from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from app.model import PowerGLU


class Power_predictor:
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        data_path: str,
    ):
        self.device = torch.device("cpu")

        # Load the saved scaler.
        self.scaler = joblib.load(scaler_path)

        # Load the trained GRU model.
        self.model = PowerGLU(num_in=7, num_out=1)

        state_dict = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Validate the dataset path.
        data_path = Path(data_path)

        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at: {data_path}"
            )

        # Load the original household power dataset.
        self.df = pd.read_csv(
            data_path,
            sep=";",
            na_values=["?"],
            low_memory=False,
        )

        # Combine Date and Time.
        self.df["datetime"] = pd.to_datetime(
            self.df["Date"].astype(str)
            + " "
            + self.df["Time"].astype(str),
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        )

        # Convert power consumption to numeric.
        self.df["Global_active_power"] = pd.to_numeric(
            self.df["Global_active_power"],
            errors="coerce",
        )

        # Remove invalid records and sort chronologically.
        self.df = (
            self.df
            .dropna(
                subset=["datetime", "Global_active_power"]
            )
            .sort_values("datetime")
            .drop_duplicates("datetime")
            .reset_index(drop=True)
        )

        if len(self.df) < 24:
            raise ValueError(
                "The dataset must contain at least 24 valid observations."
            )

        # Generate time features.
        self.df = self.add_time_features(self.df)

        # Preserve the exact feature order used during training.
        self.feature_columns = [
            "Global_active_power",
            "month_cos",
            "month_sin",
            "weekday_cos",
            "weekday_sin",
            "hour_cos",
            "hour_sin",
        ]

        print(f"Dataset loaded successfully: {self.df.shape}")

        print(
            f"Dataset date range: "
            f"{self.df['datetime'].min()} to "
            f"{self.df['datetime'].max()}"
        )

    @staticmethod
    def add_time_features(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        df = df.copy()

        month = df["datetime"].dt.month
        weekday = df["datetime"].dt.weekday
        hour = df["datetime"].dt.hour

        # Preserve the formulas used by the existing trained model.
        df["month_cos"] = np.cos(
            2 * np.pi * month / 12
        )
        df["month_sin"] = np.cos(
            2 * np.pi * month / 12
        )

        df["weekday_cos"] = np.cos(
            2 * np.pi * weekday / 7
        )
        df["weekday_sin"] = np.cos(
            2 * np.pi * weekday / 7
        )

        df["hour_cos"] = np.cos(
            2 * np.pi * hour / 24
        )
        df["hour_sin"] = np.cos(
            2 * np.pi * hour / 24
        )

        return df

    def predict(
        self,
        start_time: datetime,
        current_power: float,
    ) -> list[dict]:

        # Validate the current power value.
        current_power = float(current_power)

        if not np.isfinite(current_power) or current_power < 0:
            raise ValueError(
                "Current power must be a finite, non-negative value."
            )

        start_time = pd.Timestamp(start_time)

        # Retrieve 23 historical observations before the selected time.
        history = self.df[
            self.df["datetime"] < start_time
        ].tail(23)

        if len(history) != 23:
            raise ValueError(
                "The selected date/time must have at least 23 "
                "historical observations in the dataset."
            )

        # Build the current observation using the user's power reading.
        current_df = pd.DataFrame({
            "datetime": [start_time],
            "Global_active_power": [current_power],
        })

        current_df = self.add_time_features(current_df)

        # Combine 23 historical observations with the current observation.
        combined = pd.concat(
            [
                history[self.feature_columns],
                current_df[self.feature_columns],
            ],
            ignore_index=True,
        )

        # Scale the 24 observations.
        sequence = self.scaler.transform(
            combined.to_numpy(dtype=np.float32)
        ).astype(np.float32)

        predictions = []

        # Generate 24 forecast points.
        for step in range(24):

            # Model input shape: (1, 24, 7).
            input_tensor = torch.tensor(
                sequence[-24:],
                dtype=torch.float32,
            ).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                prediction_scaled = float(
                    self.model(input_tensor)
                    .cpu()
                    .numpy()[0, 0]
                )

            # Inverse-transform the target feature.
            dummy = np.zeros(
                (1, len(self.feature_columns)),
                dtype=np.float32,
            )

            dummy[0, 0] = prediction_scaled

            prediction_power = float(
                self.scaler.inverse_transform(dummy)[0, 0]
            )

            # Forecast the next hour.
            prediction_time = (
                start_time + timedelta(hours=step + 1)
            )

            predictions.append({
                "date_time": prediction_time.to_pydatetime(),
                "predicted_power": round(
                    prediction_power, 4
                ),
            })

            # Construct the next observation.
            next_df = pd.DataFrame({
                "datetime": [prediction_time],
                "Global_active_power": [prediction_power],
            })

            next_df = self.add_time_features(next_df)

            next_raw = next_df[
                self.feature_columns
            ].to_numpy(dtype=np.float32)

            next_scaled = self.scaler.transform(
                next_raw
            ).astype(np.float32)

            # Preserve the model's scaled target prediction.
            next_scaled[0, 0] = prediction_scaled

            # Shift the sequence and append the forecast observation.
            sequence = np.vstack([
                sequence[1:],
                next_scaled,
            ])

        return predictions