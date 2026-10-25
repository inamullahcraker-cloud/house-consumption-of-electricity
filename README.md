# ⚡ Household Power Consumption Forecasting Web App

An end-to-end time-series forecasting application built with **PyTorch and FastAPI** to predict household electricity consumption using historical power-consumption data and time-based features.

The project demonstrates the evolution of a forecasting model, starting with a single feature and progressing toward a GRU model that incorporates temporal information.

---

## 📈 Model Development Journey

### 1. Single-Feature Forecasting

Initially, the model used only one input feature:

- `Global_active_power`

The goal was to learn patterns in historical electricity consumption and predict future power usage based on previous readings.

We experimented with two recurrent neural network architectures:

- **LSTM (Long Short-Term Memory):** Used to learn temporal patterns from sequential power-consumption data.
- **GRU (Gated Recurrent Unit):** Explored as an alternative recurrent architecture for learning sequential patterns.

In this stage, the models relied entirely on historical power-consumption values, without additional calendar or time information.

### 2. Adding Time-Based Features

In the next stage, we extended the model inputs by introducing temporal features derived from the date and time.

The final model uses seven input features:

| Feature | Description |
|---|---|
| `Global_active_power` | Historical household electricity consumption |
| `month_cos` | Cosine-based representation of the month |
| `month_sin` | Sine-based representation of the month |
| `weekday_cos` | Cosine-based representation of the day of the week |
| `weekday_sin` | Sine-based representation of the day of the week |
| `hour_cos` | Cosine-based representation of the hour |
| `hour_sin` | Sine-based representation of the hour |

These features provide the model with information about **when** electricity is being consumed, in addition to **how much** electricity is being consumed.

For example, electricity usage may vary depending on the hour of the day, day of the week, or month of the year.

> **Note:** The sine and cosine features in the current implementation are generated using the existing training formulas. Their implementation should be corrected and the model retrained together if proper cyclic encoding is desired.

### 3. Final GRU Model

The current forecasting model uses a **Gated Recurrent Unit (GRU)** network with a fully connected output layer.

**Model configuration:**
- Input features: 7
- Sequence length: 24 observations
- GRU hidden units: 15
- GRU layers: 1
- Output: Predicted `Global_active_power`
- Feature scaling: Pre-fitted `MinMaxScaler`
- Framework: PyTorch

The model processes a sequence of historical observations containing power consumption and time-based features to estimate the next power-consumption value.

The forecasting application then uses the model iteratively to generate future predictions.

---

## 📊 Prediction & Analytics Dashboard

| **Power Consumption Forecast** | **Dashboard Overview** |
|:---:|:---:|
| ![Prediction Graph](static/screenshot_prediction.png) | ![Dashboard Overview](static/screenshot_dashboard.png) |

### Key Highlights

- **Interactive Forecast Visualization:** Displays predicted power consumption using an interactive Chart.js line graph.
- **Date and Time Input:** Allows users to specify the starting date and time for a forecast.
- **Current Power Input:** Allows users to provide the current `Global_active_power` reading.
- **Temporal Feature Engineering:** Incorporates month, weekday, and hour information into model inputs.
- **PyTorch Inference:** Loads the trained GRU model and generates predictions using the saved model parameters.
- **Feature Scaling:** Uses the saved `MinMaxScaler` to maintain consistency with the model's training data.
- **FastAPI Backend:** Provides an API endpoint for processing forecasting requests.
- **Forecast Summary:** Displays useful statistics, including average and maximum predicted power consumption.

---

## 📉 Model Evaluation

To evaluate forecasting performance, the project includes a comparison between actual and predicted power-consumption values.

![Actual vs Predicted Power Consumption](static/camparsion.png)

The evaluation graph compares **200 actual observations with their corresponding predicted values**.

This visualization helps assess how closely the model follows the actual electricity-consumption pattern and highlights differences between observed and predicted values.

---

## 🛠️ Technologies Used

- **Python** — Data processing and model development
- **PyTorch** — LSTM and GRU model development
- **Pandas** — Dataset loading and preprocessing
- **NumPy** — Numerical computations and feature engineering
- **Scikit-learn** — Feature scaling
- **FastAPI** — Backend API
- **Pydantic** — Request and response validation
- **HTML & CSS** — Frontend interface
- **JavaScript** — Client-side interactions
- **Chart.js** — Forecast visualization

---

## 📁 Project Architecture

```text
power_app/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application and API routes
│   ├── model.py                # PyTorch model architecture
│   ├── schemas.py              # Request and response validation
│   └── predictor.py            # Model loading, preprocessing, and prediction
├── artifacts/
│   ├── consumption_model.pt    # Trained PyTorch model parameters
│   └── scaler.pkl              # Saved MinMaxScaler
├── static/
│   ├── style.css               # Frontend styling
│   ├── screenshot_prediction.png
│   ├── screenshot_dashboard.png
│   └── camparasion.png         # Actual vs predicted comparison
├── templates/
│   └── index.html              # Forecasting dashboard
├── data/
│   └── household_power_consumption.txt
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 How the Application Works

1. The user enters a date and time and provides the current `Global_active_power` reading.
2. The FastAPI backend validates the submitted values.
3. The predictor loads historical power-consumption data from the dataset.
4. The application prepares the historical sequence and calculates the required time-based features.
5. The saved `MinMaxScaler` transforms the model inputs.
6. The trained GRU model predicts power consumption.
7. The application iteratively generates future predictions.
8. The frontend displays the forecast in a graph and table.

---

## 🎯 Project Objective

The primary objective of this project is to investigate how time-series forecasting can be developed from a simple, single-feature model into a model that incorporates temporal information.

The development process follows three main stages:

1. **Single-feature LSTM:** Predict power consumption using historical power readings.
2. **Single-feature GRU:** Explore an alternative recurrent architecture for power forecasting.
3. **Time-feature GRU:** Extend the input from one feature to seven features by adding month-, weekday-, and hour-based information.

This progression provides a practical demonstration of recurrent neural networks, feature engineering, time-series forecasting, and deployment through a web application.

---

## ⚠️ Important Note About Forecasting

The sequence length of 24 observations does not automatically mean that the model predicts 24 hours ahead. The forecast interval must match the sampling frequency used during training.

Because the original household power-consumption dataset contains minute-level readings, verify whether the data was resampled before training. If the model was trained on minute-level data, its sequence and forecast timestamps must reflect that frequency. If it was trained on hourly data, hourly forecasting may be appropriate.

For reliable evaluation, the actual and predicted values should also be compared using the same timestamps and preprocessing procedure.

---

## 💻 Installation and Setup

Follow these steps to run the Household Power Consumption Forecasting Web App on your local machine.

### 1. Clone the Repository

First, make sure Git is installed. Open your terminal and run:

```bash
git clone https://github.com/inamullahcraker-cloud/house-consumption-of-electricity.git
```

Navigate to the project directory:

```bash
cd house-consumption-of-electricity
```

### 2. Create a Python Virtual Environment

A virtual environment keeps the project's dependencies separate from other Python projects.

**Linux / Ubuntu / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**

```powershell
python -m venv venv
venv\Scripts\activate
```

After activation, your terminal should show the virtual environment name, such as `(venv)`.

### 3. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

**Note:** PyTorch installation requirements may differ depending on whether you use a CPU or a CUDA-enabled GPU. If installation fails, follow the official [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

### 4. Verify the Project Files

Make sure your project contains the necessary application files, trained model, scaler, and dataset.

The expected structure is:

```text
house-consumption-of-electricity/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── schemas.py
│   └── predictor.py
├── artifacts/
│   ├── consumption_model.pt
│   └── scaler.pkl
├── data/
│   └── household_power_consumption.txt
├── static/
│   ├── style.css
│   ├── screenshot_prediction.png
│   ├── screenshot_dashboard.png
│   └── camparasion.png
├── templates/
│   └── index.html
├── requirements.txt
└── README.md
```

**Important:** The actual filenames and folders in your GitHub repository must match the paths used by the application. If any files are missing, add them to the repository or update the application paths accordingly.

### 5. Run the FastAPI Application

From the project root directory, execute:

```bash
uvicorn app.main:app --reload
```

If the application starts successfully, open your browser and visit:

**Web Application:**

http://127.0.0.1:8000

**Interactive API Documentation:**

http://127.0.0.1:8000/docs

The web interface allows you to enter a date and time, provide the current `Global_active_power` reading, and view the generated power-consumption forecast.

### 6. Using the Application

1. Open the web application in your browser.
2. Enter the date and time for the forecasting process.
3. Enter the current `Global_active_power` value in kilowatts.
4. Submit the form to request predictions.
5. View the predicted power consumption in the graph and results table.
6. Examine the average and maximum predicted power values displayed on the dashboard.

The application uses historical observations, time-based features, a saved GRU model, and a fitted scaler to generate predictions.

### 7. Model Artifacts and Dataset

The application requires the following files:

- **Trained model:** `artifacts/consumption_model.pt`
- **Feature scaler:** `artifacts/scaler.pkl`
- **Historical dataset:** `data/household_power_consumption.txt`

The trained model and scaler must be compatible with the model architecture and feature preprocessing used by the application.

If these files are not included in the cloned repository, obtain them separately and place them in the expected directories.

### 8. Troubleshooting

**Problem: `ModuleNotFoundError`**

Make sure the virtual environment is activated and dependencies are installed:

```bash
pip install -r requirements.txt
```

**Problem: Model file not found**

Verify that `consumption_model.pt` exists inside the `artifacts/` directory and that the path in the application is correct.

**Problem: Dataset file not found**

Check that `household_power_consumption.txt` is inside the `data/` directory.

**Problem: Port 8000 is already in use**

Start the application on another port:

```bash
uvicorn app.main:app --reload --port 8001
```

Then open http://127.0.0.1:8001.

**Problem: Predictions fail**

Check the terminal for error messages. Verify the input values, dataset path, scaler compatibility, model architecture, and feature ordering.

---

## 🔮 Future Improvements

Potential future enhancements include:

- Comparing LSTM and GRU models using consistent evaluation metrics.
- Improving temporal feature engineering.
- Evaluating predictions using MAE, RMSE, and R².
- Adding more historical consumption visualizations.
- Optimizing inference speed and memory usage.
- Deploying the application to a cloud server.
- Adding automated tests for the prediction API.
- Improving the forecasting interface for mobile devices.

---

## 👨‍💻 Author and Repository

**GitHub Repository:**  
https://github.com/inamullahcraker-cloud/house-consumption-of-electricity

Contributions, suggestions, and feedback are welcome. If you find this project useful, consider giving the repository a ⭐ on GitHub.

---
