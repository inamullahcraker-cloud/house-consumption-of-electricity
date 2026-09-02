from torch import nn ,torch
class PowerGLU(nn.Module):
    def __init__(self, num_in=1, num_out=1):
        super().__init__()
        self.gru = nn.GRU(num_in, 15, batch_first=True)
        self.fc1 = nn.Linear(15, num_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x