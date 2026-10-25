from torch import nn ,torch
class PowerGLU(nn.Module):
    def __init__(self, num_in=7, num_out=1):
        super().__init__()
        self.gru = nn.GRU(num_in, 15, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(15, num_out)

    def forward(self, x):
        x, _ = self.gru(x)
        x = x[:, -1, :]  
        x = self.fc1(x)
        return x