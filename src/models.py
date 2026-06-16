import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.2,
        use_dp: bool = False,      
    ):
        super().__init__()
        self.use_dp = use_dp

        if use_dp:
            try:
                from opacus.layers import DPLSTM
                lstm_class = DPLSTM
            except ImportError:
                import warnings
                warnings.warn(
                    "Opacus not installed; using nn.LSTM.  "
                    "Install opacus for proper DP-LSTM support.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                lstm_class = nn.LSTM
        else:
            lstm_class = nn.LSTM

        self.lstm = lstm_class(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]         
        return self.fc(last)


class CNN1DClassifier(nn.Module):

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.mean(dim=-1)          
        return self.fc(x)
