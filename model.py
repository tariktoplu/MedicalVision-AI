# medical_analyzer_project/model.py

import torch
from torch import nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, d, h, w = x.shape
        y = x.mean(dim=[2, 3, 4])
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1, 1)
        return x * y

class MR3DCNN_LSTM_Attention(nn.Module):
    def __init__(self, hidden_size=64, lstm_layers=1):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 8, 3, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.se1 = SEBlock(8)
        
        self.conv2 = nn.Conv3d(8, 16, 3, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        self.se2 = SEBlock(16)
        
        self.flatten = nn.Flatten(start_dim=2)
        self.lstm = nn.LSTM(input_size=16 * 64 * 64, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.se1(x)
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.se2(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(x.size(0), x.size(1), -1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])