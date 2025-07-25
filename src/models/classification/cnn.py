"""
    @file:              cnn.py
    @Author:            Ihssene Brahimi, Moustafa Amine Bezzahi

    @Creation Date:     06/2024
    @Last modification: 07/2024

    @Description:       This file is used to define CNN model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatNET(nn.Module):
    def __init__(self):
        super(PatNET, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16 * 8, 256)
        self.fc2 = nn.Linear(256, 2)  # Output layer for binary classification
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16 * 8)
        intermediate_output = self.fc1(x)  # Save intermediate output after fc1
        x = F.relu(intermediate_output)
        x = self.dropout(x)
        x = self.fc2(x)
        return x, intermediate_output

