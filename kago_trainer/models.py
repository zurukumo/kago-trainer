import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding="same")
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + residual)


class MyModel(nn.Module):
    def __init__(self, n_channel: int, n_output: int):
        super(MyModel, self).__init__()
        self.initial_conv = nn.Conv1d(n_channel, 256, kernel_size=3, padding="same")
        self.blocks = nn.ModuleList([ResidualBlock(256) for _ in range(10)])
        self.final_conv = nn.Conv1d(256, 1, kernel_size=3, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.initial_conv(x))
        for block in self.blocks:
            out = block(out)
        out = torch.squeeze(self.final_conv(out))
        return out
