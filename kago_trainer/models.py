import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding="same")
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x).relu()
        out = self.conv2(out)
        return (out + residual).relu()


class MyModel(nn.Module):
    def __init__(self, n_channel: int, n_output: int):
        super(MyModel, self).__init__()
        self.initial_conv = nn.Conv1d(n_channel, 256, kernel_size=3, padding="same")
        self.blocks = nn.ModuleList([ResidualBlock(256) for _ in range(10)])
        self.final_conv = nn.Conv1d(256, 1, kernel_size=3, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.initial_conv(x).relu()
        for block in self.blocks:
            out = block(out)
        out = self.final_conv(out).squeeze()
        return out
