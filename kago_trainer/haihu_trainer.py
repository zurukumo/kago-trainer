import os

import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split

from kago_trainer.mode import Mode


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding='same')
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding='same')

    def forward(self, x):
        residual = x  # 入力をそのまま保存 (残差)
        out = self.conv1(x).relu()
        out = self.conv2(out)
        return (out + residual).relu()  # 残差接続で足し合わせる


class MyModel(nn.Module):
    def __init__(self, n_channel, n_output):
        super(MyModel, self).__init__()
        self.initial_conv = nn.Conv1d(n_channel, 256, kernel_size=3, padding='same')
        self.blocks = nn.ModuleList([ResidualBlock(256) for _ in range(50)])
        self.final_conv = nn.Conv1d(256, 1, kernel_size=3, padding='same')

    def forward(self, x):
        out = self.initial_conv(x).relu()
        for block in self.blocks:
            out = block(out)
        out = self.final_conv(out).squeeze()
        return out


class HaihuTrainer:
    def __init__(self, mode: Mode, batch_size: int, n_epoch: int):
        match mode:
            case Mode.DAHAI:
                self.n_output = 34
                self.filename = 'dahai'
            case Mode.RIICHI:
                self.n_output = 2
                self.filename = 'riichi'
            case Mode.ANKAN:
                self.n_output = 34
                self.filename = 'ankan'
            case Mode.RON_DAMINKAN_PON_CHII:
                self.n_output = 7
                self.filename = 'ron_daiminkan_pon_chii'
            case _:
                raise ValueError('Invalid mode')

        # データローダーの準備
        self.prepare_data_loader()

        # モデル、最適化手法、損失関数の設定
        self.model = MyModel(self.n_channel, self.n_output)
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        for epoch in range(n_epoch):
            train_loss = self.train_model()
            accuracy, test_loss = self.evaluate_model()
            print(', '.join([
                f'Epoch {epoch+1}/{n_epoch}',
                f'Train Loss: {train_loss:.4f}',
                f'Test Loss: {test_loss:.4f}',
                f'Accuracy: {accuracy*100:.2f}%'
            ]))

        # モデルを保存
        current_dir = os.path.dirname(__file__)
        torch.save(self.model.state_dict(), os.path.join(current_dir, f'../models/{self.filename}.pth'))
        print(f'Model saved as {self.filename}_model.pth')

    def prepare_data_loader(self):
        # シード値を固定する
        torch.manual_seed(0)

        # ファイル読み込み
        dataset_path = os.path.join(os.path.dirname(__file__), f'../datasets/{self.mode.value}.pt')
        dataset_dict = torch.load(dataset_path, weights_only=True)

        # データセットの準備
        dataset = TensorDataset(dataset_dict['x'], dataset_dict['t'])
        train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # チャンネル数の設定
        self.n_channel = dataset_dict['x'].shape[1]

    # 学習用関数
    def train_model(self):
        self.model.train()
        running_loss = 0.0
        for inputs, labels in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.squeeze())
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(self.train_loader)

    # 評価用関数
    def evaluate_model(self):
        self.model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.squeeze())
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels.squeeze()).sum().item()
        return correct / total, running_loss / len(self.test_loader)
