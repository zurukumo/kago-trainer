import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split

from kago_trainer.mode import Mode


class MyModel(nn.Module):
    def __init__(self, n_channel: int, n_output: int) -> None:
        W = 34
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_channel, out_channels=32, kernel_size=5, padding='same')
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding='same')
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding='same')
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128 * W, 128)
        self.fc2 = nn.Linear(128, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x


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

        # テストデータの準備
        x, t = self.prepare_data()
        dataset = TensorDataset(x, t)
        train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # モデル、最適化手法、損失関数の設定
        self.model = MyModel(x.shape[1], self.n_output)
        optimizer = Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(n_epoch):
            train_loss = self.train_model(optimizer, criterion)
            accuracy, test_loss = self.evaluate_model(criterion)
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

    # データ準備関数
    def prepare_data(self):
        current_dir = os.path.dirname(__file__)
        dataset = torch.load(os.path.join(current_dir, f'../datasets/{self.filename}.pt'), weights_only=True)
        return dataset['x'], dataset['t']

    # 学習用関数
    def train_model(self, optimizer, criterion):
        self.model.train()
        running_loss = 0.0
        for inputs, labels in self.train_loader:
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(self.train_loader)

    # 評価用関数
    def evaluate_model(self, criterion):
        self.model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                loss = criterion(outputs, labels.squeeze())
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels.squeeze()).sum().item()
        return correct / total, running_loss / len(self.test_loader)
