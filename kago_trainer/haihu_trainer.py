import datetime
import os

import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split

from kago_trainer.mode import Mode
from kago_trainer.models import MyModel


class HaihuTrainer:
    def __init__(self, mode: Mode, batch_size: int, n_epoch: int, checkpoint_path: str | None):
        self.mode = mode
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.initial_epoch = 0
        self.logs = []

        # データローダーの準備
        self.prepare_data_loader()

        # モデルや最適化手法などの準備
        self.model = MyModel(self.n_channel, self.n_output)
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        # チェックポイントが指定されている場合は読み込む
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.initial_epoch += len(checkpoint["logs"])

            self.logs = checkpoint["logs"]
            print("Checkpoint loaded.")
            for log in self.logs:
                self.print_log(log)
            print("=============================")

        for epoch in range(self.initial_epoch, self.initial_epoch + n_epoch):
            train_loss = self.train_model()
            accuracy, test_loss = self.evaluate_model()

            log = {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "accuracy": accuracy,
            }
            self.logs.append(log)
            self.print_log(log)

        # モデルを保存
        model_path = os.path.join(os.path.dirname(__file__), f"../models/{self.model_name}.pt")
        model_path = os.path.abspath(model_path)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "logs": self.logs,
                "n_channel": self.n_channel,
                "n_output": self.n_output,
            },
            model_path,
        )
        print(f"Model saved: {model_path}")

    def prepare_data_loader(self):
        # シード値を固定する
        torch.manual_seed(0)

        # ファイル読み込み
        dataset_path = os.path.join(os.path.dirname(__file__), f"../datasets/{self.mode.value}.pt")
        dataset_dict = torch.load(dataset_path, weights_only=True)

        # データセットの準備
        dataset = TensorDataset(dataset_dict["x"], dataset_dict["t"])
        train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

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

    @property
    def model_name(self):
        dt = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        return f"{self.mode.value}_{dt}"

    @property
    def n_channel(self):
        return self.train_loader.dataset[0][0].shape[0]

    @property
    def n_output(self):
        match self.mode:
            case Mode.DAHAI:
                return 34
            case Mode.RIICHI:
                return 2
            case Mode.ANKAN:
                return 34
            case Mode.RONHO_DAMINKAN_PON_CHII:
                return 7
            case _:
                raise ValueError("Invalid mode")

    def print_log(self, log):
        max_width = len(str(self.initial_epoch + self.n_epoch))
        print(
            ", ".join(
                [
                    f"Epoch {log['epoch'] + 1:>{max_width}}/{self.initial_epoch + self.n_epoch}",
                    f"Train Loss: {log['train_loss']:.4f}",
                    f"Test Loss: {log['test_loss']:.4f}",
                    f"Accuracy: {log['accuracy'] * 100:.2f}%",
                ]
            )
        )
