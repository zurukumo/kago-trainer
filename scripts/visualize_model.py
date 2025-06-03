import os
from typing import override

import torch
from kago_utils.actions import Dahai
from kago_utils.bot import Bot
from kago_utils.game import Game
from kago_utils.hai_group import HaiGroup
from kago_utils.visualizer import Visualizer

from kago_trainer.models import MyModel


class CustomBot(Bot):
    dahai_model: MyModel | None = None
    riichi_model: MyModel | None = None

    def __init__(self, id: str) -> None:
        super().__init__(id=id)
        CustomBot.load_model()

    @override
    def select_teban_action(self) -> None:
        resolver = self.game.teban_action_resolver
        if resolver.tsumoho_candidates[self.id]:
            resolver.register_tsumoho(self, resolver.tsumoho_candidates[self.id][0])
            return
        if resolver.riichi_candidates[self.id] and self.select_riichi():
            resolver.register_riichi(self, resolver.riichi_candidates[self.id][0])
            return
        if resolver.dahai_candidates[self.id]:
            resolver.register_dahai(self, self.select_dahai(resolver.dahai_candidates[self.id]))
            return

    @override
    def select_non_teban_action(self) -> None:
        resolver = self.game.non_teban_action_resolver
        if resolver.ronho_candidates[self.id]:
            resolver.register_ronho(self, resolver.ronho_candidates[self.id][0])
            return
        if resolver.skip_candidates[self.id]:
            resolver.register_skip(self, resolver.skip_candidates[self.id][0])
            return

    def select_dahai(self, candidates: list[Dahai]) -> Dahai:
        assert CustomBot.dahai_model is not None

        input = self.make_input()
        output = CustomBot.dahai_model(input)
        choice = max(candidates, key=lambda c: output[c.hai.id // 4].item())
        return choice

    def select_riichi(self) -> bool:
        assert CustomBot.riichi_model is not None

        input = self.make_input()
        output = CustomBot.riichi_model(input)
        return bool(output[0].item() > output[1].item())

    @classmethod
    def load_model(cls) -> None:
        cls.dahai_model = MyModel(316, 34)
        filepath = os.path.join(os.path.dirname(__file__), "../models/dahai.pt")
        checkpoint = torch.load(filepath, weights_only=True)
        cls.dahai_model.load_state_dict(checkpoint["model_state_dict"])
        cls.dahai_model.eval()

        cls.riichi_model = MyModel(316, 2)
        filepath = os.path.join(os.path.dirname(__file__), "../models/riichi.pt")
        checkpoint = torch.load(filepath, weights_only=True)
        cls.riichi_model.load_state_dict(checkpoint["model_state_dict"])
        cls.riichi_model.eval()

    # TODO: haihu_parserと重複しているのでDRYにしたい
    def make_input(self) -> torch.Tensor:
        planes: list[list[int]] = []

        planes += self.juntehai_to_plane()
        planes += self.juntehai_aka_to_plane()
        planes += self.huuro_to_plane()
        planes += self.huuro_aka_to_plane()
        planes += self.kawa_to_plane()
        planes += self.kawa_aka_to_plane()
        planes += self.last_dahai_to_plane()
        planes += self.riichi_to_plane()
        planes += self.dora_to_plane()
        planes += self.bakaze_to_plane()
        planes += self.kyoku_to_plane()
        planes += self.ten_to_plane()
        planes += self.position_to_plane()

        return torch.tensor(planes, dtype=torch.float32)

    def to_planes(self, counter: list[int], depth: int) -> list[list[int]]:
        planes = [[0] * 34 for _ in range(depth)]
        for i in range(34):
            for j in range(counter[i]):
                planes[j][i] = 1
        return planes

    def juntehai_to_plane(self) -> list[list[int]]:
        # 全員の手牌(4planes * 4players)
        planes: list[list[int]] = []
        for player in self.game.players:
            if player == self:
                counter = self.juntehai.to_counter34()
                planes.extend(self.to_planes(counter, 4))
            else:
                planes.extend([[0] * 34 for _ in range(4)])

        return planes

    def juntehai_aka_to_plane(self) -> list[list[int]]:
        # 全員の純手牌・赤牌(1plane * 4players)
        planes: list[list[int]] = []
        for player in self.game.players:
            if player == self:
                counter = [0] * 34
                for hai in self.juntehai:
                    if hai.color == "r":
                        counter[hai.id // 4] += 1
                planes.append(counter)
            else:
                planes.append([0] * 34)

        return planes

    def huuro_to_plane(self) -> list[list[int]]:
        # 全員の副露(16planes * 4players)
        planes: list[list[int]] = []
        for player in self.game.players:
            for i in range(4):
                if i < len(player.huuros):
                    counter = player.huuros[i].hais.to_counter34()
                    planes.extend(self.to_planes(counter, 4))
                else:
                    planes.extend([[0] * 34 for _ in range(4)])

        return planes

    def huuro_aka_to_plane(self) -> list[list[int]]:
        # 全員の副露・赤牌(1plane * 4players)
        planes: list[list[int]] = []
        for player in self.game.players:
            counter = [0] * 34
            for huuro in player.huuros:
                for hai in huuro.hais:
                    if hai.color == "r":
                        counter[hai.id // 4] += 1
            planes.append(counter)

        return planes

    def kawa_to_plane(self) -> list[list[int]]:
        # 全員の河(20planes * 4players)
        planes: list[list[int]] = []
        for player in self.game.players:
            for i in range(20):
                if i < len(player.kawa):
                    planes.append(HaiGroup([player.kawa[i]]).to_counter34())
                else:
                    planes.append([0] * 34)

        return planes

    def kawa_aka_to_plane(self) -> list[list[int]]:
        # 全員の河の赤牌(1plane * 4players)
        planes: list[list[int]] = []
        for player in self.game.players:
            counter = [0] * 34
            for hai in player.kawa:
                if hai.color == "r":
                    counter[hai.id // 4] += 1
            planes.append(counter)

        return planes

    def last_dahai_to_plane(self) -> list[list[int]]:
        # 全員の最終打牌(1plane * 4players)
        planes: list[list[int]] = []
        for i in range(4):
            if self.last_dahai is not None and self.game.last_teban is not None and i == self.game.last_teban:
                planes.append(HaiGroup([self.last_dahai]).to_counter34())
            else:
                planes.append([0] * 34)

        return planes

    def riichi_to_plane(self) -> list[list[int]]:
        # リーチ(4planes)
        planes: list[list[int]] = []
        for player in self.game.players:
            planes.append([1] * 34 if player.is_riichi_completed else [0] * 34)

        return planes

    def dora_to_plane(self) -> list[list[int]]:
        # ドラ(4planes)
        planes = self.to_planes(HaiGroup(self.game.yama.doras).to_counter34(), 4)

        return planes

    def bakaze_to_plane(self) -> list[list[int]]:
        # 場風(4planes)
        planes: list[list[int]] = []
        for i in range(4):
            planes.append([1] * 34 if i == self.game.kyoku // 4 else [0] * 34)

        return planes

    def kyoku_to_plane(self) -> list[list[int]]:
        # 局数(4planes)
        planes: list[list[int]] = []
        for i in range(4):
            planes.append([1] * 34 if i == self.game.kyoku % 4 else [0] * 34)

        return planes

    def ten_to_plane(self) -> list[list[int]]:
        # 点数(30planes * 4players)
        planes: list[list[int]] = []
        for player in self.game.players:
            man = min(player.ten // 100, 9)
            sen = player.ten % 100 // 10
            hyaku = player.ten % 10
            for j in range(10):
                planes.append([1] * 34 if j == man else [0] * 34)
            for j in range(10):
                planes.append([1] * 34 if j == sen else [0] * 34)
            for j in range(10):
                planes.append([1] * 34 if j == hyaku else [0] * 34)

        return planes

    def position_to_plane(self) -> list[list[int]]:
        # 場所(4planes)
        planes: list[list[int]] = []
        for player in self.game.players:
            planes.append([1] * 34 if player == self else [0] * 34)

        return planes


if __name__ == "__main__":
    # ゲームの初期化
    game = Game()
    game.add_player(CustomBot("Player1"))
    game.add_player(CustomBot("Player2"))
    game.add_player(CustomBot("Player3"))
    game.add_player(CustomBot("Player4"))

    visualizer = Visualizer(game)
    visualizer.start()
