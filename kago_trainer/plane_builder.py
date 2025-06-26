import torch
from kago_utils.game import Game
from kago_utils.hai import Hai
from kago_utils.hai_group import HaiGroup
from kago_utils.player import Player

from kago_trainer.mode import Mode


class PlaneBuilder:
    mode: Mode
    game: Game
    player: Player
    debug: bool

    __slots__ = ("mode", "game", "player", "debug")

    def __init__(self, mode: Mode, game: Game, player: Player, debug: bool = False) -> None:
        self.mode = mode
        self.game = game
        self.player = player
        self.debug = debug

    def build(self) -> torch.Tensor:
        planes: list[list[int]] = []

        planes += self.juntehai_to_plane()
        planes += self.juntehai_aka_to_plane()
        planes += self.huuro_to_plane()
        planes += self.huuro_aka_to_plane()
        planes += self.kawa_to_plane()
        planes += self.kawa_aka_to_plane()
        if self.mode in [Mode.PON, Mode.CHII]:
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
            if player == self.player:
                counter = player.juntehai.to_counter34()
                planes.extend(self.to_planes(counter, 4))
            else:
                planes.extend([[0] * 34 for _ in range(4)])

        self.debug_print("juntehai_to_plane")
        self.debug_planes(planes, 4)

        return planes

    def juntehai_aka_to_plane(self) -> list[list[int]]:
        # 全員の純手牌・赤牌(1plane * 4players)
        planes: list[list[int]] = []
        for player in self.game.players:
            if player == self.player:
                counter = [0] * 34
                for hai in player.juntehai:
                    if hai.color == "r":
                        counter[hai.id // 4] += 1
                planes.append(counter)
            else:
                planes.append([0] * 34)

        self.debug_print("juntehai_aka_to_plane")
        self.debug_planes(planes, 4)

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

        self.debug_print("huuro_to_plane")
        self.debug_planes(planes, 4)

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

        self.debug_print("huuro_aka_to_plane")
        self.debug_planes(planes, 4)

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

        self.debug_print("kawa_to_plane")
        self.debug_planes(planes, 20)

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

        self.debug_print("kawa_aka_to_plane")
        self.debug_planes(planes, 4)

        return planes

    def last_dahai_to_plane(self) -> list[list[int]]:
        # 全員の最終打牌(1plane * 4players)
        planes: list[list[int]] = []
        for i in range(4):
            if self.game.last_dahai is not None and i == self.game.teban:
                planes.append(HaiGroup([self.game.last_dahai]).to_counter34())
            else:
                planes.append([0] * 34)

        self.debug_print("last_dahai_to_plane")
        self.debug_planes(planes, 4)

        return planes

    def riichi_to_plane(self) -> list[list[int]]:
        # リーチ(4planes)
        planes: list[list[int]] = []
        for player in self.game.players:
            planes.append([1] * 34 if player.is_riichi_completed else [0] * 34)

        self.debug_print("riichi_to_plane")
        self.debug_planes(planes, 4)

        return planes

    def dora_to_plane(self) -> list[list[int]]:
        # ドラ(1planes * 4)
        planes = []
        for hai in self.game.yama.opened_dora_hyouji_hais:
            if hai.code in ["9m", "9p", "9s"]:
                dora = Hai(hai.id - 32)
            elif hai.code == "4z":
                dora = Hai(hai.id - 12)
            elif hai.code == "7z":
                dora = Hai(hai.id - 8)
            else:
                dora = Hai(hai.id + 4)
            planes.append(HaiGroup([dora]).to_counter34())

        while len(planes) < 4:
            planes.append([0] * 34)

        self.debug_print("dora_to_plane")
        self.debug_planes(planes, 1)

        return planes

    def bakaze_to_plane(self) -> list[list[int]]:
        # 場風(4planes)
        planes: list[list[int]] = []
        for i in range(4):
            planes.append([1] * 34 if i == self.game.kyoku // 4 else [0] * 34)

        self.debug_print("bakaze_to_plane")
        self.debug_planes(planes, 4)

        return planes

    def kyoku_to_plane(self) -> list[list[int]]:
        # 局数(4planes)
        planes: list[list[int]] = []
        for i in range(4):
            planes.append([1] * 34 if i == self.game.kyoku % 4 else [0] * 34)

        self.debug_print("kyoku_to_plane")
        self.debug_planes(planes, 4)

        return planes

    def ten_to_plane(self) -> list[list[int]]:
        # 点数(30planes * 4players)
        planes: list[list[int]] = []
        for player in self.game.players:
            if player.ten >= 100000:
                man = 9
                sen = 9
                hyaku = 9
            else:
                man = min(player.ten // 10000, 9)
                sen = min((player.ten % 10000) // 1000, 9)
                hyaku = min((player.ten % 1000) // 100, 9)
            for j in range(10):
                planes.append([1] * 34 if j == man else [0] * 34)
            for j in range(10):
                planes.append([1] * 34 if j == sen else [0] * 34)
            for j in range(10):
                planes.append([1] * 34 if j == hyaku else [0] * 34)

        self.debug_print("ten_to_plane")
        self.debug_planes(planes, 30)

        return planes

    def position_to_plane(self) -> list[list[int]]:
        # 場所(4planes)
        planes: list[list[int]] = []
        for player in self.game.players:
            planes.append([1] * 34 if player == self.player else [0] * 34)

        self.debug_print("position_to_plane")
        self.debug_planes(planes, 4)

        return planes

    def debug_print(self, *values: object, end: str | None = "\n") -> None:
        if self.debug:
            print(*values, end=end)

    def debug_planes(self, planes: list[list[int]], n_unit: int) -> None:
        for i, plane in enumerate(planes):
            self.debug_print(i, HaiGroup.from_counter34(plane).to_code().replace("0", "5"))
            if i % n_unit == n_unit - 1:
                if i == len(planes) - 1:
                    self.debug_print("")
                else:
                    self.debug_print("============")
