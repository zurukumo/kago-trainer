import re

import torch
from kago_utils.hai import Hai
from kago_utils.hai_group import HaiGroup
from kago_utils.huuro import Ankan, Chii, Daiminkan, Kakan, Pon
from kago_utils.shanten import Shanten
from tqdm import tqdm

from kago_trainer.haihu_loader import HaihuLoader
from kago_trainer.huuro_parser import HuuroParser
from kago_trainer.mode import Mode


class HaihuParser:
    count: int
    mode: Mode
    max_case: int
    progress_bar: tqdm
    debug: bool
    loader: HaihuLoader

    x: list[list[list[int]]]
    t: list[int]

    haihu_id: str
    ts: int
    tags: list[tuple[str, dict[str, str]]]
    tag_i: int

    tehai: list[HaiGroup]
    huuro: list[list[Chii | Pon | Kakan | Daiminkan | Ankan]]
    kawa: list[list[Hai]]
    dora: list[Hai]
    riichi: list[bool]
    kyoku: int
    ten: list[int]
    last_teban: int | None
    last_tsumo: Hai | None
    last_dahai: Hai | None
    who: int

    __slots__ = (
        "count",
        "mode",
        "max_case",
        "progress_bar",
        "debug",
        "loader",
        "x",
        "t",
        "haihu_id",
        "ts",
        "tags",
        "tag_i",
        "tehai",
        "huuro",
        "kawa",
        "dora",
        "riichi",
        "kyoku",
        "ten",
        "last_teban",
        "last_tsumo",
        "last_dahai",
        "who",
    )

    def __init__(self, mode: Mode, max_case: int, debug: bool = False) -> None:
        self.count = 0
        self.mode = mode
        self.max_case = max_case
        self.progress_bar = tqdm(total=self.max_case)
        self.debug = debug
        self.loader = HaihuLoader("haihus")

        self.x = []
        self.t = []

        self.run()

        dataset = {
            "x": torch.tensor(self.x, dtype=torch.float32),
            "t": torch.tensor(self.t, dtype=torch.long),
        }
        torch.save(dataset, f"./datasets/{self.output_haihu_id}.pt")

    def run(self) -> None:
        for haihu in self.loader:
            self.haihu_id = haihu.id
            self.tags = haihu.tags
            self.ts = -1
            for tag_i, (elem, attr) in enumerate(haihu.tags):
                self.tag_i = tag_i

                # 開局
                if elem == "INIT":
                    self.parse_init_tag(attr)

                # ツモ
                elif re.match(r"[T|U|V|W][0-9]+", elem):
                    self.parse_tsumo_tag(elem)

                # 打牌
                elif re.match(r"[D|E|F|G][0-9]+", elem):
                    self.parse_dahai_tag(elem)

                # 副露
                elif elem == "N":
                    self.parse_huuro_tag(attr)

                # リーチ成立
                elif elem == "REACH" and attr["step"] == "2":
                    who = int(attr["who"])
                    self.riichi[who] = True

                # ドラ
                elif elem == "DORA":
                    self.dora.append(Hai(int(attr["hai"])))

                # 和了
                elif elem == "AGARI":
                    who = int(attr["who"])
                    self.who = who

                if self.count >= self.max_case:
                    return

    def url(self) -> str:
        return f"https://tenhou.net/0/?log={self.haihu_id}&ts={self.ts}"

    def sample_riichi(self, who: int) -> None:
        next_elem, _ = self.tags[self.tag_i + 1]

        # リーチ中
        if self.riichi[who]:
            return

        # 鳴いている
        has_naki = any([isinstance(huuro, (Chii, Pon, Kakan, Daiminkan)) for huuro in self.huuro[who]])
        if has_naki:
            return

        if Shanten(self.tehai[who]).shanten == 0:
            if next_elem == "REACH":
                y = 1
            else:
                y = 0
            self.output(who, y)

    def sample_ankan(self, who: int) -> None:
        next_elem, next_attr = self.tags[self.tag_i + 1]

        # リーチ中(向聴数と待ちが変わらない暗槓が可能)
        if self.riichi[who]:
            if self.last_tsumo is None:
                return
            if self.tehai[who].to_counter34()[self.last_tsumo.id // 4] != 4:
                return

            # ツモ前
            tehai1 = self.tehai[who] - self.last_tsumo
            shanten1 = Shanten(tehai1)
            # 暗槓後
            base_id = self.last_tsumo.id - self.last_tsumo.id % 4
            tehai2 = self.tehai[who] - HaiGroup.from_list([base_id, base_id + 1, base_id + 2, base_id + 3])
            shanten2 = Shanten(tehai2)

            if not (shanten1.shanten == shanten2.shanten == 0):
                return
            if shanten1.yuukouhai != shanten2.yuukouhai:
                return

            if next_elem == "N":
                huuro = HuuroParser.from_haihu(int(next_attr["m"]))
                if isinstance(huuro, Ankan):
                    self.output(who, 1)
            else:
                self.output(who, 0)

        # 非リーチ中
        else:
            for i in range(34):
                if self.tehai[who].to_counter34()[i] != 4:
                    continue

                if next_elem == "N":
                    huuro = HuuroParser.from_haihu(int(next_attr["m"]))
                    if isinstance(huuro, Ankan):
                        self.output(who, 1)
                else:
                    self.output(who, 0)

    def sample_ron_daiminkan_pon_chii(self) -> None:
        next_elem, next_attr = self.tags[self.tag_i + 1]
        for who in range(4):
            if who == self.last_teban:
                continue

            # 何もしない -> 0, ロン -> 1, 明槓 -> 2, ポン -> 3, 左牌をチー -> 4, 中央牌をチー -> 5, 右牌をチー -> 6
            y = 0
            if next_elem == "AGARI":
                # ダブロン、トリロンの可能性があるので全てのAGARIタグを見る
                for elem, attr in self.tags[self.tag_i + 1 :]:
                    if elem != "AGARI":
                        break
                    if int(attr["who"]) == who:
                        y = 1
            elif next_elem == "N" and int(next_attr["who"]) == who:
                huuro = HuuroParser.from_haihu(int(next_attr["m"]))
                if isinstance(huuro, Daiminkan):
                    y = 2
                elif isinstance(huuro, Pon):
                    y = 3
                elif isinstance(huuro, Chii):
                    if huuro.stolen == huuro.hais[0]:
                        y = 4
                    elif huuro.stolen == huuro.hais[1]:
                        y = 5
                    elif huuro.stolen == huuro.hais[2]:
                        y = 6

            self.output(who, y)

    def debug_print(self, *values: object, end: str | None = "\n") -> None:
        if self.debug:
            print(*values, end=end)

    def debug_planes(self, planes: list[list[int]], n_unit: int) -> None:
        for i, plane in enumerate(planes):
            self.debug_print(i, HaiGroup.from_counter34(plane).to_code())
            if i % n_unit == n_unit - 1:
                if i == len(planes) - 1:
                    self.debug_print("")
                else:
                    self.debug_print("============")

    def to_planes(self, counter: list[int], depth: int) -> list[list[int]]:
        planes = [[0] * 34 for _ in range(depth)]
        for i in range(34):
            for j in range(counter[i]):
                planes[j][i] = 1
        return planes

    def flatten(self, planes: list[list[int]]) -> list[int]:
        return sum(planes, [])

    def jun_tehai_to_plane(self, who: int) -> list[list[int]]:
        # 全員の手牌(4planes * 4players)
        planes: list[list[int]] = []
        for i in range(4):
            if i == who:
                counter = self.tehai[i].to_counter34()
                planes.extend(self.to_planes(counter, 4))
            else:
                planes.extend([[0] * 34 for _ in range(4)])

        self.debug_print("純手牌")
        self.debug_planes(planes, 4)
        return planes

    def jun_tehai_aka_to_plane(self, who: int) -> list[list[int]]:
        # 全員の純手牌・赤牌(1plane * 4players)
        planes: list[list[int]] = []
        for i in range(4):
            if i == who:
                counter = [0] * 34
                for hai in self.tehai[who].hais:
                    if hai.color == "r":
                        counter[hai.id // 4] += 1
                planes.append(counter)
            else:
                planes.append([0] * 34)

        self.debug_print("純手牌・赤牌")
        self.debug_planes(planes, 1)
        return planes

    def huuro_to_plane(self) -> list[list[int]]:
        # 全員の副露(16planes * 4players)
        planes: list[list[int]] = []
        for i in range(4):
            for j in range(4):
                if j < len(self.huuro[i]):
                    counter = self.huuro[i][j].hais.to_counter34()
                    planes.extend(self.to_planes(counter, 4))
                else:
                    planes.extend([[0] * 34 for _ in range(4)])

        self.debug_print("副露")
        self.debug_planes(planes, 16)
        return planes

    def huuro_aka_to_plane(self) -> list[list[int]]:
        # 全員の副露・赤牌(1plane * 4players)
        planes: list[list[int]] = []
        for i in range(4):
            counter = [0] * 34
            for huuro in self.huuro[i]:
                for hai in huuro.hais:
                    if hai.color == "r":
                        counter[hai.id // 4] += 1
            planes.append(counter)

        self.debug_print("副露・赤牌")
        self.debug_planes(planes, 1)
        return planes

    def kawa_to_plane(self) -> list[list[int]]:
        # 全員の河(20planes * 4players)
        planes: list[list[int]] = []
        for i in range(4):
            for j in range(20):
                if j < len(self.kawa[i]):
                    planes.append(HaiGroup([self.kawa[i][j]]).to_counter34())
                else:
                    planes.append([0] * 34)

        self.debug_print("河")
        self.debug_planes(planes, 20)
        return planes

    def kawa_aka_to_plane(self) -> list[list[int]]:
        # 全員の河の赤牌(1plane * 4players)
        planes: list[list[int]] = []
        for i in range(4):
            counter = [0] * 34
            for hai in self.kawa[i]:
                if hai.color == "r":
                    counter[hai.id // 4] += 1
            planes.append(counter)

        self.debug_print("河・赤牌")
        self.debug_planes(planes, 1)
        return planes

    def last_dahai_to_plane(self) -> list[list[int]]:
        # 全員の最終打牌(1plane * 4players)
        planes: list[list[int]] = []
        for i in range(4):
            if self.last_dahai is not None and self.last_teban is not None and i == self.last_teban:
                planes.append(HaiGroup([self.last_dahai]).to_counter34())
            else:
                planes.append([0] * 34)

        self.debug_print("最終打牌")
        self.debug_planes(planes, 1)
        return planes

    def riichi_to_plane(self) -> list[list[int]]:
        # リーチ(4planes)
        planes: list[list[int]] = []
        for i in range(4):
            planes.append([1] * 34 if self.riichi[i] else [0] * 34)

        self.debug_print("リーチ")
        self.debug_planes(planes, 1)
        return planes

    def dora_to_plane(self) -> list[list[int]]:
        # ドラ(4planes)
        planes = self.to_planes(HaiGroup(self.dora).to_counter34(), 4)

        self.debug_print("ドラ")
        self.debug_planes(planes, 4)
        return planes

    def bakaze_to_plane(self) -> list[list[int]]:
        # 場風(4planes)
        planes: list[list[int]] = []
        for i in range(4):
            planes.append([1] * 34 if i == self.kyoku // 4 else [0] * 34)

        self.debug_print("場風")
        self.debug_planes(planes, 4)
        return planes

    def kyoku_to_plane(self) -> list[list[int]]:
        # 局数(4planes)
        planes: list[list[int]] = []
        for i in range(4):
            planes.append([1] * 34 if i == self.kyoku % 4 else [0] * 34)

        self.debug_print("局")
        self.debug_planes(planes, 4)
        return planes

    def ten_to_plane(self) -> list[list[int]]:
        # 点数(30planes * 4players)
        planes: list[list[int]] = []
        for i in range(4):
            man = min(self.ten[i] // 100, 9)
            sen = self.ten[i] % 100 // 10
            hyaku = self.ten[i] % 10
            for j in range(10):
                planes.append([1] * 34 if j == man else [0] * 34)
            for j in range(10):
                planes.append([1] * 34 if j == sen else [0] * 34)
            for j in range(10):
                planes.append([1] * 34 if j == hyaku else [0] * 34)

        self.debug_print("点数")
        self.debug_planes(planes, 30)
        return planes

    def position_to_plane(self, who: int) -> list[list[int]]:
        # 場所(4planes)
        planes: list[list[int]] = []
        for i in range(4):
            planes.append([1] * 34 if i == who else [0] * 34)

        self.debug_print("場所")
        self.debug_planes(planes, 4)
        return planes

    @property
    def output_haihu_id(self) -> str:
        match self.mode:
            case Mode.DAHAI:
                return "dahai"
            case Mode.RIICHI:
                return "riichi"
            case Mode.ANKAN:
                return "ankan"
            case Mode.KAKAN:
                return "kakan"
            case Mode.RON_DAMINKAN_PON_CHII:
                return "ron_daiminkan_pon_chii"

        raise ValueError("Invalid Mode")

    def output(self, who: int, t: int) -> None:
        self.debug_print(self.url())

        planes: list[list[int]] = []

        planes += self.jun_tehai_to_plane(who)
        planes += self.jun_tehai_aka_to_plane(who)
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
        planes += self.position_to_plane(who)

        # デバッグ時は入力を待つ
        if self.debug:
            input()

        self.x.append(planes)
        self.t.append(t)

        self.count += 1
        self.progress_bar.update(1)

    def parse_init_tag(self, attr: dict[str, str]) -> None:
        self.ts += 1

        self.tehai = [HaiGroup([]) for _ in range(4)]
        self.kawa = [[] for _ in range(4)]
        self.huuro = [[] for _ in range(4)]
        self.dora = []
        self.riichi = [False] * 4
        self.kyoku = 0
        self.ten = [0] * 4

        # 配牌をパース
        for who in range(4):
            for hai in map(int, attr[f"hai{who}"].split(",")):
                self.tehai[who] += Hai(hai)

        # 局数、本場、供託、ドラをパース
        kyoku, honba, kyotaku, _, _, dora = map(int, attr["seed"].split(","))
        self.kyoku = kyoku
        self.dora.append(Hai(dora))

        # 点棒状況をパース
        for who, ten in enumerate(map(int, attr["ten"].split(","))):
            self.ten[who] = ten

        self.last_teban = None
        self.last_dahai = None
        self.last_tsumo = None

    def parse_tsumo_tag(self, elem: str) -> None:
        idx = {"T": 0, "U": 1, "V": 2, "W": 3}
        who = idx[elem[0]]
        hai = Hai(int(elem[1:]))

        self.tehai[who] += hai
        self.last_tsumo = hai

        # リーチの抽出
        if self.mode == Mode.RIICHI:
            self.sample_riichi(who)

        # 暗槓の抽出
        if self.mode == Mode.ANKAN:
            self.sample_ankan(who)

    def parse_dahai_tag(self, elem: str) -> None:
        idx = {"D": 0, "E": 1, "F": 2, "G": 3}
        who = idx[elem[0]]
        hai = Hai(int(elem[1:]))

        # 打牌の抽出
        if self.mode == Mode.DAHAI and not self.riichi[who]:
            self.output(who, hai.id // 4)

        # 打牌の処理
        self.kawa[who].append(hai)
        self.tehai[who] -= hai
        self.last_dahai = hai
        self.last_teban = who

        # ロン、ミンカン、ポン、チーの抽出
        if self.mode == Mode.RON_DAMINKAN_PON_CHII:
            self.sample_ron_daiminkan_pon_chii()

    def parse_huuro_tag(self, attr: dict[str, str]) -> None:
        who = int(attr["who"])
        m = int(attr["m"])
        huuro = HuuroParser.from_haihu(m)

        match huuro:
            case Chii():
                self.tehai[who] -= huuro.hais - huuro.stolen
                self.huuro[who].append(huuro)

            case Pon():
                self.tehai[who] -= huuro.hais - huuro.stolen
                self.huuro[who].append(huuro)

            case Kakan():
                for i, h in enumerate(self.huuro[who]):
                    if (
                        isinstance(h, Pon)
                        and h.hais[0].suit == huuro.hais[0].suit
                        and h.hais[0].number == huuro.hais[0].number
                    ):
                        new_huuro = h.to_kakan()
                        self.huuro[who][i] = new_huuro
                        self.tehai[who] -= new_huuro.added
                        break

            case Daiminkan():
                self.tehai[who] -= huuro.hais - huuro.stolen
                self.huuro[who].append(huuro)

            case Ankan():
                self.tehai[who] -= huuro.hais
                self.huuro[who].append(huuro)
