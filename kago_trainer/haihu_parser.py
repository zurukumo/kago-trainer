import os
import re
from typing import Generator

import torch
from kago_utils.hai import Hai136
from kago_utils.hai_group import Hai34Group, Hai136Group
from kago_utils.huuro import Ankan, Chii, Daiminkan, Kakan, Pon
from kago_utils.shanten import Shanten
from tqdm import tqdm

from kago_trainer.huuro_parser import HuuroParser
from kago_trainer.mode import Mode


class HaihuParser():
    YEARS = [2015, 2016, 2017]

    count: int
    mode: Mode
    max_case: int
    debug: bool
    progress_bar: tqdm

    x: list[list[list[int]]]
    t: list[int]

    filename: str
    ts: int
    actions: list[tuple[str, dict[str, str]]]
    action_i: int

    tehai: list[Hai136Group]
    huuro: list[list[Chii | Pon | Kakan | Daiminkan | Daiminkan]]
    kawa: list[list[Hai136]]
    dora: list[Hai136]
    riichi: list[bool]
    kyoku: int
    ten: list[int]
    last_teban: int | None
    last_tsumo: Hai136 | None
    last_dahai: Hai136 | None
    who: int

    __slots__ = (
        'count', 'mode', 'max_case', 'debug', 'progress_bar',
        'x', 't',
        'filename', 'ts', 'actions', 'action_i',
        'tehai', 'huuro', 'kawa', 'dora', 'riichi', 'kyoku', 'ten',
        'last_teban', 'last_tsumo', 'last_dahai', 'who'
    )

    def __init__(self, mode: Mode, max_case: int, debug: bool = False) -> None:
        self.count = 0
        self.mode = mode
        self.max_case = max_case
        self.debug = debug
        self.progress_bar = tqdm(total=self.max_case)

        self.x = []
        self.t = []

        self.run()

        dataset = {
            "x": torch.tensor(self.x, dtype=torch.float32),
            "t": torch.tensor(self.t, dtype=torch.long)
        }
        torch.save(dataset, f"./datasets/{self.output_filename}.pt")

    def run(self) -> None:
        for filepath, filename in self.list_xml_files():
            self.filename = filename
            self.ts = -1
            self.actions = self.parse_actions(filepath)

            for action_i, (elem, attr) in enumerate(self.actions):
                self.action_i = action_i

                # 開局
                if elem == 'INIT':
                    self.parse_init_tag(attr)

                # ツモ
                elif re.match(r'[T|U|V|W][0-9]+', elem):
                    self.parse_tsumo_tag(elem)

                # 打牌
                elif re.match(r'[D|E|F|G][0-9]+', elem):
                    self.parse_dahai_tag(elem)

                # 副露
                elif elem == 'N':
                    self.parse_huuro_tag(attr)

                # リーチ成立
                elif elem == 'REACH' and attr['step'] == '2':
                    who = int(attr['who'])
                    self.riichi[who] = True

                # ドラ
                elif elem == 'DORA':
                    self.dora.append(Hai136(int(attr['hai'])))

                # 和了
                elif elem == 'AGARI':
                    who = int(attr['who'])
                    self.who = who

                if self.count >= self.max_case:
                    return

    def list_xml_files(self) -> Generator[tuple[str, str], None, None]:
        for year in HaihuParser.YEARS:
            for filename in os.listdir(f'./haihus/xml{year}'):
                filepath = f'./haihus/xml{year}/{filename}'
                yield filepath, filename

    def parse_actions(self, filename: str) -> list[tuple[str, dict[str, str]]]:
        with open(filename, 'r') as xml:
            actions = []
            for elem, attr in re.findall(r'<(.*?)[ /](.*?)/?>', xml.read()):
                attr = dict(re.findall(r'\s?(.*?)="(.*?)"', attr))
                actions.append((elem, attr))

        return actions

    def url(self) -> str:
        return f'https://tenhou.net/0/?log={self.filename.replace('.xml', '')}&ts={self.ts}'

    def sample_riichi(self, who: int) -> None:
        next_elem, _ = self.actions[self.action_i + 1]

        # リーチ中
        if self.riichi[who]:
            return

        # 鳴いている
        has_naki = any([isinstance(huuro, (Chii, Pon, Kakan, Daiminkan)) for huuro in self.huuro[who]])
        if has_naki:
            return

        if Shanten(self.tehai[who]).shanten == 0:
            if next_elem == 'REACH':
                y = 1
            else:
                y = 0
            self.output(who, y)

    def sample_ankan(self, who: int) -> None:
        next_elem, next_attr = self.actions[self.action_i + 1]

        # リーチ中(向聴数と待ちが変わらない暗槓が可能)
        if self.riichi[who]:
            if self.last_tsumo is None:
                return
            if self.tehai[who].to_hai34_group().to_counter()[self.last_tsumo.to_hai34()] != 4:
                return

            # ツモ前
            shanten1 = Shanten(self.tehai[who] - Hai136Group([self.last_tsumo]))
            # 暗槓後
            shanten2 = Shanten(self.tehai[who] - Hai34Group([self.last_tsumo] * 4))

            if not (shanten1.shanten == shanten2.shanten == 0):
                return
            if shanten1.yuukouhai != shanten2.yuukouhai:
                return

            if next_elem == 'N':
                huuro = HuuroParser.from_haihu(int(next_attr['m']))
                if isinstance(huuro, Ankan):
                    self.output(who, 1)
            else:
                self.output(who, 0)

        # 非リーチ中
        else:
            for i in range(34):
                if self.tehai[who].to_hai34_group().to_counter()[i] != 4:
                    continue

                if next_elem == 'N':
                    huuro = HuuroParser.from_haihu(int(next_attr['m']))
                    if isinstance(huuro, Ankan):
                        self.output(who, 1)
                else:
                    self.output(who, 0)

    def sample_ron_daiminkan_pon_chii(self) -> None:
        next_elem, next_attr = self.actions[self.action_i + 1]
        for who in range(4):
            if who == self.last_teban:
                continue

            # 何もしない -> 0, ロン -> 1, 明槓 -> 2, ポン -> 3, 左牌をチー -> 4, 中央牌をチー -> 5, 右牌をチー -> 6
            y = 0
            if next_elem == 'AGARI':
                # ダブロン、トリロンの可能性があるので全てのAGARIタグを見る
                for elem, attr in self.actions[self.action_i + 1:]:
                    if elem != 'AGARI':
                        break
                    if int(attr['who']) == who:
                        y = 1
            elif next_elem == 'N' and int(next_attr['who']) == who:
                huuro = HuuroParser.from_haihu(int(next_attr['m']))
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
            self.debug_print(i, Hai34Group.from_counter(plane).to_string())
            if i % n_unit == n_unit - 1:
                if i == len(planes) - 1:
                    self.debug_print('')
                else:
                    self.debug_print('============')

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
                counter = self.tehai[i].to_hai34_group().to_counter()
                planes.extend(self.to_planes(counter, 4))
            else:
                planes.extend([[0] * 34 for _ in range(4)])

        self.debug_print("手牌")
        self.debug_planes(planes, 4)
        return planes

    def jun_tehai_aka_to_plane(self, who: int) -> list[list[int]]:
        # 全員の純手牌・赤牌(1plane * 4players)
        planes: list[list[int]] = []
        for i in range(4):
            if i == who:
                counter = [0] * 34
                for hai in self.tehai[who].hais:
                    if hai.is_aka():
                        counter[hai.to_hai34().id] += 1
                planes.append(counter)
            else:
                planes.append([0] * 34)

        self.debug_print("純手牌・赤牌")
        self.debug_planes(planes, 1)
        return planes

    def huuro_to_plane(self) -> list[list[int]]:
        # 全員の副露(4planes * 4players)
        planes: list[list[int]] = []
        for i in range(4):
            tmp = Hai136Group([])
            for huuro in self.huuro[i]:
                tmp += huuro.hais
            counter = tmp.to_hai34_group().to_counter()
            planes.extend(self.to_planes(counter, 4))

        self.debug_print("副露")
        self.debug_planes(planes, 4)
        return planes

    def kawa_to_plane(self) -> list[list[int]]:
        # 全員の河(20planes * 4players)
        planes: list[list[int]] = []
        for i in range(4):
            for j in range(20):
                if j < len(self.kawa[i]):
                    planes.append(Hai34Group([self.kawa[i][j].to_hai34()]).to_counter())
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
                if hai.is_aka():
                    counter[hai.to_hai34().id] += 1
            planes.append(counter)

        self.debug_print("河・赤牌")
        self.debug_planes(planes, 1)
        return planes

    def last_dahai_to_plane(self) -> list[list[int]]:
        # 全員の最終打牌(1plane * 4players)
        planes: list[list[int]] = []
        for i in range(4):
            if self.last_dahai is not None and self.last_teban is not None and i == self.last_teban:
                planes.append(Hai34Group([self.last_dahai.to_hai34()]).to_counter())
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
        planes = self.to_planes(Hai136Group(self.dora).to_hai34_group().to_counter(), 4)

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
    def output_filename(self) -> str:
        match self.mode:
            case Mode.DAHAI:
                return 'dahai'
            case Mode.RIICHI:
                return 'riichi'
            case Mode.ANKAN:
                return 'ankan'
            case Mode.KAKAN:
                return 'kakan'
            case Mode.RON_DAMINKAN_PON_CHII:
                return 'ron_daiminkan_pon_chii'

        raise ValueError('Invalid Mode')

    def output(self, who: int, t: int) -> None:
        self.debug_print(self.url())

        planes: list[list[int]] = []

        planes += self.jun_tehai_to_plane(who)
        planes += self.jun_tehai_aka_to_plane(who)
        planes += self.huuro_to_plane()
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

        self.tehai = [Hai136Group([]) for _ in range(4)]
        self.kawa = [[] for _ in range(4)]
        self.huuro = [[] for _ in range(4)]
        self.dora = []
        self.riichi = [False] * 4
        self.kyoku = 0
        self.ten = [0] * 4

        # 配牌をパース
        for who in range(4):
            for hai in map(int, attr[f'hai{who}'].split(',')):
                self.tehai[who] += Hai136(hai)

        # 局数、本場、供託、ドラをパース
        kyoku, honba, kyotaku, _, _, dora = map(int, attr['seed'].split(','))
        self.kyoku = kyoku
        self.dora.append(Hai136(dora))

        # 点棒状況をパース
        for who, ten in enumerate(map(int, attr['ten'].split(','))):
            self.ten[who] = ten

        self.last_teban = None
        self.last_dahai = None
        self.last_tsumo = None

    def parse_tsumo_tag(self, elem: str) -> None:
        idx = {'T': 0, 'U': 1, 'V': 2, 'W': 3}
        who = idx[elem[0]]
        hai = Hai136(int(elem[1:]))

        self.tehai[who] += hai
        self.last_tsumo = hai

        # リーチの抽出
        if self.mode == Mode.RIICHI:
            self.sample_riichi(who)

        # 暗槓の抽出
        if self.mode == Mode.ANKAN:
            self.sample_ankan(who)

    def parse_dahai_tag(self, elem: str) -> None:
        idx = {'D': 0, 'E': 1, 'F': 2, 'G': 3}
        who = idx[elem[0]]
        hai = Hai136(int(elem[1:]))

        # 打牌の抽出
        if self.mode == Mode.DAHAI and not self.riichi[who]:
            self.output(who, hai.to_hai34().id)

        # 打牌の処理
        self.kawa[who].append(hai)
        self.tehai[who] -= hai
        self.last_dahai = hai
        self.last_teban = who

        # ロン、ミンカン、ポン、チーの抽出
        if self.mode == Mode.RON_DAMINKAN_PON_CHII:
            self.sample_ron_daiminkan_pon_chii()

    def parse_huuro_tag(self, attr: dict[str, str]) -> None:
        who = int(attr['who'])
        m = int(attr['m'])
        huuro = HuuroParser.from_haihu(m)

        match huuro:
            case Chii():
                self.tehai[who] -= (huuro.hais - huuro.stolen)
                self.huuro[who].append(huuro)

            case Pon():
                self.tehai[who] -= (huuro.hais - huuro.stolen)
                self.huuro[who].append(huuro)

            case Kakan():
                for i, h in enumerate(self.huuro[who]):
                    if isinstance(h, Pon) and h.hais == huuro.hais:
                        new_huuro = h.to_kakan()
                        self.huuro[who][i] = new_huuro
                        self.tehai[who] -= new_huuro.added
                        break

            case Daiminkan():
                self.tehai[who] -= (huuro.hais - huuro.stolen)
                self.huuro[who].append(huuro)

            case Ankan():
                self.tehai[who] -= huuro.hais
                self.huuro[who].append(huuro)
