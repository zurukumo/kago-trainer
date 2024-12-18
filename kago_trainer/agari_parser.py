import os
import re
from typing import Generator
from uuid import uuid4

from kago_utils.agari import Agari
from kago_utils.game import Game
from kago_utils.hai import Hai
from kago_utils.hai_group import HaiGroup
from kago_utils.player import Player
from tqdm import tqdm

from .huuro_parser import HuuroParser

YAKU = (
    "門前清自摸和",
    "立直",
    "一発",
    "槍槓",
    "嶺上開花",
    "海底摸月",
    "河底撈魚",
    "平和",
    "断幺九",
    "一盃口",
    "自風 東",
    "自風 南",
    "自風 西",
    "自風 北",
    "場風 東",
    "場風 南",
    "場風 西",
    "場風 北",
    "役牌 白",
    "役牌 發",
    "役牌 中",
    "両立直",
    "七対子",
    "混全帯幺九",
    "一気通貫",
    "三色同順",
    "三色同刻",
    "三槓子",
    "対々和",
    "三暗刻",
    "小三元",
    "混老頭",
    "二盃口",
    "純全帯幺九",
    "混一色",
    "清一色",
    "",
    "天和",
    "地和",
    "大三元",
    "四暗刻",
    "四暗刻単騎",
    "字一色",
    "緑一色",
    "清老頭",
    "九蓮宝燈",
    "純正九蓮宝燈",
    "国士無双",
    "国士無双１３面",
    "大四喜",
    "小四喜",
    "四槓子",
    "ドラ",
    "裏ドラ",
    "赤ドラ",
)

JOKYO_YAKU = (
    "門前清自摸和",
    "立直",
    "一発",
    "槍槓",
    "嶺上開花",
    "海底摸月",
    "河底撈魚",
    "両立直",
    "天和",
    "地和",
    "ドラ",
    "裏ドラ",
    "赤ドラ",
)


def convert_yaku_from_tenhou(tyaku: list[int], tyakuman: list[int]) -> dict[str, int]:
    yaku = Agari.initialize_yaku()

    for i in range(0, len(tyaku), 2):
        yaku_id = tyaku[i]
        han = tyaku[i + 1]
        yaku[YAKU[yaku_id]] = han

    for i in range(len(tyakuman)):
        yaku_id = tyakuman[i]
        yaku[YAKU[yaku_id]] = 13

    return yaku


class AgariParser:
    YEARS = [2015, 2016, 2017]

    count: int
    max_case: int
    progress_bar: tqdm

    filename: str
    ts: int
    kyoku: int
    actions: list[tuple[str, dict[str, str]]]

    __slots__ = (
        "count",
        "max_case",
        "progress_bar",
        "filename",
        "ts",
        "kyoku",
        "actions",
    )

    def __init__(self, max_case: int | None) -> None:
        self.count = 0
        self.max_case = max_case if max_case is not None else self.count_files()
        self.progress_bar = tqdm(total=self.max_case)

        self.run()

    def run(self) -> None:
        for filepath, filename in self.list_xml_files():
            self.filename = filename
            self.ts = -1
            self.actions = self.parse_actions(filepath)
            kyoku = -1

            for action_i, (elem, attr) in enumerate(self.actions):
                if elem == "INIT":
                    self.ts += 1
                    kyoku = int(attr["seed"].split(",")[0])
                elif elem == "AGARI":
                    honba = int(attr["ba"].split(",")[0])
                    kyoutaku = int(attr["ba"].split(",")[1])
                    hai = list(map(int, attr["hai"].split(",")))
                    m = list(map(int, attr["m"].split(","))) if "m" in attr else []
                    machi = int(attr["machi"])
                    yaku = list(map(int, attr["yaku"].split(","))) if "yaku" in attr else []
                    yakuman = list(map(int, attr["yakuman"].split(","))) if "yakuman" in attr else []
                    who = int(attr["who"])
                    from_who = int(attr["fromWho"])
                    sc = list(map(int, attr["sc"].split(",")))

                    game = Game()
                    for _ in range(4):
                        game.add_player(Player(str(uuid4())))

                    player = game.find_player_by_zaseki(who)
                    is_daburon = self.actions[action_i - 1][0] == "AGARI"

                    game.kyoku = kyoku
                    game.honba = honba
                    game.kyoutaku = kyoutaku
                    player.zaseki = who
                    player.juntehai = HaiGroup.from_list(hai)
                    player.huuros = [HuuroParser.from_haihu(code) for code in m[::-1]]

                    if who == from_who:
                        player.last_tsumo = Hai(machi)
                        game.last_teban = who
                    else:
                        player.juntehai -= Hai(machi)
                        game.last_dahai = Hai(machi)
                        game.last_teban = from_who

                    tenhou_yaku = convert_yaku_from_tenhou(yaku, yakuman)
                    jokyo_yaku = dict(((k, v) if k in JOKYO_YAKU else (k, 0) for k, v in tenhou_yaku.items()))

                    # mock
                    Agari.get_jokyo_yaku = lambda _: jokyo_yaku

                    agari = Agari(game, player, is_daburon)
                    tenhou_ten_movement = [i * 100 for i in sc[1::2]]
                    if agari.ten_movement != tenhou_ten_movement:
                        print("NOT MATCH")
                        print(f"ex: {tenhou_yaku}")
                        print(f"re: {agari.yaku}")
                        print(f"ex: {tenhou_ten_movement}")
                        print(f"re: {agari.ten_movement}")
                        print(self.url())
                        exit()

            self.count += 1
            self.progress_bar.update(1)
            if self.count >= self.max_case:
                return

    def count_files(self) -> int:
        return sum(len(os.listdir(f"./haihus/xml{year}")) for year in AgariParser.YEARS)

    def list_xml_files(self) -> Generator[tuple[str, str], None, None]:
        for year in AgariParser.YEARS:
            for filename in os.listdir(f"./haihus/xml{year}"):
                filepath = f"./haihus/xml{year}/{filename}"
                yield filepath, filename

    def parse_actions(self, filename: str) -> list[tuple[str, dict[str, str]]]:
        with open(filename, "r") as xml:
            actions = []
            for elem, attr in re.findall(r"<(.*?)[ /](.*?)/?>", xml.read()):
                attr = dict(re.findall(r'\s?(.*?)="(.*?)"', attr))
                actions.append((elem, attr))

        return actions

    def url(self) -> str:
        return f'https://tenhou.net/0/?log={self.filename.replace('.xml', '')}&ts={self.ts}'
