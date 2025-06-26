import re

import torch
from kago_utils.actions import Ankan, Chii, Dahai, Daiminkan, Kakan, KyuushuKyuuhai, Pon, Riichi, Ronho, Skip, Tsumoho
from kago_utils.game import Game, State
from kago_utils.hai import Hai
from kago_utils.hai_group import HaiGroup
from kago_utils.player import Player
from tqdm import tqdm

from kago_trainer.haihu_loader import HaihuItem, HaihuLoader
from kago_trainer.huuro_parser import HuuroParser
from kago_trainer.mode import Mode
from kago_trainer.plane_builder import PlaneBuilder
from kago_trainer.yama_generator import YamaGenerator


class HaihuParser:
    game: Game

    count: int
    mode: Mode
    max_count: int
    progress_bar: tqdm
    debug: bool

    x: torch.Tensor | None
    t: torch.Tensor | None

    haihu_id: str
    ts: int
    tags: list[tuple[str, dict[str, str]]]
    tag_i: int

    __slots__ = (
        "game",
        "count",
        "mode",
        "max_count",
        "progress_bar",
        "debug",
        "x",
        "t",
        "haihu_id",
        "ts",
        "tags",
        "tag_i",
    )

    def __init__(self, mode: Mode, max_count: int, debug: bool = False) -> None:
        self.mode = mode
        self.max_count = max_count
        self.debug = debug
        self.count = 0
        self.progress_bar = tqdm(total=self.max_count)
        self.x = None
        self.t = None

    def run(self) -> None:
        for haihu in HaihuLoader(root_dir="haihus"):
            self.parse_file(haihu)
            if self.count >= self.max_count:
                break

        dataset = {"x": self.x, "t": self.t}
        torch.save(dataset, f"./datasets/{self.output_haihu_id}.pt")

    def parse_file(self, haihu: HaihuItem) -> None:
        self.haihu_id = haihu.id
        # 後めくりのDORAタグが悪さをするのでDORAタグはすべて削除する
        self.tags = [tag for tag in haihu.tags if tag[0] != "DORA"]
        self.ts = -1
        for tag_i, (elem, attr) in enumerate(self.tags):
            self.tag_i = tag_i

            # 初期化
            if elem == "mjloggm":
                self.parse_mjloggm()

            # 乱数
            if elem == "SHUFFLE":
                self.parse_shuffle_tag(attr)

            # ルール
            if elem == "GO":
                if not self.is_valid_rule(attr):
                    return

            # 開局
            elif elem == "INIT":
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
            elif elem == "REACH":
                self.parse_riichi_tag(attr)

            # ドラ
            elif elem == "DORA":
                continue

            # 和了
            elif elem == "AGARI":
                self.parse_agari_tag(attr)

            # 流局
            elif elem == "RYUUKYOKU":
                self.parse_ryuukyoku_tag(attr)

            if self.count >= self.max_count:
                return

    def parse_mjloggm(self) -> None:
        self.game = Game()
        self.game.add_player(Player("1"))
        self.game.add_player(Player("2"))
        self.game.add_player(Player("3"))
        self.game.add_player(Player("4"))

    def parse_shuffle_tag(self, attr: dict[str, str]) -> None:
        seed = attr["seed"].split(",")[1]
        yama_generator = YamaGenerator(seed)

        # Yamaのshuffleメソッドをモックする
        self.game.yama.shuffle = lambda: [Hai(id) for id in yama_generator.generate()]  # type: ignore[method-assign]

    # TODO: 将来的にはparse_go_tagというメソッド名に変更したい。
    # TODO: 将来的にはRuleクラスを実装して、GOタグの中身も解析するようにしたい。
    def is_valid_rule(self, attr: dict[str, str]) -> bool:
        _type = int(attr["type"])

        # 東風戦
        if (_type & 0x08) == 0:
            return False

        return True

    def parse_init_tag(self, attr: dict[str, str]) -> None:
        self.ts += 1

        self.wait_prev_state("init_kyoku")

        # 手牌チェック
        juntehai1 = sorted([int(hai) for hai in attr["hai0"].split(",")])
        juntehai2 = sorted([int(hai) for hai in attr["hai1"].split(",")])
        juntehai3 = sorted([int(hai) for hai in attr["hai2"].split(",")])
        juntehai4 = sorted([int(hai) for hai in attr["hai3"].split(",")])
        assert self.game.players[0].juntehai.to_list() == juntehai1, (
            self.game.players[0].juntehai.to_code(),
            HaiGroup.from_list(juntehai1).to_code(),
        )
        assert self.game.players[1].juntehai.to_list() == juntehai2
        assert self.game.players[2].juntehai.to_list() == juntehai3
        assert self.game.players[3].juntehai.to_list() == juntehai4

        # 局数、本場、供託、ドラ表示牌チェック
        kyoku, honba, kyoutaku, _, _, dora_hyouji_hai = map(int, attr["seed"].split(","))
        assert self.game.kyoku == kyoku
        assert self.game.honba == honba, (self.game.honba, honba)
        assert self.game.kyoutaku == kyoutaku
        assert self.game.yama.opened_dora_hyouji_hais[0].id == dora_hyouji_hai

        # 点数チェック
        ten = [int(t) * 100 for t in attr["ten"].split(",")]
        assert [player.ten for player in self.game.players] == ten, ([player.ten for player in self.game.players], ten)

    def parse_tsumo_tag(self, elem: str) -> None:
        idx = {"T": 0, "U": 1, "V": 2, "W": 3}
        who = idx[elem[0]]
        hai = Hai(int(elem[1:]))

        teban_player = self.game.players[who]

        self.wait_prev_state(["tsumo", "rinshan_tsumo"])

        if self.mode == Mode.RIICHI and not teban_player.is_riichi_completed:
            next_elem, _ = self.get_next_tag()
            self.output(who, int(next_elem == "REACH"))

        assert teban_player.last_tsumo == hai

    def parse_dahai_tag(self, elem: str) -> None:
        idx = {"D": 0, "E": 1, "F": 2, "G": 3}
        who = idx[elem[0]]
        hai = Hai(int(elem[1:]))

        self.wait_state("wait_teban_action")

        teban_player = self.game.players[who]

        # 打牌の抽出(リーチ時以外)
        if self.mode == Mode.DAHAI and not teban_player.is_riichi_completed:
            self.output(who, hai.id // 4)

        # 打牌の処理
        self.game.teban_action_resolver.register_dahai(teban_player, Dahai(hai))

        # 打牌の次に来る最初の副露か和了か流局を検出
        for next_elem, next_attr in self.tags[self.tag_i + 1 :]:
            # 副露か和了が来るならそのまま副露か和了をする
            if next_elem in ["N", "AGARI"]:
                break
            # 三家和了がくるならロンを3つ登録する
            elif next_elem == "RYUUKYOKU" and "type" in next_attr and next_attr["type"] == "ron3":
                self.wait_state("wait_non_teban_action")
                for player in self.game.players:
                    if not player.is_teban:
                        self.game.non_teban_action_resolver.register_ronho(player, Ronho())
                break
            # ツモか流局が来るならスキップを登録する
            elif re.match(r"[T|U|V|W][0-9]+", next_elem) or next_elem == "RYUUKYOKU":
                self.wait_state("wait_non_teban_action")
                for player in self.game.players:
                    if not player.is_teban:
                        self.game.non_teban_action_resolver.register_skip(player, Skip())
                break

        self.next_step()

    def parse_huuro_tag(self, attr: dict[str, str]) -> None:
        who = int(attr["who"])
        m = int(attr["m"])

        me = self.game.players[who]
        huuro = HuuroParser.from_haihu(m)

        match huuro:
            case Chii():
                self.wait_state("wait_non_teban_action")
                self.game.non_teban_action_resolver.register_chii(me, huuro)
                for player in self.game.players:
                    if player != me:
                        self.game.non_teban_action_resolver.register_skip(player, Skip())
                self.next_step()

            case Pon():
                self.wait_state("wait_non_teban_action")
                self.game.non_teban_action_resolver.register_pon(me, huuro)
                for player in self.game.players:
                    if player != me:
                        self.game.non_teban_action_resolver.register_skip(player, Skip())
                self.next_step()

            case Kakan():
                self.wait_state("wait_teban_action")
                self.game.teban_action_resolver.register_kakan(me, huuro)
                next_elem, _ = self.get_next_tag()
                # 槍槓じゃなかったらスキップを登録する
                if next_elem != "AGARI":
                    self.wait_state("wait_chankan_action")
                    for player in self.game.players:
                        if not player.is_teban:
                            self.game.chankan_action_resolver.register_skip(player, Skip())
                self.next_step()

            case Daiminkan():
                self.wait_state("wait_non_teban_action")
                self.game.non_teban_action_resolver.register_daiminkan(me, huuro)
                for player in self.game.players:
                    if player != me:
                        self.game.non_teban_action_resolver.register_skip(player, Skip())
                self.next_step()

            case Ankan():
                self.wait_state("wait_teban_action")
                self.game.teban_action_resolver.register_ankan(me, huuro)
                self.next_step()

    def parse_riichi_tag(self, attr: dict[str, str]) -> None:
        who = int(attr["who"])
        step = int(attr["step"])

        # リーチ宣言
        if step == 1:
            self.wait_state("wait_teban_action")
            me = self.game.players[who]
            resolver = self.game.teban_action_resolver
            resolver.register_riichi(me, Riichi())
            self.next_step()

        # リーチ成立
        elif step == 2:
            return

    def parse_agari_tag(self, attr: dict[str, str]) -> None:
        who = int(attr["who"])
        from_who = int(attr["fromWho"])

        me = self.game.players[who]

        if who == from_who:
            self.wait_state("wait_teban_action")
            self.game.teban_action_resolver.register_tsumoho(me, Tsumoho())
            self.next_step()
        else:
            self.wait_state(["wait_non_teban_action", "wait_chankan_action"])
            if self.game.state == "wait_non_teban_action":
                self.game.non_teban_action_resolver.register_ronho(me, Ronho())
            elif self.game.state == "wait_chankan_action":
                self.game.chankan_action_resolver.register_ronho(me, Ronho())

            # 最後の和了の場合は未登録のプレイヤーにスキップを登録する
            next_elem, _ = self.get_next_tag()
            if next_elem != "AGARI":
                for player in self.game.players:
                    if self.game.non_teban_action_resolver.choice[player.id] is None:
                        self.game.non_teban_action_resolver.register_skip(player, Skip())
                self.next_step()
            # まだ和了が続く場合は次のAGARIタグをパースする
            else:
                return

        if "owari" in attr:
            owari = attr["owari"]

            self.wait_state("syuukyoku")

            ten = [int(t) * 100 for t in owari.split(",")[::2]]
            assert [player.ten for player in self.game.players] == ten, (
                [player.ten for player in self.game.players],
                ten,
            )

    def parse_ryuukyoku_tag(self, attr: dict[str, str]) -> None:
        _type = attr.get("type", "none")

        if _type == "yao9":
            self.wait_state("wait_teban_action")
            self.game.teban_action_resolver.register_kyuushu_kyuuhai(self.game.teban_player, KyuushuKyuuhai())

        self.wait_prev_state(
            [
                "ryuukyoku",
                "nagashi_mangan",
                "kyuushu_kyuuhai",
                "yoncha_riichi",
                "suuhuu_renda",
                "suukan_sanryou",
                "sancha_houra",
            ]
        )

        if "owari" in attr:
            owari = attr["owari"]

            self.wait_state("syuukyoku")
            self.next_step()

            ten = [int(t) * 100 for t in owari.split(",")[::2]]
            assert [player.ten for player in self.game.players] == ten, (
                [player.ten for player in self.game.players],
                ten,
            )

    def wait_state(self, states: State | list[State]) -> None:
        if isinstance(states, list):
            while self.game.state not in states:
                self.next_step()
        else:
            while self.game.state != states:
                self.next_step()

    def wait_prev_state(self, states: State | list[State]) -> None:
        if isinstance(states, list):
            while self.game.prev_state not in states:
                self.next_step()
        else:
            while self.game.prev_state != states:
                self.next_step()

    def next_step(self) -> None:
        if self.mode == Mode.PON and self.game.state == "wait_non_teban_action":
            resolver = self.game.non_teban_action_resolver
            for player in self.game.players:
                if len(resolver.pon_candidates[player.id]) > 0:
                    self.output(player.zaseki, int(isinstance(resolver.choice[player.id], Pon)))

        elif self.mode == Mode.CHII and self.game.state == "wait_non_teban_action":
            resolver = self.game.non_teban_action_resolver
            for player in self.game.players:
                if len(resolver.chii_candidates[player.id]) > 0:
                    self.output(player.zaseki, int(isinstance(resolver.choice[player.id], Chii)))

        self.game.step()

    def output(self, who: int, t: int) -> None:
        # 既に十分な数のデータが集まっている場合は何もしない
        if self.count >= self.max_count:
            return

        self.debug_print(self.url)

        plane_builder = PlaneBuilder(self.mode, self.game, self.game.players[who], self.debug)
        planes = plane_builder.build()

        self.debug_print(f"t: {t}")

        # デバッグ時は入力を待つ
        if self.debug:
            input()

        # xとtの形式は初回のplanesが返ってきて初めて判明する
        if self.x is None or self.t is None:
            self.x = torch.empty((self.max_count, *planes.shape), dtype=torch.float32)
            self.t = torch.empty((self.max_count,), dtype=torch.long)

        self.x[self.count] = planes
        self.t[self.count] = t

        self.count += 1
        self.progress_bar.update(1)

    @property
    def url(self) -> str:
        return f"https://tenhou.net/0/?log={self.haihu_id}&ts={self.ts}"

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
            case Mode.RONHO_DAMINKAN_PON_CHII:
                return "ronho_daiminkan_pon_chii"

        raise ValueError("Invalid Mode")

    def get_next_tag(self) -> tuple[str, dict[str, str]]:
        if self.tag_i + 1 < len(self.tags):
            return self.tags[self.tag_i + 1]
        else:
            return ("NONE", {})

    def debug_print(self, *values: object, end: str | None = "\n") -> None:
        if self.debug:
            print(*values, end=end)
