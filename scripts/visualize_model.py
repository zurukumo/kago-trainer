import os
from typing import override

import torch
from kago_utils.actions import Dahai
from kago_utils.bot import Bot
from kago_utils.game import Game
from kago_utils.visualizer import Visualizer

from kago_trainer.mode import Mode
from kago_trainer.models import MyModel
from kago_trainer.plane_builder import PlaneBuilder


class CustomBot(Bot):
    dahai_model: MyModel | None = None
    riichi_model: MyModel | None = None

    plane_builder: PlaneBuilder

    __slots__ = ("plane_builder",)

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

    @override
    def select_chankan_action(self) -> None:
        resolver = self.game.chankan_action_resolver
        if resolver.ronho_candidates[self.id]:
            resolver.register_ronho(self, resolver.ronho_candidates[self.id][0])
            return
        if resolver.skip_candidates[self.id]:
            resolver.register_skip(self, resolver.skip_candidates[self.id][0])
            return

    def select_dahai(self, candidates: list[Dahai]) -> Dahai:
        assert CustomBot.dahai_model is not None

        input = PlaneBuilder(Mode.DAHAI, self.game, self, debug=False).build()
        output = CustomBot.dahai_model(input)
        choice = max(candidates, key=lambda c: output[c.hai.id // 4].item())
        return choice

    def select_riichi(self) -> bool:
        assert CustomBot.riichi_model is not None

        input = PlaneBuilder(Mode.RIICHI, self.game, self, debug=False).build()
        output = CustomBot.riichi_model(input)
        return bool(output[0].item() > output[1].item())

    @classmethod
    def load_model(cls) -> None:
        dahai_checkpoint_filepath = cls._get_latest_checkpoint_path("dahai")
        dahai_checkpoint = torch.load(dahai_checkpoint_filepath, weights_only=True)
        cls.dahai_model = MyModel(dahai_checkpoint["n_channel"], dahai_checkpoint["n_output"])
        cls.dahai_model.load_state_dict(dahai_checkpoint["model_state_dict"])
        cls.dahai_model.eval()

        riichi_checkpoint_filepath = cls._get_latest_checkpoint_path("riichi")
        riichi_checkpoint = torch.load(riichi_checkpoint_filepath, weights_only=True)
        cls.riichi_model = MyModel(dahai_checkpoint["n_channel"], riichi_checkpoint["n_output"])
        cls.riichi_model.load_state_dict(riichi_checkpoint["model_state_dict"])
        cls.riichi_model.eval()

    @classmethod
    def _get_latest_checkpoint_path(cls, prefix: str) -> str:
        model_dir = os.path.join(os.path.dirname(__file__), "../models")
        model_files = [f for f in os.listdir(model_dir) if f.startswith(prefix) and f.endswith(".pt")]
        if not model_files:
            raise FileNotFoundError(f"No model files found with prefix '{prefix}' in {model_dir}")
        latest_model_file = max(model_files)
        return os.path.join(model_dir, latest_model_file)


if __name__ == "__main__":
    # ゲームの初期化
    game = Game()
    game.add_player(CustomBot("Player1"))
    game.add_player(CustomBot("Player2"))
    game.add_player(CustomBot("Player3"))
    game.add_player(CustomBot("Player4"))

    visualizer = Visualizer(game)
    visualizer.start()
