import questionary

from kago_trainer.haihu_parser import HaihuParser
from kago_trainer.mode import Mode

if __name__ == "__main__":
    mode = questionary.select("mode?", choices=[mode.value for mode in Mode]).ask()
    max_count = int(questionary.text("max_count?", default="150000").ask())
    debug = questionary.confirm("debug?", default=False).ask()

    parser = HaihuParser(mode=Mode(mode), max_count=max_count, debug=debug)
    parser.run()
