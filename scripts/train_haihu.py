import questionary

from core.haihu_trainer import HaihuTrainer
from core.mode import Mode

if __name__ == '__main__':
    mode = questionary.select('mode?', choices=[mode.value for mode in Mode]).ask()
    batch_size = int(questionary.text('batch_size?', default='32').ask())
    n_epoch = int(questionary.text('n_epoch?', default='10').ask())
    HaihuTrainer(mode=Mode(mode), batch_size=batch_size, n_epoch=n_epoch)
