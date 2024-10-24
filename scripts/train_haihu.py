import os

import questionary

from kago_trainer.haihu_trainer import HaihuTrainer
from kago_trainer.mode import Mode

if __name__ == '__main__':
    mode = questionary.select('mode?', choices=[mode.value for mode in Mode]).ask()
    batch_size = int(questionary.text('batch_size?', default='32').ask())
    n_epoch = int(questionary.text('n_epoch?', default='10').ask())

    # チェックポイント候補を列挙する処理
    model_dir = os.path.join(os.path.dirname(__file__), '../models')
    checkpoint_paths = []
    for filename in os.listdir(model_dir):
        if filename.startswith(mode) and filename.endswith(('.pt', '.pth')):
            checkpoint_paths.append(filename)
    checkpoint_paths.sort()
    checkpoint_path = questionary.select('checkpoint_path?', choices=["None"] + checkpoint_paths).ask()

    if checkpoint_path == "None":
        checkpoint_path = None
    else:
        checkpoint_path = os.path.join(model_dir, checkpoint_path)

    HaihuTrainer(mode=Mode(mode), batch_size=batch_size, n_epoch=n_epoch, checkpoint_path=checkpoint_path)
