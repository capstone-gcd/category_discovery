import torch
from pytorch_lightning.callbacks import Callback

import os


class PretrainCheckpointCallback(Callback):
    def on_save_checkpoint(self, trainer, pl_module):
        arch_name = pl_module.hparams.arch if 'resnet' in pl_module.hparams.arch else '_'.join([pl_module.hparams.arch, str(pl_module.hparams.patch_size)])
        checkpoint_filename = (
            "-".join(
                [
                    "pretrain",
                    arch_name,
                    pl_module.hparams.dataset,
                    pl_module.hparams.comment,
                ]
            )
            + ".cp"
        )
        checkpoint_path = os.path.join(pl_module.hparams.checkpoint_dir, checkpoint_filename)
        if trainer.global_rank == 0:
            torch.save(pl_module.model.state_dict(), checkpoint_path)
