import sched
import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from pl_bolts.optimizers import lr_scheduler
from pytorch_lightning.metrics import Accuracy

from utils.data import get_datamodule
from utils.nets import MultiHeadEncoder
from utils.callbacks import PretrainCheckpointCallback

from argparse import ArgumentParser
from datetime import datetime


parser = ArgumentParser()
parser.add_argument("--dataset", default="CIFAR10", type=str, help="dataset")
parser.add_argument("--flip", action='store_true', default=False, help="use different classes")
parser.add_argument("--download", default=False, action="store_true", help="wether to download")
parser.add_argument("--data_dir", default="/path/to/dataset", type=str, help="data directory")
parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
parser.add_argument("--checkpoint_dir", default="checkpoints", type=str, help="checkpoint dir")
parser.add_argument("--batch_size", default=256, type=int, help="batch size")
parser.add_argument("--num_workers", default=16, type=int, help="number of workers")
parser.add_argument("--img_size", default=32, type=int, help="size of input images")
parser.add_argument("--arch", default="vit_base", type=str, help="backbone architecture")
parser.add_argument("--patch_size", default=4, type=int, help="patch size of vision transformers")
parser.add_argument('--optim', default='sgd', type=str, help="optimizer")
parser.add_argument("--scheduler", default='cosine', type=str, help='learning rate scheduler')
parser.add_argument("--base_lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--min_lr", default=0.001, type=float, help="min learning rate")
parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
parser.add_argument("--weight_decay_opt", default=1.0e-4, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")
parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
parser.add_argument("--comment", default=datetime.now().strftime("%b%d_%H-%M-%S"), type=str)
parser.add_argument("--project", default="kcc_pretrain", type=str, help="wandb project")
parser.add_argument("--entity", default="dhk", type=str, help="wandb entity")
parser.add_argument("--offline", default=False, action="store_true", help="disa`ble wandb")
parser.add_argument("--num_labeled_classes", default=5, type=int, help="number of labeled classes")
parser.add_argument("--num_unlabeled_classes", default=5, type=int, help="number of unlab classes")
parser.add_argument("--pretrained", type=str, default=None, help="pretrained checkpoint path")
parser.add_argument("--freeze", action='store_true', help="pretrained encoder checkpoint path")


class Pretrainer(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters({k: v for (k, v) in kwargs.items() if not callable(v)})

        # build model
        self.model = MultiHeadEncoder(
            arch=self.hparams.arch,
            patch_size=self.hparams.patch_size,
            low_res="CIFAR" in self.hparams.dataset,
            num_labeled=self.hparams.num_labeled_classes,
            num_unlabeled=self.hparams.num_unlabeled_classes,
            num_heads=None,
            pretrained=self.hparams.pretrained,
            freeze=self.hparams.freeze
        )

        # metrics
        self.accuracy = Accuracy()

    def configure_optimizers(self):
        if self.hparams.optim == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.hparams.base_lr
            )
        elif self.hparams.optim == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams.base_lr,
                momentum=self.hparams.momentum_opt,
                weight_decay=self.hparams.weight_decay_opt,
            )
        if self.hparams.scheduler == 'cosine':
            scheduler = lr_scheduler.LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.hparams.warmup_epochs,
                max_epochs=self.hparams.max_epochs,
                warmup_start_lr=self.hparams.min_lr,
                eta_min=self.hparams.min_lr,
            )
        return [optimizer], [scheduler]
    
    def unpack_batch(self, batch):
        images, labels = batch
        if self.hparams.flip:
            return images, labels-self.hparams.num_unlabeled_classes
        else:
            return images, labels
            

    def training_step(self, batch, batch_idx):
        images, labels = self.unpack_batch(batch)

        # normalize prototypes
        self.model.normalize_prototypes()

        # forward
        outputs = self.model(images)

        # supervised cross entropy loss
        loss_supervised = torch.stack(
            [F.cross_entropy(o / self.hparams.temperature, labels) for o in outputs["logits_lab"]]
        ).mean()

        # log
        results = {
            "loss_supervised": loss_supervised,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }
        self.log_dict(results, on_step=False, on_epoch=True, sync_dist=True)

        # reweight loss
        return loss_supervised

    def validation_step(self, batch, batch_idx):
        images, labels = self.unpack_batch(batch)

        # forward
        logits = self.model(images)["logits_lab"]
        _, preds = logits.max(dim=-1)

        # calculate loss and accuracy
        loss_supervised = F.cross_entropy(logits, labels)
        acc = self.accuracy(preds, labels)

        # log
        results = {
            "val/loss_supervised": loss_supervised,
            "val/acc": acc,
        }
        self.log_dict(results, on_step=False, on_epoch=True)
        return results


def main(args):
    # build datamodule
    dm = get_datamodule(args, "pretrain")

    # logger
    arch_name = args.arch if 'resnet' in args.arch else '_'.join([args.arch, str(args.patch_size)])
    run_name = "-".join(["pretrain", arch_name, args.dataset, args.comment])
    wandb_logger = pl.loggers.WandbLogger(
        save_dir=args.log_dir,
        name=run_name,
        project=args.project,
        entity=args.entity,
        offline=args.offline,
    )

    accelerator = "ddp" if args.gpus > 1 else None
    model = Pretrainer(**args.__dict__)
    trainer = pl.Trainer.from_argparse_args(
        args, logger=wandb_logger, callbacks=[PretrainCheckpointCallback()], accelerator=accelerator
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
