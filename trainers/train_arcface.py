import sys
import os
import logging
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

sys.path.append("..")
import face_lib.models
from face_lib.utils import DataLoaderX, MXFaceDataset, cfg
from face_lib import models as mlib, utils
from face_lib.parser_cfg import training_args
from face_lib.trainer import TrainerBase
from face_lib.models import PartialFC

torch.backends.cudnn.bencmark = True


class Trainer(TrainerBase):
    def _model_loader(self):
        self.backbone = mlib.model_dict[self.model_args.backbone.name](
            **utils.pop_element(self.model_args.backbone, "name"),
        )

        if self.args.pretrained_backbone:
            try:
                backbone_pth = os.path.join(self.args.pretrained_backbone)
                self.backbone.load_state_dict(
                    torch.load(backbone_pth, map_location=torch.device(self.local_rank))
                )
                if self.rank is 0:
                    logging.info("backbone resume successfully!")
            except (FileNotFoundError, KeyError, IndexError, RuntimeError):
                raise

        margin_softmax = face_lib.models.criterions_dict[self.model_args.criterion]()
        self.module_partial_fc = PartialFC(
            rank=self.rank,
            local_rank=self.local_rank,
            world_size=self.world_size,
            resume=args.resume,
            batch_size=self.model_args.batch_size,
            margin_softmax=margin_softmax,
            num_classes=self.model_args.num_classes,
            sample_rate=self.model_args.sample_rate,
            embedding_size=self.model_args.embedding_size,
            prefix=self.args.root / "output",
        )

        self.opt_backbone = torch.optim.SGD(
            params=[{"params": self.backbone.parameters()}],
            lr=self.model_args.opt_backbone.lr
            / 512
            * self.model_args.batch_size
            * self.world_size,
            momentum=self.model_args.opt_backbone.momentum,
            weight_decay=self.model_args.opt_backbone.weight_decay,
        )

        self.opt_pfc = torch.optim.SGD(
            params=[{"params": self.module_partial_fc.parameters()}],
            lr=self.model_args.opt_pfc.lr
            / 512
            * self.model_args.batch_size
            * self.world_size,
            momentum=self.model_args.opt_pfc.momentum,
            weight_decay=self.model_args.opt_pfc.weight_decay,
        )

        self.scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.opt_backbone
        )
        self.scheduler_pfc = torch.optim.lr_scheduler.LambdaLR(optimizer=self.opt_pfc)

    def _data_loader(self):
        self.trainset = MXFaceDataset(self.dataset_args.path)
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.trainset, shuffle=True
        )
        self.train_loader = DataLoaderX(
            local_rank=self.local_rank,
            dataset=self.trainset,
            batch_size=self.model_args.batch_size,
            sampler=self.train_sampler if self.model_args.is_distributed else None,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        self.total_step = int(
            len(self.trainset)
            / self.model_args.batch_size
            / self.world_size
            * self.model_args.num_epoch
        )
        if self.rank is 0:
            logging.info("Total Step is: %d" % self.total_step)

        self.callback_verification = face_lib.utils.CallBackVerification(
            2000, self.rank, self.model_args.val_targets, self.model_args.rec
        )
        self.callback_logging = face_lib.utils.CallBackLogging(
            50,
            self.rank,
            self.total_step,
            self.model_args.batch_size,
            self.world_size,
            None,
        )
        self.callback_checkpoint = face_lib.utils.CallBackModelCheckpoint(
            self.rank, self.args.root / "output"
        )

    def _main_loop(self):
        loss = face_lib.utils.AverageMeter()
        global_step = 0
        grad_scaler = (
            face_lib.utils.MaxClipGradScaler(
                self.model_args.batch_size,
                128 * self.model_args.batch_size,
                growth_interval=100,
            )
            if self.model_args.fp16
            else None
        )
        for epoch in range(self.start_epoch, self.model_args.num_epoch):
            self.train_sampler.set_epoch(epoch)
            for step, (img, label) in enumerate(self.train_loader):
                global_step += 1
                features = torch.nn.functional.normalize(self.backbone(img))
                x_grad, loss_v = self.module_partial_fc.forward_backward(
                    label, features, self.opt_pfc
                )
                if self.model_args.fp16:
                    features.backward(grad_scaler.scale(x_grad))
                    grad_scaler.unscale_(self.opt_backbone)
                    torch.nn.utils.clip_grad_norm_(
                        self.backbone.parameters(), max_norm=5, norm_type=2
                    )
                    grad_scaler.step(self.opt_backbone)
                    grad_scaler.update()
                else:
                    features.backward(x_grad)
                    torch.nn.utils.clip_grad_norm_(
                        self.backbone.parameters(), max_norm=5, norm_type=2
                    )
                    self.opt_backbone.step()

                self.opt_pfc.step()
                self.module_partial_fc.update()
                self.opt_backbone.zero_grad()
                self.opt_pfc.zero_grad()
                loss.update(loss_v, 1)
                self.callback_logging(
                    global_step, loss, epoch, self.model_args.fp16, grad_scaler
                )
                self.callback_verification(global_step, self.backbone)
            self.callback_checkpoint(global_step, self.backbone, self.module_partial_fc)
            self.scheduler_backbone.step()
            self.scheduler_pfc.step()
        dist.destroy_process_group()

    def _model_evaluate(self, epoch=0):
        pass

    def _model_train(self, epoch=0):
        pass

    def _report_settings(self):
        pass


if __name__ == "__main__":
    args = training_args()
    writer = SummaryWriter(args.root)
    faceu = Trainer(args, writer)
    faceu.train_runner()
