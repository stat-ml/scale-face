import os
import sys
import logging
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
from torch.nn.utils import clip_grad_norm_

sys.path.append(".")
from face_lib import models as mlib, utils
from face_lib.models import PartialFC
from face_lib.parser_cfg import parse_args_scale
from face_lib.trainer import TrainerBase
from face_lib.utils import FACE_METRICS
from face_lib.utils.utils_amp import MaxClipGradScaler
from face_lib.utils.utils_logging import AverageMeter, init_logging
from face_lib.utils.utils_callback import (
    CallBackVerification, CallBackLogging, CallBackModelCheckpoint)
from face_lib.datasets import (
    MXFaceDataset, MXFaceDatasetDistorted,
    MXFaceDatasetGauss, SyntheticDataset,
    DataLoaderX, ProductsDataset)

from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True


def _set_evaluation_metric_yaml(config: dict):
    metric_name = config.name
    name_elem = config["name"]
    config.pop("name")
    metric = lambda model: FACE_METRICS[metric_name](model, **config)
    config["name"] = name_elem
    return metric


class Trainer(TrainerBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        init_logging(
            log_root=logging.getLogger(),
            rank=self.local_rank,
            models_root=self.model_args.output)

    def _data_loader(self):

        if not "dataset" in self.model_args:
            raise KeyError("Do not have dataset name in config")

        if self.model_args.dataset == "synthetic":
            self.train_set = SyntheticDataset(
                local_rank=self.local_rank)
        elif self.model_args.dataset == "distortion":
            self.train_set = MXFaceDatasetDistorted(
                root_dir=self.model_args.rec,
                local_rank=self.local_rank)
        elif self.model_args.dataset == "gauss":
            self.train_set = MXFaceDatasetGauss(
                root_dir=self.model_args.rec,
                local_rank=self.local_rank)
        elif self.model_args.dataset == "ms1m":
            self.train_set = MXFaceDataset(
                root_dir=self.model_args.rec,
                local_rank=self.local_rank)
        elif self.model_args.dataset == "products":
            self.train_set = ProductsDataset(
                root_dir=self.model_args.rec,
                local_rank=self.local_rank
            )
        else:
            raise KeyError("Don't know this name of dataset")

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_set,
            shuffle=True)
        self.train_loader = DataLoaderX(
            local_rank=self.local_rank,
            dataset=self.train_set,
            batch_size=self.model_args.batch_size,
            sampler=self.train_sampler,
            num_workers=2,
            pin_memory=True,
            drop_last=True)

    def _model_loader(self):

        self.backbone = mlib.model_dict[self.model_args.backbone.name](
            **utils.pop_element(self.model_args.backbone, "name"),
        )
        self.scale_predictor = mlib.scale_predictors[self.model_args.scale_predictor.name](
            **utils.pop_element(self.model_args.scale_predictor, "name")
        )
        self.backbone = self.backbone.to(self.device)
        self.scale_predictor = self.scale_predictor.to(self.device)

        if self.model_args.resume:
            try:
                backbone_pth = os.path.join(self.model_args.source, "backbone.pth")
                self.backbone.load_state_dict(
                    torch.load(backbone_pth, map_location=torch.device(self.local_rank)))
                if self.local_rank == 0:
                    logging.info("backbone resume successfully!")
            except (FileNotFoundError, KeyError, IndexError, RuntimeError):
                if self.local_rank == 0:
                    logging.info("resume fail, backbone init successfully!")

            if "scale_source" in self.model_args and self.model_args.scale_source is not None:
                scale_ckpt = torch.load(
                    self.model_args.scale_source, map_location=torch.device(self.local_rank))
                self.scale_predictor.load_state_dict(scale_ckpt["scale_predictor"])
                logging.info("Initialized scale_predictor with a pretrained model")
            else:
                logging.info("Initialized scale_predictor from the scratch")

        if not self.model_args.freeze_backbone:
            self.backbone = torch.nn.parallel.DistributedDataParallel(
                module=self.backbone, broadcast_buffers=False, device_ids=[self.local_rank])
        for p in self.scale_predictor.parameters():
            dist.broadcast(p, 0)
        self.scale_predictor = torch.nn.parallel.DistributedDataParallel(
            module=self.scale_predictor, broadcast_buffers=False, device_ids=[self.local_rank])

        if self.model_args.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        else:
            self.backbone.train()
        self.scale_predictor.train()

        criterion = mlib.criterions_dict[self.model_args.loss]()
        self.module_partial_fc = PartialFC(
            rank=self.rank,
            local_rank=self.local_rank,
            world_size=self.world_size,
            resume=self.model_args.resume,
            batch_size=self.model_args.batch_size,
            margin_softmax=criterion,
            num_classes=self.model_args.num_classes,
            sample_rate=self.model_args.sample_rate,
            embedding_size=self.model_args.embedding_size,
            prefix=os.path.join(self.model_args.output, "checkpoints"),
            source=self.model_args.source)


        self.learnable_parameters = []
        if not self.model_args.freeze_backbone:
            self.learnable_parameters += list(self.backbone.parameters())
        if self.model_args.loss == "arcface_scale":
            self.learnable_parameters += list(self.scale_predictor.parameters())

        param_groups = []
        if not self.model_args.freeze_backbone:
            param_groups.append({
                "params": self.backbone.parameters(),
                "lr": self.model_args.lr / 512 * self.model_args.batch_size * self.world_size,
                "momentum": self.model_args.momentum,
                "weight_decay": self.model_args.weight_decay
            })
        if self.model_args.loss == "arcface_scale":
            param_groups.append({
                "params": self.scale_predictor.parameters(),
                "lr": self.model_args.scale_lr / 512 * self.model_args.batch_size * self.world_size,
                "momentum": self.model_args.momentum,
                "weight_decay": self.model_args.weight_decay
            })

        self.opt_backbone = torch.optim.SGD(param_groups)

        if self.model_args.freeze_backbone:
            for p in self.module_partial_fc.parameters():
                p.requires_grad = False
        self.opt_pfc = torch.optim.SGD(
            params=[{'params': self.module_partial_fc.parameters()}],
            lr=self.model_args.lr / 512 * self.model_args.batch_size * self.world_size,
            momentum=self.model_args.momentum,
            weight_decay=self.model_args.weight_decay)

        num_image = len(self.train_set)
        total_batch_size = self.model_args.batch_size * self.world_size
        self.warmup_step = num_image // total_batch_size * self.model_args.warmup_epoch
        self.total_step = num_image // total_batch_size * self.model_args.num_epoch

        def lr_step_func(current_step):
            decay_step = [x * num_image // total_batch_size for x in self.model_args.decay_epoch]
            if current_step < self.warmup_step:
                return current_step / self.warmup_step
            else:
                return 0.1 ** len([m for m in decay_step if m <= current_step])

        self.scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.opt_backbone, lr_lambda=lr_step_func)

        if not self.model_args.freeze_backbone:
            self.scheduler_pfc = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.opt_pfc, lr_lambda=lr_step_func)

        self.grad_amp = MaxClipGradScaler(
            self.model_args.batch_size,
            128 * self.model_args.batch_size,
            growth_interval=100
        ) if self.model_args.fp16 else None

    def _model_evaluate(self, epoch=0):
        pass
    # @torch.no_grad()
    # def _model_evaluate(self, epoch=0):
    #     self.backbone.eval()
    #     if self.model_args.head:
    #         self.head.eval()
    #     for metric in self.evaluation_configs:
    #         if metric.name == "lfw_6000_pairs":
    #             utils.accuracy_lfw_6000_pairs(
    #                 self.backbone,
    #                 self.head,
    #                 metric.lfw_path,
    #                 metric.lfw_pairs_txt_path,
    #                 N=metric.N,
    #                 n_folds=metric.n_folds,
    #                 device=self.device,
    #                 board=True,
    #                 board_writer=self.board,
    #                 board_iter=epoch,
    #             )

    def _model_train(self, epoch=0):
        pass

    def _main_loop(self):

        val_target = self.model_args.val_targets
        callback_verification = CallBackVerification(
            2000,
            self.rank,
            val_target,
            self.model_args.rec)
        callback_logging = CallBackLogging(
            5,
            self.rank,
            self.total_step,
            self.model_args.batch_size,
            self.world_size,
            None)
        save_dir = os.path.join(self.model_args.output, "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        callback_checkpoint = CallBackModelCheckpoint(
            self.rank,
            save_dir)

        loss = AverageMeter()
        start_epoch = 0
        global_step = 0

        for epoch in range(start_epoch, self.model_args.num_epoch):

            self.train_sampler.set_epoch(epoch)

            for step, (img, label) in enumerate(self.train_loader):
                global_step += 1

                output = self.backbone(img)
                output.update(self.scale_predictor(**output))

                features = F.normalize(output["feature"], dim=1)
                scale = output["scale"]

                x_grad, s_grad, loss_v = self.module_partial_fc.forward_backward(
                    label, features, self.opt_pfc, scale=scale)

                if self.model_args.fp16:
                    if not self.model_args.freeze_backbone:
                        features.backward(self.grad_amp.scale(x_grad), retain_graph=True)
                    scale.backward(self.grad_amp.scale(s_grad))

                    self.grad_amp.unscale_(self.opt_backbone)
                    clip_grad_norm_(self.learnable_parameters, max_norm=5, norm_type=2)
                    self.grad_amp.step(self.opt_backbone)
                    self.grad_amp.update()

                else:
                    features.backward(x_grad)
                    scale.backward(self.grad_amp.scale(s_grad))
                    clip_grad_norm_(self.learnable_parameters, max_norm=5, norm_type=2)
                    self.opt_backbone.step()

                self.opt_pfc.step()
                self.module_partial_fc.update()
                self.opt_backbone.zero_grad()
                self.opt_pfc.zero_grad()
                loss.update(loss_v, 1)
                callback_logging(
                    global_step=global_step,
                    loss=loss,
                    epoch=epoch,
                    fp16=self.model_args.fp16,
                    learning_rate=self.scheduler_backbone.get_last_lr()[0],
                    grad_scaler=self.grad_amp)
                callback_verification(global_step, self.backbone)
                self.scheduler_backbone.step()
                if not self.model_args.freeze_backbone:
                    self.scheduler_pfc.step()

            callback_checkpoint(
                global_step,
                self.backbone,
                self.module_partial_fc,
                scale_predictor=self.scale_predictor)

    def _report_settings(self):
        if self.local_rank == 0:
            delimiter_str = "-" * 16
            print("%sEnvironment Versions%s" % (delimiter_str, delimiter_str))
            print("- Python    : {}".format(sys.version.strip().split("|")[0]))
            print("- PyTorch   : {}".format(torch.__version__))
            print("- TorchVison: {}".format(torchvision.__version__))
            print("- USE_GPU   : {}".format(self.device))
            print("-" * 52)
            print("- Backbone   : {}".format(self.backbone))
            print("- Head   : {}".format(self.scale_predictor))

            for key, value in self.model_args.items():
                num_space = 25 - len(key)
                print(": " + key + " " * num_space + str(value))
            print("-" * 52)


if __name__ == "__main__":
    args = parse_args_scale()
    faceu = Trainer(args)
    faceu.train_runner()
