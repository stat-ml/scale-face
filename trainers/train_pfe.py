import os
import sys
import torch
import torch.distributed as dist
import numpy as np
import torchvision

sys.path.append(".")
from face_lib.datasets import MS1MDatasetPFE, DataLoaderX, ms1m_collate_fn
from face_lib.utils import FACE_METRICS
from face_lib import models as mlib, utils
from face_lib.parser_cfg import training_args
from face_lib.trainer import TrainerBase

from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.bencmark = True


def _set_evaluation_metric_yaml(config: dict):
    metric_name = config.name
    name_elem = config["name"]
    config.pop("name")
    metric = lambda model: FACE_METRICS[metric_name](model, **config)
    config["name"] = name_elem
    return metric


class Trainer(TrainerBase):
    def _model_loader(self):
        self.backbone = mlib.model_dict[self.model_args.backbone["name"]](
            **utils.pop_element(self.model_args.backbone, "name"),
        )

        if self.model_args.head:
            self.head = mlib.heads[self.model_args.head.name](
                **utils.pop_element(self.model_args.head, "name"),
            )
            self.head_criterion = mlib.criterions_dict[
                self.model_args.head.criterion.name
            ](
                **utils.pop_element(self.model_args.head.criterion, "name"),
            )

        self.start_epoch = 0
        if self.args.resume:
            model_dict = torch.load(self.args.resume, map_location=self.device)
            self.start_epoch = model_dict["epoch"]
            self.backbone.load_state_dict(model_dict["backbone"])
            self.head.load_state_dict(model_dict["head"])

        if self.model_args.pretrained_backbone and self.args.resume is None:
            backbone_dict = torch.load(
                self.model_args.pretrained_backbone, map_location=self.device
            )
            self.backbone.load_state_dict(backbone_dict)

        # TODO: we can write nn.Module wrapper to deal with parametrization like
        # freeze and etc.

        if self.model_args.backbone.learnable is False:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        if self.model_args.head and self.model_args.head.learnable is False:
            for p in self.head.parameters():
                p.requires_grad = False
            self.head.eval()

        learnable_parameters = []
        if self.model_args.backbone.learnable is True:
            learnable_parameters += list(self.backbone.parameters())
        if self.model_args.head and self.model_args.head.learnable is True:
            learnable_parameters += list(self.head.parameters())

        self.optimizer = utils.optimizers_map[self.model_args.optimizer.name](
            [{"params": learnable_parameters}],
            **utils.pop_element(self.model_args.optimizer, "name"),
        )

        self.scheduler = utils.scheduler_map[self.model_args.scheduler.name](
            self.optimizer,
            **utils.pop_element(self.model_args.scheduler, "name")
        )

        if self.device:
            self.backbone = self.backbone.to(self.device)
            if self.head:
                self.head = self.head.to(self.device)

        if self.model_args.is_distributed:
            for p in self.head.parameters():
                dist.broadcast(p, 0)
            self.head = self.head = torch.nn.parallel.DistributedDataParallel(
                module=self.head, broadcast_buffers=False,
                device_ids=[self.local_rank])
            self.head.train()

        # set evaluation metrics
        self.evaluation_metrics, self.evaluation_configs = [], []
        if len(self.model_args.evaluation_configs) > 0:
            for item in self.model_args.evaluation_configs:
                self.evaluation_metrics.append(_set_evaluation_metric_yaml(item))
                self.evaluation_configs.append(item)

        print("Model loading was finished")

    def _data_loader(self):
        if self.model_args.dataset.name == 'ms1m':
            self.trainset = MS1MDatasetPFE(
                root_dir=self.model_args.dataset.path,
                num_face_pb=self.model_args.dataset.num_face_pb,
                local_rank=self.rank)

            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset, shuffle=True)

            self.trainloader = DataLoaderX(
                local_rank=self.rank, dataset=self.trainset,
                batch_size=self.model_args.dataset.batch_size, sampler=train_sampler,
                num_workers=0, pin_memory=True, drop_last=True, collate_fn=ms1m_collate_fn)
        else:
            raise NotImplementedError("Dataset is not implemented")

        print("Data loading was finished ...")

    @torch.no_grad()
    def _model_evaluate(self, epoch=0):
        self.backbone.eval()
        if self.model_args.head:
            self.head.eval()
        for metric in self.evaluation_configs:
            if metric.name == "lfw_6000_pairs":
                pass
                # Calculating accuracy does not seem reasonable in terms of PFE
                # utils.accuracy_lfw_6000_pairs(
                #     self.backbone,
                #     self.head,
                #     metric.lfw_path,
                #     metric.lfw_pairs_txt_path,
                #     N=metric.N,
                #     n_folds=metric.n_folds,
                #     device=self.device,
                #     board=True,
                #     board_writer=self.board,
                #     board_iter=epoch,
                # )

    def _model_train(self, epoch=0):
        if self.model_args.backbone.learnable:
            self.backbone.train()
        else:
            self.backbone.eval()
        if self.model_args.head.learnable:
            self.head.train()
        else:
            self.head.eval()
        self.head_criterion.train()

        loss_recorder, batch_acc = [], []
        for idx, (img, gty) in enumerate(self.trainloader):

            img.requires_grad = False
            gty.requires_grad = False
            if self.device:
                img = img.to(self.device)
                gty = gty.to(self.device)

            feature, sig_feat = self.backbone(img)

            print(sig_feat.view(sig_feat.size(0), -1).size())
            print(sig_feat.size())
            sig_feat_dict = {"bottleneck_feature":  sig_feat}

            log_sig_sq = self.head(**sig_feat_dict)

            # Create argument dict for ProbLoss
            outputs = {"gty": gty}
            outputs.update({"feature", feature})
            outputs.update(log_sig_sq)

            loss = self.head_criterion.forward(self.device, feature, gty, log_sig_sq)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_recorder.append(loss.item())
            train_loss = float(np.mean(loss_recorder))
            if (idx + 1) % self.model_args.logging.print_freq == 0:
                print(
                    "epoch : %2d|%2d, iter : %4d|%4d,  loss : %.4f"
                    % (
                        epoch,
                        self.model_args.epochs,
                        idx + 1,
                        len(self.trainloader),
                        train_loss,
                    )
                )
        print("train_loss : %.4f" % train_loss)
        return train_loss

    def _main_loop(self):
        min_train_loss = self.__class__._INF

        for epoch in range(self.start_epoch, self.model_args.epochs):
            print(f"{('*' * 16)}Epoch {epoch}{('*' * 16)}")
            train_loss = self._model_train(epoch)
            self._model_evaluate(epoch)

            if min_train_loss > train_loss:
                print("%sNew SOTA was found%s" % ("*" * 16, "*" * 16))
                min_train_loss = train_loss
                filename = os.path.join(self.checkpoints_path, "sota.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "backbone": self.backbone.state_dict(),
                        "head": self.head.module.state_dict() if \
                            self.model_args.is_distributed else \
                            self.head.state_dict(),
                        "train_loss": min_train_loss,
                    },
                    filename,
                )

            if epoch % self.model_args.logging.save_freq == 0:
                filename = "epoch_%d_train_loss_%.4f.pth" % (epoch, train_loss)
                savename = os.path.join(self.checkpoints_path, filename)
                torch.save(
                    {
                        "epoch": epoch,
                        "backbone": self.backbone.state_dict(),
                        "head": self.head.module.state_dict() if \
                            self.model_args.is_distributed else \
                            self.head.state_dict(),
                        "train_loss": train_loss,
                    },
                    savename,
                )
        print("Finished training")

    def _report_settings(self):
        str = "-" * 16
        print("%sEnvironment Versions%s" % (str, str))
        print("- Python    : {}".format(sys.version.strip().split("|")[0]))
        print("- PyTorch   : {}".format(torch.__version__))
        print("- TorchVison: {}".format(torchvision.__version__))
        print("- USE_GPU   : {}".format(self.device))
        print("-" * 52)
        print("- Backbone   : {}".format(self.backbone.__class__))
        print("- Head   : {}".format(self.head))
        print("- Backbone Criterion   : {}".format(self.backbone_criterion))
        print("- Head Criterion   : {}".format(self.head_criterion))
        print("-" * 52)


if __name__ == "__main__":
    args = training_args()
    writer = SummaryWriter(args.root)
    faceu = Trainer(args, writer)
    faceu.train_runner()
