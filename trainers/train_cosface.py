import os
import sys
import torch
import numpy as np
import torchvision

sys.path.append(".")
from face_lib.utils import Dataset, cfg, FACE_METRICS
from face_lib.utils.imageprocessing import preprocess
from face_lib import models as mlib, utils
from face_lib.parser_cfg import training_args
from face_lib.trainer import TrainerBase

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
    def _model_loader(self):
        self.backbone = mlib.model_dict[self.model_args.backbone.name](
            **utils.pop_element(self.model_args.backbone, "name"),
        )
        if self.model_args.backbone.criterion:
            self.backbone_criterion = mlib.criterions_dict[
                self.model_args.backbone.criterion.name
            ](
                **utils.pop_element(self.model_args.backbone.criterion, "name"),
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

        learnable_parameters = (
            list(self.backbone.parameters())
            if self.model_args.backbone.learnable is True
            else []
        )
        learnable_parameters += (
            list(self.head.parameters())
            if self.model_args.head and self.model_args.head.learnable is True
            else []
        )

        self.optimizer = utils.optimizers_map[self.model_args.optimizer.name](
            [{"params": learnable_parameters}],
            **utils.pop_element(self.model_args.optimizer, "name"),
        )

        if self.device:
            self.backbone = self.backbone.to(self.device)
            if self.head:
                self.head = self.head.to(self.device)

        # set evaluation metrics
        self.evaluation_metrics, self.evaluation_configs = [], []
        if len(self.model_args.evaluation_configs) > 0:
            for item in self.model_args.evaluation_configs:
                self.evaluation_metrics.append(_set_evaluation_metric_yaml(item))
                self.evaluation_configs.append(item)

    def _data_loader(self):
        batch_format = {
            "size": self.model_args.batch_size,
            "num_classes": self.model_args.batch_size
            // self.model_args.num_classes_batch,
        }
        train_proc_func = lambda images: preprocess(
            images, self.model_args.in_size, is_training=True
        )
        self.trainset = Dataset(
            self.model_args.path_list, preprocess_func=train_proc_func
        )
        self.trainset.start_batch_queue(batch_format)

    @torch.no_grad()
    def _model_evaluate(self, epoch=0):
        self.backbone.eval()
        if self.model_args.head:
            self.head.eval()
        for metric in self.evaluation_configs:
            if metric.name == "lfw_6000_pairs":
                utils.accuracy_lfw_6000_pairs(
                    self.backbone,
                    self.head,
                    metric.lfw_path,
                    metric.lfw_pairs_txt_path,
                    N=metric.N,
                    n_folds=metric.n_folds,
                    device=self.device,
                    board=True,
                    board_writer=self.board,
                    board_iter=epoch,
                )

    def _model_train(self, epoch=0):
        if self.model_args.backbone.learnable is True:
            self.backbone.train()
        if self.model_args.head and self.model_args.head.learnable is True:
            self.head.train()

        loss_recorder, batch_acc = [], []
        for idx in range(self.model_args.iterations):
            _global_iteration = epoch * self.model_args.iterations + idx
            self.optimizer.zero_grad()

            batch = self.trainset.pop_batch_queue()
            img = torch.from_numpy(batch["image"]).permute(0, 3, 1, 2).to(self.device)
            gty = torch.from_numpy(batch["label"]).to(self.device)

            outputs = {"gty": gty}

            outputs.update(self.backbone(img))

            if self.head:
                outputs.update(self.head(**outputs))
                loss = self.head_criterion(device=self.device, **outputs)
            else:
                loss = self.backbone_criterion(**outputs)
            loss.backward()
            self.optimizer.step()
            loss_recorder.append(loss.item())

            if (idx + 1) % self.model_args.logging.print_freq == 0 or self.args.debug:
                print(
                    "epoch : %2d|%2d, iter : %2d|%2d, loss : %.4f"
                    % (
                        epoch,
                        self.model_args.epochs,
                        idx,
                        self.model_args.iterations,
                        np.mean(loss_recorder),
                    )
                )
                self.board.add_scalar(
                    f"train/{self.head_criterion if self.head else self.backbone_criterion}_loss_mean",
                    np.mean(loss_recorder),
                    _global_iteration,
                )

                if (idx + 1) % (self.model_args.logging.print_freq * 50) == 0:
                    for metric in self.evaluation_configs:
                        if metric.name == "lfw_dilemma":
                            visual_img = utils.visualize_ambiguity_dilemma_lfw(
                                self.backbone,
                                self.backbone_criterion,
                                metric.lfw_path,
                                pfe_head=self.head,
                                criterion_head=self.head_criterion,
                                board=True,
                                device=self.device,
                            )
                            self.board.add_image(
                                "ambiguity_dilemma_lfw",
                                visual_img.transpose(2, 0, 1),
                                _global_iteration,
                            )
                        if metric.name == "lfw_dilemma":
                            pass
                            """
                            utils.visualize_low_high_similarity_pairs(
                                self.backbone,
                                self.backbone_criterion,
                                metric.lfw_path,
                                metric.lfw_pairs_txt_path,
                                pfe_head=self.head,
                                criterion_head=self.head_criterion,
                                board=True,
                                device=self.device,
                            )
                            """
            if self.args.debug:
                # break loop if debug flag is True
                break
        train_loss = np.mean(loss_recorder)
        print("train_loss : %.4f" % train_loss)
        return train_loss

    def _main_loop(self):
        min_train_loss = self.__class__._INF

        for epoch in range(self.start_epoch, self.model_args.epochs):
            train_loss = self._model_train(epoch)
            self._model_evaluate(epoch)
            if min_train_loss > train_loss:
                print("%snew SOTA was found%s" % ("*" * 16, "*" * 16))
                min_train_loss = train_loss
                filename = os.path.join(self.checkpoints_path, "sota.pth.tar")
                torch.save(
                    {
                        "epoch": epoch,
                        "backbone": self.backbone.state_dict(),
                        "head": self.head.state_dict() if self.head else None,
                        "train_loss": min_train_loss,
                    },
                    filename,
                )

            if epoch % self.model_args.logging.save_freq == 0:
                filename = "epoch_%d_train_loss_%.4f.pth.tar" % (epoch, train_loss)
                savename = os.path.join(self.checkpoints_path, filename)
                torch.save(
                    {
                        "epoch": epoch,
                        "backbone": self.backbone.state_dict(),
                        "head": self.head.state_dict() if self.head else None,
                        "train_loss": train_loss,
                    },
                    savename,
                )

    def _report_settings(self):
        str = "-" * 16
        print("%sEnvironment Versions%s" % (str, str))
        print("- Python    : {}".format(sys.version.strip().split("|")[0]))
        print("- PyTorch   : {}".format(torch.__version__))
        print("- TorchVison: {}".format(torchvision.__version__))
        print("- USE_GPU   : {}".format(self.device))
        print("-" * 52)
        print("- Backbone   : {}".format(self.backbone))
        print("- Head   : {}".format(self.head))
        print("- Backbone Criterion   : {}".format(self.backbone_criterion))
        print("- Head Criterion   : {}".format(self.head_criterion))
        print("-" * 52)


if __name__ == "__main__":
    args = training_args()
    writer = SummaryWriter(args.root)
    faceu = Trainer(args, writer)
    faceu.train_runner()
