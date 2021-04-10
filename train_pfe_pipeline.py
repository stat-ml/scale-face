import os
import sys
import torch
import numpy as np
import torchvision

import model.losses
import utils
from utils.dataset import Dataset
from utils.imageprocessing import preprocess
from utils import FACE_METRICS
from utils import cfg

import model as mlib
from parser_cfg import training_args

torch.backends.cudnn.bencmark = True


def _set_evaluation_metric_yaml(yaml_path: str):
    config = cfg.load_config(yaml_path)
    metric_name = config.name
    config.pop("name")
    metric = lambda model: FACE_METRICS[metric_name](model, **config)
    return metric


class Trainer:
    def __init__(self, args):
        self.args = args
        # Load configurations
        self.model_args = cfg.load_config(args.model_config)
        self.optim_args = cfg.load_config(args.optimizer_config)
        self.dataset_args = cfg.load_config(args.dataset_config)
        self.env_args = cfg.load_config(args.env_config)

        self.backbone = None
        self.head = None

        self.model = dict()
        self.data = dict()

        # TODO: add distributed
        self.device = (
            "cuda" if (self.env_args.use_gpu and torch.cuda.is_available) else "cpu"
        )
        # create directory for experiment

        self.checkpoints_path = self.args.root / "checkpoints"
        os.makedirs(self.checkpoints_path)

        # set evaluation metrics
        self.evaluation_metrics = []
        if len(self.args.evaluation_configs) > 0:
            for item in self.args.evaluation_configs:
                self.evaluation_metrics.append(_set_evaluation_metric_yaml(item))

    def _model_loader(self):
        self.backbone = mlib.model_dict[self.model_args.backbone.name](
            **utils.pop_element(self.model_args.backbone, "name"),
        )

        if self.model_args.head:
            self.head = mlib.heads[self.model_args.head.name](
                **utils.pop_element(self.model_args.head, "name"),
            )

        self.criterion = mlib.criterions_dict[self.model_args.criterion.name](
            **utils.pop_element(self.model_args.criterion, "name"),
        )

        self.start_epoch = 0
        if self.args.resume:
            model_dict = torch.load(self.args.resume, map_location=self.device)
            self.start_epoch = model_dict["epoch"]
            self.backbone.load_state_dict(model_dict["backbone"])
            self.head.load_state_dict(model_dict["head"])

        if self.args.pretrained_backbone and self.args.resume is None:
            backbone_dict = torch.load(
                self.args.pretrained_backbone, map_location=self.device
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

    def _data_loader(self):
        batch_format = {
            "size": self.args.batch_size,
            "num_classes": self.args.num_classes_batch,
        }
        train_proc_func = lambda images: preprocess(
            images, self.dataset_args.in_size, is_training=True
        )
        self.trainset = Dataset(self.dataset_args.path, preprocess_func=train_proc_func)
        self.trainset.start_batch_queue(batch_format)

    def _model_train(self, epoch=0):
        if self.model_args.backbone.learnable is True:
            self.backbone.train()
        if self.model_args.head and self.model_args.head.learnable is True:
            self.head.train()

        loss_recorder, batch_acc = [], []
        for idx in range(self.args.iterations):
            self.optimizer.zero_grad()

            batch = self.trainset.pop_batch_queue()
            img = torch.from_numpy(batch["image"]).permute(0, 3, 1, 2).to(self.device)
            gty = torch.from_numpy(batch["label"]).to(self.device)

            outputs = {"gty": gty}
            outputs.update(self.backbone(img))
            if self.head:
                outputs.update(self.head(**outputs))
            loss = self.criterion(**outputs)

            loss.backward()
            self.optimizer.step()

            loss_recorder.append(loss.item())

            if (idx + 1) % self.args.print_freq == 0:
                print(
                    "epoch : %2d|%2d, iter : %2d|%2d, loss : %.4f"
                    % (
                        epoch,
                        self.args.epochs,
                        idx,
                        self.args.iterations,
                        np.mean(loss_recorder),
                    )
                )
            if self.args.debug:
                # break loop if debug flag is True
                break
        train_loss = np.mean(loss_recorder)
        print("train_loss : %.4f" % train_loss)
        return train_loss

    def _main_loop(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            train_loss = self._model_train(epoch)

            import pdb

            pdb.set_trace()

            if min_train_loss > train_loss:
                print("%snew SOTA was found%s" % ("*" * 16, "*" * 16))
                min_train_loss = train_loss
                filename = os.path.join(self.checkpoints_path, "sota.pth.tar")
                torch.save(
                    {
                        "epoch": epoch,
                        "backbone": self.backbone.state_dict(),
                        "head": self.head.state_dict(),
                        "train_loss": min_train_loss,
                    },
                    filename,
                )

            if epoch % self.args.save_freq == 0:
                filename = "epoch_%d_train_loss_%.4f.pth.tar" % (epoch, train_loss)
                savename = os.path.join(self.checkpoints_path, filename)
                torch.save(
                    {
                        "epoch": epoch,
                        "backbone": self.backbone.state_dict(),
                        "head": self.head.state_dict(),
                        "train_loss": train_loss,
                    },
                    savename,
                )

    def train_runner(self):
        self._report_settings()
        self._model_loader()
        self._data_loader()
        self._main_loop()

    def _report_settings(self):
        str = "-" * 16
        print("%sEnvironment Versions%s" % (str, str))
        print("- Python    : {}".format(sys.version.strip().split("|")[0]))
        print("- PyTorch   : {}".format(torch.__version__))
        print("- TorchVison: {}".format(torchvision.__version__))
        print("- USE_GPU   : {}".format(self.device))
        print("-" * 52)


if __name__ == "__main__":
    faceu = Trainer(training_args())
    faceu.train_runner()
