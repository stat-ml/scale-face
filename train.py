import os
import sys
import time
import torch
import random
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
import torch.nn.functional as F

from utils.dataset import Dataset
from utils.imageprocessing import preprocess
from utils import cfg
from utils import optims

import model as mlib
from parser_cfg import training_args

torch.backends.cudnn.bencmark = True

from tqdm import tqdm


class Trainer:
    def __init__(self, args):
        self.args = args
        # Load configurations
        self.model_args = cfg.load_config(args.model_config)
        self.optim_args = cfg.load_config(args.optimizer_config)
        self.dataset_args = cfg.load_config(args.dataset_config)
        self.env_args = cfg.load_config(args.env_config)

        self.model = dict()
        self.data = dict()

        # TODO: add distributed
        self.device = (
            "cuda" if (self.env_args.use_gpu and torch.cuda.is_available) else "cpu"
        )

        # create directory for experiment

        self.checkpoints_path = self.args.root / "checkpoints"
        os.makedirs(self.checkpoints_path)

    def _model_loader(self):
        self.model["backbone"] = mlib.model_dict[self.model_args.backbone]()
        if self.args.resume:
            model_dict = torch.load(args.resume, map_location=self.device)
            self.start_epoch = model_dict["epoch"]
            self.model["backbone"].load_state_dict(model_dict["backbone"])

        self.model["uncertain"] = mlib.UncertaintyHead(self.model_args.in_feats)

        if self.args.resume:
            self.model["uncertain"].load_state_dict(model_dict["uncertain"])

        self.model["criterion"] = mlib.criterions_dict[self.model_args.criterion.name](
            mean=self.model_args.criterion.mean
        )

        if self.args.freeze_backbone:
            for p in self.model["backbone"].parameters():
                p.requires_grad = False

        self.model["optimizer"] = optims.optimizers_map[self.optim_args.name](
            [{"params": self.model["uncertain"].parameters()}],
            lr=self.optim_args.base_lr,
            weight_decay=self.optim_args.weight_decay,
            momentum=self.optim_args.momentum,
            nesterov=self.optim_args.nesterov,
        )

        if self.device:
            self.model["backbone"] = self.model["backbone"].to(self.device)
            self.model["uncertain"] = self.model["uncertain"].to(self.device)
            self.model["criterion"] = self.model["criterion"].to(self.device)

    def _data_loader(self):
        batch_format = {
            "size": self.args.batch_size,
            "num_classes": self.args.num_classes_batch,
        }
        self.trainset = Dataset(self.dataset_args.path)
        proc_func = lambda images: preprocess(images, self.dataset_args.in_size, True)
        self.trainset.start_batch_queue(batch_format, proc_func=proc_func)

    def _model_train(self, epoch=0):

        if self.args.freeze_backbone:
            self.model["backbone"].eval()
        self.model["uncertain"].train()

        loss_recorder, batch_acc = [], []
        for idx in range(self.args.iterations):
            self.model["optimizer"].zero_grad()

            batch = self.trainset.pop_batch_queue()
            img = torch.from_numpy(batch["image"]).permute(0, 3, 1, 2).to(self.device)
            gty = torch.from_numpy(batch["label"]).to(self.device)

            feature, sig_feat = self.model["backbone"](img)
            log_sig_sq = self.model["uncertain"](sig_feat)
            loss = self.model["criterion"](feature, log_sig_sq, gty)

            loss.backward()
            self.model["optimizer"].step()

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
        train_loss = np.mean(loss_recorder)
        print("train_loss : %.4f" % train_loss)
        return train_loss

    def _main_loop(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            train_loss = self._model_train(epoch)

            if min_train_loss > train_loss:
                print("%snew SOTA was found%s" % ("*" * 16, "*" * 16))
                min_train_loss = train_loss
                filename = os.path.join(self.checkpoints_path, "sota.pth.tar")
                torch.save(
                    {
                        "epoch": epoch,
                        "backbone": self.model["backbone"].state_dict(),
                        "uncertain": self.model["uncertain"].state_dict(),
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
                        "backbone": self.model["backbone"].state_dict(),
                        "uncertain": self.model["uncertain"].state_dict(),
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
