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

import model as mlib
import dataset as dlib
from config import training_args

torch.backends.cudnn.bencmark = True

from IPython import embed

from tqdm import tqdm


def my_collate_fn(batch):

    imgs, gtys = [], []
    for pid_imgs, gty in batch:
        imgs.extend(pid_imgs)
        gtys.extend([gty] * len(pid_imgs))
    return (torch.stack(imgs, dim=0), torch.Tensor(gtys).long())


class MetricFace(dlib.VerifyFace):
    def __init__(self, args):

        dlib.VerifyFace.__init__(self, args)
        self.args = args
        self.model = dict()
        self.data = dict()
        self.device = args.use_gpu and torch.cuda.is_available()

    def _report_settings(self):
        """ Report the settings """

        str = "-" * 16
        print("%sEnvironment Versions%s" % (str, str))
        print("- Python    : {}".format(sys.version.strip().split("|")[0]))
        print("- PyTorch   : {}".format(torch.__version__))
        print("- TorchVison: {}".format(torchvision.__version__))
        print("- USE_GPU   : {}".format(self.device))
        print("-" * 52)

    def _model_loader(self):
        self.model["backbone"] = mlib.SphereNet20()
        self.model["backbone"].load_state_dict(torch.load("sphere20a_20171020.pth"))
        self.model["uncertain"] = mlib.UncertaintyHead(self.args.in_feats)
        self.model["criterion"] = mlib.MLSLoss(mean=False)

        if self.args.freeze_backbone:
            for p in self.model["backbone"].parameters():
                p.requires_grad = False

        self.model["optimizer"] = torch.optim.SGD(
            [{"params": self.model["uncertain"].parameters()}],
            lr=self.args.base_lr,
            weight_decay=self.args.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        self.model["scheduler"] = torch.optim.lr_scheduler.MultiStepLR(
            self.model["optimizer"],
            milestones=self.args.lr_adjust,
            gamma=self.args.gamma,
        )
        if self.device:
            self.model["backbone"] = self.model["backbone"].cuda()
            self.model["uncertain"] = self.model["uncertain"].cuda()
            self.model["criterion"] = self.model["criterion"].cuda()

        if len(self.args.resume) > 2:
            checkpoint = torch.load(
                self.args.resume, map_location=lambda storage, loc: storage
            )
            self.model["backbone"].load_state_dict(checkpoint["backbone"])
            print(
                "Resuming the train process at %3d epoches ..." % self.args.start_epoch
            )
        print("Model loading was finished ...")

    def _data_loader(self):
        self.trainset = Dataset(
            "/gpfs/gpfs0/r.karimov/pfe/list_casia_mtcnncaffe_aligned_nooverlap.txt"
        )
        proc_func = lambda images: preprocess(images, True)
        self.trainset.start_batch_queue(batch_format, proc_func=proc_func)

    def _model_train(self, epoch=0):

        self.model["backbone"].eval()
        self.model["uncertain"].train()

        loss_recorder, batch_acc = [], []
        for idx in range(3000):
            batch = self.trainset.pop_batch_queue()
            img = torch.from_numpy(batch["image"]).permute(0, 3, 1, 2)
            gty = torch.from_numpy(batch["label"])

            if self.device:
                img = img.cuda()
                gty = gty.cuda()

            feature, sig_feat = self.model["backbone"](img)  # TODO
            log_sig_sq = self.model["uncertain"](sig_feat)
            loss = self.model["criterion"](feature, log_sig_sq, gty)
            self.model["optimizer"].zero_grad()
            loss.backward()
            self.model["optimizer"].step()
            loss_recorder.append(loss.item())
            if (idx + 1) % self.args.print_freq == 0:
                print(
                    "epoch : %2d|%2d, iter : %2d|%2d, loss : %.4f"
                    % (
                        epoch,
                        self.args.end_epoch,
                        idx,
                        3000,
                        np.mean(loss_recorder),
                    )
                )
                filename = os.path.join(self.args.save_to, "current.pth.tar")
                torch.save(
                    {
                        "epoch": epoch,
                        "backbone": self.model["backbone"].state_dict(),
                        "uncertain": self.model["uncertain"].state_dict(),
                    },
                    filename,
                )
        train_loss = np.mean(loss_recorder)
        print("train_loss : %.4f" % train_loss)
        return train_loss

    def _verify_lfw(self):

        self._eval_lfw()

        self._k_folds()

        best_thresh, lfw_acc = self._eval_runner()

        return best_thresh, lfw_acc

    def _main_loop(self):

        if not os.path.exists(self.args.save_to):
            os.mkdir(self.args.save_to)

        max_lfw_acc, min_train_loss = 0.0, 100
        for epoch in range(self.args.start_epoch, self.args.end_epoch + 1):

            start_time = time.time()

            train_loss = self._model_train(epoch)
            self.model["scheduler"].step()

            end_time = time.time()
            print("Single epoch cost time : %.2f mins" % ((end_time - start_time) / 60))

            if min_train_loss > train_loss:

                print("%snew SOTA was found%s" % ("*" * 16, "*" * 16))
                # max_lfw_acc = max(max_lfw_acc, lfw_acc)
                min_train_loss = train_loss
                filename = os.path.join(self.args.save_to, "sota.pth.tar")
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
                savename = os.path.join(self.args.save_to, filename)
                torch.save(
                    {
                        "epoch": epoch,
                        "backbone": self.model["backbone"].state_dict(),
                        "uncertain": self.model["uncertain"].state_dict(),
                        "train_loss": train_loss,
                    },
                    savename,
                )

            if self.args.is_debug:
                break

    def train_runner(self):

        self._report_settings()

        self._model_loader()

        self._data_loader()

        self._main_loop()


if __name__ == "__main__":

    faceu = MetricFace(training_args())
    faceu.train_runner()
