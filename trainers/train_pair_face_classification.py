import os
import sys
import torch
import torch.distributed as dist
import numpy as np
import torchvision

sys.path.append(".")
from face_lib.datasets import MS1MDatasetPFE, MS1MDatasetRandomPairs, DataLoaderX
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

        if self.model_args.pair_classifier:
            self.pair_classifier = mlib.pair_classifiers[self.model_args.pair_classifier.name](
                **utils.pop_element(self.model_args.pair_classifier, "name"),
            )
            self.pair_classifier_criterion = mlib.criterions_dict[
                self.model_args.pair_classifier.criterion.name
            ](
#                **utils.pop_element(self.model_args.pair_classifier.criterion, "name"),
            )

        self.start_epoch = 0
        if self.args.resume:
            model_dict = torch.load(self.args.resume, map_location=self.device)
            self.start_epoch = model_dict["epoch"]
            self.backbone.load_state_dict(model_dict["backbone"])
            self.pair_classifier.load_state_dict(model_dict["pair_classifier"])

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

        if self.model_args.pair_classifier and self.model_args.pair_classifier.learnable is False:
            for p in self.head.parameters():
                p.requires_grad = False
            self.pair_classifier.eval()

        learnable_parameters = []
        if self.model_args.backbone.learnable is True:
            learnable_parameters += list(self.backbone.parameters())
        if self.model_args.pair_classifier and self.model_args.pair_classifier.learnable is True:
            learnable_parameters += list(self.pair_classifier.parameters())


        self.optimizer = utils.optimizers_map[self.model_args.optimizer.name](
            [{"params": learnable_parameters}],
            **utils.pop_element(self.model_args.optimizer, "name"),
        )

        self.scheduler = utils.scheduler_map[self.model_args.scheduler.name](
            self.optimizer, **utils.pop_element(self.model_args.scheduler, "name")
        )

        if self.device:
            self.backbone = self.backbone.to(self.device)
            if self.pair_classifier:
                self.pair_classifier = self.pair_classifier.to(self.device)

        if self.model_args.is_distributed:
            for p in self.pair_classifier.parameters():
                dist.broadcast(p, 0)
            self.pair_classifier = self.pair_classifier = torch.nn.parallel.DistributedDataParallel(
                module=self.pair_classifier, broadcast_buffers=False, device_ids=[self.local_rank]
            )
            self.pair_classifier.train()

        # set evaluation metrics
        self.evaluation_metrics, self.evaluation_configs = [], []
        if len(self.model_args.evaluation_configs) > 0:
            for item in self.model_args.evaluation_configs:
                self.evaluation_metrics.append(_set_evaluation_metric_yaml(item))
                self.evaluation_configs.append(item)

        print("Model loading was finished")

    def _data_loader(self):
        if self.model_args.dataset.name == "ms1m":
            self.trainset = MS1MDatasetRandomPairs(
                root_dir=self.model_args.dataset.path,
                in_size=self.model_args.in_size,
            )

            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset, shuffle=True
            )

            self.trainloader = DataLoaderX(
                local_rank=self.rank,
                dataset=self.trainset,
                batch_size=self.model_args.dataset.batch_size,
                sampler=train_sampler,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
            )
        else:
            raise NotImplementedError("Dataset is not implemented")

        print("Data loading was finished ...")

    @torch.no_grad()
    def _model_evaluate(self, epoch=0):
        self.backbone.eval()
        if self.model_args.pair_classifier:
            self.pair_classifier.eval()
        for metric in self.evaluation_configs:
            if metric.name == "lfw_6000_pairs":
                # Calculating accuracy does not seem reasonable in terms of PFE
                ac_res = utils.accuracy_lfw_6000_pairs_binary_classification(
                    self.backbone,
                    self.pair_classifier,
                    metric.lfw_path,
                    metric.lfw_pairs_txt_path,
                    N=metric.N,
                    n_folds=metric.n_folds,
                    device=self.device,
                    board=True,
                    board_writer=self.board,
                    board_iter=epoch,
                )
                print(ac_res)

    def _model_train(self, epoch=0):
        if self.model_args.backbone.learnable:
            self.backbone.train()
        else:
            self.backbone.eval()
        if self.model_args.pair_classifier.learnable:
            self.pair_classifier.train()
        else:
            self.pair_classifier.eval()
        self.pair_classifier_criterion.train()

        loss_recorder, batch_acc = [], []
        for idx, (first_img, second_img, label) in enumerate(self.trainloader):

            _global_iteration = epoch * self.model_args.iterations + idx

            first_img.requires_grad = False
            second_img.requires_grad = False
            label.requires_grad = False
            if self.device:
                first_img = first_img.to(self.device)
                second_img = second_img.to(self.device)
                label = label.to(self.device)

            first_outputs = self.backbone(first_img)
            second_outputs = self.backbone(second_img)

            # print("first_outputs", first_outputs["feature"].shape)
            # print("second_outputs", second_outputs["feature"].shape)

            feature_stacked = torch.cat((first_outputs["feature"], second_outputs["feature"]), dim=1)
            #print("stacked_shape_tarin", feature_stacked.shape)

            outputs = {"feature": feature_stacked}

            # print("feature_stacked", outputs["feature"])
            # print("feature_stacked", outputs["feature"].shape)

            outputs.update(self.pair_classifier(**outputs))
            outputs.update({"label": label})

            loss = self.pair_classifier_criterion(outputs["pair_classifiers_output"], label)

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
                self.board.add_scalar(
                    f"train/{self.pair_classifier_criterion if self.pair_classifier else self.backbone_criterion}_loss_mean",
                    np.mean(loss_recorder),
                    _global_iteration,
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
                        "pair_classifier": self.pair_classifier.module.state_dict()
                        if self.model_args.is_distributed
                        else self.pair_classifier.state_dict(),
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
                        "pair_classifier": self.pair_classifier.module.state_dict()
                        if self.model_args.is_distributed
                        else self.pair_classifier.state_dict(),
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
        print("- Pair_classifier   : {}".format(self.pair_classifier))
        print("- Backbone Criterion   : {}".format(self.backbone_criterion))
        print("- Pair_classifier Criterion   : {}".format(self.pair_classifier_criterion))
        print("-" * 52)


if __name__ == "__main__":
    args = training_args()
    writer = SummaryWriter(args.root)
    faceu = Trainer(args, writer)
    faceu.train_runner()
