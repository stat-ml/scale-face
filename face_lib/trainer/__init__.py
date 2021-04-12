import os
import abc
import torch
import torch.distributed as dist
from face_lib.utils import cfg, FACE_METRICS


def _set_evaluation_metric_yaml(yaml_path: str):
    config = cfg.load_config(yaml_path)
    metric_name = config.name
    config.pop("name")
    metric = lambda model: FACE_METRICS[metric_name](model, **config)
    return metric


class TrainerBase(metaclass=abc.ABCMeta):
    _INF = -1e10

    def __init__(self, args, board):
        self.args = args
        # Load configurations
        self.board = board
        self.model_args = cfg.load_config(args.model_config)
        self.optim_args = cfg.load_config(args.optimizer_config)
        self.dataset_args = cfg.load_config(args.dataset_config)
        self.env_args = cfg.load_config(args.env_config)

        self.backbone = None
        self.head = None
        self.backbone_criterion = None
        self.head_criterion = None

        self.model = dict()
        self.data = dict()

        self.device = (
            "cuda" if (self.env_args.use_gpu and torch.cuda.is_available) else "cpu"
        )
        # create directory for experiment

        self.checkpoints_path = self.args.root / "checkpoints"
        os.makedirs(self.checkpoints_path)

        # set evaluation metrics
        self.evaluation_metrics, self.evaluation_configs = [], []
        if len(self.args.evaluation_configs) > 0:
            for item in self.args.evaluation_configs:
                self.evaluation_metrics.append(_set_evaluation_metric_yaml(item))
                self.evaluation_configs.append(cfg.load_config(item))

        # Set up distributed train
        if self.args.is_distributed:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
            dist_url = "tcp://{}:{}".format(
                os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"]
            )
            dist.init_process_group(
                backend=self.args.distr_backend,
                init_method=dist_url,
                rank=self.rank,
                world_size=self.world_size,
            )
            self.local_rank = args.local_rank
            torch.cuda.set_device(self.local_rank)
        self.start_epoch = 0

    def train_runner(self):
        self._model_loader()
        self._report_settings()
        self._data_loader()
        self._main_loop()

    @abc.abstractmethod
    def _data_loader(self):
        """
        TODO: docs
        """
        ...

    @abc.abstractmethod
    def _model_evaluate(self, epoch: int):
        """
        TODO: docs
        """
        ...

    @abc.abstractmethod
    def _model_train(self, epoch: int):
        """
        TODO: docs
        """
        ...

    @abc.abstractmethod
    def _main_loop(self):
        """
        TODO: docs
        """
        ...

    @abc.abstractmethod
    def _report_settings(self):
        """
        TODO: docs
        """
        ...
