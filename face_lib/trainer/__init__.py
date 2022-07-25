import os
import abc
import torch
import torch.distributed as dist
from face_lib.utils import cfg
from torch.utils.tensorboard import SummaryWriter

class TrainerBase(metaclass=abc.ABCMeta):
    _INF = 1e10

    def __init__(self, args, board=None):
        self.args = args
        # Load configurations
        self.model_args = cfg.load_config(self.args.model_config)

        if board is None:
            tensorboard_path = os.path.join(self.model_args.output, "tensorboard")
            os.makedirs(tensorboard_path, exist_ok=True)
            self.board = SummaryWriter(tensorboard_path)
        else:
            self.board = board

        self.backbone = None
        self.head = None
        self.backbone_criterion = None
        self.head_criterion = None

        self.model = dict()
        self.data = dict()

        # create directory for experiment

        if "root" not in self.args:
            self.args.root = self.model_args.output
        self.checkpoints_path = os.path.join(self.args.root, "checkpoints")
        os.makedirs(self.checkpoints_path, exist_ok=self.args.tmp)

        # Set up distributed train
        if self.model_args.is_distributed:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
            dist_url = "tcp://{}:{}".format(
                os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"]
            )
            dist.init_process_group(
                backend="nccl",
                init_method=dist_url,
                rank=self.rank,
                world_size=self.world_size,
            )
            self.local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(self.local_rank)
        self.start_epoch = 0
        self.device = f"cuda:{self.local_rank}" if torch.cuda.is_available else "cpu"

    def train_runner(self):
        self._data_loader()
        self._model_loader()
        self._report_settings()
        self._main_loop()

    @abc.abstractmethod
    def _model_loader(self):
        """
        TODO: docs
        """
        ...

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
