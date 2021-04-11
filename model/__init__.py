from .base import FaceModule
from .losses import MLSLoss, AngleLoss
from .spherenet import SphereNet20
from .heads import PFEHead
from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100
from .partial_fc import PartialFC

model_dict = {
    "spherenet20": SphereNet20,
}

criterions_dict = {"mlsloss": MLSLoss, "angle_loss": AngleLoss}

heads = {"pfe_head": PFEHead}
