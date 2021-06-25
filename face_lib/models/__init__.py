from .base import FaceModule
from .losses import MLSLoss, AngleLoss, ArcFace, CosFace, MLS
from .spherenet import SphereNet20
from .heads import PFEHead, PFEHeadAdjustable
from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100
from .partial_fc import PartialFC

model_dict = {
    "spherenet20": SphereNet20,
    "partial_fc": PartialFC,
    "iresnet50": iresnet50,
}

criterions_dict = {
    "mlsloss": MLSLoss,
    "angle_loss": AngleLoss,
    "arcface": ArcFace,
    "cosface": CosFace,
}

heads = {
    "pfe_head": PFEHead,
    "pfe_head_adjustable": PFEHeadAdjustable,
}
