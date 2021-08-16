from .base import FaceModule
from .losses import MLSLoss, AngleLoss, ArcFace, CosFace, MLS, ProbLoss
from .spherenet import SphereNet20
from .heads import PFEHead, PFEHeadAdjustable, ProbHead
from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet50_normalized
from .partial_fc import PartialFC

model_dict = {
    "spherenet20": SphereNet20,
    "partial_fc": PartialFC,
    "iresnet50": iresnet50,
    "iresnet50_normalized": iresnet50_normalized,
}

criterions_dict = {
    "mlsloss": MLSLoss,
    "angle_loss": AngleLoss,
    "arcface": ArcFace,
    "cosface": CosFace,
    "probloss": ProbLoss,
}

heads = {
    "pfe_head": PFEHead,
    "pfe_head_adjustable": PFEHeadAdjustable,
    "prob_head": ProbHead,
}
