from .base import FaceModule
from .losses import MLSLoss, AngleLoss, ArcFace, CosFace, MLS, ProbLoss
from .spherenet import SphereNet20
from .heads import PFEHead, PFEHeadAdjustable, ProbHead
from .pair_classifiers import Perceptron, Perceptron2, Perceptron2BN, SmartCosine, Bilinear
from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet50_normalized
from .partial_fc import PartialFC
from .style_gan import StyleGanDiscriminator
from torch.nn import BCELoss, CrossEntropyLoss

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
    "bce_loss": BCELoss,
    "cross_entropy_loss": CrossEntropyLoss,
}

heads = {
    "pfe_head": PFEHead,
    "pfe_head_adjustable": PFEHeadAdjustable,
    "prob_head": ProbHead,
    "perceptron2": Perceptron2,
    "smart_cosine": SmartCosine,
}

classifiers = {
    "perceptron2": Perceptron2,
    "smart_cosine": SmartCosine,
    "bilinear": Bilinear
}
