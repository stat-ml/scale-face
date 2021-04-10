from .base import FaceModule
from .losses import MLSLoss, AngleLoss
from .spherenet import SphereNet20
from .heads import PFEHead

model_dict = {
    "spherenet20": SphereNet20,
}

criterions_dict = {"mlsloss": MLSLoss, "angle_loss": AngleLoss}

heads = {"pfe_head": PFEHead}
