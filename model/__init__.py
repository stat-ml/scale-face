from .mls_loss import MLSLoss
from .spherenet import SphereNet20
from .resnet import resnet_zoo
from .mobilenet import MobileFace
from .uncertainty_head import UncertaintyHead


model_dict = {
    "spherenet20": SphereNet20,
}

criterions_dict = {"mlsloss": MLSLoss}
