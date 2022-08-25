"""
Level 2 wc
"""
from torch import nn
import torch


def get_model(name, num_classes=4, checkpoint=None):
    """
    Builds a model from a predefined zoo
    """
    if name == 'resnet9':
        model = ResNet9(num_classes)
    elif name == 'resnet9_backbone':
        model = Resnet9Backbone()
    else:
        raise ValueError('Incorrect model name')

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
    return model


def get_confidence_model(name, num_classes, backbone_checkpoint=None):
    if name == 'resnet9_scale':
        backbone = get_model('resnet9', num_classes, checkpoint=backbone_checkpoint)
        ScaleFace(backbone, num_features=128)

    """
    Builds a model from a pretrained zoo with returning the both prediction and uncertainty
    """
    return None


class ScaleFace(nn.Module):
    def __init__(self, backbone, num_features):
        super().__init__()
        self.backbone = backbone
        self.num_features = num_features

        self.linear = nn.Linear(num_features, num_features)
        self.scale_layer = nn.Linear(num_features, 1)

    def forward(self, x):
        with torch.no_grad:
            self.backbone(x)
            x = self.backbone.features
        embeddings = self.linear(x)
        self.scale = self.scale_layer(x)
        return embeddings, self.scale


class PFEHead(nn.Module):
    def __init__(self, in_feat=512, learnable=True):
        super(PFEHead, self).__init__()
        self.learnable = learnable
        self.fc1 = nn.Linear(in_feat * 6 * 7, in_feat)
        self.bn1 = nn.BatchNorm1d(in_feat, affine=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_feat, in_feat)
        self.bn2 = nn.BatchNorm1d(in_feat, affine=False)
        self.gamma = nn.Parameter(torch.Tensor([1e-4]))
        self.beta = nn.Parameter(torch.Tensor([-7.0]))

    def forward(self, x):
        x = x / x.norm(dim=-1, keepdim=True)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        x = self.gamma * x + self.beta
        x = torch.log(1e-6 + torch.exp(x))
        return x



class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x): return x + self.module(x)


class ResNet9(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = None
        self.backbone = Resnet9Backbone()
        self.head = nn.Linear(128, num_classes)


    def forward(self, x):
        self.features = self.backbone(x)
        x = self.head(self.features)
        return x


class Resnet9Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        def conv(in_size, out_size, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_size),
                nn.ELU()
            )

        def residual(channels):
            return Residual(nn.Sequential(
                conv(channels, channels, kernel_size=3, stride=1, padding=1),
                conv(channels, channels, kernel_size=3, stride=1, padding=1),
            ))

        self.backbone = nn.Sequential(
            conv(3, 64, kernel_size=3, stride=1, padding=1),
            conv(64, 128, kernel_size=5, stride=2, padding=2),
            residual(128),
            conv(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            residual(256),
            conv(256, 128, kernel_size=3, stride=1, padding=0),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.backbone(x)



class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        def conv(in_size, out_size, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_size),
                nn.ELU()
            )

        self.layers = nn.Sequential(
            conv(3, 64, kernel_size=5, stride=2, padding=2),
            conv(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv(64, 32, kernel_size=3, stride=1, padding=1),
            conv(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ELU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

