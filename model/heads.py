import torch
import torch.nn as nn
from torch.nn import Parameter
from model import FaceModule


class PFEHead(FaceModule):
    def __init__(self, in_feat=512, **kwargs):
        super(PFEHead, self).__init__(kwargs)
        self.fc1 = nn.Linear(in_feat * 6 * 7, in_feat)
        self.bn1 = nn.BatchNorm1d(in_feat, affine=True)
        self.relu = nn.ReLU(in_feat)
        self.fc2 = nn.Linear(in_feat, in_feat)
        self.bn2 = nn.BatchNorm1d(in_feat, affine=False)
        self.gamma = Parameter(torch.Tensor([1.0]))
        self.beta = Parameter(torch.Tensor([0.0]))

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        x = self.gamma * x + self.beta
        x = torch.log(1e-6 + torch.exp(x))
        return {"log_sigma": x}
