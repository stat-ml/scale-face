import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from face_lib.models import FaceModule


class Perceptron(torch.nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(1024, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        output = self.fc(x)
        output = self.sigmoid(x)
        return output


class Feedforward(torch.nn.Module):
    def __init__(self, input_size=1024, hidden_size=1024, **kwargs):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, **kwargs):
        x: torch.Tensor = kwargs["feature"]
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return {"head_output": output}


class FeedForwardBN(torch.nn.Module):
    def __init__(self, input_size=1024, hidden_size=512, **kwargs):
        super(FeedForwardBN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
        #self.bn1 = nn.BatchNorm1d(input_size)
        self.bn2 = nn.BatchNorm1d(self.hidden_size)

    def forward(self, **kwargs):
        x: torch.Tensor = kwargs["feature"]
        hidden = self.fc1(x)
        relu = self.relu(self.bn2(hidden))
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return {"head_output": output}


class PFEHead(FaceModule):
    def __init__(self, in_feat=512, **kwargs):
        super(PFEHead, self).__init__(**kwargs)
        self.fc1 = nn.Linear(in_feat * 6 * 7, in_feat)
        self.bn1 = nn.BatchNorm1d(in_feat, affine=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_feat, in_feat)
        self.bn2 = nn.BatchNorm1d(in_feat, affine=False)
        self.gamma = Parameter(torch.Tensor([1e-4]))
        self.beta = Parameter(torch.Tensor([-7.0]))

    def forward(self, **kwargs):
        x: torch.Tensor = kwargs["bottleneck_feature"]
        x = x / x.norm(dim=-1, keepdim=True)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        x = self.gamma * x + self.beta
        x = torch.log(1e-6 + torch.exp(x))
        return {"log_sigma": x}


class PFEHeadAdjustable(FaceModule):
    def __init__(self, in_feat=512, out_feat=512, **kwargs):
        super(PFEHeadAdjustable, self).__init__(**kwargs)
        self.fc1 = Parameter(torch.Tensor(out_feat, in_feat))
        self.bn1 = nn.BatchNorm1d(out_feat, affine=True)
        self.relu = nn.ReLU()
        self.fc2 = Parameter(torch.Tensor(out_feat, out_feat))
        self.bn2 = nn.BatchNorm1d(out_feat, affine=False)
        self.gamma = Parameter(torch.Tensor([1.0]))
        self.beta = Parameter(torch.Tensor([0.0]))

        nn.init.kaiming_normal_(self.fc1)
        nn.init.kaiming_normal_(self.fc2)

    def forward(self, **kwargs):
        x: torch.Tensor = kwargs["bottleneck_feature"]
        x = self.relu(self.bn1(F.linear(x, F.normalize(self.fc1))))
        x = self.bn2(F.linear(x, F.normalize(self.fc2)))  # 2*log(sigma)
        x = self.gamma * x + self.beta
        x = torch.log(1e-6 + torch.exp(x))  # log(sigma^2)
        return {"log_sigma": x}


class ProbHead(FaceModule):
    def __init__(self, in_feat=512, **kwargs):
        super(ProbHead, self).__init__(kwargs)
        # TODO: remove hard coding here
        self.fc1 = nn.Linear(in_feat * 7 * 7, in_feat)
        self.bn1 = nn.BatchNorm1d(in_feat, affine=True)
        self.relu = nn.ReLU(in_feat)
        self.fc2 = nn.Linear(in_feat, 1)
        self.bn2 = nn.BatchNorm1d(1, affine=False)
        self.gamma = Parameter(torch.Tensor([1e-4]))
        self.beta = Parameter(torch.Tensor([-7.0]))

    def forward(self, **kwargs):
        x: torch.Tensor = kwargs["bottleneck_feature"]
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        x = self.gamma * x + self.beta
        x = torch.log(1e-6 + torch.exp(x))
        return {"log_sigma": x}
