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


class Perceptron2(torch.nn.Module):
    def __init__(self, input_size=1024, hidden_size=1024, **kwargs):
        super(Perceptron2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 2)
        #self.sigmoid = torch.nn.Sigmoid()
        self.log_softmax = torch.nn.LogSoftmax()

    def forward(self, **kwargs):
        x: torch.Tensor = kwargs["feature"]
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        #output = self.sigmoid(output)
        output = self.log_softmax(output)
        return {"pair_classifiers_output": output}  # FIX this


class Perceptron2BN(torch.nn.Module):
    def __init__(self, input_size=1024, hidden_size=512, **kwargs):
        super(Perceptron2BN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(self.hidden_size)

    def forward(self, **kwargs):
        x: torch.Tensor = kwargs["feature"]
        hidden = self.fc1(x)
        relu = self.relu(self.bn1(hidden))
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return {"pair_classifiers_output": output}