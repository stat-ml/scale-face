import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from face_lib.models import FaceModule


# MLP - multilayer perceptron. Architecture: len(hidden_layers_size), [linear layers] -> linear(x, 2) -> log_softmax.
# hidden_layers_size - tuple with layers size.
class MLP(torch.nn.Module):
    def __init__(self, in_feat=1024, hidden_layers_size=[512, 126], out_feat=2, **kwargs):
        super(MLP, self).__init__()
        print(kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.hidden = nn.ModuleList()
        self.relu = nn.ModuleList()

        if not hidden_layers_size:
            pass
        else:
            self.hidden.append(nn.Linear(self.in_feat, hidden_layers_size[0]))

        for k in range(len(hidden_layers_size) - 1):
            self.hidden.append(nn.Linear(hidden_layers_size[k], hidden_layers_size[k+1]))
            self.relu.append(nn.ReLU())

        self.relu.append(nn.ReLU())

        if not hidden_layers_size:
            self.hidden.append(nn.Linear(self.in_feat, self.out_feat))
        else:
            self.hidden.append(nn.Linear(hidden_layers_size[-1], self.out_feat))

        self.log_softmax = torch.nn.LogSoftmax()

    def forward(self, **kwargs):
        x: torch.Tensor = kwargs["feature"]
        for r, h in enumerate(self.hidden):
            x = h(x)
            if r < (len(self.hidden) - 1):
                x = self.relu[r](x)
        output = self.log_softmax(x)
        return {"pair_classifiers_output": output}

# Deprecated
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
        return {"pair_classifiers_output": output}

# Deprecated
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