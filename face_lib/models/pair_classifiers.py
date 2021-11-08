import torch
import torch.nn as nn


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


class SmartCosine(torch.nn.Module):
    def __init__(self, input_size=1024, bias=True, smart_init=False, **kwargs):
        super(SmartCosine, self).__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(self.input_size // 2, 2, bias=bias)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

        if smart_init:
            self._smart_init()

    def forward(self, **kwargs):
        x: torch.Tensor = kwargs["feature"]
        x1, x2 = x[:, :self.input_size // 2], x[:, self.input_size // 2:]
        output = self.log_softmax(self.fc1(x1 * x2))
        return {"pair_classifiers_output": output}

    def _smart_init(self):
        torch.nn.init.constant_(self.fc1.weight[0, :], -1.)
        torch.nn.init.constant_(self.fc1.weight[1, :], 1.)


class Bilinear(torch.nn.Module):
    def __init__(self, input_size=1024, **kwargs):
        super(Bilinear, self).__init__()
        self.input_size = input_size
        self.fc = torch.nn.Bilinear(self.input_size // 2, self.input_size // 2, 2)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, **kwargs):
        x: torch.Tensor = kwargs["feature"]
        x1, x2 = x[:, :self.input_size // 2], x[:, self.input_size // 2:]
        output = self.log_softmax(self.fc(x1, x2))
        return {"pair_classifiers_output": output}
