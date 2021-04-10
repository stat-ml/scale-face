import torch.nn


class FaceModule(torch.nn.Module):
    def __init__(self, learnable: bool = True):
        super().__init__()
        self.learnable = learnable
