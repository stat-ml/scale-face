from torch import nn


class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x): return x + self.module(x)



class ResNet9(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = None
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
        self.head = nn.Linear(128, num_classes)


    def forward(self, x):
        self.features = self.backbone(x)
        return self.head(self.features)


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

