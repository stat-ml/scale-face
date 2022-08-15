from torch import nn


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

