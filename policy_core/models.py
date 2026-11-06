"""Two-hidden-layer ReLU policy with two available actions."""

from torch import nn


class Policy(nn.Module):
    def __init__(self, width, input_dim=4):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim, width),
                nn.Linear(width, width),
                nn.Linear(width, 2),
            ]
        )

    def forward(self, x):
        x = self.layers[0](x).relu()
        x = self.layers[1](x).relu()
        return self.layers[2](x)
