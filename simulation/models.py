"""Policy models used by the simulation experiments."""

import torch
from torch import nn


class SievePolicy(nn.Module):
    """A fixed-width ReLU network whose depth indexes policy complexity."""

    def __init__(self, input_dim, depth, num_actions, device):
        super().__init__()
        self.depth = depth
        width = 2 * input_dim
        self.activations = nn.ModuleList(
            [nn.ReLU() for _ in range(depth)] + [nn.Identity()]
        )
        layers = []
        for index in range(depth + 1):
            if index == 0:
                layers.append(nn.Linear(input_dim, width, bias=True))
            elif index == depth:
                layers.append(nn.Linear(width, num_actions, bias=True))
            else:
                layers.append(nn.Linear(width, width, bias=True))
        self.layers = nn.ModuleList(layers)
        self.to(device)

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x


class LinearPolicy(nn.Module):
    """Linear policy used by the SNLS benchmark."""

    def __init__(self, input_dim, num_actions, device):
        super().__init__()
        self.layer = nn.Linear(input_dim, num_actions, bias=False)
        self.to(device)

    def forward(self, x):
        return self.layer(x)


class PropensityModel(nn.Module):
    """Propensity network retained to preserve the original RNG sequence."""

    def __init__(self, args, input_dim, num_actions, device):
        super().__init__()
        self.depth = args.Prob_depth
        self.activations = nn.ModuleList([nn.Identity()])
        self.dropouts = nn.ModuleList(
            [nn.Dropout(probability) for probability in args.Prob_dropout_list]
        )
        layers = []
        for index in range(self.depth + 1):
            if index == 0:
                layers.append(nn.Linear(input_dim, args.Prob_neurons_list[index]))
            elif index == self.depth:
                layers.append(nn.Linear(args.Prob_neurons_list[index - 1], num_actions))
            else:
                layers.append(
                    nn.Linear(
                        args.Prob_neurons_list[index - 1],
                        args.Prob_neurons_list[index],
                    )
                )
        self.layers = nn.ModuleList(layers)
        self.to(device)

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < self.depth:
                x = self.dropouts[index](x)
        return self.activations[0](x)


def build_policy(policy_data, depth, baseline, device):
    if baseline == "None":
        return SievePolicy(
            policy_data["dim_X"], depth, policy_data["num_treat"], device
        )
    return LinearPolicy(policy_data["dim_X"], policy_data["num_treat"], device)


def set_trainable(module, trainable):
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)
