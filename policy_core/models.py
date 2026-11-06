"""Two-hidden-layer policies and the training propensity network."""
import torch
from torch import nn

POLICY_INITIALIZATIONS = ("default", "kaiming_relu")

def initialize_sieve_layers(layers, depth, initialization):
    """Initialize a plain ReLU sieve without using data or teacher parameters."""
    if initialization == "default":
        return
    if initialization != "kaiming_relu":
        raise ValueError(
            f"Unknown policy initialization {initialization!r}; "
            f"expected one of {POLICY_INITIALIZATIONS}"
        )
    for layer in layers[:depth]:
        nn.init.kaiming_normal_(
            layer.weight,
            mode="fan_in",
            nonlinearity="relu",
        )
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    output_layer = layers[depth]
    nn.init.xavier_normal_(output_layer.weight)
    if output_layer.bias is not None:
        nn.init.zeros_(output_layer.bias)


class CategoricalSievePolicy(nn.Module):
    def __init__(
        self,
        input_dim,
        depth,
        num_actions,
        device,
        width=None,
        initialization="default",
    ):
        super().__init__()
        self.depth = depth
        width = input_dim + 1 if width is None else int(width)
        self.width = width
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
        self.initialization = initialization
        initialize_sieve_layers(self.layers, depth, initialization)
        self.to(device)

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x


class SimulationPropensityModel(nn.Module):
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


def set_trainable(module, trainable):
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)


def build_policy(policy_data, depth, baseline, profile, device):
    return CategoricalSievePolicy(policy_data["dim_X"], depth, policy_data["num_treat"],
        device, width=policy_data["args"].policy_hidden_width,
        initialization=policy_data["args"].policy_initialization)


def build_propensity_model(args, policy_data, profile, device):
    return SimulationPropensityModel(args, policy_data["dim_X"], policy_data["num_treat"], device)
