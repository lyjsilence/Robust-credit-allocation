import torch
import torch.nn as nn
from torch.distributions.normal import Normal

def act_fun(key, *args):
    act_funs = {"Identity": nn.Identity(), "ELU": nn.ELU(), "Hardshrink": nn.Hardshrink(),
                "Hardsigmoid": nn.Hardsigmoid(), "Hardtanh": nn.Hardtanh(), "Hardswish": nn.Hardswish(),
                "LeakyReLU": nn.LeakyReLU(), "LogSigmoid": nn.LogSigmoid(), "PReLU": nn.PReLU(), "ReLU": nn.ReLU(),
                "ReLU6": nn.ReLU6(), "RReLU": nn.RReLU(), "SELU": nn.SELU(), "CELU": nn.CELU(), "GELU": nn.GELU(),
                "Sigmoid": nn.Sigmoid(), "SiLU": nn.SiLU(), "Mish": nn.Mish(), "Softplus": nn.Softplus(),
                "Softshrink": nn.Softshrink(), "Softsign": nn.Softsign(), "Tanh": nn.Tanh(),
                "Tanhshrink": nn.Tanhshrink(), "GLU": nn.GLU(), "Softmin": nn.Softmin(), "Softmax": nn.Softmax(),
                "Softmax2d": nn.Softmax2d(), "LogSoftmax": nn.LogSoftmax()}

    if key == "MultiheadAttention":
        if len(args) != 2:
            print("Warning：MultiheadAttention requires the inputs: embedding dimension and the number of heads.")
            embed_dim = int(input("Input embed_dim: "))
            num_heads = int(input("Input num_heads: "))
            return nn.MultiheadAttention(embed_dim, num_heads)
        return nn.MultiheadAttention(*args)

    elif key == "Threshold":
        if len(args) != 2:
            print("Warning：Threshold requires the inputs: threshold and value.")
            threshold = float(input("Input threshold: "))
            value = float(input("Input value: "))
            return nn.Threshold(threshold, value)
        return nn.Threshold(*args)

    elif key == "AdaptiveLogSoftmaxWithLoss":
        if len(args) != 3:
            print("Warning：AdaptiveLogSoftmaxWithLoss requires the inputs: "
                  "input features, number of classes, and cutoffs.")
            in_features = int(input("Input in_features: "))
            n_classes = int(input("Input n_classes: "))
            cutoffs = list(map(int, input("Input cutoffs（separated by comma): ").split(',')))
            return nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs)
        return nn.AdaptiveLogSoftmaxWithLoss(*args)
    elif key in act_funs:
        return act_funs[key]

    else:
        print("Warning：No such activations are found.")
        return None


class DR_model(nn.Module):
    def __init__(self, args, input_dim, device):
        super(DR_model, self).__init__()
        # model definition
        self.args = args
        # self.num_classes = num_classes
        self.MLP_depth = args.MLP_depth
        MLP_act_list = [act_fun(key) for key in args.MLP_act_list]
        self.MLP_act_list = nn.ModuleList(MLP_act_list)
        self.dropout_list = nn.ModuleList([nn.Dropout(_) for _ in args.MLP_dropout_list])
        self.MLP_neurons_list = args.MLP_neurons_list

        # model construction
        model_list = []
        for i in range(self.MLP_depth + 1):
            if i == 0:
                model_list.append(nn.Linear(input_dim, args.MLP_neurons_list[i], bias=True))
            elif i == self.MLP_depth:
                model_list.append(nn.Linear(args.MLP_neurons_list[i - 1], 1, bias=True))
            else:
                model_list.append(nn.Linear(args.MLP_neurons_list[i - 1], args.MLP_neurons_list[i], bias=True))
        self.model_list = nn.ModuleList(model_list)
        self.to(device)

    def forward(self, x, t):
        feat = torch.cat((t, x), dim=1)
        for i in range(self.MLP_depth + 1):
            feat = self.model_list[i](feat)
            if i < self.MLP_depth:
                feat = self.MLP_act_list[i](feat)

                feat = self.dropout_list[i](feat)
        return feat


class Prob_model(nn.Module):
    def __init__(self, args, input_dim, device, Nclasses=None):
        super(Prob_model, self).__init__()
        # model definition
        self.args = args
        self.Prob_depth = args.Prob_depth

        Prob_act_list = [nn.Identity()]
        self.Prob_act_list = nn.ModuleList(Prob_act_list)
        self.dropout_list = nn.ModuleList([nn.Dropout(_) for _ in args.Prob_dropout_list])
        self.Prob_neurons_list = args.Prob_neurons_list
        # model construction
        model_list = []
        for i in range(self.Prob_depth + 1):
            if i == 0:
                model_list.append(nn.Linear(input_dim, args.Prob_neurons_list[i], bias=True))
            elif i == self.Prob_depth:
                model_list.append(nn.Linear(args.Prob_neurons_list[i - 1], Nclasses, bias=True))
            else:
                model_list.append(nn.Linear(args.Prob_neurons_list[i - 1], args.Prob_neurons_list[i], bias=True))
        self.model_list = nn.ModuleList(model_list)

        self.to(device)

    def compute_loss(self, z, delta_logp, params, reg_states, reg_coeffs, test=False):
        mean, var = torch.chunk(params, 2, dim=1)
        var = self.softplus(var)

        logpz = torch.sum(self.base_dist(mean, var).log_prob(z), dim=1, keepdim=True)
        logpx = logpz - delta_logp

        if test:
            return logpx
        else:
            nll = -torch.mean(logpx)

            reg_states = tuple(torch.mean(rs) for rs in reg_states)
            if reg_coeffs:
                reg_loss = sum(
                    reg_state * coeff for reg_state, coeff in zip(reg_states, reg_coeffs) if coeff != 0
                )
            return nll, reg_loss

    def base_dist(self, mean, var):
        return Normal(mean, var)

    def forward(self, X):
        feat = X
        for i in range(self.Prob_depth + 1):
            feat = self.model_list[i](feat)
            if i < self.Prob_depth:
                feat = self.dropout_list[i](feat)
        feat = self.Prob_act_list[0](feat)

        return feat


# Normalizing Flow
class LayerNorm(nn.Module):
    def __init__(self, hidden, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.hidden = hidden
        self.eps = eps
        self.alpha = nn.Parameter(torch.randn(hidden))
        self.beta = nn.Parameter(torch.randn(hidden))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x-mean) / std * self.alpha + self.beta


# Numerical MLP
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=[64, 64, 64], activation="ReLU", dropout=0.1, device='cpu'):
        super(MLP, self).__init__()
        self.activation = act_fun(activation)
        self.dim = [dim_in] + dim_hidden + [dim_out]
        self.linears = nn.ModuleList([nn.Linear(self.dim[i-1], self.dim[i]) for i in range(1, len(self.dim))])
        self.layernorms = nn.ModuleList([LayerNorm(hidden) for hidden in dim_hidden])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(dim_hidden))])

    def forward(self, x):
        for i in range(len(self.dim)-2):
            x = self.linears[i](x)
            x = x + self.layernorms[i](x)
            x = self.activation(x)
            x = self.dropouts[i](x)
        x = self.linears[-1](x)
        return x

class simple_MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=[64, 64, 64], activation="ReLU", dropout=0.1, device='cpu'):
        super(simple_MLP, self).__init__()
        self.activation = act_fun(activation)
        self.dim = [dim_in] + dim_hidden + [dim_out]
        self.linears = nn.ModuleList([nn.Linear(self.dim[i-1], self.dim[i]) for i in range(1, len(self.dim))])
        self.layernorms = nn.ModuleList([LayerNorm(hidden) for hidden in dim_hidden])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(dim_hidden))])

    def forward(self, x):
        for i in range(len(self.dim)-2):
            x = self.linears[i](x)
            x = self.activation(x)
            x = self.dropouts[i](x)
        x = self.linears[-1](x)
        return x

