import numpy as np
import models
import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizationProblem:
    def __init__(self, pol_dict, device):
        self.iteration = 0
        self.tr_propensity_score_path = pol_dict["tr_propensity_score_path"]
        self.propensity_met = pol_dict['propensity_met']
        self.device = device
        self.args = pol_dict["args"]
        self.num_epochs = pol_dict["args"].optimization_epochs
        self.lr = pol_dict["args"].lr
        self.weight_decay = pol_dict["args"].weight_decay
        self.propensity_met = pol_dict["propensity_met"]
        self.estimator_choices = pol_dict["estimator_choices"]
        self.opt_procedure = pol_dict["opt_procedure"]
        self.amb_radius = pol_dict["amb_radius"]

        self.X = torch.from_numpy(pol_dict["X"]).to(torch.float32).to(device)
        self.X_test = torch.from_numpy(pol_dict["X_test"]).to(torch.float32).to(device)
        self.Y = torch.from_numpy(pol_dict["Y"]).to(torch.float32).to(device)
        self.Y_test = torch.from_numpy(pol_dict["Y_test"]).to(torch.float32).to(device)
        self.T = torch.from_numpy(pol_dict["T"]).to(torch.int).to(device)
        self.T_test = torch.from_numpy(pol_dict["T_test"]).to(torch.int).to(device)
        self.prop_score = torch.from_numpy(pol_dict["prop_score"]).to(torch.float32).to(device)
        self.prop_score_test = torch.from_numpy(pol_dict["prop_score_test"]).to(torch.float32).to(device)

    def model_eval(self, model, cal_T, softmax_fun, eval):
        if eval == False:
            X, T, est_prop_score = self.X, self.T, self.prop_score
        else:
            X, T, est_prop_score = self.X_test, self.T_test, self.prop_score_test

        if self.propensity_met == 'unknown':
            model.load_state_dict(torch.load(self.tr_propensity_score_path))
            est_prop_score_mat = softmax_fun(model(X))
            est_prop_score = torch.clamp(est_prop_score_mat[torch.arange(est_prop_score_mat.size(0)), T.squeeze()], min=0.01)
            est_prop_score = est_prop_score.reshape(-1, 1)

        indi_or_kernel_eval = torch.exp(-F.cross_entropy(cal_T/0.05, T.squeeze().to(torch.int64), reduction="none").reshape(-1, 1))

        if self.estimator_choices == "normalized_estimator":
            estimator_norm = indi_or_kernel_eval / est_prop_score

        elif self.estimator_choices == "unnormalized_estimator":
            estimator_norm = torch.Tensor([1]).to(self.device)

        return indi_or_kernel_eval, estimator_norm, est_prop_score

    def obj_fun(self, model, robust_met, alpha_or_alpha_list, cal_T, objective="update_pi", eval=False):
        softmax_fun = nn.Softmax(dim=1)
        if eval == False:
            Y = self.Y
        else:
            Y = self.Y_test

        ind_or_kernel_eval, estimator_norm, est_prop_score = \
            self.model_eval(model, cal_T, softmax_fun, eval)

        if objective == "update_pi":
            if robust_met == "non-robust":
                samples_eval = ind_or_kernel_eval * Y / est_prop_score
                obj = -torch.mean(samples_eval) / torch.mean(estimator_norm)

            elif robust_met == "robust":
                log_alpha = alpha_or_alpha_list[-1]
                log_alpha.requires_grad_(False)
                alpha = F.softplus(log_alpha)

                samples_eval = (ind_or_kernel_eval * torch.exp(-Y / alpha)) / est_prop_score
                obj = torch.mean(samples_eval) / torch.mean(estimator_norm)
            return obj

        elif objective == "update_alpha":
            with torch.enable_grad():
                alpha = torch.nn.functional.softplus(alpha_or_alpha_list)
                samples_eval = ind_or_kernel_eval * torch.exp(-Y / alpha) / est_prop_score
                obj = alpha * torch.log(torch.mean(samples_eval) /
                                        torch.mean(estimator_norm)) + alpha * self.amb_radius
            return obj


class NN_Sieves(nn.Module):
    def __init__(self, pol_dict, input_dim, device, Nlayers, Nclasses=None):
        super(NN_Sieves, self).__init__()
        # model definition
        self.num_layers = Nlayers
        nn_sieve_act_list = [models.act_fun("ReLU") for _ in range(Nlayers)]
        nn_sieve_act_list.append(nn.Identity())
        # nn_sieve_act_list.append(nn.Softmax(dim=1))
        self.nn_sieve_act_list = nn.ModuleList(nn_sieve_act_list)
        # todo: here also use this formula, where the neurons is too large, should it be modified?
        self.nn_sieve_neurons = 2 * input_dim

        self.pol_dict = pol_dict

        # model construction
        model_list = []
        for i in range(self.num_layers + 1):
            if i == 0:
                model_list.append(nn.Linear(input_dim, self.nn_sieve_neurons, bias=True))
            elif i == self.num_layers:
                model_list.append(nn.Linear(self.nn_sieve_neurons, Nclasses, bias=True))
            else:
                model_list.append(nn.Linear(self.nn_sieve_neurons, self.nn_sieve_neurons, bias=True))
        self.model_list = nn.ModuleList(model_list)

        self.to(device)

    def Vk_computation(self, k, dim_X, num_neural_met="default", Nwidth=3):
        '''
        Computation of Vk (see Theorem 7 in the paper: Nearly-tight VC-dimension and Pseudodimension Bounds for
         Piecewise Linear Neural Networks Peter L. Bartlett, Nick Harvey, Christopher Liaw, Abbas Mehrabian;
          20(63):1−17, 2019.)
        '''
        if num_neural_met == "default":
            neural_width = Nwidth
        elif num_neural_met == "non-default":
            neural_width = np.power(3, dim_X + 3) * np.max((dim_X, 3))
        # According to the paper, p+1 represents the number of piecewise polynomial. Since the whole policy classes
        # use neural networks whose activation function is relu and relu itself has two piecewise polynomial, so we
        # need to set p+1 = 2, implying that p=1.
        p = 1
        # According to the paper, d represents the value such that the degree of the piecewise polynomial is no more
        # than d. Since the whole policy classes use neural networks whose activation function is relu and relu
        # itself is formed based on two functions, one is constant and another function is linear function, so the
        # degree of the piecewise polynomial is no more than 1, implying that we set d=1.
        d = 1
        L, W_dict, Wi_dict = k + 1, {}, {}
        ki_dict = {"number of units at the" + str(i) + "th layer": neural_width if i < L else 1
                   for i in range(1, L + 1)}
        R = 0
        for i in range(1, L + 1):
            if i == 1:
                W_dict["W_" + str(i)] = neural_width * dim_X + neural_width
                Wi_dict["W_" + str(i)] = W_dict["W_" + str(i)]
            else:
                if i == L:
                    W_dict["W_" + str(i)] = 1 * neural_width + 1
                else:
                    W_dict["W_" + str(i)] = neural_width * neural_width + neural_width
                if d == 0:
                    Wi_dict["W_" + str(i)] = W_dict["W_" + str(i)]
                else:
                    Wi_dict["W_" + str(i)] = Wi_dict["W_" + str(i - 1)] + W_dict["W_" + str(i)]
                # W_dict["W_" + str(i)] = 1 * neural_width + 1 if i == L else neural_width * neural_width + neural_width
                # Wi_dict["W_" + str(i)] = W_dict["W_" + str(i)] if d == 0 else Wi_dict["W_" + str(i - 1)] + W_dict["W_" + str(i)]
            R = R + ki_dict["number of units at the" + str(i) + "th layer"] * (1 + (i - 1) * (d ** (i - 1)))
        # Calculate W
        W = sum(W_dict.values())
        # Calculate L_bar
        L_bar = 1 / W * sum(Wi_dict.values())
        # # Calculate R
        # R = sum(ki_dict["number of units at the" + str(i) + "th layer"] * (1 + (i - 1) * (d ** (i - 1)))
        #         for i in range(1, L + 1))
        Vk = L + L_bar * W * np.log2(4 * np.exp(1) * p * R * np.log2(2 * np.exp(1) * p * R))

        return Vk

    def forward(self, x):
        for i in range(self.num_layers + 1):
            x = self.model_list[i](x)
            x = self.nn_sieve_act_list[i](x)

        return x


class NN_baseline(nn.Module):
    def __init__(self, pol_dict, input_dim, device, Nlayers, Nclasses=None):
        super(NN_baseline, self).__init__()
        # model construction
        self.model = nn.Linear(input_dim, Nclasses, bias=False)
        self.model.to(device)

    def forward(self, x):
        x = self.model(x)
        return x


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)