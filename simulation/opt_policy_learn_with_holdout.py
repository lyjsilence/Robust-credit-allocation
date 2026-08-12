import os
import optimization_problem_with_holdout as op
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import alternative_optimization_with_holdout as aopt


def largek_comput(pol_dict, device, lambda_var, max_depth="computable", Ndepth=5, num_neural="default", Num_width=3):
    if max_depth == "noncomputable":
        return Ndepth

    elif max_depth == "computable" and Ndepth is None:
        k, stop_criterion, dim_X = 1, True, pol_dict["dim_X"]
        n = pol_dict["X_test"].shape[0]
        Nclasses = pol_dict["num_treat"]
        while stop_criterion:
            policy_class = op.NN_Sieves(pol_dict, pol_dict["dim_X"], device, Nlayers=k, Nclasses=Nclasses)
            Vk = policy_class.Vk_computation(k, dim_X, num_neural_met=num_neural, Nwidth=Num_width)
            if Vk < (n ** lambda_var):
                k += 1
            else:
                stop_criterion = False
        return k


def data_driven_procedure(K, W_nk, n_test):
    tau_knh = np.sqrt(K / n_test)
    Q_nk = W_nk - tau_knh

    return Q_nk


def pi_alpha_alter(pol_dict, num_classes, num_layers, exp_id, device):
    '''Prevent wrong inputs of holdout_position'''
    alpha_diff, args, dim_X = np.inf, pol_dict["args"], pol_dict["dim_X"]
    problem = op.OptimizationProblem(pol_dict, device)

    # define the learnable policy class
    if args.baseline == 'None':
        policy = op.NN_Sieves(pol_dict, dim_X, device, Nlayers=num_layers, Nclasses=num_classes)
    else:
        policy = op.NN_baseline(pol_dict, dim_X, device, Nlayers=num_layers, Nclasses=num_classes)

    # define the learnable alpha
    log_alpha = nn.Parameter(torch.tensor(1.0))
    log_alpha_list = [log_alpha]
    log_alpha_list_numpy = [log_alpha.item()]
    alpha_list_numpy = [F.softplus(log_alpha).item()]

    X = torch.from_numpy(pol_dict["X"]).to(torch.float32).to(device)
    robust_met, opt_procedure, dist_shift \
        = pol_dict["robust_met"], pol_dict["opt_procedure"], pol_dict["dist_shift"]
    if args.baseline == 'None':
        opt_pth = os.path.join(f"./models/Simulation/{args.type}/{robust_met}",
                               f"opt_pol_{args.mode}_dist_{dist_shift}_{opt_procedure}-exp_id_{exp_id}_{num_layers}.pth")
    else:
        opt_pth = os.path.join(f"./models/Simulation/{args.type}/{robust_met}",
                               f"opt_pol_{args.mode}_dist_{dist_shift}_{opt_procedure}_{args.baseline}-exp_id_{exp_id}_{num_layers}.pth")

    model = models.Prob_model(pol_dict["args"], pol_dict["dim_X"], device, Nclasses=num_classes)

    max_iter = args.max_iter if robust_met == "robust" else 1

    for k_idx in range(0, max_iter):
        '''Update pi'''
        if k_idx > 0:
            policy.load_state_dict(torch.load(opt_pth))

        # show_evolution = True
        # if show_evolution == True:
        #     opt_pth = os.path.join(f"./models/Simulation_evolution/{args.type}/{robust_met}",
        #                        f"opt_pol_dist_{dist_shift}_{opt_procedure}-exp_id_{exp_id}_{num_layers}-iter_{k_idx}.pth")

        op.set_requires_grad(policy, flag=True)
        obj_fun_policy = lambda cal_T: problem.obj_fun(model, robust_met, log_alpha_list, cal_T, objective="update_pi")
        _, W_nk = aopt.optimization(obj_fun_policy, X, pol_dict, policy, opt_pth, device, k_idx, log_alpha)

        if robust_met == "robust":
            '''Update alpha, fix pi'''
            policy.load_state_dict(torch.load(opt_pth))
            policy.eval()
            op.set_requires_grad(policy, flag=False)

            cal_T = policy(X)
            obj_fun_alpha = lambda alpha: problem.obj_fun(model, robust_met, alpha, cal_T=cal_T, objective="update_alpha")
            new_log_alpha = aopt.optimization_alpha(obj_fun_alpha, pol_dict, log_alpha_list, device, k_idx, policy)

            log_alpha_list_numpy.append(new_log_alpha.item())
            alpha_list_numpy.append(F.softplus(new_log_alpha).item())
            alpha_diff = np.abs(alpha_list_numpy[-1] - alpha_list_numpy[-2])
            if alpha_diff < 0.001:
                break

    return opt_pth, W_nk, alpha_list_numpy


def policy_learning(pol_dict, exp_id, device):
    num_classes = pol_dict["num_treat"]
    if pol_dict["opt_procedure"] == "non-data-driven":
        num_layer_list, new_policy_dict = [2], {}

    elif pol_dict["opt_procedure"] == "data-driven":
        # Find the largest k such that Vk < n^lambda. Here, lambda lies in (0, 1). This largest k helps us determine the
        # stopping criterion when running the data-driven procedure.
        ################################################################################################################
        largek = largek_comput(pol_dict, device, lambda_var=0.9, max_depth="computable",
                               Ndepth=None, num_neural="default", Num_width=5)
        num_layer_list, new_policy_dict = [_ for _ in range(1, largek + 1)], {}
        ################################################################################################################

    opt_pol_dict = {}
    opt_pol_dict['num_layer_list'] = num_layer_list

    for k in num_layer_list:
        print(f'Optimize for k={k}...')
        pol_dict[f"alpha_list_{k}"] = None if pol_dict["robust_met"] == "non-robust" else [pol_dict["init_alpha"]]

        policy_pth, W_nk, pol_dict[f"alpha_list_{k}"] \
            = pi_alpha_alter(pol_dict, num_classes, k, exp_id, device)

        if pol_dict["opt_procedure"] == "data-driven":
            if k == 1:
                Q_nk = data_driven_procedure(k, W_nk, n_test=pol_dict["X_val"].shape[0])
                opt_k = k
            else:
                old_Q_nk = Q_nk
                Q_nk = data_driven_procedure(k, W_nk, n_test=pol_dict["X_val"].shape[0])
                if old_Q_nk < Q_nk:
                    opt_k = k

            opt_pol_dict["opt_k"] = opt_k
            opt_pol_dict[f"policy_{k}"], opt_pol_dict[f"values_{k}"] = policy_pth, W_nk
        else:
            opt_pol_dict["opt_k"] = num_layer_list[-1]
            opt_pol_dict["opt_policy"], opt_pol_dict["opt_values"] = policy_pth, W_nk

    return opt_pol_dict


def policy_evaluation_dro(pol_dict, opt_pol_dict, evaluator_type, amb_radius, device):
    '''Initialize alpha'''
    args = pol_dict['args']

    opt_k = opt_pol_dict["opt_k"]

    if evaluator_type == "robust":
        log_alpha = torch.tensor([pol_dict[f'alpha_list_{opt_k}'][-1]]).to(device)
    elif evaluator_type == 'non-robust':
        log_alpha = torch.tensor([pol_dict['init_alpha']]).to(device)

    if args.baseline == 'None':
        policy = op.NN_Sieves(pol_dict, pol_dict["dim_X"], device, Nlayers=opt_k, Nclasses=pol_dict["num_treat"])
    else:
        policy = op.NN_baseline(pol_dict, pol_dict["dim_X"], device, Nlayers=opt_k, Nclasses=pol_dict["num_treat"])

    if pol_dict["opt_procedure"] == 'data-driven':
        evaluated_policy = opt_pol_dict[f'policy_{opt_k}']
    else:
        evaluated_policy = opt_pol_dict['opt_policy']

    # load the parameters of policy network and fix it
    policy.load_state_dict(torch.load(evaluated_policy))
    policy.eval()
    op.set_requires_grad(policy, flag=False)

    # load the propensity score training model
    model = models.Prob_model(pol_dict["args"], pol_dict["dim_X"], device, Nclasses=args.num_treat)


    if args.baseline in ['None', 'SNLN']:
        X_test, T_test, Y_test, prop_score_test \
            = torch.tensor(pol_dict['X_test']).to(device), torch.tensor(pol_dict['T_test']).to(device),  \
              torch.tensor(pol_dict['Y_test']).to(device), torch.tensor(pol_dict['prop_score_test']).to(device)

        Q_DRO, cal_T = aopt.evaluation(policy, X_test, T_test, Y_test, prop_score_test, log_alpha, args, amb_radius=amb_radius)
        Q_IPW = aopt.eval_IPW(policy, X_test, T_test, Y_test, prop_score_test, args)

    else:
        # optimize the new alpha and calculate Q_DRO
        problem = op.OptimizationProblem(pol_dict, device)
        # calculate the assigned policy
        cal_T = policy(torch.FloatTensor(pol_dict['X_test']).to(device))

        obj_fun_alpha = lambda alpha: problem.obj_fun(model, args.robust_met, alpha, cal_T=cal_T, objective="update_alpha", eval=True)
        new_log_alpha, neg_Q = aopt.optimization_alpha(obj_fun_alpha, pol_dict, log_alpha, k_idx='eval', eval=True)
        Q_DRO = -neg_Q
        Q_IPW = 1

    pol_dict[f'Q_DRO'] = Q_DRO.item()
    pol_dict[f'Q_IPW'] = Q_IPW.item()

    print(f'radius={amb_radius}', Q_DRO)

    # plot the decision boundary
    if args.plot_boundary:
        cal_T_idx = torch.max(cal_T, dim=1).indices.reshape(-1, 1)
        import matplotlib.pyplot as plt
        x1 = pol_dict['X_test'][:, 0]
        x2 = pol_dict['X_test'][:, 1]
        plt.figure(figsize=(6.2, 6.2))
        plt.scatter(x1, x2, c=cal_T_idx.detach().cpu().numpy(), s=6, alpha=0.85)
        plt.title(f'Estimate for {args.type} case using ({args.baseline})')
        plt.show()

    return pol_dict


def policy_evaluation_dro_test(pol_dict, opt_pol_dict, evaluator_type, amb_radius, device):
    '''Initialize alpha'''
    args = pol_dict['args']

    opt_k = opt_pol_dict["opt_k"]
    print('optimal_k=', opt_k)

    for k in opt_pol_dict['num_layer_list']:
        # load alpha
        log_alpha = torch.tensor([pol_dict[f'alpha_list_{k}'][-1]])
        # load policy network
        policy = op.NN_Sieves(pol_dict, pol_dict["dim_X"], device, Nlayers=k, Nclasses=pol_dict["num_treat"])
        evaluated_policy = opt_pol_dict[f'policy_{k}']


        # load the parameters of policy network and fix it
        policy.load_state_dict(torch.load(evaluated_policy))
        policy.eval()
        op.set_requires_grad(policy, flag=False)

        # load the propensity score training model
        model = models.Prob_model(pol_dict["args"], pol_dict["dim_X"], device, Nclasses=args.num_treat)

        X_test, T_test, Y_test, prop_score_test \
            = torch.tensor(pol_dict['X_test']).to(device), torch.tensor(pol_dict['T_test']).to(device),  \
              torch.tensor(pol_dict['Y_test']).to(device), torch.tensor(pol_dict['prop_score_test']).to(device)

        Q_DRO, cal_T = aopt.evaluation(policy, X_test, T_test, Y_test, prop_score_test, log_alpha, args, amb_radius=amb_radius)
        Q_IPW = aopt.eval_IPW(policy, X_test, T_test, Y_test, prop_score_test, args)

        pol_dict[f'Q_DRO'] = Q_DRO
        pol_dict[f'Q_IPW'] = Q_IPW

        print(f'when k={k}, radius={amb_radius}', Q_DRO)

    return pol_dict

def policy_evaluation_min(pol_dict, opt_pol_dict, device):
    if pol_dict["opt_procedure"] == 'data-driven':
        opt_k = opt_pol_dict["opt_k"]
        evaluated_policy = opt_pol_dict[f'policy_{opt_k}']
        policy = op.NN_Sieves(pol_dict, pol_dict["dim_X"], device, Nlayers=opt_k, Nclasses=pol_dict["num_treat"])
    else:
        evaluated_policy = opt_pol_dict['opt_policy']
        policy = None

    policy.load_state_dict(torch.load(evaluated_policy))

    softmax_fun = nn.Softmax(dim=1)
    Q_list = []
    # extract the shifted env and weight
    for i, env in enumerate(pol_dict['shift_env']):
        X_test, Y_all = env['X'], env['Y_all']

        cal_T = policy(torch.FloatTensor(X_test).to(device))
        cal_T_idx = torch.max(softmax_fun(cal_T), dim=1).indices.detach().cpu().numpy()
        Y_select = Y_all[np.arange(Y_all.shape[0]), cal_T_idx]
        Q_list.append(np.mean(Y_select))

    pol_dict[f'Q_min'] = np.min(Q_list)
    return pol_dict

