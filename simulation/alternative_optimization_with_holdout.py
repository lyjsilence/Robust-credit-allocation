import torch
import torch.optim as optim
from scipy.optimize import minimize

########################################################################################################################
def evaluation(policy, X, T, Y, prop_score, log_alpha, args, amb_radius=None):

    cal_T = policy(X)

    softmax_fun = torch.nn.Softmax(dim=1)
    cal_T_idx = torch.max(softmax_fun(cal_T), dim=1).indices.reshape(-1, 1)

    indi_or_kernel_eval = torch.where(cal_T_idx == T, 1, 0)

    if args.estimator_choices == "normalized_estimator":
        estimator_norm = indi_or_kernel_eval / prop_score
    else:
        estimator_norm = torch.Tensor([1])

    alpha = torch.nn.functional.softplus(log_alpha)
    samples_eval = indi_or_kernel_eval * torch.exp(-Y / alpha) / prop_score

    if amb_radius == None:
        amb_radius = args.amb_radius

    neg_Q = alpha * torch.log(torch.mean(samples_eval) / torch.mean(estimator_norm)) + alpha * amb_radius

    return -neg_Q, cal_T


def eval_IPW(policy, X, T, Y, prop_score, args):
    cal_T = policy(X)

    softmax_fun = torch.nn.Softmax(dim=1)
    cal_T_idx = torch.max(softmax_fun(cal_T), dim=1).indices.reshape(-1, 1)

    indi_or_kernel_eval = torch.where(cal_T_idx == T, 1, 0)

    if args.estimator_choices == "normalized_estimator":
        estimator_norm = indi_or_kernel_eval / prop_score
    else:
        estimator_norm = torch.Tensor([1])

    Q_IPW = torch.mean(indi_or_kernel_eval * Y / prop_score) / torch.mean(estimator_norm)
    return Q_IPW


def optimization(obj_fun, X, pol_dict, policy, opt_pth, device, k_idx, log_alpha):
    args = pol_dict["args"]

    policy_optimizer = optim.Adam(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_value = -torch.inf
    step, patience = 0, 10

    if args.verbose:
        print('Training for policy network...')
    for epoch in range(args.optimization_epochs):
        # eliminate the gradient
        policy_optimizer.zero_grad()

        cal_T = policy(X)

        obj = obj_fun(cal_T)
        # calculate the gradient and update the parameters
        obj.backward()
        policy_optimizer.step()

        # evaluation the model
        with torch.no_grad():
            X_val, T_val, Y_val, prop_score_val \
                = pol_dict['X_val'], pol_dict['T_val'], pol_dict['Y_val'], pol_dict['prop_score_val']
            X_val, T_val, Y_val, prop_score_val \
                = torch.tensor(X_val).to(device), torch.tensor(T_val).to(device), \
                  torch.tensor(Y_val).to(device), torch.tensor(prop_score_val).to(device),
            if pol_dict["robust_met"] == "robust":
                validation_value, _ = evaluation(
                    policy, X_val, T_val, Y_val, prop_score_val, log_alpha, args
                )
            else:
                validation_value = eval_IPW(
                    policy, X_val, T_val, Y_val, prop_score_val, args
                )

            if validation_value > best_value + 0.001:
                best_value = validation_value
                step = 0
                torch.save(policy.state_dict(), opt_pth)
            else:
                step += 1
                min_epochs = 100 if pol_dict["robust_met"] == "non-robust" else 0
                if step >= patience and epoch + 1 >= min_epochs:
                    if args.verbose:
                        print(f'Early stopping at step {epoch} due to small improvement.')
                    break

        # output the running status
        if args.verbose:
                if (epoch + 1) % 5 == 0:
                    print(f"Iteration: {k_idx}, Epoch [{epoch + 1}/{args.optimization_epochs}], "
                      f"Objective: {obj.item():.4f}, validation value: {validation_value.item():.4f}")

    policy.load_state_dict(torch.load(opt_pth))
    return obj.item(), best_value.item()


def optimization_alpha(obj_fun, pol_dict, alpha_list, device, k_idx, policy, eval=False):
    args = pol_dict["args"]
    log_alpha = alpha_list[-1]
    log_alpha.requires_grad_(True)

    alpha_optimizer = optim.Adam([log_alpha], lr=args.lr, weight_decay=args.weight_decay)
    best_value = -torch.inf
    best_alpha = log_alpha.detach().clone()
    step, patience = 0, 10

    if args.verbose:
        print('Training for alpha...')
    for epoch in range(args.optimization_epochs):
        # eliminate the gradient
        alpha_optimizer.zero_grad()

        obj = obj_fun(log_alpha)
        # calculate the gradient and update the parameters
        obj.backward()
        alpha_optimizer.step()

        # evaluation the model
        with torch.no_grad():
            X_val, T_val, Y_val, prop_score_val \
                = pol_dict['X_val'], pol_dict['T_val'], pol_dict['Y_val'], pol_dict['prop_score_val']
            X_val, T_val, Y_val, prop_score_val \
                = torch.tensor(X_val).to(device), torch.tensor(T_val).to(device), \
                  torch.tensor(Y_val).to(device), torch.tensor(prop_score_val).to(device),
            Q_DRO_val, _ = evaluation(policy, X_val, T_val, Y_val, prop_score_val, log_alpha, args)

            if Q_DRO_val > best_value + 0.001:
                best_value = Q_DRO_val
                best_alpha = log_alpha.detach().clone()
                step = 0
            else:
                step += 1
                if step >= patience:
                    if args.verbose:
                        print(f'Early stopping at step {epoch} due to small improvement.')
                    break

        # output the running status
        if args.verbose:
            if (epoch + 1) % 5 == 0:
                print(f"Iteration: {k_idx}, Epoch [{epoch + 1}/{args.optimization_epochs}], "
                      f"Objective: {obj.item():.4f}, Q_DRO_val: {Q_DRO_val.item():.4f}")

    with torch.no_grad():
        log_alpha.copy_(best_alpha)

    if eval == True:
        return log_alpha, best_value.item()
    else:
        return log_alpha
