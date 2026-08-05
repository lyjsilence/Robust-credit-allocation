import opt_policy_learn_with_holdout as opt_policy_learn
import optimization_problem_with_holdout as op
import basics_with_holdout as utils
import numpy as np
import argparse
import random
import os
import joblib
import torch

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_boundary(pol_dict, device):
    '''Initialize alpha'''
    args = pol_dict['args']

    for i in range(100):
        if args.baseline == 'None':
            evaluated_policy = f'./models/Simulation/{args.type}/robust\\opt_pol_dist_0.0_data-driven-exp_id_0_2.pth'
        else:
            evaluated_policy = f'./models/Simulation/{args.type}/robust\\opt_pol_dist_0.0_non-data-driven_SNLN-exp_id_0_2.pth'

        if args.baseline == 'None':
            policy = op.NN_Sieves(pol_dict, pol_dict["dim_X"], device, Nlayers=2, Nclasses=pol_dict["num_treat"])
        else:
            policy = op.NN_baseline(pol_dict, pol_dict["dim_X"], device, Nlayers=2, Nclasses=pol_dict["num_treat"])

        # load the parameters of policy network and fix it
        policy.load_state_dict(torch.load(evaluated_policy))
        policy.eval()
        op.set_requires_grad(policy, flag=False)

        # calculate the assigned policy
        cal_T = policy(torch.FloatTensor(pol_dict['X_test']).to(device))
        cal_T_idx = torch.max(cal_T, dim=1).indices.reshape(-1, 1)

        utils.set_large_fonts()
        import matplotlib.pyplot as plt
        x1 = pol_dict['X_test'][:, 0]
        x2 = pol_dict['X_test'][:, 1]
        plt.figure(figsize=(6.5, 6))
        plt.scatter(x1, x2, c=cal_T_idx.detach().cpu().numpy(), s=6, alpha=0.85)
        plt.xlim([-1.05, 1.05])
        plt.ylim([-1.05, 1.05])
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.tight_layout()
        plt.show()
        #plt.savefig(f'./models/Simulation_evolution/nonlinear/robust\\iter_{k_idx}.png')


def plot_evolution(pol_dict, device):
    args = pol_dict['args']
    opt_k = 1

    if opt_k == 1:
        iter_all = 86
    else:
        iter_all = 88

    for iter in range(iter_all):
        evaluated_policy = f'./models/Simulation_evolution/nonlinear/robust\\opt_pol_dist_0.0_data-driven-exp_id_0_{opt_k}-iter_{iter}.pth'


        '''Initialize alpha'''
        if args.baseline == 'None':
            policy = op.NN_Sieves(pol_dict, pol_dict["dim_X"], device, Nlayers=opt_k, Nclasses=pol_dict["num_treat"])
        else:
            policy = op.NN_baseline(pol_dict, pol_dict["dim_X"], device, Nlayers=opt_k, Nclasses=pol_dict["num_treat"])


        # load the parameters of policy network and fix it
        policy.load_state_dict(torch.load(evaluated_policy))
        policy.eval()
        op.set_requires_grad(policy, flag=False)

        # calculate the assigned policy
        cal_T = policy(torch.FloatTensor(pol_dict['X_test']).to(device))
        cal_T_idx = torch.max(cal_T, dim=1).indices.reshape(-1, 1)

        utils.set_large_fonts()
        import matplotlib.pyplot as plt
        x1 = pol_dict['X_test'][:, 0]
        x2 = pol_dict['X_test'][:, 1]
        plt.figure(figsize=(6.5, 6))
        plt.scatter(x1, x2, c=cal_T_idx.detach().cpu().numpy(), s=6, alpha=0.85)
        plt.xlim([-1.05, 1.05])
        plt.ylim([-1.05, 1.05])
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'./models/Simulation_evolution/nonlinear/robust\\iter_{iter}.png')
        plt.close()


'''This script focuses on data-driven procedure.'''
parser = argparse.ArgumentParser(description="Credit Allocation")
parser.add_argument("--type", type=str, default='nonlinear', choices=['linear', 'nonlinear'])

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--density_epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--weight_decay", type=float, default=0)

parser.add_argument("-num_treat", type=int, default=2)
parser.add_argument("-max_iter", type=int, default=100)
parser.add_argument("-amb_radius", type=float, default=0.2)

parser.add_argument("-norm_met", type=str, default="normalized", choices=["normalized", "unnormalized"])
parser.add_argument("-robust_met", type=str, default="robust", choices=["robust", "non-robust"])
parser.add_argument("-propensity_met", type=str, default="known", choices=["unknown", "known"])
parser.add_argument("-opt_procedure", type=str, default="non-data-driven", choices=["data-driven", "non-data-driven"])

parser.add_argument("-alpha_lb", type=float, default=0.1)
parser.add_argument("--dist_shift", type=float, default=0.0, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
parser.add_argument("--test_size", type=int, default=2500, choices=[500, 2000, 2500])

parser.add_argument("-optimization_epochs", type=int, default=1000)
parser.add_argument("-estimator_choices", type=str, default="normalized_estimator",
                    choices=["normalized_estimator", "unnormalized_estimator"])

parser.add_argument("-baseline", type=str, default='None', choices=['None', 'SNLN', 'KLLN'])
args = parser.parse_args()


if __name__ == '__main__':
    num_exp = 100
    set_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    '''The policy list that can be run.'''

    if not os.path.exists(f"./results/Simulation/{args.type}/{args.robust_met}"):
        os.makedirs(f"./results/Simulation/{args.type}/{args.robust_met}")

    if not os.path.exists(f"./models/Simulation/{args.type}/{args.robust_met}"):
        os.makedirs(f"./models/Simulation/{args.type}/{args.robust_met}")

    if args.baseline != 'None':
        args.opt_procedure = 'non-data-driven'

    num_exp = 1
    for i in range(100):
        print(i)
        load_data = joblib.load(f"./data/Simulation/{args.type}/Shift_{args.dist_shift}-exp-{str(i)}.pkl")

        '''Policy Learning'''
        train_dict, test_dict = load_data["train"], load_data["test"]

        dict_train_forOPL = utils.initialize_setting(device, train_dict, test_dict, args)

        '''policy evaluation'''
        eval_dict = {}
        dict_evaluation = dict_train_forOPL

        '''
        plot_evolution(
            pol_dict=dict_evaluation,
            device=device,
        )
        '''

        plot_boundary(
            pol_dict=dict_evaluation,
            device=device,
        )
