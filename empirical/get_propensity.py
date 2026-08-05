import pandas as pd
from itertools import product
import opt_policy_learn_with_holdout as opt_policy_learn
import basics_with_holdout as basics
import numpy as np
import argparse
import random
import os
from torch.utils.data import DataLoader, Dataset
import torch
import models
import training
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

class sim_dataset_reg(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return (self.X[item], self.y[item])


def create_dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size)


def initialize_setting_real(device, train_dict, args):
    val_dict = {}
    train_val_ratio = int(0.2 * train_dict["X"].shape[0])

    for key in train_dict.keys():
        val_dict[key] = train_dict[key][:train_val_ratio, :]
        train_dict[key] = train_dict[key][train_val_ratio:, :]

    pol_dict = {"X": train_dict["X"], "T": train_dict["T"], "Y": train_dict["Y"],
                "X_val": val_dict["X"], "T_val": val_dict["T"], "Y_val": val_dict["Y"],
                "propensity_met": args.propensity_met, "alpha_lb": None, "alpha_ub": None,
                "weight_decay": args.weight_decay, "amb_radius": args.amb_radius, "norm_met": args.norm_met,
                "robust_met": args.robust_met, "opt_procedure": args.opt_procedure,
                "estimator_choices": args.estimator_choices, "num_treat": args.num_treat, "dist_shift": args.dist_shift}

    input_dim_T, input_dim_X = pol_dict["T"].shape[1], pol_dict["X"].shape[1]

    '''training for the propensity score model'''
    dataset_train = sim_dataset_reg(train_dict["X"], train_dict["T"])
    dataset_val = sim_dataset_reg(val_dict["X"], val_dict["T"])
    # set up the models for training
    lr = models.Prob_model(input_dim_X, device, Nclasses=args.num_treat)
    # dataloader
    dl_train = create_dataloader(dataset_train, batch_size=2048, shuffle=True)
    dl_val = create_dataloader(dataset_val, batch_size=val_dict["X"].shape[0], shuffle=False)

    lr = training.train_Prob_model(args, lr, dl_train, dl_val, device)
    # 保存模型参数
    torch.save(lr.state_dict(), "models\Empirical\prop_model.pt")





'''This script focuses on data-driven procedure.'''
parser = argparse.ArgumentParser(description="Credit Allocation")
parser.add_argument("--type", type=str, default='nonlinear', choices=['linear', 'nonlinear'])

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--density_epochs", type=int, default=30)
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--weight_decay", type=float, default=0)

parser.add_argument("-num_treat", type=int, default=20)
parser.add_argument("-max_iter", type=int, default=100)
parser.add_argument("-amb_radius", type=float, default=0.1)

parser.add_argument("-norm_met", type=str, default="normalized", choices=["normalized", "unnormalized"])
parser.add_argument("-robust_met", type=str, default="robust", choices=["robust", "non-robust"])
parser.add_argument("-propensity_met", type=str, default="unknown", choices=["unknown", "known"])
parser.add_argument("-opt_procedure", type=str, default="data-driven", choices=["data-driven", "non-data-driven"])

parser.add_argument("-alpha_lb", type=float, default=0.1)
parser.add_argument("--dist_shift", type=float, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# parser.add_argument("--test_size", type=int, default=2500, choices=[500, 2000, 2500])

parser.add_argument("-optimization_epochs", type=int, default=1000)
parser.add_argument("-estimator_choices", type=str, default="normalized_estimator",
                    choices=["normalized_estimator", "unnormalized_estimator"])

parser.add_argument("-baseline", type=str, default='None', choices=['None', 'SNLN', 'KLLN'])
parser.add_argument("-verbose", type=bool, default=True, help='Print the training process or not')
parser.add_argument("-plot_boundary", type=bool, default=False, help='Plot the decision boundary in evaluation')
# parser.add_argument('-mode', type=str, default='drift', choices=['shift', 'drift'])

args = parser.parse_args()

if __name__ == '__main__':
    datasets, econ_levels = range(1, 8), range(1, 7)
    all_pairs = list(product(datasets, econ_levels))

    set_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    '''The policy list that can be run.'''
    # save the results
    if not os.path.exists(f"./results/Empirical/{args.type}/{args.robust_met}"):
        os.makedirs(f"./results/Empirical/{args.type}/{args.robust_met}")
    # save the models
    if not os.path.exists(f"./models/Empirical/{args.type}/{args.robust_met}"):
        os.makedirs(f"./models/Empirical/{args.type}/{args.robust_met}")

    if args.baseline != 'None':
        args.opt_procedure = 'non-data-driven'

    data = []
    for (s, e) in all_pairs:
        data.append(pd.read_csv(f'data/Empirical/data_Q{str(s)}_econ{str(e)}.csv'))
    data = pd.concat(data, axis=0)

    drop_columns = ['Unnamed: 0', 'account_id', 'birthday', 'id_card_prov', 'address', 'address_name',
                    'bd_credit_limit', 'bd_credit_limit_next']
    data.drop(drop_columns, axis=1, inplace=True)
    data.dropna(axis=0, how='any', inplace=True)

    # normalization for the covariates
    covariates = [
        'gender', 'age', 'account_age', 'econ_level_from_address', 'econ_level_from_idcard', # demographical related
        'n_visit', 'n_sku', 'n_shop', 'n_brand', 'n_mbp', 'n_pcp', # browse related
        'ord_item_qtty', 'ord_amt_bef_dsct', 'ord_amt_aft_dsct', 'ord_pay_actual', 'ord_amt_freight', 'num_sku', # order related
        'loanamount', 'planfee', 'planfeedsct', 'num_loan', # borrow related
        'amt_repay', 'num_repay', 'amt_refund', 'num_refund', # repay/refund related
        'rate_order_plan', 'rate_bill_plan', 'rate_over', # interest rate related
        'credit_line_group', 'credit_usage', 'sdpamt_pd' # credit related
    ]
    treatment = ['credit_line_group_next']
    outcome = ['profit']

    normalization = pd.DataFrame(columns=['variable', 'min', 'max'])

    for i, c in enumerate(covariates):
        covar_max = np.max(data[c])
        covar_min = np.min(data[c])
        data[c] = data[c] / covar_max
        data[c] = data[c] / covar_max
        normalization.loc[i, :] = [c, covar_min, covar_max]

    # normalization for the outcomes
    min_outcome, max_outcome = np.min(data[outcome]), np.max(data[outcome])
    normalization.loc[i+1, :] = [outcome[0], min_outcome, max_outcome]
    data[outcome] = (data[outcome] - min_outcome) / (max_outcome - min_outcome)
    data[outcome] = (data[outcome] - min_outcome) / (max_outcome - min_outcome)

    normalization.to_csv('models/Empirical/normalization.csv', index=False)
    train_dict = {"X": np.array(data[covariates]),
                  "T": np.array(data[treatment]),
                  "Y": np.array(data[outcome])}

    initialize_setting_real(device, train_dict, args)

