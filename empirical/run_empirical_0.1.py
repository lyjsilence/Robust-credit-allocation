import pandas as pd
from itertools import product
import opt_policy_learn_with_holdout as opt_policy_learn
import basics_with_holdout as basics
from train_test_build import sample_test_block, sample_train_connected_max_margin, cells_to_pairs
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


'''This script focuses on data-driven procedure.'''
parser = argparse.ArgumentParser(description="Credit Allocation")
parser.add_argument("--type", type=str, default='nonlinear', choices=['linear', 'nonlinear'])

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--density_epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--weight_decay", type=float, default=0)

parser.add_argument("-num_treat", type=int, default=5)
parser.add_argument("-max_iter", type=int, default=100)
parser.add_argument("-amb_radius", type=float, default=0.1)

parser.add_argument("-norm_met", type=str, default="normalized", choices=["normalized", "unnormalized"])
parser.add_argument("-robust_met", type=str, default="robust", choices=["robust", "non-robust"])
parser.add_argument("-propensity_met", type=str, default="known", choices=["unknown", "known"])
parser.add_argument("-opt_procedure", type=str, default="data-driven", choices=["data-driven", "non-data-driven"])

parser.add_argument("-alpha_lb", type=float, default=0.1)
parser.add_argument("--dist_shift", type=float, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# parser.add_argument("--test_size", type=int, default=2500, choices=[500, 2000, 2500])

parser.add_argument("-optimization_epochs", type=int, default=1000)
parser.add_argument("-estimator_choices", type=str, default="normalized_estimator",
                    choices=["normalized_estimator", "unnormalized_estimator"])

parser.add_argument("-baseline", type=str, default='SNLN', choices=['None', 'SNLN', 'KLLN'])
parser.add_argument("-verbose", type=bool, default=True, help='Print the training process or not')
parser.add_argument("-plot_boundary", type=bool, default=False, help='Plot the decision boundary in evaluation')
# parser.add_argument('-mode', type=str, default='drift', choices=['shift', 'drift'])

args = parser.parse_args()

if __name__ == '__main__':
    age, econ_levels = np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5, 6])
    all_pairs = list(product(econ_levels, age))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    '''The policy list that can be run.'''
    # save the results
    if not os.path.exists(f"./results/Empirical/{args.type}/{args.robust_met}/delta={args.amb_radius}"):
        os.makedirs(f"./results/Empirical/{args.type}/{args.robust_met}/delta={args.amb_radius}")
    # save the models
    if not os.path.exists(f"./models/Empirical/{args.type}/{args.robust_met}/delta={args.amb_radius}"):
        os.makedirs(f"./models/Empirical/{args.type}/{args.robust_met}/delta={args.amb_radius}")

    if args.baseline != 'None':
        args.opt_procedure = 'non-data-driven'

    normalization = pd.read_csv('models/Empirical/normalization.csv')
    for exp_id in range(0, 500):
        set_seed(exp_id)
        print('-' * 20, exp_id, '-' * 20)
        '''
        test_pairs = random.sample(all_pairs, 5)
        test_set = set(test_pairs)
        train_pairs = [p for p in all_pairs if p not in test_set]
        '''

        test_cells = sample_test_block(len(econ_levels), len(age), k=4, seed=exp_id)
        test_pairs = cells_to_pairs(econ_levels, age, test_cells)
        train_cells = sample_train_connected_max_margin(
            len(econ_levels), len(age), test_pairs, k_train=16, seed=exp_id, n_trials_per_D=4000)
        train_pairs = cells_to_pairs(econ_levels, age, train_cells)

        data_test = []
        for (e_test, a_test) in test_pairs:
            for s_test in np.random.choice([1, 3, 5, 7], 1, replace=False):
                data_test.append(pd.read_csv(f'data/Empirical/data_Q{str(s_test)}_econ{str(e_test)}_age{str(a_test)}_semi.csv'))
        data_test = pd.concat(data_test, axis=0)

        data_train = []
        for (e_train, a_train) in train_pairs:
            for s_train in np.random.choice([2, 4, 6, 8], 1, replace=False):
                data_train.append(pd.read_csv(f'data/Empirical/data_Q{str(s_train)}_econ{str(e_train)}_age{str(a_train)}_semi.csv'))
        data_train = pd.concat(data_train, axis=0)

        drop_columns = ['Unnamed: 0', 'account_id', 'birthday', 'id_card_prov', 'address', 'address_name',
                        'credit_usage', ]
        data_train.drop(drop_columns, axis=1, inplace=True)
        data_test.drop(drop_columns, axis=1, inplace=True)
        data_train.dropna(axis=0, how='any', inplace=True)
        data_test.dropna(axis=0, how='any', inplace=True)

        # normalization for the covariates
        covariates = [
            'gender', 'age', 'account_age', 'econ_level_from_address', 'econ_level_from_idcard',  # demographical related
            'n_visit', 'n_sku', 'n_shop', 'n_brand', 'n_mbp', 'n_pcp', # browse related
            'ord_item_qtty', 'ord_amt_bef_dsct', 'ord_amt_aft_dsct', 'ord_pay_actual', 'ord_amt_freight', 'num_sku', # order related
            'loanamount', 'planfee', 'planfeedsct', 'num_loan', # borrow related
            'amt_repay', 'num_repay', 'amt_refund', 'num_refund', # repay/refund related
            'rate_order_plan', 'rate_bill_plan', 'rate_over', # interest rate related
            'sdpamt_pd', 'bd_credit_limit',  # credit related
        ]
        treatment = ['A_sim']
        prop_sim = ['pi0_sim']
        outcome = ['profit_sim']

        for c in covariates:
            # covar_max = normalization[normalization['variable'] == c]['max'].values[0]
            covar_max = np.max(data_train[c])
            data_train[c] = data_train[c] / covar_max
            data_test[c] = data_test[c] / covar_max

        # normalization for the outcomes
        # min_outcome = normalization[normalization['variable'] == outcome[0]]['min'].values[0]
        # max_outcome = normalization[normalization['variable'] == outcome[0]]['max'].values[0]
        # min_outcome, max_outcome = np.min(data_train[outcome]), np.max(data_train[outcome])
        def tanh_reward_transform(Y, m, s, eps=1e-8):
            Y = np.asarray(Y, dtype=float)
            Z = (Y - m) / max(s, eps)
            return 0.5 * (1.0 + np.tanh(Z))

        def fit_asymmetric_tanh_transform(Y_train, lam_neg=1.3, tau=2.0, eps=1e-8):
            Y_train = np.asarray(Y_train, dtype=float)

            m = np.quantile(Y_train, 0.50)
            q10 = np.quantile(Y_train, 0.10)
            q90 = np.quantile(Y_train, 0.90)

            s_neg = max(m - q10, eps)
            s_pos = max(q90 - m, eps)

            return {
                "m": m,
                "s_neg": s_neg,
                "s_pos": s_pos,
                "lam_neg": lam_neg,
                "tau": tau
            }

        def asymmetric_tanh_transform(Y, params):
            Y = np.asarray(Y, dtype=float)

            m = params["m"]
            s_neg = params["s_neg"]
            s_pos = params["s_pos"]
            lam_neg = params["lam_neg"]
            tau = params["tau"]

            z = np.where(
                Y >= m,
                (Y - m) / s_pos,
                lam_neg * (Y - m) / s_neg
            )

            Y_tilde = 0.5 * (1.0 + np.tanh(z / tau))
            return Y_tilde


        params = fit_asymmetric_tanh_transform(data_train[outcome], lam_neg=1.3, tau=2.0)

        data_train[outcome] = asymmetric_tanh_transform(data_train[outcome], params)
        data_test[outcome] = asymmetric_tanh_transform(data_test[outcome], params)

        # training-only calibration
        # m = np.quantile(data_train[outcome], 0.25)  # or np.median(Y_train)
        # s = np.quantile(data_train[outcome], 0.75) - np.quantile(data_train[outcome], 0.25)  # IQR
        # data_train[outcome] = tanh_reward_transform(data_train[outcome], m, s)
        # data_test[outcome] = tanh_reward_transform(data_test[outcome], m, s)

        train_dict = {"X": np.array(data_train[covariates]),
                      "T": np.array(data_train[treatment]),
                      "prop_score": np.array(data_train[prop_sim]),
                      "Y": np.array(data_train[outcome])}

        test_dict = {"X": np.array(data_test[covariates]),
                     "T": np.array(data_test[treatment]),
                     "prop_score": np.array(data_test[prop_sim]),
                     "Y": np.array(data_test[outcome])}

        dict_train_forOPL = basics.initialize_setting_real(device, train_dict, test_dict, args)
        pol_dict_train = opt_policy_learn.policy_learning(pol_dict=dict_train_forOPL, exp_id=exp_id, device=device)

        #policy evaluation
        eval_dict = {}

        for delta in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]:
            dict_evaluation = dict_train_forOPL

            pol_dict_test = opt_policy_learn.policy_evaluation_dro(
                pol_dict=dict_evaluation,
                opt_pol_dict=pol_dict_train,
                evaluator_type=args.robust_met,
                amb_radius=delta,
                device=device,
            )

            q_keys = [k for k in pol_dict_test if isinstance(k, str) and k.startswith('Q')]
            for k in q_keys:
                eval_dict[args.robust_met + '_' + k + '_delta_' + str(delta)] = pol_dict_test[k]
            if args.baseline == 'None':
                joblib.dump(eval_dict,
                            f"results/Empirical/{args.type}/{args.robust_met}/delta={args.amb_radius}/Shift_{args.dist_shift}-exp_id-{exp_id}-DRO.pkl")
            else:
                joblib.dump(eval_dict,
                            f"results/Empirical/{args.type}/{args.robust_met}/delta={args.amb_radius}/Shift_{args.dist_shift}-{args.baseline}-exp_id-{exp_id}-DRO.pkl")
