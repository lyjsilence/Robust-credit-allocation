import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import opt_policy_learn_with_holdout as opt_policy_learn
import basics_with_holdout as basics
import optimization_problem_with_holdout as op
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


def initialize_setting_real(device, train_dict, args):
    pol_dict = {"X": train_dict["X"], "T": train_dict["T"], "Y": train_dict["Y"],
                "propensity_met": args.propensity_met, "alpha_lb": None, "alpha_ub": None,
                "weight_decay": args.weight_decay, "amb_radius": args.amb_radius, "norm_met": args.norm_met,
                "robust_met": args.robust_met, "opt_procedure": args.opt_procedure,
                "estimator_choices": args.estimator_choices, "num_treat": args.num_treat, "dist_shift": args.dist_shift}

    input_dim_T, input_dim_X = pol_dict["T"].shape[1], pol_dict["X"].shape[1]
    input_dim = input_dim_T + input_dim_X

    '''training for the propensity score model'''
    pol_dict["prop_score"] = train_dict["prop_score"]

    pol_dict["args"] = args
    pol_dict["dim_T"], pol_dict["dim_X"], pol_dict["dim_TX"] = input_dim_T, input_dim_X, input_dim

    # If Y takes normalized values, then M should be M = np.max((np.max(train_dict["normalized_Y"]),
    pol_dict["M"] = np.max(train_dict["Y"])

    pol_dict["alpha_ub"] = 1 / args.amb_radius * pol_dict["M"]
    pol_dict["alpha_lb"] = args.alpha_lb
    pol_dict["init_alpha"] = np.random.uniform(pol_dict["alpha_lb"], pol_dict["alpha_ub"])

    return pol_dict


def policy_evaluation_dro(pol_dict, device):
    '''Initialize alpha'''
    args = pol_dict['args']

    opt_k = 1

    if args.baseline == 'None':
        policy = op.NN_Sieves(pol_dict, pol_dict["dim_X"], device, Nlayers=opt_k, Nclasses=pol_dict["num_treat"])
    else:
        policy = op.NN_baseline(pol_dict, pol_dict["dim_X"], device, Nlayers=opt_k, Nclasses=pol_dict["num_treat"])

    evaluated_policy = f'models/Empirical/nonlinear/robust/delta=0.3/opt_pol_dist_0_data-driven-exp_id_0_{opt_k}.pth'

    # load the parameters of policy network and fix it
    policy.load_state_dict(torch.load(evaluated_policy))
    policy.eval()
    op.set_requires_grad(policy, flag=False)

    X, T, Y, prop_score \
        = torch.tensor(pol_dict['X']).to(device), torch.tensor(pol_dict['T']).to(device),  \
          torch.tensor(pol_dict['Y']).to(device), torch.tensor(pol_dict['prop_score']).to(device)

    cal_T = policy(X.to(torch.float32))
    softmax_fun = torch.nn.Softmax(dim=1)
    # cal_T_idx = torch.max(softmax_fun(cal_T), dim=1).indices.reshape(-1, 1)
    probs = softmax_fun(cal_T)  # shape: (n, K)
    cal_T_idx = torch.multinomial(probs, num_samples=1)  # shape: (n, 1)
    return cal_T_idx.detach().cpu().numpy()


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

parser.add_argument("-baseline", type=str, default='None', choices=['None', 'SNLN', 'KLLN'])
parser.add_argument("-verbose", type=bool, default=True, help='Print the training process or not')
parser.add_argument("-plot_boundary", type=bool, default=False, help='Plot the decision boundary in evaluation')
# parser.add_argument('-mode', type=str, default='drift', choices=['shift', 'drift'])

args = parser.parse_args()


def plot_discrete_hist(
    df,
    q_col='Q',
    sim_col='A_sim',
    robust_col='A_robust',
    normalize=False,
    figsize=(12, 6),
    save_path=None,
    age_layout="2row_center"  # 新增： "2row_center" 或 "1x5"
):
    # -------- 基本检查 --------
    data = df[[q_col, sim_col, robust_col]].dropna().copy()
    q_values = sorted(data[q_col].unique())

    support = sorted(set(data[sim_col].unique()).union(set(data[robust_col].unique())))
    x = np.arange(len(support))

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 12,
    })

    q_col_lower = str(q_col).lower()

    # -------- 布局控制 --------
    if q_col_lower == "q":
        fig, ax_grid = plt.subplots(2, 4, figsize=figsize, dpi=180, sharex=True, sharey=True)
        axes = ax_grid.flatten()

    elif q_col_lower in {"econ_level_fron_address", "econ_level_from_address"}:
        fig, ax_grid = plt.subplots(2, 3, figsize=figsize, dpi=180, sharex=True, sharey=True)
        axes = ax_grid.flatten()

    elif q_col_lower == "age_group":
        if age_layout == "1x5":
            fig, ax_grid = plt.subplots(1, 5, figsize=figsize, dpi=180, sharex=True, sharey=True)
            axes = ax_grid.flatten()
        else:
            # 默认：2行，上3列，下2列居中（2×6 GridSpec）
            fig = plt.figure(figsize=figsize, dpi=180)
            gs = fig.add_gridspec(nrows=2, ncols=6, wspace=0.35, hspace=0.55)

            ax1 = fig.add_subplot(gs[0, 0:2])
            ax2 = fig.add_subplot(gs[0, 2:4], sharex=ax1, sharey=ax1)
            ax3 = fig.add_subplot(gs[0, 4:6], sharex=ax1, sharey=ax1)
            ax4 = fig.add_subplot(gs[1, 1:3], sharex=ax1, sharey=ax1)
            ax5 = fig.add_subplot(gs[1, 3:5], sharex=ax1, sharey=ax1)

            # 下排间距 & 上移微调（按你之前调好的来）
            gap = 0.02
            up = 0.04
            p4 = ax4.get_position()
            p5 = ax5.get_position()
            ax4.set_position([p4.x0 - gap/2, p4.y0 + up, p4.width - gap/2, p4.height])
            ax5.set_position([p5.x0 + gap/2, p5.y0 + up, p5.width - gap/2, p5.height])

            axes = [ax1, ax2, ax3, ax4, ax5]

    else:
        fig, ax_grid = plt.subplots(2, 3, figsize=figsize, dpi=180, sharex=True, sharey=True)
        axes = ax_grid.flatten()

    width = 0.38
    ylabel = "Proportion" if normalize else "Frequency"

    # -------- 主循环绘图 --------
    for ax, q in zip(axes, q_values):
        sub = data[data[q_col] == q]

        sim_counts = sub[sim_col].value_counts().reindex(support, fill_value=0).sort_index()
        robust_counts = sub[robust_col].value_counts().reindex(support, fill_value=0).sort_index()

        if normalize:
            sim_sum = sim_counts.values.sum()
            robust_sum = robust_counts.values.sum()
            sim_vals = sim_counts.values / sim_sum if sim_sum > 0 else sim_counts.values
            robust_vals = robust_counts.values / robust_sum if robust_sum > 0 else robust_counts.values
        else:
            sim_vals = sim_counts.values
            robust_vals = robust_counts.values

        ax.bar(x - width/2, sim_vals, width=width, alpha=0.85,
               label='True Credit Limit Adjustment', edgecolor='black', linewidth=0.7)
        ax.bar(x + width/2, robust_vals, width=width, alpha=0.75,
               label='Robust Credit Limit Adjustment', edgecolor='black', linewidth=0.7)

        if q_col_lower == "q":
            title = f'Quarter {q}'
        elif q_col_lower == "age_group":
            title = f'Age Group {q}'
        elif q_col_lower in {"econ_level_fron_address", "econ_level_from_address"}:
            title = f'Region {q}'
        else:
            title = f'{q_col} = {q}'
        ax.set_title(title)

        ax.grid(axis='y', linestyle='--', alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 隐藏多余子图
    if len(q_values) < len(axes):
        for ax in axes[len(q_values):]:
            ax.set_visible(False)

    # 横轴刻度统一
    for ax in axes[:len(q_values)]:
        ax.set_xticks(x)
        ax.set_xticklabels(support, rotation=0)

    # 公共标签
    if q_col_lower == 'q':
        fig.supxlabel('Treatment Levels', fontsize=13, y=0.1)
        fig.supylabel(ylabel, fontsize=13, x=0.05)
    else:
        fig.supxlabel('Treatment Levels', fontsize=13, y=0.15)
        fig.supylabel(ylabel, fontsize=13, x=0.04)

    # 图例
    first_ax = next(ax for ax in axes if ax.get_visible())
    handles, labels = first_ax.get_legend_handles_labels()

    if q_col_lower == 'q':
        fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
        plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.92])
    else:
        fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1))
        plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.93])

    # 注意：这里不要再额外调用一次 tight_layout（你原来最后那行要删掉）

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    age, econ_levels = np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5, 6])
    all_pairs = list(product(age, econ_levels))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    '''The policy list that can be run.'''
    dataset = []
    for s in [1, 2, 3, 4, 5, 6, 7, 8]:
        for (a, e) in all_pairs:
            data = pd.read_csv(f'data/Empirical/data_Q{str(s)}_econ{str(e)}_age{str(a)}_semi.csv')
            data['Q'] = s
            dataset.append(data)
    dataset = pd.concat(dataset, axis=0)
    drop_columns = ['Unnamed: 0', 'account_id', 'birthday', 'id_card_prov', 'address', 'address_name',
                    'credit_usage', ]
    dataset.drop(drop_columns, axis=1, inplace=True)
    dataset.dropna(axis=0, how='any', inplace=True)

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
        covar_max = np.max(dataset[c])
        dataset[c] = dataset[c] / covar_max


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


    params = fit_asymmetric_tanh_transform(dataset[outcome], lam_neg=1.3, tau=2.0)

    dataset[outcome] = asymmetric_tanh_transform(dataset[outcome], params)

    train_dict = {"X": np.array(dataset[covariates]),
                  "T": np.array(dataset[treatment]),
                  "prop_score": np.array(dataset[prop_sim]),
                  "Y": np.array(dataset[outcome])}

    dict_evaluation = initialize_setting_real(device, train_dict, args)

    T_eval = policy_evaluation_dro(
        pol_dict=dict_evaluation,
        device=device,
    )

    dataset = dataset[['Q', 'age_group', 'econ_level_from_address', 'A_sim']]
    dataset['econ_level_from_address'] = (dataset['econ_level_from_address'] * 6).astype(int)
    dataset['A_robust'] = T_eval
    plot_discrete_hist(
        dataset,
        q_col='age_group',
        sim_col='A_sim',
        robust_col='A_robust',
        age_layout="1x5",
        normalize=False,  # 如果你想比较分布形状，可以改成 True
        figsize=(10, 3)
    )


    # dataset.to_csv()