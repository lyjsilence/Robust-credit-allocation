import pandas as pd
import random
import os
import models
import six
import training
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from scipy.stats import norm


def save_data(df, file_path):
    df.to_excel(file_path, index=False, engine='openpyxl')


def cleaning(file, args):
    '''Data cleaning for different purpose are different. Here, we undergo several process for data cleaning'''
    df = pd.read_excel(pd.ExcelFile(file), keep_default_na=False)

    '''Cleaning: Filter out those samples which do not have stable dose of warfarin and stable observed INR'''
    filter_idx = df[((df["Subject Reached Stable Dose of Warfarin"] == "NA") |
                    (df["Subject Reached Stable Dose of Warfarin"] == 0)) |
                    (df["Therapeutic Dose of Warfarin"] == "NA") |
                    (df["INR on Reported Therapeutic Dose of Warfarin"] == "NA")].index
    df.drop(filter_idx, inplace=True)

    '''Cleaning: Classify the age group.'''
    # Convert the age group to categorical variable
    age_encode, group = pd.factorize(df["Age"])
    # map = pd.DataFrame({"Age": group, "Age encode": range(len(group))})
    df["Age_encode"] = age_encode
    # Insert the column Age_encode after the column Age
    df.insert(loc=df.columns.values.tolist().index("Age") + 1, column="Age_encode", value=df.pop("Age_encode"))
    # Delete those rows' samples where the age group is not classified and the column "Age"
    df.drop(df[df.Age == "NA"].index, inplace=True)
    df.drop("Age", axis=1, inplace=True)

    '''Cleaning: Remove those samples without height and weight values.'''
    columns_to_strip = ["Height (cm)", "Weight (kg)"]
    df = df[df[columns_to_strip].ne('').all(axis=1)]
    # Remove those samples without height values.
    df.drop(df[df["Height (cm)"] == "NA"].index, axis=0, inplace=True)
    # Remove those samples without weight values.
    df.drop(df[df["Weight (kg)"] == "NA"].index, axis=0, inplace=True)
    # # Remove those samples without height values.
    # df.drop(df[(df["Height (cm)"] == "") | (df["Height (cm)"] == "NA")].index, axis=0, inplace=True)
    # # Remove those samples without weight values.
    # df.drop(df[(df["Weight (kg)"] == "") | (df["Weight (kg)"] == "NA")].index, axis=0, inplace=True)

    '''Cleaning: Add a column of calculated BMI.'''
    df["BMI"] = df["Weight (kg)"] / ((df["Height (cm)"] / 100) ** 2)
    df.insert(loc=df.columns.values.tolist().index("Weight (kg)") + 1, column="BMI", value=df.pop("BMI"))

    '''Cleaning: Set those samples without values in the columns "Target INR" to 0.'''
    df.loc[df["Target INR"] == "NA", "Target INR"] = 0

    '''Cleaning: One-hot Encoding of variables.'''
    for input_var, num_var, output_var in zip(args.one_hoc_input, args.num_one_hoc, args.one_hoc_output):
        for i in range(num_var):
            df[output_var + " == " + str(i)] = 0
            df.loc[(df[input_var] == i), output_var + " == " + str(i)] = 1
            df.insert(loc=df.columns.values.tolist().index(input_var) + i + 1,
                      column=output_var + " == " + str(i), value=df.pop(output_var + " == " + str(i)))
        df.drop(input_var, axis=1, inplace=True)

    '''Cleaning: Delete those variables that are not used.'''
    for var_delete in args.var_delete:
        df.drop(var_delete, axis=1, inplace=True)

    return df


def true_prop_score(z, t, theta, lr, treat_met):
    if treat_met == "continuous":
        mu_t = np.mean(t)
        sigma_t = np.std(t)

        return np.array([norm.pdf(t[i], loc=mu_t+sigma_t*np.sqrt(theta)*z[i],
                                  scale=sigma_t*np.sqrt(1-theta)) for i in range(t.shape[0])])


'''torch version of kernel_cal'''
def kernel_cal(kernel_met, device):
    if kernel_met == "gaussian":
        return lambda u: torch.exp(-0.5 * u ** 2) / (torch.sqrt(2 * torch.tensor(np.pi).to(device)))

    elif kernel_met == "epanechnikov":
        return lambda u: torch.abs(0.75 * (1 - u ** 2) * (torch.where(torch.abs(u) <= 1, 1, 0)))
# def kernel_cal(kernel_met):
#     if kernel_met == "gaussian":
#         return lambda u: np.exp(-0.5 * u ** 2) / (np.sqrt(2 * np.pi))
#
#     elif kernel_met == "epanechnikov":
#         return lambda u: np.abs(0.75 * (1 - u ** 2) * (np.where(np.abs(u) <= 1, 1, 0)))


def data_split(sim_data_dict, train_size, test_size):
    '''Total number of samples'''
    total_sample = train_size + test_size
    all_idx = {i for i in range(total_sample)}
    random.seed(0)
    train_idx = sorted(random.sample(all_idx, train_size))
    test_idx = sorted(all_idx.difference(train_idx))

    pd_train, pd_test = pd.DataFrame(), pd.DataFrame()
    data_train_dict, data_test_dict = {}, {}
    for key, value in sim_data_dict.items():
        if value.ndim == 1:
            train_array, test_array = value[train_idx], value[test_idx]
            '''It is odd that there is a problem about the format of particular entries. Here, we simply delete them.'''
            train_array, test_array = np.delete(train_array, (-1)), np.delete(test_array, (-1))
            data_train_dict[key], data_test_dict[key] = train_array, test_array
            pd_train[key], pd_test[key] = train_array, test_array

        else:
            train_array, test_array = value[train_idx, :], value[test_idx, :]
            '''It is odd that there is a problem about the format of particular entries. Here, we simply delete them.'''
            train_array, test_array = np.delete(train_array, (-1), axis=0), np.delete(test_array, (-1), axis=0)
            data_train_dict[key], data_test_dict[key] = train_array, test_array

            pd_train = pd.concat([pd_train, pd.DataFrame(train_array,
                                                         columns=[f"{key}_{i + 1}"
                                                                  for i in range(value.shape[1])])], axis=1)
            pd_test = pd.concat([pd_test, pd.DataFrame(test_array,
                                                       columns=[f"{key}_{i + 1}"
                                                                for i in range(value.shape[1])])], axis=1)
    with pd.ExcelWriter(os.path.join("./data", "check_data.xlsx")) as writer:
        # use to_excel function and specify the sheet_name and index to store the dataframe in specified sheet
        pd_train.to_excel(writer, sheet_name="train", index=False)
        pd_test.to_excel(writer, sheet_name="test", index=False)

    return data_train_dict, data_test_dict


def treat_discretization(cont_T, t_lo, t_hi, input_type="numpy", n_bins=10):
    '''Discretize the treatment vector 'tau' according to uniform binning.'''
    if input_type == "numpy":
        bins = np.linspace(t_lo, t_hi, n_bins + 1)
        T_binned = np.digitize(cont_T, bins) - 1
        T_binned = np.clip(T_binned, 0, n_bins - 1)
    elif input_type == "tensor":
        bins = torch.linspace(t_lo, t_hi, n_bins + 1)
        T_binned = torch.bucketize(cont_T, bins) - 1
        T_binned = torch.clamp(T_binned, 0, n_bins - 1)

    return T_binned


def total_derivative(x, t, logp, dx, dlogp, unused_context):
    del logp, dlogp, unused_context

    directional_dx = torch.autograd.grad(dx, x, dx, create_graph=True)[0]

    try:
        u = torch.full_like(dx, 1/x.numel(), requires_grad=True)
        tmp = torch.autograd.grad( (u*dx).sum(), t, create_graph=True)[0]
        partial_dt = torch.autograd.grad(tmp.sum(), u, create_graph=True)[0]

        total_deriv = directional_dx + partial_dt
    except RuntimeError as e:
        if 'One of the differentiated Tensors' in e.__str__():
            raise RuntimeError('No partial derivative with respect to time. Use mathematically equivalent '
                               '"directional_derivative" regularizer instead')

    tdv2 = total_deriv.pow(2).view(x.size(0), -1)

    return 0.5*tdv2.mean(dim=-1)


def directional_derivative(x, t, logp, dx, dlogp, unused_context):
    del t, logp, dlogp, unused_context

    directional_dx = torch.autograd.grad(dx, x, dx, create_graph=True)[0]
    ddx2 = directional_dx.pow(2).view(x.size(0),-1)

    return 0.5*ddx2.mean(dim=-1)


def quadratic_cost(x, t, logp, dx, dlogp, unused_context):
    del x, logp, dlogp, t, unused_context
    dx = dx.view(dx.shape[0], -1)
    return 0.5*dx.pow(2).mean(dim=-1)


def jacobian_frobenius_regularization_fn(x, t, logp, dx, dlogp, context):
    sh = x.shape
    del logp, dlogp, t, dx, x
    sqjac = context.sqjacnorm

    return context.sqjacnorm


def creat_reg_fns(density_dict):
    REGULARIZATION_FNS = {
        "kinetic_energy": quadratic_cost,
        "jacobian_norm2": jacobian_frobenius_regularization_fn,
        "directional_penalty": directional_derivative
    }

    reg_fns = []
    reg_coeffs = []

    for arg_key, reg_fn in six.iteritems(REGULARIZATION_FNS):
        if density_dict[arg_key] is not None:
            reg_fns.append(reg_fn)
            reg_coeffs.append(density_dict[arg_key])

    reg_fns = tuple(reg_fns)
    reg_coeffs = tuple(reg_coeffs)
    return reg_fns, reg_coeffs


def get_transforms(model):

    def sample_fn(z, logpz=None):
        if logpz is not None:
            return model(z, logpz, reverse=True)
        else:
            return model(z, reverse=True)

    def density_fn(x, logpx=None):
        if logpx is not None:
            return model(x, logpx, reverse=False)
        else:
            return model(x, reverse=False)

    return sample_fn, density_fn


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


def data_normalization(data, scal_met="Min-Max Scaling", feat_lbd=None, feat_ubd=None):
    if scal_met == "Min-Max Scaling":
        norm_tensor = ((data - data.min()) / (data.max() - data.min())) * (feat_ubd - feat_lbd) + feat_lbd

    elif scal_met == "Standardization":
        norm_tensor = (data - data.mean()) / data.std()

    elif scal_met == "Robust Scaling":
        Q1 = torch.quantile(data, 0.25, dim=0)
        Q3 = torch.quantile(data, 0.75, dim=0)

        norm_tensor = (data - Q1) / (Q3 - Q1)

    return norm_tensor


def initialize_setting(device, train_dict, test_dict, args):
    val_dict = {}
    train_val_ratio = int(0.2 * train_dict["X"].shape[0])

    for key in train_dict.keys():
        val_dict[key] = train_dict[key][:train_val_ratio, :]
        train_dict[key] = train_dict[key][train_val_ratio:, :]

    if args.dist_shift == 0:
        for key in test_dict.keys():
            test_dict[key] = test_dict[key][:args.test_size, :]
        shift_env, shift_q = None, None
    else:
        shift_env = test_dict['envs']
        test_dict = test_dict['base']

    pol_dict = {"X": train_dict["X"], "T": train_dict["T"], "Y": train_dict["Y"], "Y_all": train_dict["Y_all"],
                "X_val": val_dict["X"], "T_val": val_dict["T"], "Y_val": val_dict["Y"], "Y_all_val": val_dict["Y_all"],
                "X_test": test_dict["X"], "T_test": test_dict["T"], "Y_test": test_dict["Y"], "Y_all_test": test_dict["Y_all"],
                "shift_env": shift_env,
                "propensity_met": args.propensity_met, "alpha_lb": None, "alpha_ub": None,
                "weight_decay": args.weight_decay, "amb_radius": args.amb_radius, "norm_met": args.norm_met,
                "robust_met": args.robust_met, "opt_procedure": args.opt_procedure,
                "estimator_choices": args.estimator_choices, "num_treat": args.num_treat,
                "dist_shift": args.dist_shift}

    input_dim_T, input_dim_X = pol_dict["T"].shape[1], pol_dict["X"].shape[1]
    input_dim = input_dim_T + input_dim_X

    pol_dict["prop_score"] = train_dict["prop_score"]
    pol_dict["prop_score_val"] = val_dict["prop_score"]
    pol_dict["prop_score_test"] = test_dict["prop_score"]
    path_regress, path_prob = None, None

    '''training for the propensity score model'''

    if args.propensity_met == "unknown":
        dataset_train = sim_dataset_reg(train_dict["X"], train_dict["T"])
        dataset_val = sim_dataset_reg(val_dict["X"], val_dict["T"])
        # set up the models for training
        lr = models.Prob_model(args, input_dim_X, device, Nclasses=args.num_treat)
        # dataloader
        dl_train = create_dataloader(dataset_train, args.batch_size, shuffle=True)
        dl_val = create_dataloader(dataset_val, val_dict["X"].shape[0], shuffle=False)
        path_prob = training.train_Prob_model(args, lr, dl_train, dl_val, device, test_dict['prop_score'])

    pol_dict["tr_regression_path"], pol_dict["tr_propensity_score_path"] = path_regress, path_prob
    pol_dict["args"] = args
    pol_dict["dim_T"], pol_dict["dim_X"], pol_dict["dim_TX"] = input_dim_T, input_dim_X, input_dim

    # If Y takes normalized values, then M should be M = np.max((np.max(train_dict["normalized_Y"]),
    # np.max(test_dict["normalized_Y"])))
    pol_dict["M"] = np.max((np.max(train_dict["Y"]), np.max(test_dict["Y"])))

    pol_dict["alpha_ub"] = 1 / args.amb_radius * pol_dict["M"]
    pol_dict["alpha_lb"] = args.alpha_lb
    pol_dict["init_alpha"] = np.random.uniform(pol_dict["alpha_lb"], pol_dict["alpha_ub"])

    return pol_dict


def set_large_fonts():
    """设置所有图形的大字体参数"""
    plt.rcParams.update({
        # 字体相关
        'font.size': 18,
        'font.family': ['sans-serif'],
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.weight': 'normal',

        # 标题相关
        'axes.titlesize': 18,
        'axes.titleweight': 'bold',
        'figure.titlesize': 20,
        'figure.titleweight': 'bold',

        # 坐标轴相关
        'axes.labelsize': 20,
        'axes.labelweight': 'bold',
        'axes.linewidth': 1.2,

        # 刻度相关
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,

        # 图例相关
        'legend.fontsize': 12,
        'legend.title_fontsize': 13,
        'legend.frameon': True,
        'legend.framealpha': 0.8,

        # 线条相关
        'lines.linewidth': 2,
        'lines.markersize': 8,

        # 图形相关
        'figure.figsize': [10, 6],
        'figure.dpi': 100,
        'figure.autolayout': True,

        # 保存相关
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })