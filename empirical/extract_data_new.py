import pandas as pd
import numpy as np
import os
import warnings
import re
warnings.filterwarnings('ignore')
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union
from dataclasses import dataclass


def extract_order_info():
    # 省份ID -> 全称映射
    id_to_name = {
        1: "北京市",
        2: "上海市",
        3: "天津市",
        4: "重庆市",
        5: "河北省",
        6: "山西省",
        7: "河南省",
        8: "辽宁省",
        9: "吉林省",
        10: "黑龙江省",
        11: "内蒙古自治区",
        12: "江苏省",
        13: "山东省",
        14: "安徽省",
        15: "浙江省",
        16: "福建省",
        17: "湖北省",
        18: "湖南省",
        19: "广东省",
        20: "广西壮族自治区",
        21: "江西省",
        22: "四川省",
        23: "海南省",
        24: "贵州省",
        25: "云南省",
        26: "西藏自治区",
        27: "陕西省",
        28: "甘肃省",
        29: "青海省",
        30: "宁夏回族自治区",
        31: "新疆维吾尔自治区",
        32: "台湾省",
        52993: "港澳"
    }

    # 经济分档
    econ_level_map = {
        # level 6
        "西藏自治区": 6, "上海市": 6, "福建省": 6,

        # level 5
        "北京市": 5, "浙江省": 5, "江苏省": 5, "天津市": 5, "四川省": 5,

        # level 4
        "江西省": 4, "安徽省": 4, "河北省": 4, "黑龙江省": 4, "山东省": 4, "内蒙古自治区": 4, "吉林省": 4,

        # level 3
        "重庆市": 3, "湖北省": 3, "湖南省": 3, "辽宁省": 3, "广东省": 3,  "新疆维吾尔自治区": 3,

        # level 2
        "贵州省": 2, "宁夏回族自治区": 2, "河南省": 2, "山西省": 2,

        # level 1
        "广西壮族自治区": 1, "青海省": 1, "云南省": 1, "甘肃省": 1, "海南省": 1, "陕西省": 1,
    }

    for Q in [1, 2, 3, 4, 5, 6, 7, 8]:
        df_orders_1 = pd.read_csv(f'D:\Data\JD_wq\data\8_processed\\t={3*Q-2}\orders.csv')
        df_orders_2 = pd.read_csv(f'D:\Data\JD_wq\data\8_processed\\t={3*Q-1}\orders.csv')
        df_orders_3 = pd.read_csv(f'D:\Data\JD_wq\data\8_processed\\t={3*Q}\orders.csv')
        df_orders = pd.concat([df_orders_1, df_orders_2, df_orders_3], axis=0).sort_values(by=['account_id'])

        df_orders['coupon'] = df_orders['ord_pay_jqdq'] + df_orders['ord_pay_gift_card'] + df_orders['ord_pay_acct_bal']\
                              + df_orders['ord_pay_jbean'] +df_orders['ord_pay_fcoupon']
        select_ord_feat = ['account_id', 'ord_item_qtty', 'ord_amt_bef_dsct', 'ord_amt_aft_dsct', 'ord_pay_actual',
                           'ord_amt_freight', 'num_sku']
        df_orders = df_orders[select_ord_feat]
        # 先对数值列求和
        num_cols = ['ord_item_qtty', 'ord_amt_bef_dsct', 'ord_amt_aft_dsct',
                    'ord_pay_actual', 'ord_amt_freight', 'num_sku']
        df_orders = df_orders.groupby('account_id')[num_cols].sum()

        df_clicks_1 = pd.read_csv(f'D:\Data\JD_wq\data\8_processed\\t={3*Q-2}\clicks.csv')
        df_clicks_2 = pd.read_csv(f'D:\Data\JD_wq\data\8_processed\\t={3*Q-1}\clicks.csv')
        df_clicks_3 = pd.read_csv(f'D:\Data\JD_wq\data\8_processed\\t={3*Q}\clicks.csv')
        df_clicks = pd.concat([df_clicks_1, df_clicks_2, df_clicks_3], axis=0).sort_values(by=['account_id'])

        df_clicks['n_mbp'] = df_clicks['n_mb'] / df_clicks['n_visit']
        df_clicks['n_pcp'] = df_clicks['n_pc'] / df_clicks['n_visit']
        select_clk_feat = ['account_id', 'n_visit', 'n_sku', 'n_shop', 'n_brand', 'n_mbp', 'n_pcp']
        df_clicks = df_clicks[select_clk_feat]
        df_click = df_clicks.groupby('account_id').agg({
            'n_visit': 'sum',
            'n_sku': 'sum',
            'n_shop': 'sum',
            'n_brand': 'sum',
            'n_mbp': 'mean',
            'n_pcp': 'mean'
        }).reset_index()

        df_loans_1 = pd.read_csv(f'D:\Data\JD_wq\data\8_processed\\t={3*Q-2}\loans.csv')
        df_loans_2 = pd.read_csv(f'D:\Data\JD_wq\data\8_processed\\t={3*Q-1}\loans.csv')
        df_loans_3 = pd.read_csv(f'D:\Data\JD_wq\data\8_processed\\t={3*Q}\loans.csv')
        df_loans = pd.concat([df_loans_1, df_loans_2, df_loans_3], axis=0).sort_values(by=['account_id'])

        select_loan_feat = ['account_id', 'loanamount', 'planfee', 'planfeedsct', 'num_loan', 'amt_repay', 'num_repay',
                            'amt_refund', 'num_refund']
        df_loans = df_loans[select_loan_feat]
        df_loans = df_loans.groupby('account_id').agg('sum').reset_index()

        df_bills_1 = pd.read_csv(f'D:\Data\JD_wq\data\8_processed\\t={3*Q-2}\\bills.csv')
        df_bills_2 = pd.read_csv(f'D:\Data\JD_wq\data\8_processed\\t={3*Q-1}\\bills.csv')
        df_bills_3 = pd.read_csv(f'D:\Data\JD_wq\data\8_processed\\t={3*Q}\\bills.csv')
        df_bills = pd.concat([df_bills_1, df_bills_2, df_bills_3], axis=0).sort_values(by=['account_id'])

        df_bills['credit_usage'] = (df_bills['bd_credit_limit_available'] + df_bills['bd_tmp_credit_limit_available']) \
                                   / (df_bills['bd_credit_limit'] + df_bills['bd_tmp_credit_limit'])
        select_bill_feat = ['account_id', 'rate_order_plan', 'rate_bill_plan', 'rate_over', 'bd_credit_limit',
                            'credit_usage', 'sdpamt_pd']
        df_bills = df_bills[select_bill_feat]
        df_bills = df_bills.groupby('account_id').agg({
            'rate_order_plan': 'mean',
            'rate_bill_plan': 'mean',
            'rate_over': 'mean',
            'bd_credit_limit': 'mean',
            'credit_usage': 'mean',
            'sdpamt_pd': 'max'
        }).reset_index()

        df_demo = pd.read_csv(r'D:\Data\JD_wq\data\8_processed\user_demo.csv')
        df_demo["address_name"] = df_demo["address"].map(id_to_name)
        df_demo["econ_level_from_address"] = df_demo["address_name"].map(econ_level_map)

        # id_card_prov 列 (中文全称 → econ_level)
        df_demo["econ_level_from_idcard"] = df_demo["id_card_prov"].map(econ_level_map)

        df = pd.merge(left=df_demo, right=df_click, on='account_id', how='inner')
        df = pd.merge(left=df, right=df_orders, on='account_id', how='inner')
        df = pd.merge(left=df, right=df_loans, on='account_id', how='inner')
        df = pd.merge(left=df, right=df_bills, on='account_id', how='inner')

        df['profit'] = df['ord_pay_actual'] * 0.1 - df['sdpamt_pd']
        cuts = [-np.inf, 23.06, 26.48, 30.00, 34.69, np.inf]
        labels = [1, 2, 3, 4, 5]

        df["age_group"] = pd.cut(
            df["age"],
            bins=cuts,
            labels=labels,
            right=True,
            include_lowest=True
        ).astype(int)
        df.to_csv(f'data/Empirical/data_Q{Q}.csv', index=False, encoding="utf-8-sig")


def add_next_month():
    for Q in [1, 2, 3, 4, 5, 6, 7]:
        df_now = pd.read_csv(f'data/Empirical/data_Q{Q}.csv')
        df_next = pd.read_csv(f'data/Empirical/data_Q{Q+1}.csv')
        df_next = df_next[['account_id', 'bd_credit_limit', 'profit']]
        df_next.rename(columns={'bd_credit_limit': 'bd_credit_limit_next', 'profit': 'profit_next'}, inplace=True)
        df_now = pd.merge(left=df_now, right=df_next, on='account_id', how='left')

        df_now.loc[df_now["bd_credit_limit_next"].isna(), "bd_credit_limit_next"] = df_now.loc[df_now["bd_credit_limit_next"].isna(), "bd_credit_limit"]
        df_now.loc[df_now["profit_next"].isna(), "profit_next"] = 0

        df_now.to_csv(f'data/Empirical/data_Q{Q}.csv', index=False, encoding="utf-8-sig")


def description():
    data_list = []
    for Q in [1, 2, 3, 4, 5, 6, 7, 8]:
        data_Q = pd.read_csv(f'data/Empirical/data_Q{str(Q)}_semi.csv')
        data_Q.dropna(inplace=True)
        des_Q = data_Q.describe()
        if not os.path.exists('data/Empirical/description'):
            os.makedirs('data/Empirical/description')
        des_Q.to_csv(f'data/Empirical/description/des_Q{str(Q)}_semi.csv')
        data_list.append(data_Q)

    data_list = pd.concat(data_list, axis=0)
    data_list.dropna(inplace=True)
    des = data_list.describe()
    des.to_csv(f'data/Empirical/description/des_total_semi.csv')

    for E in [1, 2, 3, 4, 5, 6]:
        data_E = data_list[data_list['econ_level_from_address'] == E]
        data_E.dropna(inplace=True)
        des_E = data_E.describe()

        des_E.to_csv(f'data/Empirical/description/des_E{str(E)}_semi.csv')

    for A in [1, 2, 3, 4, 5]:
        data_A = data_list[data_list['age_group'] == A]
        data_A.dropna(inplace=True)
        des_A = data_A.describe()

        des_A.to_csv(f'data/Empirical/description/des_A{str(A)}_semi.csv')



def data_split():
    for Q in [1, 2, 3, 4, 5, 6, 7, 8]:
        data_Q = pd.read_csv(f"data\Empirical\data_Q{str(Q)}_semi.csv", low_memory=False)
        # data_Q = data_Q[data_Q['bd_credit_limit'] <= np.quantile(data_Q['bd_credit_limit'], q=0.99)]

        '''
        data_Q_next = pd.read_csv(f"data\Empirical\data_Q{str(Q+1)}.csv", low_memory=False)
        data_Q_next = data_Q_next[['account_id', 'bd_credit_limit', 'ord_pay_actual', 'sdpamt_pd']]
        data_Q_next.rename(columns={'ord_pay_actual': 'ord_pay_actual_next', 'sdpamt_pd': 'sdpamt_pd_next',
                                    'bd_credit_limit': 'bd_credit_limit_next'}, inplace=True)
        data_Q = pd.merge(left=data_Q, right=data_Q_next, on='account_id', how='inner')
        '''
        for level in [1, 2, 3, 4, 5, 6]:
            data_Q_econ = data_Q[data_Q["econ_level_from_address"] == level]

            for age in [1, 2, 3, 4, 5]:
                data_Q_econ_age = data_Q_econ[data_Q_econ["age_group"] == age]

                # 离散化credit line作为treatment
                # data_Q_econ_age['credit_line_group'] = np.floor(data_Q_econ_age['bd_credit_limit'] / 1000).astype('Int64')
                # data_Q_econ_age['credit_line_group'] = data_Q_econ_age['credit_line_group'].clip(lower=0, upper=19)
                # data_Q_econ['credit_line_group_next'] = np.floor(data_Q_econ['bd_credit_limit_next'] / 1000).astype('Int64')
                # data_Q_econ['credit_line_group_next'] = data_Q_econ['credit_line_group_next'].clip(lower=0, upper=19)

                data_Q_econ_age.to_csv(f'data\Empirical\data_Q{str(Q)}_econ{str(level)}_age{str(age)}_semi.csv')

                # 删除无用的feature
                # data_Q_econ.drop(['birthday', 'id_card_prov', 'address', 'address_name'], axis=1, inplace=True)
                # 'ord_amt_bef_dsct', 'ord_amt_aft_dsct', 'ord_pay_actual', 'sdpamt_pd'


'''The following code are the process of DGP'''
# ----------------------------- helpers -----------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    m = np.max(logits, axis=axis, keepdims=True)
    ex = np.exp(logits - m)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + eps)


def _winsorize(x: np.ndarray, lo_q: float = 0.01, hi_q: float = 0.99) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo = np.quantile(x, lo_q)
    hi = np.quantile(x, hi_q)
    return np.clip(x, lo, hi)


def _solve_logit_intercept_for_target_rate(
    linpred: np.ndarray, target_rate: float, max_iter: int = 60
) -> float:
    """
    Find b such that mean(sigmoid(b + linpred)) approx target_rate.
    Uses bisection on b in a wide range.
    """
    target_rate = float(np.clip(target_rate, 1e-4, 1 - 1e-4))
    lo, hi = -30.0, 30.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p = _sigmoid(mid + linpred).mean()
        if p < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------- DGP configuration -----------------------------

@dataclass
class SemiSynthConfig:
    K: int = 20  # actions 0..K-1
    credit_step: float = 1000.0  # credit amount per level
    promo_quarters: Tuple[int, ...] = (2, 4, 6, 8)

    # Behavior policy / propensity
    prop_temperature: float = 1.2     # higher -> smoother propensities
    prop_sigma: float = 2.5           # controls concentration around "center" action
    min_prop: float = 1e-3            # numerical floor for pi0

    # Spending / margin process
    base_growth: float = 0.05         # baseline growth from historical to future
    margin_nonpromo: float = 0.085    # ~8.5% margin (within 5-10%)
    margin_promo: float = 0.060       # promo quarter lower margin
    promo_spend_boost: float = 0.18   # stronger credit->spend response under promotion

    # Default process
    target_default_rate: float = 0.14     # overall event rate (keeps Q75=0 plausible)
    promo_default_mult: float = 1.12      # promo quarter slightly higher default risk
    loss_scale_mult: float = 0.55         # scales severity in currency units

    # Nonlinearity knobs
    spend_sat_scale: float = 7.5          # saturation for credit effect
    risk_sat_scale: float = 6.0           # saturation for risk effect

    # Noise
    spend_noise_sd: float = 0.22          # log-normal noise on future spend
    loss_noise_shape: float = 2.0         # gamma shape for loss severity noise

    # Calibration
    calibrate_each_quarter: bool = True
    verbose: bool = True


# ----------------------------- core semi-synthetic generator -----------------------------

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple


# ----------------------------- helpers -----------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    m = np.max(logits, axis=axis, keepdims=True)
    ex = np.exp(logits - m)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + eps)


def _winsorize(x: np.ndarray, lo_q: float = 0.01, hi_q: float = 0.99) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo = np.quantile(x, lo_q)
    hi = np.quantile(x, hi_q)
    return np.clip(x, lo, hi)


def _solve_logit_intercept_for_target_rate(
    linpred: np.ndarray, target_rate: float, max_iter: int = 60
) -> float:
    target_rate = float(np.clip(target_rate, 1e-4, 1 - 1e-4))
    lo, hi = -30.0, 30.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p = _sigmoid(mid + linpred).mean()
        if p < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------- new DGP configuration -----------------------------

@dataclass
class SemiSynthConfig:
    # 5 local credit-adjustment actions relative to current limit
    # 0:-10%, 1:-5%, 2:keep, 3:+5%, 4:+10%
    action_deltas: Tuple[float, ...] = (-0.10, -0.05, 0.00, 0.05, 0.10)
    promo_quarters: Tuple[int, ...] = (2, 4, 6, 8)

    # behavior policy
    prop_temperature: float = 0.9
    prop_sigma: float = 0.055
    min_prop: float = 1e-3

    # spending / revenue
    base_growth: float = 0.03
    margin_nonpromo: float = 0.085
    margin_promo: float = 0.060
    promo_spend_boost: float = 0.20

    # default process
    target_default_rate_keep: float = 0.10
    promo_default_mult: float = 1.10
    loss_scale_mult: float = 0.75

    # noise
    spend_noise_sd: float = 0.18
    loss_noise_shape: float = 2.0

    verbose: bool = True


# ----------------------------- core semi-synthetic generator -----------------------------

def generate_semi_synth_credit_quarters(
    input_dir: str,
    output_dir: str,
    cfg: SemiSynthConfig = SemiSynthConfig(),
    file_pattern: str = "data_Q{}.csv",
    seed: int = 123,
    save_csv: bool = True,
    save_npz: bool = True,
) -> pd.DataFrame:
    """
    Reads data_Q1.csv ... data_Q8.csv from input_dir.
    Adds:
      - A_sim: action in {0,1,2,3,4}
      - delta_pct_sim: {-10,-5,0,5,10}
      - bd_credit_limit_new_sim: adjusted credit line after action
      - pi0_sim: logging propensity of chosen action
      - profit_sim: observed profit under chosen action

    Saves:
      - CSV with appended columns
      - NPZ with profit_all (n,5), pi0_all (n,5), new_limit_all (n,5)
    """
    os.makedirs(output_dir, exist_ok=True)
    rng_global = np.random.default_rng(seed)

    deltas = np.array(cfg.action_deltas, dtype=np.float64)   # (5,)
    K = len(deltas)

    summaries = []

    for q in range(1, 9):
        path = os.path.join(input_dir, file_pattern.format(q))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing quarter file: {path}")

        df = pd.read_csv(path)
        n = df.shape[0]

        def col(name, default=0.0):
            if name not in df.columns:
                raise KeyError(f"Column '{name}' not found in {path}")
            v = df[name].copy()
            if v.isna().any():
                v = v.fillna(default)
            return v.to_numpy(dtype=np.float64)

        # ---------- covariates ----------
        gender = col("gender")
        age = col("age")
        account_age = col("account_age")
        econ_res = col("econ_level_from_address")
        econ_birth = col("econ_level_from_idcard")

        n_visit = col("n_visit")
        n_sku = col("n_sku")
        n_shop = col("n_shop")
        n_brand = col("n_brand")
        n_mbp = col("n_mbp")
        n_pcp = col("n_pcp")

        ord_item_qtty = col("ord_item_qtty")
        ord_amt_bef_dsct = col("ord_amt_bef_dsct")
        ord_amt_aft_dsct = col("ord_amt_aft_dsct")
        ord_pay_actual = col("ord_pay_actual")
        ord_amt_freight = col("ord_amt_freight")
        num_sku = col("num_sku")

        loanamount = col("loanamount")
        planfee = col("planfee")
        planfeedsct = col("planfeedsct")
        num_loan = col("num_loan")
        amt_repay = col("amt_repay")
        num_repay = col("num_repay")
        amt_refund = col("amt_refund")
        num_refund = col("num_refund")

        rate_order_plan = col("rate_order_plan")
        rate_bill_plan = col("rate_bill_plan")
        rate_over = col("rate_over")
        sdpamt_pd = col("sdpamt_pd")
        current_limit = col("bd_credit_limit")
        # credit_usage = col("credit_usage", default=0.5)

        current_limit_safe = np.maximum(current_limit, 1000.0)

        # ---------- latent factors ----------
        eng = (
            0.55 * np.log1p(_winsorize(n_visit))
            + 0.35 * np.log1p(_winsorize(n_sku))
            + 0.25 * np.log1p(_winsorize(n_shop))
            + 0.20 * np.log1p(_winsorize(n_brand))
            + 0.10 * (n_mbp - n_pcp)
        )
        eng = _zscore(eng)

        spend_hist = _zscore(np.log1p(_winsorize(ord_pay_actual)))
        order_hist = _zscore(np.log1p(_winsorize(num_sku + 1.0)))
        borrow_hist = _zscore(np.log1p(_winsorize(loanamount + 1.0)))
        repay_ratio = (amt_repay + 1.0) / (loanamount + 50.0)
        repay_ratio = _winsorize(repay_ratio, 0.01, 0.99)
        repay_ratio_z = _zscore(repay_ratio)
        overdue = _zscore(np.log1p(_winsorize(sdpamt_pd)))
        # util = _zscore(_winsorize(credit_usage, 0.01, 0.99))

        # business-driven risk / opportunity signals
        young = _zscore(-(age - 30.0))      # younger -> larger
        rich = _zscore(econ_res)            # richer region -> larger
        tenure = _zscore(account_age)
        low_limit = _zscore(-np.log1p(current_limit_safe))  # lower current limit -> larger

        # constrained but good users benefit more from credit increases
        demand_score = (
            0.55 * eng
            + 0.30 * spend_hist
            + 0.20 * order_hist
            + 0.30 * rich
            + 0.18 * tenure
            + 0.20 * low_limit
        )

        # risky users should not receive aggressive increases
        risk_score = (
            0.85 * overdue
            - 0.55 * repay_ratio_z
            + 0.35 * borrow_hist
            + 0.45 * young
            - 0.55 * rich
            - 0.10 * tenure
        )

        # vulnerable subgroup: young + overdue + poorer region + high utilization
        vulnerable = _sigmoid(0.9 * overdue + 0.7 * young - 0.8 * rich)

        is_promo = 1.0 if (q in cfg.promo_quarters) else 0.0

        # ------------------------------------------------------------------
        # (1) Logging policy over 5 local actions
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # (1) Logging policy over 5 local actions
        # ------------------------------------------------------------------
        delta_all = deltas[None, :]  # (1,5), values in {-0.10,-0.05,0,0.05,0.10}

        # 受约束程度：当前额度较低、历史支付较高、浏览活跃，说明加额更可能转化为消费
        constrained = _sigmoid(
            1.00 * low_limit
            + 0.55 * eng
            + 0.25 * spend_hist
            - 0.35 * overdue
            - 0.20 * borrow_hist
        )

        # user-specific optimal delta in [-0.10, 0.10]
        # 比原来更分散，让不同子群体更明显地落到不同档位
        delta_star = 0.10 * np.tanh(
            1.20 * constrained
            + 0.55 * np.maximum(rich, 0.0)
            - 1.35 * vulnerable
            - 0.35 * young
            + 0.30 * is_promo
            - 0.20
        )  # shape (n,)

        # logging center follows delta_star, but keep local adjustment behavior
        center_delta = np.clip(
            delta_star
            + 0.015 * np.tanh(0.6 * eng - 0.6 * overdue)
            + 0.010 * np.tanh(0.6 * rich)
            - 0.010 * np.tanh(0.5 * young),
            -0.095, 0.095
        )[:, None]

        # 中间动作仍更常见，但不要过强压制两端
        locality_penalty = np.array([0.30, 0.08, 0.00, 0.08, 0.30], dtype=np.float64)[None, :]

        quad = -0.5 * ((delta_all - center_delta) / cfg.prop_sigma) ** 2
        hetero = (
                0.45 * constrained[:, None] * delta_all
                - 0.55 * vulnerable[:, None] * np.maximum(delta_all, 0.0)
                + 0.15 * rich[:, None] * delta_all
                - 0.12 * young[:, None] * np.maximum(delta_all, 0.0)
        )

        logits = (quad - locality_penalty + hetero) / cfg.prop_temperature

        pi0_all = _softmax(logits, axis=1)
        pi0_all = np.clip(pi0_all, cfg.min_prop, 1.0)
        pi0_all = pi0_all / pi0_all.sum(axis=1, keepdims=True)

        A = np.array([rng_global.choice(K, p=pi0_all[i]) for i in range(n)], dtype=np.int64)
        pi0_chosen = pi0_all[np.arange(n), A]
        chosen_delta = deltas[A]

        new_limit_obs = np.maximum(current_limit_safe * (1.0 + chosen_delta), 0.0)
        new_limit_all = np.maximum(current_limit_safe[:, None] * (1.0 + delta_all), 0.0)

        # ------------------------------------------------------------------
        # (2) Potential outcomes for all 5 actions
        # ------------------------------------------------------------------

        # baseline future payment
        base_pay = (
                0.78 * _winsorize(ord_pay_actual)
                + 150.0 * np.maximum(eng, 0.0)
                + 80.0 * np.maximum(order_hist, 0.0)
                + 60.0 * np.maximum(rich, 0.0)
        )
        base_pay = np.maximum(base_pay, 100.0)

        # gap between chosen delta and ideal delta
        delta_gap = (delta_all - delta_star[:, None]) / 0.05
        delta_gap_abs = np.abs(delta_gap)
        up = np.maximum(delta_all, 0.0)
        down = np.maximum(-delta_all, 0.0)

        # ---------- Future payment ----------
        # upward adjustment helps more for constrained users and in promotion quarters
        pay_gain = (
                (0.030 + 0.065 * constrained[:, None] + 0.020 * np.maximum(rich[:, None], 0.0))
                * (delta_all / 0.05)
                + 0.035 * is_promo * np.maximum(delta_all, 0.0) / 0.05
        )

        # 强化相邻动作之间的差异：偏离 ideal delta 会明显降低未来支付转化效率
        # 这里从原来的 0.03 * gap^2 提高到更强的惩罚
        pay_penalty = 0.060 * (delta_gap_abs ** 1.8)

        eps_spend = rng_global.normal(0.0, cfg.spend_noise_sd, size=n)[:, None]
        mult_spend = np.exp(eps_spend)

        pay_multiplier = 1.0 + cfg.base_growth + pay_gain - pay_penalty
        pay_multiplier = np.maximum(pay_multiplier, 0.50)

        pay_all = base_pay[:, None] * pay_multiplier * mult_spend
        pay_all = np.maximum(pay_all, 0.0)

        # quarter-specific margin
        margin_rate = cfg.margin_promo if is_promo else cfg.margin_nonpromo
        revenue_all = margin_rate * pay_all

        # ---------- default probability ----------
        base_risk = (
                1.00 * overdue
                - 0.65 * repay_ratio_z
                + 0.35 * borrow_hist
                + 0.48 * young
                - 0.60 * rich
                - 0.12 * tenure
        )

        # if action exceeds user's ideal delta, risk rises sharply
        delta_excess = np.maximum(delta_all - delta_star[:, None], 0.0) / 0.05
        delta_below = np.maximum(delta_star[:, None] - delta_all, 0.0) / 0.05

        ref_lin = base_risk
        b0 = _solve_logit_intercept_for_target_rate(ref_lin, cfg.target_default_rate_keep)

        # 比原来更强地惩罚“过度加额”
        p_def_all = _sigmoid(
            b0
            + base_risk[:, None]
            + 0.85 * delta_excess
            + 1.10 * vulnerable[:, None] * (delta_excess ** 2)
            - 0.22 * delta_below
        )

        if is_promo:
            p_def_all = np.clip(p_def_all * cfg.promo_default_mult, 1e-6, 0.95)
        else:
            p_def_all = np.clip(p_def_all, 1e-6, 0.95)

        u_def = rng_global.uniform(size=n)[:, None]
        default_event_all = (u_def < p_def_all).astype(np.float64)

        # ---------- default severity ----------
        sev_base = (
                100.0
                + 0.10 * _winsorize(loanamount)[:, None]
                + 0.012 * current_limit_safe[:, None]
                + 125.0 * _sigmoid(0.9 * overdue)[:, None]
                + 90.0 * delta_excess
                + 65.0 * vulnerable[:, None] * delta_excess
        )
        sev_base = np.maximum(sev_base, 0.0)

        gamma_noise = rng_global.gamma(
            shape=cfg.loss_noise_shape,
            scale=1.0 / cfg.loss_noise_shape,
            size=(n, K)
        )
        loss_all = cfg.loss_scale_mult * default_event_all * sev_base * gamma_noise

        # ---------- adjustment cost ----------
        # 进一步强化“偏离合适调额”的代价，尤其对高额度用户更明显
        adj_cost = (
                           18.0 + 0.0045 * current_limit_safe
                   )[:, None] * (delta_gap_abs ** 2)

        # ---------- final profit ----------
        profit_all = revenue_all - loss_all - adj_cost
        profit_obs = profit_all[np.arange(n), A] # (1,5), values in {-0.10,-0.05,0,0.05,0.10}

        '''
        oracle_a =np.argmax(profit_all, axis=1)
        vals, cnts = np.unique(oracle_a, return_counts=True)
        # print("Q", q, dict(zip(vals, cnts)))
        print("Q", q, profit_all.mean(axis=0))
        
        print(q, np.min(profit_obs), np.quantile(profit_obs, q=[0.1, 0.3, 0.5, 0.7, 0.9]), np.max(profit_obs))
        '''

        # ------------------------------------------------------------------
        # save outputs
        # ------------------------------------------------------------------
        df_out = df.copy()
        df_out["A_sim"] = A
        df_out["delta_pct_sim"] = chosen_delta * 100.0
        df_out["bd_credit_limit_new_sim"] = new_limit_obs
        df_out["pi0_sim"] = pi0_chosen
        df_out["profit_sim"] = profit_obs

        if save_csv:
            out_csv = os.path.join(output_dir, f"data_Q{q}_semi.csv")
            df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

        if save_npz:
            out_npz = os.path.join(output_dir, f"data_Q{q}_semi.npz")
            np.savez_compressed(
                out_npz,
                A=A.astype(np.int64),
                delta_values=deltas.astype(np.float32),
                delta_chosen=chosen_delta.astype(np.float32),
                pi0_all=pi0_all.astype(np.float32),
                pi0_chosen=pi0_chosen.astype(np.float32),
                profit_obs=profit_obs.astype(np.float32),
                profit_all=profit_all.astype(np.float32),
                pay_all=pay_all.astype(np.float32),
                loss_all=loss_all.astype(np.float32),
                new_limit_all=new_limit_all.astype(np.float32),
                quarter=np.full(n, q, dtype=np.int64),
                is_promo=np.full(n, int(is_promo), dtype=np.int64),
            )

        # diagnostics
        chosen_default_rate = default_event_all[np.arange(n), A].mean()
        action_counts = np.bincount(A, minlength=K)

        summaries.append({
            "quarter": q,
            "promo": int(is_promo),
            "n": n,
            "A_dec10": int(action_counts[0]),
            "A_dec5": int(action_counts[1]),
            "A_keep": int(action_counts[2]),
            "A_inc5": int(action_counts[3]),
            "A_inc10": int(action_counts[4]),
            "pi0_min": float(pi0_all.min()),
            "pi0_mean_chosen": float(pi0_chosen.mean()),
            "profit_mean": float(np.mean(profit_obs)),
            "profit_std": float(np.std(profit_obs)),
            "profit_q25": float(np.quantile(profit_obs, 0.25)),
            "profit_q50": float(np.quantile(profit_obs, 0.50)),
            "profit_q75": float(np.quantile(profit_obs, 0.75)),
            "chosen_default_rate": float(chosen_default_rate),
            "new_limit_mean": float(np.mean(new_limit_obs)),
        })
        '''
        if cfg.verbose:
            print(
                f"[Q{q}] promo={int(is_promo)}, n={n}, "
                f"actions={action_counts.tolist()}, "
                f"profit_mean={np.mean(profit_obs):.2f}, "
                f"default_rate={chosen_default_rate:.3f}"
            )
        '''
    summary_df = pd.DataFrame(summaries)
    if cfg.verbose:
        print("\n=== Semi-synthetic summary ===")
        print(summary_df)

    return summary_df


if __name__ == '__main__':
    # 调用函数
    extract_order_info()
    cfg = SemiSynthConfig(
        action_deltas=(-0.10, -0.05, 0.00, 0.05, 0.10),
        verbose=True
    )

    generate_semi_synth_credit_quarters(
        input_dir='data/Empirical',
        output_dir='data/Empirical',
        cfg=cfg
    )
    description()
    data_split()