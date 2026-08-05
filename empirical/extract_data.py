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
    Reads 8 quarter CSVs (data_Q1.csv ... data_Q8.csv) from input_dir.
    Adds:
      - A_sim: simulated current credit line group in {0,...,K-1}
      - pi0_sim: logging propensity of chosen action
      - profit_sim: observed profit for chosen action
    Optionally saves:
      - CSV with appended columns
      - NPZ with arrays including full potential outcomes profit_all (n,K) and pi0_all (n,K)

    Returns a summary DataFrame with per-quarter diagnostics.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng_global = np.random.default_rng(seed)

    summaries = []

    for q in range(1, 9):
        path = os.path.join(input_dir, file_pattern.format(q))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing quarter file: {path}")

        df = pd.read_csv(path)
        n = df.shape[0]

        # --- extract covariates (as arrays) ---
        # robust to minor dtype issues; fill NA conservatively
        def col(name, default=0.0):
            if name not in df.columns:
                raise KeyError(f"Column '{name}' not found in {path}")
            v = df[name].copy()
            if v.isna().any():
                v = v.fillna(default)
            return v.to_numpy(dtype=np.float64)

        gender = col("gender")
        age = col("age")
        account_age = col("account_age")
        econ_res = col("econ_level_from_address")     # 1..6
        econ_birth = col("econ_level_from_idcard")    # 1..6

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
        num_sku = col("num_sku")  # number of orders (your naming)

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
        sdpamt_pd = col("sdpamt_pd")  # overdue amount (historical)

        df['credit_group'] = (df['bd_credit_limit'] // 1000).clip(upper=19).astype(int)
        credit_hist = col("credit_group")
        # --------- engineered latent factors ---------
        # Engagement: browsing intensity with diminishing returns + channel mix
        eng = (
            0.55 * np.log1p(_winsorize(n_visit))
            + 0.35 * np.log1p(_winsorize(n_sku))
            + 0.25 * np.log1p(_winsorize(n_shop))
            + 0.20 * np.log1p(_winsorize(n_brand))
            + 0.10 * (n_mbp - n_pcp)
        )
        eng = _zscore(eng)

        # Purchasing power proxy: historical payment and order counts
        spend_hist = np.log1p(_winsorize(ord_pay_actual))
        spend_hist = _zscore(spend_hist)

        # Credit usage proxy: historical borrowing and loans
        credit_use = 0.6 * np.log1p(_winsorize(loanamount)) + 0.25 * np.log1p(_winsorize(num_loan + 1.0))
        credit_use = _zscore(credit_use)

        # Repayment quality proxy: repay-to-borrow ratio and overdue
        repay_ratio = (amt_repay + 1.0) / (loanamount + 50.0)
        repay_ratio = _winsorize(repay_ratio, 0.01, 0.99)
        overdue = np.log1p(_winsorize(sdpamt_pd))
        overdue = _zscore(overdue)
        repay_ratio_z = _zscore(repay_ratio)

        # Demographic / region signals: as you requested
        young = _zscore(age)             # younger -> larger
        rich = _zscore(econ_res)                   # richer region -> larger
        tenure = _zscore(account_age)

        is_promo = 1.0 if (q in cfg.promo_quarters) else 0.0

        # --------- (1) simulate logging propensities for A (behavior policy) ---------
        # Center action: mostly follow historical credit, but adjust for region/age/engagement.
        # Younger -> lower credit; richer region -> higher credit; high engagement -> higher credit.
        center = (
            credit_hist
            + 1.15 * rich
            - 1.05 * young
            + 0.55 * eng
            + 0.25 * tenure
        )
        center = np.clip(center, 0.0, cfg.K - 1.0)

        a_grid = np.arange(cfg.K, dtype=np.float64)[None, :]  # (1,K)
        center2 = center[:, None]  # (n,1)

        # Quadratic preference around center + mild systematic tilt toward moderate actions
        quad = -0.5 * ((a_grid - center2) / cfg.prop_sigma) ** 2
        tilt = -0.015 * (a_grid - (cfg.K - 1) / 2.0) ** 2 / ((cfg.K - 1) ** 2)  # discourages extremes

        # Add small heterogeneous logits (nonlinear) so propensities depend on X
        # (helps realism and ensures confounding structure)
        hetero = (
            0.18 * eng[:, None] * np.tanh((a_grid - 6.0) / 4.0)
            + 0.12 * rich[:, None] * np.tanh((a_grid - 10.0) / 4.0)
            - 0.15 * overdue[:, None] * np.tanh((a_grid - 12.0) / 3.0)
        )

        logits = (quad + tilt + hetero) / cfg.prop_temperature
        pi0_all = _softmax(logits, axis=1)
        pi0_all = np.clip(pi0_all, cfg.min_prop, 1.0)
        pi0_all = pi0_all / pi0_all.sum(axis=1, keepdims=True)

        # Sample one action A
        A = np.array([rng_global.choice(cfg.K, p=pi0_all[i]) for i in range(n)], dtype=np.int64)
        pi0_chosen = pi0_all[np.arange(n), A]

        # Map action to credit amount (currency)
        credit_amt = cfg.credit_step * A.astype(np.float64)  # 0,1000,...,19000

        # --------- (2) generate potential outcomes: profit(a) for all a ---------
        # Spending process: future payment depends on historical payment + engagement + credit line (saturating)
        # Promotion quarters increase responsiveness but reduce margin (concept drift).
        a_amt = cfg.credit_step * np.arange(cfg.K, dtype=np.float64)[None, :]  # (1,K)
        a_level = np.arange(cfg.K, dtype=np.float64)[None, :]

        # nonlinear "demand lift" from credit: saturating in credit, stronger for engaged users and richer regions
        # promotion modifies the mapping (concept drift)
        lift_strength = (
            0.18
            + 0.12 * _sigmoid(0.9 * eng)            # higher engagement -> higher sensitivity
            + 0.08 * _sigmoid(0.8 * rich)           # richer -> higher spending response
        )[:, None]

        promo_lift = (1.0 + is_promo * cfg.promo_spend_boost)

        credit_lift = promo_lift * lift_strength * np.tanh(a_level / cfg.spend_sat_scale)  # (n,K)

        # add a nonlinear interaction: credit helps more when prior spend is moderate (not already saturated)
        hist_gate = (1.0 - _sigmoid(1.2 * spend_hist))[:, None]  # high past spend -> smaller marginal lift
        credit_lift = credit_lift * (0.65 + 0.55 * hist_gate)

        # base future payment level: correlated with historical pay and engagement
        base_pay = np.expm1(np.clip(_zscore(np.log1p(_winsorize(ord_pay_actual))) * 0.75 + 0.45 * eng, -2.5, 3.0))
        # scale base_pay back to currency magnitude near ord_pay_actual
        base_pay = 0.75 * _winsorize(ord_pay_actual) + 0.25 * base_pay

        # multiplicative spend noise shared across actions (keeps counterfactual coherence)
        eps_spend = rng_global.normal(0.0, cfg.spend_noise_sd, size=n)
        mult_spend = np.exp(eps_spend)[:, None]

        pay_all = (base_pay[:, None] * (1.0 + cfg.base_growth + credit_lift)) * mult_spend
        pay_all = np.maximum(pay_all, 0.0)

        # Margin rate: promotion reduces unit margin (concept drift)
        margin_rate = (cfg.margin_promo if is_promo else cfg.margin_nonpromo)
        revenue_all = margin_rate * pay_all

        # Default process: event probability increases with younger age, poorer region, worse repayment/overdue,
        # and increases with higher credit (treatment risk).
        # Also slightly higher in promotion quarters.
        risk_lin = (
            0.85 * overdue
            - 0.55 * repay_ratio_z
            + 0.35 * credit_use
            + 0.50 * young
            - 0.55 * rich
            - 0.10 * tenure
        )
        # treatment-dependent risk: nonlinear in credit and interacts with young/overdue
        treat_risk = (
            0.45 * np.tanh(a_level / cfg.risk_sat_scale)
            + 0.25 * (young[:, None] > 0.5).astype(np.float64) * np.tanh((a_level - 8.0) / 3.0)
            + 0.25 * (overdue[:, None] > 0.5).astype(np.float64) * np.tanh((a_level - 10.0) / 3.0)
        )

        # Calibrate intercept to hit a target baseline default event rate at the "typical" action near historical credit.
        # Use the chosen action distribution as reference (fast and stable).
        ref_action = np.clip(np.rint(center).astype(int), 0, cfg.K - 1)
        ref_lin = risk_lin + treat_risk[np.arange(n), ref_action]
        b0 = _solve_logit_intercept_for_target_rate(ref_lin, cfg.target_default_rate)
        p_def_base = _sigmoid(b0 + risk_lin[:, None] + treat_risk)
        p_def_all = np.clip(p_def_base * (cfg.promo_default_mult if is_promo else 1.0), 1e-6, 0.95)

        # default event noise shared across actions: one uniform draw per person
        u_def = rng_global.uniform(size=n)[:, None]
        default_event_all = (u_def < p_def_all).astype(np.float64)

        # default severity: depends on credit amount and borrowing history; gamma noise
        # ensure heavy-tail but many zeros due to default_event_all
        sev_base = (
            0.08 * a_amt
            + 0.20 * _winsorize(loanamount)[:, None]
            + 60.0 * _sigmoid(0.9 * overdue)[:, None]
        )
        sev_base = np.maximum(sev_base, 0.0)

        gamma_noise = rng_global.gamma(shape=cfg.loss_noise_shape, scale=1.0 / cfg.loss_noise_shape, size=(n, cfg.K))
        loss_all = cfg.loss_scale_mult * default_event_all * sev_base * gamma_noise

        profit_all = revenue_all - loss_all

        # observed outcome under chosen A
        profit_obs = profit_all[np.arange(n), A]

        # --------- save outputs ---------
        df_out = df.copy()
        df_out["A_sim"] = A
        df_out["pi0_sim"] = pi0_chosen
        df_out["profit_sim"] = profit_obs

        if save_csv:
            out_csv = os.path.join(output_dir, f"data_Q{q}_semi.csv")
            df_out.to_csv(out_csv, index=False)

        if save_npz:
            out_npz = os.path.join(output_dir, f"data_Q{q}_semi.npz")
            np.savez_compressed(
                out_npz,
                X=df_out.to_numpy(),                 # full table, convenient but large
                A=A.astype(np.int64),
                pi0_all=pi0_all.astype(np.float32),
                pi0_chosen=pi0_chosen.astype(np.float32),
                profit_obs=profit_obs.astype(np.float32),
                profit_all=profit_all.astype(np.float32),
                quarter=np.full(n, q, dtype=np.int64),
                is_promo=np.full(n, int(is_promo), dtype=np.int64),
            )

        # --------- diagnostics ---------
        # Compare scales to your real-world magnitudes: pay ~ a few thousand; loss sparse; profit ~ O(100)
        default_rate_obs = float((profit_obs < 0).mean())  # proxy; not exact default rate
        # exact default-event at chosen action
        chosen_default_event = default_event_all[np.arange(n), A].mean()

        summaries.append({
            "quarter": q,
            "n": n,
            "promo": int(is_promo),
            "A_mean": float(A.mean()),
            "A_std": float(A.std()),
            "pi0_min": float(pi0_all.min()),
            "pi0_mean_chosen": float(pi0_chosen.mean()),
            "pay_hist_mean": float(np.mean(ord_pay_actual)),
            "profit_mean": float(np.mean(profit_obs)),
            "profit_std": float(np.std(profit_obs)),
            "chosen_default_rate": float(chosen_default_event),
            "profit_q25": float(np.quantile(profit_obs, 0.25)),
            "profit_q50": float(np.quantile(profit_obs, 0.50)),
            "profit_q75": float(np.quantile(profit_obs, 0.75)),
        })

        if cfg.verbose:
            print(
                f"[Q{q}] n={n}, promo={int(is_promo)}, "
                f"A_mean={A.mean():.2f}, pi0_min={pi0_all.min():.4g}, "
                f"profit_mean={np.mean(profit_obs):.2f}, "
                f"chosen_default_rate={chosen_default_event:.3f}"
            )

    summary_df = pd.DataFrame(summaries)
    if cfg.verbose:
        print("\n=== Semi-synthetic summary (per quarter) ===")
        print(summary_df)

    return summary_df


if __name__ == '__main__':
    # 调用函数
    extract_order_info()
    generate_semi_synth_credit_quarters(input_dir='data/Empirical', output_dir='data/Empirical')
    description()
    data_split()