import pandas as pd
from functools import reduce


def extract_orders_id(threshold):
    # 定义月份列表
    month_id = ['2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06',
                '2019-07', '2019-08', '2019-09', '2019-10', '2019-11', '2019-12']

    # 定义上半年和下半年的月份
    first_half_months = month_id[:6]  # 1-6月
    second_half_months = month_id[6:]  # 7-12月

    def find_negative_users():
        # 创建一个集合来存储有负成本用户的pin（集合自动去重）
        negative_users = set()

        # 处理每个月的数据
        for month in month_id:
            # 加载数据
            file_path = f'C:\Data\JD database\month data\{month}\orders.csv'
            df = pd.read_csv(file_path)
            print(f"正在检查 {month} 的数据...")

            # 找出成本为负值的行
            negative_cost_rows = df[df['ord_prc_stk'] < 0]
            negative_price_rows = df[df['ord_amt_aft_dsct'] < 0]
            negative_payment_rows = df[df['ord_pay_actual'] < 0]

            if not negative_cost_rows.empty:
                # 获取有负成本用户的pin
                month_negative_pins = negative_cost_rows['pin'].unique().tolist()

                # 添加到总集合中
                negative_users.update(month_negative_pins)

                print(f"  {month} 有 {len(negative_cost_rows)} 条负成本记录，涉及 {len(month_negative_pins)} 个用户")
            else:
                print(f"  {month} 没有负成本记录")

            if not negative_price_rows.empty:
                # 获取有负成本用户的pin
                month_negative_pins = negative_price_rows['pin'].unique().tolist()

                # 添加到总集合中
                negative_users.update(month_negative_pins)

                print(f"  {month} 有 {len(negative_price_rows)} 条负价格记录，涉及 {len(month_negative_pins)} 个用户")
            else:
                print(f"  {month} 没有负价格记录")

            if not negative_payment_rows.empty:
                # 获取有负成本用户的pin
                month_negative_pins = negative_payment_rows['pin'].unique().tolist()

                # 添加到总集合中
                negative_users.update(month_negative_pins)

                print(f"  {month} 有 {len(negative_payment_rows)} 条负支付记录，涉及 {len(month_negative_pins)} 个用户")
            else:
                print(f"  {month} 没有负支付记录")

        # 将集合转换为列表
        negative_cost_users_list = list(negative_users)

        return negative_cost_users_list

    # 调用函数
    negative_users = find_negative_users()

    # 初始化结果字典
    results = {'counts': [], 'payments': [], 'prices': [], 'costs': [], 'profits': []}

    # 处理每个月的数据
    for month in month_id:
        # 加载数据
        file_path = f'C:\Data\JD database\month data\{month}\orders.csv'
        df = pd.read_csv(file_path)
        print(f"已加载: {file_path}")

        # 剔除有负成本记录的用户
        df = df[~df['pin'].isin(negative_users)]

        # 统计订单数量
        count_df = df.groupby('pin').size().reset_index(name=f'{month}_count')
        results['counts'].append(count_df)

        # 统计支付金额
        payment_df = df.groupby('pin')['ord_pay_actual'].sum().reset_index(name=f'{month}_payment')
        results['payments'].append(payment_df)

        # 统计订单金额
        price_df = df.groupby('pin')['ord_amt_aft_dsct'].sum().reset_index(name=f'{month}_price')
        results['prices'].append(price_df)

        # 统计成本
        cost_df = df.groupby('pin')['ord_prc_stk'].sum().reset_index(name=f'{month}_cost')
        results['costs'].append(cost_df)

        # 利润
        profit_df = pd.merge(price_df[['pin', f'{month}_price']],
                             cost_df[['pin', f'{month}_cost']],
                             on='pin', how='outer').fillna(0)

        # 计算利润
        profit_df[f'{month}_profit'] = profit_df[f'{month}_price'] - profit_df[f'{month}_cost']

        # 只保留pin和利润列
        profit_df = profit_df[['pin', f'{month}_profit']]
        results['profits'].append(profit_df)

    # 合并所有月份的数据
    def merge_all(data_list):
        merged = data_list[0]
        for df in data_list[1:]:
            merged = pd.merge(merged, df, on='pin', how='outer')
        return merged.fillna(0)

    # 合并统计数据
    final_df = merge_all([merge_all(results['counts']),
                          merge_all(results['payments']),
                          merge_all(results['prices']),
                          merge_all(results['costs']),
                          merge_all(results['profits'])])

    # 计算总计
    count_cols = [col for col in final_df.columns if '_count' in col]
    payment_cols = [col for col in final_df.columns if '_payment' in col]
    price_cols = [col for col in final_df.columns if '_price' in col]
    cost_cols = [col for col in final_df.columns if '_cost' in col]
    profit_cols = [col for col in final_df.columns if '_profit' in col]

    final_df['total_count'] = final_df[count_cols].sum(axis=1)
    final_df['total_payment'] = final_df[payment_cols].sum(axis=1)
    final_df['total_price'] = final_df[price_cols].sum(axis=1)
    final_df['total_cost'] = final_df[cost_cols].sum(axis=1)
    final_df['total_profit'] = final_df[profit_cols].sum(axis=1)

    # 计算上半年的统计指标
    # 1. 上半年订单数
    first_half_count_cols = [f'{month}_count' for month in first_half_months]
    final_df['first_half_count'] = final_df[first_half_count_cols].sum(axis=1)

    # 2. 上半年支付总额
    first_half_payment_cols = [f'{month}_payment' for month in first_half_months]
    final_df['first_half_payment'] = final_df[first_half_payment_cols].sum(axis=1)

    # 3. 上半年订单折扣后总额
    first_half_price_cols = [f'{month}_price' for month in first_half_months]
    final_df['first_half_price'] = final_df[first_half_price_cols].sum(axis=1)

    # 4. 上半年成本总额
    first_half_cost_cols = [f'{month}_cost' for month in first_half_months]
    final_df['first_half_cost'] = final_df[first_half_cost_cols].sum(axis=1)

    # 5. 上半年毛利
    first_half_profit_cols = [f'{month}_profit' for month in first_half_months]
    final_df['first_half_profit'] = final_df[first_half_profit_cols].sum(axis=1)

    # 计算下半年的统计指标
    # 1. 下半年订单数
    second_half_count_cols = [f'{month}_count' for month in second_half_months]
    final_df['second_half_count'] = final_df[second_half_count_cols].sum(axis=1)

    # 2. 下半年支付总额
    second_half_payment_cols = [f'{month}_payment' for month in second_half_months]
    final_df['second_half_payment'] = final_df[second_half_payment_cols].sum(axis=1)

    # 3. 下半年订单总额
    second_half_price_cols = [f'{month}_price' for month in second_half_months]
    final_df['second_half_price'] = final_df[second_half_price_cols].sum(axis=1)

    # 4. 下半年成本总额
    second_half_cost_cols = [f'{month}_cost' for month in second_half_months]
    final_df['second_half_cost'] = final_df[second_half_cost_cols].sum(axis=1)

    # 5. 下半年毛利
    second_half_profit_cols = [f'{month}_profit' for month in second_half_months]
    final_df['second_half_profit'] = final_df[second_half_profit_cols].sum(axis=1)

    # 重新排列列，使结果更清晰
    # 首先确定所有列的顺序
    pin_col = ['pin']

    # 月度详细数据
    monthly_cols = []
    for month in month_id:
        monthly_cols.extend([f'{month}_count', f'{month}_payment', f'{month}_price', f'{month}_cost', f'{month}_profit'])

    # 上半年汇总
    first_half_summary = ['first_half_count', 'first_half_payment', 'first_half_price', 'first_half_cost', 'first_half_profit']

    # 下半年汇总
    second_half_summary = ['second_half_count', 'second_half_payment', 'second_half_price', 'second_half_cost', 'second_half_profit']

    # 全年汇总
    yearly_summary = ['total_count', 'total_payment', 'total_price', 'total_cost', 'total_profit']

    # 重新组织列顺序
    final_df = final_df[pin_col + monthly_cols + first_half_summary + second_half_summary + yearly_summary]

    # 删除上半年交易小于10单的
    # final_df = final_df[final_df['first_half_count'] >= 10]

    # 删除运营成本小于0的异常值
    final_df = final_df[final_df['first_half_cost'] >= 0]
    final_df = final_df[final_df['second_half_cost'] >= 0]
    final_df = final_df[final_df['first_half_profit'] >= 0]
    final_df = final_df[final_df['second_half_profit'] >= 0]

    # 显示结果
    print("\n用户月度统计汇总：")
    print(f"共统计了 {len(final_df)} 个用户")

    # 保存结果
    output_file = '用户月度统计汇总.csv'
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n统计结果已保存到: {output_file}")

    return final_df



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
        "上海市": 6, "北京市": 6, "台湾省": 6, "港澳": 6,

        # level 5
        "浙江省": 5, "江苏省": 5, "天津市": 5,

        # level 4
        "福建省": 4, "广东省": 4, "辽宁省": 4, "山东省": 4, "内蒙古自治区": 4,

        # level 3
        "重庆市": 3, "湖北省": 3, "湖南省": 3, "安徽省": 3, "河北省": 3,
        "江西省": 3, "海南省": 3,

        # level 2
        "四川省": 2, "陕西省": 2, "吉林省": 2, "宁夏回族自治区": 2, "黑龙江省": 2, "河南省": 2, "山西省": 2,

        # level 1
        "广西壮族自治区": 1, "新疆维吾尔自治区": 1, "青海省": 1, "云南省": 1, "贵州省": 1, "西藏自治区": 1, "甘肃省": 1,
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

        df.to_csv(f'data/Empirical/data_Q{Q}.csv', index=False, encoding="utf-8-sig")


if __name__ == '__main__':
    # 调用函数
    # result_df = extract_orders_id(threshold=10)
    extract_order_info()