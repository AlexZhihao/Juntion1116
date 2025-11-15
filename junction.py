"""
Fortum Junction 2025 – Baseline ML Pipeline  (兼容老版 pandas & lightgbm)

- 读取官方训练数据 (Excel)
- 构建特征（时间 + 滞后 + 价格）
- 训练一个全局 LightGBM 模型（所有 group 共用）
- 生成 48 小时小时级预测 & 12 个月月度预测
- 按官方模板导出 CSV（分号分隔，逗号为小数点）

需要：
    pip install pandas numpy scikit-learn lightgbm
"""

from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

# ======================== 配置 ========================

TRAINING_XLSX = "20251111_JUNCTION_training.xlsx"
HOURLY_TEMPLATE_CSV = "20251111_JUNCTION_example_hourly.csv"
MONTHLY_TEMPLATE_CSV = "20251111_JUNCTION_example_monthly.csv"

OUTPUT_HOURLY_CSV = "my_hourly_forecast.csv"
OUTPUT_MONTHLY_CSV = "my_monthly_forecast.csv"

# 9 月做验证集
VALID_START = pd.Timestamp("2024-09-01 00:00:00")


# ======================== 工具函数 ========================

def to_naive_datetime(s):
    """把任何 Series 转成不带时区的 datetime64[ns]."""
    s = pd.to_datetime(s)
    try:
        s = s.dt.tz_localize(None)
    except Exception:
        pass
    return s


def remove_categories(df):
    """把 DataFrame 里所有 Categorical 列转换成普通类型。"""
    df = df.copy()
    for c in df.columns:
        if str(df[c].dtype) == "category":
            df[c] = df[c].astype(object)
    return df


def add_time_features(df, time_col="measured_at"):
    """从时间戳生成常用时间特征（不带时区）。"""
    df = df.copy()
    df[time_col] = to_naive_datetime(df[time_col])

    df["hour"] = df[time_col].dt.hour
    df["dayofweek"] = df[time_col].dt.dayofweek
    df["month"] = df[time_col].dt.month
    df["year"] = df[time_col].dt.year
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    return df


def add_lag_features(df, group_col="group_id", target_col="consumption",
                     lags=(1, 24, 168)):
    """为每个 group 添加滞后特征 + 24h 滚动统计。"""
    df = remove_categories(df)
    df = df.sort_values([group_col, "measured_at"]).copy()

    for lag in lags:
        df["lag_%d" % lag] = df.groupby(group_col)[target_col].shift(lag)

    df["rolling_24h_mean"] = (
        df.groupby(group_col)[target_col]
        .shift(1)
        .rolling(window=24, min_periods=12)
        .mean()
    )
    df["rolling_24h_std"] = (
        df.groupby(group_col)[target_col]
        .shift(1)
        .rolling(window=24, min_periods=12)
        .std()
    )

    return df


def add_price_lags(df, price_col="eur_per_mwh", group_col="group_id",
                   lags=(1, 24)):
    """价格滞后特征（价格是全局的，不分 group，这里简单复制）。"""
    df = df.sort_values([group_col, "measured_at"]).copy()
    for lag in lags:
        df["price_lag_%d" % lag] = df.groupby(group_col)[price_col].shift(lag)
    return df


def to_eu_csv(df, path):
    """
    保存为欧洲格式 CSV：
    - 分号分隔
    - 小数点用逗号
    """
    df_str = df.copy()
    for col in df_str.columns:
        if (pd.api.types.is_float_dtype(df_str[col]) or
                pd.api.types.is_integer_dtype(df_str[col])):
            df_str[col] = (
                df_str[col].astype(float)
                .map(lambda x: f"{x:.9f}")
                .str.replace(".", ",")
            )
    df_str.to_csv(path, sep=";", index=False, encoding="utf-8")


# ======================== 1. 读取训练数据 ========================

def load_training_data():
    print("Loading training Excel...")
    xls = pd.ExcelFile(TRAINING_XLSX)

    groups_df = pd.read_excel(xls, sheet_name="groups")
    cons_wide = pd.read_excel(xls, sheet_name="training_consumption")
    prices = pd.read_excel(xls, sheet_name="training_prices")

    cons_wide["measured_at"] = to_naive_datetime(cons_wide["measured_at"])
    prices["measured_at"] = to_naive_datetime(prices["measured_at"])

    # 宽表 → 长表
    value_cols = [c for c in cons_wide.columns if c != "measured_at"]
    cons_long = cons_wide.melt(
        id_vars="measured_at",
        value_vars=value_cols,
        var_name="group_id",
        value_name="consumption",
    )
    cons_long["group_id"] = cons_long["group_id"].astype(int)

    df = cons_long.merge(prices, on="measured_at", how="left")
    df = remove_categories(df)

    print("Consumption rows:", len(df))
    print("Time range:", df["measured_at"].min(), "->", df["measured_at"].max())
    print("Groups:", df["group_id"].nunique())

    return df, groups_df, prices


# ======================== 2. 小时级特征 & 模型 ========================

def prepare_hourly_features(df):
    df = add_time_features(df, time_col="measured_at")
    df = add_lag_features(df, group_col="group_id", target_col="consumption")
    df = add_price_lags(df, price_col="eur_per_mwh", group_col="group_id")

    feature_cols = [
        "group_id",
        "hour",
        "dayofweek",
        "month",
        "year",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "lag_1",
        "lag_24",
        "lag_168",
        "rolling_24h_mean",
        "rolling_24h_std",
        "eur_per_mwh",
        "price_lag_1",
        "price_lag_24",
    ]

    df_model = df.dropna(subset=feature_cols + ["consumption"]).copy()
    return df_model, feature_cols


def train_hourly_model(df_model, feature_cols):
    mask_train = df_model["measured_at"] < VALID_START
    mask_valid = ~mask_train

    train = df_model[mask_train]
    valid = df_model[mask_valid]

    X_train = train[feature_cols].copy()
    y_train = train["consumption"].values
    X_valid = valid[feature_cols].copy()
    y_valid = valid["consumption"].values

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    print("Training hourly LightGBM model...")
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train", "valid"],
        num_boost_round=2000,
    )

    y_pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, y_pred)
    print("Validation MAE (hourly): %.4f" % mae)

    return model


# ======================== 3. 48 小时预测 ========================

def forecast_48h_hourly(model, full_df, prices, feature_cols):
    last_time = full_df["measured_at"].max()
    last_time = pd.to_datetime(last_time)
    print("Last observed timestamp:", last_time)

    future_times = [last_time + timedelta(hours=i) for i in range(1, 49)]
    all_groups = sorted(full_df["group_id"].unique())

    hist = full_df[["measured_at", "group_id", "consumption", "eur_per_mwh"]].copy()
    hist["measured_at"] = to_naive_datetime(hist["measured_at"])

    preds_records = []

    for t in future_times:
        print("Forecasting for", t)
        new_rows = []
        for g in all_groups:
            price_val = prices.loc[prices["measured_at"] == t, "eur_per_mwh"]
            price_val = price_val.iloc[0] if not price_val.empty else np.nan
            new_rows.append(
                {
                    "measured_at": t,
                    "group_id": g,
                    "consumption": np.nan,
                    "eur_per_mwh": price_val,
                }
            )

        new_df = pd.DataFrame(new_rows)
        tmp_hist = pd.concat([hist, new_df], ignore_index=True)

        tmp_hist = add_time_features(tmp_hist, time_col="measured_at")
        tmp_hist = add_lag_features(tmp_hist, group_col="group_id", target_col="consumption")
        tmp_hist = add_price_lags(tmp_hist, price_col="eur_per_mwh", group_col="group_id")

        to_pred = tmp_hist[tmp_hist["measured_at"] == t].copy()
        X = to_pred[feature_cols].copy()

        y_hat = model.predict(X)
        to_pred["consumption"] = y_hat
        preds_records.append(to_pred[["measured_at", "group_id", "consumption"]])

        hist = pd.concat(
            [hist, to_pred[["measured_at", "group_id", "consumption", "eur_per_mwh"]]],
            ignore_index=True,
        )

    preds_long = pd.concat(preds_records, ignore_index=True)

    preds_wide = preds_long.pivot_table(
        index="measured_at", columns="group_id", values="consumption"
    ).reset_index()

    preds_wide.columns = ["measured_at"] + [str(c) for c in preds_wide.columns[1:]]
    preds_wide = preds_wide.sort_values("measured_at")

    return preds_wide


# ======================== 4. 月度模型（不用 datetime 做索引） ========================

def add_month_index(df, time_col="measured_at"):
    """给 DataFrame 添加 year / month / ym_index（如 202409）。"""
    df[time_col] = to_naive_datetime(df[time_col])
    df["year"] = df[time_col].dt.year
    df["month"] = df[time_col].dt.month
    df["ym_index"] = df["year"] * 100 + df["month"]
    return df


def prepare_monthly_training(full_df):
    df = full_df.copy()
    df = add_month_index(df, time_col="measured_at")

    monthly = (
        df.groupby(["group_id", "year", "month", "ym_index"])["consumption"]
        .sum()
        .reset_index()
        .rename(columns={"consumption": "month_consumption"})
    )

    monthly = monthly.sort_values(["group_id", "ym_index"])

    monthly["lag_1m"] = monthly.groupby("group_id")["month_consumption"].shift(1)
    monthly["lag_12m"] = monthly.groupby("group_id")["month_consumption"].shift(12)

    monthly = monthly.dropna(subset=["lag_1m", "lag_12m"]).reset_index(drop=True)

    feature_cols = ["group_id", "year", "month", "ym_index", "lag_1m", "lag_12m"]
    return monthly, feature_cols


def train_monthly_model(monthly_df, feature_cols):
    last_idx = monthly_df["ym_index"].max()
    # 大约后 6 个月作为验证
    valid_threshold = last_idx - 6

    mask_train = monthly_df["ym_index"] < valid_threshold
    mask_valid = ~mask_train

    train = monthly_df[mask_train]
    valid = monthly_df[mask_valid]

    X_train = train[feature_cols].copy()
    y_train = train["month_consumption"].values
    X_valid = valid[feature_cols].copy()
    y_valid = valid["month_consumption"].values

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    print("Training monthly LightGBM model...")
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train", "valid"],
        num_boost_round=2000,
    )

    y_pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, y_pred)
    print("Validation MAE (monthly): %.4f" % mae)

    return model


def add_one_month(ym_index):
    """ym_index 如 202409 -> 返回下一月的 (ym_index, year, month)."""
    year = int(ym_index // 100)
    month = int(ym_index % 100)
    month += 1
    if month > 12:
        month = 1
        year += 1
    return year * 100 + month, year, month


def forecast_12m_monthly(model, full_df, monthly_feature_cols):
    df = full_df.copy()
    df = add_month_index(df, time_col="measured_at")

    monthly = (
        df.groupby(["group_id", "year", "month", "ym_index"])["consumption"]
        .sum()
        .reset_index()
        .rename(columns={"consumption": "month_consumption"})
    )

    last_idx = monthly["ym_index"].max()
    print("Last historical ym_index:", last_idx)

    all_groups = sorted(monthly["group_id"].unique())

    monthly = monthly.sort_values(["group_id", "ym_index"])

    preds = []
    current_idx = last_idx

    for _ in range(12):
        next_idx, next_year, next_month = add_one_month(current_idx)
        print("Forecasting month:", next_year, next_month)

        new_rows = []
        for g in all_groups:
            new_rows.append(
                {
                    "group_id": g,
                    "year": next_year,
                    "month": next_month,
                    "ym_index": next_idx,
                }
            )

        new_df = pd.DataFrame(new_rows)
        monthly = pd.concat([monthly, new_df], ignore_index=True)
        monthly = monthly.sort_values(["group_id", "ym_index"])

        monthly["lag_1m"] = monthly.groupby("group_id")["month_consumption"].shift(1)
        monthly["lag_12m"] = monthly.groupby("group_id")["month_consumption"].shift(12)

        cur = monthly[
            (monthly["ym_index"] == next_idx)
        ].copy()

        cur["lag_1m"] = cur["lag_1m"].fillna(monthly["month_consumption"].median())
        cur["lag_12m"] = cur["lag_12m"].fillna(cur["lag_1m"])

        X = cur[monthly_feature_cols].copy()
        y_hat = model.predict(X)

        cur["month_consumption"] = y_hat
        preds.append(cur[["group_id", "year", "month", "ym_index", "month_consumption"]])

        monthly.loc[monthly["ym_index"] == next_idx, "month_consumption"] = y_hat
        current_idx = next_idx

    preds_long = pd.concat(preds, ignore_index=True)

    # 构造 measured_at = 当月 1 日 00:00
    preds_long["measured_at"] = preds_long.apply(
        lambda r: pd.Timestamp(int(r["year"]), int(r["month"]), 1),
        axis=1,
    )

    preds_wide = preds_long.pivot_table(
        index="measured_at", columns="group_id", values="month_consumption"
    ).reset_index()

    preds_wide.columns = ["measured_at"] + [str(c) for c in preds_wide.columns[1:]]
    preds_wide = preds_wide.sort_values("measured_at")

    return preds_wide


# ======================== 5. 对齐模板并导出 ========================

def align_with_template(preds_wide, template_path):
    template = pd.read_csv(template_path, sep=";")
    cols = list(template.columns)

    out = preds_wide.copy()
    out["measured_at"] = to_naive_datetime(out["measured_at"])
    out["measured_at"] = out["measured_at"].dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    for c in cols:
        if c not in out.columns:
            out[c] = 0.0

    out = out[cols]
    return out


# ======================== main ========================

def main():
    full_df, groups_df, prices = load_training_data()

    # 小时级模型
    hourly_df, hourly_feature_cols = prepare_hourly_features(full_df)
    hourly_model = train_hourly_model(hourly_df, hourly_feature_cols)

    hourly_preds_wide = forecast_48h_hourly(
        hourly_model, full_df, prices, hourly_feature_cols
    )
    hourly_submission = align_with_template(hourly_preds_wide, HOURLY_TEMPLATE_CSV)
    print("Saving hourly forecast to:", OUTPUT_HOURLY_CSV)
    to_eu_csv(hourly_submission, OUTPUT_HOURLY_CSV)

    # 月度模型
    monthly_df, monthly_feature_cols = prepare_monthly_training(full_df)
    monthly_model = train_monthly_model(monthly_df, monthly_feature_cols)

    monthly_preds_wide = forecast_12m_monthly(
        monthly_model, full_df, monthly_feature_cols
    )
    monthly_submission = align_with_template(
        monthly_preds_wide, MONTHLY_TEMPLATE_CSV
    )
    print("Saving monthly forecast to:", OUTPUT_MONTHLY_CSV)
    to_eu_csv(monthly_submission, OUTPUT_MONTHLY_CSV)

    print("Done.")


if __name__ == "__main__":
    main()
