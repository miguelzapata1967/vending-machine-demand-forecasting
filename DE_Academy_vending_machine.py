"""
DE Academy — Vending Machine Demand Forecasting Project
Full pipeline: data cleaning -> EDA -> feature engineering -> baseline
forecasts -> statistical models (ARIMA/SARIMAX) -> ML models (RF/XGBoost/LightGBM)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ============================================================
# Setup — output path + reusable helpers
# ============================================================
output_dir = r'C:\Users\mexar\OneDrive\DE_Academy\Python\Vending Machine Project'
os.makedirs(output_dir, exist_ok=True)


def save_plot(filename):
    """Save the current matplotlib figure into the project output folder."""
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")


results = []  # collects {model, MAE, RMSE, MAPE} dicts across every forecasting model


def evaluate_forecast(actual, predicted, model_name):
    actual = pd.Series(actual).reset_index(drop=True)
    predicted = pd.Series(predicted).reset_index(drop=True)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual.replace(0, np.nan))) * 100
    print(f"{model_name}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
    return {'model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


# ============================================================
# Load data
# ============================================================
df1 = pd.read_csv('C:\\Users\\mexar\\OneDrive\\DE_Academy\\CSVs\\Inventory_Turnover.csv')
df2 = pd.read_csv('C:\\Users\\mexar\\OneDrive\\DE_Academy\\CSVs\\Restock_data.csv')

# Verifying data info is clean - nonnull
print(df2.info())
print(df1.info())

# verifying for duplicates
print(df1.duplicated().sum())
print(df2.duplicated().sum())

# --- Clean the sku column: strips hidden whitespace/newlines that fragment
# --- identical SKUs into "different" groups during groupby operations
df1['sku'] = df1['sku'].str.strip()

# verifying for convertion errors
df1['dispense_date'] = pd.to_datetime(df1['dispense_date'])
df2['restock_date'] = pd.to_datetime(df2['restock_date'])

# ============================================================
# Feature Engineering — calendar attributes for time series analysis
# ============================================================
df1['day_of_week'] = df1['dispense_date'].dt.day_name()
df1['day_of_week_num'] = df1['dispense_date'].dt.dayofweek
df1['day_of_month'] = df1['dispense_date'].dt.day
df1['month'] = df1['dispense_date'].dt.month
df1['quarter'] = df1['dispense_date'].dt.quarter
df1['is_weekend'] = df1['dispense_date'].dt.dayofweek >= 5

print(df1[['dispense_date', 'day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend']].head(10))

# Weekend vs Weekday split
weekend_avg = df1.groupby('is_weekend')['qty_dispensed'].mean()
print(weekend_avg)

plt.figure(figsize=(6, 5))
df1.groupby('is_weekend')['qty_dispensed'].mean().plot(kind='bar', color=['steelblue', 'crimson'])
plt.title('Average Qty Dispensed: Weekday vs. Weekend')
plt.xlabel('Is Weekend')
plt.ylabel('Average Qty Dispensed')
plt.xticks([0, 1], ['Weekday', 'Weekend'], rotation=0)
plt.tight_layout()
save_plot('feature_weekday_vs_weekend_avg_dispensed.png')
plt.show()

# verifying for 0 or negatives values in df1 and df2
print(df1[(df1['qty_dispensed'] <= 0) | (df1['package_qty'] <= 0)].shape[0])
print(df2[df2['total'] <= 0].shape[0])

# matching columns for merging
df1_renamed = df1.rename(columns={'dispense_date': 'proj1_date'})
df2_renamed = df2.rename(columns={'restock_date': 'proj1_date'})

# merging tables on device_id column
# not using inner join because we want to keep all records from df1 and only matching records from df2
merged_df = pd.merge(df1_renamed, df2_renamed, on=['device_id'], how='left')

print(merged_df)

# Calculate summary statistics for the 'package_qty', 'qty_dispensed', and 'total' columns
summary = merged_df[['package_qty', 'qty_dispensed', 'total']].agg(['mean', 'median', 'std', 'min', 'max'])
print(summary)

# Calculating Cardinalities for sku, device_id, global_order_id
print(f"Unique SKUs: {merged_df['sku'].nunique()}")
print(f"Unique Device IDs: {merged_df['device_id'].nunique()}")
print(f"Unique Global Order IDs: {merged_df['global_order_id'].nunique()}")

# comparing before and after merging
print("SKUs before merge (df1):", df1['sku'].nunique())
print("SKUs after merge (merged_df):", merged_df['sku'].nunique())
print("Device IDs before merge (df1):", df1['device_id'].nunique())
print("Device IDs after merge (merged_df):", merged_df['device_id'].nunique())
print("Global Order IDs before merge (df2):", df2['global_order_id'].nunique())
print("Global Order IDs after merge (merged_df):", merged_df['global_order_id'].nunique())

# ============================================================
# IQR — ranges and outlier detection (package_qty, qty_dispensed, total)
# ============================================================
for col in ['package_qty', 'qty_dispensed', 'total']:
    Q1 = merged_df[col].quantile(0.25)
    Q3 = merged_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_count = merged_df[(merged_df[col] < lower) | (merged_df[col] > upper)].shape[0]
    print(f"{col}: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}, outliers={outlier_count}")

# ============================================================
# Overall Trends & Seasonality — daily and weekly qty_dispensed
# ============================================================
daily1 = df1.groupby('dispense_date')['qty_dispensed'].sum().reset_index()
print(daily1.head())
daily2 = df2.groupby('restock_date')['total'].sum().reset_index()
print(daily2.head())

weekly1 = df1.set_index('dispense_date')['qty_dispensed'].resample('W').sum().reset_index()
print(weekly1.head())
weekly2 = df2.set_index('restock_date')['total'].resample('W').sum().reset_index()
print(weekly2.head())

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].plot(daily1['dispense_date'], daily1['qty_dispensed'], color='steelblue')
axes[0].set_title('Daily Total Qty Dispensed')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Qty Dispensed')

axes[1].plot(weekly1['dispense_date'], weekly1['qty_dispensed'], color='darkorange')
axes[1].set_title('Weekly Total Qty Dispensed')
axes[1].set_xlabel('Week')
axes[1].set_ylabel('Qty Dispensed')
plt.tight_layout()
save_plot('trend_daily_weekly_qty_dispensed.png')
plt.show()

# ============================================================
# Demand Variability by SKU — daily CV and High/Low variance buckets
# ============================================================
daily_by_sku = df1.groupby(['sku', 'dispense_date'])['qty_dispensed'].sum().reset_index()

sku_stats = daily_by_sku.groupby('sku')['qty_dispensed'].agg(['mean', 'std']).reset_index()
sku_stats['cv'] = sku_stats['std'] / sku_stats['mean']

print(sku_stats.sort_values('cv', ascending=False))

median_cv = sku_stats['cv'].median()
sku_stats['variance_category'] = sku_stats['cv'].apply(
    lambda x: 'High Variance' if x >= median_cv else 'Low Variance'
)
print(f"Median CV threshold: {median_cv:.3f}")
print(sku_stats['variance_category'].value_counts())

plt.figure(figsize=(6, 5))
sku_stats['variance_category'].value_counts().plot(kind='bar', color=['crimson', 'steelblue'])
plt.title('SKU Count by Demand Variability Category')
plt.xlabel('Category')
plt.ylabel('Number of SKUs')
plt.xticks(rotation=0)
plt.tight_layout()
save_plot('sku_demand_variability_category_counts.png')
plt.show()

# ============================================================
# Product Turnover — rank SKUs by total qty_dispensed & transactions
# ============================================================
sku_turnover = df1.groupby('sku').agg(
    total_qty_dispensed=('qty_dispensed', 'sum'),
    total_transactions=('sku', 'count')
).reset_index()

sku_turnover = sku_turnover.sort_values('total_qty_dispensed', ascending=False)
print(sku_turnover)

top_n = 5
top_skus = sku_turnover.head(top_n)['sku'].tolist()
print(top_skus)

plt.figure(figsize=(10, 5))
top_10 = sku_turnover.head(10)
plt.bar(top_10['sku'], top_10['total_qty_dispensed'], color='#800020')
plt.title('Top 10 SKUs by Total Quantity Dispensed')
plt.xlabel('SKU')
plt.ylabel('Total Qty Dispensed')
plt.xticks(rotation=45)
plt.tight_layout()
save_plot('product_turnover_top10_skus.png')
plt.show()

# ============================================================
# Seasonality per top SKU — monthly & day-of-week patterns
# ============================================================
for rank, sku in enumerate(top_skus, start=1):
    subset = df1[df1['sku'] == sku]

    monthly_pattern = subset.groupby('month')['qty_dispensed'].sum()
    dow_pattern = subset.groupby('day_of_week')['qty_dispensed'].sum()

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_pattern = dow_pattern.reindex(day_order)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].bar(monthly_pattern.index, monthly_pattern.values, color='steelblue')
    axes[0].set_title(f'Top {rank} SKU — Monthly Total Qty Dispensed')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Qty Dispensed')

    axes[1].bar(dow_pattern.index, dow_pattern.values, color='darkorange')
    axes[1].set_title(f'Top {rank} SKU — Day-of-Week Total Qty Dispensed')
    axes[1].set_xlabel('Day of Week')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    save_plot(f'seasonality_top{rank}_sku_monthly_dow.png')
    plt.show()

# Formal seasonal decomposition (weekly period) for the single top SKU
top_sku_daily = daily_by_sku[daily_by_sku['sku'] == top_skus[0]].set_index('dispense_date')['qty_dispensed']
top_sku_daily = top_sku_daily.asfreq('D', fill_value=0)

if len(top_sku_daily) >= 60:
    decomposition = seasonal_decompose(top_sku_daily, model='additive', period=7)
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(f'Seasonal Decomposition — Top SKU (Additive, weekly period)', y=1.02)
    plt.tight_layout()
    save_plot('seasonal_decomposition_top_sku.png')
    plt.show()

# ============================================================
# Device Utilization — dispensing volume vs. restock frequency
# ============================================================
device_dispense = df1.groupby('device_id').agg(
    total_qty_dispensed=('qty_dispensed', 'sum'),
    total_dispense_transactions=('device_id', 'count')
).reset_index()

print(device_dispense.head())

device_restock = df2.groupby('device_id').agg(
    total_restock_events=('device_id', 'count'),
    total_restock_qty=('total', 'sum')
).reset_index()

print(device_restock.head())

device_summary = pd.merge(device_dispense, device_restock, on='device_id', how='outer')
device_summary = device_summary.fillna(0)

print(device_summary.head())

qty_median = device_summary['total_qty_dispensed'].median()
restock_median = device_summary['total_restock_events'].median()


def classify_device(row):
    if row['total_qty_dispensed'] >= qty_median and row['total_restock_events'] >= restock_median:
        return 'Overused (high volume + frequent restocks)'
    elif row['total_qty_dispensed'] >= qty_median and row['total_restock_events'] < restock_median:
        return 'High volume, low restocks (stockout risk)'
    elif row['total_qty_dispensed'] < qty_median and row['total_restock_events'] >= restock_median:
        return 'Low volume, frequent restocks (inefficient)'
    else:
        return 'Underused (low volume + low restocks)'


device_summary['device_category'] = device_summary.apply(classify_device, axis=1)

print(device_summary['device_category'].value_counts())

stockout_risk = device_summary[device_summary['device_category'] == 'High volume, low restocks (stockout risk)']
print(stockout_risk.sort_values('total_qty_dispensed', ascending=False))

underused = device_summary[device_summary['device_category'] == 'Underused (low volume + low restocks)']
print(underused.sort_values('total_qty_dispensed'))

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=device_summary,
    x='total_restock_events',
    y='total_qty_dispensed',
    hue='device_category',
    palette={
        'Overused (high volume + frequent restocks)': '#800020',
        'High volume, low restocks (stockout risk)': 'darkorange',
        'Low volume, frequent restocks (inefficient)': 'steelblue',
        'Underused (low volume + low restocks)': 'gray'
    },
    s=80
)
plt.axvline(restock_median, color='black', linestyle='--', alpha=0.4)
plt.axhline(qty_median, color='black', linestyle='--', alpha=0.4)
plt.title('Device Utilization: Dispensing Volume vs. Restock Frequency')
plt.xlabel('Total Restock Events')
plt.ylabel('Total Qty Dispensed')
plt.tight_layout()
save_plot('device_utilization_scatter.png')
plt.show()

print(df2[['device_id', 'total']].head())

# ============================================================
# Correlation Analysis — Pearson/Spearman + lag correlation
# ============================================================
daily_dispense = df1.groupby('dispense_date')['qty_dispensed'].sum().reset_index()
daily_dispense.columns = ['date', 'qty_dispensed']

daily_restock = df2.groupby('restock_date').agg(
    restock_frequency=('device_id', 'count'),
    restock_expenditure=('total', 'sum')
).reset_index()
daily_restock.columns = ['date', 'restock_frequency', 'restock_expenditure']

print(daily_dispense.head())
print(daily_restock.head())

daily_combined = pd.merge(daily_dispense, daily_restock, on='date', how='outer').fillna(0)
daily_combined = daily_combined.sort_values('date').reset_index(drop=True)

print(daily_combined.head())
print(daily_combined.shape)

pairs = [
    ('qty_dispensed', 'restock_frequency'),
    ('qty_dispensed', 'restock_expenditure'),
    ('restock_frequency', 'restock_expenditure')
]

for col1, col2 in pairs:
    pearson_r, pearson_p = pearsonr(daily_combined[col1], daily_combined[col2])
    spearman_r, spearman_p = spearmanr(daily_combined[col1], daily_combined[col2])
    print(f"{col1} vs {col2}:")
    print(f"  Pearson  r = {pearson_r:.3f}, p = {pearson_p:.4f}")
    print(f"  Spearman r = {spearman_r:.3f}, p = {spearman_p:.4f}\n")

corr_matrix = daily_combined[['qty_dispensed', 'restock_frequency', 'restock_expenditure']].corr(method='pearson')
print(corr_matrix)

plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Pearson Correlation Matrix (Daily)')
plt.tight_layout()
save_plot('correlation_heatmap_daily.png')
plt.show()


def lag_correlation(series1, series2, max_lag=14):
    lag_res = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            s1 = series1[-lag:]
            s2 = series2[: len(series2) + lag]
        elif lag > 0:
            s1 = series1[: len(series1) - lag]
            s2 = series2[lag:]
        else:
            s1, s2 = series1, series2

        if len(s1) > 1 and len(s2) > 1:
            r, _ = pearsonr(s1, s2)
            lag_res.append({'lag': lag, 'pearson_r': r})
    return pd.DataFrame(lag_res)


lag_results = lag_correlation(
    daily_combined['restock_frequency'].values,
    daily_combined['qty_dispensed'].values,
    max_lag=14
)

print(lag_results)

plt.figure(figsize=(10, 5))
plt.plot(lag_results['lag'], lag_results['pearson_r'], marker='o', color='#800020')
plt.axvline(0, color='black', linestyle='--', alpha=0.5)
plt.axhline(0, color='gray', linestyle='-', alpha=0.3)
plt.title('Lag Correlation: Restock Frequency vs. Qty Dispensed')
plt.xlabel('Lag (days) — negative = restock follows demand, positive = restock precedes demand')
plt.ylabel('Pearson r')
plt.tight_layout()
save_plot('correlation_lag_restock_vs_dispense.png')
plt.show()

best_lag = lag_results.loc[lag_results['pearson_r'].abs().idxmax()]
print(f"Strongest correlation at lag = {best_lag['lag']} days, r = {best_lag['pearson_r']:.3f}")

# ============================================================
# Lag Features — historical demand at t-1, t-7, t-14, t-28 days
# ============================================================
daily_qty = daily1.set_index('dispense_date').asfreq('D', fill_value=0).reset_index()

print(daily_qty.head())

daily_qty['qty_lag_1'] = daily_qty['qty_dispensed'].shift(1)
daily_qty['qty_lag_7'] = daily_qty['qty_dispensed'].shift(7)
daily_qty['qty_lag_14'] = daily_qty['qty_dispensed'].shift(14)
daily_qty['qty_lag_28'] = daily_qty['qty_dispensed'].shift(28)

print(daily_qty.head(30))

# Keeping missing NaN rows on record for an auditable review, then
# dropping them before any correlation/modeling work downstream.
lag_cols = ['qty_lag_1', 'qty_lag_7', 'qty_lag_14', 'qty_lag_28']
daily_qty['has_missing_lag'] = daily_qty[lag_cols].isna().any(axis=1)


def missing_lag_label(row):
    missing = [col for col in lag_cols if pd.isna(row[col])]
    return ', '.join(missing) if missing else 'None'


daily_qty['missing_lag_columns'] = daily_qty.apply(missing_lag_label, axis=1)

print(daily_qty[['dispense_date', 'qty_dispensed'] + lag_cols + ['missing_lag_columns']].head(30))

pending_review = daily_qty[daily_qty['has_missing_lag']].copy()

category_summary = pending_review.groupby('missing_lag_columns').agg(
    row_count=('dispense_date', 'count'),
    total_qty_dispensed=('qty_dispensed', 'sum'),
    earliest_date=('dispense_date', 'min'),
    latest_date=('dispense_date', 'max')
).reset_index().sort_values('row_count', ascending=False)

print("Pending decision — rows with incomplete lag history:")
print(category_summary)

pending_review.to_csv(os.path.join(output_dir, 'lag_features_pending_review.csv'), index=False)
category_summary.to_csv(os.path.join(output_dir, 'lag_features_missing_summary.csv'), index=False)

daily_qty_clean = daily_qty[~daily_qty['has_missing_lag']].drop(
    columns=['has_missing_lag', 'missing_lag_columns']
).reset_index(drop=True)

print(f"Rows before drop: {daily_qty.shape[0]}")
print(f"Rows after drop:  {daily_qty_clean.shape[0]}")
print(f"Rows excluded:    {daily_qty.shape[0] - daily_qty_clean.shape[0]}")

total_all = daily_qty['qty_dispensed'].sum()
total_dropped = pending_review['qty_dispensed'].sum()
pct_dropped = (total_dropped / total_all) * 100
print(f"Total volume in dropped rows: {total_dropped} ({pct_dropped:.2f}% of all dispensed volume)")

# ============================================================
# Rolling Window Statistics — 7-day and 30-day rolling mean/std
# per SKU and per device
# ============================================================
# Per SKU
daily_by_sku_full = (
    daily_by_sku.set_index('dispense_date')
    .groupby('sku')['qty_dispensed']
    .resample('D')
    .sum()
    .fillna(0)
    .reset_index()
)
daily_by_sku_full = daily_by_sku_full.sort_values(['sku', 'dispense_date'])

daily_by_sku_full['rolling_mean_7d'] = (
    daily_by_sku_full.groupby('sku')['qty_dispensed']
    .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
)
daily_by_sku_full['rolling_std_7d'] = (
    daily_by_sku_full.groupby('sku')['qty_dispensed']
    .transform(lambda x: x.rolling(window=7, min_periods=1).std())
)
daily_by_sku_full['rolling_mean_30d'] = (
    daily_by_sku_full.groupby('sku')['qty_dispensed']
    .transform(lambda x: x.rolling(window=30, min_periods=1).mean())
)
daily_by_sku_full['rolling_std_30d'] = (
    daily_by_sku_full.groupby('sku')['qty_dispensed']
    .transform(lambda x: x.rolling(window=30, min_periods=1).std())
)

print(daily_by_sku_full.head(20))

# Per device
daily_dispense_by_device = df1.groupby(['device_id', 'dispense_date'])['qty_dispensed'].sum().reset_index()

daily_by_device_full = (
    daily_dispense_by_device.set_index('dispense_date')
    .groupby('device_id')['qty_dispensed']
    .resample('D')
    .sum()
    .fillna(0)
    .reset_index()
)
daily_by_device_full = daily_by_device_full.sort_values(['device_id', 'dispense_date'])

daily_by_device_full['rolling_mean_7d'] = (
    daily_by_device_full.groupby('device_id')['qty_dispensed']
    .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
)
daily_by_device_full['rolling_std_7d'] = (
    daily_by_device_full.groupby('device_id')['qty_dispensed']
    .transform(lambda x: x.rolling(window=7, min_periods=1).std())
)
daily_by_device_full['rolling_mean_30d'] = (
    daily_by_device_full.groupby('device_id')['qty_dispensed']
    .transform(lambda x: x.rolling(window=30, min_periods=1).mean())
)
daily_by_device_full['rolling_std_30d'] = (
    daily_by_device_full.groupby('device_id')['qty_dispensed']
    .transform(lambda x: x.rolling(window=30, min_periods=1).std())
)

print(daily_by_device_full.head(20))

top_sku_id = top_skus[0]
subset = daily_by_sku_full[daily_by_sku_full['sku'] == top_sku_id]

plt.figure(figsize=(14, 5))
plt.plot(subset['dispense_date'], subset['qty_dispensed'], alpha=0.3, label='Daily (raw)')
plt.plot(subset['dispense_date'], subset['rolling_mean_7d'], label='7-day rolling avg', color='steelblue')
plt.plot(subset['dispense_date'], subset['rolling_mean_30d'], label='30-day rolling avg', color='#800020')
plt.title('Top SKU — Rolling Averages of Daily Qty Dispensed')
plt.xlabel('Date')
plt.ylabel('Qty Dispensed')
plt.legend()
plt.tight_layout()
save_plot('rolling_avg_top_sku.png')
plt.show()

sku_avg_rolling_std = daily_by_sku_full.groupby('sku')['rolling_std_7d'].mean().reset_index()
sku_avg_rolling_std.columns = ['sku', 'avg_rolling_std_7d']
print(sku_avg_rolling_std.sort_values('avg_rolling_std_7d', ascending=False).head(10))

# ============================================================
# Restock Signal Features — days since last restock,
# cumulative restock cost over 7 and 30 days (per device)
# ============================================================
daily_restock_by_device = df2.groupby(['device_id', 'restock_date']).agg(
    restock_events=('device_id', 'count'),
    restock_cost=('total', 'sum')
).reset_index()

daily_restock_full = (
    daily_restock_by_device.set_index('restock_date')
    .groupby('device_id')[['restock_events', 'restock_cost']]
    .resample('D')
    .sum()
    .reset_index()
)
daily_restock_full = daily_restock_full.sort_values(['device_id', 'restock_date'])

daily_restock_full['restock_date_if_event'] = daily_restock_full['restock_date'].where(
    daily_restock_full['restock_events'] > 0
)
daily_restock_full['last_restock_date'] = (
    daily_restock_full.groupby('device_id')['restock_date_if_event'].ffill()
)
daily_restock_full['days_since_last_restock'] = (
    daily_restock_full['restock_date'] - daily_restock_full['last_restock_date']
).dt.days

no_prior_restock = daily_restock_full['days_since_last_restock'].isna().sum()
print(f"Rows with no prior restock on record: {no_prior_restock}")

daily_restock_full['cum_restock_cost_7d'] = (
    daily_restock_full.groupby('device_id')['restock_cost']
    .transform(lambda x: x.rolling(window=7, min_periods=1).sum())
)
daily_restock_full['cum_restock_cost_30d'] = (
    daily_restock_full.groupby('device_id')['restock_cost']
    .transform(lambda x: x.rolling(window=30, min_periods=1).sum())
)

daily_restock_full = daily_restock_full.drop(columns=['restock_date_if_event'])

print(daily_restock_full[['device_id', 'restock_date', 'restock_cost',
                           'days_since_last_restock', 'cum_restock_cost_7d', 'cum_restock_cost_30d']].head(20))

sample_device = daily_restock_full['device_id'].iloc[0]
subset = daily_restock_full[daily_restock_full['device_id'] == sample_device]

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].plot(subset['restock_date'], subset['days_since_last_restock'], color='#800020')
axes[0].set_title(f'Device {sample_device} — Days Since Last Restock')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Days Since Last Restock')

axes[1].plot(subset['restock_date'], subset['cum_restock_cost_7d'], label='7-day cumulative cost', color='steelblue')
axes[1].plot(subset['restock_date'], subset['cum_restock_cost_30d'], label='30-day cumulative cost', color='darkorange')
axes[1].set_title(f'Device {sample_device} — Cumulative Restock Cost')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Cumulative Cost')
axes[1].legend()

plt.tight_layout()
save_plot('restock_signal_features_sample_device.png')
plt.show()

# ============================================================
# PREDICTIVE MODELING — FORECASTING
# ============================================================

# ---- Train/test split (chronological, last 30 days held out) ----
ts = daily1.set_index('dispense_date')['qty_dispensed'].asfreq('D', fill_value=0)

test_size = 30
train, test = ts.iloc[:-test_size], ts.iloc[-test_size:]

print(f"Train: {train.index.min()} to {train.index.max()} ({len(train)} days)")
print(f"Test:  {test.index.min()} to {test.index.max()} ({len(test)} days)")

# ---- Baseline 1: Naive (tomorrow = today) ----
naive_forecast = pd.Series(train.iloc[-1], index=test.index)
results.append(evaluate_forecast(test, naive_forecast, 'Naive (last value)'))

# ---- Baseline 2: Seasonal Naive (7-day lag) ----
seasonal_naive_values = train.iloc[-7:].values
seasonal_naive_values = np.tile(seasonal_naive_values, int(np.ceil(len(test) / 7)))[:len(test)]
seasonal_naive_forecast = pd.Series(seasonal_naive_values, index=test.index)
results.append(evaluate_forecast(test, seasonal_naive_forecast, 'Seasonal Naive (7-day)'))

# ---- Baseline 3: Holt-Winters Exponential Smoothing ----
hw_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=7).fit()
hw_forecast = hw_model.forecast(test_size)
results.append(evaluate_forecast(test, hw_forecast, 'Holt-Winters'))

plt.figure(figsize=(14, 6))
plt.plot(train.index[-60:], train.iloc[-60:], label='Train (last 60 days)', color='gray')
plt.plot(test.index, test, label='Actual', color='black', linewidth=2)
plt.plot(test.index, naive_forecast, label='Naive', linestyle='--')
plt.plot(test.index, seasonal_naive_forecast, label='Seasonal Naive (7d)', linestyle='--')
plt.plot(test.index, hw_forecast, label='Holt-Winters', linestyle='--')
plt.title('Baseline Forecast Comparison')
plt.xlabel('Date')
plt.ylabel('Qty Dispensed')
plt.legend()
plt.tight_layout()
save_plot('baseline_forecast_comparison.png')
plt.show()

# ---- Stationarity check ----
adf_result = adfuller(train)
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")
if adf_result[1] > 0.05:
    print("Series is likely non-stationary — differencing needed (set d=1 or higher).")
else:
    print("Series is likely stationary.")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(train, lags=40, ax=axes[0])
plot_pacf(train, lags=40, ax=axes[1])
plt.tight_layout()
save_plot('acf_pacf_diagnostic.png')
plt.show()

# ---- ARIMA ----
arima_model = ARIMA(train, order=(7, 1, 1))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=test_size)
results.append(evaluate_forecast(test, arima_forecast, 'ARIMA(7,1,1)'))
print(arima_fit.summary())

# ---- SARIMAX (seasonal, weekly period) ----
sarimax_model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarimax_fit = sarimax_model.fit(disp=False)
sarimax_forecast = sarimax_fit.forecast(steps=test_size)
results.append(evaluate_forecast(test, sarimax_forecast, 'SARIMAX(1,1,1)(1,1,1,7)'))
print(sarimax_fit.summary())

plt.figure(figsize=(14, 6))
plt.plot(test.index, test, label='Actual', color='black', linewidth=2)
plt.plot(test.index, hw_forecast, label='Holt-Winters', linestyle='--')
plt.plot(test.index, arima_forecast, label='ARIMA', linestyle='--')
plt.plot(test.index, sarimax_forecast, label='SARIMAX', linestyle='--')
plt.title('Statistical Models vs. Actual Demand')
plt.xlabel('Date')
plt.ylabel('Qty Dispensed')
plt.legend()
plt.tight_layout()
save_plot('statistical_models_forecast_comparison.png')
plt.show()

# ============================================================
# MACHINE LEARNING / TREE MODELS — Random Forest, XGBoost, LightGBM
# ============================================================

# ---- Assemble ML feature matrix from lag + rolling + calendar features ----
ml_df = daily_qty.copy()  # dispense_date, qty_dispensed, qty_lag_1/7/14/28 (+ audit cols)

ml_df['day_of_week_num'] = ml_df['dispense_date'].dt.dayofweek
ml_df['day_of_month'] = ml_df['dispense_date'].dt.day
ml_df['month'] = ml_df['dispense_date'].dt.month
ml_df['quarter'] = ml_df['dispense_date'].dt.quarter
ml_df['is_weekend'] = (ml_df['dispense_date'].dt.dayofweek >= 5).astype(int)

ml_df['rolling_mean_7d'] = ml_df['qty_dispensed'].rolling(window=7, min_periods=1).mean()
ml_df['rolling_std_7d'] = ml_df['qty_dispensed'].rolling(window=7, min_periods=1).std()
ml_df['rolling_mean_30d'] = ml_df['qty_dispensed'].rolling(window=30, min_periods=1).mean()
ml_df['rolling_std_30d'] = ml_df['qty_dispensed'].rolling(window=30, min_periods=1).std()

feature_cols = [
    'qty_lag_1', 'qty_lag_7', 'qty_lag_14', 'qty_lag_28',
    'day_of_week_num', 'day_of_month', 'month', 'quarter', 'is_weekend',
    'rolling_mean_7d', 'rolling_std_7d', 'rolling_mean_30d', 'rolling_std_30d'
]

ml_df_clean = ml_df.dropna(subset=feature_cols).reset_index(drop=True)
print(f"ML-ready rows: {ml_df_clean.shape[0]} (dropped {ml_df.shape[0] - ml_df_clean.shape[0]} rows with incomplete features)")

# ---- Train/test split — final 30 days as holdout ----
X = ml_df_clean[feature_cols]
y = ml_df_clean['qty_dispensed']
dates = ml_df_clean['dispense_date']

ml_test_size = 30
X_train, X_test = X.iloc[:-ml_test_size], X.iloc[-ml_test_size:]
y_train, y_test = y.iloc[:-ml_test_size], y.iloc[-ml_test_size:]
dates_train, dates_test = dates.iloc[:-ml_test_size], dates.iloc[-ml_test_size:]

print(f"Train: {dates_train.min()} to {dates_train.max()} ({len(X_train)} rows)")
print(f"Test:  {dates_test.min()} to {dates_test.max()} ({len(X_test)} rows)")

# ---- TimeSeriesSplit cross-validation folds ----
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
    print(f"Fold {fold}: train rows {len(train_idx)}, val rows {len(val_idx)}, "
          f"val dates {dates_train.iloc[val_idx].min()} to {dates_train.iloc[val_idx].max()}")


def cross_validate_model(model, X_cv, y_cv, cv_splitter):
    cv_maes = []
    for train_idx, val_idx in cv_splitter.split(X_cv):
        X_tr, X_val = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
        y_tr, y_val = y_cv.iloc[train_idx], y_cv.iloc[val_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        cv_maes.append(mean_absolute_error(y_val, preds))
    return np.mean(cv_maes), np.std(cv_maes)


# ---- Random Forest ----
rf_model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
results.append(evaluate_forecast(y_test, rf_pred, 'Random Forest'))

# ---- XGBoost ----
xgb_model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
results.append(evaluate_forecast(y_test, xgb_pred, 'XGBoost'))

# ---- LightGBM ----
lgbm_model = LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
lgbm_model.fit(X_train, y_train)
lgbm_pred = lgbm_model.predict(X_test)
results.append(evaluate_forecast(y_test, lgbm_pred, 'LightGBM'))

# ---- Cross-validated MAE for each tree model ----
for name, model in [
    ('Random Forest', RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)),
    ('XGBoost', XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)),
    ('LightGBM', LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42))
]:
    mean_mae, std_mae = cross_validate_model(model, X_train, y_train, tscv)
    print(f"{name} — CV MAE: {mean_mae:.2f} (+/- {std_mae:.2f})")

# ---- Feature importance (XGBoost) ----
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(importances)

plt.figure(figsize=(10, 6))
plt.barh(importances['feature'], importances['importance'], color='#800020')
plt.title('XGBoost Feature Importance')
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
save_plot('xgboost_feature_importance.png')
plt.show()

# ---- Tree model forecasts vs. actual ----
plt.figure(figsize=(14, 6))
plt.plot(dates_test, y_test, label='Actual', color='black', linewidth=2)
plt.plot(dates_test, rf_pred, label='Random Forest', linestyle='--')
plt.plot(dates_test, xgb_pred, label='XGBoost', linestyle='--')
plt.plot(dates_test, lgbm_pred, label='LightGBM', linestyle='--')
plt.title('Tree Model Forecasts vs. Actual Demand')
plt.xlabel('Date')
plt.ylabel('Qty Dispensed')
plt.legend()
plt.tight_layout()
save_plot('tree_models_forecast_comparison.png')
plt.show()

# ============================================================
# FINAL MODEL COMPARISON — every model, sorted by RMSE
# ============================================================
results_df = pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True)
print(results_df)

results_df.to_csv(os.path.join(output_dir, 'model_comparison_results.csv'), index=False)
print(f"Saved: {os.path.join(output_dir, 'model_comparison_results.csv')}")
