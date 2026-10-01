# Technical Documentation — Vending Machine Demand Forecasting

[Project Introduction](README.md) · [Analysis & Findings](readme_analysis.md)

## Code & Attribution

[`DE_Academy_vending_machine.py`](DE_Academy_vending_machine.py) is the Python script identified for this project. It was added to the local repository in commit `bfaecbf` (843 lines). Upload completion has not been confirmed in the provided terminal output.

The work is AI-assisted: Claude generated the analysis and advanced modeling. This document describes the supplied outputs and code sections visible in the project screenshot. It is not a full source-code audit or a claim that the workflow has been independently reproduced.

The visible script sections include LightGBM fitting, Random Forest/XGBoost/LightGBM cross-validation calls, XGBoost feature-importance export, tree-model comparison charts, and a model-results table sorted by RMSE. Exact input paths, feature definitions, evaluation settings, and all dependencies still require inspection of the complete script.

## Documented Workflow

| Stage | Purpose | Supporting outputs |
| --- | --- | --- |
| Data preparation and exploration | Examine dispensing, SKU behavior, and replenishment activity | Trends, turnover, device utilization, correlation charts |
| Time-series exploration | Examine rolling demand and seasonal/lag structure | Rolling averages, decomposition, seasonality, ACF/PACF |
| Feature preparation | Create historical-demand and calendar features; audit missing lags | Lag audit CSVs and restock-signal chart |
| Model comparison | Compare baselines, statistical methods, and tree models | Forecast charts and model metrics CSV |
| Inventory estimates | Translate demand and variability into planning quantities | SKU safety-stock and reorder-point CSV |
| Device planning | Propose review frequencies and replenishment priorities | Device schedule CSV and priority chart |

The available exported metrics cover Naive, Seasonal Naive, Holt-Winters, ARIMA, SARIMAX, Random Forest, XGBoost, and LightGBM. Full model settings must be obtained from the generating code.

## Tools & Dependencies

The project uses Python. The visible code references Pandas, Matplotlib, Random Forest, XGBoost, and LightGBM. The statistical outputs indicate additional time-series tooling, but exact imports and library versions have not been verified here.

No pinned dependency file or tested installation instructions are established by the supplied artifacts. Inspect the imports and record a tested Python version and dependency versions before publishing a `requirements.txt` or claiming reproducibility.

## Features & Missing Values

The outputs describe lagged dispensing quantities, seven- and thirty-day rolling statistics, and calendar information. The missing-feature summary identifies gaps in lag-1, lag-7, lag-14, and lag-28 columns across 28 early-date rows.

Initial lag gaps can occur because insufficient historical observations exist. The presentation attributes missing windows to offline maintenance and mentions rolling-median imputation; those claims require verification in the code and operational records.

For model evaluation, check that:

- Each lag and rolling statistic uses only observations available before the target prediction time.
- Missing-value processing does not use future observations.
- Training, tuning, and evaluation preserve chronological order.
- All models are evaluated on comparable dates and horizons.
- Multi-step forecasts do not assume future actual demand is available unless that is explicitly the evaluation design.

## Evaluation Metrics

[`data/model_comparison_results.csv`](data/model_comparison_results.csv) records MAE, RMSE, and MAPE for eight models. Lower values indicate smaller errors under each metric. The analysis README contains the reported values and their business interpretation.

Before reproducing the evaluation, document the target aggregation level, training/test dates, forecast horizon, cross-validation splits, handling of zero actual values in MAPE, and whether model tuning used a separate validation period.

## Inventory Calculation Assumptions

The presentation specifies a 95% service-level assumption, `Z = 1.65`, and an estimated two-day lead time. The SKU export includes rolling means, rolling standard deviations, lead time, safety stock, and reorder point.

Verify the actual formulas and units in the script. Also document which demand window is used, whether the service level means cycle service level or fill rate, and how fractional estimates should be rounded for physical inventory. Fixed lead-time assumptions need revision if real supplier lead times vary.

## Data Exports

| File | Contents |
| --- | --- |
| [model_comparison_results.csv](data/model_comparison_results.csv) | Eight model error summaries |
| [sku_safety_stock_reorder_points.csv](data/sku_safety_stock_reorder_points.csv) | Inventory estimates for 82 SKUs |
| [device_restock_schedule.csv](data/device_restock_schedule.csv) | Priorities and proposed review frequencies for five devices |
| [lag_features_missing_summary.csv](data/lag_features_missing_summary.csv) | Summary of missing lag combinations |
| [lag_features_pending_review.csv](data/lag_features_pending_review.csv) | Feature records awaiting review |

These CSVs are analytical outputs. They do not establish that the original source dataset is included in `data/`.

## Graph Files

All graph paths are relative to the repository root:

- [trend_daily_weekly_qty_dispensed.png](graphs/trend_daily_weekly_qty_dispensed.png) — Dispensing trends.
- [product_turnover_top10_skus.png](graphs/product_turnover_top10_skus.png) — Leading SKUs by turnover.
- [feature_weekday_vs_weekend_avg_dispensed.png](graphs/feature_weekday_vs_weekend_avg_dispensed.png) — Weekday/weekend comparison.
- [device_utilization_scatter.png](graphs/device_utilization_scatter.png) — Device classification visualization.
- [device_restock_priority_tiers.png](graphs/device_restock_priority_tiers.png) — Device replenishment priorities.
- [correlation_heatmap_daily.png](graphs/correlation_heatmap_daily.png) — Daily correlation analysis.
- [correlation_lag_restock_vs_dispense.png](graphs/correlation_lag_restock_vs_dispense.png) — Lag-correlation analysis.
- [rolling_avg_top_sku.png](graphs/rolling_avg_top_sku.png) — Rolling-demand view.
- [seasonal_decomposition_top_sku.png](graphs/seasonal_decomposition_top_sku.png) — Decomposition of leading-SKU demand.
- [seasonality_top1_sku_monthly_dow.png](graphs/seasonality_top1_sku_monthly_dow.png) — Seasonality view for leading SKU 1.
- [seasonality_top2_sku_monthly_dow.png](graphs/seasonality_top2_sku_monthly_dow.png) — Seasonality view for leading SKU 2.
- [seasonality_top3_sku_monthly_dow.png](graphs/seasonality_top3_sku_monthly_dow.png) — Seasonality view for leading SKU 3.
- [seasonality_top4_sku_monthly_dow.png](graphs/seasonality_top4_sku_monthly_dow.png) — Seasonality view for leading SKU 4.
- [seasonality_top5_sku_monthly_dow.png](graphs/seasonality_top5_sku_monthly_dow.png) — Seasonality view for leading SKU 5.
- [acf_pacf_diagnostic.png](graphs/acf_pacf_diagnostic.png) — Time-series diagnostics.
- [baseline_forecast_comparison.png](graphs/baseline_forecast_comparison.png) — Baseline forecasts.
- [statistical_models_forecast_comparison.png](graphs/statistical_models_forecast_comparison.png) — Statistical-model comparison.
- [tree_models_forecast_comparison.png](graphs/tree_models_forecast_comparison.png) — Tree-model comparison.
- [xgboost_feature_importance.png](graphs/xgboost_feature_importance.png) — Reported XGBoost feature importance.
- [sku_demand_variability_category_counts.png](graphs/sku_demand_variability_category_counts.png) — SKU variability categories.
- [restock_signal_features_sample_device.png](graphs/restock_signal_features_sample_device.png) — Example restock-signal features.

## Reproduction Status

The Python script is now identified, which improves the documentation beyond an outputs-only project. Its presence does not yet verify that every reported output can be regenerated.

Before running it:

1. Inspect its input filenames and obtain the corresponding source data.
2. Replace local absolute paths with repository-relative paths where appropriate.
3. Confirm that charts are saved in `graphs/` and result exports in `data/`; moving existing outputs does not automatically update script paths.
4. Install and record the dependencies in an isolated environment.
5. Validate feature timing, missing-value handling, and model evaluation settings.
6. Run the workflow and compare regenerated metrics and outputs with the committed files.

Once inputs, dependencies, and paths are configured, the intended command from the repository root is:

```powershell
python .\DE_Academy_vending_machine.py
```

This command has not been tested against the reorganized repository. Document dataset provenance and redistribution permissions alongside the input requirements.
