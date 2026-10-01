# Vending Machine Demand Forecasting & Inventory Planning

**Data Engineering Academy learning project | Miguel Angel Zapata**

This project explores how historical vending-machine demand and replenishment data can support inventory planning, device prioritization, and restocking decisions. It brings together exploratory outputs, forecasting comparisons, SKU-level inventory calculations, and a proposed operational roadmap.

## Project Context & AI Contribution

This project was developed as part of my Data Engineering Academy studies. **Claude performed the analysis and generated the advanced forecasting work, calculations, and recommendations.** I used the project as a learning exercise to explore how Python and analytical methods connect operational data to business decisions.

The models, model-selection results, and inventory calculations in this repository should be understood as AI-generated analytical work. I do not claim to have independently developed or validated the complete forecasting pipeline. My involvement centered on working through the project with AI assistance, asking questions, and studying the outputs and their business meaning.

The purpose of sharing this project is to document my learning process and developing analytical understanding. The presence of a script, chart, or report does not by itself establish independent authorship or mastery of the methods used. ChatGPT/Codex assisted with preparing this README from the supplied project artifacts.

## Business Problem

Restocking on a fixed schedule can lead to unnecessary visits at slower devices and insufficient inventory at faster devices. The project explores three questions:

1. How does dispensing demand vary over time, across SKUs, and across devices?
2. Which forecasting approaches perform best on the reported evaluation metrics?
3. How could demand variability inform inventory buffers and replenishment priorities?

The intended business application is proactive replenishment. Reduced costs, fewer stockouts, and improved route efficiency are proposed outcomes—not measured achievements of this project.

## Analytical Scope

The supplied artifacts cover:

- Daily and weekly dispensing trends, weekday/weekend comparisons, and SKU turnover.
- Rolling averages, seasonal decomposition, and monthly/day-of-week patterns for leading SKUs.
- Device utilization, replenishment activity, and correlation/lag-correlation analysis.
- Lag features, a missing-feature audit, and restock-signal examples.
- Baseline, statistical, and tree-based forecasting comparisons.
- SKU safety-stock and reorder-point estimates, device priorities, and an operational roadmap.

These are documented outputs of the AI-assisted exercise. Complete source data and code for every output are required before the full workflow can be reproduced and independently checked.

## Reported Forecasting Results

The following values are taken from `model_comparison_results.csv`. MAE and RMSE are expressed in the forecast target's units; MAPE is a percentage. Lower values indicate smaller errors on the corresponding metric.

| Model | MAE | RMSE | MAPE (%) |
| --- | ---: | ---: | ---: |
| XGBoost | 91.28 | 231.41 | 18.83 |
| LightGBM | 100.08 | 238.53 | 39.37 |
| Random Forest | 87.72 | 243.91 | 22.21 |
| Seasonal Naive (7-day) | 85.70 | 256.57 | 36.69 |
| SARIMAX(1,1,1)(1,1,1,7) | 93.65 | 262.52 | 41.55 |
| ARIMA(7,1,1) | 103.60 | 265.38 | 46.79 |
| Holt-Winters | 99.96 | 266.81 | 41.80 |
| Naive (last value) | 108.60 | 267.50 | 48.45 |

**XGBoost has the lowest reported MAPE and RMSE; Seasonal Naive has the lowest MAE.** Model preference therefore depends on the evaluation metric and business objective. An 18.83% MAPE should not be described as “81.17% accuracy.”

The presentation describes these as holdout results. The exported metrics alone do not establish the train/test dates, forecast horizon, tuning procedure, or whether all models used an identical evaluation window. These results have not been independently reproduced for this README.

## Interpretation & Operational Proposals

### Demand patterns and anomalies

The project includes trend, seasonal, and rolling-demand outputs. Its presentation calls for investigating an unusual mid-February demand spike with field operations to distinguish a real business event from a telemetry issue. An unexplained spike should remain an investigation item rather than receive an assumed explanation.

### Predictive features

The deck reports approximately 62% XGBoost importance for the seven-day rolling mean, followed by lagged demand and calendar features. This describes the fitted model's feature-importance measure; it does not establish causation. Confirming that rolling features exclude the prediction day's target is essential before accepting the performance results.

### Inventory planning

`sku_safety_stock_reorder_points.csv` contains estimates for **82 SKUs**. The deck describes a **95% service-level assumption**, a **Z value of 1.65**, and an **estimated two-day lead time**. These assumptions are planning inputs, not verified supplier performance or achieved service levels.

Before implementation, inventory estimates would need validation against actual lead times, available stock, machine capacity, pack sizes, stockout history, and the chosen service-level definition. Dispensing alone may understate demand when products are unavailable.

### Device replenishment

`device_restock_schedule.csv` contains **five devices** and proposed review frequencies. The presentation also includes illustrative facilities, dispatch times, and batch sizes. These should be treated as operational scenarios pending validation, rather than confirmed locations or optimized routes.

The deck's classification table accounts for four devices, while the schedule CSV lists five. The deck and CSV also differ in some replenishment/review intervals. These differences should be reconciled before using the plan operationally. A review interval is not necessarily a restock interval.

## Limitations & Items Requiring Validation

- **Reproducibility:** Supporting charts and exported results do not replace the full forecasting code, source dataset, dependency versions, and evaluation configuration.
- **Time-series validation:** Verify chronological splits, consistent evaluation windows, and a clearly defined forecast horizon. Ensure lag and rolling features use only information available at prediction time.
- **Missing lag features:** The missing-feature summary contains 28 rows with early-date lag gaps. Such gaps can arise naturally at the beginning of a series. The deck's claim that missing windows resulted from offline maintenance and were imputed requires supporting code and operational evidence.
- **Correlation:** The deck reports a strong negative dispensing/restock correlation. Correlation alone cannot confirm a reactive restocking policy or explain its cause.
- **Stockouts and routing:** Volume-based categories cannot prove that stockout risk is absent. Route recommendations need location, travel-time, capacity, and service-window data.
- **Business impact:** No measured cost savings, stockout reduction, or production deployment is established by the supplied artifacts.
- **Data provenance:** The supplied outputs do not fully document the original dataset's source, collection process, or redistribution terms. Confirm these before publishing raw data.

## Project Artifacts

The filenames below identify supplied project outputs. Their locations can be adjusted when organizing the GitHub repository.

| Artifact | Purpose |
| --- | --- |
| `Vending Machine Demand Forecasting & Operational Roadmap.pptx` | Project presentation and proposed roadmap |
| `model_comparison_results.csv` | Exported forecast error metrics |
| `sku_safety_stock_reorder_points.csv` | SKU inventory planning estimates |
| `device_restock_schedule.csv` | Proposed device priorities and review intervals |
| `lag_features_missing_summary.csv` | Summary of rows missing lag features |
| `lag_features_pending_review.csv` | Records retained for further review |
| `trend_daily_weekly_qty_dispensed.png` | Dispensing trends |
| `product_turnover_top10_skus.png` | Leading SKUs by turnover |
| `feature_weekday_vs_weekend_avg_dispensed.png` | Weekday/weekend comparison |
| `device_utilization_scatter.png` | Device classification visualization |
| `device_restock_priority_tiers.png` | Device replenishment priorities |
| `correlation_heatmap_daily.png` | Daily correlation analysis |
| `correlation_lag_restock_vs_dispense.png` | Lag-correlation analysis |
| `rolling_avg_top_sku.png` | Rolling-demand view |
| `seasonal_decomposition_top_sku.png` | Decomposition of leading-SKU demand |
| `seasonality_top1_sku_monthly_dow.png` through `seasonality_top5_sku_monthly_dow.png` | Seasonality views for five leading SKUs |
| `acf_pacf_diagnostic.png` | Time-series diagnostics |
| `baseline_forecast_comparison.png` | Baseline forecasts |
| `statistical_models_forecast_comparison.png` | Statistical-model comparison |
| `tree_models_forecast_comparison.png` | Tree-model comparison |
| `xgboost_feature_importance.png` | Reported XGBoost feature importance |
| `sku_demand_variability_category_counts.png` | SKU variability categories |
| `restock_signal_features_sample_device.png` | Example restock-signal features |

## Learning Focus & Next Steps

This exercise introduces the connection between demand forecasting, inventory buffers, replenishment priorities, and operational decisions. My learning focus is understanding what each output means, questioning assumptions, distinguishing findings from recommendations, and explaining the business problem accurately.

The next steps are to collect the complete generating code, reproduce the reported results, document dataset provenance and evaluation settings, reconcile the scheduling artifacts, and validate inventory assumptions with operational data. A small replenishment pilot could then measure service levels, stockouts, visit frequency, and costs against a baseline.

## Author & Acknowledgments

**Miguel Angel Zapata** — Data Engineering Academy student transitioning from security operations and investigations into data analytics.

[GitHub Portfolio](https://github.com/miguelzapata1967)

Acknowledgments: Data Engineering Academy for the learning context; Claude for the analytical work; ChatGPT/Codex for README preparation. This is an educational portfolio project, with AI involvement disclosed above.
