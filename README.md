# README – Energy Consumption Forecasting (Fortum Junction 2025 Challenge)

## 1. Overview

This project implements a complete, reproducible machine learning pipeline for the Fortum Junction 2025 Energy Consumption Forecasting Challenge.  
It provides:

- End-to-end data loading and preprocessing  
- Feature engineering for hourly and monthly prediction tasks  
- Two global LightGBM regression models  
- Recursive forecasting for:
  - 48-hour ahead hourly consumption
  - 12-month ahead aggregated monthly consumption
- Output aligned with the official submission templates  
- Export in European CSV format (semicolon separator, comma decimal)

The objective is to produce reliable baseline forecasts for 112 customer groups using historical consumption and electricity price data.

---

## 2. Dataset Description

The official dataset (`20251111_JUNCTION_training.xlsx`) contains three sheets:

### 2.1 training_consumption  
- Hourly consumption data for 112 customer groups  
- Coverage: 2021-01-01 00:00 → 2024-09-30 23:00  
- Provided in wide format; converted to long format for modeling:  
  - `measured_at`, `group_id`, `consumption`

### 2.2 training_prices  
- Hourly electricity spot prices  
- Columns: `measured_at`, `eur_per_mwh`  
- Used to generate price lag features.

### 2.3 groups  
- Group metadata (not required by the baseline model)

---

## 3. Model Architecture

Two separate LightGBM models are used:

### 3.1 Hourly Model  
- A single global model for all groups  
- Predicts hourly consumption  
- Used for 48-hour recursive forecasting  
- Inputs include lag features, rolling statistics, cyclical encodings, and electricity prices.

### 3.2 Monthly Model  
- Aggregates hourly data into monthly totals  
- Trains a LightGBM model on monthly features  
- Used to predict monthly consumption 12 months ahead  
- Inputs include month, year, and lag values (1-month and 12-month lags).

---

## 4. Feature Engineering

### 4.1 Hourly Features  
| Category | Features |
|----------|----------|
| Time features | hour, dayofweek, month, year, is_weekend |
| Cyclical encodings | hour_sin, hour_cos |
| Lag features | lag_1, lag_24, lag_168 |
| Statistical features | rolling_24h_mean, rolling_24h_std |
| Price-based features | eur_per_mwh, price_lag_1, price_lag_24 |

### 4.2 Monthly Features  
- month  
- year  
- lag_1m  
- lag_12m  

These features capture seasonality, recent trends, and annual recurrence patterns.

---

## 5. Forecasting Methodology

### 5.1 48-hour Hourly Forecast  
A recursive process is used:

1. Identify the final timestamp in the training data.  
2. For each of the next 48 hours:  
   - Construct a temporary history containing all known and predicted values.  
   - Recompute lag and time features.  
   - Predict consumption for all groups.  
   - Append predictions to history for the next iteration.  

### 5.2 12-month Monthly Forecast  

1. Aggregate the historical dataset into monthly totals.  
2. Identify the final available month.  
3. For each of the next 12 months:  
   - Insert a placeholder row for all groups.  
   - Recompute lag and time features.  
   - Predict consumption.  
   - Update the dataset so predictions become future lags.  

---

## 6. Submission Format

Two CSV files are produced:

### 6.1 my_hourly_forecast.csv  
- 48 rows × 113 columns  
- Columns: `measured_at` and 112 group IDs  
- Timestamps formatted as `YYYY-MM-DDTHH:MM:SS.000Z`  
- Semicolon-separated  
- Decimal comma format

### 6.2 my_monthly_forecast.csv  
- 12 rows × 113 columns  
- Same column structure and formatting rules.

Both outputs are aligned to the official template column order.

---

## 7. How to Run the Project

### 7.1 Install required dependencies

pip install pandas numpy scikit-learn lightgbm
###7.2 Place the required input files in the working directory:
20251111_JUNCTION_training.xlsx  
20251111_JUNCTION_example_hourly.csv  
20251111_JUNCTION_example_monthly.csv

###7.3 Run the pipeline
python junction.py

###7.4 Outputs generated
my_hourly_forecast.csv  
my_monthly_forecast.csv


These files are fully compliant with the submission format.

##8. Project Structure
project/
│
├── junction.py                      # Full ML pipeline
├── README.md                        # Documentation
│
├── 20251111_JUNCTION_training.xlsx
├── 20251111_JUNCTION_example_hourly.csv
├── 20251111_JUNCTION_example_monthly.csv
│
├── my_hourly_forecast.csv
└── my_monthly_forecast.csv

##9. Model Evaluation

Both models include internal validation:

Hourly model: last month of data used as validation

Monthly model: last six months used as validation

The script prints the Mean Absolute Error (MAE) for both:

Example:

Validation MAE (hourly): 0.XXX  
Validation MAE (monthly): XX.XXX

##10. Future Improvements

Several enhancements can significantly improve performance:

Integration of external weather data (temperature, wind chill, daylight).

Inclusion of holiday and event features.

Group embeddings or deep hybrid models.

Probabilistic and quantile forecasting.

Anomaly detection and robust preprocessing.

Multi-step forecasting models trained specifically for long-horizon prediction.
