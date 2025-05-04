# FORECASTING BRENT CRUDE OIL PRICE AND OECD OIL DEMAND

This repository contains a Python program that automates the end‑to‑end forecasting of two key energy time series:  
  **1. Brent crude oil spot price** (USD per barrel)  
  **2. OECD refined petroleum products consumption** (thousand barrels per day)  

Data are fetched directly from the U.S. Energy Information Administration (EIA) API v2, cleaned and resampled to a uniform month‑end frequency, 
modeled with two complementary approaches (Prophet and XGBoost), and evaluated via both a one‑year hold‑out test and five‑fold time‑series cross‑validation. 
The final 24‑month forecasts are saved as CSV files, and a suite of visualizations is produced.  

## Repository Contents
```
├── forecast_crude_oil_and_oecd_demand.py   # Main forecasting script
├── data/                                   # Output folder for CSV forecasts
│   ├── brent_forecast.csv
│   └── oecd_forecast.csv
├── figures/                                # Generated plot images
│   ├── Figure_1.png  …  Figure_7.png
├── README.md                               # This file
└── requirements.txt                        # Python dependencies
```

## Prerequisites
- Python 3.8 or later
- A valid EIA API key (register at https://www.eia.gov/opendata/)  

## Installation
1. Clone this repository:
```
git clone https://github.com/your‑username/oil-forecasting.git
cd oil-forecasting
```
2. (Optional) Create and activate a virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate       # Windows
```
3. Install required Python packages:
```
pip install -r requirements.txt
```

## Configuration
Edit the top of `forecast_crude_oil_and_oecd_demand.py` to set your EIA API key:
```
API_KEY = "YOUR_EIA_API_KEY_HERE"
```

## Usage
Run the forecasting script:
```
python forecast_crude_oil_and_oecd_demand.py
```
This will:  
1. Fetch Brent price and OECD demand series from the EIA API.
2. Clean and resample each series to month‑end.  
3. Perform Exploratory Data Analysis (correlation heatmap, pairplot, feature‑importance).
4. Train two models per series:
   - **Prophet** (additive time‑series)
   - **XGBoost** (lag‑feature regression)
5. Evaluate on a 12‑month hold‑out and via 5‑fold time‑series cross‑validation.
6. Generate and display plots.
7. Save 24‑month forecasts to data/brent_forecast.csv and data/oecd_forecast.csv.

## Code Structure
- Data fetching
  - `fetch_eia_series(series_id, api_key)`
  - `fetch_oecd_demand(api_key)`
- Preprocessing
  - `prepare_monthly_df(df)`
  - Interpolation and scaling within `main()`
- Exploratory Data Analysis
  - Correlation heatmap (Figure 1)
  - Pairplot of scaled series (Figure 2)
  - Feature importance via XGBoost (Figure 3)
- Modeling & Evaluation
  - `train_and_forecast(df, periods, series_name)`
     - Fits Prophet and XGBoost
     - Computes hold‑out RMSE/MAE/MAPE
     - Calls `cross_validate_model(...)` for 5‑fold CV
     - Plots test vs. forecast and full forecast
- Cross-Validation
  - `cross_validate_model(X, y, model, n_splits=5)`

## Output
- CSV forecasts in `data/`:
  - `brent_forecast.csv`: 24‑month ahead Brent price predictions
  - `oecd_forecast.csv`: 24‑month ahead OECD demand predictions
- Figures displayed:
  - Figure 1: Correlation heatmap
  - Figure 2: Pairplot of scaled series
  - Figure 3: XGBoost feature importances
  - Figure 4–7: Test vs. forecast and full‑forecast plots for each series
- Terminal:
  - Error statistics and 5-fold cross-validation

## Dependencies
Listed in `requirements.txt`:  
```
pandas
numpy
requests
prophet
scikit-learn
xgboost
matplotlib
seaborn
```

## Extending the Project
Possible enhancements:
- Incorporate additional exogenous predictors (GDP, inventory, rig counts).
- Experiment with SARIMAX or ensemble models.
- Automate periodic re‑training and deployment.



  
