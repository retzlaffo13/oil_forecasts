"""
forecast_crude_oil_and_oecd_demand.py

Team Project
ITSCM 180: Programming for Business Applications
Owen Retzlaff, Jolea Wallisch

This program fetches Brent crude oil monthly spot prices from the EIA APIv2 backward compatibility endpoint,
fetches OECD total refined petroleum products consumption via EIA international API v2,
builds predictive models using Prophet, visualizes historical data, forecasts,and correlation,
and evaluates model performance using error metrics.

Usage:
  pip install requests pandas prophet matplotlib scikit-learn plotly
  python forecast_crude_oil_and_oecd_demand.py
"""

# Imports
import requests
import pandas as pd
from prophet import Prophet
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import numpy as np
import os

# Configuration
API_KEY = 'T6EU4Q8Q4bDlYScp7C9bHxHjcJoimBZcCMpuH4NY'
BRENT_SERIES_ID = 'PET.RBRTE.M'

# Fetch functions
def fetch_eia_series(series_id, api_key):
    url = f'https://api.eia.gov/v2/seriesid/{series_id}?api_key={api_key}'
    resp = requests.get(url); resp.raise_for_status()
    data = resp.json().get('response')
    if not data or 'data' not in data:
        raise RuntimeError(f"Error fetching series {series_id}")
    df = pd.DataFrame(data['data'])
    df.rename(columns={'period':'date','value':'value'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df.dropna(subset=['value'], inplace=True)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df[['value']]


def fetch_oecd_demand(api_key):
    url = 'https://api.eia.gov/v2/international/data'
    params = {
        'api_key': api_key,
        'frequency': 'monthly',
        'data[0]': 'value',
        'facets[activityId][]': '2',
        'facets[productId][]': '54',
        'facets[countryRegionId][]': 'OECD',
        'facets[unit][]': 'TBPD',
        'sort[0][column]':'period','sort[0][direction]':'asc',
        'offset':'0','length':'5000'
    }
    resp = requests.get(url, params=params); resp.raise_for_status()
    data = resp.json().get('response')
    if not data or 'data' not in data:
        raise RuntimeError("Error fetching OECD demand")
    df = pd.DataFrame(data['data'])
    df.rename(columns={'period':'date','value':'value'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df.dropna(subset=['value'], inplace=True)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    return df[['value']]


def prepare_monthly_df(df):
    monthly = df.resample('ME').mean().reset_index()
    monthly.rename(columns={'date':'ds','value':'y'}, inplace=True)
    return monthly[['ds','y']]

def cross_validate_model(X, y, model, n_splits=5):
    """
    Run TimeSeriesSplit CV on (X, y) with the given model,
    printing RMSE/MAE/MAPE for each fold and the average ± std.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses, maes, mapes = [], [], []

    for i, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        mae  = mean_absolute_error(y_te, y_pred)
        mape = np.mean(np.abs((y_te - y_pred) / y_te)) * 100

        print(f"  Fold {i:>2} → RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.1f}%")
        rmses.append(rmse); maes.append(mae); mapes.append(mape)

    print(f"\n  CV avg → RMSE: {np.mean(rmses):.2f} ± {np.std(rmses):.2f}, "
          f"MAE: {np.mean(maes):.2f} ± {np.std(maes):.2f}, "
          f"MAPE: {np.mean(mapes):.1f}% ± {np.std(mapes):.1f}%\n")

def train_and_forecast(df, periods, series_name, unit):
    if len(df) < 24:
        raise ValueError(f"Need ≥24 points for {series_name}, got {len(df)}")
    train = df[:-12]; test = df[-12:]

    # ── Prophet ──
    m = Prophet()
    m.fit(train)
    fut= m.make_future_dataframe(periods=periods, freq='ME')
    fc = m.predict(fut)

    # Hold-out metrics
    idx    = fc.set_index('ds')
    true_p = test['y'].values
    pred_p = idx.reindex(test['ds'])['yhat'].values
    rmse_p = np.sqrt(mean_squared_error(true_p, pred_p))
    mae_p  = mean_absolute_error(true_p, pred_p)
    mape_p = np.mean(np.abs((true_p - pred_p)/true_p))*100

    print(f"\n{series_name} – Hold-Out Test")
    print(f"  Prophet → RMSE: {rmse_p:.2f}, MAE: {mae_p:.2f}, MAPE: {mape_p:.1f}%")

    # ── XGBoost on lagged features ──
    df_xgb = df.set_index('ds').copy()
    for lag in (1,2,3,6,12):
        df_xgb[f'lag_{lag}'] = df_xgb['y'].shift(lag)
    df_xgb.dropna(inplace=True)

    X       = df_xgb.drop('y', axis=1)
    y_xgb   = df_xgb['y']
    X_train = X.iloc[:-12]; y_train = y_xgb.iloc[:-12]
    X_test  = X.iloc[-12:]; y_test  = y_xgb.iloc[-12:]

    model_xgb = xgb.XGBRegressor(objective='reg:squarederror')
    model_xgb.fit(X_train, y_train)
    pred_xgb = model_xgb.predict(X_test)

    rmse_x = np.sqrt(mean_squared_error(y_test, pred_xgb))
    mae_x  = mean_absolute_error(y_test, pred_xgb)
    mape_x = np.mean(np.abs((y_test - pred_xgb)/y_test))*100

    print(f"  XGBoost → RMSE: {rmse_x:.2f}, MAE: {mae_x:.2f}, MAPE: {mape_x:.1f}%")

    # ── Time-Series CV for XGBoost ──
    print(f"\n{series_name} – 5-Fold Time-Series CV")
    cross_validate_model(X, y_xgb, xgb.XGBRegressor(objective='reg:squarederror'), n_splits=5)

    # Plot Prophet results
    cutoff = df['ds'].max() - pd.DateOffset(years=3)
    mask_train = train['ds'] >= cutoff
    mask_test  = test ['ds'] >= cutoff

    plt.figure(figsize=(10,6))
    plt.plot(train.loc[mask_train,'ds'], train.loc[mask_train,'y'],    label='Train')
    plt.plot(test .loc[mask_test, 'ds'], test .loc[mask_test, 'y'],    label='Actual')
    plt.plot(test .loc[mask_test, 'ds'], pred_p[mask_test],            label='Prophet Forecast')
    plt.plot(test .loc[mask_test, 'ds'], pred_xgb[mask_test],          label='XGBoost Forecast', linestyle='--')
    plt.xlabel('Date (Year-Month)')
    plt.ylabel(unit)
    plt.title(f"{series_name} Test vs Forecast (Last 3 Years)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Full Prophet forecast
    cutoff = df['ds'].max() - pd.DateOffset(years=3)
    fc20 = fc[fc['ds'] >= cutoff].copy()
    m.history = m.history[m.history['ds'] >= cutoff].copy()
    fig = m.plot(fc20)
    ax  = fig.gca()
    ax.set_xlabel('Date (Year-Month)')
    ax.set_ylabel(unit)
    plt.title(f"{series_name} Full Forecast (Last 3 Years)")
    plt.tight_layout()
    plt.show()

    return fc

def price_vs_demand(br_m, od_m):
    eda_df = pd.merge(br_m.rename(columns={'y':'brent'}),od_m.rename(columns={'y':'oecd'}),on='ds',how='inner').set_index('ds')

    # Ensure a contiguous monthly index & interpolate any small gaps
    eda_df = eda_df.asfreq('ME')
    eda_df.interpolate(method='time', inplace=True)

    # Standardize both series for comparative plots
    scaler = StandardScaler()
    eda_df[['brent_scaled','oecd_scaled']] = scaler.fit_transform(eda_df[['brent','oecd']])

    # Correlation heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(eda_df[['brent','oecd']].corr(),annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation: Brent Price vs OECD Demand')
    plt.xlabel('Series')
    plt.ylabel('Series')
    plt.tight_layout()
    plt.show()

    # Pairplot of the scaled series
    sns.pairplot(eda_df[['brent_scaled','oecd_scaled']].dropna(),diag_kind='kde')
    plt.suptitle('Pairplot of Scaled Brent Price & OECD Demand', y=1.02)
    plt.show()

    # Feature Importance via XGBoost
    feat_df = eda_df.copy()
    for lag in (1, 2, 3, 6, 12):
        feat_df[f'brent_lag_{lag}'] = feat_df['brent'].shift(lag)
        feat_df[f'oecd_lag_{lag}']  = feat_df['oecd'].shift(lag)
    feat_df.dropna(inplace=True)

    X_feat = feat_df.drop(['brent','oecd','brent_scaled','oecd_scaled'], axis=1)
    y_feat = feat_df['brent']
    model_feat = xgb.XGBRegressor(objective='reg:squarederror')
    model_feat.fit(X_feat, y_feat)

    importances = pd.Series(model_feat.feature_importances_,index=X_feat.columns).nlargest(10).sort_values()

    plt.figure(figsize=(8, 6))
    importances.plot(kind='barh')
    plt.title('Top 10 Feature Importances for Predicting Brent Price')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

def main():
    os.makedirs('data', exist_ok=True)

    print("Fetching Brent...")
    br   = fetch_eia_series(BRENT_SERIES_ID, API_KEY)
    br_m = prepare_monthly_df(br)

    print("\nFetching OECD demand...\n")
    od   = fetch_oecd_demand(API_KEY)
    od_m = prepare_monthly_df(od)

    # Joint price/demand correlation and feature importance
    price_vs_demand(br_m, od_m)
    
    # Model and forecast individual series
    fc_br = train_and_forecast(br_m, 24, 'Brent Price (USD)', 'USD / barrel')
    fc_od = train_and_forecast(od_m, 24, 'OECD Demand (TBPD)', 'Thousand barrels per day')

    # Save forecasts
    fc_br[['ds','yhat']].to_csv('data/brent_forecast.csv', index=False)
    fc_od[['ds','yhat']].to_csv('data/oecd_forecast.csv', index=False)

    print("\nForecast files saved.")

if __name__ == '__main__':
    main()