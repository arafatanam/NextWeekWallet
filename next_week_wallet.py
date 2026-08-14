import os
import time
import warnings

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from dotenv import load_dotenv

load_dotenv()

warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")


class config:
    HORIZONS = [1, 2, 3, 4]
    TRAIN_FRACTION = 0.8
    RANDOM_STATE = 42
    DATA_DIR = "data"
    RSI_WINDOW = 14
    VOL_WINDOWS = (10, 20)
    REL_VOLUME_WINDOW = 20
    DISPLAY_YEARS = 2
    BACKTEST_INITIAL_FRACTION = 0.7
    BACKTEST_STEP_WEEKS = 12
    TUNE_ITER = 25
    DATE_FMT = "%d-%m-%Y"


os.makedirs(config.DATA_DIR, exist_ok=True)

print("NextWeekWallet - next-week stock forecasting tool")
print()

API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
if not API_KEY:
    raise RuntimeError(
        "No Alpha Vantage key found.\n"
        "Copy .env.example to .env and add your key, or run:\n"
        "  export ALPHA_VANTAGE_API_KEY=your_key_here\n"
        "Get a free key at https://www.alphavantage.co/support/#api-key"
    )


## 1. Ticker input
TICKER = input("Enter stock ticker symbol (e.g. AAPL): ").strip().upper()
if not TICKER:
    raise RuntimeError("No ticker entered.")


## 2. Data loading
def _cache_path(ticker: str) -> str:
    return os.path.join(config.DATA_DIR, f"{ticker}_weekly_adj.csv")


def _download_from_alpha_vantage(ticker: str) -> pd.DataFrame:
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_WEEKLY_ADJUSTED",
        "symbol": ticker,
        "apikey": API_KEY,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    key = "Weekly Adjusted Time Series"
    if key not in payload:
        raise RuntimeError(f"Alpha Vantage did not return price data for {ticker}. Raw response: {payload}")

    raw = pd.DataFrame.from_dict(payload[key], orient="index")
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()
    raw = raw.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. adjusted close": "adj_close",
        "6. volume": "volume",
        "7. dividend amount": "dividend",
    })
    raw = raw.astype(float)

    out = pd.DataFrame(index=raw.index)
    out["open"] = raw["open"]
    out["high"] = raw["high"]
    out["low"] = raw["low"]
    out["close"] = raw["adj_close"]
    out["volume"] = raw["volume"]
    return out


def load_prices(ticker: str, force_refresh: bool = False) -> pd.DataFrame:
    path = _cache_path(ticker)
    if os.path.exists(path) and not force_refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = _download_from_alpha_vantage(ticker)
    df.to_csv(path)
    time.sleep(1)
    return df


prices = load_prices(TICKER)


## 3. Feature engineering
def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def _bollinger_percent_b(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return (close - lower) / (upper - lower + 1e-12)


def _atr_pct(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean() / df["close"]


def _stochastic_k(df: pd.DataFrame, window: int = 14) -> pd.Series:
    lowest_low = df["low"].rolling(window).min()
    highest_high = df["high"].rolling(window).max()
    return (df["close"] - lowest_low) / (highest_high - lowest_low + 1e-12) * 100


def build_feature_block(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    ret = df["close"].pct_change()
    out["ret_1w"] = ret

    for lag in range(1, 6):
        out[f"ret_lag_{lag}"] = ret.shift(lag)

    for window in (5, 10, 20):
        out[f"mom_{window}w"] = df["close"].pct_change(window)

    for window in (10, 20, 50):
        sma = df["close"].rolling(window).mean()
        out[f"close_over_sma_{window}"] = df["close"] / sma

    for window in config.VOL_WINDOWS:
        out[f"vol_{window}w"] = ret.rolling(window).std()

    out["rsi_14"] = _rsi(df["close"], config.RSI_WINDOW)

    macd_line, macd_signal, macd_hist = _macd(df["close"])
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist

    out["bollinger_pct_b"] = _bollinger_percent_b(df["close"])
    out["atr_pct"] = _atr_pct(df)
    out["stochastic_k"] = _stochastic_k(df)
    out["ret_skew_20w"] = ret.rolling(20).skew()

    vol_avg = df["volume"].rolling(config.REL_VOLUME_WINDOW).mean()
    out["rel_volume"] = df["volume"] / vol_avg

    month = df.index.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    return out


def make_horizon_dataset(feature_block: pd.DataFrame, df: pd.DataFrame, horizon: int, keep_latest_incomplete: bool = False) -> pd.DataFrame:
    out = feature_block.copy()
    out["target"] = df["close"].shift(-horizon) / df["close"] - 1
    if keep_latest_incomplete:
        out = out.dropna(subset=[c for c in out.columns if c != "target"])
    else:
        out = out.dropna()
    return out


def feature_columns(feats: pd.DataFrame) -> list:
    return [c for c in feats.columns if c != "target"]


feature_block = build_feature_block(prices)
tuning_set = make_horizon_dataset(feature_block, prices, horizon=1)
cols = feature_columns(tuning_set)


## 4. Hyperparameter tuning (once, reused across horizons)
n_train = int(len(tuning_set) * config.TRAIN_FRACTION)
X_tune_raw = tuning_set[cols].values[:n_train]
y_tune = tuning_set["target"].values[:n_train]

tune_scaler = StandardScaler().fit(X_tune_raw)
X_tune = tune_scaler.transform(X_tune_raw)

MODEL_SPECS = {
    "ridge": (
        Ridge(),
        {"alpha": [0.1, 1, 5, 10, 20, 50, 100]},
    ),
    "random_forest": (
        RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=-1),
        {
            "n_estimators": [200, 300, 400],
            "max_depth": [3, 4, 5, 6],
            "min_samples_leaf": [10, 15, 25, 35],
            "max_features": ["sqrt", "log2", 0.5],
        },
    ),
    "hist_gb": (
        HistGradientBoostingRegressor(random_state=config.RANDOM_STATE),
        {
            "max_iter": [100, 200, 300],
            "max_depth": [2, 3, 4, None],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "l2_regularization": [0, 0.1, 1, 5],
        },
    ),
    "xgboost": (
        XGBRegressor(random_state=config.RANDOM_STATE, n_jobs=-1, objective="reg:squarederror"),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_lambda": [0.1, 1, 5],
        },
    ),
}


def tune_model(estimator, param_dist, X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        estimator, param_dist, n_iter=config.TUNE_ITER, cv=tscv,
        scoring="neg_root_mean_squared_error", random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_


tuned_models = {
    name: tune_model(estimator, param_dist, X_tune, y_tune)
    for name, (estimator, param_dist) in MODEL_SPECS.items()
}


## 5. Walk-forward backtest and forecast, per horizon
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    dir_acc = float((np.sign(y_true) == np.sign(y_pred)).mean())
    ic, _ = spearmanr(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "DirAcc": dir_acc, "IC": float(ic) if ic is not None else np.nan}


def walk_forward_backtest(feat_df: pd.DataFrame, cols: list, models: dict):
    X = feat_df[cols].values
    y = feat_df["target"].values
    idx = feat_df.index
    n = len(feat_df)
    pos = int(n * config.BACKTEST_INITIAL_FRACTION)

    pred_dates, actuals = [], []
    preds_by_model = {name: [] for name in models}

    while pos < n:
        end = min(pos + config.BACKTEST_STEP_WEEKS, n)
        X_train_raw, y_train = X[:pos], y[:pos]
        X_test_raw, y_test = X[pos:end], y[pos:end]

        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        for name, model in models.items():
            m = clone(model)
            m.fit(X_train, y_train)
            preds_by_model[name].extend(m.predict(X_test).tolist())

        actuals.extend(y_test.tolist())
        pred_dates.extend(idx[pos:end].tolist())
        pos = end

    return pd.DatetimeIndex(pred_dates), np.array(actuals), {k: np.array(v) for k, v in preds_by_model.items()}


last_price_date = prices.index[-1]
last_close = float(prices["close"].iloc[-1])

horizon_results = {}
forecast_rows = []

for h in config.HORIZONS:
    dataset_h = make_horizon_dataset(feature_block, prices, horizon=h)
    bt_dates, bt_actual, bt_preds = walk_forward_backtest(dataset_h, cols, tuned_models)

    bt_preds["ensemble"] = np.mean(np.vstack([bt_preds[name] for name in tuned_models]), axis=0)
    bt_preds["naive_baseline"] = prices["close"].pct_change(h).reindex(bt_dates).values

    metrics_h = pd.DataFrame(
        [{"model": name, **evaluate(bt_actual, preds)} for name, preds in bt_preds.items()]
    ).set_index("model").sort_values("RMSE")

    candidates = metrics_h.drop(index="naive_baseline")
    best_name = candidates.index[0]
    residual_std = float((bt_actual - bt_preds[best_name]).std())

    dataset_all = make_horizon_dataset(feature_block, prices, horizon=h)
    dataset_live = make_horizon_dataset(feature_block, prices, horizon=h, keep_latest_incomplete=True)

    X_all_raw = dataset_all[cols].values
    y_all = dataset_all["target"].values
    final_scaler = StandardScaler().fit(X_all_raw)
    X_all = final_scaler.transform(X_all_raw)
    X_latest = final_scaler.transform(dataset_live.iloc[[-1]][cols].values)

    final_preds = {}
    for name, model in tuned_models.items():
        final_model = clone(model)
        final_model.fit(X_all, y_all)
        final_preds[name] = float(final_model.predict(X_latest)[0])
    final_preds["ensemble"] = float(np.mean(list(final_preds.values())))

    predicted_return = final_preds[best_name]
    forecast_date = last_price_date + pd.Timedelta(weeks=h)
    predicted_price = last_close * (1 + predicted_return)
    price_low = last_close * (1 + predicted_return - residual_std)
    price_high = last_close * (1 + predicted_return + residual_std)
    dollar_value = 1 * (1 + predicted_return)
    dollar_low = 1 * (1 + predicted_return - residual_std)
    dollar_high = 1 * (1 + predicted_return + residual_std)
    dir_acc_pct = metrics_h.loc[best_name, "DirAcc"] * 100

    horizon_results[h] = {
        "bt_dates": bt_dates, "bt_actual": bt_actual, "bt_preds": bt_preds,
        "metrics": metrics_h, "best_name": best_name,
    }

    forecast_rows.append({
        "Weeks ahead": h,
        "Estimated date": forecast_date.strftime(config.DATE_FMT),
        "Estimated price ($)": round(predicted_price, 2),
        "Likely range ($)": f"{price_low:.2f} - {price_high:.2f}",
        "Value of $1 invested today ($)": round(dollar_value, 4),
        "$1 likely range ($)": f"{dollar_low:.4f} - {dollar_high:.4f}",
        "How often right before (%)": round(dir_acc_pct, 1),
        "Method (see README)": best_name,
    })

forecast_df = pd.DataFrame(forecast_rows).set_index("Weeks ahead")


## 6. Plain-English summary
print("=" * 62)
print(f"{TICKER} forecast summary")
print("=" * 62)
print(f"Data range:   {prices.index.min().strftime(config.DATE_FMT)} to {prices.index.max().strftime(config.DATE_FMT)}  ({len(prices)} weekly bars)")
print(f"Last price:   {last_close:.2f} on {last_price_date.strftime(config.DATE_FMT)}")
print("=" * 62)
print()

for row in forecast_rows:
    low, high = row["Likely range ($)"].split(" - ")
    print(
        f"In {row['Weeks ahead']} week(s), around {row['Estimated date']}: "
        f"{TICKER} could be worth about ${row['Estimated price ($)']:.2f} a share "
        f"(likely between ${float(low):.2f} and ${float(high):.2f}). "
        f"Every $1 invested today could become about ${row['Value of $1 invested today ($)']:.2f}. "
        f"This method has gotten the direction right about {row['How often right before (%)']:.1f}% "
        f"of the time in similar past weeks."
    )

print()
print("Note: these are statistical estimates based on historical patterns.")
print("Actual results will vary. This is not financial advice.")
print("=" * 62)

forecast_df


## 7. Chart 1: actual vs predicted (1-week horizon, walk-forward, last N years)
print(f"\nChart 1 shows what actually happened each week vs. what each method would have")
print(f"predicted, over the last {config.DISPLAY_YEARS} years. Click a name in the legend to show or hide it.\n")

h1 = horizon_results[1]
cutoff = h1["bt_dates"].max() - pd.DateOffset(years=config.DISPLAY_YEARS)
mask = h1["bt_dates"] >= cutoff
display_dates = h1["bt_dates"][mask]
best_h1 = h1["best_name"]

trace_labels = {
    "ridge": "Ridge (predicted)",
    "random_forest": "Random Forest (predicted)",
    "hist_gb": "HistGradientBoosting (predicted)",
    "xgboost": "XGBoost (predicted)",
    "ensemble": "Ensemble (predicted)",
    "naive_baseline": "Naive baseline",
}

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=display_dates, y=h1["bt_actual"][mask],
    name="Actual return", mode="lines", line=dict(color="black", width=2),
))

for name, label in trace_labels.items():
    visible = True if name in (best_h1, "naive_baseline") else "legendonly"
    fig1.add_trace(go.Scatter(
        x=display_dates, y=h1["bt_preds"][name][mask],
        name=label, mode="lines", visible=visible, opacity=0.8,
    ))

fig1.update_layout(
    title=f"{TICKER}: actual vs predicted weekly return (out-of-sample, last {config.DISPLAY_YEARS}y)",
    xaxis_title="Date", yaxis_title="Weekly return",
    hovermode="x unified",
    legend_title_text="Click to show/hide",
)
fig1.update_xaxes(tickformat="%d-%m-%Y", showspikes=True, spikemode="across")
fig1.update_yaxes(tickformat=".2%", showspikes=True)
fig1.show()


## 8. Chart 2: value of $1 invested over time (last N years)
print(f"\nChart 2 shows what $1 would have grown into if you had followed each method's")
print(f"buy/hold-cash signal each week, compared with simply buying and holding.\n")

buy_hold_curve = np.cumprod(1 + h1["bt_actual"][mask])

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=display_dates, y=buy_hold_curve,
    name="Buy and hold", mode="lines", line=dict(color="black", width=2),
))

for name, label in trace_labels.items():
    if name == "naive_baseline":
        continue
    position = (h1["bt_preds"][name][mask] > 0).astype(float)
    strategy_ret = position * h1["bt_actual"][mask]
    curve = np.cumprod(1 + strategy_ret)
    visible = True if name == best_h1 else "legendonly"
    fig2.add_trace(go.Scatter(
        x=display_dates, y=curve,
        name=f"{label.replace(' (predicted)', '')} strategy", mode="lines", visible=visible, opacity=0.8,
    ))

fig2.update_layout(
    title=f"{TICKER}: growth of $1, model strategy vs buy and hold (last {config.DISPLAY_YEARS}y)",
    xaxis_title="Date", yaxis_title="Value of $1",
    hovermode="x unified",
    legend_title_text="Click to show/hide",
)
fig2.update_xaxes(tickformat="%d-%m-%Y", showspikes=True, spikemode="across")
fig2.update_yaxes(showspikes=True)
fig2.show()


## 9. Advanced detail (optional, technical) — see README.md for a glossary
print("\nAdvanced detail below is optional. See README.md for what each term means.\n")
for h in config.HORIZONS:
    print(f"--- {h}-week horizon: model comparison ---")
    print(horizon_results[h]["metrics"])
    print()
