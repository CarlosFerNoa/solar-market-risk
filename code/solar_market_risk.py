from pathlib import Path
# Provides an object-oriented interface for filesystem paths.
# Used to manage file paths and directories in a clean, cross-platform way.

import pandas as pd
# Core library for data manipulation and analysis.
# Used to handle time series, DataFrames, indexing, grouping, and aggregation.

import numpy as np
# Fundamental numerical computing library.
# Used for array operations, mathematical functions, and efficient numerical routines.

import yfinance as yf
# Interface to Yahoo Finance.
# Used to download historical market data such as prices and indices.

from scipy import stats
# Statistical functions and distributions.
# Used here for correlation analysis (e.g., Pearson and Spearman correlations).

import statsmodels.api as sm
# Statistical modeling library.
# Used to estimate regression models (OLS) and compute robust inference
# such as HAC / Newey–West standard errors.

import matplotlib.pyplot as plt
# Plotting library.
# Used for basic visualization, such as residual diagnostics and time series plots.

from typing import Optional
# Type hinting utilities.
# Used to make function signatures clearer and improve code readability.


# ============================================================
# Import dataset (SILSO/SIDC Sunspot Number)
# ============================================================

def fetch_monthly_sunspot_number(
    url: str = "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv",
    use_month_end_timestamp: bool = True,
    cache_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch the monthly mean total Sunspot Number (SN) from SILSO/SIDC (Version 2.0).

    Data source:
      - File: SN_m_tot_V2.0.csv (semicolon-separated)
      - Contents (per row):
          year, month, decimal_year, sn_value, sn_std, n_obs, is_definitive
        where:
          - sn_value == -1 indicates missing value
          - is_definitive: 1 definitive, 0 provisional

    Parameters
    ----------
    url : str
        Direct CSV URL for the monthly mean total sunspot number.
    use_month_end_timestamp : bool
        If True, timestamps are set to the month end (e.g., 2020-01-31).
        If False, timestamps are set to the month start (e.g., 2020-01-01).
    cache_csv_path : Optional[str]
        If provided, saves the raw downloaded CSV to this path, and will reuse it
        if the file already exists.

    Returns
    -------
    pd.DataFrame
        Indexed by datetime (monthly). Columns:
          - sunspot_number (float)
          - sunspot_std (float)
          - n_obs (int)
          - is_definitive (int: 0/1)
          - decimal_year (float)
    """
    if cache_csv_path is not None:
        cache_path = Path(cache_csv_path)
        if cache_path.exists():
            df_raw = pd.read_csv(cache_path, sep=";", header=None)
        else:
            df_raw = pd.read_csv(url, sep=";", header=None)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df_raw.to_csv(cache_path, index=False, header=False)
    else:
        df_raw = pd.read_csv(url, sep=";", header=None)

    df_raw = df_raw.rename(
        columns={
            0: "year",
            1: "month",
            2: "decimal_year",
            3: "sunspot_number",
            4: "sunspot_std",
            5: "n_obs",
            6: "is_definitive",
        }
    )

    # Enforce dtypes and clean missing markers
    df_raw["year"] = df_raw["year"].astype("int32")
    df_raw["month"] = df_raw["month"].astype("int32")
    df_raw["decimal_year"] = df_raw["decimal_year"].astype("float64")

    df_raw["sunspot_number"] = df_raw["sunspot_number"].astype("float64")
    df_raw["sunspot_std"] = df_raw["sunspot_std"].astype("float64")
    df_raw["n_obs"] = df_raw["n_obs"].astype("int32")
    df_raw["is_definitive"] = df_raw["is_definitive"].astype("int8")

    # -1 indicates missing values in SILSO files
    df_raw["sunspot_number"] = df_raw["sunspot_number"].replace(-1.0, np.nan)

    # Build timestamp
    periods = pd.PeriodIndex.from_fields(
        year=df_raw["year"].to_numpy(),
        month=df_raw["month"].to_numpy(),
        freq="M",
    )

    if use_month_end_timestamp:
        ts = periods.to_timestamp(how="end")
    else:
        ts = periods.to_timestamp(how="start")

    df = df_raw.drop(columns=["year", "month"]).copy()
    df.index = pd.DatetimeIndex(ts, name="date").normalize()


    return df.sort_index()



# ============================================================
# Import dataset (Yahoo Finance)
# ============================================================

def download_market_data_yahoo(
    ticker: str,
    start: str = "1900-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download daily market data from Yahoo Finance and return a standardized
    DataFrame with a DatetimeIndex and canonical columns.

    Returns columns:
        ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    """
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned by Yahoo Finance for ticker='{ticker}'.")

    # If MultiIndex columns appear, reduce to the requested ticker level.
    if isinstance(df.columns, pd.MultiIndex):
        # Common yfinance layout: columns like ('Close', 'SPY') or ('SPY', 'Close')
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        elif ticker in df.columns.get_level_values(0):
            df = df.xs(ticker, axis=1, level=0)
        else:
            # Fallback: if only one ticker was downloaded, squeeze the first level
            df.columns = df.columns.get_level_values(-1)

    # Standardize column names
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    # Keep only canonical columns (some tickers may not have all)
    canonical_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    existing_cols = [c for c in canonical_cols if c in df.columns]
    df = df[existing_cols].copy()

    # Enforce DatetimeIndex and sorting
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df

def build_monthly_risk_solar_dataframe(
    df_sunspot,
    df_market,
    look_ahead: bool
):
    """
    Build a monthly DataFrame combining market risk and solar activity.

    This function aligns monthly solar activity data with daily market prices,
    computes a monthly market risk metric based on realized volatility, and
    returns a clean monthly dataset suitable for correlation or regression
    analysis.

    The function supports two alignment modes:
      - Predictive (anti-lookahead): solar activity from month M is associated
        with market risk in month M+1.
      - Contemporaneous (retrospective): solar activity and market risk are
        aligned within the same month.

    Parameters
    ----------
    df_sunspot : pd.DataFrame
        DataFrame containing monthly solar activity data.
        Must have a DatetimeIndex (month-end timestamps) and a column
        named 'sunspot_number'.

    df_market : pd.DataFrame
        DataFrame containing daily market price data.
        Must have a DatetimeIndex (daily frequency) and a column
        named 'adj_close'.

    look_ahead : bool
        Controls the temporal alignment of solar activity:
        - If True, solar activity from the SAME month is used
          (contemporaneous / retrospective analysis).
        - If False, solar activity from the PREVIOUS month is used
          (anti-lookahead / predictive setup).

    Returns
    -------
    df_monthly : pd.DataFrame
        Monthly DataFrame indexed by month-end timestamps with the following columns:

        - 'monthly_risk':
            Realized market risk measured as the within-month standard deviation
            of daily log-returns (ddof = 0).

        - 'solar_feature':
            Monthly solar activity aligned according to the chosen look-ahead
            configuration.

        All rows containing missing values are removed.
    """
    # ============================================================
    # 1) Align monthly solar data to daily market dates
    # ============================================================

    s_monthly = df_sunspot["sunspot_number"].copy()
    s_monthly_period = s_monthly.copy()
    s_monthly_period.index = s_monthly_period.index.to_period("M")

    market_month = df_market.index.to_period("M")
    solar_month_for_day = market_month if look_ahead else (market_month - 1)

    s_solar_daily = pd.Series(
        s_monthly_period.reindex(solar_month_for_day).to_numpy(),
        index=df_market.index,
        name="solar_feature"
    )

    # ============================================================
    # 2) Build monthly risk metric from daily market data
    # ============================================================

    prices = df_market["adj_close"].astype(np.float64).to_numpy()

    log_returns = np.empty(prices.shape[0], dtype=np.float64)
    log_returns[:] = np.nan
    log_returns[1:] = np.log(prices[1:] / prices[:-1])

    log_returns = pd.Series(
        log_returns,
        index=df_market.index,
        name="log_return"
    )

    realized_vol = log_returns.groupby(
        df_market.index.to_period("M")
    ).std(ddof=0)

    solar_monthly = s_solar_daily.groupby(
        df_market.index.to_period("M")
    ).first()

    # ============================================================
    # 3) Final monthly dataset
    # ============================================================

    df_monthly = pd.DataFrame(
        {
            "monthly_risk": realized_vol,
            "solar_feature": solar_monthly,
        }
    )

    df_monthly.index = df_monthly.index.to_timestamp("M")
    df_monthly = df_monthly.dropna()

    return df_monthly




def print_correlation_analysis(df_monthly):
    """
    Print correlation metrics between solar activity and monthly market risk.

    This function computes and prints:
      - Pearson correlation (linear association)
      - Spearman correlation (rank-based, monotonic association)

    Parameters
    ----------
    df_monthly : pd.DataFrame
        Monthly DataFrame containing at least the columns:
        - 'solar_feature'
        - 'monthly_risk'
    """
    x = df_monthly["solar_feature"].to_numpy(dtype=np.float64)
    y = df_monthly["monthly_risk"].to_numpy(dtype=np.float64)

    # Pearson correlation (linear association)
    pearson_corr, pearson_pval = stats.pearsonr(x, y)

    # Spearman correlation (rank-based, monotonic association)
    spearman_corr, spearman_pval = stats.spearmanr(x, y)

    print("Correlation analysis:")
    print(f"Pearson correlation  : {pearson_corr:.6g} (p-value = {pearson_pval:.3g})")
    print(f"Spearman correlation : {spearman_corr:.6g} (p-value = {spearman_pval:.3g})")


def run_ols_baseline_monthly_risk_vs_solar(
    df_monthly,
    hac_lags: int = 12,
    plot_residuals: bool = True,
    titulo_residuos: str = "OLS residuals (HAC fit)"
):
    """
    Fit an OLS baseline model: monthly_risk ~ solar_feature, and report both
    classical and HAC (Newey–West) standard errors.

    Parameters
    ----------
    df_monthly : pd.DataFrame
        Monthly DataFrame with columns:
        - 'solar_feature'
        - 'monthly_risk'
        And a DatetimeIndex (month-end timestamps).

    hac_lags : int, default=12
        Number of lags for HAC (Newey–West) robust covariance.

    plot_residuals : bool, default=True
        If True, plot HAC residuals over time.

    titulo_residuos : str, default="OLS residuals (HAC fit)"
        Title for the residual plot.

    Returns
    -------
    out : dict
        Dictionary containing fitted results and key metrics:
        - 'ols'
        - 'ols_hac'
        - 'beta_solar'
        - 't_solar'
        - 'p_solar'
        - 'r2'
    """
    # --- 1) Prepare X and y ---
    x = df_monthly["solar_feature"].to_numpy(dtype=np.float64)
    y = df_monthly["monthly_risk"].to_numpy(dtype=np.float64)

    X = sm.add_constant(x, has_constant="add")  # [const, solar_feature]

    # --- 2) Fit OLS ---
    ols = sm.OLS(y, X, missing="drop").fit()

    # --- 3) Fit OLS with HAC (Newey–West) standard errors ---
    ols_hac = ols.get_robustcov_results(cov_type="HAC", maxlags=hac_lags)

    print("=== OLS (classical SE) ===")
    print(ols.summary())

    print("\n=== OLS (HAC / Newey–West SE) ===")
    print(ols_hac.summary())

    # --- 4) Quick residual diagnostics plot ---
    if plot_residuals:
        resid = ols_hac.resid
        fig = plt.figure(figsize=(10, 4))
        plt.plot(df_monthly.index, resid)
        plt.title(titulo_residuos)
        plt.xlabel("Date")
        plt.ylabel("Residual")
        plt.show()

    # --- 5) Report key numbers cleanly ---
    beta = float(ols_hac.params[1])
    tval = float(ols_hac.tvalues[1])
    pval = float(ols_hac.pvalues[1])
    r2   = float(ols.rsquared)

    print("\nKey results (HAC):")
    print(f"beta_solar = {beta:.6g}")
    print(f"t-stat     = {tval:.3f}")
    print(f"p-value    = {pval:.3g}")
    print(f"R^2 (OLS)  = {r2:.6g}")

    return {
        "ols": ols,
        "ols_hac": ols_hac,
        "beta_solar": beta,
        "t_solar": tval,
        "p_solar": pval,
        "r2": r2,
    }



def ols_stability_table_three_periods(
    df_monthly,
    hac_lags: int = 12
):
    """
    Build a compact stability table by splitting the full monthly dataset into
    three consecutive subperiods and running the same OLS + HAC estimation in
    each subperiod.

    This function is designed for robustness / stability checks:
      - No verbose statsmodels summaries
      - No plots
      - Returns a small DataFrame with key metrics per period

    Parameters
    ----------
    df_monthly : pd.DataFrame
        Monthly DataFrame with columns:
        - 'solar_feature'
        - 'monthly_risk'
        And a DatetimeIndex (month-end timestamps).

    hac_lags : int, default=12
        Number of lags for HAC (Newey–West) robust covariance.

    Returns
    -------
    df_out : pd.DataFrame
        Compact table with one row per subperiod and columns:
        - 'n_obs'
        - 'start'
        - 'end'
        - 'beta_solar'
        - 't_solar'
        - 'p_solar'
        - 'r2'
    """
    n = len(df_monthly)
    n1 = n // 3
    n2 = 2 * n // 3

    df_p1 = df_monthly.iloc[:n1]
    df_p2 = df_monthly.iloc[n1:n2]
    df_p3 = df_monthly.iloc[n2:]

    def _rodar_ols_hac(df_sub):
        x = df_sub["solar_feature"].to_numpy(dtype=np.float64)
        y = df_sub["monthly_risk"].to_numpy(dtype=np.float64)

        X = sm.add_constant(x, has_constant="add")
        ols = sm.OLS(y, X, missing="drop").fit()
        ols_hac = ols.get_robustcov_results(cov_type="HAC", maxlags=hac_lags)

        beta = float(ols_hac.params[1])
        tval = float(ols_hac.tvalues[1])
        pval = float(ols_hac.pvalues[1])
        r2   = float(ols.rsquared)

        return beta, tval, pval, r2

    beta1, t1, p1, r21 = _rodar_ols_hac(df_p1)
    beta2, t2, p2, r22 = _rodar_ols_hac(df_p2)
    beta3, t3, p3, r23 = _rodar_ols_hac(df_p3)

    df_out = pd.DataFrame(
        {
            "period": ["P1", "P2", "P3"],
            "n_obs": [len(df_p1), len(df_p2), len(df_p3)],
            "start": [df_p1.index.min(), df_p2.index.min(), df_p3.index.min()],
            "end":   [df_p1.index.max(), df_p2.index.max(), df_p3.index.max()],
            "beta_solar": [beta1, beta2, beta3],
            "t_solar":    [t1, t2, t3],
            "p_solar":    [p1, p2, p3],
            "r2":         [r21, r22, r23],
        }
    ).set_index("period")

    return df_out

