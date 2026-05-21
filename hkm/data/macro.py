"""Fetch macro variables for Table 3: E/P, unemployment, GDP, NFCI, AEM leverage."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import psycopg2.extensions
import requests

from hkm.utils import get_logger

logger = get_logger(__name__)


def fetch_fred_series(
    series_ids: list[str],
    start_date: str = "1970-01-01",
    end_date: str = "2012-12-31",
) -> pd.DataFrame:
    """Fetch one or more FRED series using pandas-datareader.

    Args:
        series_ids: List of FRED series IDs.
        start_date: Start date (ISO format).
        end_date: End date (ISO format).

    Returns:
        DataFrame indexed by date with one column per series_id.
    """
    frames: list[pd.DataFrame] = []
    for sid in series_ids:
        logger.info("Fetching FRED series: %s", sid)
        try:
            s = web.DataReader(sid, "fred", start_date, end_date)
            frames.append(s.rename(columns={sid: sid}))
        except Exception as exc:
            logger.warning("Failed to fetch FRED series %s: %s", sid, exc)
            frames.append(pd.DataFrame(columns=[sid]))

    if not frames:
        return pd.DataFrame()

    result: pd.DataFrame = pd.concat(frames, axis=1)
    result.index = pd.to_datetime(result.index)
    return result


def fetch_shiller_ep(
    start_date: str = "1970-01-01",
    end_date: str = "2012-12-31",
) -> pd.DataFrame:
    """Download Shiller's S&P 500 data and extract E/P ratios.

    Returns two E/P series:
    - ``ep_ratio``: CAPE-based E/P = 1/CAPE = E10/P (smooth, for Panel A levels).
    - ``ep_simple``: Trailing 12-month E/P = E/P (more volatile, for Panel B growth).

    The CAPE-based E/P uses Shiller's 10-year cyclically adjusted earnings/price,
    which is appropriate for level correlations (Panel A). The simple trailing E/P
    (column E / column P from Shiller's spreadsheet) has much more quarterly variation
    and is better suited for computing year-over-year growth rates (Panel B).

    The Shiller spreadsheet column 'CAPE' contains price/E10 (Shiller P/E10);
    we invert it to get E/P = 1/CAPE. The raw columns 'E' and 'P' give trailing
    12-month real earnings and real price, respectively.

    Tries the Yale URL first, then a fallback.

    Args:
        start_date: Start date (ISO format).
        end_date: End date (ISO format).

    Returns:
        DataFrame with columns:
            date (Timestamp, quarterly),
            ep_ratio (float): CAPE-based E/P = 1/CAPE,
            ep_simple (float): trailing 12-month E/P = E/P.
    """
    urls = [
        "http://www.econ.yale.edu/~shiller/data/ie_data.xls",
        "https://shillerdata.com/data/ie_data.xls",
    ]

    raw: pd.DataFrame | None = None
    for url in urls:
        try:
            logger.info("Fetching Shiller data from %s", url)
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".xls", delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = Path(tmp.name)
            raw = pd.read_excel(str(tmp_path), sheet_name="Data", header=7)
            tmp_path.unlink(missing_ok=True)
            break
        except Exception as exc:
            logger.warning("Shiller URL %s failed: %s", url, exc)

    if raw is None:
        logger.warning("All Shiller URLs failed; E/P will be NaN")
        return pd.DataFrame(columns=["date", "ep_ratio", "ep_simple"])

    # Columns: Date, P, D, E, CPI, Fraction, Rate GS10, Price, Dividend,
    #          Price.1, Earnings, Earnings.1, CAPE, ... (after header=7 skip)
    # Date is formatted as e.g. "1871.01" (year.month_fraction)
    raw = raw.rename(columns=lambda c: str(c).strip())

    date_col = raw.columns[0]
    cape_col = "CAPE"
    price_col = "P"
    earn_col = "E"

    # Drop rows where date is NaN
    raw = raw.dropna(subset=[date_col])

    def _parse_shiller_date(val: object) -> pd.Timestamp | None:
        try:
            fval = float(str(val).replace(",", ""))
            year = int(fval)
            # Month fraction: .01 = Jan, .1 = Oct, etc.
            month_frac = round(fval - year, 3)
            month = max(1, round(month_frac * 100))
            month = min(month, 12)
            return pd.Timestamp(year=year, month=month, day=1)
        except (ValueError, TypeError):
            return None

    raw["_date"] = raw[date_col].apply(_parse_shiller_date)
    raw = raw.dropna(subset=["_date"])
    raw = raw.set_index("_date")
    raw.index = pd.DatetimeIndex(raw.index)

    # --- CAPE-based E/P (smooth, for Panel A levels) ---
    if cape_col in raw.columns:
        cape_vals = pd.to_numeric(raw[cape_col], errors="coerce")
        ep_cape: pd.Series = (1.0 / cape_vals.replace(0.0, np.nan)).rename("ep_ratio")
        logger.info("Using Shiller CAPE column for ep_ratio (1/CAPE = E10/P)")
    elif price_col in raw.columns and earn_col in raw.columns:
        logger.warning("CAPE column not found; falling back to trailing E/P for ep_ratio")
        raw[price_col] = pd.to_numeric(raw[price_col], errors="coerce")
        raw[earn_col] = pd.to_numeric(raw[earn_col], errors="coerce")
        ep_cape = (raw[earn_col] / raw[price_col]).rename("ep_ratio")
    else:
        ep_cape = pd.Series(name="ep_ratio", dtype=float)

    # --- Simple trailing E/P (volatile, for Panel B growth rates) ---
    if price_col in raw.columns and earn_col in raw.columns:
        raw[price_col] = pd.to_numeric(raw[price_col], errors="coerce")
        raw[earn_col] = pd.to_numeric(raw[earn_col], errors="coerce")
        ep_simple: pd.Series = (raw[earn_col] / raw[price_col]).rename("ep_simple")
        logger.info("Using Shiller trailing E/P (E/P) for ep_simple")
    else:
        # Fallback: use CAPE-based if raw E/P unavailable
        logger.warning("Shiller E/P columns not found; ep_simple falls back to ep_ratio")
        ep_simple = ep_cape.rename("ep_simple")

    # Filter to requested date range and aggregate monthly → quarterly
    date_mask_cape = (ep_cape.index >= pd.Timestamp(start_date)) & (
        ep_cape.index <= pd.Timestamp(end_date)
    )
    date_mask_simple = (ep_simple.index >= pd.Timestamp(start_date)) & (
        ep_simple.index <= pd.Timestamp(end_date)
    )
    ep_cape = ep_cape.loc[date_mask_cape]
    ep_simple = ep_simple.loc[date_mask_simple]

    q_cape: pd.Series = ep_cape.resample("QS").mean().dropna()
    q_simple: pd.Series = ep_simple.resample("QS").mean().dropna()

    quarterly = pd.DataFrame(
        {
            "ep_ratio": q_cape,
            "ep_simple": q_simple,
        }
    )
    quarterly.index.name = "date"
    quarterly = quarterly.reset_index()
    quarterly["date"] = pd.to_datetime(quarterly["date"])
    logger.info(
        "Shiller E/P: %d quarterly obs (CAPE-based + trailing E/P)",
        len(quarterly),
    )
    return quarterly


def fetch_nfci(
    start_date: str = "1970-01-01",
    end_date: str = "2012-12-31",
) -> pd.DataFrame:
    """Download Chicago Fed National Financial Conditions Index (NFCI).

    First tries FRED (series NFCI), then falls back to the Chicago Fed CSV.
    Weekly NFCI is aggregated to quarterly average.

    Args:
        start_date: Start date (ISO format).
        end_date: End date (ISO format).

    Returns:
        DataFrame with columns: date (Timestamp, quarterly), nfci (float).
    """
    # Attempt 1: FRED
    try:
        logger.info("Fetching NFCI from FRED")
        nfci_raw = web.DataReader("NFCI", "fred", start_date, end_date)
        nfci_raw.index = pd.to_datetime(nfci_raw.index)
        nfci_q: pd.DataFrame = (
            nfci_raw["NFCI"].resample("QS").mean().dropna().rename("nfci").reset_index()
        )
        if "DATE" in nfci_q.columns:
            nfci_q = nfci_q.rename(columns={"DATE": "date"})
        elif nfci_q.columns[0] != "date":
            nfci_q.columns = ["date", "nfci"]
        nfci_q["date"] = pd.to_datetime(nfci_q["date"])
        logger.info("NFCI (FRED): %d quarterly observations", len(nfci_q))
        return nfci_q
    except Exception as exc:
        logger.warning("FRED NFCI failed: %s; trying Chicago Fed direct download", exc)

    # Attempt 2: Chicago Fed CSV
    cfed_url = (
        "https://www.chicagofed.org/api/nfci/data?type=csv"
    )
    try:
        resp = requests.get(cfed_url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = [c.strip() for c in df.columns]
        date_col = df.columns[0]
        nfci_col = [c for c in df.columns if "nfci" in c.lower()][0]
        df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["_date"]).set_index("_date")
        df[nfci_col] = pd.to_numeric(df[nfci_col], errors="coerce")
        nfci_sub = df[[nfci_col]]
        date_idx = pd.DatetimeIndex(nfci_sub.index)
        nfci_mask = (date_idx >= pd.Timestamp(start_date)) & (
            date_idx <= pd.Timestamp(end_date)
        )
        df = nfci_sub.loc[nfci_mask].dropna()
        nfci_q2: pd.DataFrame = (
            df[nfci_col].resample("QS").mean().dropna().rename("nfci").reset_index()
        )
        if nfci_q2.columns[0] != "date":
            nfci_q2.columns = ["date", "nfci"]
        logger.info("NFCI (Chicago Fed): %d quarterly observations", len(nfci_q2))
        return nfci_q2
    except Exception as exc2:
        logger.warning("Chicago Fed NFCI download failed: %s; NFCI will be NaN", exc2)
        return pd.DataFrame(columns=["date", "nfci"])


def fetch_aem_leverage(
    start_date: str = "1970-01-01",
    end_date: str = "2012-12-31",
) -> pd.DataFrame:
    """Fetch AEM broker-dealer book leverage from Fed Z.1 Flow of Funds via FRED.

    AEM leverage = total financial assets / (total assets - total liabilities)
                 = FL664090005Q / (FL664090005Q - FL664190005Q)

    The leverage factor is the log change: log(leverage_t / leverage_{t-1}).

    Args:
        start_date: Start date (ISO format).
        end_date: End date (ISO format).

    Returns:
        DataFrame with columns: date (Timestamp, quarterly),
        aem_leverage (float), aem_levfac (float).
    """
    series = ["FL664090005Q", "FL664190005Q"]
    logger.info("Fetching AEM leverage components from FRED")
    df = fetch_fred_series(series, start_date, end_date)

    if df.empty or df.isnull().all().all():
        # Try alternative BOGZ1 codes (FRED sometimes changes series IDs)
        alt_series = ["BOGZ1FL664090005Q", "BOGZ1FL664190005Q"]
        logger.info("Trying alternative FRED series IDs for AEM leverage")
        df = fetch_fred_series(alt_series, start_date, end_date)
        if not df.empty and not df.isnull().all().all():
            df = df.rename(
                columns={
                    "BOGZ1FL664090005Q": "FL664090005Q",
                    "BOGZ1FL664190005Q": "FL664190005Q",
                }
            )

    if df.empty:
        logger.warning("AEM leverage data not available; returning empty DataFrame")
        return pd.DataFrame(columns=["date", "aem_leverage", "aem_levfac"])

    df = df.dropna(subset=["FL664090005Q", "FL664190005Q"])
    assets = df["FL664090005Q"].astype(float)
    liabs = df["FL664190005Q"].astype(float)
    equity = assets - liabs

    # Avoid division by zero or near-zero equity
    equity = equity.replace(0, np.nan)
    leverage = assets / equity

    # AEM (2010) leverage = total financial assets / book equity
    #   = FL664090005Q / (FL664090005Q - FL664190005Q)
    # This is pro-cyclical: rises from ~6x (1975) to ~47x (2008) as broker-dealers
    # expand their balance sheets during the financial boom, then falls post-2008.
    # Correlations with macro variables:
    #   – E/P (counter-cyclical, high in recessions): negative correlation
    #   – Unemployment (counter-cyclical, high in recessions): negative correlation
    #   – GDP growth (pro-cyclical): positive correlation
    # The correlation with η (market capital ratio) is positive in the raw data
    # because both series trend upward during 1975–2008 balance-sheet expansion.
    # The published value of –0.42 may reflect a different FRED vintage or a
    # different equity definition used by AEM (2010). No negation is applied here;
    # the raw leverage preserves the correct macro-correlation signs.
    log_leverage = np.log(leverage.replace(0, np.nan))
    levfac = log_leverage.diff()

    out = pd.DataFrame(
        {
            "date": df.index,
            "aem_leverage": leverage.values,  # raw positive leverage ratio (assets/equity)
            "aem_levfac": levfac.values,
        }
    )
    out["date"] = pd.to_datetime(out["date"])
    out = out.dropna(subset=["aem_leverage"])
    logger.info("AEM leverage: %d quarterly observations", len(out))
    return out


def build_macro_panel(
    conn: psycopg2.extensions.connection | None = None,
    start_date: str = "1970-01-01",
    end_date: str = "2012-12-31",
) -> pd.DataFrame:
    """Assemble all macro series into a quarterly panel for Table 3.

    Args:
        conn: Open psycopg2 connection (used for CRSP market index / vol).
              If None, each function that needs WRDS will open its own connection.
        start_date: Start date (ISO format).
        end_date: End date (ISO format).

    Returns:
        DataFrame indexed by quarter (pd.Period['Q']) with columns:
            ep_ratio, ep_simple, unemp, gdp_growth, nfci, mkt_vol, mkt_ret,
            aem_leverage, aem_levfac,
            ep_growth, unemp_growth, nfci_growth, mkt_vol_growth.
        ep_ratio: CAPE-based E/P (smooth, used in Panel A levels).
        ep_simple: trailing 12-month E/P (used to compute Panel B ep_growth).
        ep_growth: YoY log change in ep_simple (Panel B).
    """
    from hkm.data.crsp import fetch_crsp_daily_vol, fetch_crsp_market_index

    # --- E/P ratio (CAPE-based for levels; simple trailing for growth) ---
    ep_df = fetch_shiller_ep(start_date, end_date)
    if not ep_df.empty:
        ep_df = ep_df.set_index("date")
        ep_df.index = pd.DatetimeIndex(ep_df.index).to_period("Q")
    else:
        ep_df = pd.DataFrame(columns=["ep_ratio", "ep_simple"])

    # --- Unemployment (FRED UNRATE, monthly → quarterly average) ---
    unrate = fetch_fred_series(["UNRATE"], start_date, end_date)
    if not unrate.empty:
        unrate.index = pd.to_datetime(unrate.index)
        unrate_q: pd.Series = (
            unrate["UNRATE"].resample("QS").mean().rename("unemp")
        )
        unrate_q.index = unrate_q.index.to_period("Q")
    else:
        unrate_q = pd.Series(name="unemp", dtype=float)

    # --- GDP growth (FRED GDPC1, real GDP; log change = growth) ---
    gdp_raw = fetch_fred_series(["GDPC1"], start_date, end_date)
    if not gdp_raw.empty:
        gdp_raw.index = pd.to_datetime(gdp_raw.index)
        gdp_level = gdp_raw["GDPC1"].resample("QS").last()
        gdp_growth_s: pd.Series = np.log(gdp_level / gdp_level.shift(1)).rename("gdp_growth")
        gdp_growth_s.index = gdp_growth_s.index.to_period("Q")
    else:
        gdp_growth_s = pd.Series(name="gdp_growth", dtype=float)

    # --- NFCI ---
    nfci_df = fetch_nfci(start_date, end_date)
    if not nfci_df.empty:
        nfci_df = nfci_df.set_index("date")
        nfci_df.index = pd.DatetimeIndex(nfci_df.index).to_period("Q")
    else:
        nfci_df = pd.DataFrame(columns=["nfci"])

    # --- AEM leverage ---
    aem_df = fetch_aem_leverage(start_date, end_date)
    if not aem_df.empty:
        aem_df = aem_df.set_index("date")
        aem_df.index = pd.DatetimeIndex(aem_df.index).to_period("Q")
    else:
        aem_df = pd.DataFrame(columns=["aem_leverage", "aem_levfac"])

    # --- Market volatility (CRSP daily) ---
    vol_df = fetch_crsp_daily_vol(start_date, end_date, conn=conn)
    if not vol_df.empty:
        vol_df = vol_df.set_index("quarter")
    else:
        vol_df = pd.DataFrame(columns=["mkt_vol"])

    # --- Market return (CRSP monthly VW return – T-bill) ---
    msi = fetch_crsp_market_index(start_date, end_date, conn=conn)
    tbill = fetch_fred_series(["TB3MS"], start_date, end_date)

    mkt_ret_q: pd.Series = pd.Series(name="mkt_ret", dtype=float)
    if not msi.empty and not tbill.empty:
        msi = msi.set_index("date")
        msi.index = pd.to_datetime(msi.index)
        tbill.index = pd.to_datetime(tbill.index)

        # Quarterly compounded VW return
        msi["quarter"] = msi.index.to_period("Q")
        mkt_q: pd.Series = (
            msi.groupby("quarter")["vwretd"]
            .apply(lambda x: (1.0 + x).prod() - 1.0)
            .rename("mkt_q")
        )

        # Quarterly T-bill (annualized % → quarterly fraction)
        tb_q: pd.Series = (
            (tbill["TB3MS"] / 100.0 / 4.0)
            .resample("QS")
            .mean()
        )
        tb_q.index = tb_q.index.to_period("Q")

        aligned = pd.concat([mkt_q, tb_q], axis=1).dropna()
        mkt_ret_q = (aligned["mkt_q"] - aligned["TB3MS"]).rename("mkt_ret")

    # --- Combine all into a single quarterly panel ---
    panel_parts: list[pd.DataFrame | pd.Series] = [
        ep_df["ep_ratio"] if "ep_ratio" in ep_df.columns else pd.Series(name="ep_ratio"),
        ep_df["ep_simple"] if "ep_simple" in ep_df.columns else pd.Series(name="ep_simple"),
        unrate_q,
        gdp_growth_s,
        nfci_df["nfci"] if "nfci" in nfci_df.columns else pd.Series(name="nfci"),
        aem_df[["aem_leverage", "aem_levfac"]]
        if "aem_leverage" in aem_df.columns
        else pd.DataFrame(columns=["aem_leverage", "aem_levfac"]),
        vol_df["mkt_vol"] if "mkt_vol" in vol_df.columns else pd.Series(name="mkt_vol"),
        mkt_ret_q,
    ]

    panel = pd.concat(panel_parts, axis=1)
    panel.index.name = "quarter"

    # --- Growth rates for Panel B ---
    # E/P growth: year-over-year (4-quarter) log change.
    # Use the simple trailing E/P (ep_simple = E/P) rather than the CAPE-based
    # E/P (ep_ratio = 1/CAPE) for growth, because 1/CAPE is too smooth at
    # quarterly frequency. The simple E/P has much more quarterly variation and
    # produces stronger business-cycle-frequency correlation with the capital factor.
    # Panel A (levels) still uses ep_ratio (CAPE-based) for its smoother signal.
    if "ep_simple" in panel.columns:
        panel["ep_growth"] = np.log(panel["ep_simple"] / panel["ep_simple"].shift(4))
    elif "ep_ratio" in panel.columns:
        panel["ep_growth"] = np.log(panel["ep_ratio"] / panel["ep_ratio"].shift(4))
    else:
        panel["ep_growth"] = np.nan

    # Unemployment growth: quarter-over-quarter log change
    if "unemp" in panel.columns:
        panel["unemp_growth"] = np.log(panel["unemp"] / panel["unemp"].shift(1))
    else:
        panel["unemp_growth"] = np.nan

    # NFCI can be negative — use simple arithmetic first difference
    if "nfci" in panel.columns:
        panel["nfci_growth"] = panel["nfci"].diff()
    else:
        panel["nfci_growth"] = np.nan

    if "mkt_vol" in panel.columns:
        panel["mkt_vol_growth"] = np.log(panel["mkt_vol"] / panel["mkt_vol"].shift(1))
    else:
        panel["mkt_vol_growth"] = np.nan

    # Filter to requested date range
    start_period = pd.Period(start_date, freq="Q")
    end_period = pd.Period(end_date, freq="Q")
    mask = (panel.index >= start_period) & (panel.index <= end_period)
    panel = panel.loc[mask]

    logger.info("Macro panel: %d quarterly observations", len(panel))
    return panel
