"""Compute Table 3: pairwise time-series correlations of capital ratios and macro variables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import psycopg2.extensions

from hkm.data.intermediary import build_capital_factor, build_capital_ratio
from hkm.data.macro import build_macro_panel, fetch_aem_leverage
from hkm.utils import get_logger, wrds_connection

logger = get_logger(__name__)

# Row labels for Panel A (levels)
_PANEL_A_ROWS = [
    "Market capital",
    "Book capital",
    "AEM leverage",
    "E/P",
    "Unemployment",
    "GDP",
    "Financial conditions",
    "Market volatility",
]

# Row labels for Panel B (factors / growth rates)
_PANEL_B_ROWS = [
    "Market capital factor",
    "Book capital factor",
    "AEM leverage factor",
    "Market excess return",
    "E/P growth",
    "Unemployment growth",
    "GDP growth",
    "Financial conditions growth",
    "Market volatility growth",
]

# Column labels for both panels
_PANEL_A_COLS = ["Market capital", "Book capital", "AEM leverage"]
_PANEL_B_COLS = ["Market capital factor", "Book capital factor", "AEM leverage factor"]


def compute_table3(
    conn: psycopg2.extensions.connection | None = None,
    start_date: str = "1970-01-01",
    end_date: str = "2012-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Table 3: pairwise time-series correlations.

    Args:
        conn: Open psycopg2 WRDS connection, or None to open one internally.
        start_date: Start date for sample (ISO format, default 1970-01-01).
        end_date: End date for sample (ISO format, default 2012-12-31).

    Returns:
        (panel_a, panel_b): Tuple of two DataFrames.

        panel_a: shape (8, 3), correlations of levels.
            Index: Market capital, Book capital, AEM leverage, E/P, Unemployment,
                   GDP, Financial conditions, Market volatility.
            Columns: Market capital, Book capital, AEM leverage.

        panel_b: shape (9, 3), correlations of factors/growth rates.
            Index: Market capital factor, Book capital factor, AEM leverage factor,
                   Market excess return, E/P growth, Unemployment growth, GDP growth,
                   Financial conditions growth, Market volatility growth.
            Columns: Market capital factor, Book capital factor, AEM leverage factor.
    """

    def _run(c: psycopg2.extensions.connection) -> tuple[pd.DataFrame, pd.DataFrame]:
        return _compute_table3_with_conn(c, start_date, end_date)

    if conn is not None:
        return _run(conn)
    with wrds_connection() as c:
        return _run(c)


def _compute_table3_with_conn(
    conn: psycopg2.extensions.connection,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Internal implementation of Table 3 with an open connection."""
    logger.info("Computing Table 3 (%s – %s)", start_date, end_date)

    # ---- Step 1: Build quarterly capital ratios ----
    logger.info("Building quarterly capital ratio η_t")
    cap_df = build_capital_ratio(conn=conn, frequency="Q", start_date=start_date, end_date=end_date)

    # Convert to quarterly period index
    cap_df.index = cap_df.index.to_period("Q")
    eta = cap_df["eta"].dropna()
    book_cap = cap_df["book_capital"].dropna()

    # ---- Step 2: Build capital factors ----
    logger.info("Building capital ratio factors")
    mkt_factor = build_capital_factor(eta, frequency="Q")
    book_factor = build_capital_factor(book_cap, frequency="Q")

    # ---- Step 3: Fetch AEM leverage ----
    logger.info("Fetching AEM leverage")
    aem_df = fetch_aem_leverage(start_date, end_date)
    if not aem_df.empty:
        aem_df = aem_df.set_index("date")
        aem_df.index = pd.DatetimeIndex(aem_df.index).to_period("Q")
        aem_leverage = aem_df["aem_leverage"]
        aem_levfac = aem_df["aem_levfac"]
    else:
        aem_leverage = pd.Series(name="aem_leverage", dtype=float)
        aem_levfac = pd.Series(name="aem_levfac", dtype=float)

    # ---- Step 4: Build macro panel ----
    logger.info("Building macro panel")
    macro = build_macro_panel(conn=conn, start_date=start_date, end_date=end_date)

    # ---- Step 5: Align on common quarterly dates ----
    def _get(col: str) -> pd.Series[float]:
        """Return macro column as Series, or empty Series if missing."""
        return macro[col] if col in macro.columns else pd.Series(dtype=float)

    # Panel A: levels — use gdp_growth for "GDP" (paper footnote confirms this)
    panel_a_inputs = pd.DataFrame(
        {
            "Market capital": eta,
            "Book capital": book_cap,
            "AEM leverage": aem_leverage,
            "E/P": _get("ep_ratio"),
            "Unemployment": _get("unemp"),
            "GDP": _get("gdp_growth"),
            "Financial conditions": _get("nfci"),
            "Market volatility": _get("mkt_vol"),
        }
    )

    # Panel B: factors / growth rates
    panel_b_inputs = pd.DataFrame(
        {
            "Market capital factor": mkt_factor,
            "Book capital factor": book_factor,
            "AEM leverage factor": aem_levfac,
            "Market excess return": _get("mkt_ret"),
            "E/P growth": _get("ep_growth"),
            "Unemployment growth": _get("unemp_growth"),
            "GDP growth": _get("gdp_growth"),
            "Financial conditions growth": _get("nfci_growth"),
            "Market volatility growth": _get("mkt_vol_growth"),
        }
    )

    # Filter to requested date range
    start_q = pd.Period(start_date, freq="Q")
    end_q = pd.Period(end_date, freq="Q")

    for df in (panel_a_inputs, panel_b_inputs):
        mask = (df.index >= start_q) & (df.index <= end_q)
        df.loc[~mask, :] = np.nan

    # ---- Step 6: Compute pairwise correlations ----
    panel_a = _compute_correlations(panel_a_inputs, _PANEL_A_ROWS, _PANEL_A_COLS)
    panel_b = _compute_correlations(panel_b_inputs, _PANEL_B_ROWS, _PANEL_B_COLS)

    n_obs_a = panel_a_inputs[_PANEL_A_COLS].dropna().shape[0]
    n_obs_b = panel_b_inputs[_PANEL_B_COLS].dropna().shape[0]
    logger.info(
        "Table 3 complete: Panel A has ~%d obs, Panel B has ~%d obs",
        n_obs_a,
        n_obs_b,
    )
    return panel_a, panel_b


def _compute_correlations(
    data: pd.DataFrame,
    row_labels: list[str],
    col_labels: list[str],
) -> pd.DataFrame:
    """Compute pairwise Pearson correlations between row variables and column variables.

    For each (row, col) pair, uses pairwise complete observations (dropna on pair).
    Sets diagonal to 1.0 for variables in both row_labels and col_labels.

    Args:
        data: DataFrame with all variables as columns.
        row_labels: Row variable names (must be columns of data).
        col_labels: Column variable names (must be columns of data).

    Returns:
        DataFrame of shape (len(row_labels), len(col_labels)) with correlation values.
        Missing columns in data produce NaN rows/columns.
    """
    result = pd.DataFrame(np.nan, index=row_labels, columns=col_labels, dtype=float)

    for row in row_labels:
        for col in col_labels:
            if row not in data.columns or col not in data.columns:
                continue
            if row == col:
                # Perfect self-correlation
                valid = data[row].dropna()
                if len(valid) >= 3:
                    result.loc[row, col] = 1.0
                continue
            row_s = data[row]
            col_s = data[col]
            # Ensure we have Series (not DataFrame) even with duplicate column names
            if isinstance(row_s, pd.DataFrame):
                row_s = row_s.iloc[:, 0]
            if isinstance(col_s, pd.DataFrame):
                col_s = col_s.iloc[:, 0]
            combined = pd.concat([row_s, col_s], axis=1, keys=["_r", "_c"]).dropna()
            if len(combined) < 3:
                logger.warning("Too few observations for corr(%s, %s): %d", row, col, len(combined))
                continue
            result.loc[row, col] = float(combined["_r"].corr(combined["_c"]))

    return result
