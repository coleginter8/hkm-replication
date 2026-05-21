"""Build primary dealer capital ratios (η_t) and capital ratio factors."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import psycopg2.extensions
import statsmodels.api as sm

from hkm.data.compustat import fetch_compustat_quarterly
from hkm.data.crsp import fetch_crsp_monthly
from hkm.data.dealers import PRIMARY_DEALERS, Dealer, find_dealer_identifiers, get_active_dealers
from hkm.utils import get_logger

logger = get_logger(__name__)

_SAMPLE_END = datetime.date(2012, 12, 31)


def _get_ccm_link(
    gvkeys: list[str],
    conn: psycopg2.extensions.connection,
) -> pd.DataFrame:
    """Fetch CRSP-Compustat link table for given GVKEYs.

    Returns DataFrame with columns: gvkey, permno, linkdt, linkenddt.
    """
    from hkm.data.wrds_connect import run_query

    if not gvkeys:
        return pd.DataFrame(columns=["gvkey", "permno", "linkdt", "linkenddt"])

    placeholders = ", ".join(f"'{g}'" for g in gvkeys)
    sql = f"""
        SELECT gvkey, lpermno AS permno, linkdt, linkenddt
        FROM crsp.ccmxpf_linktable
        WHERE gvkey IN ({placeholders})
          AND linktype IN ('LU', 'LC', 'LS')
          AND linkprim IN ('P', 'C')
        ORDER BY gvkey, linkdt
    """
    df = run_query(sql, conn)
    df["gvkey"] = df["gvkey"].astype(str).str.zfill(6)
    df["permno"] = df["permno"].astype(int)
    df["linkdt"] = pd.to_datetime(df["linkdt"])
    df["linkenddt"] = pd.to_datetime(df["linkenddt"])
    return df


def _resolve_permnos(
    dealers: list[Dealer],
    conn: psycopg2.extensions.connection,
) -> dict[str, list[tuple[int, pd.Timestamp, pd.Timestamp]]]:
    """Map each known GVKEY to a list of (permno, linkdt, linkenddt) from CCM.

    Returns:
        Dict mapping gvkey → list of (permno, linkdt, linkenddt) tuples.
        linkenddt is NaT when the link is currently valid.
    """
    known_gvkeys = [d.gvkey for d in dealers if d.gvkey is not None]
    if not known_gvkeys:
        return {}

    link_df = _get_ccm_link(known_gvkeys, conn)

    gvkey_map: dict[str, list[tuple[int, pd.Timestamp, pd.Timestamp]]] = {}
    for _, row in link_df.iterrows():
        gv = str(row["gvkey"]).zfill(6)
        end = row["linkenddt"] if pd.notna(row["linkenddt"]) else pd.Timestamp("2012-12-31")
        entry = (int(row["permno"]), row["linkdt"], end)
        gvkey_map.setdefault(gv, []).append(entry)

    return gvkey_map


def build_capital_ratio(
    conn: psycopg2.extensions.connection,
    frequency: str = "Q",
    start_date: str = "1960-01-01",
    end_date: str = "2012-12-31",
) -> pd.DataFrame:
    """Compute the primary dealer capital ratio η_t.

    η_t = Σ_i ME_{i,t} / Σ_i (ME_{i,t} + BD_{i,t})

    where ME = CRSP market equity = |prc| × shrout (in $thousands)
          BD = Compustat book debt = AT − CEQ (in $millions, converted to $thousands)

    Also computes book_capital_ratio = Σ CEQ / Σ AT.

    Args:
        conn: Open psycopg2 WRDS connection.
        frequency: 'Q' for quarterly (last month of quarter), 'M' for monthly.
        start_date: Earliest date (ISO format).
        end_date: Latest date (ISO format).

    Returns:
        DataFrame indexed by date (monthly or period-end dates) with columns:
            eta (float): market capital ratio
            book_capital (float): book capital ratio = Σ CEQ / Σ AT
            n_dealers (int): number of dealers with data in that period
    """
    # Step 1: resolve identifiers
    logger.info("Resolving dealer identifiers via WRDS name search")
    dealers = find_dealer_identifiers(list(PRIMARY_DEALERS), conn)

    # Step 2: fetch Compustat data for dealers with known GVKEYs
    known_gvkeys = list({d.gvkey for d in dealers if d.gvkey is not None})
    logger.info("Fetching Compustat quarterly for %d GVKEYs", len(known_gvkeys))
    comp_df = fetch_compustat_quarterly(known_gvkeys, start_date, end_date, conn=conn)

    # Step 3: resolve PERMNOs via CCM link table
    gvkey_to_permnos = _resolve_permnos(dealers, conn)
    all_permnos = list({p for links in gvkey_to_permnos.values() for p, _, _ in links})
    logger.info("Fetching CRSP monthly for %d PERMNOs", len(all_permnos))
    crsp_df = fetch_crsp_monthly(all_permnos, start_date, end_date, conn=conn)

    # Build PERMNO → GVKEY reverse map (time-aware)
    permno_to_gvkey: dict[int, str] = {}
    for gv, links in gvkey_to_permnos.items():
        for perm, _, _ in links:
            permno_to_gvkey[perm] = gv

    crsp_df["gvkey"] = crsp_df["permno"].map(permno_to_gvkey)
    crsp_df = crsp_df.dropna(subset=["gvkey"])

    # Step 4: build the monthly date grid
    date_range = pd.date_range(start=start_date, end=end_date, freq="ME")

    if frequency == "Q":
        # Keep only last month of each quarter (March, June, September, December)
        date_range = date_range[date_range.month.isin([3, 6, 9, 12])]

    rows: list[dict[str, object]] = []

    for t in date_range:
        t_date = t.date()
        t_timestamp = pd.Timestamp(t)

        # Identify active dealers at t
        active = get_active_dealers(t_date)
        active_gvkeys = {d.gvkey for d in active if d.gvkey is not None}

        sum_me = 0.0
        sum_bd = 0.0
        sum_ceq = 0.0
        sum_at = 0.0
        n = 0

        for gvkey in active_gvkeys:
            # Get most recent Compustat filing available at t.
            # Per HKM, book balance-sheet data should be aligned to the calendar
            # quarter in which the data were *reported* (rdq), not the fiscal
            # quarter-end (datadate). Using rdq ensures that book capital is only
            # counted once the filing is publicly available, preventing look-ahead
            # bias and improving alignment with the CRSP market-equity dates.
            comp_sub = comp_df[comp_df["gvkey"] == gvkey].copy()
            if comp_sub.empty:
                continue
            # Use rdq (report date) for alignment if available; otherwise datadate.
            # The "available at t" date is the earlier of rdq and datadate + 3 months.
            if "rdq" in comp_sub.columns and comp_sub["rdq"].notna().any():
                # For rows where rdq is available, use it as the availability date.
                # For rows where rdq is NaN, fall back to datadate.
                avail_date = comp_sub["rdq"].fillna(
                    comp_sub["datadate"] + pd.DateOffset(months=3)
                )
            else:
                avail_date = comp_sub["datadate"] + pd.DateOffset(months=3)
            comp_sub = comp_sub[avail_date <= t_timestamp]
            if comp_sub.empty:
                continue
            latest_comp = comp_sub.loc[comp_sub["datadate"].idxmax()]
            atq: float = float(latest_comp["atq"])  # $millions
            ceqq: float = float(latest_comp["ceqq"])  # $millions
            book_debt: float = atq - ceqq  # $millions

            # Get CRSP market equity for this gvkey at month t
            # Find PERMNOs active at t for this gvkey
            links = gvkey_to_permnos.get(gvkey, [])
            perm_t: list[int] = [
                perm
                for perm, ldt, lend in links
                if ldt <= t_timestamp <= lend
            ]

            if not perm_t:
                # Fall back to using any permno for this gvkey
                perm_t = [perm for perm, _, _ in links]

            if not perm_t:
                continue

            # Get month-end ME from CRSP (exact match on month)
            crsp_sub = crsp_df[
                (crsp_df["gvkey"] == gvkey)
                & (crsp_df["date"].dt.year == t.year)
                & (crsp_df["date"].dt.month == t.month)
            ]
            if crsp_sub.empty:
                continue

            me_thousands: float = float(crsp_sub["me"].iloc[0])  # $thousands

            # Convert Compustat to $thousands for consistent units
            bd_thousands: float = book_debt * 1000.0  # $millions → $thousands
            at_thousands: float = atq * 1000.0
            ceq_thousands: float = ceqq * 1000.0

            sum_me += me_thousands
            sum_bd += bd_thousands
            sum_ceq += ceq_thousands
            sum_at += at_thousands
            n += 1

        if n == 0 or (sum_me + sum_bd) == 0:
            rows.append(
                {"date": t_timestamp, "eta": np.nan, "book_capital": np.nan, "n_dealers": 0}
            )
        else:
            eta = sum_me / (sum_me + sum_bd)
            book_cap = sum_ceq / sum_at if sum_at > 0 else np.nan
            rows.append(
                {"date": t_timestamp, "eta": eta, "book_capital": book_cap, "n_dealers": n}
            )

    result = pd.DataFrame(rows).set_index("date")
    n_valid = result["eta"].notna().sum()
    logger.info(
        "Capital ratio built: %d periods, %d valid η observations (frequency=%s)",
        len(result),
        n_valid,
        frequency,
    )
    return result


def build_capital_factor(
    eta: pd.Series,
    frequency: str = "Q",
) -> pd.Series:
    """Compute the capital ratio factor = AR(1) innovations scaled by lagged ratio.

    Fits OLS AR(1): η_t = ρ_0 + ρ × η_{t-1} + u_t on the full sample.
    Returns u_t / η_{t-1}.

    The estimated ρ should be approximately 0.94 (paper footnote 22).

    Args:
        eta: pd.Series of capital ratio values indexed by date/period.
        frequency: 'Q' for quarterly, 'M' for monthly (informational only).

    Returns:
        pd.Series of same index as eta with capital ratio factor values.
        First element is NaN.
    """
    eta_lag = eta.shift(1)
    valid = eta.notna() & eta_lag.notna()

    if valid.sum() < 5:
        logger.warning("Insufficient data to fit AR(1) for capital ratio factor")
        return pd.Series(np.nan, index=eta.index, name="capital_factor")

    x_mat = sm.add_constant(eta_lag[valid])
    y = eta[valid]
    res = sm.OLS(y, x_mat).fit()

    rho = float(res.params.iloc[1])
    logger.info("AR(1) capital ratio: ρ = %.4f (expected ~0.94)", rho)

    resids = pd.Series(np.nan, index=eta.index, dtype=float)
    resids[valid] = res.resid.values

    factor = resids / eta_lag
    factor.name = "capital_factor"
    return factor
