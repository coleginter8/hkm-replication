"""Compute Table 2: average sizes of primary dealers relative to comparison groups."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import psycopg2.extensions

from hkm.data.compustat import fetch_compustat_all_quarterly, fetch_compustat_quarterly
from hkm.data.crsp import fetch_crsp_all_monthly, fetch_crsp_monthly
from hkm.data.dealers import PRIMARY_DEALERS, find_dealer_identifiers, get_active_dealers
from hkm.utils import get_logger, wrds_connection

logger = get_logger(__name__)

_GROUPS = ["BD", "Banks", "Cmpust"]
_ITEMS = ["Total assets", "Book debt", "Book equity", "Market equity"]

_PERIODS: dict[str, tuple[str, str]] = {
    "1960-2012": ("1960-01-01", "2012-12-31"),
    "1960-1990": ("1960-01-01", "1989-12-31"),
    "1990-2012": ("1990-01-01", "2012-12-31"),
}

# SIC filters for each comparison group
_GROUP_SIC: dict[str, dict[str, object]] = {
    "BD": {"sic_filter": "BD"},
    "Banks": {"sic_filter": "Banks"},
    "Cmpust": {"sic_filter": None},
}

_CRSP_GROUP_SIC: dict[str, dict[str, object]] = {
    "BD": {"sic_codes": ["6211", "6221"]},
    "Banks": {"sic_range": (6000, 6299)},
    "Cmpust": {},
}


def _get_ccm_link_for_gvkeys(
    gvkeys: list[str],
    conn: psycopg2.extensions.connection,
) -> pd.DataFrame:
    """Fetch CCM link rows for a list of GVKEYs.

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
    df["linkenddt"] = pd.to_datetime(df["linkenddt"].fillna("2099-12-31"))
    return df


def compute_table2(
    conn: psycopg2.extensions.connection | None = None,
    start_date: str = "1960-01-01",
    end_date: str = "2012-12-31",
) -> pd.DataFrame:
    """Compute Table 2: primary dealer size relative to comparison groups.

    For each calendar month, computes the ratio of dealer aggregate balance-sheet
    items to each comparison group aggregate, then averages over three sub-periods.

    Args:
        conn: Open psycopg2 WRDS connection, or None to open one internally.
        start_date: Earliest date (ISO format, default 1960-01-01).
        end_date: Latest date (ISO format, default 2012-12-31).

    Returns:
        DataFrame of shape (3, 12):
            Index: ['1960-2012', '1960-1990', '1990-2012']
            Columns: MultiIndex (item × group), items = Total assets, Book debt,
                Book equity, Market equity; groups = BD, Banks, Cmpust.
    """

    def _run(c: psycopg2.extensions.connection) -> pd.DataFrame:
        return _compute_table2_with_conn(c, start_date, end_date)

    if conn is not None:
        return _run(conn)
    with wrds_connection() as c:
        return _run(c)


def _compute_table2_with_conn(
    conn: psycopg2.extensions.connection,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Internal implementation of Table 2 with an open connection."""
    logger.info("Computing Table 2 (%s – %s)", start_date, end_date)

    # ---- Step 1: Resolve dealer identifiers ----
    dealers = find_dealer_identifiers(list(PRIMARY_DEALERS), conn)
    known_gvkeys = list({d.gvkey for d in dealers if d.gvkey is not None})
    logger.info("Dealers with GVKEY: %d", len(known_gvkeys))

    # ---- Step 2: Fetch Compustat quarterly for dealers ----
    dealer_comp = fetch_compustat_quarterly(known_gvkeys, start_date, end_date, conn=conn)

    # ---- Step 3: Get dealer PERMNOs via CCM link ----
    link_df = _get_ccm_link_for_gvkeys(known_gvkeys, conn)
    all_dealer_permnos = list(link_df["permno"].unique().astype(int))
    logger.info("Fetching CRSP monthly for %d dealer PERMNOs", len(all_dealer_permnos))
    dealer_crsp = fetch_crsp_monthly(all_dealer_permnos, start_date, end_date, conn=conn)

    # Build gvkey → permno mapping (time-aware)
    gvkey_to_links: dict[str, list[tuple[int, pd.Timestamp, pd.Timestamp]]] = {}
    for _, row in link_df.iterrows():
        gv = str(row["gvkey"]).zfill(6)
        entry = (int(row["permno"]), row["linkdt"], row["linkenddt"])
        gvkey_to_links.setdefault(gv, []).append(entry)

    # ---- Fetch CRSP SIC codes for dealer PERMNOs (for group classification) ----
    from hkm.data.wrds_connect import run_query

    dealer_sic_df = pd.DataFrame(columns=["permno", "siccd", "namedt", "nameendt"])
    if all_dealer_permnos:
        perm_ph = ", ".join(str(p) for p in all_dealer_permnos)
        sic_sql = f"""
            SELECT permno, siccd, namedt, nameendt
            FROM crsp.msenames
            WHERE permno IN ({perm_ph})
            ORDER BY permno, namedt
        """
        dealer_sic_raw = run_query(sic_sql, conn)
        if not dealer_sic_raw.empty:
            dealer_sic_raw["permno"] = dealer_sic_raw["permno"].astype(int)
            dealer_sic_raw["siccd"] = dealer_sic_raw["siccd"].astype(float)
            dealer_sic_raw["namedt"] = pd.to_datetime(dealer_sic_raw["namedt"])
            dealer_sic_raw["nameendt"] = pd.to_datetime(
                dealer_sic_raw["nameendt"].fillna("2099-12-31")
            )
            dealer_sic_df = dealer_sic_raw

    def _get_dealer_sic_at(permno: int, date: pd.Timestamp) -> float | None:
        """Return CRSP historical SIC code for a permno at a given date."""
        sub = dealer_sic_df[
            (dealer_sic_df["permno"] == permno)
            & (dealer_sic_df["namedt"] <= date)
            & (dealer_sic_df["nameendt"] >= date)
        ]
        if sub.empty:
            return None
        return float(sub["siccd"].iloc[0])

    def _dealer_in_group(permno: int, date: pd.Timestamp, grp: str) -> bool:
        """Return True if dealer PERMNO is in the given comparison group at date."""
        sic = _get_dealer_sic_at(permno, date)
        if sic is None:
            # Unknown SIC — include in Cmpust only
            return grp == "Cmpust"
        sic_int = int(sic)
        if grp == "BD":
            return sic_int in (6211, 6221)
        if grp == "Banks":
            return 6000 <= sic_int <= 6299
        return True  # Cmpust includes all

    # ---- Step 4: Fetch comparison group data (Compustat + CRSP) ----
    logger.info("Fetching comparison group Compustat data")
    group_comp: dict[str, pd.DataFrame] = {}
    for grp in _GROUPS:
        sf = _GROUP_SIC[grp]["sic_filter"]
        group_comp[grp] = fetch_compustat_all_quarterly(
            sic_filter=sf,  # type: ignore[arg-type]
            start_date=start_date,
            end_date=end_date,
            conn=conn,
        )

    logger.info("Fetching comparison group CRSP data")
    group_crsp: dict[str, pd.DataFrame] = {}
    for grp in _GROUPS:
        kw = _CRSP_GROUP_SIC[grp]
        group_crsp[grp] = fetch_crsp_all_monthly(
            start_date=start_date,
            end_date=end_date,
            conn=conn,
            **kw,  # type: ignore[arg-type]
        )

    # ---- Step 5: Build monthly ratio time series ----
    date_range = pd.date_range(start=start_date, end=end_date, freq="ME")

    # Pre-index group Compustat by (gvkey, datadate) for fast lookup
    # For each group: pivot to gvkey → sorted datadates
    for grp in _GROUPS:
        group_comp[grp] = group_comp[grp].sort_values(["gvkey", "datadate"])

    # Monthly ratio storage: {(item, group): list of float}
    ratio_records: list[dict[str, object]] = []

    for t in date_range:
        t_date: datetime.date = t.date()
        t_ts = pd.Timestamp(t)
        t_year = t.year
        t_month = t.month

        # ---- Dealer aggregates at t (computed per comparison group) ----
        # Per paper: dealers ⊆ BD group means only BD-SIC dealers go in BD numerator.
        # Historical SIC from CRSP msenames determines group membership.
        active = get_active_dealers(t_date)
        active_gvkeys = {d.gvkey for d in active if d.gvkey is not None}

        # Per dealer: compute balance-sheet items + ME + PERMNO (for SIC lookup)
        dealer_items: list[dict[str, float | int | str]] = []

        for gvkey in active_gvkeys:
            comp_sub = dealer_comp[
                (dealer_comp["gvkey"] == gvkey) & (dealer_comp["datadate"] <= t_ts)
            ]
            if comp_sub.empty:
                continue
            latest = comp_sub.loc[comp_sub["datadate"].idxmax()]

            # Find active PERMNO at t
            links = gvkey_to_links.get(gvkey, [])
            perms_at_t = [p for p, ld, le in links if ld <= t_ts <= le]
            if not perms_at_t:
                perms_at_t = [p for p, _, _ in links]
            if not perms_at_t:
                continue

            crsp_sub = dealer_crsp[
                (dealer_crsp["permno"].isin(perms_at_t))
                & (dealer_crsp["date"].dt.year == t_year)
                & (dealer_crsp["date"].dt.month == t_month)
            ]
            if crsp_sub.empty:
                continue

            perm_used = int(crsp_sub["permno"].iloc[0])
            me_m = float(crsp_sub["me"].iloc[0]) / 1000.0  # thousands → millions
            dealer_items.append(
                {
                    "gvkey": gvkey,
                    "permno": perm_used,
                    "ta": float(latest["atq"]),
                    "bd": float(latest["book_debt"]),
                    "be": float(latest["ceqq"]),
                    "me": me_m,
                }
            )

        if not dealer_items:
            # No dealer data for this month; skip
            continue

        # ---- Comparison group aggregates at t ----
        rec: dict[str, object] = {"date": t_ts}

        for grp in _GROUPS:
            # Dealer aggregate (numerator): only dealers in this comparison group
            d_ta = 0.0
            d_bd_g = 0.0
            d_be = 0.0
            d_me = 0.0
            for di in dealer_items:
                perm = int(di["permno"])
                if _dealer_in_group(perm, t_ts, grp):
                    d_ta += float(di["ta"])
                    d_bd_g += float(di["bd"])
                    d_be += float(di["be"])
                    d_me += float(di["me"])

            rec[f"d_ta_{grp}"] = d_ta
            rec[f"d_bd_{grp}"] = d_bd_g
            rec[f"d_be_{grp}"] = d_be
            rec[f"d_me_{grp}"] = d_me

            # Comparison group aggregate (denominator)
            gc = group_comp[grp]
            gc_t = gc[gc["datadate"] <= t_ts]
            if not gc_t.empty:
                gc_latest = gc_t.loc[gc_t.groupby("gvkey")["datadate"].idxmax()]
                g_ta = float(gc_latest["atq"].sum())
                g_bd = float(gc_latest["book_debt"].sum())
                g_be = float(gc_latest["ceqq"].sum())
            else:
                g_ta = g_bd = g_be = np.nan

            gc2 = group_crsp[grp]
            gc2_t = gc2[
                (gc2["date"].dt.year == t_year) & (gc2["date"].dt.month == t_month)
            ]
            g_me = float(gc2_t["me"].sum()) / 1000.0 if not gc2_t.empty else np.nan

            rec[f"g_ta_{grp}"] = g_ta
            rec[f"g_bd_{grp}"] = g_bd
            rec[f"g_be_{grp}"] = g_be
            rec[f"g_me_{grp}"] = g_me

        ratio_records.append(rec)

    if not ratio_records:
        logger.warning("No monthly observations computed for Table 2")
        cols = pd.MultiIndex.from_product([_ITEMS, _GROUPS], names=["item", "group"])
        return pd.DataFrame(
            index=list(_PERIODS.keys()), columns=cols, dtype=float
        )

    monthly = pd.DataFrame(ratio_records).set_index("date")

    # Compute ratios (per-group dealer numerator vs. group denominator)
    for grp in _GROUPS:
        monthly[f"ratio_ta_{grp}"] = monthly[f"d_ta_{grp}"] / monthly[f"g_ta_{grp}"]
        monthly[f"ratio_bd_{grp}"] = monthly[f"d_bd_{grp}"] / monthly[f"g_bd_{grp}"]
        monthly[f"ratio_be_{grp}"] = monthly[f"d_be_{grp}"] / monthly[f"g_be_{grp}"]
        monthly[f"ratio_me_{grp}"] = monthly[f"d_me_{grp}"] / monthly[f"g_me_{grp}"]

    # Replace inf/negative with NaN
    monthly = monthly.replace([np.inf, -np.inf], np.nan)

    # ---- Step 6: Average over sub-periods ----
    cols = pd.MultiIndex.from_product([_ITEMS, _GROUPS], names=["item", "group"])
    result = pd.DataFrame(index=list(_PERIODS.keys()), columns=cols, dtype=float)

    item_to_col = {
        "Total assets": "ratio_ta",
        "Book debt": "ratio_bd",
        "Book equity": "ratio_be",
        "Market equity": "ratio_me",
    }

    for period_label, (p_start, p_end) in _PERIODS.items():
        mask = (monthly.index >= pd.Timestamp(p_start)) & (
            monthly.index <= pd.Timestamp(p_end)
        )
        sub = monthly.loc[mask]
        if sub.empty:
            continue
        for item in _ITEMS:
            col_base = item_to_col[item]
            for grp in _GROUPS:
                col = f"{col_base}_{grp}"
                if col in sub.columns:
                    val = float(sub[col].mean(skipna=True))
                    result.loc[period_label, (item, grp)] = val

    logger.info("Table 2 computation complete")
    return result
