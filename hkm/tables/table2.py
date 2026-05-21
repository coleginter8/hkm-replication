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

        # ---- Dealer aggregates at t ----
        # Per HKM footnote 19, ALL active primary dealers appear in the numerator
        # regardless of their holding company's SIC code.
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

        # ---- Dealer aggregate (numerator): ALL active dealers for ALL groups ----
        # Per HKM footnote 19: "define the total broker-dealer sector as the set of
        # US primary dealers PLUS any firms with a broker-dealer SIC code (6211 or
        # 6221). Note that had we instead relied on the SIC code definition of
        # broker-dealers, we would miss important dealers that are subsidiaries of
        # holding companies not classified as broker-dealers, for instance JP Morgan."
        # Therefore ALL active primary dealers appear in the numerator regardless of
        # their holding company's SIC code.
        d_ta_all = sum(float(di["ta"]) for di in dealer_items)
        d_bd_all = sum(float(di["bd"]) for di in dealer_items)
        d_be_all = sum(float(di["be"]) for di in dealer_items)
        d_me_all = sum(float(di["me"]) for di in dealer_items)

        # Set of dealer GVKEYs active at t (for denominator construction)
        active_dealer_gvkeys: set[str] = {str(di["gvkey"]) for di in dealer_items}

        # Set of dealer PERMNOs active at t (for CRSP ME denominator construction)
        active_dealer_permnos: set[int] = {int(di["permno"]) for di in dealer_items}

        # ---- Comparison group aggregates at t ----
        rec: dict[str, object] = {"date": t_ts}

        for grp in _GROUPS:
            rec[f"d_ta_{grp}"] = d_ta_all
            rec[f"d_bd_{grp}"] = d_bd_all
            rec[f"d_be_{grp}"] = d_be_all
            rec[f"d_me_{grp}"] = d_me_all

            # Comparison group denominator (book items):
            # = all active primary dealers  +  non-dealer firms in this group
            # This matches HKM: "total broker-dealer sector = primary dealers PLUS
            # any firms with BD SIC code." We sum dealer TA directly (not via
            # group_comp, which only covers dealers with BD/Banks/Cmpust SIC) and add
            # the non-dealer portion from group_comp to avoid double-counting.
            gc = group_comp[grp]
            gc_t = gc[gc["datadate"] <= t_ts]
            if not gc_t.empty:
                gc_latest = gc_t.loc[gc_t.groupby("gvkey")["datadate"].idxmax()]
                # Exclude dealer GVKEYs from group_comp denominator (they are
                # already counted in d_ta_all / d_bd_all / d_be_all above).
                gc_non_dealer = gc_latest[
                    ~gc_latest["gvkey"].isin(active_dealer_gvkeys)
                ]
                g_ta = d_ta_all + float(gc_non_dealer["atq"].sum())
                g_bd = d_bd_all + float(gc_non_dealer["book_debt"].sum())
                g_be = d_be_all + float(gc_non_dealer["ceqq"].sum())
            else:
                g_ta = d_ta_all if d_ta_all > 0 else np.nan
                g_bd = d_bd_all if d_bd_all > 0 else np.nan
                g_be = d_be_all if d_be_all > 0 else np.nan

            # Market equity denominator: dealer ME + non-dealer group-SIC CRSP ME.
            # Per HKM footnote 19: the total BD sector = primary dealers PLUS
            # any SIC 6211/6221 firms. Some dealers (e.g. JPMorgan SIC 6020) are
            # in d_me_all but NOT in the SIC-filtered CRSP pull for BD group.
            # Therefore: g_me = d_me_all + Σ ME_{j ∈ group_crsp, j ∉ dealers}.
            # This prevents ME/BD ratio from ever exceeding 1.0.
            gc2 = group_crsp[grp]
            gc2_t = gc2[
                (gc2["date"].dt.year == t_year) & (gc2["date"].dt.month == t_month)
            ]
            if not gc2_t.empty:
                # Non-dealer CRSP firms in this group's SIC filter
                gc2_non_dealer = gc2_t[
                    ~gc2_t["permno"].isin(active_dealer_permnos)
                ]
                non_dealer_crsp_me = float(gc2_non_dealer["me"].sum()) / 1000.0
                # Total group ME = dealer ME (all dealers, regardless of SIC) +
                # non-dealer group-SIC ME from CRSP
                g_me = d_me_all + non_dealer_crsp_me
            else:
                g_me = d_me_all if d_me_all > 0 else np.nan

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
