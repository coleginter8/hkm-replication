"""Fetch quarterly Compustat balance-sheet data for dealers and comparison groups."""

from __future__ import annotations

import pandas as pd
import psycopg2.extensions

from hkm.data.wrds_connect import run_query
from hkm.utils import get_logger, wrds_connection

logger = get_logger(__name__)


def fetch_compustat_quarterly(
    gvkeys: list[str],
    start_date: str = "1960-01-01",
    end_date: str = "2012-12-31",
    conn: psycopg2.extensions.connection | None = None,
) -> pd.DataFrame:
    """Fetch quarterly Compustat data for the given GVKEYs.

    Args:
        gvkeys: List of 6-digit Compustat GVKEYs.
        start_date: Earliest datadate to include (ISO format).
        end_date: Latest datadate to include (ISO format).
        conn: Open psycopg2 connection, or None to open one internally.

    Returns:
        DataFrame with columns: gvkey (str), datadate (Timestamp), atq (float),
        ceqq (float), book_debt (float = atq - ceqq), fyearq (int), fqtr (int).
        Only rows where atq > 0 and ceqq is not null are returned.
    """
    if not gvkeys:
        logger.warning("fetch_compustat_quarterly called with empty gvkeys list")
        return pd.DataFrame(
            columns=["gvkey", "datadate", "rdq", "atq", "ceqq", "book_debt", "fyearq", "fqtr"]
        )

    placeholders = ", ".join(f"'{g}'" for g in gvkeys)
    sql = f"""
        SELECT gvkey, datadate, rdq, atq, ceqq, fyearq, fqtr
        FROM comp.fundq
        WHERE gvkey IN ({placeholders})
          AND datadate BETWEEN '{start_date}' AND '{end_date}'
          AND datafmt = 'STD'
          AND indfmt = 'INDL'
          AND popsrc = 'D'
          AND consol = 'C'
          AND atq IS NOT NULL
          AND atq > 0
          AND ceqq IS NOT NULL
        ORDER BY gvkey, datadate
    """

    def _execute(c: psycopg2.extensions.connection) -> pd.DataFrame:
        df = run_query(sql, c)
        if df.empty:
            return pd.DataFrame(
                columns=["gvkey", "datadate", "rdq", "atq", "ceqq", "book_debt", "fyearq", "fqtr"]
            )
        df["gvkey"] = df["gvkey"].astype(str).str.zfill(6)
        df["datadate"] = pd.to_datetime(df["datadate"])
        df["rdq"] = pd.to_datetime(df["rdq"], errors="coerce")
        df["atq"] = df["atq"].astype(float)
        df["ceqq"] = df["ceqq"].astype(float)
        df["book_debt"] = df["atq"] - df["ceqq"]
        df["fyearq"] = df["fyearq"].astype(int)
        df["fqtr"] = df["fqtr"].astype(int)
        logger.info(
            "Compustat: fetched %d rows for %d GVKEYs, date range %s–%s",
            len(df),
            len(gvkeys),
            start_date,
            end_date,
        )
        return df

    if conn is not None:
        return _execute(conn)
    with wrds_connection() as c:
        return _execute(c)


def fetch_compustat_all_quarterly(
    sic_filter: str | None = None,
    start_date: str = "1960-01-01",
    end_date: str = "2012-12-31",
    conn: psycopg2.extensions.connection | None = None,
    lookback_months: int = 18,
) -> pd.DataFrame:
    """Fetch quarterly Compustat data for comparison groups (BD, Banks, all firms).

    Uses comp.funda.sich (historical SIC from annual filings) joined to comp.fundq
    to identify which firms belong to each comparison group at each point in time.
    This is the correct approach per HKM (2017) footnote 19: "the total broker-dealer
    sector as the set of US primary dealers plus any firms with a broker-dealer SIC
    code (6211 or 6221)."

    For BD/Banks groups, firms are additionally filtered to US-listed ordinary common
    stocks (crsp.msenames.shrcd IN (10, 11)) via the CCM link table to exclude foreign
    firms (e.g., Credit Suisse Group, Nomura Holdings) from the comparison group.

    The annual historical SIC (comp.funda.sich) is matched to quarterly filings via
    fyearq = fyear, so each quarterly filing uses the SIC code from the corresponding
    annual report. This is superior to comp.names.sic (which reflects only the final,
    current SIC code).

    The query extends back by ``lookback_months`` before ``start_date`` so that at
    each target date t, the most recent filing within 18 months of t is available.
    This avoids sparse-coverage issues for early years (pre-1978) where delayed
    filings might otherwise cause firms to be excluded from the comparison group.

    Args:
        sic_filter: 'BD' for sich IN ('6211', '6221'),
                    'Banks' for sich BETWEEN '6000' AND '6299',
                    None for all firms (no SIC filter, but still US-listed).
        start_date: Earliest target date (ISO format). The SQL query fetches filings
                    from ``start_date - lookback_months`` to ``end_date``.
        end_date: Latest datadate to include (ISO format).
        conn: Open psycopg2 connection, or None to open one internally.
        lookback_months: Number of months before start_date to include in the SQL
                    query to support firms with delayed filings (default 18).

    Returns:
        DataFrame with columns: gvkey (str), datadate (Timestamp), atq (float),
        ceqq (float), sich (str), book_debt (float), fyearq (int), fqtr (int).
        sich is the historical annual SIC from comp.funda.
    """

    def _execute(c: psycopg2.extensions.connection) -> pd.DataFrame:
        filter_label = sic_filter if sic_filter else "All"

        # Extend the SQL fetch window back by lookback_months to capture filings
        # that belong to comparison groups but arrived slightly before start_date.
        fetch_start = (
            pd.Timestamp(start_date) - pd.DateOffset(months=lookback_months)
        ).strftime("%Y-%m-%d")

        # Build SIC filter for comp.funda.sich (integer column in WRDS PostgreSQL)
        if sic_filter == "BD":
            sic_clause = "AND a.sich IN (6211, 6221)"
            crsp_join = """
                JOIN crsp.ccmxpf_linktable lk
                    ON q.gvkey = lk.gvkey
                   AND lk.linktype IN ('LU', 'LC', 'LS')
                   AND lk.linkprim IN ('P', 'C')
                   AND q.datadate BETWEEN lk.linkdt
                       AND COALESCE(lk.linkenddt, '2099-12-31'::date)
                JOIN crsp.msenames e
                    ON lk.lpermno = e.permno
                   AND q.datadate BETWEEN e.namedt
                       AND COALESCE(e.nameendt, '2099-12-31'::date)
                   AND e.shrcd IN (10, 11)
            """
        elif sic_filter == "Banks":
            sic_clause = "AND a.sich BETWEEN 6000 AND 6299"
            crsp_join = """
                JOIN crsp.ccmxpf_linktable lk
                    ON q.gvkey = lk.gvkey
                   AND lk.linktype IN ('LU', 'LC', 'LS')
                   AND lk.linkprim IN ('P', 'C')
                   AND q.datadate BETWEEN lk.linkdt
                       AND COALESCE(lk.linkenddt, '2099-12-31'::date)
                JOIN crsp.msenames e
                    ON lk.lpermno = e.permno
                   AND q.datadate BETWEEN e.namedt
                       AND COALESCE(e.nameendt, '2099-12-31'::date)
                   AND e.shrcd IN (10, 11)
            """
        else:
            sic_clause = ""
            crsp_join = """
                JOIN crsp.ccmxpf_linktable lk
                    ON q.gvkey = lk.gvkey
                   AND lk.linktype IN ('LU', 'LC', 'LS')
                   AND lk.linkprim IN ('P', 'C')
                   AND q.datadate BETWEEN lk.linkdt
                       AND COALESCE(lk.linkenddt, '2099-12-31'::date)
                JOIN crsp.msenames e
                    ON lk.lpermno = e.permno
                   AND q.datadate BETWEEN e.namedt
                       AND COALESCE(e.nameendt, '2099-12-31'::date)
                   AND e.shrcd IN (10, 11)
            """

        # Join fundq to funda (annual) on gvkey + fyearq = fyear to get historical
        # SIC from the annual report that corresponds to each quarterly filing.
        # DISTINCT ON (q.gvkey, q.datadate) prevents duplicates from multiple CRSP links.
        # The lookback window (fetch_start → end_date) ensures that at each target
        # month t, the most recent filing within the prior 18 months is available.
        sql = f"""
            SELECT DISTINCT ON (q.gvkey, q.datadate)
                   q.gvkey,
                   q.datadate,
                   q.atq,
                   q.ceqq,
                   CAST(a.sich AS VARCHAR) AS sich,
                   q.fyearq,
                   q.fqtr
            FROM comp.fundq q
            JOIN comp.funda a
                ON q.gvkey = a.gvkey
               AND q.fyearq = a.fyear
               AND a.datafmt = 'STD'
               AND a.indfmt = 'INDL'
               AND a.popsrc = 'D'
               AND a.consol = 'C'
               {sic_clause}
            {crsp_join}
            WHERE q.datadate BETWEEN '{fetch_start}' AND '{end_date}'
              AND q.datafmt = 'STD'
              AND q.indfmt = 'INDL'
              AND q.popsrc = 'D'
              AND q.consol = 'C'
              AND q.atq IS NOT NULL
              AND q.atq > 0
              AND q.ceqq IS NOT NULL
            ORDER BY q.gvkey, q.datadate
        """
        df = run_query(sql, c)
        if df.empty:
            logger.warning("No data found for sic_filter=%s", filter_label)
            return pd.DataFrame(
                columns=["gvkey", "datadate", "atq", "ceqq", "sich", "book_debt", "fyearq", "fqtr"]
            )

        df["gvkey"] = df["gvkey"].astype(str).str.zfill(6)
        df["datadate"] = pd.to_datetime(df["datadate"])
        df["atq"] = df["atq"].astype(float)
        df["ceqq"] = df["ceqq"].astype(float)
        df["sich"] = df["sich"].fillna("").astype(str)
        df["book_debt"] = df["atq"] - df["ceqq"]
        df = df.dropna(subset=["fyearq", "fqtr"])
        df["fyearq"] = df["fyearq"].astype(int)
        df["fqtr"] = df["fqtr"].astype(int)
        logger.info(
            "Compustat all: fetched %d rows, sic_filter=%s, date range %s–%s",
            len(df),
            filter_label,
            start_date,
            end_date,
        )
        return df

    if conn is not None:
        return _execute(conn)
    with wrds_connection() as c:
        return _execute(c)
