"""Static primary dealer universe from Table A.1 of He, Kelly & Manela (2017).

Contains the complete list of US-based primary dealers with their CRSP/Compustat
identifiers (where known), active date ranges per Table A.1, and helper functions
for date-based filtering and runtime identifier resolution.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, replace

import psycopg2.extensions

from hkm.utils import get_logger

logger = get_logger(__name__)

_SAMPLE_END = datetime.date(2012, 12, 31)


@dataclass(frozen=True)
class Dealer:
    """A primary dealer with Compustat/CRSP identifiers and activity dates."""

    name: str
    gvkey: str | None  # 6-digit zero-padded Compustat key, or None
    permno: int | None  # CRSP PERMNO, or None
    start: datetime.date  # First active date per Table A.1
    end: datetime.date | None  # Last active date; None = "Current" (treated as 2012-12-31)


PRIMARY_DEALERS: list[Dealer] = [
    # ---- Major US dealers with known Compustat GVKEYs ----
    Dealer(
        "Goldman Sachs",
        gvkey="114628",  # GOLDMAN SACHS GROUP INC (verified in comp.names)
        permno=None,
        start=datetime.date(1974, 12, 4),
        end=None,
    ),
    Dealer(
        "Merrill Lynch",
        gvkey="007267",  # MERRILL LYNCH & CO INC (verified in comp.names)
        permno=None,
        start=datetime.date(1960, 5, 19),
        end=datetime.date(2010, 11, 1),
    ),
    Dealer(
        "Lehman Brothers",
        gvkey="030128",  # LEHMAN BROTHERS HOLDINGS INC (verified in comp.names)
        permno=None,
        start=datetime.date(1976, 11, 25),
        end=datetime.date(2008, 9, 22),
    ),
    Dealer(
        "Lehman Brothers (first run)",
        gvkey="030128",  # same entity
        permno=None,
        start=datetime.date(1973, 2, 22),
        end=datetime.date(1974, 1, 29),
    ),
    Dealer(
        "Morgan Stanley",
        gvkey="012124",  # MORGAN STANLEY (verified in comp.names)
        permno=None,
        start=datetime.date(1978, 2, 1),
        end=None,
    ),
    Dealer(
        "Citigroup",
        gvkey="003243",  # CITIGROUP INC (verified in comp.names)
        permno=None,
        start=datetime.date(1961, 6, 15),
        end=None,
    ),
    Dealer(
        "Drexel Burnham",
        gvkey=None,  # No Compustat match found for Drexel Burnham Lambert
        permno=None,
        start=datetime.date(1960, 5, 19),
        end=datetime.date(1990, 3, 28),
    ),
    Dealer(
        "Paine Webber",
        gvkey="008299",  # PAINE WEBBER GROUP (verified in comp.names)
        permno=None,
        start=datetime.date(1976, 11, 25),
        end=datetime.date(2000, 12, 4),
    ),
    Dealer(
        "Dean Witter Reynolds",
        gvkey="003823",  # DEAN WITTER REYNOLDS ORG INC (+ 027867 for Discover era)
        permno=None,
        start=datetime.date(1977, 11, 2),
        end=datetime.date(1998, 4, 30),
    ),
    # ---- Dealers with known GVKEYs (verified) ----
    Dealer(
        "Salomon Smith Barney",
        gvkey="008537",  # CITIGROUP GLOBAL MKTS HLDGS (formerly Smith Barney Holdings, SIC 6211)
        permno=None,
        start=datetime.date(1960, 5, 19),
        end=datetime.date(2003, 4, 6),
    ),
    Dealer(
        "Bear Stearns",
        gvkey="011818",  # BEAR STEARNS COMPANIES INC (verified in comp.names)
        permno=None,
        start=datetime.date(1981, 6, 10),
        end=datetime.date(2008, 10, 1),
    ),
    Dealer(
        "JP Morgan",
        gvkey="002968",  # JPMORGAN CHASE & CO (verified in comp.names)
        permno=None,
        start=datetime.date(1960, 5, 19),
        end=None,
    ),
    Dealer(
        "Chemical Bank",
        gvkey=None,
        permno=None,
        start=datetime.date(1960, 5, 19),
        end=datetime.date(1996, 3, 31),
    ),
    Dealer(
        "Manufacturers Hanover",
        gvkey="007003",  # MANUFACTURERS HANOVER CORP (verified in comp.names)
        permno=None,
        start=datetime.date(1983, 8, 31),
        end=datetime.date(1991, 12, 31),
    ),
    Dealer(
        "Bankers Trust",
        gvkey="002029",  # BANKERS TRUST CORP (verified in comp.names)
        permno=None,
        start=datetime.date(1960, 5, 19),
        end=datetime.date(1997, 10, 22),
    ),
    Dealer(
        "Continental",
        gvkey=None,
        permno=None,
        start=datetime.date(1960, 5, 19),
        end=datetime.date(1991, 8, 30),
    ),
    Dealer(
        "Discount Corp.",
        gvkey=None,
        permno=None,
        start=datetime.date(1960, 5, 19),
        end=datetime.date(1993, 8, 10),
    ),
    Dealer(
        "First Boston",
        gvkey="004684",  # FIRST BOSTON INC (verified in comp.names)
        permno=None,
        start=datetime.date(1960, 5, 19),
        end=datetime.date(1993, 10, 11),
    ),
    Dealer(
        "Kidder Peabody",
        gvkey=None,
        permno=None,
        start=datetime.date(1978, 11, 17),
        end=datetime.date(1994, 4, 15),
    ),
    Dealer(
        "Bank of America",
        gvkey="007647",  # BANK OF AMERICA CORP (verified in comp.names)
        permno=None,
        start=datetime.date(1999, 5, 17),
        end=datetime.date(2010, 11, 1),
    ),
    Dealer(
        "Prudential",
        gvkey=None,
        permno=None,
        start=datetime.date(1975, 10, 29),
        end=datetime.date(2000, 12, 1),
    ),
    Dealer(
        "Dillon Read",
        gvkey=None,
        permno=None,
        start=datetime.date(1988, 6, 24),
        end=datetime.date(1997, 9, 2),
    ),
    Dealer(
        "Harris",
        gvkey=None,
        permno=None,
        start=datetime.date(1965, 7, 15),
        end=datetime.date(1995, 5, 31),
    ),
    Dealer(
        "Aubrey Lanston",
        gvkey=None,
        permno=None,
        start=datetime.date(1960, 5, 19),
        end=datetime.date(2000, 4, 17),
    ),
    Dealer(
        "Blyth Eastman Dillon",
        gvkey=None,
        permno=None,
        start=datetime.date(1974, 12, 5),
        end=datetime.date(1979, 12, 31),
    ),
    Dealer(
        "Carroll McEntee",
        gvkey=None,
        permno=None,
        start=datetime.date(1976, 9, 29),
        end=datetime.date(1994, 5, 6),
    ),
    Dealer(
        "DLJ",
        gvkey=None,
        permno=None,
        start=datetime.date(1974, 3, 6),
        end=datetime.date(1983, 1, 16),
    ),
    Dealer(
        "DLJ (second run)",
        gvkey=None,
        permno=None,
        start=datetime.date(1995, 10, 25),
        end=datetime.date(2000, 12, 31),
    ),
    Dealer(
        "First Chicago",
        gvkey=None,
        permno=None,
        start=datetime.date(1960, 5, 19),
        end=datetime.date(1999, 3, 31),
    ),
    Dealer(
        "First Interstate",
        gvkey=None,
        permno=None,
        start=datetime.date(1964, 7, 31),
        end=datetime.date(1988, 6, 17),
    ),
    Dealer(
        "Midland-Montagu",
        gvkey=None,
        permno=None,
        start=datetime.date(1975, 8, 13),
        end=datetime.date(1990, 7, 26),
    ),
    Dealer(
        "NationsBanc",
        gvkey=None,
        permno=None,
        start=datetime.date(1993, 7, 6),
        end=datetime.date(1999, 5, 16),
    ),
    Dealer(
        "Security Pacific",
        gvkey=None,
        permno=None,
        start=datetime.date(1986, 12, 11),
        end=datetime.date(1991, 1, 17),
    ),
    Dealer(
        "White Weld",
        gvkey=None,
        permno=None,
        start=datetime.date(1976, 2, 26),
        end=datetime.date(1978, 4, 18),
    ),
    Dealer(
        "Becker",
        gvkey=None,
        permno=None,
        start=datetime.date(1958, 5, 8),
        end=datetime.date(1984, 9, 10),
    ),
    Dealer(
        "CF Childs",
        gvkey=None,
        permno=None,
        start=datetime.date(1960, 5, 19),
        end=datetime.date(1965, 6, 29),
    ),
    Dealer(
        "Chase",
        gvkey="002943",  # CHASE MANHATTAN CORP -OLD (verified in comp.names)
        permno=None,
        start=datetime.date(1970, 10, 12),
        # Chase Manhattan filed its last Compustat quarterly report in 1995Q4
        # after merging with Chemical Bank (effective 1996-03-31). The entity
        # became JPMorgan Chase via the 2000 merger; JPMorgan (gvkey 002968) covers
        # the post-merger combined entity.  Truncate here to avoid carrying stale
        # 1995 balance-sheet data forward through 2001.
        end=datetime.date(1995, 12, 31),
    ),
    Dealer(
        "First National Bank of Boston",
        gvkey=None,
        permno=None,
        start=datetime.date(1983, 3, 21),
        end=datetime.date(1985, 11, 17),
    ),
    Dealer(
        "Pollock",
        gvkey=None,
        permno=None,
        start=datetime.date(1960, 5, 19),
        end=datetime.date(1987, 2, 3),
    ),
    Dealer(
        "Second District",
        gvkey=None,
        permno=None,
        start=datetime.date(1961, 3, 15),
        end=datetime.date(1983, 8, 27),
    ),
]


def get_active_dealers(as_of: datetime.date) -> list[Dealer]:
    """Return dealers active on the given date.

    A dealer is active if start <= as_of <= end.
    Dealers with end=None are treated as active through 2012-12-31.
    """
    effective_end = _SAMPLE_END
    result: list[Dealer] = []
    for d in PRIMARY_DEALERS:
        dealer_end = d.end if d.end is not None else effective_end
        if d.start <= as_of <= dealer_end:
            result.append(d)
    return result


# Name fragments for fuzzy matching in find_dealer_identifiers.
# Maps each dealer name to a list of SQL ILIKE patterns.
_NAME_PATTERNS: dict[str, list[str]] = {
    "Salomon Smith Barney": ["%salomon%"],
    "Bear Stearns": ["%bear stearns%"],
    "JP Morgan": ["%jpmorgan%", "%jp morgan%", "%j.p. morgan%"],
    "Chemical Bank": ["%chemical bank%"],
    "Manufacturers Hanover": ["%manufacturers hanover%"],
    "Bankers Trust": ["%bankers trust%"],
    "Continental": ["%continental illinois%"],
    "Discount Corp.": ["%discount corp%"],
    "First Boston": ["%first boston%"],
    "Kidder Peabody": ["%kidder%peabody%", "%kidder, peabody%"],
    "Bank of America": ["%bank of america%"],
    "Prudential": ["%prudential%bache%", "%prudential securities%"],
    "Dillon Read": ["%dillon read%"],
    "Harris": ["%harris upham%"],
    "Aubrey Lanston": ["%aubrey lanston%"],
    "Blyth Eastman Dillon": ["%blyth%eastman%"],
    "Carroll McEntee": ["%carroll%mcentee%"],
    "DLJ": ["%donaldson%lufkin%", "%dlj%"],
    "DLJ (second run)": ["%donaldson%lufkin%", "%dlj%"],
    "First Chicago": ["%first chicago%"],
    "First Interstate": ["%first interstate%"],
    "Midland-Montagu": ["%midland%montagu%"],
    "NationsBanc": ["%nationsbanc%", "%nations banc%"],
    "Security Pacific": ["%security pacific%"],
    "White Weld": ["%white weld%"],
    "Becker": ["%a.g. becker%", "%ag becker%"],
    "CF Childs": ["%childs%"],
    "Chase": ["%chase manhattan%"],
    "First National Bank of Boston": ["%first national bank%boston%"],
    "Pollock": ["%pollock%"],
    "Second District": ["%second district%"],
}


def find_dealer_identifiers(
    dealers: list[Dealer],
    conn: psycopg2.extensions.connection,
) -> list[Dealer]:
    """Attempt to fill missing gvkey/permno by fuzzy name-matching against Compustat/CRSP.

    For each dealer with gvkey=None, queries comp.names using ILIKE patterns derived
    from the dealer name.  For each dealer with permno=None (but gvkey known or found),
    queries the CRSP-Compustat link table to resolve PERMNO.

    Args:
        dealers: List of Dealer objects (may have gvkey/permno = None).
        conn: An open psycopg2 WRDS connection.

    Returns:
        New list of Dealer objects with identifiers filled in where matches were found.
        Unmatched dealers keep gvkey=None / permno=None and are logged as misses.
    """
    from hkm.data.wrds_connect import run_query  # avoid circular at module level

    updated: list[Dealer] = []

    # Step 1: resolve missing GVKEYs via comp.names fuzzy matching
    for dealer in dealers:
        if dealer.gvkey is not None:
            updated.append(dealer)
            continue

        patterns = _NAME_PATTERNS.get(dealer.name, [])
        if not patterns:
            logger.warning("No name patterns defined for dealer '%s'; skipping", dealer.name)
            updated.append(dealer)
            continue

        conditions = " OR ".join(f"conm ILIKE '{p}'" for p in patterns)
        sql = f"SELECT gvkey, conm FROM comp.names WHERE {conditions} ORDER BY gvkey LIMIT 5"
        try:
            df = run_query(sql, conn)
        except Exception as exc:
            logger.warning("GVKEY lookup failed for '%s': %s", dealer.name, exc)
            updated.append(dealer)
            continue

        if df.empty:
            logger.warning("No Compustat match for dealer '%s'", dealer.name)
            updated.append(dealer)
        else:
            gvkey = str(df["gvkey"].iloc[0]).zfill(6)
            conm = df["conm"].iloc[0]
            logger.info("Matched '%s' → gvkey=%s (%s)", dealer.name, gvkey, conm)
            updated.append(replace(dealer, gvkey=gvkey))

    # Step 2: resolve PERMNOs for dealers that now have a GVKEY but no PERMNO
    gvkeys_needing_permno = [d.gvkey for d in updated if d.gvkey is not None and d.permno is None]
    if not gvkeys_needing_permno:
        return updated

    placeholders = ", ".join(f"'{g}'" for g in gvkeys_needing_permno)
    link_sql = f"""
        SELECT gvkey, lpermno AS permno
        FROM crsp.ccmxpf_linktable
        WHERE gvkey IN ({placeholders})
          AND linktype IN ('LU', 'LC', 'LS')
          AND linkprim IN ('P', 'C')
        ORDER BY gvkey, linkdt
    """
    try:
        link_df = run_query(link_sql, conn)
    except Exception as exc:
        logger.warning("PERMNO lookup via CCM link table failed: %s", exc)
        return updated

    # Build gvkey → permno map (take first / primary match)
    gvkey_to_permno: dict[str, int] = {}
    for _, row in link_df.iterrows():
        gv = str(row["gvkey"]).zfill(6)
        if gv not in gvkey_to_permno:
            gvkey_to_permno[gv] = int(row["permno"])

    final: list[Dealer] = []
    for dealer in updated:
        if dealer.gvkey is not None and dealer.permno is None:
            perm = gvkey_to_permno.get(dealer.gvkey)
            if perm is not None:
                logger.info(
                    "Resolved PERMNO for '%s' (gvkey=%s): %d",
                    dealer.name,
                    dealer.gvkey,
                    perm,
                )
                final.append(replace(dealer, permno=perm))
            else:
                logger.warning(
                    "No PERMNO found for '%s' (gvkey=%s) in CCM link table",
                    dealer.name,
                    dealer.gvkey,
                )
                final.append(dealer)
        else:
            final.append(dealer)

    return final
