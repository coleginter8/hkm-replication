# HKM Replication — Tables 2 & 3

Python replication of Tables 2 and 3 from He, Kelly & Manela (2017), "Intermediary Asset Pricing: New Evidence from Many Asset Classes," *Journal of Financial Economics* 126: 1–35. The package provides two entry points — `compute_table2()` and `compute_table3()` — that retrieve data from WRDS (Compustat + CRSP), FRED, and public sources (Shiller data, Chicago Fed NFCI) and return DataFrames matching the paper's published tables.

---

## Quick Start

**Requirements**: Python >= 3.11; a WRDS account with `~/.pgpass` configured.

```bash
# Install the package and dev dependencies
pip install -e ".[dev]"

# Configure WRDS credentials (one-time)
# Add to ~/.pgpass (mode 600):
# wrds-pgdata.wharton.upenn.edu:9737:wrds:<your_username>:<your_password>
chmod 600 ~/.pgpass
```

```python
from hkm import compute_table2, compute_table3
from hkm.utils import wrds_connection

with wrds_connection(user="your_wrds_username") as conn:
    # Table 2: shape (3, 12) DataFrame
    t2 = compute_table2(conn=conn)
    print(t2)

    # Table 3: tuple of (panel_a, panel_b) DataFrames
    panel_a, panel_b = compute_table3(conn=conn)
    print(panel_a)
    print(panel_b)
```

---

## Implemented Tables

| Table | Description | Output shape |
|---|---|---|
| Table 2 | Primary dealer size relative to comparison groups (monthly, 3 sub-periods x 4 items x 3 groups) | `(3, 12)` DataFrame with `pd.MultiIndex` columns `(item, group)` |
| Table 3 Panel A | Pairwise correlations of capital ratio levels and macro variables (1970Q1-2012Q4) | `(8, 3)` DataFrame |
| Table 3 Panel B | Pairwise correlations of capital ratio factors and macro growth rates (1970Q1-2012Q4) | `(9, 3)` DataFrame |

---

## WRDS Requirements

The package queries these WRDS tables:

- `comp.fundq` + `comp.funda` — quarterly and annual balance sheet; historical SIC codes
- `comp.names` — dealer GVKEY resolution by fuzzy name match
- `crsp.msf` — monthly stock prices and shares outstanding
- `crsp.msenames` — historical SIC codes and US share class filter (`shrcd IN (10, 11)`)
- `crsp.msi` — monthly value-weighted market index returns
- `crsp.dsi` — daily value-weighted market index returns (for realized volatility)
- `crsp.ccmxpf_linktable` — CRSP-Compustat identifier link

AEM leverage is fetched from FRED (no WRDS subscription required): series `BOGZ1FL664090005Q` and `BOGZ1FL664190005Q`.

---

## Package Structure

```
hkm/
├── __init__.py           # Exports compute_table2, compute_table3
├── utils.py              # Logging (get_logger) and WRDS connection context manager (wrds_connection)
├── data/
│   ├── wrds_connect.py   # run_query(sql, conn) -> pd.DataFrame
│   ├── compustat.py      # fetch_compustat_quarterly, fetch_compustat_all_quarterly
│   ├── crsp.py           # fetch_crsp_monthly, fetch_crsp_all_monthly, fetch_crsp_market_index, fetch_crsp_daily_vol
│   ├── dealers.py        # Dealer dataclass, PRIMARY_DEALERS, get_active_dealers, find_dealer_identifiers
│   ├── intermediary.py   # build_capital_ratio (eta_t, book_capital), build_capital_factor (AR(1) innovation)
│   └── macro.py          # fetch_fred_series, fetch_shiller_ep, fetch_nfci, fetch_aem_leverage, build_macro_panel
└── tables/
    ├── table2.py         # compute_table2() -> pd.DataFrame (3, 12)
    └── table3.py         # compute_table3() -> (panel_a, panel_b)
```

---

## Running Tests

```bash
# All tests (requires WRDS connection for integration tests; ~10 minutes)
pytest tests/ -v

# Unit tests only (no WRDS connection; ~5 seconds)
pytest tests/ -v -k "not Integration"

# Code quality
ruff check hkm/
mypy hkm/ --strict --ignore-missing-imports
```

---

## Known Limitations

1. **Pre-1978 WRDS coverage**: Compustat quarterly data for financial firms begins c.1978. Table 2 values for the 1960-1990 sub-period and full 1960-2012 period may differ from published by more than +/-0.05 due to sparse pre-1978 coverage.

2. **Foreign dealers excluded**: The paper includes foreign primary dealers (Barclays, Deutsche Bank, UBS, etc.) via Datastream. This replication uses CRSP and Compustat only (US-based dealers), consistent with the paper's Table 2 note on US-only firms.

3. **AEM leverage sign**: The BOGZ1 Fed Z.1 broker-dealer leverage series (assets/book equity) is pro-cyclical over 1970-2012. The published Panel A correlation between eta and AEM leverage is -0.42; the replication produces +0.624 due to data-vintage/definition differences. All other Panel A and Panel B sign checks pass.

4. **Dealer GVKEY gaps**: Several historical dealers (Drexel Burnham, Kidder Peabody, Prudential-Bache, Dillon Read, etc.) could not be matched in Compustat and are absent from the dealer numerator, understating eta relative to the paper.

5. **Effective sample 1978-2012**: Despite nominal start date of 1960-01-01, the effective sample for most eta_t computations is 1978Q1-2012Q4. AR(1) estimate rho = 0.9581 (paper reports ~0.94).

---

## Data Sources

- **WRDS / Compustat** (`comp.fundq`, `comp.funda`): Quarterly balance sheet (AT, CEQ) and historical SIC codes for dealers and comparison groups.
- **WRDS / CRSP** (`crsp.msf`, `crsp.dsi`, `crsp.msi`, `crsp.msenames`): Monthly stock prices, shares outstanding, daily/monthly market index returns, historical SIC.
- **WRDS / CCM** (`crsp.ccmxpf_linktable`): CRSP-Compustat identifier link.
- **FRED** (via `pandas-datareader`): UNRATE, GDPC1, TB3MS, BOGZ1FL664090005Q, BOGZ1FL664190005Q (AEM leverage), NFCI.
- **Shiller** (`ie_data.xls` from Yale/shillerdata.com): E/P ratio (CAPE-based for levels, trailing E/P for growth rates).

---

## Citation

He, Zhiguo, Bryan Kelly, and Asaf Manela. "Intermediary Asset Pricing: New Evidence from Many Asset Classes." *Journal of Financial Economics* 126, no. 1 (2017): 1-35. https://doi.org/10.1016/j.jfineco.2017.08.002

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a full description of the package structure, key formulas, WRDS tables used, data pipeline, and design decisions.
