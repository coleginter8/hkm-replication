# HKM (2017) Replication: Tables 2 and 3

This package replicates Tables 2 and 3 from He, Kelly & Manela (2017),
"Intermediary Asset Pricing: New Evidence from Many Asset Classes,"
*Journal of Financial Economics* 126: 1–35.

Table 2 reports the average size of primary US broker-dealers relative to three
comparison groups (all broker-dealers, all commercial banks, all Compustat firms)
across four balance-sheet items (total assets, book debt, book equity, market equity)
for three sub-periods (1960–2012, 1960–1990, 1990–2012).

Table 3 reports pairwise time-series correlations (1970Q1–2012Q4) between the
primary dealer market capital ratio, book capital ratio, AEM leverage, and several
macro variables — both in levels (Panel A) and as AR(1)-based factors (Panel B).

---

## Data Sources

- **WRDS / Compustat** (`comp.fundq`): Quarterly balance sheet data (AT, CEQ) for primary dealers and comparison groups.
- **WRDS / CRSP** (`crsp.msf`, `crsp.dsi`, `crsp.msi`, `crsp.msenames`): Monthly stock prices, shares outstanding, and daily/monthly market index returns.
- **WRDS / CCM** (`crsp.ccmxpf_linktable`): CRSP–Compustat identifier link.
- **FRED** (via `pandas-datareader`): UNRATE, GDPC1, TB3MS, FL664090005Q, FL664190005Q (AEM leverage).
- **Shiller** (`ie_data.xls` from Yale): E/P ratio.
- **Chicago Fed** (FRED series `NFCI`): National Financial Conditions Index.

---

## Setup

### 1. WRDS credentials

Create `~/.pgpass` with the following entry (replace `<password>`):

```
wrds-pgdata.wharton.upenn.edu:9737:wrds:coleginter:<password>
```

Ensure the file has restricted permissions:
```bash
chmod 600 ~/.pgpass
```

### 2. Install the package

```bash
pip install -e ".[dev]"
```

### 3. Run checks

```bash
ruff check hkm/
mypy hkm/ --strict --ignore-missing-imports
pytest tests/ -x --tb=short
```

---

## Usage

```python
from hkm import compute_table2, compute_table3
from hkm.utils import wrds_connection

with wrds_connection() as conn:
    t2 = compute_table2(conn=conn)
    panel_a, panel_b = compute_table3(conn=conn)

print(t2)
print(panel_a)
print(panel_b)
```

The connection context manager reads credentials from `~/.pgpass` automatically.

---

## Package Structure

```
hkm/
├── __init__.py           # Exports compute_table2, compute_table3
├── utils.py              # Logging and WRDS connection context manager
├── data/
│   ├── wrds_connect.py   # SQL query helper
│   ├── compustat.py      # Compustat quarterly balance-sheet fetcher
│   ├── crsp.py           # CRSP monthly/daily stock and index data
│   ├── dealers.py        # Primary dealer static list + identifier resolution
│   ├── intermediary.py   # Capital ratio η_t and capital factor construction
│   └── macro.py          # FRED, Shiller, NFCI macro series
└── tables/
    ├── table2.py         # compute_table2()
    └── table3.py         # compute_table3()
```
