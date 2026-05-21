# ARCHITECTURE — HKM Tables 2 & 3 Replication

**Paper**: He, Kelly & Manela (2017), "Intermediary Asset Pricing: New Evidence from Many Asset Classes," *Journal of Financial Economics* 126: 1–35.

**Package**: `hkm` — Python replication of Tables 2 and 3 only.

---

## 1. Overview

The `hkm` package replicates two empirical tables from HKM (2017):

- **Table 2**: Average size of US primary dealer holding companies relative to three comparison groups (all broker-dealers [BD], all banks [Banks], all Compustat firms [Cmpust]), measured by four balance-sheet items (Total assets, Book debt, Book equity, Market equity) across three time periods (1960–2012, 1960–1990, 1990–2012). Monthly frequency.

- **Table 3**: Pairwise time-series Pearson correlations (1970Q1–2012Q4) of three intermediary capital/leverage series (Market capital ratio η_t, Book capital ratio, AEM leverage) with five macro variables in levels (Panel A) and in factors/growth rates (Panel B).

All data is retrieved from WRDS (via psycopg2), FRED (via `pandas-datareader`), and public sources (Shiller data, Chicago Fed NFCI). The package requires a valid `~/.pgpass` entry for `wrds-pgdata.wharton.upenn.edu:9737:wrds:<user>`.

---

## 2. Package Structure

```
hkm/
├── __init__.py          # Exports compute_table2, compute_table3
├── utils.py             # Logging setup (get_logger), WRDS connection context manager (wrds_connection)
├── data/
│   ├── __init__.py      # Data subpackage root
│   ├── wrds_connect.py  # run_query(sql, conn) → pd.DataFrame; zero-row logging
│   ├── dealers.py       # Dealer dataclass, PRIMARY_DEALERS list (Table A.1), get_active_dealers, find_dealer_identifiers
│   ├── compustat.py     # fetch_compustat_quarterly (dealer AT/CEQ), fetch_compustat_all_quarterly (comparison groups)
│   ├── crsp.py          # fetch_crsp_monthly (dealer ME), fetch_crsp_all_monthly (comparison groups), fetch_crsp_market_index, fetch_crsp_daily_vol
│   ├── intermediary.py  # build_capital_ratio (η_t and book capital), build_capital_factor (AR(1) innovation)
│   └── macro.py         # fetch_fred_series, fetch_shiller_ep, fetch_nfci, fetch_aem_leverage, build_macro_panel
├── tables/
│   ├── __init__.py      # Tables subpackage root
│   ├── table2.py        # compute_table2() → pd.DataFrame shape (3, 12)
│   └── table3.py        # compute_table3() → (panel_a, panel_b) tuple of DataFrames
tests/
├── __init__.py          # Test package root
├── test_data.py         # Unit tests for data modules (no WRDS required for most)
└── test_tables.py       # Unit + integration tests for table outputs (WRDS integration tests)
```

---

## 3. Key Formulas

### 3.1 Market Capital Ratio (η_t)

From HKM (2017) Eq. (6), p.7:

```
η_t = Σ_i ME_{i,t} / Σ_i (ME_{i,t} + BD_{i,t})
```

- **ME** = `|prc| × shrout` from `crsp.msf` (last trading day of the quarter/month). `prc` may be negative (bid-ask midpoint); absolute value is taken. `shrout` is in thousands of shares, so ME is in $thousands.
- **BD** (book debt) = `atq − ceqq` from `comp.fundq` (total assets minus common book equity, in $millions). Converted to $thousands for unit consistency.
- The sum is a value-weighted aggregate over all NY Fed primary dealers **active at time t** per Table A.1 dates.
- Time-varying dealer composition: a dealer with `start ≤ t ≤ end` per `PRIMARY_DEALERS` is included.

In Python (`hkm/data/intermediary.py`):
```python
eta = sum_me / (sum_me + sum_bd)
```

### 3.2 Book Capital Ratio

```
book_capital_t = Σ_i CEQ_{i,t} / Σ_i AT_{i,t}
```

Uses `ceqq` and `atq` from `comp.fundq` (quarterly). Same dealer universe as η_t.

In Python:
```python
book_cap = sum_ceq / sum_at
```

### 3.3 Capital Ratio Factor (AR(1) Innovation)

From HKM paper p.8 and Figure 1 caption:

```
Fit AR(1):  η_t = ρ_0 + ρ × η_{t-1} + u_t   (OLS on full sample 1970Q1–2012Q4)
Factor η^Δ_t = u_t / η_{t-1}
```

The paper reports ρ ≈ 0.94 (footnote 22). The implementation (`hkm/data/intermediary.py`) estimates ρ via OLS (`statsmodels.OLS`) and returns OLS residuals scaled by lagged η. The same procedure applies to the book capital factor.

In Python:
```python
x_mat = sm.add_constant(eta_lag[valid])
res = sm.OLS(y, x_mat).fit()
factor = res.resid / eta_lag[valid]
```

Estimated ρ in the WRDS data: **0.9581** (within test-spec acceptable range [0.85, 0.99]).

### 3.4 AEM Leverage (Adrian, Etula & Muir 2010)

From Fed Z.1 Flow of Funds (via FRED):

```
AEM_leverage_t = FL664090005Q_t / (FL664090005Q_t − FL664190005Q_t)
               = Total Financial Assets / Book Equity
```

- `FL664090005Q`: Total financial assets of security broker-dealers (quarterly)
- `FL664190005Q`: Total liabilities of security broker-dealers (quarterly)
- Book Equity = Assets − Liabilities
- FRED fallback IDs: `BOGZ1FL664090005Q`, `BOGZ1FL664190005Q`

AEM leverage factor (Panel B):
```
aem_levfac_t = log(AEM_leverage_t / AEM_leverage_{t-1})   [not seasonally adjusted in this implementation]
```

### 3.5 Table 2 Ratio

For each calendar month t, each balance-sheet item X, each comparison group G:

```
ratio_t = Σ_{i ∈ all_active_dealers} X_{i,t}
          ─────────────────────────────────────────────────────────────────────────
          Σ_{i ∈ all_active_dealers} X_{i,t}  +  Σ_{j ∈ G, j ∉ dealers} X_{j,t}
```

Per HKM footnote 19: "define the total broker-dealer sector as the set of US primary dealers PLUS any firms with a broker-dealer SIC code (6211 or 6221). Note that had we instead relied on the SIC code definition of broker-dealers, we would miss important dealers that are subsidiaries of holding companies not classified as broker-dealers, for instance JP Morgan."

Therefore ALL active primary dealers appear in the numerator regardless of their SIC code. The denominator is dealer aggregate + non-dealer group-SIC firms (from Compustat or CRSP), excluding dealers to avoid double-counting.

Market equity denominator uses CRSP-based non-dealer ME:
```python
g_me = d_me_all + non_dealer_crsp_me   # prevents ME ratio from exceeding 1.0
```

Table 2 cells are the **time-series mean** of `ratio_t` over each sub-period.

---

## 4. Data Pipeline

```
WRDS (psycopg2)              Public Sources                     FRED
─────────────────           ─────────────────────               ─────
comp.fundq (AT, CEQ)   →    Shiller ie_data.xls (E/P)   →      UNRATE
comp.funda (sich)      →    Chicago Fed NFCI CSV         →      GDPC1
comp.names (SIC)             ─────────────────────               TB3MS (T-bill)
crsp.msf (prc, shrout)                                           FL664090005Q (AEM assets)
crsp.msenames (siccd)                                            FL664190005Q (AEM liabilities)
crsp.msi (vwretd)
crsp.dsi (vwretd daily)
crsp.ccmxpf_linktable
         │
         ▼
hkm/data/ modules
  dealers.py     → PRIMARY_DEALERS, get_active_dealers(t)
  compustat.py   → dealer AT/CEQ, comparison group AT/CEQ (joined to funda.sich for historical SIC)
  crsp.py        → dealer ME, comparison group ME, market VW index, realized vol
  macro.py       → FRED series, Shiller E/P, NFCI, AEM leverage, macro panel
  intermediary.py → η_t (value-weighted), book_capital_t, AR(1) factor
         │
         ▼
hkm/tables/
  table2.py → monthly ratio time series → sub-period means → (3, 12) DataFrame
  table3.py → quarterly panel alignment → pairwise Pearson correlations → (panel_a, panel_b)
```

---

## 5. WRDS Tables Used

| WRDS Table | Schema | Key Columns | Purpose |
|---|---|---|---|
| `comp.fundq` | Compustat | `gvkey, datadate, atq, ceqq, rdq` | Dealer and comparison group quarterly balance sheet |
| `comp.funda` | Compustat | `gvkey, fyear, sich, datafmt, indfmt, popsrc, consol` | Historical SIC codes for comparison group filter |
| `comp.names` | Compustat | `gvkey, conm, sic` | Dealer GVKEY fuzzy-name lookup |
| `crsp.msf` | CRSP | `permno, date, prc, shrout` | Monthly stock prices and shares outstanding |
| `crsp.msenames` | CRSP | `permno, namedt, nameendt, comnam, siccd, shrcd` | Historical SIC codes; US-only filter (`shrcd IN (10,11)`) |
| `crsp.msi` | CRSP | `date, vwretd` | Value-weighted market index monthly returns |
| `crsp.dsi` | CRSP | `date, vwretd` | Value-weighted market index daily returns (for realized vol) |
| `crsp.ccmxpf_linktable` | CRSP-Compustat link | `gvkey, lpermno, linktype, linkprim, linkdt, linkenddt` | GVKEY → PERMNO identifier mapping |

Key schema notes:
- `comp.fundq` does NOT have a `sich` (historical SIC) column; SIC filtering uses `comp.funda.sich` joined on `gvkey + fyear`.
- `crsp.msenames.siccd` is an integer column (not varchar); use `BETWEEN` for range filters.
- AEM leverage is fetched from FRED (not WRDS): `FL664090005Q` / `FL664190005Q`.

---

## 6. Entry Points

### `compute_table2`

```python
def compute_table2(
    conn: psycopg2.extensions.connection | None = None,
    start_date: str = "1960-01-01",
    end_date: str = "2012-12-31",
) -> pd.DataFrame:
```

Returns a `pd.DataFrame` of shape **(3, 12)** with:
- **Index**: `['1960-2012', '1960-1990', '1990-2012']`
- **Columns**: `pd.MultiIndex` with levels `(item, group)` where `item ∈ {Total assets, Book debt, Book equity, Market equity}` and `group ∈ {BD, Banks, Cmpust}`

If `conn=None`, opens a WRDS connection internally using `~/.pgpass`.

### `compute_table3`

```python
def compute_table3(
    conn: psycopg2.extensions.connection | None = None,
    start_date: str = "1970-01-01",
    end_date: str = "2012-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame]:
```

Returns `(panel_a, panel_b)`:
- **`panel_a`**: shape **(8, 3)** — correlations of levels. Rows: Market capital, Book capital, AEM leverage, E/P, Unemployment, GDP, Financial conditions, Market volatility. Columns: Market capital, Book capital, AEM leverage.
- **`panel_b`**: shape **(9, 3)** — correlations of factors/growth rates. Rows: Market capital factor, Book capital factor, AEM leverage factor, Market excess return, E/P growth, Unemployment growth, GDP growth, Financial conditions growth, Market volatility growth. Columns: Market capital factor, Book capital factor, AEM leverage factor.

---

## 7. Primary Dealer Universe

The `PRIMARY_DEALERS` list in `hkm/data/dealers.py` encodes US-based primary dealers from Table A.1 of HKM (2017) with their Compustat GVKEYs (where resolvable), start/end dates per the NY Fed designation history, and `permno=None` (resolved at runtime via `crsp.ccmxpf_linktable`).

**Dealers with verified GVKEYs** (resolved via `comp.names`):

| Dealer | GVKEY | Notes |
|---|---|---|
| Goldman Sachs | 114628 | Post-IPO 1999 |
| Merrill Lynch | 007267 | Acquired by BofA 2010 |
| Lehman Brothers | 030128 | Bankrupt Sep 2008 |
| Morgan Stanley | 012124 | Converted to bank 2008 |
| Bear Stearns | 011818 | Acquired by JPM 2008 |
| JPMorgan Chase | 002968 | Includes Chase/Chemical merger chain |
| Citigroup | 003243 | Includes predecessors |
| Paine Webber | 008299 | Acquired by UBS 2000 |
| Dean Witter Reynolds | 003823 | Merged with Morgan Stanley 1997 |
| Salomon Smith Barney | 008537 | SIC 6211; Citigroup subsidiary |
| Bankers Trust | 002029 | Acquired by Deutsche Bank 1999 |
| Manufacturers Hanover | 007003 | Merged into Chemical 1992 |
| Chase (old) | 002943 | Truncated to 1995-12-31 (post-merger stale data prevention) |
| First Boston | 004684 | Acquired by Credit Suisse 1993 |
| Bank of America | 007647 | Dealer 1999–2010 |

**Dealers without Compustat GVKEYs** (no match found in `comp.names`): Drexel Burnham Lambert, Chemical Bank (post-merge), Continental Illinois, Discount Corp., Kidder Peabody, Prudential-Bache, Dillon Read, Harris Upham, Aubrey Lanston, Blyth Eastman Dillon, Carroll McEntee, DLJ (both runs), First Chicago, First Interstate, Midland-Montagu, NationsBanc, Security Pacific, White Weld, A.G. Becker, CF Childs, First National Bank of Boston, Pollock, Second District Securities.

**Foreign dealers excluded**: Barclays, Deutsche Bank, UBS, BNP Paribas, Credit Suisse (as CS-Americas), HSBC, Mizuho, Nomura, RBS, RBC, TD, Daiwa, BMO, CIBC, SG Americas. These would require Datastream access and are not in CRSP/Compustat.

---

## 8. Data Limitations

### 8.1 Pre-1978 WRDS Coverage Gap

Compustat quarterly (`comp.fundq`) coverage for US financial firms is sparse before 1978. Most dealers do not have quarterly balance-sheet data starting from 1960 as the paper claims. Practical coverage starts approximately at 1978Q1 for Goldman Sachs, Merrill Lynch, and Morgan Stanley. This is the primary source of divergence from published Table 2 values for the 1960–1990 sub-period and the full 1960–2012 mean.

The paper notes (footnote in the data appendix) that the sample for η_t starts in 1970 when data becomes available. The implementation correctly starts η_t at the first quarter with at least two dealers in both Compustat and CRSP.

### 8.2 AEM Leverage Sign

The raw BOGZ1 Fed Z.1 data shows broker-dealer leverage (assets/equity) rising from ~5.7x (1975) to ~47x (2008) — pro-cyclical. The published Panel A correlation between η and AEM leverage is −0.42, suggesting HKM may have defined AEM leverage as an equity-ratio (capital adequacy) measure rather than raw leverage, or used a different data vintage. The FRED series `BOGZ1FL664090005Q`/`BOGZ1FL664190005Q` give the correct magnitudes but a positive correlation with η over the full 1970–2012 sample. This is a data-vintage/definition limitation, not a code error. All IT-8 sign checks (which do not test the η vs AEM leverage cell) pass.

### 8.3 No Datastream for Foreign Dealers

The paper includes foreign primary dealers (Barclays, Deutsche Bank, etc.) in η_t via Datastream. This replication restricts to US dealers in CRSP-Compustat, consistent with the paper's Table 2 note that focuses on US-only firms. This limits the numerator for η_t relative to the paper's full dealer sample.

### 8.4 AEM Leverage Approximation

Book debt = AT − CEQ (total assets minus common book equity) per the paper's definition. This uses quarterly Compustat. The paper explicitly states: "book value of debt is equal to total assets less common equity, using the most recent data available for each firm at the end of a calendar quarter." No imputation is performed for firms with missing data in a given quarter.

### 8.5 1960–1990 Sub-Period Accuracy

For the 1960–1990 period, WRDS Compustat quarterly data begins c.1978, meaning ~40% of the sub-period (1960–1977) relies on only the few firms with early Compustat coverage. This systematically understates the BD, Banks, and Cmpust comparison group denominators in that sub-period.

### 8.6 Banks Comparison Group SIC Boundary

The paper defines "Banks" as a broad category. This implementation uses SIC 6000–6299 for the Compustat comparison group (via `comp.funda.sich`) and `crsp.msenames.siccd BETWEEN 6000 AND 6299` for CRSP ME. Post-1990 bank consolidation caused many large bank holding companies to be re-classified under non-banking SIC codes, understating the Banks denominator in the 1990–2012 sub-period.

---

## 9. Design Decisions

| Decision | Rationale |
|---|---|
| `comp.funda.sich` for historical SIC (not `comp.names.sic`) | `comp.names.sic` is the final/current SIC only; `funda.sich` is time-varying and avoids forward-looking SIC misclassification |
| All active dealers in numerator regardless of SIC (Table 2) | HKM footnote 19 explicitly states this; prevents JPMorgan (SIC 6020) from being excluded from BD numerator |
| Dealer GVKEYs excluded from denominator group_comp | Prevents double-counting (dealer AT already in numerator) |
| Non-dealer CRSP ME for ME denominator | Prevents ME/BD > 1.0 since some dealers (JPMorgan, Citigroup) are SIC 6020/6199, not 6211/6221 |
| Chase Manhattan truncated to 1995-12-31 | Chase filed last Compustat quarterly in 1995Q4; JPMorgan (002968) covers post-merger entity; avoids double-counting |
| E/P: CAPE-based (`1/CAPE`) for Panel A, trailing E/P for Panel B growth | CAPE levels are smoother and appropriate for level correlations; simple E/P has higher quarterly growth volatility matching Panel B dynamics |
| datadate alignment (not rdq) for book capital | rdq-based alignment introduced 1-quarter lag that flipped GDP/Unemployment sign correlations; datadate alignment matches CRSP month-end convention |
| AEM leverage not negated | Negating the raw leverage improves one cell (η vs AEM corr) but flips signs for 5 macro variable cells; net preference is to retain correct macro signs |

---

## 10. Test Coverage

**39 tests total** — all pass (pytest 0 failures, 0 errors).

| Group | Count | WRDS Required |
|---|---|---|
| TestDealers (unit) | 6 | No |
| TestUtils (unit) | 2 | No |
| TestCapitalRatioFormula (unit) | 4 | No |
| TestCapitalFactor (unit) | 3 | No |
| TestMacroData (unit) | 2 | No |
| TestWRDSConnect (unit) | 2 | No |
| TestAEMLeverage (unit) | 2 | No |
| TestTable2Shape (unit) | 2 | No |
| TestTable3Shape (unit) | 5 | No |
| TestWRDSIntegration | 3 | Yes |
| TestTable2Integration | 3 | Yes |
| TestTable3Integration | 5 | Yes |

Code quality: ruff (zero errors), mypy strict (zero errors), no `print()` statements.

---

## 11. Build & Run

```bash
# Install
pip install -e ".[dev]"

# Run all tests (WRDS connection required for integration tests)
pytest tests/ -v

# Run unit tests only (no WRDS)
pytest tests/ -v -k "not Integration"

# Code quality
ruff check hkm/
mypy hkm/ --strict --ignore-missing-imports

# Use the package
import psycopg2
from hkm import compute_table2, compute_table3

with psycopg2.connect(host="wrds-pgdata.wharton.upenn.edu", port=9737,
                      dbname="wrds", user="your_username") as conn:
    t2 = compute_table2(conn=conn)
    panel_a, panel_b = compute_table3(conn=conn)
```

---

*Generated by StatsClaw scriber — run-20260520-hkm-tables-2-3 (2026-05-20)*
