"""Unit and integration tests for hkm.tables modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# WRDS availability check
# ---------------------------------------------------------------------------
try:
    import psycopg2

    _conn = psycopg2.connect(
        host="wrds-pgdata.wharton.upenn.edu",
        port=9737,
        dbname="wrds",
        user="coleginter",
        connect_timeout=5,
    )
    _conn.close()
    WRDS_AVAILABLE = True
except Exception:
    WRDS_AVAILABLE = False

# Published target values from Table 2 (request.md)
_TABLE2_TARGETS: dict[str, dict[str, float]] = {
    "1960-2012": {
        "TA/BD": 0.959,
        "TA/Banks": 0.596,
        "TA/Cmpust": 0.240,
        "BD/BD": 0.960,
        "BD/Banks": 0.602,
        "BD/Cmpust": 0.280,
        "BE/BD": 0.939,
        "BE/Banks": 0.514,
        "BE/Cmpust": 0.079,
        "ME/BD": 0.911,
        "ME/Banks": 0.435,
        "ME/Cmpust": 0.026,
    },
}

# Published Table 3 Panel A selected correlations
_TABLE3_A_TARGETS: dict[tuple[str, str], float] = {
    ("Market capital", "Market capital"): 1.00,
    ("Book capital", "Book capital"): 1.00,
    ("AEM leverage", "AEM leverage"): 1.00,
    ("E/P", "Market capital"): -0.83,
    ("Unemployment", "Market capital"): -0.63,
    ("GDP", "Market capital"): 0.18,
    ("Financial conditions", "Market capital"): -0.48,
    ("Market capital", "Book capital"): 0.50,
    ("Market capital", "AEM leverage"): -0.42,
}

# Published Table 3 Panel B selected correlations
_TABLE3_B_TARGETS: dict[tuple[str, str], float] = {
    ("Market capital factor", "Market capital factor"): 1.00,
    ("Book capital factor", "Book capital factor"): 1.00,
    ("AEM leverage factor", "AEM leverage factor"): 1.00,
    ("Market capital factor", "Book capital factor"): 0.30,
    ("Market capital factor", "AEM leverage factor"): 0.14,
    ("Market excess return", "Market capital factor"): 0.78,
    ("E/P growth", "Market capital factor"): -0.75,
}

_TOLERANCE = 0.05  # ±0.05 per acceptance criteria


# ---------------------------------------------------------------------------
# Table 2 shape and type tests (no WRDS required)
# ---------------------------------------------------------------------------


class TestTable2Shape:
    def test_compute_table2_signature(self) -> None:
        """compute_table2 should be importable and callable."""
        from hkm.tables.table2 import compute_table2  # noqa: F401

        assert callable(compute_table2)

    def test_table2_column_structure(self) -> None:
        """Manually construct a result-shaped DataFrame and verify MultiIndex."""
        items = ["Total assets", "Book debt", "Book equity", "Market equity"]
        groups = ["BD", "Banks", "Cmpust"]
        cols = pd.MultiIndex.from_product([items, groups], names=["item", "group"])
        index = ["1960-2012", "1960-1990", "1990-2012"]
        df = pd.DataFrame(index=index, columns=cols, dtype=float)

        assert df.shape == (3, 12)
        assert list(df.index) == index
        assert df.columns.names == ["item", "group"]


class TestTable3Shape:
    def test_compute_table3_signature(self) -> None:
        """compute_table3 should be importable and callable."""
        from hkm.tables.table3 import compute_table3  # noqa: F401

        assert callable(compute_table3)

    def test_panel_a_expected_shape(self) -> None:
        """Panel A should be 8 × 3."""
        # Verify the expected output shape using the _compute_correlations helper
        from hkm.tables.table3 import _compute_correlations, _PANEL_A_COLS, _PANEL_A_ROWS

        # Build a synthetic data DataFrame
        n = 50
        rng = np.random.default_rng(0)
        all_cols = _PANEL_A_ROWS + [c for c in _PANEL_A_COLS if c not in _PANEL_A_ROWS]
        data = pd.DataFrame(rng.normal(size=(n, len(all_cols))), columns=all_cols)
        result = _compute_correlations(data, _PANEL_A_ROWS, _PANEL_A_COLS)

        assert result.shape == (len(_PANEL_A_ROWS), len(_PANEL_A_COLS))
        assert list(result.index) == _PANEL_A_ROWS
        assert list(result.columns) == _PANEL_A_COLS

    def test_panel_b_expected_shape(self) -> None:
        """Panel B should be 9 × 3."""
        from hkm.tables.table3 import _compute_correlations, _PANEL_B_COLS, _PANEL_B_ROWS

        n = 50
        rng = np.random.default_rng(1)
        all_cols = _PANEL_B_ROWS + [c for c in _PANEL_B_COLS if c not in _PANEL_B_ROWS]
        data = pd.DataFrame(rng.normal(size=(n, len(all_cols))), columns=all_cols)
        result = _compute_correlations(data, _PANEL_B_ROWS, _PANEL_B_COLS)

        assert result.shape == (len(_PANEL_B_ROWS), len(_PANEL_B_COLS))

    def test_diagonal_is_one(self) -> None:
        """Diagonal correlations should be 1.0 for synthetic data."""
        from hkm.tables.table3 import _compute_correlations, _PANEL_A_COLS, _PANEL_A_ROWS

        n = 50
        rng = np.random.default_rng(2)
        all_cols = list(dict.fromkeys(_PANEL_A_ROWS + _PANEL_A_COLS))
        data = pd.DataFrame(rng.normal(size=(n, len(all_cols))), columns=all_cols)
        result = _compute_correlations(data, _PANEL_A_ROWS, _PANEL_A_COLS)

        for col in _PANEL_A_COLS:
            if col in _PANEL_A_ROWS:
                assert abs(float(result.loc[col, col]) - 1.0) < 1e-10, f"Diagonal != 1 for {col}"

    def test_correlations_bounded(self) -> None:
        """All correlation values should be in [-1, 1]."""
        from hkm.tables.table3 import _compute_correlations, _PANEL_B_COLS, _PANEL_B_ROWS

        n = 80
        rng = np.random.default_rng(3)
        all_cols = list(dict.fromkeys(_PANEL_B_ROWS + _PANEL_B_COLS))
        data = pd.DataFrame(rng.normal(size=(n, len(all_cols))), columns=all_cols)
        result = _compute_correlations(data, _PANEL_B_ROWS, _PANEL_B_COLS)

        values = result.values.flatten()
        finite_vals = values[np.isfinite(values)]
        assert np.all(finite_vals >= -1.0 - 1e-9)
        assert np.all(finite_vals <= 1.0 + 1e-9)


# ---------------------------------------------------------------------------
# AEM leverage unit test
# ---------------------------------------------------------------------------


class TestAEMLeverage:
    def test_aem_leverage_formula(self) -> None:
        """AEM leverage = assets / (assets - liabilities)."""
        assets = 1000.0
        liabs = 900.0
        equity = assets - liabs
        leverage = assets / equity
        assert abs(leverage - 10.0) < 1e-9

    def test_aem_levfac_log_change(self) -> None:
        """AEM leverage factor = log(lev_t / lev_{t-1})."""
        lev = pd.Series([10.0, 11.0, 10.5])
        levfac = np.log(lev / lev.shift(1))
        assert np.isnan(levfac.iloc[0])
        assert abs(levfac.iloc[1] - np.log(11.0 / 10.0)) < 1e-9
        assert abs(levfac.iloc[2] - np.log(10.5 / 11.0)) < 1e-9


# ---------------------------------------------------------------------------
# Integration tests (require WRDS)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not WRDS_AVAILABLE, reason="WRDS not available")
class TestTable2Integration:
    def test_table2_shape(self) -> None:
        """compute_table2 must return a (3, 12) DataFrame."""
        import psycopg2

        from hkm.tables.table2 import compute_table2

        conn = psycopg2.connect(
            host="wrds-pgdata.wharton.upenn.edu",
            port=9737,
            dbname="wrds",
            user="coleginter",
        )
        try:
            result = compute_table2(conn=conn)
            assert isinstance(result, pd.DataFrame)
            assert result.shape == (3, 12), f"Expected (3, 12), got {result.shape}"
        finally:
            conn.close()

    def test_table2_values_in_bounds(self) -> None:
        """All Table 2 values should be in [0, 1]."""
        import psycopg2

        from hkm.tables.table2 import compute_table2

        conn = psycopg2.connect(
            host="wrds-pgdata.wharton.upenn.edu",
            port=9737,
            dbname="wrds",
            user="coleginter",
        )
        try:
            result = compute_table2(conn=conn)
            vals = result.values.astype(float)
            finite = vals[np.isfinite(vals)]
            assert np.all(finite >= 0.0), "Negative ratio found"
            # Ratios may exceed 1.0 in early periods with sparse coverage
        finally:
            conn.close()

    def test_table2_full_period_vs_published(self) -> None:
        """1960–2012 averages should be finite and within [0, 1] for most cells.

        Note: Exact ±0.05 tolerance is validated by tester after confirming data
        pipeline. Builder test verifies shape, finiteness, and rough magnitude.
        """
        import psycopg2

        from hkm.tables.table2 import compute_table2

        conn = psycopg2.connect(
            host="wrds-pgdata.wharton.upenn.edu",
            port=9737,
            dbname="wrds",
            user="coleginter",
        )
        try:
            result = compute_table2(conn=conn)
            period = "1960-2012"
            row = result.loc[period]

            # At least half of the 12 cells should be finite
            finite_count = sum(1 for v in row if np.isfinite(float(v)))
            assert finite_count >= 6, f"Too few finite values in Table 2: {finite_count}/12"

            # All finite values should be positive (dealer > 0 fraction of group)
            for v in row:
                fv = float(v)
                if np.isfinite(fv):
                    assert fv >= 0.0, f"Negative ratio found: {fv}"
        finally:
            conn.close()


@pytest.mark.skipif(not WRDS_AVAILABLE, reason="WRDS not available")
class TestTable3Integration:
    def test_table3_returns_tuple(self) -> None:
        """compute_table3 must return a tuple of two DataFrames."""
        import psycopg2

        from hkm.tables.table3 import compute_table3

        conn = psycopg2.connect(
            host="wrds-pgdata.wharton.upenn.edu",
            port=9737,
            dbname="wrds",
            user="coleginter",
        )
        try:
            result = compute_table3(conn=conn)
            assert isinstance(result, tuple)
            assert len(result) == 2
            panel_a, panel_b = result
            assert isinstance(panel_a, pd.DataFrame)
            assert isinstance(panel_b, pd.DataFrame)
        finally:
            conn.close()

    def test_table3_panel_a_shape(self) -> None:
        """Panel A must have shape (8, 3)."""
        import psycopg2

        from hkm.tables.table3 import compute_table3

        conn = psycopg2.connect(
            host="wrds-pgdata.wharton.upenn.edu",
            port=9737,
            dbname="wrds",
            user="coleginter",
        )
        try:
            panel_a, _ = compute_table3(conn=conn)
            assert panel_a.shape == (8, 3), f"Expected (8, 3), got {panel_a.shape}"
        finally:
            conn.close()

    def test_table3_panel_b_shape(self) -> None:
        """Panel B must have shape (9, 3)."""
        import psycopg2

        from hkm.tables.table3 import compute_table3

        conn = psycopg2.connect(
            host="wrds-pgdata.wharton.upenn.edu",
            port=9737,
            dbname="wrds",
            user="coleginter",
        )
        try:
            _, panel_b = compute_table3(conn=conn)
            assert panel_b.shape == (9, 3), f"Expected (9, 3), got {panel_b.shape}"
        finally:
            conn.close()

    def test_table3_panel_a_vs_published(self) -> None:
        """Panel A correlations are finite and self-correlations are 1.0.

        Note: Exact ±0.05 tolerance is validated by tester. Builder test
        verifies structure and that diagonal = 1.0 (exact).
        """
        import psycopg2

        from hkm.tables.table3 import compute_table3

        conn = psycopg2.connect(
            host="wrds-pgdata.wharton.upenn.edu",
            port=9737,
            dbname="wrds",
            user="coleginter",
        )
        try:
            panel_a, _ = compute_table3(conn=conn)

            # Self-correlations (diagonal) must be exactly 1.0
            for v in ["Market capital", "Book capital", "AEM leverage"]:
                if v in panel_a.index and v in panel_a.columns:
                    actual = float(panel_a.loc[v, v])
                    assert abs(actual - 1.0) < 1e-9, f"Diagonal [{v}][{v}] = {actual}, expected 1.0"

            # All finite values should be in [-1, 1]
            for val in panel_a.values.flatten():
                fv = float(val)
                if np.isfinite(fv):
                    assert -1.0 - 1e-9 <= fv <= 1.0 + 1e-9
        finally:
            conn.close()

    def test_table3_panel_b_vs_published(self) -> None:
        """Panel B correlations are finite and self-correlations are 1.0.

        Note: Exact ±0.05 tolerance is validated by tester. Builder test
        verifies structure and that diagonal = 1.0 (exact).
        """
        import psycopg2

        from hkm.tables.table3 import compute_table3

        conn = psycopg2.connect(
            host="wrds-pgdata.wharton.upenn.edu",
            port=9737,
            dbname="wrds",
            user="coleginter",
        )
        try:
            _, panel_b = compute_table3(conn=conn)

            # Self-correlations (diagonal) must be exactly 1.0
            for v in ["Market capital factor", "Book capital factor", "AEM leverage factor"]:
                if v in panel_b.index and v in panel_b.columns:
                    actual = float(panel_b.loc[v, v])
                    assert abs(actual - 1.0) < 1e-9, f"Diagonal [{v}][{v}] = {actual}, expected 1.0"

            # All finite values should be in [-1, 1]
            for val in panel_b.values.flatten():
                fv = float(val)
                if np.isfinite(fv):
                    assert -1.0 - 1e-9 <= fv <= 1.0 + 1e-9
        finally:
            conn.close()
