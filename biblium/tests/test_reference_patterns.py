# -*- coding: utf-8 -*-
"""
Regression tests for ``biblium.reference_patterns`` (new in 2.19).

All expectations are computed by hand on tiny synthetic inputs; the tests are
deterministic and hermetic (no network, no files except a temporary figure).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from biblium import reference_patterns as RP


def _ref(author: str, year: int) -> str:
    return f"{author}, Title, Journal, 10, 2, pp. 1-20, ({year})"


# ---------------------------------------------------------------------------
# RPYS: centered vs backward median
# ---------------------------------------------------------------------------

class TestRPYSBackward:
    @pytest.fixture
    def df(self):
        # cited years: 2000 x2, 2001, 2003, 2007 -> 2002, 2004-2006 have 0
        return pd.DataFrame({
            "References": [
                "; ".join([_ref("A", 2000), _ref("B", 2000), _ref("C", 2001)]),
                "; ".join([_ref("D", 2003), _ref("E", 2007)]),
            ]
        })

    def test_rolling_median_helper(self):
        s = pd.Series([2, 1, 0, 1], dtype=float)
        assert RP._rolling_median(s, 3, "centered").tolist() == pytest.approx([1.5, 1.0, 1.0, 0.5])
        assert RP._rolling_median(s, 3, "backward").tolist() == pytest.approx([2.0, 1.5, 1.0, 1.0])

    def test_centered_matches_compute_rpys(self, df):
        from biblium.utilsbib import compute_rpys
        a = RP.compute_rpys_backward(df, median_window=3, mode="centered")
        b = compute_rpys(df, median_window=3)
        assert a["Cited Year"].tolist() == list(range(2000, 2008))
        assert a["Citations"].tolist() == [2, 1, 0, 1, 0, 0, 0, 1]
        assert a["Median"].tolist() == pytest.approx(b["Median"].tolist())
        assert a["Deviation"].tolist() == pytest.approx(b["Deviation"].tolist())

    def test_backward_median_and_deviation_pct(self, df):
        out = RP.compute_rpys_backward(df, median_window=3, mode="backward").set_index("Cited Year")
        assert out.loc[2000, "Median"] == 2.0           # median(2)
        assert out.loc[2001, "Median"] == 1.5           # median(2, 1)
        assert out.loc[2002, "Median"] == 1.0           # median(2, 1, 0)
        assert out.loc[2003, "Median"] == 1.0           # median(1, 0, 1)
        assert out.loc[2001, "Deviation"] == -0.5
        assert out.loc[2001, "Deviation %"] == pytest.approx(-100 / 3)
        # 2006: window (0, 0, 0) -> median 0 -> Deviation % NaN
        assert out.loc[2006, "Median"] == 0.0 and np.isnan(out.loc[2006, "Deviation %"])
        assert out.loc[2007, "Median"] == 0.0           # median(0, 0, 1)
        assert out.loc[2007, "Deviation"] == 1.0

    def test_invalid_mode_and_missing_column(self, df):
        with pytest.raises(ValueError):
            RP.compute_rpys_backward(df, mode="forward")
        assert RP.compute_rpys_backward(pd.DataFrame({"x": [1]})).empty


# ---------------------------------------------------------------------------
# Cited-reference time series
# ---------------------------------------------------------------------------

class TestCitedReferenceTimeseries:
    @pytest.fixture
    def df(self):
        smith = "Smith, J., Alpha, J, 1, pp. 1-2, (2001)"
        jones = "Jones, K., Beta, J, 2, pp. 3-4, (2002)"
        return pd.DataFrame({
            "Year": [2010, 2011, 2011],
            "References": [
                f"{smith}; {jones}",
                # lowercase variant with trailing period; duplicate Jones in the same doc
                f"{smith.lower()}.; {jones}; {jones}",
                f"{smith}; Lee, Gamma, no year here",
            ],
        })

    def test_normalized_counts_and_spelling(self, df):
        ts = RP.cited_reference_timeseries(df, min_citations=2)
        assert list(ts.columns) == ["Reference", "Cited Year", "Citing Year", "Citations"]
        assert set(ts["Reference"].str[:5]) == {"Smith", "Jones"}
        smith = ts[ts["Reference"].str.startswith("Smith")].set_index("Citing Year")
        assert smith["Citations"].to_dict() == {2010: 1, 2011: 2}
        assert smith["Cited Year"].unique().tolist() == [2001.0]
        jones = ts[ts["Reference"].str.startswith("Jones")].set_index("Citing Year")
        assert jones["Citations"].to_dict() == {2010: 1, 2011: 1}   # duplicate inside doc counted once
        # Lee cited once -> dropped by min_citations
        assert not ts["Reference"].str.startswith("Lee").any()

    def test_min_citations_one_keeps_year_nan(self, df):
        ts = RP.cited_reference_timeseries(df, min_citations=1)
        lee = ts[ts["Reference"].str.startswith("Lee")]
        assert len(lee) == 1 and np.isnan(lee["Cited Year"].iloc[0])

    def test_without_normalization(self, df):
        ts = RP.cited_reference_timeseries(df, min_citations=2, normalize=False)
        smith = ts[ts["Reference"].str.startswith("Smith")]
        assert smith["Citations"].sum() == 2   # the lowercase variant is a separate reference (1 cit.)

    def test_missing_year_rows_ignored(self):
        df = pd.DataFrame({"Year": [2010, np.nan], "References": [_ref("A", 2000), _ref("A", 2000)]})
        ts = RP.cited_reference_timeseries(df, min_citations=1)
        assert ts["Citations"].sum() == 1


# ---------------------------------------------------------------------------
# Pattern classification
# ---------------------------------------------------------------------------

def _ts(series: dict) -> pd.DataFrame:
    rows = []
    for ref, yrs in series.items():
        for y, n in yrs.items():
            rows.append({"Reference": ref, "Cited Year": 1999.0, "Citing Year": y, "Citations": n})
    return pd.DataFrame(rows)


class TestClassifyReferencePatterns:
    @pytest.fixture
    def ts(self):
        return _ts({
            "hot": {2019: 1, 2022: 2, 2023: 5, 2024: 8},
            "constant": {y: 3 for y in range(2010, 2021)},
            "lifecycle": {2010: 1, 2011: 3, 2012: 6, 2013: 3, 2014: 1, 2015: 1},
            "beauty": {2000: 1, 2002: 1, 2004: 1, 2006: 5, 2007: 6, 2008: 7, 2009: 6},
            "insufficient": {2015: 4, 2016: 2},
            "other": {2010: 5, 2011: 1, 2012: 1, 2013: 5},
        })

    @pytest.fixture
    def cls(self, ts):
        return RP.classify_reference_patterns(ts).set_index("Reference")

    def test_patterns(self, cls):
        assert cls.loc["hot", "Pattern"] == "Hot Paper"
        assert cls.loc["constant", "Pattern"] == "Constant Performer"
        assert cls.loc["lifecycle", "Pattern"] == "Life Cycle"
        assert cls.loc["beauty", "Pattern"] == "Sleeping Beauty"
        assert cls.loc["insufficient", "Pattern"] == "Insufficient data"
        assert cls.loc["other", "Pattern"] == "Other"

    def test_hot_statistics(self, cls):
        # current year = 2024; recent window 2022-2024 -> 15 of 16 citations
        assert cls.loc["hot", "Recent share"] == pytest.approx(15 / 16)
        assert cls.loc["hot", "Last citing year"] == 2024
        assert cls.loc["hot", "N citing years"] == 4 and cls.loc["hot", "Span"] == 6

    def test_constant_statistics(self, cls):
        r = cls.loc["constant"]
        assert r["Span"] == 11 and r["Total citations"] == 33
        assert r["Mean per year"] == pytest.approx(3.0) and r["CV"] == pytest.approx(0.0)
        assert r["Peak year"] == 2010 and r["Dormancy years"] == 0

    def test_lifecycle_statistics(self, cls):
        r = cls.loc["lifecycle"]
        assert r["Peak year"] == 2012 and r["Peak citations"] == 6
        assert r["Dormancy years"] == 1   # 2010 has 1 citation, 2011 has 3
        assert r["Cited Year"] == 1999.0

    def test_sleeping_beauty_statistics(self, cls):
        r = cls.loc["beauty"]
        # full series 2000-2009: 1,0,1,0,1,0,5,6,7,6 -> 6 dormant years
        assert r["Dormancy years"] == 6 and r["Total citations"] == 27
        assert r["CV"] == pytest.approx(np.sqrt(7.61) / 2.7)

    def test_rule_order_hot_before_constant(self):
        # constant series up to the current year with a strong recent share -> Hot Paper wins
        ts = _ts({"x": {y: 3 for y in range(2020, 2025)}, "y": {2000: 1, 2001: 1, 2002: 1}})
        out = RP.classify_reference_patterns(ts, hot_recent_share=0.5).set_index("Reference")
        assert out.loc["x", "Pattern"] == "Hot Paper"
        out = RP.classify_reference_patterns(ts, hot_recent_share=0.9, constant_min_span=5).set_index("Reference")
        assert out.loc["x", "Pattern"] == "Constant Performer"

    def test_current_year_override(self, ts):
        # with current_year far in the future nothing is a Hot Paper
        out = RP.classify_reference_patterns(ts, current_year=2040).set_index("Reference")
        assert out.loc["hot", "Pattern"] != "Hot Paper"
        assert out.loc["hot", "Recent share"] == 0.0

    def test_sleeping_beauty_needs_awakening(self):
        ts = _ts({"never": {2000: 1, 2003: 1, 2006: 1, 2009: 1}})
        out = RP.classify_reference_patterns(ts).set_index("Reference")
        assert out.loc["never", "Dormancy years"] == 10
        assert out.loc["never", "Pattern"] != "Sleeping Beauty"

    def test_empty_input(self):
        out = RP.classify_reference_patterns(pd.DataFrame(columns=["Reference", "Citing Year", "Citations"]))
        assert out.empty and "Pattern" in out.columns

    def test_summary(self, ts):
        cls = RP.classify_reference_patterns(ts)
        s = RP.rpys_pattern_summary(cls).set_index("Pattern")
        assert s["N"].sum() == 6 and s["%"].sum() == pytest.approx(100.0)
        assert s.loc["Hot Paper", "Top references"] == "hot"
        assert list(s.index) == ["Hot Paper", "Sleeping Beauty", "Constant Performer",
                                 "Life Cycle", "Other", "Insufficient data"]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

class TestPlots:
    def test_plot_rpys_extended(self, tmp_path):
        df = pd.DataFrame({"References": ["; ".join([_ref("A", 2000), _ref("B", 2002)])]})
        rp = RP.compute_rpys_backward(df, median_window=3)
        fig, ax = RP.plot_rpys_extended(rp, highlight_years=[2000], f_name=str(tmp_path / "rpys.png"), dpi=50)
        assert (tmp_path / "rpys.png").exists()
        assert hasattr(ax, "deviation_ax")
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_reference_patterns(self):
        ts = _ts({"a": {2010: 1, 2011: 2, 2012: 3}, "b": {2010: 2, 2012: 1, 2013: 1}})
        cls = RP.classify_reference_patterns(ts)
        fig, ax = RP.plot_reference_patterns(cls, ts, top_n=2)
        assert len(ax.lines) == 2
        import matplotlib.pyplot as plt
        plt.close(fig)
