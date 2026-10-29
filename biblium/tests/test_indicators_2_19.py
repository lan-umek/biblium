# -*- coding: utf-8 -*-
"""
Regresijski testi za ``biblium.indicators`` (novo v 2.19).

Vse kontrole so ročno izračunane na majhnih, sintetičnih vhodih; testi so
deterministični in hermetični (brez mreže, brez datotek).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biblium import indicators as I


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

class TestCSS:
    def test_thresholds_iterative_means(self):
        # vsota 140 / 20 = 7; nad 7: {20, 30, 40, 50} → 35; nad 35: {40, 50} → 45
        vals = np.array([0, 0, 0, 0, 0, 0, 20, 30, 40, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], float)
        thr = I._css_thresholds(vals, 4)
        assert thr == pytest.approx([7.0, 35.0, 45.0])

    def test_classes_assignment(self):
        df = pd.DataFrame({"Cited by": [0, 2, 4, 10, 20, 100]})
        cls, summ = I.characteristic_scores_and_scales(df)
        # mean = 22.67 → b1; nad: {100} → b2 = 100; nad 100: prazno → brez b3
        assert summ.loc[0, "b1"] == pytest.approx(136 / 6)
        assert summ.loc[0, "b2"] == pytest.approx(100.0)
        assert np.isnan(summ.loc[0, "b3"])
        assert list(cls.astype(str)) == ["Poorly cited"] * 5 + ["Fairly cited"]
        assert summ.loc[0, "N Poorly cited"] == 5 and summ.loc[0, "% Fairly cited"] == pytest.approx(100 / 6)

    def test_perfect_square_h_like_example(self):
        # klasičen primer: 1..10 → b1=5.5; nad: 6..10 → 8; nad 8: 9,10 → 9.5
        df = pd.DataFrame({"Cited by": range(1, 11)})
        cls, summ = I.characteristic_scores_and_scales(df)
        assert summ.loc[0, ["b1", "b2", "b3"]].tolist() == pytest.approx([5.5, 8.0, 9.5])
        assert (cls == "Outstandingly cited").sum() == 1  # samo 10

    def test_by_year_uses_global_when_small(self):
        df = pd.DataFrame({"Cited by": [1, 2, 3, 40, 50, 60], "Year": [2000] * 3 + [2001] * 3})
        cls, summ = I.characteristic_scores_and_scales(df, year_col="Year", min_year_size=10)
        yrs = summ[summ["Scope"] != "all"]
        assert len(yrs) == 2
        assert (yrs["b1"] == summ.loc[0, "b1"]).all()  # globalni pragovi

    def test_nan_citations_unclassified(self):
        df = pd.DataFrame({"Cited by": [1, np.nan, 3]})
        cls, summ = I.characteristic_scores_and_scales(df)
        assert pd.isna(cls.iloc[1]) and summ.loc[0, "N"] == 2


# ---------------------------------------------------------------------------
# Percentilni indikatorji
# ---------------------------------------------------------------------------

class TestPercentileIndicators:
    @pytest.fixture
    def df(self):
        # dve leti po 10 dokumentov; citati 1..10 v vsakem letu
        rows = []
        for y in (2000, 2001):
            for c in range(1, 11):
                rows.append({"Year": y, "Cited by": c,
                             "Source title": "A" if c > 5 else "B"})
        return pd.DataFrame(rows)

    def test_mncs_and_pp_top10(self, df):
        out = I.percentile_indicators(df, "Source title", min_docs=1, sep=";").set_index("Source title")
        # povprečje leta = 5.5; A ima 6..10 → MNCS = 8/5.5
        assert out.loc["A", "MNCS"] == pytest.approx(8 / 5.5)
        assert out.loc["B", "MNCS"] == pytest.approx(3 / 5.5)
        # top 10 % leta: rang pct > 90 → samo c=10 (pct=100) → A: 2/10 = 20 %
        assert out.loc["A", "PP(top 10%)"] == pytest.approx(20.0)
        assert out.loc["A", "P(top 1%)"] == 2  # pct 100 > 99 → oba najbolj citirana

        assert out.loc["B", "PP(top 10%)"] == 0.0
        assert out.loc["A", "P"] == 10 and out.loc["A", "TCS"] == 80
        assert out.loc["A", "h"] == 7  # 10,10,9,9,8,8,7,7,6,6 → h=7

    def test_min_docs_filter_and_top_n(self, df):
        out = I.percentile_indicators(df, "Source title", min_docs=11)
        assert out.empty
        out = I.percentile_indicators(df, "Source title", min_docs=1, top_n=1)
        assert len(out) == 1 and out.iloc[0]["Source title"] == "A"

    def test_multi_valued_entities_full_counting(self):
        df = pd.DataFrame({"Year": [2000, 2000], "Cited by": [10, 0], "Authors": ["X; Y", "X"]})
        out = I.percentile_indicators(df, "Authors", min_docs=1).set_index("Authors")
        assert out.loc["X", "P"] == 2 and out.loc["Y", "P"] == 1
        assert out.loc["Y", "MNCS"] == pytest.approx(10 / 5)


# ---------------------------------------------------------------------------
# Necitiranost
# ---------------------------------------------------------------------------

class TestUncitedness:
    def test_shares_and_cumulative(self):
        df = pd.DataFrame({"Year": [2000, 2000, 2001, 2001, 2001],
                           "Cited by": [0, 5, 0, 0, 3]})
        out = I.uncitedness(df, current_year=2001).set_index("Year")
        assert out.loc[2000, "Uncited %"] == 50.0
        assert out.loc[2001, "Uncited %"] == pytest.approx(200 / 3)
        assert out.loc[2001, "Cumulative Uncited %"] == pytest.approx(60.0)
        assert out.loc[2000, "Age"] == 1 and out.loc[2001, "Age"] == 0

    def test_nan_counts_as_uncited(self):
        df = pd.DataFrame({"Year": [2000, 2000], "Cited by": [np.nan, 1]})
        out = I.uncitedness(df)
        assert out.loc[0, "Uncited"] == 1

    def test_grouped(self):
        df = pd.DataFrame({"Year": [2000] * 4, "Cited by": [0, 1, 0, 0],
                           "Document Type": ["Article", "Article", "Review", "Review"]})
        out = I.uncitedness(df, group_col="Document Type").set_index("Document Type")
        assert out.loc["Article", "Uncited %"] == 50.0 and out.loc["Review", "Uncited %"] == 100.0


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

class TestBootstrap:
    def test_constant_vector_has_zero_width(self):
        r = I.bootstrap_ci([5, 5, 5, 5], "mean", n_boot=100)
        assert r["estimate"] == 5 and r["ci_low"] == 5 and r["ci_high"] == 5 and r["se"] == 0

    def test_deterministic_with_seed(self):
        x = [1, 2, 3, 4, 50, 6, 7, 8, 9, 10]
        a = I.bootstrap_ci(x, "mean", n_boot=300, seed=7)
        b = I.bootstrap_ci(x, "mean", n_boot=300, seed=7)
        c = I.bootstrap_ci(x, "mean", n_boot=300, seed=8)
        assert a == b and a["ci_low"] != c["ci_low"]

    def test_interval_contains_estimate_and_is_sane(self):
        rng = np.random.default_rng(0)
        x = rng.poisson(5, 200)
        r = I.bootstrap_ci(x, "mean", n_boot=500)
        assert r["ci_low"] <= r["estimate"] <= r["ci_high"]
        # SE povprečja ≈ sd/sqrt(n)
        assert r["se"] == pytest.approx(x.std(ddof=1) / np.sqrt(200), rel=0.3)

    def test_h_index_and_custom_statistic(self):
        x = [10, 8, 5, 4, 3]
        assert I.bootstrap_ci(x, "h_index", n_boot=10)["estimate"] == 4
        assert I.bootstrap_ci(x, lambda v: float(v.max()), n_boot=10)["estimate"] == 10

    def test_basic_method_and_empty(self):
        r = I.bootstrap_ci([1, 2, 3], "mean", n_boot=50, method="basic")
        assert r["ci_low"] <= r["ci_high"]
        assert np.isnan(I.bootstrap_ci([], "mean")["estimate"])
        with pytest.raises(ValueError):
            I.bootstrap_ci([1, 2], "mean", method="nope")

    def test_by_group(self):
        df = pd.DataFrame({"Cited by": [1, 2, 3, 10, 20, 30], "Source title": ["A"] * 3 + ["B"] * 3})
        out = I.bootstrap_ci_by_group(df, "Source title", min_docs=3, n_boot=50)
        assert list(out["Source title"]) == ["B", "A"]
        assert out.loc[0, "estimate"] == 20


# ---------------------------------------------------------------------------
# Starost referenc
# ---------------------------------------------------------------------------

class TestReferenceAge:
    def test_price_index_and_ages(self):
        refs = ("Smith, J., Title one, Journal, 1, pp. 1-2, (2018); "
                "Doe, A., Title two, Journal, 2, pp. 3-4, (2010); "
                "Roe, B., Title three, Journal, 3, pp. 5-6, (2020); "
                "No year here")
        df = pd.DataFrame({"Year": [2020], "References": [refs]})
        per = I.reference_ages(df)
        assert per.loc[0, "N refs"] == 4 and per.loc[0, "N refs with year"] == 3
        assert per.loc[0, "Mean ref age"] == pytest.approx((2 + 10 + 0) / 3)
        assert per.loc[0, "Median ref age"] == 2
        assert per.loc[0, "Price index"] == pytest.approx(200 / 3)  # 2 od 3 ≤ 5 let
        by, ov = I.reference_age_profile(df)
        assert ov["n_refs"] == 3 and ov["citing_half_life"] == 2 and by.loc[0, "N docs"] == 1

    def test_future_refs_clamped_and_outliers_dropped(self):
        df = pd.DataFrame({"Year": [2020], "References": ["A, T, J, (2022); B, T, J, (1700)"]})
        per = I.reference_ages(df, max_age=100)
        assert per.loc[0, "N refs with year"] == 1 and per.loc[0, "Mean ref age"] == 0

    def test_missing_rows(self):
        df = pd.DataFrame({"Year": [2020, np.nan], "References": [np.nan, "A, T, J, (2010)"]})
        per = I.reference_ages(df)
        assert per["N refs with year"].tolist() == [0, 0]
        by, ov = I.reference_age_profile(df)
        assert by.empty and np.isnan(ov["price_index"])


# ---------------------------------------------------------------------------
# Citirane revije
# ---------------------------------------------------------------------------

class TestCitedSources:
    def test_scopus_extractor(self):
        ex = I._source_from_scopus_reference
        assert ex("Smith, J., A title, Scientometrics, 10, 2, pp. 1-20, (2010)") == "Scientometrics"
        assert ex("Smith, J., A title, Journal of Informetrics, 4, (2011)") == "Journal of Informetrics"
        assert ex("Smith, J., A title, Nature, 505, 7484, pp. 543-545, (2014)") == "Nature"
        assert ex("Smith, J., (2010)") is None  # brez revije

    def test_table_and_stats(self):
        refs1 = "A, X., T1, Scientometrics, 1, pp. 1-2, (2010); B, Y., T2, Nature, 2, (2011)"
        refs2 = "C, Z., T3, scientometrics, 3, (2012)"
        df = pd.DataFrame({"References": [refs1, refs2], "Source title": ["Scientometrics", "Other"]})
        tab, st = I.cited_sources(df)
        assert tab.iloc[0]["Cited Source"] == "Scientometrics"  # najpogostejši zapis velikosti črk
        assert tab.iloc[0]["Citations"] == 2 and tab.iloc[0]["Citing Docs"] == 2
        assert tab.iloc[0]["Publishing Docs"] == 1 and bool(tab.iloc[0]["In Corpus"])
        assert st["n_refs"] == 3 and st["n_refs_with_source"] == 3 and st["n_distinct_sources"] == 2
        assert st["self_reliance"] == pytest.approx(200 / 3)
        assert st["jaccard_overlap"] == pytest.approx(1 / 3)  # {sci, nature} ∪ {sci, other}
        assert st["hhi"] == pytest.approx((2 / 3) ** 2 + (1 / 3) ** 2)

    def test_empty(self):
        tab, st = I.cited_sources(pd.DataFrame({"References": [np.nan]}))
        assert tab.empty and st["n_refs"] == 0


# ---------------------------------------------------------------------------
# Koncentracija
# ---------------------------------------------------------------------------

class TestConcentration:
    def test_gini_known_values(self):
        assert I.gini_coefficient([1, 1, 1, 1]) == pytest.approx(0.0)
        # [0,0,0,1] → G = (n-1)/n = 0.75
        assert I.gini_coefficient([0, 0, 0, 1]) == pytest.approx(0.75)
        assert I.gini_coefficient([1, 2, 3, 4]) == pytest.approx(0.25)
        assert np.isnan(I.gini_coefficient([0, 0]))

    def test_theil(self):
        assert I.theil_index([3, 3, 3]) == pytest.approx(0.0)
        # [1, 3]: mu=2 → 0.5*(0.5 ln0.5 + 1.5 ln1.5)
        assert I.theil_index([1, 3]) == pytest.approx(0.5 * (0.5 * np.log(0.5) + 1.5 * np.log(1.5)))

    def test_lorenz_curve_points(self):
        x, y = I.lorenz_curve([1, 1, 2])
        assert x.tolist() == pytest.approx([0, 1 / 3, 2 / 3, 1])
        assert y.tolist() == pytest.approx([0, 0.25, 0.5, 1.0])

    def test_concentration_table(self):
        df = pd.DataFrame({"Authors": ["A; B", "A", "A; C", "B"], "Cited by": [10, 0, 0, 5]})
        tab, st = I.concentration(df, "Authors")
        assert tab.iloc[0]["Item"] == "A" and tab.iloc[0]["Value"] == 3
        assert st["n_items"] == 3 and st["total"] == 6
        assert st["top20_share"] == pytest.approx(50.0)  # ceil(0.2*3)=1 → A: 3/6
        assert st["share_of_items_for_half"] == pytest.approx(100 / 3)
        tab2, st2 = I.concentration(df, "Authors", value_col="Cited by")
        assert tab2.set_index("Item").loc["B", "Value"] == 15


# ---------------------------------------------------------------------------
# Kariere avtorjev
# ---------------------------------------------------------------------------

class TestAuthorCareers:
    def test_basic(self):
        df = pd.DataFrame({
            "Authors": ["A; B", "A", "B; A", "C"],
            "Year": [2010, 2012, 2015, 2015],
            "Cited by": [10, 4, 2, 100],
        })
        out = I.author_careers(df, min_docs=1, current_year=2019).set_index("Author")
        a = out.loc["A"]
        assert a["Docs"] == 3 and a["First Year"] == 2010 and a["Last Year"] == 2015
        assert a["Academic Age"] == 10 and a["Span"] == 6 and a["Active Years"] == 3
        assert a["h"] == 2 and a["m-quotient"] == pytest.approx(0.2)
        assert a["First-author Docs"] == 2 and a["First-author %"] == pytest.approx(200 / 3)
        assert out.loc["B", "First-author Docs"] == 1
        assert out.loc["C", "Academic Age"] == 5

    def test_min_docs(self):
        df = pd.DataFrame({"Authors": ["A", "B; A"], "Year": [2000, 2001], "Cited by": [1, 1]})
        assert list(I.author_careers(df, min_docs=2)["Author"]) == ["A"]


# ---------------------------------------------------------------------------
# Heapsov zakon
# ---------------------------------------------------------------------------

class TestHeaps:
    def test_exact_power_law_recovered(self):
        # Konstruiramo zaporedje, kjer je V(n) = n (vse besede nove) → β = 1, K = 1
        df = pd.DataFrame({"kw": [f"w{i}" for i in range(50)], "Year": range(50)})
        curve, p = I.fit_heaps_law(df, col="kw", order_col="Year")
        assert p["beta"] == pytest.approx(1.0) and p["K"] == pytest.approx(1.0) and p["r2"] == pytest.approx(1.0)
        assert p["vocabulary"] == 50 and p["n_tokens"] == 50

    def test_sublinear_and_case_normalisation(self):
        df = pd.DataFrame({"kw": ["A; b", "a; B", "a; b; c"], "Year": [1, 2, 3]})
        curve, p = I.fit_heaps_law(df, col="kw")
        assert p["vocabulary"] == 3 and p["n_tokens"] == 7 and p["beta"] < 1

    def test_deterministic_random_order(self):
        df = pd.DataFrame({"kw": ["a; b", "c", "a; d", "e; f; a"]})
        c1, p1 = I.fit_heaps_law(df, col="kw", order_col=None, seed=1)
        c2, p2 = I.fit_heaps_law(df, col="kw", order_col=None, seed=1)
        assert p1 == p2 and c1.equals(c2)


# ---------------------------------------------------------------------------
# BiblioStats ovoji (brez datotek)
# ---------------------------------------------------------------------------

class TestBiblioStatsWrappers:
    @pytest.fixture
    def bib(self):
        pytest.importorskip("biblium.bibstats")
        from biblium.bibstats import BiblioStats
        df = pd.DataFrame({
            "Authors": ["A; B", "A", "C; A", "B", "D"],
            "Title": list("vwxyz"),
            "Year": [2010, 2011, 2012, 2012, 2013],
            "Cited by": [10, 0, 3, 7, 1],
            "Source title": ["S1", "S1", "S2", "S2", "S3"],
            "Author Keywords": ["k1; k2", "k1", "k3; k2", "k4", "k1; k5"],
            "References": ["X, T, Journal S1, (2005); Y, T, Journal S2, (2009)", np.nan,
                           "Z, T, Journal S1, (2010)", "Q, T, Journal S9, (1990)", "W, T, Journal S1, (2012)"],
            "Document Type": ["Article"] * 5,
        })
        return BiblioStats(df=df, db="scopus", res_folder=None)

    def test_all_methods_run_and_store(self, bib):
        bib.citation_css(by_year=False)
        assert "CSS Class" in bib.df.columns and bib.css_summary_df.loc[0, "N"] == 5
        pi = bib.percentile_indicators("Source title", min_docs=1)
        assert set(pi["Source title"]) == {"S1", "S2", "S3"}
        un = bib.uncitedness()
        assert un["Uncited"].sum() == 1
        r = bib.indicator_ci("mean", n_boot=50)
        assert r["estimate"] == pytest.approx(4.2) and bib.indicator_ci_result is r
        ra = bib.reference_age()
        assert "Price index" in bib.df.columns and bib.reference_age_stats["n_refs"] == 5
        cs = bib.cited_sources()
        assert cs.iloc[0]["Cited Source"] == "Journal S1" and cs.iloc[0]["Citations"] == 3
        conc = bib.concentration("Authors")
        assert conc.iloc[0]["Item"] == "A" and "Authors" in bib.concentration_stats
        ac = bib.author_careers(min_docs=1)
        assert ac.set_index("Author").loc["A", "Docs"] == 3
        hp = bib.fit_heaps_law()
        assert hp["vocabulary"] == 5
        # methods_text pozna nove analize
        txt_en, txt_sl = bib.methods_text("en"), bib.methods_text("sl")
        for needle in ("Characteristic Scores", "Leiden", "Price", "Gini", "m-quotient", "Heaps", "bootstrap"):
            assert needle in txt_en, needle
        assert "Priceovim indeksom" in txt_sl and "Heapsovim" in txt_sl
