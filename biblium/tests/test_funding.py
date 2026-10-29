# -*- coding: utf-8 -*-
"""Tests for ``biblium.funding`` (new in 2.19): parsing of Scopus funding
entries, flags, full-counting funder statistics and the funded/unfunded
comparison. Hand-computed expectations on tiny synthetic frames."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from biblium import funding as F


class TestParsing:
    def test_colon_style(self):
        assert F.parse_funding_details("European Commission, EC: 123") == [
            {"funder": "European Commission", "acronym": "EC", "grant": "123"}]

    def test_colon_several_grants(self):
        out = F.parse_funding_details("National Natural Science Foundation of China, NSFC: 71573121, 71873092")
        assert out == [{"funder": "National Natural Science Foundation of China", "acronym": "NSFC",
                        "grant": "71573121, 71873092"}]

    def test_no_grant(self):
        assert F.parse_funding_details("Javna Agencija za Raziskovalno Dejavnost RS, ARRS") == [
            {"funder": "Javna Agencija za Raziskovalno Dejavnost RS", "acronym": "ARRS", "grant": None}]
        assert F.parse_funding_details("Jacobs Foundation") == [
            {"funder": "Jacobs Foundation", "acronym": None, "grant": None}]

    def test_parenthesised_grant_and_duplicates(self):
        s = "Chinese Academy of Sciences, CAS, (XDA28020102); Chinese Academy of Sciences, CAS"
        out = F.parse_funding_details(s)
        assert len(out) == 2 and out[0]["grant"] == "XDA28020102" and out[1]["grant"] is None
        merged = F.parse_funding_details(s, merge_duplicates=True)
        assert merged == [{"funder": "Chinese Academy of Sciences", "acronym": "CAS", "grant": "XDA28020102"}]

    def test_country_part_and_several_acronyms(self):
        out = F.parse_funding_details("Ministry of Higher Education, Malaysia, MOHE, (20190101LRGS)")
        assert out == [{"funder": "Ministry of Higher Education, Malaysia", "acronym": "MOHE",
                        "grant": "20190101LRGS"}]
        out = F.parse_funding_details("Universidad de Extremadura, UEx, UNEX")
        assert out[0]["funder"] == "Universidad de Extremadura" and out[0]["acronym"] == "UEx"

    def test_grant_list_with_commas_in_parentheses(self):
        out = F.parse_funding_details("Generalitat de Catalunya, (2017-SGR-1409, 2017-SGR-1537)")
        assert out == [{"funder": "Generalitat de Catalunya", "acronym": None,
                        "grant": "2017-SGR-1409, 2017-SGR-1537"}]

    def test_grant_only_and_empty(self):
        assert F.parse_funding_details("(2020004)") == []
        assert F.parse_funding_details("") == []
        assert F.parse_funding_details(None) == []
        assert F.parse_funding_details(np.nan) == []
        assert F.parse_funding_details(" ; ; ") == []

    def test_custom_separator(self):
        out = F.parse_funding_details("A, AA: 1 | B", sep="|")
        assert [d["funder"] for d in out] == ["A", "B"]


class TestNormalize:
    def test_basic(self):
        assert F.normalize_funder("  The  National   Science Foundation. ") == "national science foundation"
        assert F.normalize_funder("Cardiff University") == "cardiff university"
        assert F.normalize_funder(None) == ""

    def test_aliases(self):
        assert F.normalize_funder("NSFC") == "national natural science foundation of china"
        assert F.normalize_funder("Horizon 2020 Framework Programme", "H2020") == "european commission"
        assert F.normalize_funder("Some Programme", "EC") == "european commission"
        assert F.normalize_funder("Javna Agencija za Raziskovalno Dejavnost RS", "ARRS") == \
            F.normalize_funder("Slovenian Research Agency") == "slovenian research and innovation agency"
        assert F.funder_display_name("european commission") == "European Commission"
        assert F.funder_display_name("cardiff university", "Cardiff University ") == "Cardiff University"


@pytest.fixture
def frame():
    # 6 docs, 2 years; NaN citation → 0 for MNCS
    return pd.DataFrame({
        "Year": [2020, 2020, 2020, 2021, 2021, 2021],
        "Cited by": [10, 20, 0, 4, np.nan, 8],
        "Funding Details": [
            "National Science Foundation, NSF, (1); National Institutes of Health, NIH",  # 2 funders
            "NSF: 2",                                                                    # alias → NSF
            None,                                                                        # unfunded
            "Horizon 2020 Framework Programme, H2020, (3); European Commission, EC",     # merged → EC
            "",                                                                          # text only
            "Cardiff University",
        ],
        "Funding Texts": [None, None, None, None, "We thank X for support.", None],
        "Author Keywords": ["a; b", "a; c", "d", "a; b", "e", "b"],
    })


class TestFlags:
    def test_add_funding_flags(self, frame):
        out = F.add_funding_flags(frame)
        assert out["Funded"].tolist() == [True, True, False, True, True, True]
        assert out["N funders"].tolist() == [2, 1, 0, 1, 0, 1]
        assert out.loc[0, "Funders"] == "National Science Foundation; National Institutes of Health"
        assert out.loc[3, "Funders"] == "European Commission"
        assert pd.isna(out.loc[2, "Funders"]) and pd.isna(out.loc[4, "Funders"])
        assert "Funded" not in frame.columns  # input not mutated

    def test_no_text_column(self, frame):
        out = F.add_funding_flags(frame.drop(columns=["Funding Texts"]))
        assert out["Funded"].tolist() == [True, True, False, True, False, True]


class TestFunderCounts:
    def test_full_counting_and_stats(self, frame):
        fc = F.funder_counts(frame).set_index("Funder")
        # NSF: docs 0 and 1; NIH: doc 0; EC: doc 3; Cardiff: doc 5 → 5 funder-docs from 4 funded docs
        assert fc["Docs"].to_dict() == {"National Science Foundation": 2, "National Institutes of Health": 1,
                                        "European Commission": 1, "Cardiff University": 1}
        assert fc.loc["National Science Foundation", "% of funded docs"] == pytest.approx(50.0)
        assert fc.loc["National Science Foundation", "Total citations"] == 30
        assert fc.loc["National Science Foundation", "Mean citations"] == pytest.approx(15.0)
        # year means: 2020 → 10; 2021 → (4+0+8)/3 = 4
        assert fc.loc["National Science Foundation", "MNCS"] == pytest.approx((1.0 + 2.0) / 2)
        assert fc.loc["European Commission", "MNCS"] == pytest.approx(1.0)
        assert fc.loc["Cardiff University", "MNCS"] == pytest.approx(2.0)
        assert fc.loc["National Science Foundation", "Acronym"] == "NSF"
        assert fc.loc["National Science Foundation", "First year"] == 2020
        assert fc.loc["Cardiff University", "Last year"] == 2021
        # sorted by Docs desc, then citations desc
        assert list(fc.index)[:2] == ["National Science Foundation", "National Institutes of Health"]

    def test_min_docs_top_n(self, frame):
        assert F.funder_counts(frame, min_docs=2)["Funder"].tolist() == ["National Science Foundation"]
        assert len(F.funder_counts(frame, top_n=2)) == 2
        assert F.funder_counts(pd.DataFrame({"Funding Details": [None]})).empty


class TestByYear:
    def test_table(self, frame):
        t = F.funding_by_year(frame).set_index("Year")
        assert t["N docs"].tolist() == [3, 3]
        assert t["Funded"].tolist() == [2, 3]
        assert t.loc[2020, "Funded %"] == pytest.approx(200 / 3)
        assert t.loc[2021, "Funded %"] == pytest.approx(100.0)
        assert t.loc[2020, "Mean citations funded"] == pytest.approx(15.0)
        assert t.loc[2020, "Mean citations unfunded"] == pytest.approx(0.0)
        assert t.loc[2021, "Mean citations funded"] == pytest.approx(6.0)   # NaN ignored in mean
        assert np.isnan(t.loc[2021, "Mean citations unfunded"])


class TestFundedVsUnfunded:
    def test_stats(self, frame):
        r = F.funded_vs_unfunded(frame)
        assert r["n_funded"] == 5 and r["n_unfunded"] == 1
        # funded citations (NaN → 0): 10, 20, 4, 0, 8 → mean 8.4, median 8
        assert r["mean_cit_funded"] == pytest.approx(8.4)
        assert r["median_cit_funded"] == pytest.approx(8.0)
        assert r["mean_cit_unfunded"] == pytest.approx(0.0)
        # MNCS: funded 1, 2, 1, 0, 2 → 1.2; unfunded doc 2 → 0
        assert r["mncs_funded"] == pytest.approx(1.2)
        assert r["mncs_unfunded"] == pytest.approx(0.0)
        assert np.isnan(r["mncs_ratio"])  # division by zero → NaN
        assert r["share_uncited_funded"] == pytest.approx(0.2)
        assert r["share_uncited_unfunded"] == pytest.approx(1.0)
        assert 0 <= r["mannwhitney_p"] <= 1

    def test_mannwhitney_matches_scipy(self):
        from scipy.stats import mannwhitneyu
        df = pd.DataFrame({"Cited by": [1, 5, 9, 2, 3, 0, 7, 4],
                           "Funding Details": ["F"] * 4 + [None] * 4, "Year": [2020] * 8})
        r = F.funded_vs_unfunded(df, text_col=None)
        exp = mannwhitneyu([1, 5, 9, 2], [3, 0, 7, 4], alternative="two-sided")
        assert r["mannwhitney_u"] == pytest.approx(exp.statistic)
        assert r["mannwhitney_p"] == pytest.approx(exp.pvalue)
        assert r["mncs_ratio"] == pytest.approx((17 / 4) / (14 / 4))

    def test_empty_group(self):
        df = pd.DataFrame({"Cited by": [1, 2], "Funding Details": ["F", "G"], "Year": [2020, 2020]})
        r = F.funded_vs_unfunded(df, text_col=None)
        assert r["n_unfunded"] == 0 and np.isnan(r["mannwhitney_p"]) and np.isnan(r["mean_cit_unfunded"])


class TestNetworks:
    def test_cooccurrence(self, frame):
        M = F.funder_cooccurrence(frame, min_docs=1)
        assert M.shape == (4, 4)
        assert M.loc["National Science Foundation", "National Science Foundation"] == 2
        assert M.loc["National Science Foundation", "National Institutes of Health"] == 1
        assert M.loc["National Institutes of Health", "National Science Foundation"] == 1
        assert M.loc["European Commission", "National Science Foundation"] == 0
        assert F.funder_cooccurrence(frame, min_docs=2).shape == (1, 1)

    def test_keyword_profile(self, frame):
        p = F.funder_keyword_profile(frame, top_n_funders=1, top_n_keywords=2)
        assert p["Funder"].unique().tolist() == ["National Science Foundation"]
        assert p.set_index("Keyword")["Docs"].to_dict() == {"a": 2, "b": 1}


class TestPlots:
    def test_plots_render(self, frame, tmp_path):
        import matplotlib.pyplot as plt
        f1 = tmp_path / "trend.png"
        fig, ax = F.plot_funding_trend(F.funding_by_year(frame), f_name=str(f1), dpi=50)
        assert f1.exists()
        plt.close(fig)
        f2 = tmp_path / "top.png"
        fig, ax = F.plot_top_funders(F.funder_counts(frame), top_n=3, f_name=str(f2), dpi=50)
        assert f2.exists()
        assert len(ax.get_yticklabels()) == 3
        plt.close(fig)
