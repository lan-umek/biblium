# -*- coding: utf-8 -*-
"""Paneli za ``biblium.indicators`` (2.19): CSS, percentilni indikatorji,
koncentracija, necitiranost, bootstrap, kariere, Heaps, starost referenc,
citirane revije."""
from __future__ import annotations

import pandas as pd

from biblium.gui.panels.simple import SimpleAnalysisPanel, Opt

ENTITY_COLS = ["Source title", "Authors", "Author Keywords", "Index Keywords", "Affiliations",
               "Countries", "Document Type"]


def _stats_table(d: dict, skip=("lorenz_x", "lorenz_y")) -> pd.DataFrame:
    rows = [(k, v) for k, v in d.items() if k not in skip and not hasattr(v, "shape")]
    return pd.DataFrame(rows, columns=["Statistic", "Value"])


class CSSPanel(SimpleAnalysisPanel):
    title = "CSS Citation Classes"
    icon = "🏅"
    description = "Characteristic Scores and Scales (Glänzel & Schubert 1988): poorly / fairly / remarkably / outstandingly cited."
    OPTIONS = [
        Opt("by_year", "Thresholds per year", "check", True),
        Opt("n_classes", "Number of classes", "spin", 4, from_=2, to=6),
        Opt("min_year_size", "Min docs per year", "spin", 10, from_=1, to=1000),
    ]
    INFO_TEXT = """
CHARACTERISTIC SCORES AND SCALES (CSS)
──────────────────────────────────────
β1 = mean of all citation counts; β2 = mean of documents above β1;
β3 = mean of documents above β2. Classes: ≤β1 poorly cited, (β1,β2] fairly,
(β2,β3] remarkably, >β3 outstandingly cited. Parameter-free and robust to the
skewness of citation distributions; thresholds can be computed per year.

Glänzel, W., & Schubert, A. (1988). Characteristic scores and scales in
assessing citation impact. Journal of Information Science, 14(2), 123–127.
"""

    def compute(self, bib, o):
        summ = bib.citation_css(by_year=o["by_year"], n_classes=o["n_classes"], min_year_size=o["min_year_size"])
        from biblium.indicators import plot_css_classes
        counts = bib.df["CSS Class"].value_counts(dropna=True)
        res = {"summary": [(k, int(v)) for k, v in counts.items()], "tables": {"Summary": summ}}
        if o["by_year"]:
            res["plots"] = {"By year": lambda fig: plot_css_classes(summ, ax=fig.add_subplot(111))}
        return res


class PercentileIndicatorsPanel(SimpleAnalysisPanel):
    title = "Percentile Indicators"
    icon = "📊"
    description = "Leiden-style indicators per entity: P, TCS, MCS, MNCS, PP(top 10%), PP(top 1%), h."
    OPTIONS = [
        Opt("entity", "Entity column", "columns", "Source title", values=ENTITY_COLS),
        Opt("min_docs", "Min documents", "spin", 5, from_=1, to=1000),
        Opt("top_n", "Top N", "spin", 50, from_=5, to=1000),
        Opt("sort_by", "Sort by", "combo", "PP(top 10%)", values=["PP(top 10%)", "MNCS", "P", "TCS", "h", "PP(top 1%)"]),
    ]
    INFO_TEXT = """
PERCENTILE INDICATORS (LEIDEN RANKING STYLE)
────────────────────────────────────────────
MNCS = mean of citations / mean citations of the publication year.
PP(top X%) = share of documents whose within-year percentile rank is strictly
above 100−X. Full counting (each document counts fully for each entity).

Waltman, L. et al. (2012). The Leiden ranking 2011/2012. JASIST, 63(12), 2419–2432.
"""

    def compute(self, bib, o):
        df = bib.percentile_indicators(entity=o["entity"], min_docs=o["min_docs"], top_n=o["top_n"], sort_by=o["sort_by"])

        def draw(fig):
            ax = fig.add_subplot(111)
            top = df.head(20).iloc[::-1]
            ax.barh(top[o["entity"]].astype(str), top["PP(top 10%)"], color="#3b82f6")
            ax.set_xlabel("PP(top 10%)"); ax.set_title("Share of top-10% documents")
        return {"tables": {"Indicators": df}, "plots": {"PP(top 10%)": draw}}


class ConcentrationPanel(SimpleAnalysisPanel):
    title = "Concentration (Lorenz)"
    icon = "📐"
    description = "Gini, Theil, HHI, top shares and Lorenz curve of output (or citations) per entity."
    OPTIONS = [
        Opt("entity", "Entity column", "columns", "Authors", values=ENTITY_COLS),
        Opt("citations", "Weight by citations", "check", False),
    ]

    def compute(self, bib, o):
        tab = bib.concentration(entity=o["entity"], value_col="Cited by" if o["citations"] else None)
        col = bib._get_column(o["entity"])
        st = bib.concentration_stats[col]
        from biblium.indicators import plot_lorenz_curve
        return {
            "summary": [("Gini", f"{st['gini']:.3f}"), ("Theil", f"{st['theil']:.3f}"), ("HHI", f"{st['hhi']:.4f}"),
                        ("Top 10% share", f"{st.get('top10_share', float('nan')):.1f} %")],
            "tables": {"Items": tab, "Statistics": _stats_table(st)},
            "plots": {"Lorenz curve": lambda fig: plot_lorenz_curve(st, label=col, ax=fig.add_subplot(111))},
        }


class UncitednessPanel(SimpleAnalysisPanel):
    title = "Uncitedness"
    icon = "🫥"
    description = "Share of uncited documents by publication year (optionally by group)."
    OPTIONS = [Opt("group", "Group column (optional)", "columns", "", values=["", "Document Type", "Source title"])]

    def compute(self, bib, o):
        g = o["group"] or None
        tab = bib.uncitedness(group=g)
        from biblium.indicators import plot_uncitedness
        gcol = bib._get_column(g) if g else None
        total = tab["N"].sum(); unc = tab["Uncited"].sum()
        return {"summary": [("Documents", int(total)), ("Uncited", int(unc)), ("Uncited %", f"{100 * unc / total:.1f}" if total else "-")],
                "tables": {"By year": tab},
                "plots": {"Trend": lambda fig: plot_uncitedness(tab, group_col=gcol, ax=fig.add_subplot(111))}}


class BootstrapCIPanel(SimpleAnalysisPanel):
    title = "Bootstrap CI"
    icon = "🎲"
    description = "Bootstrap confidence intervals for citation indicators, overall or per group."
    OPTIONS = [
        Opt("statistic", "Statistic", "combo", "mean", values=["mean", "median", "h_index", "total", "uncited_share"]),
        Opt("by", "Group column (optional)", "columns", "", values=["", "Source title", "Authors", "Year", "Document Type"]),
        Opt("n_boot", "Bootstrap samples", "spin", 1000, from_=100, to=20000),
        Opt("min_docs", "Min docs per group", "spin", 5, from_=2, to=1000),
        Opt("seed", "Seed", "spin", 42, from_=0, to=99999),
    ]

    def compute(self, bib, o):
        by = o["by"] or None
        res = bib.indicator_ci(statistic=o["statistic"], by=by, n_boot=o["n_boot"], seed=o["seed"], min_docs=o["min_docs"])
        if by is None:
            return {"summary": [("Estimate", f"{res['estimate']:.2f}"), ("CI low", f"{res['ci_low']:.2f}"),
                                ("CI high", f"{res['ci_high']:.2f}"), ("SE", f"{res['se']:.2f}")],
                    "tables": {"Result": pd.DataFrame([res])}}

        def draw(fig):
            ax = fig.add_subplot(111)
            d = res.head(20).iloc[::-1]
            y = range(len(d))
            ax.errorbar(d["estimate"], y, xerr=[d["estimate"] - d["ci_low"], d["ci_high"] - d["estimate"]],
                        fmt="o", color="#3b82f6", ecolor="#94a3b8", capsize=3)
            ax.set_yticks(list(y)); ax.set_yticklabels(d[bib._get_column(by)].astype(str))
            ax.set_xlabel(o["statistic"]); ax.set_title("Bootstrap confidence intervals")
        return {"tables": {"By group": res}, "plots": {"Intervals": draw}}


class AuthorCareersPanel(SimpleAnalysisPanel):
    title = "Author Careers"
    icon = "🧑‍🔬"
    description = "Within-corpus academic age, productivity per active year, local h-index and m-quotient."
    OPTIONS = [
        Opt("min_docs", "Min documents", "spin", 2, from_=1, to=100),
        Opt("top_n", "Top N", "spin", 50, from_=5, to=1000),
        Opt("sort_by", "Sort by", "combo", "Docs", values=["Docs", "h", "m-quotient", "Total Citations", "Docs per Active Year"]),
    ]

    def compute(self, bib, o):
        df = bib.author_careers(min_docs=o["min_docs"], top_n=o["top_n"], sort_by=o["sort_by"])

        def draw(fig):
            ax = fig.add_subplot(111)
            ax.scatter(df["Academic Age"], df["h"], s=20 + 10 * df["Docs"], alpha=0.6, color="#3b82f6")
            for _, r in df.head(10).iterrows():
                ax.annotate(str(r["Author"]), (r["Academic Age"], r["h"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel("Academic age (years in corpus)"); ax.set_ylabel("Local h-index")
        return {"tables": {"Careers": df}, "plots": {"Age vs h": draw}}


class HeapsLawPanel(SimpleAnalysisPanel):
    title = "Heaps' Law"
    icon = "📈"
    description = "Vocabulary growth V(n) = K·n^β of keywords (β < 1: sublinear growth)."
    OPTIONS = [
        Opt("field", "Keyword column", "columns", "Author Keywords", values=["Author Keywords", "Index Keywords"]),
        Opt("order_by", "Order documents by", "combo", "Year", values=["Year", "random"]),
        Opt("seed", "Seed", "spin", 42, from_=0, to=99999),
    ]

    def compute(self, bib, o):
        params = bib.fit_heaps_law(field=o["field"], order_by=None if o["order_by"] == "random" else o["order_by"], seed=o["seed"])
        from biblium.indicators import plot_heaps_law
        curve = bib.heaps_df
        return {"summary": [("β", f"{params['beta']:.3f}"), ("K", f"{params['K']:.2f}"), ("R²", f"{params['r2']:.3f}"),
                            ("Vocabulary", params["vocabulary"])],
                "tables": {"Parameters": pd.DataFrame([params]), "Curve": curve},
                "plots": {"Heaps": lambda fig: plot_heaps_law(curve, params, ax=fig.add_subplot(111))}}


class ReferenceAgePanel(SimpleAnalysisPanel):
    title = "Reference Age / Price"
    icon = "⏳"
    description = "Price index, median reference age and citing half-life by publication year."
    OPTIONS = [
        Opt("price_window", "Price window (years)", "spin", 5, from_=1, to=20),
        Opt("add_to_df", "Add per-document columns", "check", True),
    ]

    def compute(self, bib, o):
        by = bib.reference_age(price_window=o["price_window"], add_to_df=o["add_to_df"])
        st = bib.reference_age_stats
        from biblium.indicators import plot_reference_age_profile
        res = {"summary": [("Price index", f"{st['price_index']:.1f} %"), ("Median age", f"{st['median_ref_age']:.1f}"),
                           ("Citing half-life", f"{st['citing_half_life']:.1f}"), ("References", st["n_refs"])],
               "tables": {"By year": by, "Overall": _stats_table(st)}}
        if len(by):
            res["plots"] = {"Profile": lambda fig: plot_reference_age_profile(by, ax=fig.add_subplot(111))}
        return res


class CitedSourcesPanel(SimpleAnalysisPanel):
    title = "Cited Sources"
    icon = "📚"
    description = "Knowledge base: most cited journals in the reference lists and overlap with publishing journals."
    OPTIONS = [Opt("top_n", "Top N", "spin", 30, from_=5, to=500)]

    def compute(self, bib, o):
        tab = bib.cited_sources(top_n=o["top_n"])
        st = bib.cited_sources_stats

        def draw(fig):
            ax = fig.add_subplot(111)
            d = tab.head(20).iloc[::-1]
            colors = ["#3b82f6" if x else "#94a3b8" for x in d["In Corpus"]]
            ax.barh(d["Cited Source"].astype(str).str.slice(0, 50), d["Citations"], color=colors)
            ax.set_xlabel("Citations in reference lists"); ax.set_title("Blue = corpus also publishes here")
        return {"summary": [("Distinct sources", st.get("n_distinct_sources", 0)),
                            ("Self-reliance", f"{st.get('self_reliance', float('nan')):.1f} %"),
                            ("Top-10 share", f"{st.get('top10_share', float('nan')):.1f} %"),
                            ("Jaccard overlap", f"{st.get('jaccard_overlap', float('nan')):.3f}")],
                "tables": {"Cited sources": tab, "Statistics": _stats_table(st)}, "plots": {"Top sources": draw}}
