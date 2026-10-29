# -*- coding: utf-8 -*-
"""Paneli 2.19: RPYS z vzorci citiranih referenc, dual-map overlay, snowballing."""
from __future__ import annotations

import pandas as pd

from biblium.gui.panels.simple import SimpleAnalysisPanel, Opt


class RPYSPatternsPanel(SimpleAnalysisPanel):
    title = "RPYS & Reference Patterns"
    icon = "🔭"
    description = "Reference Publication Year Spectroscopy (centered/backward median) and classification of cited references: Hot Paper, Constant Performer, Life Cycle, Sleeping Beauty."
    OPTIONS = [
        Opt("median_mode", "Median window", "combo", "centered", values=["centered", "backward"]),
        Opt("median_window", "Window (years)", "spin", 5, from_=3, to=15),
        Opt("min_citations", "Min citations of a reference", "spin", 3, from_=2, to=100),
        Opt("top_n", "References to plot", "spin", 8, from_=3, to=25),
    ]
    INFO_TEXT = """
RPYS AND CITED-REFERENCE PATTERNS
─────────────────────────────────
RPYS (Marx et al. 2014) counts cited references per reference publication year
and highlights years deviating from a rolling median (centered, or backward as
in bibliometrix 5.1). Each cited reference's citation history (citing years)
is classified: Hot Paper (most citations in the last years), Sleeping Beauty
(long dormancy, then awakening), Constant Performer (long, even citation
stream), Life Cycle (rise and decline), Other / Insufficient data.
"""

    def compute(self, bib, o):
        from biblium import reference_patterns as RP
        cls = bib.rpys_patterns(min_citations=o["min_citations"], median_mode=o["median_mode"],
                                median_window=o["median_window"])
        summ = bib.reference_pattern_summary_df
        rp = bib.rpys_ext_df
        ts = bib.reference_timeseries_df
        peaks = rp.nlargest(5, "Deviation")[["Cited Year", "Citations", "Median", "Deviation"]]
        return {"summary": [(str(r["Pattern"]), int(r["N"])) for _, r in summ.iterrows()][:4],
                "tables": {"Patterns": cls, "Pattern summary": summ, "RPYS": rp, "Peak years": peaks},
                "plots": {"RPYS": lambda fig: RP.plot_rpys_extended(rp, ax=fig.add_subplot(111)),
                          "Reference curves": lambda fig: RP.plot_reference_patterns(cls, ts, top_n=o["top_n"], ax=fig.add_subplot(111))}}


class DualMapPanel(SimpleAnalysisPanel):
    title = "Dual-Map Overlay"
    icon = "🗺️"
    description = "Citing disciplines (journals of the corpus) vs. cited disciplines (journals in reference lists), mapped through Scopus ASJC."
    OPTIONS = [
        Opt("level", "Discipline level", "combo", "Area", values=["Area", "Science", "Field"]),
        Opt("fractional", "Fractional counting", "check", True),
        Opt("top_citing", "Citing areas shown", "spin", 12, from_=3, to=40),
        Opt("top_cited", "Cited areas shown", "spin", 12, from_=3, to=40),
    ]
    INFO_TEXT = """
DUAL-MAP OVERLAY (ASJC-BASED)
─────────────────────────────
CiteSpace's dual-map overlay places citing journals on one base map and cited
journals on another, with ribbons showing knowledge flows. Biblium approximates
it with the Scopus ASJC classification bundled in 'additional files': the left
stack shows the disciplines of the journals in which the corpus publishes, the
right stack the disciplines of the journals it cites. Flow width ∝ number of
citations (fractional counting splits multi-discipline journals).

Chen, C., & Leydesdorff, L. (2014). Patterns of connections and movements in
dual-map overlays. JASIST, 65(2), 334–351.
"""

    def compute(self, bib, o):
        from biblium import dual_map as DM
        flows = bib.dual_map(level=o["level"], fractional=o["fractional"])
        st = bib.dual_map_stats
        if flows.empty:
            raise ValueError("No flows could be mapped (sources not recognised in the ASJC list).")
        return {"summary": [("Docs mapped", f"{100 * st.get('coverage_docs', 0):.0f} %"),
                            ("Refs mapped", f"{100 * st.get('coverage_refs', 0):.0f} %"),
                            ("Self-discipline flow", f"{100 * st.get('self_discipline_share', 0):.0f} %"),
                            ("Cited areas", st.get("n_cited_areas", 0))],
                "tables": {"Flows": flows, "Summary by citing area": bib.dual_map_summary_df,
                           "Matrix": bib.dual_map_matrix_df.reset_index()},
                "plots": {"Dual map": lambda fig: DM.plot_dual_map(flows, st, top_citing=o["top_citing"], top_cited=o["top_cited"], ax=fig.add_subplot(111))}}


class SnowballPanel(SimpleAnalysisPanel):
    title = "Snowball (OpenAlex)"
    icon = "❄️"
    description = "Citation-based corpus expansion via OpenAlex: references (backward) and citing works (forward) of the seed documents. Requires internet."
    OPTIONS = [
        Opt("direction", "Direction", "combo", "both", values=["both", "backward", "forward"]),
        Opt("rounds", "Rounds", "spin", 1, from_=1, to=3),
        Opt("min_links", "Min links to seed set", "spin", 2, from_=1, to=50),
        Opt("max_seeds", "Max seeds (0 = all DOIs)", "spin", 50, from_=0, to=5000),
        Opt("max_candidates", "Max candidates", "spin", 1000, from_=50, to=20000),
        Opt("year_min", "Year from (0 = any)", "spin", 0, from_=0, to=2100),
        Opt("email", "Contact e-mail (polite pool)", "entry", ""),
    ]
    INFO_TEXT = """
SNOWBALLING / CASCADING CITATION EXPANSION
──────────────────────────────────────────
Starting from the DOIs of the loaded corpus (seeds), OpenAlex is queried for
the references of each seed (backward) and the works citing it (forward).
Candidates linked to at least 'min links' seed documents are retained, so the
base set grows with the works most connected to the root set (cf. CiteSpace's
cascading citation expansion; pyBiblioNet's root/base sets). Export the table
and load it as an OpenAlex dataset to analyse the expanded corpus.
"""

    def compute(self, bib, o):
        from biblium import snowball as SB
        dcol = bib._get_column("DOI")
        seeds = [d for d in bib.df[dcol].dropna().astype(str) if d.strip()]
        if o["max_seeds"]:
            seeds = seeds[: o["max_seeds"]]
        if not seeds:
            raise ValueError("No DOIs in the dataset to use as seeds.")
        df = bib.snowball(seeds=seeds, direction=o["direction"], rounds=o["rounds"], min_links=o["min_links"],
                          max_candidates=o["max_candidates"], email=o["email"] or None,
                          year_min=o["year_min"] or None)
        st = bib.snowball_stats
        summ = SB.expansion_summary(df)
        return {"summary": [("Seeds", len(seeds)), ("Candidates", int((df["Direction"] != "seed").sum()) if "Direction" in df.columns else len(df))],
                "message": f"Stats: {st}",
                "tables": {"Candidates": df, "By round": summ},
                "plots": {"Expansion": lambda fig: SB.plot_expansion(df, ax=fig.add_subplot(111))}}
