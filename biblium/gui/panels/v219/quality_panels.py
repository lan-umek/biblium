# -*- coding: utf-8 -*-
"""Paneli 2.19: PRISMA, dopolnjevanje metapodatkov, lestvice revij, financiranje."""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog

import pandas as pd

from biblium.gui.panels.simple import SimpleAnalysisPanel, Opt


class PrismaPanel(SimpleAnalysisPanel):
    title = "PRISMA Diagram"
    icon = "🧾"
    description = "PRISMA 2020 flow diagram from the numbers of your search/screening process (identification → screening → included)."
    OPTIONS = [
        Opt("db_records", "Records from databases", "spin", 0, from_=0, to=10_000_000, card="🔍 Identification"),
        Opt("register_records", "Records from registers", "spin", 0, from_=0, to=10_000_000, card="🔍 Identification"),
        Opt("other_records", "Records from other sources", "spin", 0, from_=0, to=10_000_000, card="🔍 Identification"),
        Opt("duplicates_removed", "Duplicates removed", "spin", 0, from_=0, to=10_000_000, card="🔍 Identification"),
        Opt("records_excluded", "Records excluded (screening)", "spin", 0, from_=0, to=10_000_000, card="🧹 Screening"),
        Opt("reports_not_retrieved", "Reports not retrieved", "spin", 0, from_=0, to=10_000_000, card="🧹 Screening"),
        Opt("reports_excluded", "Reports excluded (eligibility)", "spin", 0, from_=0, to=10_000_000, card="🧹 Screening"),
        Opt("exclusion_reasons", "Exclusion reasons (a=3; b=2)", "entry", "", card="🧹 Screening"),
        Opt("studies_included", "Studies included (0 = current dataset)", "spin", 0, from_=0, to=10_000_000, card="✅ Included"),
        Opt("title", "Title", "entry", "PRISMA 2020 flow diagram", card="✅ Included"),
    ]
    INFO_TEXT = """
PRISMA 2020
───────────
Enter the numbers you know; derived boxes (records screened, reports sought,
reports assessed, studies included) are computed when left at 0. Exclusion
reasons are given as 'reason=count' pairs separated by ';'.

Page, M. J. et al. (2021). The PRISMA 2020 statement. BMJ, 372, n71.
"""

    def compute(self, bib, o):
        from biblium import prisma as P
        reasons = None
        if o["exclusion_reasons"]:
            reasons = {}
            for part in str(o["exclusion_reasons"]).split(";"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    try:
                        reasons[k.strip()] = int(v)
                    except ValueError:
                        pass
        n_final = o["studies_included"] or (len(bib.df) if bib is not None else None)
        db = o["db_records"] or (n_final if not (o["register_records"] or o["other_records"]) else 0)
        counts = P.PrismaCounts(
            db_records=db, register_records=o["register_records"] or None, other_records=o["other_records"] or None,
            duplicates_removed=o["duplicates_removed"] or 0, records_excluded=o["records_excluded"] or 0,
            reports_not_retrieved=o["reports_not_retrieved"] or 0,
            reports_excluded=o["reports_excluded"] or (sum(reasons.values()) if reasons else 0),
            exclusion_reasons=reasons, studies_included=n_final,
        )
        bib.prisma(counts=counts, title=o["title"], plot=False)
        return {"summary": [("Identified", counts.db_records), ("Screened", counts.records_screened),
                            ("Included", counts.studies_included)],
                "tables": {"Counts": bib.prisma_df},
                "plots": {"PRISMA": lambda fig: P.plot_prisma(counts, title=o["title"], ax=fig.add_subplot(111))}}


class MetadataCompletionPanel(SimpleAnalysisPanel):
    title = "Metadata Completion"
    icon = "🩹"
    description = "Audit missing fields and fill them by DOI from OpenAlex / Crossref (only empty cells, with provenance and undo). Completion requires internet."
    OPTIONS = [
        Opt("mode", "Action", "combo", "audit only", values=["audit only", "complete (OpenAlex → Crossref)", "complete (Crossref only)", "complete (OpenAlex only)", "undo last completion"]),
        Opt("fields", "Fields (comma-separated, empty = all fillable)", "entry", ""),
        Opt("max_rows", "Max rows to query (0 = all)", "spin", 200, from_=0, to=100000),
        Opt("email", "Contact e-mail (polite pool)", "entry", ""),
    ]
    INFO_TEXT = """
MISSING DATA AUDIT AND COMPLETION
─────────────────────────────────
Analogue of bibliometrix 5.4 completeMetadata(): for documents with a DOI the
missing fields (abstract, year, source, authors, citations, …) are looked up
in OpenAlex and/or Crossref. Existing values are never overwritten; every
change is logged (row, field, old, new, source) and can be undone.
"""

    def compute(self, bib, o):
        mode = o["mode"]
        audit = bib.audit_missing()
        if mode == "audit only":
            return {"summary": [("Rows", len(bib.df)), ("Fields with gaps", int((audit["N missing"] > 0).sum()))],
                    "tables": {"Missing data": audit}}
        if mode == "undo last completion":
            bib.undo_complete_metadata()
            return {"message": "Last completion undone.", "tables": {"Missing data": bib.audit_missing()}}
        sources = {"complete (OpenAlex → Crossref)": ("openalex", "crossref"), "complete (Crossref only)": ("crossref",),
                   "complete (OpenAlex only)": ("openalex",)}[mode]
        fields = [f.strip() for f in str(o["fields"]).split(",") if f.strip()] or None
        prov = bib.complete_metadata(fields=fields, sources=sources, email=o["email"] or None,
                                     max_rows=o["max_rows"] or None)
        after = bib.audit_missing()
        return {"summary": [("Cells filled", len(prov)), ("Rows changed", int(prov["Row index"].nunique()) if len(prov) else 0)],
                "tables": {"Provenance": prov, "Summary": bib.metadata_completion_summary_df, "Missing after": after,
                           "Missing before": audit}}


class JournalRankingPanel(SimpleAnalysisPanel):
    title = "Journal Rankings"
    icon = "🏆"
    description = "Match sources to a journal ranking list (SCImago SJR CSV, JCR, ABS, ABDC … by ISSN or title), summarise by quartile and optionally filter the dataset."
    OPTIONS = [
        Opt("path", "Ranking file (csv/xlsx)", "entry", ""),
        Opt("title_col", "Title column (auto if empty)", "entry", ""),
        Opt("issn_col", "ISSN column (auto if empty)", "entry", ""),
        Opt("quartile_col", "Quartile column (auto if empty)", "entry", ""),
        Opt("rank_col", "Rank column (auto if empty)", "entry", ""),
        Opt("filter_quartiles", "Filter dataset to quartiles (e.g. Q1,Q2; empty = no filter)", "entry", ""),
        Opt("keep_unranked", "Keep unranked when filtering", "check", False),
    ]
    RUN_LABEL = "Match & Summarise"

    def _create_options(self):
        super()._create_options()
        from biblium.gui.widgets.buttons import ThemedButton
        ThemedButton(self.options_content, text="Browse ranking file…", icon="📂", style="secondary",
                     command=self._browse, theme=self.theme_name).pack(fill=tk.X, padx=8, pady=(0, 8))

    def _browse(self):
        p = filedialog.askopenfilename(filetypes=[("Ranking lists", "*.csv *.xlsx *.xls"), ("All", "*.*")])
        if p:
            opt, w = self._widgets["path"]
            try:
                w.entry.delete(0, tk.END); w.entry.insert(0, p)
            except Exception:
                try:
                    w.set(p)
                except Exception:
                    pass

    def compute(self, bib, o):
        from biblium import journal_rankings as JR
        if not o["path"]:
            raise ValueError("Choose a ranking list file (e.g. SCImago CSV export).")
        kw = {k: o[k] for k in ("title_col", "issn_col", "quartile_col", "rank_col") if o[k]}
        summ = bib.add_journal_ranking(o["path"], **kw)
        by_year = bib.journal_ranking_by_year_df
        matched = int(bib.df["Ranking match"].notna().sum())
        msg = None
        if o["filter_quartiles"]:
            qs = [q.strip().upper() for q in str(o["filter_quartiles"]).split(",") if q.strip()]
            sub = bib.filter_by_journal_ranking(quartiles=qs, keep_unranked=o["keep_unranked"], inplace=True)
            msg = f"Dataset filtered to {qs}: {len(sub)} documents remain (reload the data to restore)."
        match_tab = (bib.df["Ranking match"].fillna("none").value_counts().rename_axis("Match type")
                     .reset_index(name="Documents"))
        return {"summary": [("Matched docs", matched), ("Match rate", f"{100 * matched / max(1, len(bib.df)):.0f} %"),
                            ("Journals in list", len(bib.journal_ranking_df))],
                "message": msg,
                "tables": {"By quartile": summ, "By year": by_year, "Match types": match_tab},
                "plots": {"Quartiles by year": lambda fig: JR.plot_ranking_distribution(by_year, ax=fig.add_subplot(111))}}


class FundingPanel(SimpleAnalysisPanel):
    title = "Funding Analysis"
    icon = "💶"
    description = "Funders, funding trend and impact of funded vs. unfunded documents (Scopus 'Funding Details' / 'Funding Texts')."
    OPTIONS = [
        Opt("top_n", "Top N funders", "spin", 30, from_=5, to=500),
        Opt("min_docs", "Min documents per funder", "spin", 1, from_=1, to=100),
    ]

    def compute(self, bib, o):
        from biblium import funding as F
        funders = bib.funding_analysis(top_n=o["top_n"], min_docs=o["min_docs"])
        by_year = bib.funding_by_year_df
        comp = bib.funding_comparison
        comp_df = pd.DataFrame([(k, v) for k, v in comp.items()], columns=["Statistic", "Value"])
        p = comp.get("p_value", comp.get("mannwhitney_p", float("nan")))
        return {"summary": [("Funded docs", comp.get("n_funded", 0)), ("Unfunded docs", comp.get("n_unfunded", 0)),
                            ("MNCS funded", f"{comp.get('mncs_funded', float('nan')):.2f}"),
                            ("MNCS unfunded", f"{comp.get('mncs_unfunded', float('nan')):.2f}")],
                "message": f"Mann–Whitney U test (citations funded vs. unfunded): p = {p:.4f}" if p == p else None,
                "tables": {"Funders": funders, "By year": by_year, "Funded vs unfunded": comp_df},
                "plots": {"Top funders": lambda fig: F.plot_top_funders(funders, ax=fig.add_subplot(111)),
                          "Trend": lambda fig: F.plot_funding_trend(by_year, ax=fig.add_subplot(111))}}
