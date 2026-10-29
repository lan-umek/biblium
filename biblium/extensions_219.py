# -*- coding: utf-8 -*-
"""
biblium.extensions_219
======================

``Extensions219Mixin`` — ovoji ``BiblioStats`` za drugi paket novosti 2.19:
PRISMA, dopolnjevanje metapodatkov, lestvice revij, financiranje, RPYS
vzorci, diakrone mreže / časovnica klastrov / structural variation / fuzzy
klastri, dual-map overlay, snowballing (OpenAlex) in VOSviewer-slog
ekstrakcije izrazov. Vsaka metoda shrani rezultat kot atribut, zabeleži
parametre v ``_analysis_params`` (za ``methods_text``) in ob nastavljenem
``res_folder`` zapiše xlsx / sliko (prek ``_save_table`` / ``_save_plot``).

Vsa računska logika je v samostojnih modulih (``biblium.prisma``,
``biblium.metadata_completion``, ``biblium.journal_rankings``,
``biblium.funding``, ``biblium.reference_patterns``,
``biblium.network_dynamics``, ``biblium.dual_map``, ``biblium.snowball``,
``biblium.term_extraction``), ki so uporabni tudi brez ``BiblioStats``.

Version: 2.19.0
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

import pandas as pd


class Extensions219Mixin:
    """Mixin z metodami 2.19 (drugi paket). Pričakuje ``self.df``,
    ``self._get_column``, ``self._save_table``, ``self._save_plot``,
    ``self._entity_sep``, ``self._record_params`` iz ``BiblioStats``."""

    # ------------------------------------------------------------------ helpers
    def _ref_col(self, references_col="References"):
        return references_col if references_col in self.df.columns else self._get_column(references_col)

    def _plot_fig(self, name: str, drawer, **kw):
        """Nariše s podano funkcijo ``drawer(ax=None, **kw)`` in shrani sliko."""
        fig_ax = drawer(**kw)
        self._save_plot(name)
        return fig_ax

    # ------------------------------------------------------------------ PRISMA
    def prisma(self, counts=None, n_identified=None, n_after_dedup=None, n_final=None,
               n_screened_out=None, title=None, plot=True, **kwargs):
        """
        PRISMA 2020 diagram poteka. Podaj ``counts`` (``biblium.prisma.PrismaCounts``)
        ali osnovne številke (``n_identified``, ``n_after_dedup``, ``n_final``,
        ...); manjkajoče ``n_final`` = trenutni ``len(self.df)``. Shrani
        ``self.prisma_counts`` in ``self.prisma_df``.
        """
        from biblium import prisma as P
        if counts is None:
            if n_final is None:
                n_final = len(self.df)
            if n_after_dedup is None:
                n_after_dedup = getattr(self, "_n_after_dedup", n_identified if n_identified is not None else n_final)
            if n_identified is None:
                n_identified = getattr(self, "_n_identified", n_after_dedup)
            counts = P.prisma_counts_from_pipeline(n_identified=n_identified, n_after_dedup=n_after_dedup,
                                                   n_final=n_final, n_screened_out=n_screened_out, **kwargs)
        counts.fill_derived()
        self.prisma_counts = counts
        self.prisma_df = counts.to_dataframe()
        self._record_params("prisma", n_identified=n_identified, n_final=n_final)
        self._save_table(self.prisma_df, "prisma counts")
        if plot:
            P.plot_prisma(counts, title=title)
            self._save_plot("prisma flow diagram")
        return self.prisma_df

    # ------------------------------------------------------------------ metadata completion
    def audit_missing(self, fields=None):
        """Revizija manjkajočih vrednosti po poljih (``self.missing_audit_df``)."""
        from biblium.metadata_completion import audit_missing
        self.missing_audit_df = audit_missing(self.df, fields=fields)
        self._save_table(self.missing_audit_df, "missing data audit")
        return self.missing_audit_df

    def complete_metadata(self, fields=None, sources=("openalex", "crossref"), overwrite=False,
                          email=None, max_rows=None, delay=0.1, **kwargs):
        """
        Dopolni manjkajoča polja po DOI iz OpenAlex/Crossref (bibliometrix
        ``completeMetadata``). Spremeni ``self.df`` (samo prazne celice, razen
        ``overwrite=True``), shrani ``self.metadata_provenance_df`` in varnostno
        kopijo za ``undo_complete_metadata()``. Zahteva mrežo (razen ob
        injiciranih ``fetch_openalex``/``fetch_crossref``).
        """
        from biblium.metadata_completion import complete_metadata, completion_summary
        self._metadata_backup = self.df.copy()
        new_df, prov = complete_metadata(self.df, fields=fields, sources=sources, overwrite=overwrite,
                                         email=email, max_rows=max_rows, delay=delay, **kwargs)
        self.df = new_df
        self.metadata_provenance_df = prov
        self.metadata_completion_summary_df = completion_summary(prov) if len(prov) else pd.DataFrame()
        self._record_params("complete_metadata", sources=list(sources), overwrite=overwrite,
                            n_changed=int(len(prov)))
        self._save_table(prov, "metadata completion provenance")
        return prov

    def undo_complete_metadata(self):
        """Razveljavi zadnje dopolnjevanje metapodatkov."""
        from biblium.metadata_completion import undo_completion
        if getattr(self, "metadata_provenance_df", None) is not None and len(self.metadata_provenance_df):
            self.df = undo_completion(self.df, self.metadata_provenance_df)
        elif getattr(self, "_metadata_backup", None) is not None:
            self.df = self._metadata_backup.copy()
        self.metadata_provenance_df = pd.DataFrame()
        return self.df

    # ------------------------------------------------------------------ journal rankings
    def add_journal_ranking(self, ranking, **load_kwargs):
        """
        Pripne lestvico revij (SJR/JCR/ABS/ABDC …; pot do csv/xlsx ali DataFrame)
        po ISSN oz. naslovu: stolpci ``Journal Rank``, ``Journal Quartile``,
        ``Journal Category``, ``Ranking match``. Shrani
        ``self.journal_ranking_summary_df`` in ``self.journal_ranking_by_year_df``.
        """
        from biblium import journal_rankings as JR
        rk = JR.load_ranking_list(ranking, **load_kwargs)
        self.journal_ranking_df = rk
        self.df = JR.match_sources(self.df, rk, source_col=self._get_column("Source title"),
                                   issn_col=self._get_column("ISSN", required=False) or "ISSN")
        self.journal_ranking_summary_df = JR.ranking_summary(
            self.df, citations_col=self._get_column("Cited by"), year_col=self._get_column("Year"))
        self.journal_ranking_by_year_df = JR.ranking_by_year(self.df, year_col=self._get_column("Year"))
        matched = int(self.df["Ranking match"].notna().sum()) if "Ranking match" in self.df.columns else 0
        self._record_params("journal_ranking", n_ranked=len(rk), n_matched=matched)
        self._save_table(self.journal_ranking_summary_df, "journal ranking summary")
        self._save_table(self.journal_ranking_by_year_df, "journal ranking by year")
        return self.journal_ranking_summary_df

    def filter_by_journal_ranking(self, quartiles=None, max_rank=None, categories=None, keep_unranked=False,
                                  inplace=False):
        """Podmnožica dokumentov po kvartilih / rangu / kategoriji lestvice."""
        from biblium.journal_rankings import filter_by_ranking
        if "Journal Quartile" not in self.df.columns:
            raise ValueError("Call add_journal_ranking() first")
        sub = filter_by_ranking(self.df, quartiles=quartiles, max_rank=max_rank, categories=categories,
                                keep_unranked=keep_unranked)
        if inplace:
            self.df = sub.reset_index(drop=True)
        return sub

    # ------------------------------------------------------------------ funding
    def funding_analysis(self, top_n=30, min_docs=1, plot=False, **plot_kwargs):
        """
        Analiza financiranja (Scopus 'Funding Details'/'Funding Texts'): zastavice
        ``Funded``/``Funders`` v ``self.df``, tabela financerjev, trend po letih,
        primerjava financirani vs. nefinancirani (MNCS, Mann–Whitney).
        Shrani ``funders_df``, ``funding_by_year_df``, ``funding_comparison``.
        """
        from biblium import funding as F
        dcol = self._get_column("Funding Details", required=False)
        tcol = self._get_column("Funding Texts", required=False)
        if dcol is None and tcol is None:
            raise ValueError("No funding columns ('Funding Details' / 'Funding Texts') in the dataset")
        kw = dict(details_col=dcol or "Funding Details", text_col=tcol or "Funding Texts")
        flagged = F.add_funding_flags(self.df, **kw)
        for c in ("Funded", "N funders", "Funders"):
            if c in flagged.columns:
                self.df[c] = flagged[c].values
        cit, year = self._get_column("Cited by"), self._get_column("Year")
        self.funders_df = F.funder_counts(self.df, details_col=kw["details_col"], citations_col=cit,
                                          year_col=year, top_n=top_n, min_docs=min_docs)
        self.funding_by_year_df = F.funding_by_year(self.df, details_col=kw["details_col"], text_col=kw["text_col"],
                                                    citations_col=cit, year_col=year)
        self.funding_comparison = F.funded_vs_unfunded(self.df, citations_col=cit, year_col=year)
        self._record_params("funding_analysis", top_n=top_n)
        self._save_table(self.funders_df, "funders")
        self._save_table(self.funding_by_year_df, "funding by year")
        if plot:
            F.plot_top_funders(self.funders_df, **plot_kwargs); self._save_plot("top funders")
            F.plot_funding_trend(self.funding_by_year_df); self._save_plot("funding trend")
        return self.funders_df

    # ------------------------------------------------------------------ RPYS patterns
    def rpys_patterns(self, references_col="References", min_citations=3, median_mode="centered",
                      median_window=5, plot=False, **classify_kwargs):
        """
        RPYS z izbiro mediane (centered/backward) in klasifikacijo citiranih
        referenc v vzorce (Hot Paper, Constant Performer, Life Cycle, Sleeping
        Beauty). Shrani ``rpys_ext_df``, ``reference_timeseries_df``,
        ``reference_patterns_df``, ``reference_pattern_summary_df``.
        """
        from biblium import reference_patterns as RP
        ref = self._ref_col(references_col)
        self.rpys_ext_df = RP.compute_rpys_backward(self.df, reference_column=ref, median_window=median_window,
                                                    mode=median_mode)
        ts = RP.cited_reference_timeseries(self.df, reference_column=ref, year_col=self._get_column("Year"),
                                           sep=self._entity_sep(ref), min_citations=min_citations)
        self.reference_timeseries_df = ts
        self.reference_patterns_df = RP.classify_reference_patterns(ts, **classify_kwargs)
        self.reference_pattern_summary_df = RP.rpys_pattern_summary(self.reference_patterns_df)
        self._record_params("rpys_patterns", median_mode=median_mode, median_window=median_window,
                            min_citations=min_citations)
        self._save_table(self.rpys_ext_df, "rpys extended")
        self._save_table(self.reference_patterns_df, "reference patterns")
        if plot:
            RP.plot_rpys_extended(self.rpys_ext_df); self._save_plot("rpys extended")
            RP.plot_reference_patterns(self.reference_patterns_df, ts); self._save_plot("reference patterns")
        return self.reference_patterns_df

    # ------------------------------------------------------------------ network dynamics
    def diachronic_networks(self, field="Author Keywords", n_slices=3, periods=None, cumulative=True,
                            top_n=50, min_count=2, min_weight=1, seed=42, animate=False, gif_name="diachronic network.gif",
                            fps=1, plot=False, **kwargs):
        """
        Sobesedne mreže po obdobjih s stabilno množico vozlišč (``period_graphs``);
        ``animate=True`` zapiše GIF (v ``res_folder/plots`` ali CWD), ``plot=True``
        statično mrežo malih slik. Shrani ``diachronic_graphs``, ``diachronic_labels``.
        """
        from biblium import network_dynamics as ND
        col = field if field in self.df.columns else self._get_column(field)
        graphs, labels = ND.period_graphs(self.df, field=col, year_col=self._get_column("Year"), periods=periods,
                                          n_slices=n_slices, cumulative=cumulative, sep=self._entity_sep(col),
                                          min_count=min_count, min_weight=min_weight, top_n=top_n, seed=seed)
        self.diachronic_graphs, self.diachronic_labels = graphs, labels
        self._record_params("diachronic_networks", field=col, n_slices=n_slices, cumulative=cumulative,
                            top_n=top_n, seed=seed)
        out = None
        if animate and graphs:
            folder = os.path.join(self.res_folder, "plots") if self.res_folder else "."
            os.makedirs(folder, exist_ok=True)
            out = ND.animate_diachronic_network(graphs, labels, os.path.join(folder, gif_name), fps=fps, seed=seed, **kwargs)
            self.diachronic_animation_path = out
        if plot and graphs:
            ND.plot_diachronic_grid(graphs, labels, seed=seed); self._save_plot("diachronic networks")
        return graphs, labels

    def fuzzy_clusters(self, field="Author Keywords", G=None, partition_attr="partition_louvain", min_membership=0.5):
        """
        Mehka dodelitev dokumentov klastrom sobesedne mreže (deleži ključnih
        besed po klastrih). Brez ``G`` uporabi zadnjo diakrono/celotno mrežo
        (zgradi jo, če je ni). Shrani ``fuzzy_membership_df``.
        """
        from biblium import network_dynamics as ND
        col = field if field in self.df.columns else self._get_column(field)
        if G is None:
            graphs = getattr(self, "diachronic_graphs", None)
            if not graphs:
                graphs, _ = ND.period_graphs(self.df, field=col, year_col=self._get_column("Year"), n_slices=1,
                                             cumulative=True, sep=self._entity_sep(col))
            G = graphs[-1]
        self.fuzzy_membership_df = ND.fuzzy_cluster_membership(self.df, G, field=col, partition_attr=partition_attr,
                                                               sep=self._entity_sep(col))
        self.fuzzy_graph = G
        self._record_params("fuzzy_clusters", field=col, partition_attr=partition_attr)
        self._save_table(self.fuzzy_membership_df, "fuzzy cluster membership")
        return self.fuzzy_membership_df

    def cluster_timeline(self, field="Author Keywords", G=None, partition_attr="partition_louvain", year_stat="mean",
                         top_labels=5, min_cluster_size=3, plot=False, **plot_kwargs):
        """
        CiteSpace-slog časovnice klastrov (vozlišča na pasovih po klastrih, x = leto).
        Klastri z manj kot ``min_cluster_size`` vozlišči so izpuščeni (louvain na
        redkih mrežah ustvari veliko samostojnih vozlišč). Shrani
        ``cluster_timeline_df``, ``cluster_timeline_labels``.
        """
        from biblium import network_dynamics as ND
        col = field if field in self.df.columns else self._get_column(field)
        if G is None:
            graphs = getattr(self, "diachronic_graphs", None)
            if not graphs:
                graphs, _ = ND.period_graphs(self.df, field=col, year_col=self._get_column("Year"), n_slices=1,
                                             cumulative=True, sep=self._entity_sep(col))
            G = graphs[-1]
        table, labels = ND.cluster_timeline(G, self.df, field=col, year_col=self._get_column("Year"),
                                            partition_attr=partition_attr, sep=self._entity_sep(col),
                                            year_stat=year_stat, top_labels=top_labels)
        if min_cluster_size and len(table):
            sizes = table["Cluster"].value_counts()
            keep = set(sizes[sizes >= min_cluster_size].index)
            table = table[table["Cluster"].isin(keep)].reset_index(drop=True)
            labels = {k: v for k, v in labels.items() if k in keep}
        self.cluster_timeline_df, self.cluster_timeline_labels, self.cluster_timeline_graph = table, labels, G
        self._record_params("cluster_timeline", field=col, year_stat=year_stat)
        self._save_table(table, "cluster timeline")
        if plot:
            ND.plot_cluster_timeline(table, labels, G=G, **plot_kwargs); self._save_plot("cluster timeline")
        return table

    def structural_variation(self, references_col="References", min_ref_citations=3, top_n_refs=150,
                             baseline_window=None, seed=42, plot=False, **kwargs):
        """
        Structural variation analysis (Chen 2012): za vsak dokument sprememba
        modularnosti, cluster linkage in centrality divergence glede na
        sociacijsko mrežo referenc pred letom objave. Shrani ``structural_variation_df``.
        """
        from biblium import network_dynamics as ND
        ref = self._ref_col(references_col)
        self.structural_variation_df = ND.structural_variation(
            self.df, reference_column=ref, year_col=self._get_column("Year"), sep=self._entity_sep(ref),
            min_ref_citations=min_ref_citations, top_n_refs=top_n_refs, baseline_window=baseline_window,
            seed=seed, **kwargs)
        self._record_params("structural_variation", min_ref_citations=min_ref_citations, top_n_refs=top_n_refs,
                            baseline_window=baseline_window, seed=seed)
        self._save_table(self.structural_variation_df, "structural variation")
        if plot and len(self.structural_variation_df):
            ND.plot_structural_variation(self.structural_variation_df); self._save_plot("structural variation")
        return self.structural_variation_df

    # ------------------------------------------------------------------ dual map
    def dual_map(self, level="Area", fractional=True, min_flow=1, references_col="References", plot=False, **plot_kwargs):
        """
        Dual-map overlay (CiteSpace analog na podlagi ASJC): discipline citirajočih
        revij ↔ discipline citiranih revij v referencah. Shrani ``dual_map_flows_df``,
        ``dual_map_stats``, ``dual_map_matrix_df``, ``dual_map_summary_df``.
        """
        from biblium import dual_map as DM
        ref = self._ref_col(references_col)
        flows, stats = DM.dual_map_flows(self.df, level=level, fractional=fractional, min_flow=min_flow,
                                         source_col=self._get_column("Source title"), references_col=ref,
                                         sep=self._entity_sep(ref))
        self.dual_map_flows_df, self.dual_map_stats = flows, stats
        self.dual_map_matrix_df = DM.dual_map_matrix(flows)
        self.dual_map_summary_df = DM.dual_map_summary(flows)
        self._record_params("dual_map", level=level, fractional=fractional)
        self._save_table(flows, "dual map flows")
        self._save_table(self.dual_map_summary_df, "dual map summary")
        if plot and len(flows):
            DM.plot_dual_map(flows, stats, **plot_kwargs); self._save_plot("dual map overlay")
        return flows

    # ------------------------------------------------------------------ snowball
    def snowball(self, seeds=None, direction="both", rounds=1, min_links=1, max_candidates=2000, email=None,
                 year_min=None, year_max=None, fetch_json=None, **kwargs):
        """
        Razširitev korpusa prek OpenAlex (reference + citirajoči članki). Semena
        so DOI-ji ali OpenAlex ID-ji; brez ``seeds`` se uporabijo DOI-ji
        trenutnega korpusa. Shrani ``snowball_df`` in ``snowball_stats``.
        Zahteva mrežo (razen ob injiciranem ``fetch_json``).
        """
        from biblium import snowball as SB
        if seeds is None:
            dcol = self._get_column("DOI")
            seeds = [d for d in self.df[dcol].dropna().astype(str) if d.strip()]
        self.snowball_df, self.snowball_stats = SB.snowball(
            seeds, direction=direction, rounds=rounds, min_links=min_links, max_candidates=max_candidates,
            email=email, year_min=year_min, year_max=year_max, fetch_json=fetch_json, **kwargs)
        self._record_params("snowball", direction=direction, rounds=rounds, min_links=min_links,
                            n_seeds=len(seeds))
        self._save_table(self.snowball_df, "snowball candidates")
        return self.snowball_df

    # ------------------------------------------------------------------ term extraction
    def extract_terms(self, columns=("Title", "Abstract"), max_len=3, min_occurrences=10, share=0.6,
                      add_column=True, col_name="Extracted Terms", plot=False, **kwargs):
        """
        VOSviewer-slog ekstrakcije izrazov iz naslovov/povzetkov z relevance
        score (van Eck & Waltman 2011). Shrani ``terms_df``, ``term_matrix``,
        ``term_list``, ``term_cooccurrence_df`` in (``add_column=True``) stolpec
        ``Extracted Terms`` v ``self.df`` za nadaljnje analize.
        """
        from biblium import term_extraction as TE
        cols = tuple(c for c in columns if c in self.df.columns)
        if not cols:
            raise ValueError(f"None of the text columns {columns} found")
        scores, X, terms = TE.extract_terms(self.df, columns=cols, max_len=max_len, min_occurrences=min_occurrences,
                                            share=share, **kwargs)
        self.terms_df, self.term_matrix, self.term_list = scores, X, terms
        self.term_cooccurrence_df = TE.term_cooccurrence(X, terms)
        if add_column:
            self.df = TE.add_terms_column(self.df, X, terms, col=col_name)
        self._record_params("extract_terms", columns=list(cols), max_len=max_len, min_occurrences=min_occurrences,
                            share=share, n_terms=len(terms))
        self._save_table(scores, "extracted terms")
        if plot and len(scores):
            TE.plot_term_relevance(scores); self._save_plot("term relevance")
        return scores


__all__ = ["Extensions219Mixin"]
