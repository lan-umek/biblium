# -*- coding: utf-8 -*-
"""
Group analysis methods for BiblioGroup.

This module provides mixin methods for group-level analysis operations.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional

import pandas as pd

from biblium import utilsbib

if TYPE_CHECKING:
    from biblium.bibgroup import BiblioGroup


class GroupAnalysisMixin:
    """Mixin providing analysis methods for BiblioGroup."""

    # Type hints for attributes from BiblioGroup
    df: pd.DataFrame
    group_matrix: pd.DataFrame
    groups: Dict[str, Any]
    group_df: Dict[str, pd.DataFrame]
    group_desc: Any
    default_separator: str
    res_folder: Optional[str]

    def compare_continuous_vars(
        self: "BiblioGroup",
        vrs: List[str] = None,
        output_format: Literal["long", "wide"] = "long",
    ) -> pd.DataFrame:
        """
        Compare continuous variables across groups.

        This method calls ``utilsbib.compare_continuous_by_binary_groups`` on
        the current dataframe and group matrix. The resulting table is stored
        as ``stats_comparison_continous_df``.

        Parameters
        ----------
        vrs : list of str, optional
            Candidate variable names. Only those present in ``self.df`` are
            used. Defaults to ["Year", "Cited by", "Entropy", "Sentiment Score"].
        output_format : {"long", "wide"}, default "long"
            Shape of the returned comparison table.

        Returns
        -------
        pd.DataFrame
            Comparison statistics.
        """
        if vrs is None:
            vrs = ["Year", "Cited by", "Entropy", "Sentiment Score"]
        vrs = [v for v in vrs if v in self.df.columns]
        self.stats_comparison_continous_df = utilsbib.compare_continuous_by_binary_groups(
            self.df, vrs, self.group_matrix, output_format=output_format
        )
        return self.stats_comparison_continous_df

    def get_group_intersections(
        self: "BiblioGroup",
        include_ids: bool = True,
        id_column: str = "Doc ID",
    ) -> pd.DataFrame:
        """
        Compute intersections between groups.

        This uses ``utilsbib.compute_group_intersections`` to determine how
        documents are shared between groups in ``self.group_matrix``.
        The result is stored in ``self.group_intersections_df`` and can
        optionally include document identifiers.

        Parameters
        ----------
        include_ids : bool, default True
            Whether to list document IDs for each intersection.
        id_column : str, default "Doc ID"
            Name of the column in ``self.df`` that contains document IDs.

        Returns
        -------
        pd.DataFrame
            Group intersection statistics.
        """
        id_col_data = self.df[id_column] if include_ids else None
        self.group_intersections_df = utilsbib.compute_group_intersections(
            self.group_matrix,
            include_ids=include_ids,
            id_column=id_col_data,
        )
        return self.group_intersections_df

    def process_keywords(
        self: "BiblioGroup",
        exclude_list: Optional[List[str]] = None,
        synonyms: Optional[Dict[str, List[str]]] = None,
        lemmatize: bool = False,
    ) -> None:
        """
        Preprocess keyword fields and update group-specific dataframes.

        Author, index and combined keyword fields are cleaned using
        ``utilsbib.preprocess_keywords``. The group-specific dataframes
        stored in ``self.group_df`` and the nested BiblioStats instances in
        ``self.groups`` are then refreshed with the updated data.

        Parameters
        ----------
        exclude_list : list or set, optional
            Keywords to remove.
        synonyms : dict, optional
            Mapping of synonym forms to a canonical keyword.
        lemmatize : bool, default False
            If True, lemmatization is applied to keyword tokens.
        """
        sep = self.default_separator

        self.df = utilsbib.preprocess_keywords(
            self.df, "Author Keywords",
            exclude_list=exclude_list, synonyms=synonyms, lemmatize=lemmatize, sep=sep
        )
        self.df = utilsbib.preprocess_keywords(
            self.df, "Index Keywords",
            exclude_list=exclude_list, synonyms=synonyms, lemmatize=lemmatize, sep=sep
        )
        self.df = utilsbib.preprocess_keywords(
            self.df, "Author and Index Keywords",
            exclude_list=exclude_list, synonyms=synonyms, lemmatize=lemmatize, sep=sep
        )

        for group_name in self.group_matrix.columns:
            mask = self.group_matrix[group_name]
            self.group_df[group_name] = self.df[mask]
            self.groups[group_name].set_data(self.group_df[group_name])

    def process_text_vars(
        self: "BiblioGroup",
        stopwords_file: Optional[str] = None,
        lang: str = "en",
        remove_numbers: bool = True,
        remove_two_letter_words: bool = True,
        extra_stopwords: Optional[Iterable[str]] = None,
        exclude_specific_stopwords: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Preprocess long text fields (abstract and title).

        The abstract and title columns are cleaned using
        ``utilsbib.process_text_column`` and the group-specific dataframes and
        nested BiblioStats instances are updated accordingly.

        Parameters
        ----------
        stopwords_file : str or path-like, optional
            Custom stop-word list to use. May contain sheets "general" and
            optionally "specific".
        lang : str, default "en"
            Language code used by the text-processing utilities.
        remove_numbers : bool, default True
            If True, numeric tokens are removed.
        remove_two_letter_words : bool, default True
            If True, 2-letter tokens are removed.
        extra_stopwords : iterable of str or None, optional
            Additional stop-words to exclude for all groups.
        exclude_specific_stopwords : iterable of str or None, optional
            Names of categories from the "specific" sheet in ``stopwords_file``
            whose words should be treated as stopwords.
        """
        common_kwargs = dict(
            stopwords_file=stopwords_file,
            lang=lang,
            remove_numbers=remove_numbers,
            remove_two_letter_words=remove_two_letter_words,
            extra_stopwords=extra_stopwords,
            exclude_specific_stopwords=exclude_specific_stopwords,
        )

        self.df = utilsbib.process_text_column(self.df, "Abstract", **common_kwargs)
        self.df = utilsbib.process_text_column(self.df, "Title", **common_kwargs)

        for group_name in self.group_matrix.columns:
            mask = self.group_matrix[group_name]
            self.group_df[group_name] = self.df[mask]
            self.groups[group_name].set_data(self.group_df[group_name])

    def get_main_info(
        self: "BiblioGroup",
        include: Optional[List[str]] = None,
        performance_mode: str = "full",
        stopwords: Optional[Any] = None,
        excluded_sources_references: Optional[List[str]] = None,
        extra_stats: bool = False,
    ) -> None:
        """
        Compute and merge main bibliometric summaries for each group.

        The method calls :meth:`BiblioStats.get_main_info` on each group in
        ``self.groups`` and then uses ``utilsbib.merge_group_performances``
        to build group-comparable tables.

        Parameters
        ----------
        include : list of str, optional
            Which blocks to include. Recognized values are
            "descriptives", "performance", "time series", "references"
            and "specific". Defaults to ["descriptives", "performance", "time series"].
        performance_mode : str, default "full"
            Performance mode forwarded to the underlying BiblioStats calls.
        stopwords : collection or None, optional
            Extra stop-words for text-based descriptives.
        excluded_sources_references : list or None, optional
            Sources to ignore in reference statistics.
        extra_stats : bool, default False
            If True, request additional descriptive statistics.
        """
        if include is None:
            include = ["descriptives", "performance", "time series"]
        else:
            include = list(include)  # Make a copy to avoid modifying input

        main_info = []
        if isinstance(self.group_desc, str) and self.group_desc == "Year" and "time series" in include:
            include.remove("time series")

        for group_name in self.group_matrix.columns:
            self.groups[group_name].get_main_info(
                include=include,
                performance_mode=performance_mode,
                stopwords=stopwords,
                excluded_sources_references=excluded_sources_references,
                extra_stats=extra_stats,
            )

        if "descriptives" in include:
            self.descriptives_df = utilsbib.merge_group_performances(
                {g: self.groups[g].descriptives_df for g in self.group_matrix.columns}
            )
            main_info.append((self.descriptives_df, "descriptives"))

        if "performance" in include:
            self.performances_df = utilsbib.merge_group_performances(
                {g: self.groups[g].performances_df for g in self.group_matrix.columns}
            )
            main_info.append((self.performances_df, "performances"))

        if "time series" in include:
            self.time_series_stats_df = utilsbib.merge_group_performances(
                {g: self.groups[g].time_series_stats_df for g in self.group_matrix.columns}
            )
            main_info.append((self.time_series_stats_df, "time-series analysis"))

        if "references" in include:
            self.references_stats_df = utilsbib.merge_group_performances(
                {g: self.groups[g].references_stats_df for g in self.group_matrix.columns}
            )
            main_info.append((self.references_stats_df, "references"))

        if self.res_folder is not None:
            utilsbib.save_descriptives_to_excel(
                main_info,
                os.path.join(self.res_folder, "tables", "main info.xlsx"),
            )

    def get_scientific_production(
        self: "BiblioGroup",
        relative_counts: bool = True,
        cumulative: bool = True,
        predict_last_year: bool = True,
        percent_change: bool = True,
        output_format: Literal["wide", "long"] = "wide",
    ) -> pd.DataFrame:
        """
        Compute scientific production statistics by group.

        Delegates to ``utilsbib.get_scientific_production_by_group`` to obtain
        per-year production and citation statistics for each group.

        Parameters
        ----------
        relative_counts : bool, default True
            If True, relative and percentage measures are included.
        cumulative : bool, default True
            If True, cumulative counts over years are added.
        predict_last_year : bool, default True
            If True, attempts to predict production for the last year.
        percent_change : bool, default True
            If True, year-on-year percentage changes are computed.
        output_format : {"wide", "long"}, default "wide"
            Shape of the returned statistics.

        Returns
        -------
        pd.DataFrame
            Scientific production statistics.
        """
        self.production_df = utilsbib.get_scientific_production_by_group(
            self.df,
            self.group_matrix,
            relative_counts=relative_counts,
            cumulative=cumulative,
            predict_last_year=predict_last_year,
            percent_change=percent_change,
            output_format=output_format,
        )
        return self.production_df

    def get_group_top_cited_documents(
        self: "BiblioGroup",
        top_n: int = 10,
        cols: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        mode: Literal["global", "local", "both"] = "global",
        title_col: str = "Title",
        ref_col: str = "References",
        cite_col: str = "Cited by",
    ) -> None:
        """
        Identify and aggregate top-cited documents within and across groups.

        For each group, this method calls the underlying BiblioStats
        ``get_top_cited_documents`` and collects global and/or local top lists
        into group-level tables.

        Parameters
        ----------
        top_n : int, default 10
            Number of top documents to select per group.
        cols : list of str or None, optional
            Additional columns to show in the output.
        filters : dict or None, optional
            Optional filters passed to the underlying selection routine.
        mode : {"global", "local", "both"}, default "global"
            Whether to return global ranking, within-group ranking, or both.
        title_col : str, default "Title"
            Column containing document titles.
        ref_col : str, default "References"
            Column containing reference lists.
        cite_col : str, default "Cited by"
            Column containing citation counts.
        """
        global_frames, local_frames = [], []

        for g in self.group_matrix.columns:
            ga = self.groups[g]
            ga.get_top_cited_documents(top_n, cols, filters, mode, title_col, ref_col, cite_col)

            if mode in {"global", "both"} and ga.top_cited_docs_global_df is not None:
                df = ga.top_cited_docs_global_df.copy()
                df["Group"] = g
                global_frames.append(df)

            if mode in {"local", "both"} and ga.top_cited_docs_local_df is not None:
                df = ga.top_cited_docs_local_df.copy()
                df["Group"] = g
                local_frames.append(df)

        if mode in {"global", "both"} and global_frames:
            self.top_cited_docs_global_group_df = pd.concat(global_frames, ignore_index=True)
            if hasattr(self, "_save_table"):
                self._save_table(self.top_cited_docs_global_group_df, "top cited documents global")

        if mode in {"local", "both"} and local_frames:
            self.top_cited_docs_local_group_df = pd.concat(local_frames, ignore_index=True)
            if hasattr(self, "_save_table"):
                self._save_table(self.top_cited_docs_local_group_df, "top cited documents local")
