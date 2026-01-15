# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 11:50:08 2025

@author: Lan.Umek
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
)
import math
import networkx as nx
import numpy as np
import os
import pandas as pd
import re
from biblium import readbib
from biblium import reportbib
from biblium import utilsbib
from biblium.base import BiblioBase
from biblium.bibplot_modules.race_bar import RaceBarMixin
from biblium.bibplot_modules.advanced import AdvancedVisualizationsMixin
from biblium.disruption import DisruptionMixin


class BiblioStats(BiblioBase, RaceBarMixin, AdvancedVisualizationsMixin, DisruptionMixin):
    """
    Core bibliometric statistics and counting functionality.
    
    This class provides methods for:
    - Counting entities (sources, authors, keywords, etc.)
    - Computing statistics and performance indicators
    - Text processing and keyword analysis
    - Country and affiliation analysis
    
    Attributes
    ----------
    df : pd.DataFrame
        The main bibliometric dataset.
    n : int
        Number of documents in the dataset.
    db : str
        Database type (e.g., "scopus", "oa", "wos").
    default_separator : str
        Default separator for multi-value fields.
    res_folder : Optional[str]
        Path to results folder, or None to disable saving.
    """

    # Class-level type hints for instance attributes
    df: pd.DataFrame
    n: int
    db: str
    default_separator: str
    res_folder: Optional[str]
    mapping: Dict[str, str]
    sources_abb_dict: Dict[str, str]
    kw_var: Optional[str]
    author_var: str
    
    # Count result DataFrames
    sources_counts_df: pd.DataFrame
    authors_counts_df: pd.DataFrame
    author_keywords_counts_df: pd.DataFrame
    index_keywords_counts_df: pd.DataFrame
    keywords_counts_df: pd.DataFrame
    affiliations_counts_df: pd.DataFrame
    document_types_counts_df: pd.DataFrame
    ca_country_counts_df: pd.DataFrame
    fields_counts_df: pd.DataFrame
    areas_counts_df: pd.DataFrame
    sciences_counts_df: pd.DataFrame
    references_counts_df: pd.DataFrame
    words_abs_counts_df: pd.DataFrame
    words_tit_counts_df: pd.DataFrame
    words_comb_counts_df: pd.DataFrame
    
    # Binary indicator DataFrames
    binary_sources_df: pd.DataFrame
    binary_authors_df: pd.DataFrame
    binary_author_keywords_df: pd.DataFrame
    binary_index_keywords_df: pd.DataFrame
    binary_keywords_df: pd.DataFrame

    def __init__(
        self,
        f_name: Optional[str] = None,
        db: str = "",
        df: Optional[pd.DataFrame] = None,
        res_folder: Optional[str] = "results",
        output_lang: str = "en",
        preprocess_level: int = 0,
        exclude_list_kw: Optional[List[str]] = None,
        synonyms_kw: Optional[Dict[str, List[str]]] = None,
        lemmatize_kw: bool = False,
        default_keywords: Literal["author", "index", "both"] = "author",
        combine_with_index_keywords: bool = False,
        concept_df: Optional[pd.DataFrame] = None,
        concept_column: Optional[str] = None,
        asjc_map_df: Optional[pd.DataFrame] = None,
        lang_of_docs: str = "en",
        fancy_output: bool = False,
        label_docs: bool = True,
        dpi: int = 600,
        cmap: str = "viridis",
        cmap_disc: str = "tab10",
        default_color: str = "lightblue",
        extra_stopwords: Optional[List[str]] = None,
        specific_stopword_categories: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize a BiblioStats analysis object.
        
        The constructor loads a bibliographic dataset (from a file or a pre-built
        DataFrame), normalizes key fields and builds helper mappings for sources,
        countries and subject areas. It also stores configuration that controls
        text pre-processing, keyword handling and where outputs are written.
        
        Parameters
        ----------
        f_name : str or path-like, optional
            Path to a bibliographic export file. Used together with `db` when
            `df` is not provided.
        db : str, default ""
            Short code of the data source (for example "scopus" or "oa").
        df : pandas.DataFrame, optional
            Pre-loaded dataframe to use instead of reading `f_name`.
        res_folder : str, default "results"
            Root folder for saving tables, plots and reports.
        output_lang : str, default "en"
            Language code used for user-facing messages.
        preprocess_level : int, default 0
            Controls how strongly `utilsbib.preprocess_df` and related text
            utilities are applied to text columns:
    
            - 0: basic loading, minimal processing.
            - 1: add country info, labels, missing-value diagnostics.
            - 2: add keyword processing and text processing (title/abstract),
                 including optional specific stopword categories.
            - 3: add science mappings and merged text fields.
            - 4: compute interdisciplinarity measures (if available).
    
        exclude_list_kw, synonyms_kw, lemmatize_kw, default_keywords,
        combine_with_index_keywords :
            Options that configure how keyword variables are cleaned and merged.
        concept_df, concept_column, asjc_map_df :
            Optional auxiliary tables used for concept mapping and Scopus ASJC
            enrichment.
        lang_of_docs : str, default "en"
            Language of the original documents (used by some NLP utilities,
            including stopword selection).
        fancy_output : bool, default False
            If True, enables more stylised Excel output.
        label_docs : bool, default True
            Whether to add a simple "Doc ID" running label column.
        dpi, cmap, cmap_disc, default_color :
            Default plotting options used by plotting helpers.
        extra_stopwords : iterable of str or None, optional
            Additional stopwords applied to all text fields during preprocessing
            (for example database-specific boilerplate). They are always used and
            are independent of any "specific" sheet.
        specific_stopword_categories : iterable of str or None, optional
            Category names taken from the "specific" sheet of the stopword Excel
            file (see `utilsbib.process_text_column`). Only words belonging to
            these categories are treated as additional stopwords.
    
            Category matching is case-insensitive after stripping. Typical values:
    
            - "Scholarly boilerplate"
            - "Methods and analysis"
            - "Publication document type terms"
            - "Scope and qualifiers"
            - "Bibliometrics specific terms"
            - "Generic science vocabulary"
            - "Section labels and boilerplate"
    
            If None or empty, the "specific" sheet is ignored and only general
            stopwords (plus `extra_stopwords`) are used.
        
        Notes
        -----
        On success the main dataframe is available as `self.df` and the
        number of documents as `self.n`.
        """
        # Initialize cache
        self._init_cache()
        
        self.db = db.lower()
        # Normalize OpenAlex variants to canonical "oa" for consistent internal checks
        if self.db in ("openalex", "open alex"):
            self.db = "oa"
        self.ldf = lambda x: utilsbib.ldf(x, l=output_lang)
        
        if df is not None:
            self.df = df
        elif f_name is not None:
            self.df = readbib.read_bibfile(f_name, self.db)
        else:
            print(
                self.ldf("No dataset has been provided."),
                self.ldf("The bibliometric analysis cannot be performed."),
            )
            return None
    
        mapping_path = os.path.join(os.path.dirname(__file__), "additional files", "mappings.xlsx")
        mapping_df = pd.read_excel(mapping_path, sheet_name="mapping")
        alias_df = pd.read_excel(mapping_path, sheet_name="alias")
        self.mapping = utilsbib.reconstruct_mapping(mapping_df, alias_df)
    
        # Default separators for multi-value fields by database
        # Most databases use "; " but OpenAlex uses "|"
        separator_map = {
            # Major databases
            "scopus": "; ",
            "oa": "|",  # OpenAlex (canonical name)
            "openalex": "|",  # OpenAlex alias
            "open alex": "|",  # OpenAlex alias
            "wos": "; ",
            "pubmed": "; ",
            "dimensions": "; ",
            "lens": "; ",
            # Engineering/CS
            "ieee": "; ",
            "dblp": "; ",
            "arxiv": "; ",
            "semantic scholar": "; ",
            "s2": "; ",
            # Social sciences
            "proquest": "; ",
            "ebsco": "; ",
            "jstor": "; ",
            "psycinfo": "; ",
            "eric": "; ",
            "econlit": "; ",
            # Other sciences
            "mathscinet": "; ",
            "inspec": "; ",
            "georef": "; ",
            "cab": "; ",
            "cinahl": "; ",
            "embase": "; ",
            "cochrane": "; ",
            # Generic formats
            "ris": "; ",
            "bibtex": " and ",  # BibTeX uses " and " for authors
            "bib": " and ",
            "endnote": "; ",
            "zotero": "; ",
            "csv": "; ",
            "crossref": "; ",
            "orcid": "; ",
        }
        self.default_separator = separator_map.get(self.db, "; ")
        self.concept_df = concept_df
    
        self.n = len(self.df)
        if label_docs:
            self.df["Doc ID"] = [f"Doc {i}" for i in range(1, self.n + 1)]
    
        if ("Source title" in self.df.columns) and (
            "Abbreviated Source Title" not in self.df.columns
        ):
            self.df["Abbreviated Source Title"] = self.df["Source title"].map(
                utilsbib.abbreviate_words
            )
    
        # mapping dictionaries
        if "Source title" in self.df.columns and "Abbreviated Source Title" in self.df.columns:
            self.sources_abb_dict = self.df.set_index("Source title")[
                "Abbreviated Source Title"
            ].to_dict()
        else:
            self.sources_abb_dict = {}
    
        if res_folder is not None:
            self.res_folder = os.path.join(os.getcwd(), res_folder)
            sub_folders = ["plots", "tables", "reports", "networks", "relations"]
            folders = [os.path.join(self.res_folder, s) for s in sub_folders]
            utilsbib.make_folders(folders)
            if fancy_output:
                self.cond_formatting, self.autofit = True, True
            else:
                self.cond_formatting, self.autofit = False, False
        else:
            self.res_folder = None
    
        self.relations = {}
    
        if preprocess_level >= 1:
            if "Cited by" in self.df.columns:
                """Compute percent rank of "Cited by" (NaN-safe): higher citations → higher %, NaNs preserved."""
                self.df["Percent Rank Cited by"] = (
                    pd.to_numeric(self.df["Cited by"], errors="coerce")
                    .rank(pct=True, method="average")
                    .mul(100)
                    .round(2)
                )
    
            self.df = utilsbib.add_ca_country_df(self.df, self.db)
            self.missings_df, self.missings = utilsbib.check_missing_values(self.df)
            self.df = utilsbib.add_document_labels_abbrev(self.df)
            doc_id_index = self.df.set_index("Doc ID")
            self.id_short_label_dict = doc_id_index.to_dict()["Document Short Label"]
            self.id_label_dict = doc_id_index.to_dict()["Document Label"]
            if self.db == "scopus":
                aff_column = "Affiliations"
                self.df, self.country_collab_matrix = utilsbib.extract_countries_from_affiliations(
                    self.df,
                    aff_column=aff_column,
                )
    
        if preprocess_level >= 2:
            self.process_keywords(
                exclude_list=exclude_list_kw,
                synonyms=synonyms_kw,
                lemmatize=lemmatize_kw,
            )
            self.process_text_vars(
                stopwords_file=utilsbib.stopwords_file,
                lang=lang_of_docs,
                extra_stopwords=extra_stopwords,
                exclude_specific_stopwords=specific_stopword_categories,
            )
            self.get_country_collaboration()
            self.df = utilsbib.build_combined_text(
                self.df,
                include_index_keywords=combine_with_index_keywords,
            )
    
            if self.concept_df is not None:
                if concept_column is None:
                    concept_column = utilsbib.first_existing(
                        self.df,
                        [
                            "Processed Combined Text",
                            "Combined Text",
                            "Processed Abstract",
                            "Processed Title",
                            "Abstract",
                            "Title",
                            "Processed Author Keywords",
                            "Author Keywords",
                        ],
                    )
    
                self.df = utilsbib.add_concept_indicators(
                    self.df,
                    self.concept_df,
                    text_col=concept_column,
                )
    
        if preprocess_level >= 3:
            if self.db == "scopus":
                cited_sciences = preprocess_level == 4
                self.add_sciences_scopus(
                    asjc_map_df=asjc_map_df,
                    cited_sciences=cited_sciences,
                )
            self.df["Author and Index Keywords"] = utilsbib.merge_keywords_columns(
                self.df,
                author_col="Author Keywords",
                index_col="Index Keywords",
                sep=self.default_separator,
            )
            # Find the actual processed column names (may vary by database)
            proc_title = utilsbib.first_existing(
                self.df, 
                ["Processed Title", "Processed TI", "Processed Document Title", "Title"]
            )
            proc_abstract = utilsbib.first_existing(
                self.df,
                ["Processed Abstract", "Processed Abstract / BHTD Critical Abstract", "Abstract", "Abstract / BHTD Critical Abstract"]
            )
            proc_author_kw = utilsbib.first_existing(
                self.df,
                ["Processed Author Keywords", "Author Keywords"]
            )
            proc_index_kw = utilsbib.first_existing(
                self.df,
                ["Processed Index Keywords", "Index Keywords"]
            )
            # Only merge if we have at least title or abstract
            if proc_title or proc_abstract:
                self.df = utilsbib.merge_text_columns(
                    self.df,
                    title_col=proc_title or "Title",
                    abstract_col=proc_abstract or "Abstract",
                    author_col=proc_author_kw or "Author Keywords",
                    index_col=proc_index_kw or "Index Keywords",
                )
        if preprocess_level >= 4:
            if hasattr(self, "cited_sciences_df"):
                self.compute_interdisciplinarity_entropy()
    
        self.describe_columns()
    
        if default_keywords == "author":
            self.kw_var = (
                "Processed Author Keywords"
                if "Processed Author Keywords" in self.df.columns
                else "Author Keywords"
            )
        elif default_keywords == "index":
            self.kw_var = (
                "Processed Index Keywords"
                if "Processed Index Keywords" in self.df.columns
                else "Index Keywords"
            )
        elif default_keywords in ["both", "author and index"]:
            if "Author and Index Keywords" not in self.df.columns:
                self.df["Author and Index Keywords"] = utilsbib.merge_keywords_columns(
                    self.df,
                    author_col="Author Keywords",
                    index_col="Index Keywords",
                    sep=self.default_separator,
                )
            self.kw_var = (
                "Processed Author and Index Keywords"
                if "Processed Author and Index Keywords" in self.df.columns
                else "Author and Index Keywords"
            )
        else:
            self.kw_var = None
    
        self.dpi = dpi
        self.cmap, self.default_color, self.cmap_disc = cmap, default_color, cmap_disc
    
        if ("Author full names" in self.df.columns) and self.db == "scopus":
            (
                self.id_to_author,
                self.author_to_id,
            ) = utilsbib.extract_author_mappings(self.df, "Author full names")
    
        if self.res_folder is not None:
            if hasattr(self, "missings_df"):
                self._save_table(self.missings_df, "missing values")
        
        # Initialize unified plotting interface
        from biblium.plotting.interface import PlotInterface
        self.plot = PlotInterface(self)
    
    def create_dashboard(
        self,
        output_path: str = "dashboard.html",
        title: str = None,
        theme: str = "light",
        **kwargs,
    ):
        """
        Create an interactive HTML dashboard.
        
        Parameters
        ----------
        output_path : str
            Output file path for the HTML dashboard.
        title : str, optional
            Dashboard title. Defaults to "Bibliometric Dashboard".
        theme : str
            Theme: "light" or "dark". Default is "light".
        **kwargs
            Additional arguments passed to DashboardConfig.
        
        Returns
        -------
        Path
            Path to the generated dashboard file.
        
        Examples
        --------
        >>> ba = BiblioAnalysis("data.csv", db="scopus")
        >>> ba.create_dashboard("my_analysis.html")
        >>> ba.create_dashboard("dark.html", theme="dark", top_n=20)
        """
        from biblium.dashboard import Dashboard, DashboardConfig
        
        config = DashboardConfig(
            title=title or "Bibliometric Dashboard",
            theme=theme,
            **kwargs,
        )
        dashboard = Dashboard(self, config=config)
        return dashboard.create(output_path)

    # =========================================================================
    # HELPER METHODS (shared across BiblioAnalysis and BiblioGroupAnalysis)
    # =========================================================================
    
    def _save_table(self, df, name, subfolder="tables"):
        """
        Save DataFrame to Excel if res_folder is set.
        
        Parameters
        ----------
        df : DataFrame
            Data to save.
        name : str
            Filename (without extension).
        subfolder : str
            Subfolder within res_folder.
        """
        if self.res_folder is not None and df is not None:
            utilsbib.to_excel_fancy(
                df,
                f_name=os.path.join(self.res_folder, subfolder, f"{name}.xlsx"),
                autofit=getattr(self, "autofit", False),
                conditional_formatting=getattr(self, "cond_formatting", False),
            )
    
    def _save_plot(self, filename_base, subfolder="plots"):
        """
        Save current matplotlib figure if res_folder is set.
        
        Parameters
        ----------
        filename_base : str
            Filename without extension.
        subfolder : str
            Subfolder within res_folder.
        """
        if self.res_folder is not None:
            path = os.path.join(self.res_folder, subfolder, filename_base)
            utilsbib.save_plot(path, dpi=self.dpi)
    
    def _get_column(self, candidates, required=True):
        """
        Find the first available column from candidates.
        
        Parameters
        ----------
        candidates : str or list
            Single column name or list of candidates in priority order.
        required : bool
            If True, raise error when no column found.
            
        Returns
        -------
        str or None
            Found column name.
        """
        if isinstance(candidates, str):
            candidates = [candidates]
        
        for col in candidates:
            if col in self.df.columns:
                return col
        
        if required:
            raise ValueError(f"None of the columns found: {candidates}")
        return None

    def filter_dataframe(
        self,
        *,
        filters: Optional[Dict[str, Dict[str, Any]]] = None,
        bradford_filter: Literal["all", "core", "core+zone2"] = "all",
        source_col: str = "Source title",
    ) -> None:
        """
        Filter `self.df` by column-wise rules and (optionally) Bradford zones, updating in place.

        Supported filter keys per column:
        - "regex_include": list of regex patterns (OR-matched)
        - "regex_exclude": list of regex patterns (OR-matched)
        - "include": list of exact values to include
        - "exclude": list of exact values to exclude
        - "min": minimum value (numeric or date-like)
        - "max": maximum value (numeric or date-like)

        Parameters
        ----------
        filters : dict | None, default=None
            Column-wise filtering rules. If None, no column filters are applied.
        bradford_filter : {"all", "core", "core+zone2"}, default="all"
            If not "all", keep only sources in Bradford core (or core+zone2) computed
            by splitting unique sources into thirds by frequency rank within the
            already column-filtered subset.
        source_col : str, default="Source title"
            Column used for Bradford filtering.

        Effects
        -------
        - Sets `self.df` to the filtered DataFrame (index reset).
        - Sets `self.df_removed` to the rows removed by filtering (index reset).
        - Returns nothing.

        Examples
        --------
        One-liner core-only Articles after 2000 with ≥1 citation:
        >>> ba.filter_dataframe(filters={"Document Type": {"include": ["Article"]}, "Year": {"min": 2000}, "Cited by": {"min": 1}}, bradford_filter="core")

        One-liner with regex include/exclude:
        >>> ba.filter_dataframe(filters={"Language of Original Document": {"regex_include": ["English|Slovenian"]}, "Document Type": {"regex_exclude": ["Editorial|Note"]}})
        """
        print(f"Initial sample size: {self.n}")
        if not hasattr(self, "df"):
            raise ValueError("self.df is not available.")

        df = self.df
        filters = filters or {}

        # Start with all rows selected
        base_mask = pd.Series(True, index=df.index)

        for col, criteria in filters.items():
            if col not in df.columns or not isinstance(criteria, dict):
                continue

            s = df[col]
            s_str = s.astype(str)

            # Regex include (OR)
            regex_inc = criteria.get("regex_include")
            if regex_inc:
                pattern = "|".join(f"(?:{p})" for p in regex_inc)
                base_mask &= s_str.str.contains(pattern, na=False, regex=True)

            # Regex exclude (OR)
            regex_exc = criteria.get("regex_exclude")
            if regex_exc:
                pattern = "|".join(f"(?:{p})" for p in regex_exc)
                base_mask &= ~s_str.str.contains(pattern, na=False, regex=True)

            # Exact include / exclude
            inc = criteria.get("include")
            if inc is not None:
                inc_mask = s.isin(inc)
                if not inc_mask.any():  # fallback if dtype mismatch
                    inc_mask = s_str.isin([str(x) for x in inc])
                base_mask &= inc_mask

            exc = criteria.get("exclude")
            if exc is not None:
                exc_mask = s.isin(exc)
                if not exc_mask.any():
                    exc_mask = s_str.isin([str(x) for x in exc])
                base_mask &= ~exc_mask

            # Min / Max (numeric first, then datetime)
            needs_min = "min" in criteria
            needs_max = "max" in criteria
            if needs_min or needs_max:
                num = pd.to_numeric(s, errors="coerce")
                if num.notna().any():
                    if needs_min:
                        base_mask &= num >= criteria["min"]
                    if needs_max:
                        base_mask &= num <= criteria["max"]
                else:
                    dt = pd.to_datetime(s, errors="coerce", utc=True)
                    if dt.notna().any():
                        if needs_min:
                            min_dt = pd.to_datetime(criteria["min"], utc=True)
                            base_mask &= dt >= min_dt
                        if needs_max:
                            max_dt = pd.to_datetime(criteria["max"], utc=True)
                            base_mask &= dt <= max_dt
                    # else: neither numeric nor datetime-like -> skip min/max

        final_mask = base_mask.copy()

        # Bradford filtering (computed on the column-filtered subset)
        if bradford_filter in {"core", "core+zone2"}:
            if source_col not in df.columns:
                raise ValueError(f'"{source_col}" column is required for Bradford filtering.')

            subset = df.loc[base_mask]
            source_counts = subset[source_col].value_counts(dropna=True)
            total_sources = len(source_counts)

            if total_sources > 0:
                third = math.ceil(total_sources / 3)
                core_sources = set(source_counts.index[:third])
                zone2_sources = set(source_counts.index[third : 2 * third])

                allowed = core_sources if bradford_filter == "core" else core_sources | zone2_sources
                final_mask = base_mask & df[source_col].isin(allowed)
            else:
                # No valid sources in the subset -> nothing passes Bradford
                final_mask = base_mask & pd.Series(False, index=df.index)

        # Split kept vs removed, update in place
        self.df_removed = df.loc[~final_mask].copy().reset_index(drop=True)
        self.df = df.loc[final_mask].copy().reset_index(drop=True)
        self.n = len(self.df)
        print(f"Sample size after filtering: {self.n}")

    def get_country_collaboration(
        self,
        normalization='jaccard',
        links_df=True,
        min_weight=1,
        region=None,
    ):
        """
        Build country collaboration matrices (raw and optionally normalized) and, if requested,
        a links DataFrame; optionally subset all outputs to a region.

        Parameters
        ----------
        normalization : str or None, default "jaccard"
            Method passed to `utilsbib.normalize_symmetric_matrix`. If None, skip normalization.
        links_df : bool, default True
            If True, build `self.countries_links_df` from the (possibly region-filtered) raw matrix.
        min_weight : int or float, default 1
            Minimum edge weight when constructing links from the matrix.
        region : str or None, default None
            If None, no regional filtering. If "EU" (case-insensitive), filter using
            `utilsbib.df_countries["EU"]`. Otherwise, treat as a continent name and filter
            via `utilsbib.df_countries["Continent"]`.

        Notes
        -----
        - `self.country_collab_matrix` and `self.country_collab_matrix_norm` are square matrices
          whose rows and columns are countries. When `region` is provided, both are
          subset to the allowed country set (intersection with their current labels).
        - `self.countries_links_df` (if built) is constructed AFTER any regional filtering.
        """
        # ---- 1) Extract countries and build matrices -----------------------------------------
        if self.db == "scopus":
            self.df, self.country_collab_matrix = utilsbib.extract_countries_from_affiliations(
                self.df, aff_column="Affiliations")
        elif self.db == "oa":
            country_col = [c for c in ["Countries of Authors", "authorships.countries"] if c in self.df.columns][0]
            self.df = utilsbib.openalex_map_country_codes(self.df,
                                                          country_col=country_col,code_dict=utilsbib.code_dct_r,
                                                          sep=self.default_separator)
            self.country_collab_matrix = utilsbib.compute_openalex_country_collaboration_matrix(self.df,
                                                                   column="Countries of Authors",
                                                                   sep=self.default_separator)
        elif self.db == "wos":
            # For WoS - extract countries from Correspondence Address or Reprint Address
            addr_col = utilsbib.first_existing(
                self.df,
                ["Correspondence Address", "Reprint Address", "C1"]
            )
            if addr_col:
                self.df, self.country_collab_matrix = utilsbib.extract_countries_from_wos_addresses(
                    self.df, addr_column=addr_col
                )
            else:
                self.country_collab_matrix = None
        else:
            # For PubMed and other databases - try to find existing country column
            country_col = utilsbib.first_existing(
                self.df,
                ["Country", "Countries", "Countries of Authors", "Affiliations Country"]
            )
            if country_col:
                # Build collaboration matrix from country column
                self.country_collab_matrix = utilsbib.compute_openalex_country_collaboration_matrix(
                    self.df,
                    column=country_col,
                    sep=self.default_separator
                )
            else:
                # No country data available - initialize empty matrix
                self.country_collab_matrix = None

        self.country_collab_matrix_norm = None
        if normalization is not None and self.country_collab_matrix is not None:
            self.country_collab_matrix_norm = utilsbib.normalize_symmetric_matrix(
                self.country_collab_matrix, method=normalization
            )

        # ---- 2) Optional regional filtering ---------------------------------------------------
        if region is not None:
            dfc = utilsbib.df_countries.copy()

            # Determine mask
            region_str = str(region).strip()
            if region_str.lower() == "eu":
                eu = dfc["EU"]
                truthy = {"1", "true", "yes", "y", "t"}
                eu_bool = eu.apply(
                    lambda v: bool(v) if isinstance(v, (bool, int))
                    else (str(v).strip().lower() in truthy) if v is not None
                    else False
                )
                mask = eu_bool
            else:
                mask = dfc["Continent"].fillna("").str.strip().str.casefold() == region_str.casefold()

            # Build allowed label set: accept common country labels and ISO-3 codes
            allowed = set()
            for col_name in ("Name", "Official name", "ISO-3"):
                if col_name in dfc.columns:
                    allowed.update(dfc.loc[mask, col_name].dropna().astype(str).str.strip())

            # Include both name->ISO3 and ISO3->name variants when possible
            iso_map = utilsbib.country_iso3_dct or {}
            inv_iso_map = {v: k for k, v in iso_map.items()} if isinstance(iso_map, dict) else {}
            expanded_allowed = set(allowed)
            expanded_allowed.update(iso_map.get(a, None) for a in allowed if a in iso_map)
            expanded_allowed.update(inv_iso_map.get(a, None) for a in allowed if a in inv_iso_map)
            allowed_labels = {str(x).strip() for x in expanded_allowed if x is not None}

            # Helper to subset a square DataFrame by labels present in index/columns
            def _subset_square(
                df_sq,
            ):
                """
                Subset a square DataFrame to keep only rows/columns with allowed labels.
                
                Parameters
                ----------
                df_sq : pd.DataFrame
                    Square DataFrame (e.g., collaboration matrix).
                    
                Returns
                -------
                pd.DataFrame
                    Filtered DataFrame with only allowed labels.
                """
                if df_sq is None or getattr(df_sq, "empty", True):
                    return df_sq
                labels = [str(x).strip() for x in df_sq.index]
                keep = [lbl for lbl in labels if lbl in allowed_labels]
                return df_sq.loc[keep, keep]

            self.country_collab_matrix = _subset_square(self.country_collab_matrix)
            if self.country_collab_matrix_norm is not None:
                self.country_collab_matrix_norm = _subset_square(self.country_collab_matrix_norm)

        # ---- 3) Build links from the (possibly filtered) RAW matrix ---------------------------
        if links_df and self.country_collab_matrix is not None:
            self.countries_links_df = utilsbib.build_links_from_matrix(
                self.country_collab_matrix, min_weight=min_weight
            )
            # Map source/target to ISO-3 for consistency
            if hasattr(self, "countries_links_df") and not self.countries_links_df.empty:
                self.countries_links_df["source"] = self.countries_links_df["source"].map(utilsbib.country_iso3_dct)
                self.countries_links_df["target"] = self.countries_links_df["target"].map(utilsbib.country_iso3_dct)
        elif links_df:
            # No country data - initialize empty DataFrame
            self.countries_links_df = pd.DataFrame(columns=["source", "target", "weight"])

    def compute_interdisciplinarity_entropy(
        self,
        counting_types=['Number of documents', 'Proportion of documents', 'Franctional number of documents'],
        concat=True,
    ):
        """
        Compute interdisciplinarity entropy indices from cited subject areas.
        
        This wrapper expects `self.cited_sciences_df` to contain per-document
        counts of subject categories (for example ASJC fields/areas/sciences).
        It calls `utilsbib.compute_interdisciplinarity_entropy` and stores the
        result in `self.entropies_df`.
        
        Parameters
        ----------
        counting_types : list of str, optional
            Which counting schemes to include (e.g. absolute, proportional,
            or fractional document counts).
        concat : bool, default True
            If True, the entropy columns are concatenated to `self.df` in place.
        """
        self.entropies_df = utilsbib.compute_interdisciplinarity_entropy(self.cited_sciences_df, counting_types=counting_types)
        if concat:
            self.df = pd.concat([self.df, self.entropies_df.reset_index(drop=True)], axis=1)

    def describe_columns(
        self,
        show: bool = False,
    ) -> None:
        """
        Opens the Excel file of variable names and descriptions,
        prints out each column in `df` alongside its description,
        and returns a function to look up individual variable descriptions.

        Parameters:
        - mapping_file: Path to the Excel file containing 'Name' and 'Description' columns.
        - df: DataFrame whose columns you want to describe.

        Returns:
        - get_description: function that takes a variable name and returns its description.
        """
        # Load the mapping of names → descriptions
        fd = os.path.dirname(__file__)
        mapping_df = pd.read_excel(os.path.join(fd, "additional files", "variable names.xlsx"), sheet_name="descriptions", usecols=["Name", "Description"])
        mapping_dict = dict(zip(mapping_df["Name"], mapping_df["Description"]))

        # Display each column with its description
        for col in self.df.columns:
            desc = mapping_dict.get(col, "No description available")
            if show:
                print(f"{col}: {desc}")

        # Return a helper function for individual lookups
        def get_description(
            var_name: str,
        ) -> str:
            """
            Get description for a variable name.
            
            Parameters
            ----------
            var_name : str
                Name of the variable/column.
                
            Returns
            -------
            str
                Description of the variable, or "No description available".
            """
            return mapping_dict.get(var_name, "No description available")

        self.column_descriptor = get_description

    def add_sciences_scopus(
        self,
        asjc_map_df: Optional[pd.DataFrame] = None,
        cited_sciences: bool = True,
    ) -> None:
        """
        Enrich Scopus data with Scopus ASJC subject areas and sciences.
        
        For Scopus datasets (`self.db == "scopus"`) this method merges in ASJC
        metadata, adds field/area/science columns to `self.df` and optionally
        derives a per-document table of cited sciences.
        
        Parameters
        ----------
        asjc_map_df : pandas.DataFrame, optional
            Preloaded mapping between sources and ASJC codes. When None, the
            mapping is read from package Excel files.
        cited_sciences : bool, default True
            If True, builds `self.cited_sciences_df` using
            `utilsbib.extract_cited_sciences`.
        """
        if self.db != "scopus":
            return None

        fd = os.path.dirname(__file__)
        if asjc_map_df is None:
            try:
                asjc_map_df = pd.read_excel(os.path.join(fd, "additional files", "sources_data_short.xlsx"))
            except:
                try:
                    asjc_map_df = pd.read_excel(os.path.join(fd, "additional files", "sources_data.xlsx"), sheet_name="Scopus Sources Oct. 2024")
                    asjc_map_df = asjc_map_df[["Source Title", "All Science Journal Classification Codes (ASJC)"]]
                except:
                    pass
        asjc_meta_df = pd.read_excel(os.path.join(fd, "additional files", "scopus subject area codes.xlsx"))
        self.df = utilsbib.enrich_bibliometric_data(self.df, asjc_map_df, asjc_meta_df)
        if cited_sciences:
            self.cited_sciences_df = utilsbib.extract_cited_sciences(self.df, asjc_map_df, asjc_meta_df)

    def process_keywords(
        self,
        exclude_list: Optional[List[str]] = None,
        synonyms: Optional[Union[pd.DataFrame, Dict[str, List[str]]]] = None,
        lemmatize: bool = False,
        *,
        normalize_compounds: Optional[Literal["space", "hyphen"]] = "space",
        fold_accents: bool = False,
        alt_separators: Optional[List[str]] = None,
        sort_keywords: bool = True,
        strip_punctuation: bool = True,
        normalize_underscores: bool = True,
    ) -> pd.DataFrame:
        """
        Process only the standard keyword columns:
        "Author Keywords", "Index Keywords", and (if present) "Author and Index Keywords".
        Missing columns are skipped. For each found column, writes "Processed {column}".

        Returns:
            pd.DataFrame: Updated self.df.
        """
        for col in ("Author Keywords", "Index Keywords", "Author and Index Keywords"):
            if col in self.df.columns:
                self.df = utilsbib.preprocess_keywords(
                    self.df,
                    col,
                    exclude_list=exclude_list,
                    synonyms=synonyms,
                    lemmatize=lemmatize,
                    sep=self.default_separator,
                    normalize_compounds=normalize_compounds,
                    fold_accents=fold_accents,
                    alt_separators=alt_separators,
                    sort_keywords=sort_keywords,
                    strip_punctuation=strip_punctuation,
                    normalize_underscores=normalize_underscores,
                )
        return self.df

    def process_text_vars(
        self,
        stopwords_file: Optional[str] = None,
        lang: str = "en",
        remove_numbers: bool = True,
        remove_two_letter_words: bool = True,
        extra_stopwords: Optional[Iterable[str]] = None,
        exclude_specific_stopwords: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """
        Clean and normalize core text variables such as abstracts and titles.
    
        Applies `utilsbib.process_text_column` to the abstract and title columns,
        performing tokenization, lowercasing, stop-word removal and optional
        number/two-letter-word filtering. Processed versions are stored in
        `self.df` (for example "Processed Abstract" and "Processed Title").
    
        Parameters
        ----------
        stopwords_file : str or path-like, optional
            Location of the stop-word list to use. May contain a "general" and
            optionally a "specific" sheet (see `utilsbib.process_text_column`).
        lang : str, default "en"
            Language code passed to the tokenizer / stop-word logic.
        remove_numbers : bool, default True
            If True, numeric tokens are removed.
        remove_two_letter_words : bool, default True
            If True, tokens of length two are removed.
        extra_stopwords : iterable of str or None, optional
            Additional stop-words to exclude (e.g. database-specific boilerplate).
            These are always used and are not affected by `exclude_specific_stopwords`.
        exclude_specific_stopwords : iterable of str or None, optional
            Names of categories from the "specific" sheet in `stopwords_file` whose
            words should be treated as stopwords. Category matching is
            case-insensitive after stripping. If None or empty, the "specific"
            sheet is ignored.
    
            Example:
                exclude_specific_stopwords = ["Publication document type terms"]
        """
        # Start from user-provided extra stopwords, but avoid mutating the input
        extra_sw = list(extra_stopwords) if extra_stopwords is not None else []
    
        # Add database-specific stopwords (kept separate from Excel "specific")
        if getattr(self, "db", None) == "scopus":
            extra_sw += ["elsevier", "reserved"]
    
        # Find Abstract column (different databases use different names)
        abstract_col = utilsbib.first_existing(
            self.df, 
            ["Abstract", "Abstract / BHTD Critical Abstract", "AB", "abstract"]
        )
        if abstract_col:
            self.df = utilsbib.process_text_column(
                self.df,
                abstract_col,
                stopwords_file=stopwords_file,
                lang=lang,
                remove_numbers=remove_numbers,
                remove_two_letter_words=remove_two_letter_words,
                extra_stopwords=extra_sw,
                exclude_specific_stopwords=exclude_specific_stopwords,
            )
    
        # Find Title column (different databases use different names)
        title_col = utilsbib.first_existing(
            self.df,
            ["Title", "TI", "Document Title", "title"]
        )
        if title_col:
            self.df = utilsbib.process_text_column(
                self.df,
                title_col,
                stopwords_file=stopwords_file,
                lang=lang,
                remove_numbers=remove_numbers,
                remove_two_letter_words=remove_two_letter_words,
                extra_stopwords=extra_sw,
                exclude_specific_stopwords=exclude_specific_stopwords,
            )

    def add_concepts(
        self,
        concepts: dict = None,
        concept_df: pd.DataFrame = None,
        concept_file: str = None,
        text_column: str = None,
        use_regex: bool = False,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Add binary concept indicator columns to the dataset.
        
        Creates new binary (0/1) columns for each concept, where 1 indicates
        that the document's text contains keywords associated with that concept.
        
        Parameters
        ----------
        concepts : dict, optional
            Dictionary mapping concept names to lists of keywords/phrases.
            Example: {"AI": ["artificial intelligence", "machine learning", "deep learn*"],
                      "Climate": ["climate change", "global warming", "carbon*"]}
        concept_df : pd.DataFrame, optional
            DataFrame where each column is a concept and rows contain keywords.
            Alternative to `concepts` dict.
        concept_file : str, optional
            Path to Excel/CSV file with concepts. Each column is a concept name,
            rows contain keywords. Alternative to `concepts` and `concept_df`.
        text_column : str, optional
            Column to search for keywords. If None, auto-detects from:
            "Processed Combined Text", "Combined Text", "Abstract", "Title",
            "Author Keywords", etc.
        use_regex : bool, default False
            If True, treat keywords as regular expressions.
            If False, use word-boundary matching with wildcard support (word*).
        verbose : bool, default True
            Print progress messages.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with added concept columns.
        
        Notes
        -----
        Keywords support:
        - Wildcards: "govern*" matches "government", "governance", "governing"
        - Phrases: "climate change" matches the exact phrase
        - Combined: "good govern*" matches "good government", "good governance"
        
        Examples
        --------
        >>> bib.add_concepts(concepts={
        ...     "Sustainability": ["sustainab*", "green", "eco-friendly"],
        ...     "Innovation": ["innovat*", "novel", "breakthrough"]
        ... })
        
        >>> bib.add_concepts(concept_file="my_concepts.xlsx", text_column="Abstract")
        """
        import re
        
        # Load concepts from file if provided
        if concept_file is not None:
            if verbose:
                print(f"Loading concepts from: {concept_file}")
            if concept_file.endswith('.xlsx') or concept_file.endswith('.xls'):
                concept_df = pd.read_excel(concept_file)
            else:
                concept_df = pd.read_csv(concept_file)
        
        # Convert dict to DataFrame
        if concepts is not None:
            # Find max length
            max_len = max(len(v) for v in concepts.values())
            # Pad shorter lists with None
            padded = {k: v + [None] * (max_len - len(v)) for k, v in concepts.items()}
            concept_df = pd.DataFrame(padded)
        
        if concept_df is None:
            raise ValueError("Must provide concepts, concept_df, or concept_file")
        
        # Auto-detect text column
        if text_column is None:
            text_column = utilsbib.first_existing(
                self.df,
                [
                    "Processed Combined Text",
                    "Combined Text",
                    "Processed Abstract",
                    "Abstract",
                    "Processed Title",
                    "Title",
                    "Processed Author Keywords",
                    "Author Keywords",
                    "Index Keywords",
                ],
            )
            if text_column is None:
                raise ValueError("No suitable text column found. Please specify text_column.")
        
        if verbose:
            print(f"Adding concept indicators...")
            print(f"  Text column: {text_column}")
            print(f"  Concepts: {list(concept_df.columns)}")
        
        if use_regex:
            # Use regex mode - treat each keyword as a regex pattern
            self.df[text_column] = self.df[text_column].fillna("").astype(str)
            
            for concept in concept_df.columns:
                patterns = (
                    concept_df[concept]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .tolist()
                )
                patterns = [p for p in patterns if p]
                
                if not patterns:
                    continue
                
                # Combine patterns with OR
                combined = "(?:" + "|".join(patterns) + ")"
                try:
                    regex = re.compile(combined, flags=re.IGNORECASE)
                    self.df[concept] = self.df[text_column].str.contains(regex, na=False).astype(int)
                except re.error as e:
                    if verbose:
                        print(f"  Warning: Invalid regex for {concept}: {e}")
                    continue
        else:
            # Use utilsbib's add_concept_indicators (supports wildcards)
            self.df = utilsbib.add_concept_indicators(
                self.df, concept_df, text_column
            )
        
        # Store concepts for reference
        self.concept_df = concept_df
        self.concept_columns = list(concept_df.columns)
        
        if verbose:
            # Show summary
            for concept in concept_df.columns:
                if concept in self.df.columns:
                    count = self.df[concept].sum()
                    pct = 100 * count / len(self.df)
                    print(f"  {concept}: {count:,} documents ({pct:.1f}%)")
        
        return self.df


    def get_production(
        self,
        relative_counts: bool = True,
        cumulative: bool = True,
        predict_last_year: bool = False,
    ) -> pd.DataFrame:
        """
        Compute annual scientific production for the current dataset.
        
        This is a thin wrapper around `utilsbib.get_scientific_production` that
        aggregates the number of documents (and optionally citations) per year and
        stores the result in `self.production_df`. When an output folder is
        configured the table is also exported to Excel.
        
        Parameters
        ----------
        relative_counts : bool, default True
            If True, adds relative and percentage measures.
        cumulative : bool, default True
            If True, adds cumulative counts over years.
        predict_last_year : bool, default False
            If True, attempts to predict production for the last year based on
            previous trends.
        """
        self.production_df = utilsbib.get_scientific_production(self.df, relative_counts=relative_counts, cumulative=cumulative, predict_last_year=predict_last_year)
        self._save_table(self.production_df, "scientific production")
        return self.production_df

    def get_relative_representation(
        self,
        category: str = "Year",
        reference_df: Optional[pd.DataFrame] = None,
        fetch_reference: bool = True,
        threshold: float = 1.0,
        plot: bool = True,
        plot_comparison: bool = False,
        filename: Optional[str] = None,
        cmap: str = "RdBu",
    ) -> pd.DataFrame:
        """
        Analyze relative representation of your dataset compared to a reference distribution.
        
        Shows over- and under-representation in percentage points for any categorical
        variable compared to global/reference data from OpenAlex.
        
        Parameters
        ----------
        category : str, default "Year"
            Category to analyze. Supported for auto-fetch:
            - "Year" : Publication year
            - "Country" : Author countries  
            - "SDG" : Sustainable Development Goals
            - "Document Type" : Article, review, etc.
            - "Open Access" : OA status
            For other categories, provide reference_df manually.
        reference_df : pd.DataFrame, optional
            Custom reference distribution with columns [category, "Count"].
            If None and fetch_reference=True, fetches from OpenAlex.
        fetch_reference : bool, default True
            Whether to fetch reference data from OpenAlex.
        threshold : float, default 1.0
            Threshold for classifying over/under-representation (pp).
        plot : bool, default True
            Whether to generate difference plot.
        plot_comparison : bool, default False
            Whether to also generate distribution comparison plot.
        filename : str, optional
            Base filename for saving (without extension).
        cmap : str, default "RdBu"
            Colormap for difference plot (red=under, blue=over).
        
        Returns
        -------
        pd.DataFrame
            Relative representation analysis.
        
        Examples
        --------
        # Year vs global (auto-fetch)
        >>> ba.get_relative_representation("Year")
        
        # SDGs vs global (auto-fetch)
        >>> ba.get_relative_representation("SDG")
        
        # Countries vs global
        >>> ba.get_relative_representation("Country")
        
        # Custom reference
        >>> ref_df = pd.DataFrame({"Year": [2020, 2021], "Count": [1000, 1100]})
        >>> ba.get_relative_representation("Year", reference_df=ref_df)
        
        # List supported reference types
        >>> from biblium.representation import list_supported_references
        >>> list_supported_references()
        """
        try:
            from biblium.representation import (
                compute_relative_representation,
                plot_relative_representation,
                plot_distribution_comparison,
                SUPPORTED_REFERENCES,
                fetch_openalex_yearly_counts,
                fetch_openalex_country_counts,
                fetch_openalex_sdg_counts,
            )
        except ImportError:
            from representation import (
                compute_relative_representation,
                plot_relative_representation,
                plot_distribution_comparison,
                SUPPORTED_REFERENCES,
                fetch_openalex_yearly_counts,
                fetch_openalex_country_counts,
                fetch_openalex_sdg_counts,
            )
        
        # Map category aliases
        category_map = {
            "year": "Year",
            "country": "Country",
            "countries": "Country",
            "sdg": "SDG",
            "sdgs": "SDG",
            "document type": "Document Type",
            "doctype": "Document Type",
            "open access": "Open Access",
            "oa": "Open Access",
        }
        category_normalized = category_map.get(category.lower(), category)
        
        # Determine the actual column in df
        col_candidates = {
            "Year": ["Year"],
            "Country": ["Countries of Authors", "Country", "Countries"],
            "SDG": ["SDG", "SDGs", "Sustainable Development Goals"],
            "Document Type": ["Document Type", "Type"],
            "Open Access": ["Open Access", "OA"],
        }
        
        actual_col = None
        for candidate in col_candidates.get(category_normalized, [category]):
            if candidate in self.df.columns:
                actual_col = candidate
                break
        
        # Special handling for SDG - multiple binary columns
        if category_normalized == "SDG":
            sdg_cols = [c for c in self.df.columns if c.startswith("SDG") and len(c) <= 5 and c[3:].isdigit()]
            if not sdg_cols:
                raise ValueError("No SDG columns found (SDG01-SDG17). Run ba.identify_sdgs() first.")
            
            # Sum each SDG column to get counts
            sdg_counts = {}
            for col in sorted(sdg_cols):
                sdg_num = int(col[3:])
                sdg_label = f"SDG {sdg_num}"
                sdg_counts[sdg_label] = self.df[col].sum()
            
            observed = pd.DataFrame(list(sdg_counts.items()), columns=["SDG", "Count"])
            actual_col = "SDG"  # For consistency
        elif actual_col is None:
            raise ValueError(f"Category '{category}' not found. Available: {self.df.columns.tolist()}")
        # Handle list-type columns (like Countries)
        elif self.df[actual_col].dtype == object and self.df[actual_col].str.contains(self.default_separator, na=False).any():
            # Split and explode
            exploded = self.df[actual_col].str.split(self.default_separator).explode().str.strip()
            exploded = exploded[exploded != ""]
            observed = exploded.value_counts().reset_index()
            observed.columns = [category_normalized, "Count"]
        else:
            observed = self.df[actual_col].value_counts().reset_index()
            observed.columns = [category_normalized, "Count"]
        
        # Get reference distribution
        if reference_df is None and fetch_reference:
            if category_normalized in SUPPORTED_REFERENCES:
                info = SUPPORTED_REFERENCES[category_normalized]
                print(f"Fetching global {category_normalized} data from OpenAlex...")
                
                # Get year range for filtering
                year_range = None
                if "Year" in self.df.columns:
                    year_range = (int(self.df["Year"].min()), int(self.df["Year"].max()))
                
                if category_normalized == "Year":
                    reference_df = fetch_openalex_yearly_counts(year_range[0], year_range[1])
                else:
                    reference_df = info["fetcher"](year_range=year_range)
                
                print(f"Fetched {len(reference_df)} categories from OpenAlex.")
            else:
                raise ValueError(
                    f"Auto-fetch not supported for '{category}'. "
                    f"Supported: {list(SUPPORTED_REFERENCES.keys())}. "
                    "Provide reference_df manually."
                )
        
        if reference_df is None:
            raise ValueError("No reference data and fetch_reference=False")
        
        # Standardize reference column name
        ref_cat_col = [c for c in reference_df.columns if c not in ["Count", "Percentage", "_key"]][0]
        if ref_cat_col != category_normalized:
            reference_df = reference_df.rename(columns={ref_cat_col: category_normalized})
        
        # Compute relative representation
        self.relative_representation_df = compute_relative_representation(
            observed, reference_df,
            category_col=category_normalized,
            count_col="Count",
            ref_count_col="Count",
            threshold=threshold,
        )
        
        # Summary
        over = (self.relative_representation_df["Difference (pp)"] > threshold).sum()
        under = (self.relative_representation_df["Difference (pp)"] < -threshold).sum()
        print(f"\nRelative Representation Summary ({category_normalized}):")
        print(f"  Over-represented (>{threshold} pp): {over} categories")
        print(f"  Under-represented (<-{threshold} pp): {under} categories")
        
        # Determine filename
        if filename is None and self.res_folder is not None:
            filename = os.path.join(self.res_folder, "plots", f"relative representation {category_normalized}")
        
        # Plot difference
        if plot:
            rotation = 45 if len(self.relative_representation_df) > 10 else 0
            plot_relative_representation(
                self.relative_representation_df,
                category_col=category_normalized,
                title=f"Relative Representation: {category_normalized}",
                cmap=cmap,
                rotation=rotation,
                filename=filename,
                dpi=self.dpi,
            )
        
        # Plot comparison
        if plot_comparison:
            comp_filename = filename + "_comparison" if filename else None
            rotation = 45 if len(self.relative_representation_df) > 10 else 0
            plot_distribution_comparison(
                self.relative_representation_df,
                category_col=category_normalized,
                title=f"Distribution Comparison: {category_normalized}",
                rotation=rotation,
                filename=comp_filename,
                dpi=self.dpi,
            )
        
        # Save to Excel
        self._save_table(self.relative_representation_df, f"relative representation {category_normalized}")
        
        return self.relative_representation_df

    # Descriptives

    def identify_sdgs(
        self,
        text_column: str = None,
        sdg_queries_path: str = None,
        return_perspectives: bool = True,
        return_dimensions: bool = True,
    ):
        """
        Identify Sustainable Development Goals (SDGs) in documents.
        
        Uses Scopus SDG queries to match text content against SDG definitions.
        Creates binary columns (SDG01-SDG17) indicating which SDGs each document
        addresses, plus perspective and dimension aggregations.
        
        Parameters
        ----------
        text_column : str, optional
            Column containing text to analyze. If None, tries:
            "Abstract", "Title", "Processed Abstract", "Processed Title"
        sdg_queries_path : str, optional
            Path to SDG queries Excel file. If None, uses built-in queries.
        return_perspectives : bool, default True
            Whether to compute SDG perspective columns (Life, Society, etc.)
        return_dimensions : bool, default True  
            Whether to compute SDG dimension columns.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with SDG columns added.
        
        Notes
        -----
        After calling this method, you can use:
        >>> ba.get_relative_representation("SDG")
        
        to compare your SDG distribution against global research trends.
        
        Examples
        --------
        >>> ba.identify_sdgs()
        >>> print(ba.df[["Title", "SDG01", "SDG02", "SDG03"]].head())
        
        >>> # Then compare against global
        >>> ba.get_relative_representation("SDG")
        """
        try:
            from biblium.sdg_identifier import identify_sdgs as _identify_sdgs
        except ImportError:
            from sdg_identifier import identify_sdgs as _identify_sdgs
        
        # Determine text column
        if text_column is None:
            for candidate in ["Abstract", "Processed Abstract", "Title", "Processed Title"]:
                if candidate in self.df.columns and self.df[candidate].notna().sum() > 0:
                    text_column = candidate
                    break
        
        if text_column is None or text_column not in self.df.columns:
            raise ValueError(
                "No text column found for SDG identification. "
                "Available columns: " + str(self.df.columns.tolist())
            )
        
        print(f"Identifying SDGs using column: {text_column}")
        
        # Run SDG identification
        self.df = _identify_sdgs(
            self.df,
            text_column=text_column,
            sdg_queries_path=sdg_queries_path,
            return_perspectives=return_perspectives,
            return_dimensions=return_dimensions,
        )
        
        # Count SDGs identified
        sdg_cols = [c for c in self.df.columns if c.startswith("SDG") and c[3:].isdigit()]
        total_sdgs = self.df[sdg_cols].sum().sum()
        docs_with_sdg = (self.df[sdg_cols].sum(axis=1) > 0).sum()
        
        print(f"\nSDG Identification Complete:")
        print(f"  Documents with at least one SDG: {docs_with_sdg} ({docs_with_sdg/len(self.df)*100:.1f}%)")
        print(f"  Total SDG assignments: {total_sdgs}")
        
        # Create summary
        sdg_summary = self.df[sdg_cols].sum().sort_values(ascending=False)
        self.sdg_counts_df = sdg_summary.reset_index()
        self.sdg_counts_df.columns = ["SDG", "Count"]
        
        print(f"\nTop SDGs in dataset:")
        print(self.sdg_counts_df.head(5).to_string(index=False))
        
        return self.df

    def get_collaboration_index(
        self,
        author_col='Author(s) ID',
        sep=';',
    ):
        """
        Compute a simple collaboration index based on co-authorship.
        
        Delegates to `utilsbib.collaboration_index` to summarize how many authors
        collaborate on the documents in `self.df` (for example, average authors
        per paper or the share of multi-authored documents). The result is stored
        as `self.collaboration_index`.
        
        Parameters
        ----------
        author_col : str, default "Author(s) ID"
            Column holding author or author-ID information.
        sep : str, default ";"
            Delimiter used to separate multiple authors in `author_col`.
        """
        self.collaboration_index = utilsbib.collaboration_index(self.df, author_col=author_col, sep=sep)

    def get_main_info(
        self,
        include: List[Literal["descriptives", "performance", "time series"]] = None,
        performance_mode: str = "full",
        stopwords: Optional[Iterable[str]] = None,
        excluded_sources_references: Optional[List[str]] = None,
        extra_stats: bool = False,
        exclude_last_year_for_growth: bool = True,
    ) -> None:
        """
        Assemble core summary tables for a bibliometric dataset.
        
        This convenience method can compute descriptive statistics, performance
        indicators and a short time-series summary, storing each result on the
        object. The created tables are also collected in `self.main_info_list`
        for easy reporting.
        
        Parameters
        ----------
        include : list of {"descriptives", "performance", "time series"}, optional
            Which blocks to compute.
        performance_mode : str, default "full"
            Passed to `utilsbib.get_performance_indicators`.
        stopwords : collection or None, optional
            Optional custom stop-words for text descriptives.
        excluded_sources_references : list or None, optional
            Passed through to reference-related descriptives.
        extra_stats : bool, default False
            If True, `utilsbib.describe_dataframe` computes additional statistics.
        
        Notes
        -----
        Typical attributes created are `self.descriptives_df`,
        `self.performances_df` and `self.time_series_stats_df`.
        """
        if include is None:
            include = ["descriptives", "performance", "time series"]
        if not hasattr(self, "main_info_list"):
            self.main_info_list = []
        if "descriptives" in include:
            # Helper to find first matching column or None
            def _find_col(candidates):
                matches = [c for c in candidates if c in self.df.columns]
                return matches[0] if matches else None
            
            ak = _find_col(["Processed Author Keywords", "Author Keywords"])
            ik = _find_col(["Processed Index Keywords", "Index Keywords"])
            abst = _find_col(["Processed Abstract", "Abstract"])
            tit = _find_col(["Processed Title", "Title"])
            
            # Build column list dynamically, skipping None
            desc_cols = [
                ("Year", "numeric"), 
                ("Source title", "string"), 
                ("Document Type", "string"),
                ("Open Access", "string"),  
                ("Cited by", "numeric")
            ]
            if ak:
                desc_cols.append((ak, "list"))
            if ik:
                desc_cols.append((ik, "list"))
            desc_cols.append(("Language of Original Document", "string"))
            if abst:
                desc_cols.append((abst, "text"))
            if tit:
                desc_cols.append((tit, "text"))
            
            self.descriptives_df = utilsbib.compute_descriptive_statistics(
                self.df, desc_cols, stopwords=stopwords, extra_stats=extra_stats, sep=self.default_separator)
            self.main_info_list.append((self.descriptives_df, "descriptives"))
        if "performance" in include:
            metrics = utilsbib.get_performance_indicators(self.df, mode=performance_mode)
            data = [("Performance indicator", name, value) for name, value in metrics]
            self.performances_df = pd.DataFrame(data, columns=["Variable", "Indicator", "Value"])
            self.main_info_list.append((self.performances_df, "performances"))
        if "time series" in include:
            if not hasattr(self, "production_df"):
                self.get_production()
            self.time_series_stats_df = utilsbib.summarize_publication_timeseries(self.production_df, exclude_last_year_for_growth=exclude_last_year_for_growth)
            self.main_info_list.append((self.time_series_stats_df, "time-series analysis"))
        if "references" in include:
            if not hasattr(self, "df_references"):
                self.df_references = utilsbib.parse_references_dataframe(self.df, excluded_sources=excluded_sources_references)
            self._references_stats_df = utilsbib.summarize_parsed_references(self.df_references)
            self.main_info_list.append((self._references_stats_df, "references"))
        if "specific" in include:
            sent = utilsbib.get_specific_indicators(self.df)
            data = [("Sentiment analysis", name, value) for name, value in sent]
            self.specific_stats_df = pd.DataFrame(data, columns=["Variable", "Indicator", "Value"])
            self.main_info_list.append((self.specific_stats_df, "specifics"))
        if self.res_folder is not None:
            f_name = os.path.join(self.res_folder, "tables", "main info.xlsx")
            utilsbib.save_descriptives_to_excel(self.main_info_list, f_name)
            print(f"Saved to {f_name}")

    # Top cited documents

    def get_top_cited_documents(
        self,
        top_n=10,
        cols=None,
        filters=None,
        mode='global',
        title_col='Title',
        ref_col='References',
        cite_col='Cited by',
        additional_cols=None,
    ):
        """
        Compute and store top-cited documents (global, local, or both), using external helper functions.
        """
        if mode in {"global", "both"}:
            self.top_cited_docs_global_df = utilsbib.select_global_top_cited_documents(
                self.df, top_n=top_n, cols=cols, filters=filters, cite_col=cite_col)
            self._save_table(self.top_cited_docs_global_df, "top cited documents global")
        if mode in {"local", "both"}:
            if self.db == "scopus":
                self.top_cited_docs_local_df = utilsbib.select_local_top_cited_documents(
                    self.df, top_n=top_n, cols=cols, filters=filters,
                    title_col=title_col, ref_col=ref_col, cite_col=cite_col)
            elif self.db == "oa":
                self.top_cited_docs_local_df=utilsbib.top_openalex_local_cited_documents(self.df,
                                                                                         top_n=top_n,
                                                                                         sep=self.default_separator,
                                                                                         cols=additional_cols)
            else:
                print("Not yet supported")
                return None

            self._save_table(self.top_cited_docs_local_df, "top cited documents local")
        if mode == "per year":
            self.df["Citations per year"] = self.df["Cited by"] / (self.df["Year"].max() - self.df["Year"] + 1)
            self.top_cited_docs_global_per_year_df = utilsbib.select_global_top_cited_documents(
                self.df, top_n=top_n, cols=cols, filters=filters, cite_col="Citations per year")
            self._save_table(self.top_cited_docs_global_per_year_df, "top cited documents global per year")

    # Counting

    # single occurecenes
    def count_sources(
        self,
        top_n: int = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Count occurrences of source titles in the dataset.
        
        Uses `utilsbib.count_occurrences` on the "Source title" column and
        stores the result as `self.sources_counts_df`. Abbreviated source
        titles are attached when available. Optionally, binary indicator
        columns for the top-N sources are created.
        
        Parameters
        ----------
        top_n : int, default 0
            If > 0, also create binary indicator columns for the top-N
            sources using `utilsbib.match_items_and_compute_binary_indicators`.
        **kwargs :
            Additional options forwarded to the binary-indicator helper.
        """
        return self._count_entity(
            column="Source title",
            item_label="Source",
            count_type="single",
            attr_name="sources_counts_df",
            binary_attr_name="binary_sources_df",
            save_name="sources counts",
            top_n=top_n,
            rename_dict=getattr(self, "sources_abb_dict", None),
            translated_column_name="Abbreviated Source Title",
            **kwargs,
        )

    def count_document_types(
        self,
        top_n: int = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Count occurrences of document types.
        
        Wraps `utilsbib.count_occurrences` on the "Document Type" column and
        stores the result as `self.document_types_counts_df`. Optionally,
        binary indicator columns for the most frequent document types are
        created.
        
        Parameters
        ----------
        top_n : int, default 0
            If > 0, also create binary indicator columns for the top-N
            document types.
        **kwargs :
            Additional options forwarded to the binary-indicator helper.
        """
        return self._count_entity(
            column="Document Type",
            item_label="Document Type",
            count_type="single",
            attr_name="document_types_counts_df",
            binary_attr_name="binary_document_types_df",
            save_name="document types counts",
            top_n=top_n,
            **kwargs,
        )

    def count_ca_countries(
        self,
        top_n: int = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Count corresponding-author countries.
        
        Ensures a "CA Country" column is present (creating it when necessary),
        then calls `utilsbib.count_occurrences` and stores the result as
        `self.ca_country_counts_df`. ISO-3 and continent labels are added and
        optionally binary indicator columns for the top-N countries are built.
        
        Parameters
        ----------
        top_n : int, default 0
            If > 0, also create binary indicator columns for the top-N
            corresponding-author countries.
        **kwargs :
            Additional options forwarded to the binary-indicator helper.
        """
        # Ensure CA Country column exists
        if "CA Country" not in self.df.columns:
            self.df = utilsbib.add_ca_country_df(self.df, self.db)
        
        # Post-process to add ISO-3 and Continent columns
        def add_country_metadata(df):
            df["ISO-3"] = df["Country"].map(utilsbib.country_iso3_dct)
            df["Continent"] = df["Country"].map(utilsbib.continent_dct)
            return df
        
        return self._count_entity(
            column="CA Country",
            item_label="Country",
            count_type="single",
            attr_name="ca_country_counts_df",
            binary_attr_name="binary_ca_countries_df",
            save_name="CA country counts",
            top_n=top_n,
            post_process=add_country_metadata,
            **kwargs,
        )

    def count_author_keywords(
        self,
        top_n: int = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Count occurrences of author keywords.
        
        Uses either "Processed Author Keywords" or "Author Keywords" (depending
        on availability) and stores counts as `self.author_keywords_counts_df`.
        Optionally, binary indicator columns for the most frequent author
        keywords are created.
        
        Parameters
        ----------
        top_n : int, default 0
            If > 0, also create binary indicator columns for the top-N
            author keywords.
        **kwargs :
            Additional options forwarded to the binary-indicator helper.
        """
        return self._count_entity(
            column=["Processed Author Keywords", "Author Keywords"],
            item_label="Keyword",
            count_type="list",
            attr_name="author_keywords_counts_df",
            binary_attr_name="binary_author_keywords_df",
            save_name="author keywords counts",
            top_n=top_n,
            **kwargs,
        )

    def count_index_keywords(
        self,
        top_n: int = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Count occurrences of index keywords.
        
        Uses either "Processed Index Keywords" or "Index Keywords" depending on
        availability and stores counts as `self.index_keywords_counts_df`.
        Optionally, binary indicator columns for the most frequent index
        keywords are created.
        
        Parameters
        ----------
        top_n : int, default 0
            If > 0, also create binary indicator columns for the top-N
            index keywords.
        **kwargs :
            Additional options forwarded to the binary-indicator helper.
        """
        return self._count_entity(
            column=["Processed Index Keywords", "Index Keywords"],
            item_label="Keyword",
            count_type="list",
            attr_name="index_keywords_counts_df",
            binary_attr_name="binary_index_keywords_df",
            save_name="index keywords counts",
            top_n=top_n,
            **kwargs,
        )

    def count_both_keywords(
        self,
        top_n: int = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Count occurrences of combined author and index keywords.
        
        Works on a combined author+index keyword column (processed if present)
        and stores counts as `self.author_and_index_keywords_counts_df`.
        Optionally, binary indicator columns for the most frequent combined
        keywords are created.
        
        Parameters
        ----------
        top_n : int, default 0
            If > 0, also create binary indicator columns for the top-N
            combined keywords.
        **kwargs :
            Additional options forwarded to the binary-indicator helper.
        """
        return self._count_entity(
            column=["Processed Author and Index Keywords", "Author and Index Keywords"],
            item_label="Keyword",
            count_type="list",
            attr_name="author_and_index_keywords_counts_df",
            binary_attr_name="binary_author_and_index_keywords_df",
            save_name="both keywords counts",
            top_n=top_n,
            **kwargs,
        )

    def count_keywords(
        self,
        which: Optional[Literal["author", "index", "both", "author and index"]] = None,
        top_n: int = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Generic keyword counter.
        
        Counts occurrences of author, index or combined keywords depending on
        the `which` argument and stores the result as `self.keywords_counts_df`.
        Optionally, binary indicator columns for the most frequent keywords are
        created.
        
        Parameters
        ----------
        which : {None, "author", "index", "both", "author and index"}, optional
            Which keyword field to use. The default picks the main keyword
            configuration stored on the object.
        top_n : int, default 0
            If > 0, also create binary indicator columns for the top-N
            keywords.
        **kwargs :
            Additional options forwarded to the binary-indicator helper.
        """
        # Determine the keyword column to use
        if which in ["both", "author and index"]:
            if "Author and Index Keywords" not in self.df.columns:
                self.df["Author and Index Keywords"] = utilsbib.merge_keywords_columns(
                    self.df, author_col="Author Keywords", 
                    index_col="Index Keywords", sep=self.default_separator
                )
            kw_column = ["Processed Author and Index Keywords", "Author and Index Keywords"]
        elif which == "author":
            kw_column = ["Processed Author Keywords", "Author Keywords"]
        elif which == "index":
            kw_column = ["Processed Index Keywords", "Index Keywords"]
        else:
            # Use default kw_var
            kw_column = getattr(self, "kw_var", None) or ["Author Keywords"]
        
        return self._count_entity(
            column=kw_column,
            item_label="Keyword",
            count_type="list",
            attr_name="keywords_counts_df",
            binary_attr_name="binary_keywords_df",
            save_name="keywords counts",
            top_n=top_n,
            **kwargs,
        )

    def count_authors(
        self,
        top_n: int = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Count occurrences of authors.
        
        Selects the most appropriate author column (full names, IDs or
        concatenated names) and uses `utilsbib.count_occurrences` to build
        `self.authors_counts_df`. For Scopus/OpenAlex data, helper mappings
        from IDs to names are attached where available. Optionally, binary
        indicator columns for the most productive authors are created.
        
        Parameters
        ----------
        top_n : int, default 0
            If > 0, also create binary indicator columns for the top-N authors.
        **kwargs :
            Additional options forwarded to the binary-indicator helper.
        """
        # Determine author column
        self.author_var = self._get_column(
            ["Author full names", "Author(s) ID", "Authors"],
            required=False
        ) or "Authors"
        
        # Post-processing for author data
        def process_authors(df):
            # Split author IDs if available
            if "Author full names" in self.df.columns:
                df = utilsbib.split_author_id(df)
            # Add OpenAlex author names
            if self.db in ["open alex", "openalex", "oa"]:
                if not hasattr(self, "oa_author_dict"):
                    self.oa_author_dict = utilsbib.openalex_build_author_id_name_dict(self.df)
                df["Author"] = df["Author(s) ID"].map(self.oa_author_dict)
            return df
        
        return self._count_entity(
            column=self.author_var,
            item_label="Author(s) ID",
            count_type="list",
            attr_name="authors_counts_df",
            binary_attr_name="binary_authors_df",
            save_name="authors counts",
            top_n=top_n,
            post_process=process_authors,
            **kwargs,
        )

    def count_affiliations(
        self,
        top_n: int = 0,
        **kwargs,
    ):
        """
        Count occurrences of affiliations.
    
        Uses the "Affiliations" column and ``utilsbib.count_occurrences`` to
        build ``self.affiliations_counts_df``. Optionally, binary indicator
        columns for the most frequent affiliations are created.
    
        Parameters
        ----------
        top_n : int, default 0
            If > 0, also create binary indicator columns for the top-N
            affiliations.
        **kwargs :
            Additional options forwarded to the binary-indicator helper.
        """
        return self._count_entity(
            column="Affiliations",
            item_label="Affiliation",
            count_type="list",
            attr_name="affiliations_counts_df",
            binary_attr_name="binary_affiliations_df",
            save_name="affiliation counts",
            top_n=top_n,
            **kwargs,
        )



    def count_fields(
        self,
        top_n: int = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Count occurrences of Scopus fields.
        
        Counts values in the "Field" column (typically ASJC fields) using
        `utilsbib.count_occurrences` and stores the results as
        `self.fields_counts_df`. Optionally, binary indicator columns for the
        most frequent fields are created.
        
        Parameters
        ----------
        top_n : int, default 0
            If > 0, also create binary indicator columns for the top-N fields.
        **kwargs :
            Additional options forwarded to the binary-indicator helper.
        """
        sep = ";" if self.db == "scopus" else self.default_separator
        return self._count_entity(
            column="Field",
            item_label="Field",
            count_type="list",
            attr_name="fields_counts_df",
            binary_attr_name="binary_fields_df",
            save_name="fields counts",
            sep=sep,
            top_n=top_n,
            **kwargs,
        )

    def count_areas(
        self,
        top_n: int = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Count occurrences of subject areas.
        
        Counts values in the "Area" column (for example Scopus subject areas)
        using `utilsbib.count_occurrences` and stores the result as
        `self.areas_counts_df`. Optionally, binary indicator columns for
        the most frequent areas are created.
        
        Parameters
        ----------
        top_n : int, default 0
            If > 0, also create binary indicator columns for the top-N areas.
        **kwargs :
            Additional options forwarded to the binary-indicator helper.
        """
        sep = ";" if self.db == "scopus" else self.default_separator
        return self._count_entity(
            column="Area",
            item_label="Area",
            count_type="list",
            attr_name="areas_counts_df",
            binary_attr_name="binary_areas_df",
            save_name="areas counts",
            sep=sep,
            top_n=top_n,
            **kwargs,
        )

    def count_sciences(
        self,
        top_n: int = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Count occurrences of broad sciences.
        
        Counts values in the "Science" column (for example Scopus broad
        sciences) using `utilsbib.count_occurrences` and stores the result as
        `self.sciences_counts_df`. Optionally, binary indicator columns for
        the most frequent sciences are created.
        
        Parameters
        ----------
        top_n : int, default 0
            If > 0, also create binary indicator columns for the top-N sciences.
        **kwargs :
            Additional options forwarded to the binary-indicator helper.
        """
        sep = ";" if self.db == "scopus" else self.default_separator
        return self._count_entity(
            column="Science",
            item_label="Science",
            count_type="list",
            attr_name="sciences_counts_df",
            binary_attr_name="binary_sciences_df",
            save_name="sciences counts",
            sep=sep,
            top_n=top_n,
            **kwargs,
        )

    # Add new or update existing functions:
    def count_references(
        self,
        min_len=50,
        top_n=0,
        **kwargs,
    ):
        """
        Count occurrences of cited references.
        
        Counts individual references in the "References" column and stores
        the result as `self.references_counts_df`. Very short entries can be
        filtered out using `min_len`. Optionally, binary indicator columns for
        the most frequently cited references are created.
        
        Parameters
        ----------
        min_len : int, default 50
            Minimum string length for a reference to be counted. For OpenAlex
            data this may be overridden internally.
        top_n : int, default 0
            If > 0, also create binary indicator columns for the top-N
            references.
        **kwargs :
            Additional options forwarded to the binary-indicator helper.
        """
        if self.db == "scopus":
            sep = ";"
        else:
            sep = self.default_separator
        print(sep)
        self.references_counts_df = utilsbib.count_occurrences(
            self.df, "References", count_type="list", item_column_name="Reference", sep=sep
        )
        if self.db == "oa":
            min_len = 1 # references from openalex are links
        self.references_counts_df = self.references_counts_df[self.references_counts_df["Reference"].str.len() >= min_len]
        if top_n > 0:
            top_items = self.references_counts_df["Reference"].head(top_n).tolist()
            _, indicators_dict = utilsbib.match_items_and_compute_binary_indicators(
                df=self.df,
                col="References",
                items_of_interest=top_items,
                value_type="list",
                separator=sep, **kwargs
            )
            self.binary_references_df = indicators_dict["binary"]
        self._save_table(self.references_counts_df, "references counts")
        return self.references_counts_df

    def count_ngrams_abstract(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        top_n: int = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Count word and phrase n-grams in abstracts.
        
        Works on the processed abstract column when available and stores
        counts as `self.words_abs_counts_df` (column "Word - Phrase").
        Optionally, binary indicator columns for the most frequent n-grams
        are created.
        
        Parameters
        ----------
        ngram_range : tuple, default (1, 2)
            The (min_n, max_n) n-gram lengths to consider.
        top_n : int, default 0
            If > 0, also create binary indicator columns for the top-N
            n-grams.
        **kwargs :
            Additional options forwarded to the binary-indicator helper.
        """
        return self._count_entity(
            column=["Processed Abstract", "Abstract"],
            item_label="Word - Phrase",
            count_type="text",
            attr_name="words_abs_counts_df",
            binary_attr_name="binary_words_abs_df",
            save_name="words abstract counts",
            ngram_range=ngram_range,
            top_n=top_n,
            **kwargs,
        )

    def count_ngrams_title(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        top_n: int = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Count word and phrase n-grams in titles.
        
        Works on the processed title column when available and stores counts
        as `self.words_tit_counts_df` (column "Word - Phrase"). Optionally,
        binary indicator columns for the most frequent n-grams are created.
        
        Parameters
        ----------
        ngram_range : tuple, default (1, 2)
            The (min_n, max_n) n-gram lengths to consider.
        top_n : int, default 0
            If > 0, also create binary indicator columns for the top-N
            n-grams.
        **kwargs :
            Additional options forwarded to the binary-indicator helper.
        """
        return self._count_entity(
            column=["Processed Title", "Title"],
            item_label="Word - Phrase",
            count_type="text",
            attr_name="words_tit_counts_df",
            binary_attr_name="binary_words_tit_df",
            save_name="words tit counts",
            ngram_range=ngram_range,
            top_n=top_n,
            **kwargs,
        )

    def count_ngrams_combined_text(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        top_n: int = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Count n-grams in the combined text column, optionally creating binary indicators and saving an Excel table.

        Prefers "Processed Combined Text"; falls back to "Combined Text". If neither exists, returns None.

        Parameters
        ----------
        ngram_range : tuple[int, int], default=(1, 2)
            Inclusive n-gram range (e.g., (1, 2) counts unigrams and bigrams).
        top_n : int, default=0
            If > 0, compute binary indicators for the top-N items.
        **kwargs :
            Forwarded to utilsbib.match_items_and_compute_binary_indicators.

        Returns
        -------
        pd.DataFrame | None
            DataFrame with counts in column "Word - Phrase", or None if no combined column is present.
        """
        return self._count_entity(
            column=["Processed Combined Text", "Combined Text"],
            item_label="Word - Phrase",
            count_type="text",
            attr_name="words_comb_counts_df",
            binary_attr_name="binary_words_comb_df",
            save_name="words combined counts",
            ngram_range=ngram_range,
            top_n=top_n,
            **kwargs,
        )

    def count_ngrams(
        self,
        ngram_range=(1, 2),
        top_n=0,
        **kwargs,
    ):
        """
        Convenience wrapper to count n-grams in both abstracts and titles.
        
        Calls `count_ngrams_abstract` and `count_ngrams_title` with the same
        arguments.
        
        Parameters
        ----------
        ngram_range : tuple, default (1, 2)
            The (min_n, max_n) n-gram lengths to consider.
        top_n : int, default 0
            If > 0, both underlying methods also create binary indicator
            columns for the top-N n-grams.
        **kwargs :
            Additional options forwarded to the underlying methods.
        """
        self.count_ngrams_abstract(ngram_range=ngram_range, top_n=top_n, **kwargs)
        self.count_ngrams_title(ngram_range=ngram_range, top_n=top_n, **kwargs)

    def count_all_countries(
        self,
        top_n: int = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Count all author countries (not only corresponding authors).
        
        Ensures that a "Countries of Authors" column exists, deriving it from
        affiliations or country codes when necessary, and then counts
        occurrences via `utilsbib.count_occurrences`. Results are stored as
        `self.all_countries_counts_df`, with ISO-3 codes attached. Optionally,
        binary indicator columns for the most frequent countries are created.
        
        Parameters
        ----------
        top_n : int, default 0
            If > 0, also create binary indicator columns for the top-N
            countries.
        **kwargs :
            Additional options forwarded to the binary-indicator helper.
        """
        if "Countries of Authors" not in self.df.columns:
            # Determine affiliation column based on database
            aff_column = None
            if self.db == "scopus":
                aff_column = "Affiliations"
            elif self.db == "wos":
                # WoS uses "Affiliations" or "Correspondence Address"
                for candidate in ["Affiliations", "Correspondence Address", "Reprint Address"]:
                    if candidate in self.df.columns:
                        aff_column = candidate
                        break
            else:
                # Generic fallback
                for candidate in ["Affiliations", "Addresses", "Address"]:
                    if candidate in self.df.columns:
                        aff_column = candidate
                        break
            
            if aff_column and aff_column in self.df.columns:
                self.df, self.country_collab_matrix = utilsbib.extract_countries_from_affiliations(self.df, aff_column=aff_column)
            else:
                # Can't extract countries - create empty column
                self.df["Countries of Authors"] = ""
                self.country_collab_matrix = None
        if self.db == "oa":
            self.df = utilsbib.openalex_map_country_codes(self.df)

        self.all_countries_counts_df = utilsbib.count_occurrences(
            self.df, "Countries of Authors", count_type="list", item_column_name="Country", sep=self.default_separator)
        if top_n > 0:
            top_items = self.all_countries_counts_df["Country"].head(top_n).tolist()
            _, indicators_dict = utilsbib.match_items_and_compute_binary_indicators(
                df=self.df,
                col="Countries of Authors",
                items_of_interest=top_items,
                value_type="list", separator=self.default_separator, **kwargs
            )
            self.binary_all_countries_df = indicators_dict["binary"]
        self._save_table(self.all_countries_counts_df, "all countries counts")

        self.all_countries_counts_df["ISO-3"] = self.all_countries_counts_df["Country"].map(utilsbib.country_iso3_dct)
        return self.all_countries_counts_df

    def count_all(
        self,
        top_n: int = 0,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Run a suite of basic counting routines in one call.
        
        Sequentially invokes several `count_*` methods (sources, document
        types, countries, keywords, authors, affiliations, references, fields
        and sciences) with the provided arguments. Useful as a quick
        one-liner after filtering.
        
        Parameters
        ----------
        top_n : int, default 0
            Passed to each underlying `count_*` method.
        **kwargs :
            Additional options forwarded to the underlying methods.
        """
        for f in [
            "count_sources", "count_document_types", "count_ca_countries", "count_author_keywords",
            "count_index_keywords", "count_authors", "count_affiliations",
            "count_references", "count_fields", "count_areas", "count_sciences",
            "count_ngrams_abstract", "count_ngrams_title", "count_all_countries"
        ]:
            try:
                getattr(self, f)(top_n=top_n, **kwargs)
            except Exception:
                print("Problem", f)
                pass

    # performance measuring

    """
    Compute performance statistics for a specific entity type (e.g., Source, Author, Country, Document Type).
    This method selects relevant items based on provided filters or top-N counts and delegates the analysis
    to a generic utility function `get_entity_stats`.

    The resulting statistics are stored in a DataFrame attribute (e.g., self.source_stats_df),
    and if `indicators=True`, a corresponding indicator matrix is also stored (e.g., self.source_indicators).

    Parameters are passed through to `get_entity_stats`, including:
    - items_of_interest: list of entities to include explicitly
    - exclude_items: list of entities to exclude
    - top_n: number of top entities to include if items_of_interest is not provided
    - regex_include / regex_exclude: optional patterns to filter entities
    - indicators: whether to return a binary document-entity indicator matrix
    - missing_as_zero: whether to treat missing entries as zeros
    - mode: analysis mode passed to performance metric function

    Relies on a user-defined `count_<entity>` method if counts_df is not provided.
    """

    def get_sources_stats(
        self,
        top_n: int = 100,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute extended statistics for sources (journals).
        
        Uses `utilsbib.get_entity_stats` on the "Source title" column.
        Includes h-index variants and other bibliometric indicators.
        
        Parameters
        ----------
        top_n : int, default 100
            Number of top sources to analyze.
        **kwargs :
            Additional options forwarded to `utilsbib.get_entity_stats`.
        
        Returns
        -------
        DataFrame
            Statistics for top sources.
        """
        def add_abbreviations(df):
            if hasattr(self, "sources_abb_dict") and self.sources_abb_dict:
                df["Abbreviated Source Title"] = df["Source"].map(self.sources_abb_dict)
            return df
        
        return self._get_entity_stats(
            column="Source title",
            item_label="Source",
            count_method_name="count_sources",
            attr_name="sources_stats_df",
            indicators_attr_name="sources_indicators",
            save_name="sources stats",
            value_type="string",
            top_n=top_n,
            post_process=add_abbreviations,
            **kwargs,
        )

    def get_document_type_stats(
        self,
        top_n: int = 100,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute extended statistics for document types.
        
        Parameters
        ----------
        top_n : int, default 100
            Number of document types to analyze.
        **kwargs :
            Additional options forwarded to `utilsbib.get_entity_stats`.
        
        Returns
        -------
        DataFrame
            Statistics for document types.
        """
        return self._get_entity_stats(
            column="Document Type",
            item_label="Document Type",
            count_method_name="count_document_types",
            attr_name="document_type_stats_df",
            indicators_attr_name="document_type_indicators",
            save_name="document types stats",
            value_type="string",
            top_n=top_n,
            **kwargs,
        )

    def get_ca_countries_stats(
        self,
        top_n: int = 100,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute extended statistics for corresponding-author countries.
        
        Parameters
        ----------
        top_n : int, default 100
            Number of countries to analyze.
        **kwargs :
            Additional options forwarded to `utilsbib.get_entity_stats`.
        
        Returns
        -------
        DataFrame
            Statistics for countries.
        """
        return self._get_entity_stats(
            column="CA Country",
            item_label="Country",
            count_method_name="count_ca_countries",
            attr_name="ca_countries_stats_df",
            indicators_attr_name="ca_countries_indicators",
            save_name="ca countries stats",
            value_type="string",
            top_n=top_n,
            **kwargs,
        )

    def get_author_keywords_stats(
        self,
        top_n: int = 100,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute extended statistics for author keywords.
        
        Parameters
        ----------
        top_n : int, default 100
            Number of keywords to analyze.
        **kwargs :
            Additional options forwarded to `utilsbib.get_entity_stats`.
        
        Returns
        -------
        DataFrame
            Statistics for author keywords.
        """
        return self._get_entity_stats(
            column=["Processed Author Keywords", "Author Keywords"],
            item_label="Keyword",
            count_method_name="count_author_keywords",
            attr_name="author_keywords_stats_df",
            indicators_attr_name="author_keywords_indicators",
            save_name="author keywords stats",
            value_type="list",
            top_n=top_n,
            **kwargs,
        )

    def get_index_keywords_stats(
        self,
        top_n: int = 100,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute extended statistics for index keywords.
        
        Parameters
        ----------
        top_n : int, default 100
            Number of keywords to analyze.
        **kwargs :
            Additional options forwarded to `utilsbib.get_entity_stats`.
        
        Returns
        -------
        DataFrame
            Statistics for index keywords.
        """
        return self._get_entity_stats(
            column=["Processed Index Keywords", "Index Keywords"],
            item_label="Keyword",
            count_method_name="count_index_keywords",
            attr_name="index_keywords_stats_df",
            indicators_attr_name="index_keywords_indicators",
            save_name="index keywords stats",
            value_type="list",
            top_n=top_n,
            **kwargs,
        )

    def get_both_keywords_stats(
        self,
        top_n: int = 100,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute extended statistics for combined author and index keywords.
        
        Parameters
        ----------
        top_n : int, default 100
            Number of keywords to analyze.
        **kwargs :
            Additional options forwarded to `utilsbib.get_entity_stats`.
        
        Returns
        -------
        DataFrame
            Statistics for combined keywords.
        """
        return self._get_entity_stats(
            column=["Processed Author and Index Keywords", "Author and Index Keywords"],
            item_label="Keyword",
            count_method_name="count_both_keywords",
            attr_name="author_and_index_keywords_stats_df",
            indicators_attr_name="author_and_index_keywords_indicators",
            save_name="author and index keywords stats",
            value_type="list",
            top_n=top_n,
            **kwargs,
        )

    def get_keywords_stats(
        self,
        keyword_types: List[str] = None,
        top_n: int = 100,
        **kwargs: Any,
    ) -> None:
        """
        Compute extended statistics for keyword fields.
        
        Parameters
        ----------
        keyword_types : list, default ['author', 'index']
            Which keyword types to analyze.
        top_n : int, default 100
            Number of keywords to analyze per type.
        **kwargs :
            Additional options forwarded to `utilsbib.get_entity_stats`.
        """
        if keyword_types is None:
            keyword_types = ["author", "index"]
        
        for keyword_type in keyword_types:
            if keyword_type == "author":
                self.get_author_keywords_stats(top_n=top_n, **kwargs)
            elif keyword_type == "index":
                self.get_index_keywords_stats(top_n=top_n, **kwargs)
            elif keyword_type in ["both", "author and index"]:
                self.get_both_keywords_stats(top_n=top_n, **kwargs)

    def get_authors_stats(
        self,
        top_n: int = 100,
        shorten_first_name: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute extended statistics for authors.
        
        Parameters
        ----------
        top_n : int, default 100
            Number of top authors to analyze.
        shorten_first_name : bool, default True
            Whether to shorten first names to initials.
        **kwargs :
            Additional options forwarded to `utilsbib.get_entity_stats`.
        
        Returns
        -------
        DataFrame
            Statistics for authors.
        """
        # Determine author variable based on database
        auth_var_map = {
            "scopus": "Author full names", 
            "oa": "Author(s) ID",
            "wos": "Author full names",
            "pubmed": "Authors",
            "dimensions": "Authors",
            "lens": "Authors",
        }
        auth_var = auth_var_map.get(self.db, "Authors")
        
        # Fallback: find first available author column
        if auth_var not in self.df.columns:
            auth_var = self._get_column(
                ["Author full names", "Authors or Inventors", "Authors", "Author(s) ID"],
                required=False
            ) or auth_var
        
        # Determine ID column
        id_col = self._get_column(["Author(s) ID"], required=False) or auth_var
        
        self.authors_stats_df, self.authors_indicators = utilsbib.get_entity_stats(
            self.df, auth_var, id_col,
            count_method=self.count_authors, 
            value_type="list", 
            sep=self.default_separator,
            top_n=top_n,
            **kwargs,
        )
        
        # Post-process author data
        if "Author(s) ID" in self.authors_stats_df.columns:
            self.authors_stats_df = utilsbib.split_author_id(self.authors_stats_df)
        
        if self.db in ["open alex", "openalex", "oa"]:
            if not hasattr(self, "oa_author_dict"):
                self.oa_author_dict = utilsbib.openalex_build_author_id_name_dict(self.df)
            self.authors_stats_df["Author"] = self.authors_stats_df["Author(s) ID"].map(self.oa_author_dict)
        
        if "Author" in self.authors_stats_df.columns and shorten_first_name:
            self.authors_stats_df["Author"] = self.authors_stats_df["Author"].str.replace(
                r"^\s*([^,]+),\s*(?i:(?:dr|prof|mr|ms|mrs).?\s*)?([A-Za-zÀ-ÖØ-öø-ÿ]).*$", 
                r"\1, \2.", 
                regex=True
            )
        
        self._save_table(self.authors_stats_df, "author stats")
        return self.authors_stats_df

    def get_fields_stats(
        self,
        top_n: int = 100,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute extended statistics for subject fields.
        
        Parameters
        ----------
        top_n : int, default 100
            Number of fields to analyze.
        **kwargs :
            Additional options forwarded to `utilsbib.get_entity_stats`.
        
        Returns
        -------
        DataFrame
            Statistics for fields.
        """
        return self._get_entity_stats(
            column="Field",
            item_label="Field",
            count_method_name="count_fields",
            attr_name="fields_stats_df",
            indicators_attr_name="fields_indicators",
            save_name="fields stats",
            value_type="list",
            top_n=top_n,
            **kwargs,
        )

    def get_areas_stats(
        self,
        top_n: int = 100,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute extended statistics for subject areas.
        
        Parameters
        ----------
        top_n : int, default 100
            Number of areas to analyze.
        **kwargs :
            Additional options forwarded to `utilsbib.get_entity_stats`.
        
        Returns
        -------
        DataFrame
            Statistics for areas.
        """
        return self._get_entity_stats(
            column="Area",
            item_label="Area",
            count_method_name="count_areas",
            attr_name="areas_stats_df",
            indicators_attr_name="areas_indicators",
            save_name="areas stats",
            value_type="list",
            top_n=top_n,
            **kwargs,
        )

    def get_sciences_stats(
        self,
        top_n: int = 100,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute extended statistics for sciences.
        
        Parameters
        ----------
        top_n : int, default 100
            Number of sciences to analyze.
        **kwargs :
            Additional options forwarded to `utilsbib.get_entity_stats`.
        
        Returns
        -------
        DataFrame
            Statistics for sciences.
        """
        return self._get_entity_stats(
            column="Science",
            item_label="Science",
            count_method_name="count_sciences",
            attr_name="sciences_stats_df",
            indicators_attr_name="sciences_indicators",
            save_name="sciences stats",
            value_type="list",
            top_n=top_n,
            **kwargs,
        )

    def get_all_sciences_stats(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Convenience wrapper to compute stats for fields, areas and sciences.
        
        Sequentially calls `get_fields_stats`, `get_areas_stats` and
        `get_sciences_stats` with the supplied keyword arguments.
        
        Parameters
        ----------
        **kwargs :
            Additional options forwarded to the underlying methods.
        """
        self.get_fields_stats(**kwargs)
        self.get_areas_stats(**kwargs)
        self.get_sciences_stats(**kwargs)

    def get_affiliations_stats(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Compute extended statistics for affiliations.
    
        Uses the "Affiliations" column with ``utilsbib.get_entity_stats`` and
        stores results as ``self.affiliations_stats_df`` and
        ``self.affiliations_indcators``. Statistics are computed using the
        full affiliation strings.
    
        After computation, two label columns are provided:
    
            - "Affiliation original": full affiliation string.
            - "Affiliation": text up to the first comma (short label).
    
        The table is exported to Excel when an output folder is configured.
    
        Parameters
        ----------
        **kwargs :
            Additional options forwarded to ``utilsbib.get_entity_stats``.
        """
        import os
    
        # Compute stats using full affiliations, consistent with counting
        self.affiliations_stats_df, self.affiliations_indcators = utilsbib.get_entity_stats(
            self.df,
            "Affiliations",
            "Affiliation",
            count_method=self.count_affiliations,
            value_type="list",
            sep=self.default_separator,
            **kwargs,
        )
    
        # Add original + shortened affiliation labels *after* stats are computed
        if "Affiliation" in self.affiliations_stats_df.columns:
            col = self.affiliations_stats_df["Affiliation"]
            self.affiliations_stats_df["Affiliation original"] = col
    
            mask = col.notna()
            self.affiliations_stats_df.loc[mask, "Affiliation"] = (
                col[mask]
                .astype(str)
                .str.split(",", n=1)
                .str[0]
                .str.strip()
            )
    
        # Save stats table
        self._save_table(self.affiliations_stats_df, "affiliations stats")
    
        return self.affiliations_stats_df, self.affiliations_indcators


    def get_ngrams_abstract_stats(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Compute statistics for n-grams in abstracts.
        
        Uses the processed or raw abstract column with `utilsbib.get_entity_stats`
        and the `count_ngrams_abstract` method. Results are stored as
        `self.ngrams_abstract_stats_df` and
        `self.ngrams_abstract_stats_indicators`, and exported to Excel when an
        output folder is configured.
        
        Parameters
        ----------
        **kwargs :
            Additional options forwarded to `utilsbib.get_entity_stats`.
        """
        v = self._get_column(["Processed Abstract", "Abstract"])
        self.ngrams_abstract_stats_df, self.ngrams_abstract_stats_indicators = utilsbib.get_entity_stats(
                self.df, v, "Word - Phrase",
                count_method=self.count_ngrams_abstract, value_type="text", sep=self.default_separator, **kwargs)
        self._save_table(self.ngrams_abstract_stats_df, "ngrams abstract stats")

    def get_ngrams_title_stats(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Compute statistics for n-grams in titles.
        
        Uses the processed or raw title column with `utilsbib.get_entity_stats`
        and the `count_ngrams_title` method. Results are stored as
        `self.ngrams_title_stats_df` and `self.ngrams_title_stats_indicators`,
        and exported to Excel when an output folder is configured.
        
        Parameters
        ----------
        **kwargs :
            Additional options forwarded to `utilsbib.get_entity_stats`.
        """
        v = self._get_column(["Processed Title", "Title"])
        self.ngrams_title_stats_df, self.ngrams_title_stats_indicators = utilsbib.get_entity_stats(
                self.df, v, "Word - Phrase",
                count_method=self.count_ngrams_title, value_type="text", sep=self.default_separator, **kwargs)
        self._save_table(self.ngrams_title_stats_df, "ngrams title stats")

    def get_ngrams_combined_text_stats(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Compute n-gram statistics for a combined text column.

        Prefers "Processed Combined Text"; falls back to "Combined Text".
        If neither column exists in self.df, returns None.

        Parameters
        ----------
        **kwargs :
            Forwarded to utilsbib.get_entity_stats (e.g., ngram_range, top_n, etc.).

        Returns
        -------
        pd.DataFrame | None
            The n-gram statistics DataFrame on success; otherwise None.
        """
        v = (
            "Processed Combined Text"
            if "Processed Combined Text" in self.df.columns
            else ("Combined Text" if "Combined Text" in self.df.columns else None)
        )
        if v is None:
            return None

        self.ngrams_combined_stats_df, self.ngrams_combined_stats_indicators = utilsbib.get_entity_stats(
            self.df,
            v,
            "Word - Phrase",
            count_method=self.count_ngrams_combined_text,
            value_type="text",
            sep=self.default_separator,
            **kwargs,
        )

        self._save_table(self.ngrams_combined_stats_df, "ngrams combined text stats")

    def get_references_stats(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Compute extended statistics for cited references.
        
        Calls `utilsbib.get_entity_stats` on the "References" column together
        with the `count_references` method. Results are stored as
        `self.references_stats_df` and `self.references_stats_indicators`, and
        exported to Excel when an output folder is configured.
        
        Parameters
        ----------
        **kwargs :
            Additional options forwarded to `utilsbib.get_entity_stats`.
        """
        self.references_stats_df, self.references_stats_indicators = utilsbib.get_entity_stats(
                self.df, "References", "Reference",
                count_method=self.count_references, value_type="list", sep=self.default_separator, openalex_add=(self.db=="oa"), **kwargs)
        self._save_table(self.references_stats_df, "references stats")

    def get_all_countries_stats(
        self,
        top_n=200,
        region=None,
        normalize_by: str = None,
        **kwargs,
    ):
        """
        Compute statistics for all countries and optionally subset results by a region.

        Parameters
        ----------
        top_n : int, default 200
            Passed to `utilsbib.get_entity_stats`.
        region : str or None, default None
            If None, no filtering is applied.
            If "EU" (case-insensitive), keep only EU member states using `utilsbib.df_countries["EU"]`.
            Otherwise, treat `region` as a continent name and match against `utilsbib.df_countries["Continent"]`.
        normalize_by : str or None, default None
            Normalize counts by population or researchers. Options:
            - None: No normalization (default)
            - "population": Publications per million inhabitants
            - "researchers": Publications per 1000 researchers
            - "both": Add both normalized columns
            Data is fetched from World Bank API (with fallback to built-in data).

        Notes
        -----
        - `self.all_countries_stats_df` is filtered by rows where "Country" is in the allowed set.
        - `self.all_countries_indcators` may be None; if it is a dict, each value is a DataFrame
          with countries as COLUMNS and will be column-filtered to the allowed set.
        """
        # Compute full stats first
        self.all_countries_stats_df, self.all_countries_indcators = utilsbib.get_entity_stats(
            self.df,
            "Countries of Authors",
            "Country",
            count_method=self.count_all_countries,
            value_type="list",
            top_n=top_n,
            sep=self.default_separator,
            **kwargs,
        )

        # Optional regional filtering
        if region is not None:
            dfc = utilsbib.df_countries.copy()

            # Build mask based on region
            region_str = str(region).strip()
            if region_str.lower() == "eu":
                # Coerce EU to boolean
                eu = dfc["EU"]
                # Accept common encodings for truthy values just in case
                truthy = {"1", "true", "yes", "y", "t"}
                eu_bool = eu.apply(
                    lambda v: bool(v)
                    if isinstance(v, (bool, int))
                    else (str(v).strip().lower() in truthy) if v is not None
                    else False
                )
                mask = eu_bool
            else:
                # Match continent case-insensitively
                mask = dfc["Continent"].fillna("").str.strip().str.casefold() == region_str.casefold()

            # Allowed country names (prefer "Name"; fall back to "Official name" if missing)
            allowed = set()
            if "Name" in dfc.columns:
                allowed.update(dfc.loc[mask, "Name"].dropna().astype(str).str.strip())
            if not allowed and "Official name" in dfc.columns:
                allowed.update(dfc.loc[mask, "Official name"].dropna().astype(str).str.strip())

            # Filter stats DataFrame rows by "Country"
            if not self.all_countries_stats_df.empty and "Country" in self.all_countries_stats_df.columns:
                self.all_countries_stats_df = (
                    self.all_countries_stats_df[
                        self.all_countries_stats_df["Country"].astype(str).str.strip().isin(allowed)
                    ].reset_index(drop=True)
                )

            # Filter each indicators DataFrame by columns (countries are columns)
            if isinstance(self.all_countries_indcators, dict) and self.all_countries_indcators:
                filtered = {}
                for k, df_ind in self.all_countries_indcators.items():
                    if hasattr(df_ind, "columns"):
                        keep_cols = [c for c in df_ind.columns if str(c).strip() in allowed]
                        filtered[k] = df_ind.loc[:, keep_cols]
                    else:
                        filtered[k] = df_ind
                self.all_countries_indcators = filtered

        # Apply normalization if requested
        if normalize_by is not None and len(self.all_countries_stats_df) > 0:
            try:
                from biblium.normalization import normalize_country_counts, get_normalization_data
            except ImportError:
                from normalization import normalize_country_counts, get_normalization_data
            
            print(f"\nNormalizing country statistics by {normalize_by}...")
            norm_data = get_normalization_data(use_api=True)
            # Find the count column - could be "Count" or "Number of documents"
            count_col = "Count"
            if "Number of documents" in self.all_countries_stats_df.columns:
                count_col = "Number of documents"
            elif "Count" in self.all_countries_stats_df.columns:
                count_col = "Count"
            else:
                # Find first numeric column after Country
                for col in self.all_countries_stats_df.columns:
                    if col != "Country" and pd.api.types.is_numeric_dtype(self.all_countries_stats_df[col]):
                        count_col = col
                        break
            
            self.all_countries_stats_df = normalize_country_counts(
                self.all_countries_stats_df,
                country_col="Country",
                count_col=count_col,
                normalize_by=normalize_by,
                normalization_data=norm_data,
            )
            # Store normalization data for reuse
            self._country_normalization_data = norm_data

        # Save as before
        self._save_table(self.all_countries_stats_df, "all countries stats")

    def get_all_items_stats(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Convenience wrapper to compute statistics for all main entity types.
        
        Sequentially calls several `get_*_stats` methods (sources, document
        types, countries, keywords, authors, affiliations, references, fields,
        areas and sciences). Errors in individual calls are caught and
        silently ignored so that partial results are still available.
        
        Parameters
        ----------
        **kwargs :
            Additional options forwarded to the underlying methods.
        """
        for f in ["get_sources_stats", "get_document_types_stats", "get_ca_countries_stats",
                  "get_author_keywords_stats", "get_index_keywords_stats", "get_authors_stats",
                  "get_affiliations_stats", "get_references_stats", "get_fields_stats", "get_areas_stats",
                  "get_sciences_stats", "get_ngrams_abstract_stats", "get_ngrams_title_stats"]:
            try:
                getattr(self, f)(**kwargs)
            except:
                pass

    def compute_reference_spectrogram(
        self,
        reference_column='References',
        include_pre_1900=False,
    ):
        """
        Compute a reference spectrogram from cited reference years.
        
        For Scopus datasets this delegates to `utilsbib.compute_reference_spectrogram`
        to build a year-by-reference intensity table and stores it as
        `self.spectrogram_df`. Other databases are currently ignored.
        
        Parameters
        ----------
        reference_column : str, default "References"
            Name of the column containing reference strings.
        include_pre_1900 : bool, default False
            If True, include references published before 1900.
        """
        if self.db == "scopus":
            self.spectrogram_df = utilsbib.compute_reference_spectrogram_scopus(self.df, reference_column=reference_column, include_pre_1900=include_pre_1900)
        elif self.db == "oa":
            pass
        else:
            print("Not yet supported")

    # factor analysis

    def conceptual_structure_analysis(
        self,
        field: str = "Author Keywords",
        dr_method: str = "MCA",
        cluster_method: str = "kmeans",
        n_clusters: int = 5,
        n_terms: int = 100,
        n_components: int = 2,
        dtm_method: str = "count",
        term_selection: str = "frequency",
        y: np.ndarray | None = None,
        min_df: int = 2,
        ngram_range: tuple = (1, 1),
        use_lemmatization: bool = False,
        pos_filter: list | None = None,
        include_terms: list | None = None,
        exclude_terms: list | None = None,
        term_regex: str | None = None,
        compute_metrics: bool = False,
        filename: str | None = "conceptual analysis",
        keyword_separator: str | None = None,
    ) -> None:
        """
        Run conceptual structure analysis on a keyword or text field.
    
        This is a high-level wrapper around ``utilsbib.conceptual_structure_analysis``.
        It computes a document-term matrix, applies dimensionality reduction and
        clustering, and stores the results (embeddings, term labels, cluster
        assignments) in ``self.conceptual_structure_d``. A helper table with
        representative words per cluster is stored as ``self.words_by_cluster_df``.
    
        When ``filename`` is not None, key tables are exported to Excel files in
        the results folder:
    
            * A multi-sheet workbook with term/doc embeddings and metrics.
            * A table of representative words per cluster.
            * A wide table with terms per cluster.
    
        Parameters
        ----------
        field : str, default "Author Keywords"
            Name of the text or keyword field to analyse. The underlying utility
            will automatically prefer processed variants when available (for
            example, "Processed Author Keywords", "Processed Abstract").
        dr_method : str, default "MCA"
            Dimensionality reduction method.
        cluster_method : str, default "kmeans"
            Clustering algorithm applied in the reduced space.
        n_clusters : int, default 5
            Number of clusters to extract (where applicable).
        n_terms : int, default 100
            Maximum number of terms to keep.
        n_components : int, default 2
            Number of dimensions in the reduced space.
        dtm_method : str, default "count"
            Document-term matrix method ("count" or "tfidf").
        term_selection : str, default "frequency"
            Term selection method ("frequency", "chi2", "mutual_info").
        y : np.ndarray or None, default None
            Target labels for supervised term selection.
        min_df : int, default 2
            Minimum document frequency for term inclusion.
        ngram_range : tuple, default (1, 1)
            N-gram range for free-text fields.
        use_lemmatization : bool, default False
            Flag forwarded to the underlying function (currently a placeholder
            if you rely on processed text columns).
        pos_filter : list or None
            POS tags to include (for free-text fields; placeholder).
        include_terms, exclude_terms, term_regex :
            Advanced term filtering arguments passed directly to
            ``utilsbib.conceptual_structure_analysis``.
        compute_metrics : bool, default False
            Whether to compute clustering quality metrics.
        filename : str or None, default "conceptual analysis"
            Base name for Excel output files in ``self.res_folder / "tables"``.
            If None, no Excel files are written.
        keyword_separator : str or None, default None
            Separator used to split keyword fields into individual keywords.
            If None, ``self.keyword_separator`` is used when present, otherwise
            ";" is assumed.
        """
        import os
    
        # Decide keyword separator: parameter > class attribute > default ";"
        if keyword_separator is None:
            keyword_separator = getattr(self, "default_separator", ";")
    
        # If filename is given, prepare a multi-sheet Excel path for full results
        excel_path = None
        if filename is not None:
            excel_path = os.path.join(
                self.res_folder,
                "tables",
                f"{filename}_conceptual_structure.xlsx",
            )
    
        # Call the utility function - it will resolve processed vs. raw field
        self.conceptual_structure_d = utilsbib.conceptual_structure_analysis(
            self.df,
            field=field,
            dr_method=dr_method,
            cluster_method=cluster_method,
            n_clusters=n_clusters,
            n_terms=n_terms,
            n_components=n_components,
            dtm_method=dtm_method,
            term_selection=term_selection,
            y=y,
            min_df=min_df,
            ngram_range=ngram_range,
            use_lemmatization=use_lemmatization,
            pos_filter=pos_filter,
            include_terms=include_terms,
            exclude_terms=exclude_terms,
            term_regex=term_regex,
            compute_metrics=compute_metrics,
            excel_path=excel_path,
            keyword_separator=keyword_separator,
        )
    
        # Representative words per cluster
        self.words_by_cluster_df = utilsbib.words_by_cluster(
            self.conceptual_structure_d["term_embeddings"],
            self.conceptual_structure_d["terms"],
            self.conceptual_structure_d["term_labels"],
        )
    
        # Optional Excel exports of key tables
        if filename is not None:
            tables_folder = os.path.join(self.res_folder, "tables")
            os.makedirs(tables_folder, exist_ok=True)
    
            f1 = os.path.join(
                tables_folder,
                f"{filename}_terms_embeddings.xlsx",
            )
            f2 = os.path.join(
                tables_folder,
                f"{filename}_terms_clusters.xlsx",
            )
    
            utilsbib.to_excel_fancy(
                self.words_by_cluster_df,
                f_name=f1,
                autofit=self.autofit,
                conditional_formatting=self.cond_formatting,
            )
            utilsbib.to_excel_fancy(
                self.conceptual_structure_d["clusters_terms_df"],
                f_name=f2,
                autofit=self.autofit,
                conditional_formatting=self.cond_formatting,
            )

    # relations computation
    from typing import Optional
    # general coocurences
    def compute_cooccurrence(
        self,
        column_name,
        items_of_interest=None,
        top_n=20,
        normalization=False,
        network=True,
        partition_network=True,
        partition_kwargs=None,
        vector_df=None,
        vector_name_col=None,
        network_filename=None,
        network_formats=['pajek'],
        count_func=None,
        count_attr=None,
        output_attr_prefix=None,
        value_type='list',
        separator=None,
        use_cache=True,
        **kwargs,
    ):
        """
        Compute co-occurrence (and optional normalized matrices) for items in a column.

        This function builds a binary document-item matrix for selected items and derives:
          - A co-occurrence matrix (and optional normalized variants),
          - An (optional) co-occurrence network with community partitions,
          - Two helper DataFrames:
              nodes_info_df: one row per node containing partition labels (for each
                 detected partition algorithm) and any numeric "vector" columns provided via vector_df.
              clusters_wide_df: a wide layout where each column is a "PartitionName ClusterId"
                 and the rows list the items (nodes) in that cluster. Column order respects the insertion
                 order of partition algorithms; within each algorithm, cluster IDs are ascending.

        Side-effects:
          - If self.res_folder is not None, both helper DataFrames are saved to
            <self.res_folder>/networks/ with filenames based on network_filename
            (or `<column_name>_cooccurrence` if `network_filename` is None).
          - If `output_attr_prefix` is provided, the helper DataFrames are also attached to `self` as:
                `{prefix}_nodes_info_df` and `{prefix}_clusters_wide_df`.

        Parameters
        ----------
        column_name : str
            DataFrame column with items (as list-strings or delimited strings).
        items_of_interest : list[str] or None, optional
            Explicit set of items to consider. If None, `top_n` and counting logic are used.
        top_n : int, default 20
            If selecting by counts, limit to top-N items.
        normalization : bool, default False
            Whether to compute normalized co-occurrence matrices.
        network : bool, default True
            Build and return a NetworkX graph from co-occurrences.
        partition_network : bool, default True
            Compute partitions/communities on the network.
        partition_kwargs : dict or None
            Extra kwargs for the partition routine.
        vector_df : pandas.DataFrame or None
            If provided, acts as the "top items" table and its numeric columns are
            added to nodes as attributes; vector_name_col indicates the name column
            (defaults to the first column).
        vector_name_col : str or None
            Column in `vector_df` containing the item names.
        network_filename : str or None
            Basename for network exports and derived tables.
        network_formats : list[str], default ["pajek"]
            Formats for network export.
        count_func : callable or None
            A function that populates `self.{count_attr}` when `items_of_interest` and
            `vector_df` are not provided.
        count_attr : str or None
            Name of the attribute on `self` to retrieve counts from after `count_func()`.
        output_attr_prefix : str or None
            If provided, results are also saved on `self` under this prefix.
        value_type : {"list","string"}, default "list"
            How values are stored in `self.df[column_name]`.
        separator : str or None
            Delimiter for "string" value_type; falls back to `self.default_separator` if present.
        **kwargs :
            Passed through to `utilsbib.select_documents`.

        Returns
        -------
        co_matrix : pandas.DataFrame
            Symmetric co-occurrence matrix for the selected items.
        co_matrices_norm : dict[str, pandas.DataFrame] or None
            Normalized variants when `normalization=True`, else {}.
        co_network : networkx.Graph or None
            The co-occurrence network (without self-loops). May be None if `network=False`.

        Notes
        -----
        - The helper DataFrames are only created when a network exists (i.e., `co_network` is not None).
        - Partition column names in `nodes_info_df` are title-cased (e.g., "Walktrap", "Infomap").
        - `clusters_wide_df` columns are ordered by partition algorithm insertion order, then by cluster id.
        """
        if separator is None:
            separator = getattr(self, "default_separator", "; ")

        if column_name not in self.df.columns:
            raise ValueError(f'Column "{column_name}" not in DataFrame.')

        top_items_df = None
        top_items_col = None
        if vector_df is not None and len(vector_df.columns) > 0:
            top_items_df = vector_df
            top_items_col = vector_name_col if vector_name_col is not None else vector_df.columns[0]

        if items_of_interest is None and top_items_df is None and count_func and count_attr:
            count_func()
            top_items_df = getattr(self, count_attr, None)
            top_items_col = column_name

        select_kwargs = {
            "df": self.df,
            "col": column_name,
            "value_type": value_type,
            "indicators": True,
            "items_of_interest": items_of_interest,
            "top_items_df": top_items_df,
            "top_items_col": top_items_col if top_items_col is not None else column_name,
            "top_n": top_n,
            "separator": separator,
        }
        select_kwargs.update(kwargs)

        _, indicators = utilsbib.select_documents(**select_kwargs)

        if not isinstance(indicators, dict) or "binary" not in indicators:
            raise RuntimeError("Expected `indicators['binary']` from select_documents, but it was missing.")

        binary_matrix = indicators["binary"]

        # Cache key for the expensive computation
        cache_key = None
        if use_cache and hasattr(self, '_make_cache_key') and hasattr(self, '_get_cached'):
            items_key = tuple(sorted(binary_matrix.columns)) if len(binary_matrix.columns) < 100 else f"cols:{len(binary_matrix.columns)}"
            cache_key = self._make_cache_key(
                "cooc", column_name, items_key,
                normalization=normalization, network=network
            )
            cached_result = self._instance_cache.get(cache_key) if self._instance_cache else None
            if cached_result is not None:
                co_matrix, co_matrices_norm, co_network, all_measures_df, all_measures_df_T = cached_result
            else:
                co_matrix, co_matrices_norm, co_network, all_measures_df, all_measures_df_T = utilsbib.compute_relation_matrix(
                    binary_matrix,
                    normalization=normalization,
                    network=network
                )
                if self._instance_cache is not None:
                    self._instance_cache[cache_key] = (co_matrix, co_matrices_norm, co_network, all_measures_df, all_measures_df_T)
        else:
            co_matrix, co_matrices_norm, co_network, all_measures_df, all_measures_df_T = utilsbib.compute_relation_matrix(
                binary_matrix,
                normalization=normalization,
                network=network
            )

        if not network:
            co_network = None

        if co_network is not None:
            co_network.remove_edges_from(nx.selfloop_edges(co_network))

        partitions = {}
        if partition_network and (co_network is not None):
            partitions = utilsbib.add_partitions(co_network, **(partition_kwargs or {}))

        # Add numeric vectors as node attributes if vector_df provided
        if co_network is not None and vector_df is not None and top_items_col is not None:
            import pandas as pd
            if not vector_df.empty:
                name_col = top_items_col
                numeric_cols = vector_df.drop(columns=[name_col], errors="ignore").select_dtypes(include="number").columns.tolist()
                if numeric_cols:
                    sub_df = vector_df[[name_col] + numeric_cols].copy()
                    attr_map = {}
                    for row in sub_df.itertuples(index=False):
                        row_dict = dict(zip(sub_df.columns, row))
                        node_name = str(row_dict.pop(name_col))
                        clean_vals = {k: float(v) for k, v in row_dict.items() if pd.notna(v) and (v == v)}
                        if clean_vals:
                            attr_map[node_name] = clean_vals
                    if attr_map:
                        set_attrs = {}
                        for node in co_network.nodes():
                            if str(node) in attr_map:
                                set_attrs[node] = attr_map[str(node)]
                        if set_attrs:
                            nx.set_node_attributes(co_network, set_attrs)

        # Save network if requested
        if (co_network is not None) and (network_filename is not None) and (self.res_folder is not None):
            out_dir = os.path.join(self.res_folder, "networks")
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.join(out_dir, network_filename)
            utilsbib.save_network(
                co_network,
                base,
                formats=network_formats,
                vector_cols=None,
                partition_attr="partition",
            )

        # Relabel nodes to strip trailing " (something)" if present
        if co_network is not None:
            try:
                co_network = nx.relabel_nodes(co_network, lambda x: str(x).split(" (")[0], copy=True)
            except Exception:
                pass

        # Remove isolated nodes copy
        co_network_no_isolated = None
        if co_network is not None:
            co_network_no_isolated = co_network.copy()
            zero_deg = [n for n, d in co_network_no_isolated.degree() if d == 0]
            if zero_deg:
                co_network_no_isolated.remove_nodes_from(zero_deg)

        # (NEW) Build helper DataFrames: nodes_info_df and clusters_wide_df
        nodes_info_df = None
        clusters_wide_df = None
        if co_network is not None:
            import pandas as pd

            # Base nodes frame
            nodes = [str(n) for n in co_network.nodes()]
            nodes_info_df = pd.DataFrame({"Item": nodes})

            # Partition columns (title-cased keys)
            if isinstance(partitions, dict) and partitions:
                for algo_key, mapping in partitions.items():
                    # Expect mapping: node -> cluster_id
                    col_name = str(algo_key).title()
                    # Use .map to align by node name
                    nodes_info_df[col_name] = nodes_info_df["Item"].map({str(k): v for k, v in mapping.items()})

                # Build clusters_wide_df: columns are "<AlgoTitle> <cluster_id>"
                series_dict = {}
                ordered_cols = []

                def _sort_key(
                    cid,
                ):
                    """Sort cluster IDs: integers first (by value), then strings alphabetically."""
                    try:
                        return (0, int(cid))
                    except (TypeError, ValueError):
                        return (1, str(cid))

                for algo_key, mapping in partitions.items():
                    algo_title = str(algo_key).title()
                    # Group nodes by cluster id
                    cluster_to_nodes = {}
                    for node, cid in mapping.items():
                        if cid is None:
                            continue
                        cluster_to_nodes.setdefault(cid, []).append(str(node))
                    # Order cluster ids
                    for cid in sorted(cluster_to_nodes.keys(), key=_sort_key):
                        col = f"{algo_title} {cid}"
                        ordered_cols.append(col)
                        # Keep simple alphabetical order inside each cluster for stability
                        series_dict[col] = pd.Series(sorted(cluster_to_nodes[cid], key=str))

                if series_dict:
                    clusters_wide_df = pd.DataFrame(series_dict)
                    clusters_wide_df = clusters_wide_df[ordered_cols]  # enforce column order

            # Vector columns from vector_df (numeric only), merged on item name
            if vector_df is not None and top_items_col is not None and not vector_df.empty:
                name_col = top_items_col
                numeric_cols = vector_df.drop(columns=[name_col], errors="ignore").select_dtypes(include="number").columns.tolist()
                if numeric_cols:
                    merge_df = vector_df[[name_col] + numeric_cols].copy()
                    merge_df = merge_df.rename(columns={name_col: "Item"})
                    nodes_info_df = nodes_info_df.merge(merge_df, on="Item", how="left")

            if self.res_folder is not None:
                out_dir = os.path.join(self.res_folder, "networks")
                os.makedirs(out_dir, exist_ok=True)
                base_name = network_filename if network_filename else f"{column_name}_cooccurrence"

                nodes_path = os.path.join(out_dir, f"{base_name}_nodes.xlsx")
                # Save nodes (partitions + vectors)
                with pd.ExcelWriter(nodes_path) as writer:
                    nodes_info_df.to_excel(writer, index=False, sheet_name="nodes")

                if clusters_wide_df is not None:
                    clusters_path = os.path.join(out_dir, f"{base_name}_clusters_wide.xlsx")
                    # Save wide cluster membership
                    with pd.ExcelWriter(clusters_path) as writer:
                        clusters_wide_df.to_excel(writer, index=False, sheet_name="clusters_wide")

        # Attach to self if prefix requested
        if output_attr_prefix:
            setattr(self, f"{output_attr_prefix}_cooccurrence_matrix", co_matrix)
            setattr(self, f"{output_attr_prefix}_cooccurrence_matrices_normalized", co_matrices_norm)
            setattr(self, f"{output_attr_prefix}_cooccurrence_network", co_network)
            setattr(self, f"{output_attr_prefix}_cooccurrence_network_no_isolated", co_network_no_isolated)
            setattr(self, f"{output_attr_prefix}_all_cooccurrences", indicators)
            setattr(self, f"{output_attr_prefix}_partitions", partitions)
            # NEW
            setattr(self, f"{output_attr_prefix}_nodes_info_df", nodes_info_df)
            setattr(self, f"{output_attr_prefix}_clusters_wide_df", clusters_wide_df)

        return co_matrix, co_matrices_norm, co_network

    # sepcific coocurences
    def get_author_keyword_cooccurrence(
        self,
        vec_stats=True,
        top_n=20,
        **kwargs,
    ):
        """
        Build author keyword co-occurrence. If available, use stats DF as vector_df
        so all numeric columns become node vectors. vector_name_col is Keyword.
        """
        if not hasattr(self, "author_keywords_stats_df") and vec_stats:
            self.get_author_keywords_stats(top_n=top_n)
        elif not hasattr(self, "author_keywords_counts_df"):
            self.count_author_keywords()

        vector_df = (
            self.author_keywords_stats_df
            if hasattr(self, "author_keywords_stats_df")
            else self.author_keywords_counts_df
        )

        keyword_col = next(
            (col for col in ["Processed Author Keywords", "Author Keywords"] if col in self.df.columns),
            None
        )
        if keyword_col is None:
            raise ValueError("No author keyword column found ('Processed Author Keywords' or 'Author Keywords').")

        self.compute_cooccurrence(
            keyword_col,
            count_func=self.count_author_keywords,
            count_attr="author_keywords_counts_df",
            output_attr_prefix="author_keyword",
            network_filename="author keyword cooccurrence",
            vector_df=vector_df,
            vector_name_col="Keyword",
            top_n=top_n,
            **kwargs
        )

    def get_index_keyword_cooccurrence(
        self,
        vec_stats=True,
        top_n=20,
        **kwargs,
    ):
        """
        Build index keyword co-occurrence. Uses stats DF as vector_df when available.
        Name column for vectors/top items is Keyword.
        """
        if not hasattr(self, "index_keywords_stats_df") and vec_stats:
            self.get_index_keywords_stats(top_n=top_n)
        elif not hasattr(self, "index_keywords_counts_df"):
            self.count_index_keywords()

        vector_df = (
            self.index_keywords_stats_df
            if hasattr(self, "index_keywords_stats_df")
            else self.index_keywords_counts_df
        )

        keyword_col = next(
            (col for col in ["Processed Index Keywords", "Index Keywords"] if col in self.df.columns),
            None
        )
        if keyword_col is None:
            raise ValueError("No index keyword column found ('Processed Index Keywords' or 'Index Keywords').")

        self.compute_cooccurrence(
            keyword_col,
            count_func=self.count_index_keywords,
            count_attr="index_keywords_counts_df",
            output_attr_prefix="ik",
            network_filename="index keyword cooccurrence",
            vector_df=vector_df,
            vector_name_col="Keyword",
            top_n=top_n,
            **kwargs
        )

    def get_ngrams_title_cooccurrence(
        self,
        ngram_range=(1, 2),
        vec_stats=True,
        top_n=20,
        **kwargs,
    ):
        """
        Build title n-grams co-occurrence. Uses stats DF as vector_df when available.
        Name column for vectors/top items is Word-Phrase. Value type is text.
        """
        if not hasattr(self, "ngrams_title_stats_df") and vec_stats:
            self.get_ngrams_title_stats(top_n=top_n)
        elif not hasattr(self, "words_tit_counts_df"):
            self.count_ngrams_title(ngram_range=ngram_range)

        vector_df = (
            self.ngrams_title_stats_df
            if hasattr(self, "ngrams_title_stats_df")
            else self.words_tit_counts_df
        )

        title_col = next((col for col in ["Processed Title", "Title"] if col in self.df.columns), None)
        if title_col is None:
            raise ValueError("No title column found ('Processed Title' or 'Title').")

        self.compute_cooccurrence(
            title_col,
            count_func=self.count_ngrams_title,
            count_attr="words_tit_counts_df",
            output_attr_prefix="ngrams_title",
            value_type="text",
            network_filename="ngrams title cooccurrence",
            vector_df=vector_df,
            vector_name_col="Word - Phrase",
            top_n=top_n,
            **kwargs
        )

    def get_ngrams_abstract_cooccurrence(
        self,
        ngram_range=(1, 2),
        vec_stats=True,
        top_n=20,
        **kwargs,
    ):
        """
        Build abstract n-grams co-occurrence. Uses stats DF as vector_df when available.
        Name column for vectors/top items is Word-Phrase. Value type is text.
        """
        if not hasattr(self, "ngrams_abstract_stats_df") and vec_stats:
            self.get_ngrams_abstract_stats(top_n=top_n)
        elif not hasattr(self, "words_abs_counts_df"):
            self.count_ngrams_abstract(ngram_range=ngram_range)

        vector_df = (
            self.ngrams_abstract_stats_df
            if hasattr(self, "ngrams_abstract_stats_df")
            else self.words_abs_counts_df
        )

        abstract_col = next((col for col in ["Processed Abstract", "Abstract"] if col in self.df.columns), None)
        if abstract_col is None:
            raise ValueError("No abstract column found ('Processed Abstract' or 'Abstract').")

        self.compute_cooccurrence(
            abstract_col,
            count_func=self.count_ngrams_abstract,
            count_attr="words_abs_counts_df",
            output_attr_prefix="ngrams_abstract",
            value_type="text",
            network_filename="ngrams abstract cooccurrence",
            vector_df=vector_df,
            vector_name_col="Word - Phrase",
            top_n=top_n,
            **kwargs
        )

    def get_co_citations(
        self,
        vec_stats=True,
        top_n=20,
        **kwargs,
    ):
        """
        Build co-citation network from the 'References' list column. Uses stats DF as vector_df
        when available. Name column for vectors/top items is Reference. Value type is list.
        """
        if not hasattr(self, "references_stats_df") and vec_stats:
            self.get_references_stats(top_n=top_n)
        elif not hasattr(self, "references_counts_df"):
            self.count_references()

        vector_df = (
            self.references_stats_df
            if hasattr(self, "references_stats_df")
            else self.references_counts_df
        )

        self.compute_cooccurrence(
            "References",
            count_func=self.count_references,
            count_attr="references_counts_df",
            output_attr_prefix="refs",
            value_type="list",
            network_filename="cocitation",
            vector_df=vector_df,
            vector_name_col="Reference",
            top_n=top_n,
            **kwargs
        )
        # Keep your renaming (if you rely on the alias elsewhere)
        utilsbib.rename_attributes(self, {"refs_cooccurrence_network": "co_citation_network"})

    def get_coauthorship(
        self,
        vec_stats=True,
        top_n=20,
        **kwargs,
    ):
        """
        Build co-authorship network from `self.author_var` (list field). Uses stats DF as vector_df
        when available. Name column for vectors/top items is Author ID. Value type is list.
        """
        if not hasattr(self, "authors_stats_df") and vec_stats:
            self.get_authors_stats(top_n=top_n)
        elif not hasattr(self, "authors_counts_df"):
            self.count_authors()

        vector_df = (
            self.authors_stats_df
            if hasattr(self, "authors_stats_df")
            else self.authors_counts_df
        )

        # Determine author column based on database
        author_col_map = {
            "scopus": "Author(s) ID", 
            "oa": "Author",
            "wos": "Author full names",
            "pubmed": "Authors",
            "dimensions": "Authors",
            "lens": "Authors",
        }
        author_col = author_col_map.get(self.db, "Authors")
        
        # Fallback: find first available author column in vector_df
        if author_col not in vector_df.columns:
            for candidate in ["Author(s) ID", "Author", "Authors", "Author full names"]:
                if candidate in vector_df.columns:
                    author_col = candidate
                    break
        self.compute_cooccurrence(
            self.author_var,
            count_func=self.count_authors,
            count_attr="authors_counts_df",
            output_attr_prefix="auth",
            value_type="list",
            network_filename="coauthorship",
            vector_df=vector_df,
            vector_name_col=author_col,
            top_n=top_n,
            **kwargs
        )
        utilsbib.rename_attributes(self, {"auth_cooccurrence_network": "co_authorship_network"})

    def get_country_collaboration_network(
        self,
        vec_stats=True,
        top_n=200,
        region=None,
        **kwargs,
    ):
        """
        Build country collaboration network from Countries of Authors (list field).
        Uses stats DF as vector_df when available. Name column for vectors/top items is Country.
        """
        if not hasattr(self, "all_countries_stats_df") and vec_stats:
            self.get_all_countries_stats(top_n=top_n, region=region)
        elif not hasattr(self, "all_countries_counts_df"):
            self.count_all_countries()
            # --- Minimal addition: region-filter counts DF -----------------------
            if region is not None and hasattr(utilsbib, "df_countries"):
                dfc = utilsbib.df_countries.copy()
                r = str(region).strip()

                if r.lower() == "eu":
                    eu = dfc["EU"]
                    truthy = {"1", "true", "yes", "y", "t"}
                    mask = eu.apply(
                        lambda v: bool(v) if isinstance(v, (bool, int))
                        else (str(v).strip().lower() in truthy) if v is not None
                        else False
                    )
                else:
                    mask = dfc["Continent"].fillna("").str.strip().str.casefold() == r.casefold()

                # Allowed country labels (names + official names + ISO-3)
                allowed = set()
                for col in ("Name", "Official name", "ISO-3"):
                    if col in dfc.columns:
                        allowed.update(dfc.loc[mask, col].dropna().astype(str).str.strip())

                # Apply to counts DF:
                # - if "Country" column exists → filter rows
                # - otherwise → treat as wide format and filter columns
                df_counts = getattr(self, "all_countries_counts_df", None)
                if df_counts is not None:
                    if "Country" in df_counts.columns:
                        self.all_countries_counts_df = (
                            df_counts[df_counts["Country"].astype(str).str.strip().isin(allowed)]
                            .reset_index(drop=True)
                        )
                    else:
                        keep_cols = [c for c in df_counts.columns if str(c).strip() in allowed]
                        # Preserve non-country index/ID columns if present (heuristic: non-numeric col names)
                        # If you prefer strict filtering, drop the next line and keep only keep_cols.
                        if not keep_cols and "Country" in df_counts.columns:
                            # fallback already handled above; keep for safety
                            pass
                        else:
                            self.all_countries_counts_df = df_counts.loc[:, keep_cols]

        vector_df = (
            self.all_countries_stats_df
            if hasattr(self, "all_countries_stats_df")
            else self.all_countries_counts_df
        )

        self.compute_cooccurrence(
            "Countries of Authors",
            count_func=self.count_all_countries,
            count_attr="all_countries_counts_df",
            output_attr_prefix="all_countries",
            value_type="list",
            network_filename="country collaboration",
            vector_df=vector_df,
            vector_name_col="Country",
            top_n=top_n,
            **kwargs
        )

    # citation network of documents

    def build_citation_network(
        self,
        threshold: int=90,
        largest_only: bool=True,
        main_path: bool=True,
        rename: str | None='short',
    ):
        """
        Build a directed citation network of documents and (optionally) compute a main path.

        Parameters
        ----------
        threshold : int, default 90
            Passed to the Scopus routine (pair-matching threshold, etc.). Ignored for OpenAlex.
        largest_only : bool, default True
            Keep only the largest weakly connected component of the network.
        main_path : bool, default True
            If True, compute and store the citation main path from the built network.
        rename : {"short","full",None}, default "short"
            If "short", relabel nodes with `self.id_short_label_dict`;
            if "full" (any non-"short" truthy string), use `self.id_label_dict`;
            if None, keep original node IDs.

        Side effects
        ------------
        Sets:
          - self.citation_network_documents : nx.DiGraph
          - self.citation_main_path         : list (only if `main_path` is True)
        """
        if self.db == "scopus":
            # Build from Scopus export
            self.citation_network_documents, _unmatched = utilsbib.build_citation_network(
                self.df,
                threshold=threshold,
                largest_only=largest_only,
            )
        elif self.db == "oa":
            refs_col = [c for c in self.df.columns if c in ["References", "referenced_works"]][0]
            # Build from OpenAlex: columns "unique-id" (work id) and "referenced_works"
            self.citation_network_documents = utilsbib.build_openalex_citation_network(
                self.df,
                id_col="unique-id",
                refs_col=refs_col,
                sep=self.default_separator,
                normalize="short",               # or "url"/"auto"
                keep_external=False,             # keep only in-dataset citations
                drop_self_loops=True,
                deduplicate=True,
                keep_largest_component=largest_only,
                path_base=None,
                save_formats=("csv", "pajek"),
            )
        else:
            raise ValueError(f"Unsupported database: {self.db!r}")

        # Compute main path if requested
        if main_path:
            self.citation_main_path = utilsbib.compute_main_path(self.citation_network_documents)
            if self.db == "oa":
                self.citation_main_path_df = readbib.harvest_openalex_links_to_df(self.citation_main_path)
                self.citation_main_path_df=readbib.add_short_and_full_labels(self.citation_main_path_df)
                self.citation_main_path_label_dict = self.citation_main_path_df.set_index("input").to_dict()["Short Label"]

        # Optional relabeling
        if rename is not None:
            if rename == "short":
                mapping = getattr(self, "id_short_label_dict", {})
            else:
                mapping = getattr(self, "id_label_dict", {})
            if mapping:
                self.citation_network_documents = nx.relabel_nodes(self.citation_network_documents, mapping, copy=True)
                if main_path and getattr(self, "citation_main_path", None):
                    self.citation_main_path = [mapping.get(doc, doc) for doc in self.citation_main_path]

    def build_historiograph(
        self,
        title_col='Title',
        year_col='Year',
        refs_col='References',
        cutoff=0.85,
        label_col='Document Short Label',
        filename='historiograph',
    ):
        """
        Build a historiograph (citation network over time) for the dataset.
        
        Delegates to `utilsbib.build_historiograph` to construct a directed
        network of citations among the documents in `self.df`, usually ordered
        by publication year. The resulting network object is stored as
        `self.historiograph` and can also be exported to network files
        (Pajek, etc.) when supported by the helper function.
        
        Parameters
        ----------
        title_col : str, default "Title"
            Column used to identify documents.
        year_col : str, default "Year"
            Column with publication years.
        refs_col : str, default "References"
            Column with reference strings.
        cutoff : float, default 0.85
            Similarity threshold for linking documents when titles are matched
            fuzzily.
        label_col : str, default "Document Short Label"
            Column used for concise node labels.
        filename : str, default "historiograph"
            Base filename for exported network files.
        """
        filename = os.path.join(self.res_folder, "networks", filename)
        self.historiograph = utilsbib.build_historiograph(self.df, title_col=title_col,
                                     year_col=year_col, refs_col=refs_col,
                                     cutoff=cutoff, label_col=label_col,
                                     save_path=filename)

    # analysis of relationships

    def _resolve_binary_key(
        self,
        concept: str,
        preferred: str='binary dataframe',
    ) -> str:
        """
        Return the first existing binary-matrix key for `concept` from `self.mapping`.
        Tries `preferred`, then "binary indicators", then "binary dataframe".
        """
        if not hasattr(self, "mapping"):
            raise AttributeError("self.mapping is not defined.")
        if concept not in self.mapping:
            raise KeyError(f'Concept "{concept}" not found in self.mapping.')
        for k in (preferred, "binary indicators", "binary dataframe"):
            if k in self.mapping[concept]:
                return k
        raise KeyError(f'No binary-matrix key for "{concept}".')

    def _ensure_binary_matrix_for_concept(
        self,
        concept: str,
        *,
        binary_key: str='binary dataframe',
        counter_key: str='counter',
        top_n: int=10,
    ) -> pd.DataFrame:
        """
        Ensure a binary matrix for `concept` exists on `self`. If missing, compute it
        by calling the mapped counter with `top_n`.
        """
        bin_key = self._resolve_binary_key(concept, binary_key)
        attr_name = self.mapping[concept][bin_key]
        if not isinstance(attr_name, str):
            raise TypeError(f'Expected string attribute name for "{concept}" under "{bin_key}".')

        df = getattr(self, attr_name, None)
        if isinstance(df, pd.DataFrame):
            return df

        if counter_key not in self.mapping[concept]:
            raise KeyError(f'Missing "{counter_key}" for concept "{concept}".')

        counter_ref = self.mapping[concept][counter_key]
        counter_fn = getattr(self, counter_ref) if isinstance(counter_ref, str) else counter_ref
        if not callable(counter_fn):
            raise AttributeError(f'Counter for "{concept}" is not callable: {counter_ref!r}.')

        try:
            ret = counter_fn(top_n=top_n)
        except TypeError:
            ret = counter_fn()

        df_after = getattr(self, attr_name, None)
        if isinstance(df_after, pd.DataFrame):
            return df_after
        if isinstance(ret, (list, tuple)) and ret and isinstance(ret[0], pd.DataFrame):
            setattr(self, attr_name, ret[0])
            return ret[0]
        if isinstance(ret, pd.DataFrame):
            setattr(self, attr_name, ret)
            return ret

        raise TypeError(
            f'Could not obtain DataFrame for "{concept}". Attribute "{attr_name}" not set '
            f'and counter did not return a DataFrame.'
        )

    def _known_matrices_from_self_mapping(
        self,
        concepts: Iterable[str],
        *,
        binary_key: str='binary dataframe',
        counter_key: str='counter',
        top_n: int=10,
    ) -> dict[str, pd.DataFrame]:
        """
        Build concept-DataFrame dict by ensuring/creating each concepts binary matrix.
        """
        return {
            c: self._ensure_binary_matrix_for_concept(
                c, binary_key=binary_key, counter_key=counter_key, top_n=top_n
            )
            for c in concepts
        }

    def _safe_name(
        self,
        s: str,
    ) -> str:
        """
        Make a string safe for filenames.
        """
        return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in s.strip().replace(" ", "_"))

    def _as_df(
        self,
        obj,
        name: str='value',
    ) -> pd.DataFrame | None:
        """
        Convert common structures to DataFrame if possible; otherwise return None.
        """
        if obj is None:
            return None
        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, pd.Series):
            return obj.to_frame(name if obj.name is None else obj.name)
        if isinstance(obj, Mapping):
            try:
                return pd.DataFrame([obj])
            except Exception:
                return None
        if isinstance(obj, (list, tuple)):
            try:
                return pd.DataFrame({name: list(obj)})
            except Exception:
                return None
        if isinstance(obj, (int, float, str, bool)):
            return pd.DataFrame({name: [obj]})
        return None

    def _write_group_xlsx(
        self,
        base_dir: str,
        c1: str,
        c2: str,
        group: str,
        sheets: dict[str, pd.DataFrame],
    ) -> None:
        """
        Write a multi-sheet XLSX file for a result group.
        """
        sheets = {k: v for k, v in sheets.items() if isinstance(v, pd.DataFrame)}
        if not sheets:
            return
        os.makedirs(base_dir, exist_ok=True)
        fn = f"{self._safe_name(c1)}__{self._safe_name(c2)}__{self._safe_name(group)}.xlsx"
        path = os.path.join(base_dir, fn)
        with pd.ExcelWriter(path) as writer:
            for sheet_name, df in sheets.items():
                writer_sheet = (sheet_name[:31]) or "Sheet"
                df.to_excel(writer, sheet_name=writer_sheet, index=True)

    def _save_relation_results(
        self,
        R,
    ) -> None:
        """
        Save DataFrame-like attributes of a Relation to grouped XLSX files under
        `self.res_folder/relations`.
        """
        base_dir = getattr(self, "res_folder", None)
        if not base_dir:
            return
        out_dir = os.path.join(base_dir, "relations")
        c1, c2 = R.concept1, R.concept2

        # Matrix
        self._write_group_xlsx(out_dir, c1, c2, "matrix", {"matrix": self._as_df(R.rm, "value")})

        # Diversity
        self._write_group_xlsx(out_dir, c1, c2, "diversity", {
            "row_metrics": self._as_df(R.diversity_row_metrics),
            "column_metrics": self._as_df(R.diversity_column_metrics),
        })

        # Bipartite
        self._write_group_xlsx(out_dir, c1, c2, "bipartite", {
            "row_stats": self._as_df(R.bipartite_row_stats),
            "column_stats": self._as_df(R.bipartite_column_stats),
            "global_stats": self._as_df(R.bipartite_global_stats),
            "row_global": self._as_df(R.bipartite_row_global),
            "column_global": self._as_df(R.bipartite_column_global),
        })

        # Clustering
        self._write_group_xlsx(out_dir, c1, c2, "cluster", {
            "clusters": self._as_df(R.clusters, "cluster"),
            "silhouette_scores": self._as_df(R.silhouette_scores, "silhouette"),
            "n_clusters": self._as_df(R.n_clusters, "n_clusters"),
        })

        # Biclustering
        self._write_group_xlsx(out_dir, c1, c2, "bicluster", {
            "row_clusters": self._as_df(R.bicluster_row_clusters, "cluster"),
            "column_clusters": self._as_df(R.bicluster_column_clusters, "cluster"),
        })

        # Correspondence
        self._write_group_xlsx(out_dir, c1, c2, "correspondence", {
            "row_coords": self._as_df(R.ca_row_coords),
            "col_coords": self._as_df(R.ca_col_coords),
            "explained_inertia": self._as_df(
                None if R.ca_explained_inertia is None
                else [{"component": i + 1, "explained_inertia": v} for i, v in enumerate(R.ca_explained_inertia)]
            ),
        })

        # Chi-square
        self._write_group_xlsx(out_dir, c1, c2, "chi2", {
            "expected": self._as_df(R.chi2_expected_df),
            "residuals": self._as_df(R.chi2_residuals_df),
            "sorted_residuals": self._as_df(R.chi2_sorted_residuals),
            "summary": self._as_df(
                {"chi2_stat": R.chi2_chi2_stat, "dof": R.chi2_dof} if R.chi2_chi2_stat is not None else None
            ),
        })

        # SVD
        self._write_group_xlsx(out_dir, c1, c2, "svd", {
            "row_projection": self._as_df(R.svd_row_projection),
            "singular_values": self._as_df(R.svd_singular_values, "singular_value"),
            "explained_variance": self._as_df(R.svd_explained_variance, "explained_variance"),
        })

        # Log-ratio
        self._write_group_xlsx(out_dir, c1, c2, "log_ratio", {
            "log_ratio": self._as_df(R.log_ratio_df),
            "expected": self._as_df(R.log_ratio_expected_df),
            "sorted_log_ratios": self._as_df(R.log_ratio_sorted_log_ratios),
        })

    # --- main entry (prefers self.*; falls back to utilsbib) ----------------------

    def relate_concepts(
        self,
        concept1: str,
        concept2: str,
        *,
        custom_matrices: dict[str, 'pd.DataFrame'] | None=None,
        binary_key: str='binary dataframe',
        counter_key: str='counter',
        top_n: int=10,
        **kwargs,
    ):
        """
        Relate two concepts using matrices from `custom_matrices` (priority) and/or `self.mapping`.

        Custom-first logic
        ------------------
        - If a concept is present in `custom_matrices`, it OVERRIDES any mapped/known matrix.
        - Known matrices are only fetched/computed for concepts that are *not* provided as custom.
        - To avoid downstream crashes, we pass **only one** matrices dict to
          `relate_concepts_general`: if any custom exists, we pass the merged dict as `custom_matrices`;
          otherwise, we pass the known dict as `known_matrices`.

        Returns
        -------
        pd.DataFrame
            The relation result from `relate_concepts_general`.

        Examples
        --------
        R = ba.relate_concepts("Author Keywords", "Index Keywords")
        R = ba.relate_concepts("Author Keywords", "X", custom_matrices={"X": x_df})
        R = ba.relate_concepts("A", "B", custom_matrices={"A": A_df, "B": B_df})
        """
        import pandas as pd

        # 1) Collect customs for the requested concepts (customs override known)
        custom: dict[str, pd.DataFrame] = {}
        if custom_matrices:
            for c in (concept1, concept2):
                if c in custom_matrices:
                    m = custom_matrices[c]
                    if not isinstance(m, pd.DataFrame):
                        raise TypeError(f'custom_matrices["{c}"] must be a pandas DataFrame.')
                    custom[c] = m

        # 2) Collect known ONLY for concepts that are NOT custom
        known: dict[str, pd.DataFrame] = {}
        mapping = getattr(self, "mapping", None)
        if isinstance(mapping, dict):
            for c in (concept1, concept2):
                if c not in custom and c in mapping:
                    km = self._known_matrices_from_self_mapping(
                        (c,), binary_key=binary_key, counter_key=counter_key, top_n=top_n
                    ).get(c)
                    if isinstance(km, pd.DataFrame):
                        known[c] = km

        # 3) Validate both concepts resolvable from either source
        missing = [c for c in (concept1, concept2) if c not in custom and c not in known]
        if missing:
            raise KeyError(f"Missing matrices for {missing}. Provide via custom_matrices or self.mapping.")

        # 4) Merge (custom overrides known); choose a single argument style for downstream
        combined = {**known, **custom}
        use_known = not bool(custom)  # if any custom present, pass as custom only

        # 5) Ensure relations store
        if getattr(self, "relations", None) is None:
            self.relations = {}

        # 6) Select implementation
        relate_fn = getattr(self, "relate_concepts_general", None)
        if not callable(relate_fn):
            from biblium import utilsbib
            relate_fn = utilsbib.relate_concepts_general

        # 7) Dispatch: only one matrices kwarg set to avoid mixed-kwargs crash
        R, _rm = relate_fn(
            concept1,
            concept2,
            known_matrices=(combined if use_known else None),
            custom_matrices=(combined if not use_known else None),
            relations=self.relations,
            **kwargs,
        )

        # 8) Symmetric storage and optional save
        self.relations.setdefault(concept1, {})[concept2] = R
        self.relations.setdefault(concept2, {})[concept1] = R
        if hasattr(self, "_save_relation_results"):
            self._save_relation_results(R)

        return R
    

    def relate_term_sets(
        self,
        terms1,
        terms2,
        *,
        concept1: str = "Author Keywords",
        concept2: str | None = None,
        custom_matrices: dict[str, "pd.DataFrame"] | None = None,  # kept for compatibility, unused
        binary_key: str = "binary dataframe",                      # unused
        counter_key: str = "counter",                              # unused
        top_n: int = 10,                                           # unused
        regex: bool = False,
        use_processed: bool = True,
        min_count: int = 1,
        **kwargs,
    ):
        """
        Relate two sets of terms (keywords or words) directly from the dataframe.
    
        This function computes co-occurrence of two user-specified term sets
        across documents, without relying on precomputed concept matrices.
    
        It supports:
          - Keywords from "Author Keywords" or "Index Keywords"
            (separated by ``self.default_separator``).
          - Words from "Title" or "Abstract".
          - Optional use of processed columns, if available:
            "Processed Author Keywords", "Processed Index Keywords",
            "Processed Title", "Processed Abstract".
          - Exact, case-insensitive matching (default) or regular expressions.
    
        Parameters
        ----------
        terms1 :
            First set of terms. A string separated by ``self.default_separator``
            or an iterable of strings.
        terms2 :
            Second set of terms. Same format as ``terms1``.
        concept1 : str, default "Author Keywords"
            First concept: one of "Author Keywords", "Index Keywords",
            "Title", "Abstract" (or a column name, if you know what you’re doing).
        concept2 : str or None, default None
            Second concept. If ``None``, uses the same as ``concept1``.
        regex : bool, default False
            If True, each term is treated as a regular expression and matched
            with ``re.search(..., re.IGNORECASE)`` against tokens.
            If False, matching is case-insensitive exact equality.
        use_processed : bool, default True
            If True and the corresponding processed column exists, it is used.
            Otherwise, the original column is used.
        min_count : int, default 1
            Minimum number of co-occurring documents for a pair to be kept
            in the result.
        custom_matrices, binary_key, counter_key, top_n, **kwargs :
            Present only for signature compatibility with other methods;
            they are ignored here.
    
        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by a MultiIndex (term1, term2), with one column
            "Number of documents" giving the number of documents where the
            two terms co-occur (at least once each in the respective concept).
        """
        import re
        import pandas as pd
        from collections import Counter
    
        if concept2 is None:
            concept2 = concept1
    
        sep = getattr(self, "default_separator", "; ")
    
        # ---- term parsing -------------------------------------------------
        def _parse_terms(raw):
            if raw is None:
                return []
            if isinstance(raw, str):
                parts = [p.strip() for p in raw.split(sep)]
            else:
                parts = [str(p).strip() for p in raw]
            return [p for p in parts if p]
    
        patterns1 = _parse_terms(terms1)
        patterns2 = _parse_terms(terms2)
    
        if not patterns1:
            raise ValueError("terms1 is empty after parsing.")
        if not patterns2:
            raise ValueError("terms2 is empty after parsing.")
    
        # ---- column / concept resolution ----------------------------------
        def _resolve_concept_column(concept: str) -> str:
            processed_map = {
                "Author Keywords": "Processed Author Keywords",
                "Index Keywords": "Processed Index Keywords",
                "Title": "Processed Title",
                "Abstract": "Processed Abstract",
            }
            raw_map = {
                "Author Keywords": "Author Keywords",
                "Index Keywords": "Index Keywords",
                "Title": "Title",
                "Abstract": "Abstract",
            }
    
            # Prefer processed if requested and present
            if use_processed:
                pcol = processed_map.get(concept)
                if pcol and pcol in self.df.columns:
                    return pcol
    
            # Fallback: raw column (or concept name as column name)
            col = raw_map.get(concept, concept)
            if col in self.df.columns:
                return col
    
            raise KeyError(f'Cannot resolve data column for concept "{concept}".')
    
        col1 = _resolve_concept_column(concept1)
        col2 = _resolve_concept_column(concept2)
    
        s1 = self.df[col1].fillna("")
        s2 = self.df[col2].fillna("")
    
        # ---- tokenization -------------------------------------------------
        def _tokenize(series: "pd.Series", concept: str):
            """Split series into lists of tokens per document."""
            # Keywords: use default separator
            if concept in ("Author Keywords", "Index Keywords"):
                return series.astype(str).str.split(sep)
            # Title / Abstract or other text: split on whitespace
            return series.astype(str).str.split()
    
        tokens1 = _tokenize(s1, concept1)
        tokens2 = _tokenize(s2, concept2)
    
        # ---- per-document pattern matching --------------------------------
        if regex:
            rx1 = [re.compile(p, re.IGNORECASE) for p in patterns1]
            rx2 = [re.compile(p, re.IGNORECASE) for p in patterns2]
    
            def _matches(tokens, rx_list):
                toks = [t for t in tokens if t]
                return [any(rx.search(tok) for tok in toks) for rx in rx_list]
    
        else:
            pats1_norm = [p.lower() for p in patterns1]
            pats2_norm = [p.lower() for p in patterns2]
    
            def _matches(tokens, pats_norm):
                toks = {t.lower() for t in tokens if t}
                return [p in toks for p in pats_norm]
    
        counts = Counter()
    
        for doc_tokens1, doc_tokens2 in zip(tokens1, tokens2):
            if not isinstance(doc_tokens1, list):
                doc_tokens1 = []
            if not isinstance(doc_tokens2, list):
                doc_tokens2 = []
    
            flags1 = _matches(doc_tokens1, rx1 if regex else pats1_norm)
            flags2 = _matches(doc_tokens2, rx2 if regex else pats2_norm)
    
            present1 = [patterns1[i] for i, flag in enumerate(flags1) if flag]
            present2 = [patterns2[i] for i, flag in enumerate(flags2) if flag]
    
            if not present1 or not present2:
                continue
    
            # Each present term1 x each present term2 gets +1 for this document
            for p1 in present1:
                for p2 in present2:
                    counts[(p1, p2)] += 1
    
        # ---- build result DataFrame ---------------------------------------
        if not counts:
            mi = pd.MultiIndex.from_arrays([[], []], names=[concept1, concept2])
            return pd.DataFrame({"Number of documents": []}, index=mi)
    
        rows = [
            (p1, p2, n)
            for (p1, p2), n in counts.items()
            if n >= min_count
        ]
        # sort: highest count first, then alphabetical
        rows.sort(key=lambda x: (-x[2], x[0].lower(), x[1].lower()))
    
        idx = pd.MultiIndex.from_tuples(
            [(r[0], r[1]) for r in rows],
            names=[concept1, concept2],
        )
        values = [r[2] for r in rows]
    
        return pd.DataFrame({"Number of documents": values}, index=idx)

    def plot_associations_top_n_pairs(
        self,
        items,
        filename_base: str='top_n_pairs_associations',
        **kwds,
    ):
        """
        Plot associations of top-N item pairs, resolving association objects via getattr(self, ...).

        Notes
        -----
        - In `self.mapping`, the value under ["associations"] is an attribute *name* (string).
          We fetch the actual object with getattr(self, ...).
        - Saves under `<res_folder>/relations/` if `filename_base` is not None.

        Parameters
        ----------
        items : str
            Mapping key (e.g., "keywords", "sources").
        filename_base : str, default "top_n_pairs_associations"
            Base filename (no extension). Saved to `<res_folder>/relations/<filename_base>_<items>`.
        **kwds :
            Forwarded to `plotbib.plot_top_n_pairs`.
            If not provided, defaults are:
                metric_column="residual", size_column="observed".
            User "x_label" and "y_label" override defaults ("groups", items).
        """
        import os
        import pandas as pd
        from biblium import plotbib

        # Resolve association object from attribute name (mapping stores strings)
        assoc_attr = None
        if hasattr(self, "mapping") and items in self.mapping:
            assoc_attr = self.mapping[items].get("associations")
        assoc_attr = assoc_attr or f"{items}_associations"

        assoc = getattr(self, assoc_attr, None)
        if assoc is None:
            raise AttributeError(f"Association object '{assoc_attr}' not found on self.")

        pairs = getattr(assoc, "chi2_sorted_residuals", None)
        if pairs is None:
            raise AttributeError(f"'{assoc_attr}' lacks 'chi2_sorted_residuals'.")

        # Convert to DataFrame if needed (expected columns: row, col, residual, observed, expected)
        if isinstance(pairs, list):
            if len(pairs) and not isinstance(pairs[0], (list, tuple)):
                raise TypeError("chi2_sorted_residuals must be a list of tuples like (row, col, residual, observed, expected).")
            sorted_pairs_df = pd.DataFrame(pairs, columns=["row", "col", "residual", "observed", "expected"])
        elif hasattr(pairs, "shape"):
            sorted_pairs_df = pairs  # already a DataFrame-like
        else:
            raise TypeError("Unsupported type for chi2_sorted_residuals; expected list of tuples or DataFrame.")

        # Prepare filename in relations/
        if filename_base is not None and getattr(self, "res_folder", None):
            filename_base = os.path.join(self.res_folder, "relations", f"{filename_base}_{items}")

        # Labels (axis titles)
        x_label = kwds.pop("x_label", "groups")
        y_label = kwds.pop("y_label", items)

        # Defaults for plotbib if caller didn't provide them
        kwds.setdefault("metric_column", "residual")
        kwds.setdefault("size_column", "observed")

        plotbib.plot_top_n_pairs(
            sorted_pairs_df,
            filename_base=filename_base,
            x_label=x_label,
            y_label=y_label,
            **kwds,
        )

    def plot_associations_correspondence_analysis(
        self,
        items,
        filename_base: str='ca_associations',
        **kwds,
    ):
        """
        Plot correspondence analysis of item associations, resolving objects via getattr(self, ...).

        Notes
        -----
        - In `self.mapping`, the values under ["associations"] and ["contingency"] are
          attribute *names* (strings). We fetch the actual objects with getattr(self, ...).
        - Saves under `<res_folder>/relations/` if `filename_base` is not None.
        """
        import os
        from biblium import plotbib

        # Resolve attributes from mapping (fallback to conventional names)
        assoc_attr = cont_attr = None
        if hasattr(self, "mapping") and items in self.mapping:
            assoc_attr = self.mapping[items].get("associations")
            cont_attr = self.mapping[items].get("contingency")
        assoc_attr = assoc_attr or f"{items}_associations"
        cont_attr = cont_attr or f"{items}_contingency"

        assoc = getattr(self, assoc_attr, None)
        cont = getattr(self, cont_attr, None)

        need = ("ca_row_coords", "ca_col_coords", "ca_explained_inertia")
        if assoc is None or not all(hasattr(assoc, a) for a in need):
            raise AttributeError(
                f"Association object '{assoc_attr}' missing or lacks required attributes: {need}."
            )
        if cont is None:
            raise AttributeError(f"Contingency matrix '{cont_attr}' missing on self.")

        # Prepare filename in relations/
        if filename_base is not None and getattr(self, "res_folder", None):
            filename_base = os.path.join(self.res_folder, "relations", f"{filename_base}_{items}")

        # Labels (axis titles)
        row_label_name = kwds.pop("row_label_name", "groups")
        col_label_name = kwds.pop("col_label_name", items)

        plotbib.plot_correspondence_analysis(
            assoc.ca_row_coords,
            assoc.ca_col_coords,
            assoc.ca_explained_inertia,
            cont,
            filename_base=filename_base,
            row_label_name=row_label_name,
            col_label_name=col_label_name,
            **kwds,
        )

    def cluster_items(
        self,
        items,
        cluster_by,
        **kwargs,
    ):
        """
        Cluster items by their co-occurrence patterns.
        
        A thin wrapper around `relate_concepts` that requests both cluster and
        bicluster statistics for the selected items. Useful for quickly
        partitioning keywords, authors or other entities based on how they
        co-occur.
        
        Parameters
        ----------
        items : str or list[str]
            Column name(s) representing the items to be clustered.
        cluster_by : str
            Column indicating the context in which items co-occur (for example,
            a document identifier or another categorical variable).
        **kwargs :
            Additional options forwarded to `relate_concepts`.
        """
        self.relate_concepts(items, cluster_by, include_stats=["cluster", "bicluster"],
                             clean_zeros=True, to_self=True, **kwargs)

    def cluster_documents(
        self,
        text_field: str='Processed Abstract',
        method: str='kmeans',
        n_clusters: int | None=None,
        k_range: range=range(2, 11),
        coupling_fields: list[str] | str | None=None,
        **vectorize_kwargs,
    ) -> None:
        """
        Wrapper around utilsbib.cluster_documents that SAVES results as attributes (no return).

        Saves
        -----
        - self.df[col]              : in-place labels (col = f"{method}_cluster")
        - self.doc_clusters_df      : labels DataFrame
        - self.cluster_matrix_df    : representation matrix (DTM or similarity)
        - self.last_cluster_method  : method used
        - self.last_cluster_column  : label column name
        Also writes {self.res_folder}/tables/{method}_clusters.xlsx if self.res_folder is set.

        Notes
        -----
        This replaces the old tuple-unpacking pattern. Do not assign the return value.
        """
        import pandas as pd  # safe import

        # compute (utils returns two values: df_out, matrix_df)
        df_out, matrix_df = utilsbib.cluster_documents(
            self.df,
            text_field=text_field,
            method=method,
            n_clusters=n_clusters,
            k_range=k_range,
            coupling_fields=coupling_fields,
            **vectorize_kwargs,
        )

        col = f"{method}_cluster"

        # save on the object
        self.df[col] = df_out[col]
        self.doc_clusters_df = df_out[[col]]
        self.cluster_matrix_df = matrix_df
        self.last_cluster_method = method
        self.last_cluster_column = col

        # optional on-disk save; assumes {res_folder}/tables exists
        if getattr(self, "res_folder", None):
            try:
                out_path = os.path.join(self.res_folder, "tables", f"{method}_clusters.xlsx")
                sizes = (
                    df_out[col]
                    .value_counts()
                    .sort_index()
                    .rename("Count")
                    .to_frame()
                )
                with pd.ExcelWriter(out_path, engine="xlsxwriter") as xlw:
                    df_out[[col]].to_excel(xlw, sheet_name="labels")
                    sizes.to_excel(xlw, sheet_name="sizes")
            except Exception:
                # keep silent; file writing is best-effort
                pass

        return None

    def cluster_entities(
        self,
        entity_column: str,
        *,
        method: str = 'kmeans',
        n_clusters: int | None = None,
        k_range: range = range(2, 11),
        features: str = 'cooccurrence',
        feature_column: str | None = None,
        min_occurrence: int = 2,
        scorer: str = 'silhouette',
        linkage_method: str = 'ward',
    ) -> dict:
        """
        Cluster entities (authors, sources, keywords, etc.) and save results.
        
        Parameters
        ----------
        entity_column : str
            Column containing entities to cluster (e.g., "Authors", "Author Keywords").
        method : str, default="kmeans"
            Clustering method: "kmeans", "hierarchical", "spectral".
        n_clusters : int or None
            Number of clusters. Auto-selected if None.
        k_range : range
            Range for auto k-selection (kmeans).
        features : str, default="cooccurrence"
            Feature type: "cooccurrence" or "documents".
        feature_column : str or None
            Column for measuring co-occurrence (if different from entity_column).
        min_occurrence : int, default=2
            Minimum occurrences to include an entity.
        scorer : str
            Metric for auto k-selection.
        linkage_method : str
            Linkage for hierarchical clustering.
        
        Returns
        -------
        dict with clustering results including clusters_df, n_clusters, etc.
        
        Side Effects
        ------------
        Saves results to self.entity_clusters_result and optionally to Excel.
        """
        result = utilsbib.cluster_entities(
            self.df,
            entity_column=entity_column,
            method=method,
            n_clusters=n_clusters,
            k_range=k_range,
            features=features,
            feature_column=feature_column,
            sep=self.default_separator,
            min_occurrence=min_occurrence,
            scorer=scorer,
            linkage_method=linkage_method,
        )
        
        # Store results
        self.entity_clusters_result = result
        self.entity_clusters_df = result["clusters_df"]
        self.last_entity_cluster_column = entity_column
        
        # Save to file if res_folder is set
        if getattr(self, "res_folder", None):
            try:
                col_safe = entity_column.replace(" ", "_").lower()
                out_path = os.path.join(self.res_folder, "tables", f"entity_clusters_{col_safe}.xlsx")
                with pd.ExcelWriter(out_path, engine="xlsxwriter") as xlw:
                    result["clusters_df"].to_excel(xlw, sheet_name="Clusters", index=False)
                    result["cluster_sizes"].to_frame("Size").to_excel(xlw, sheet_name="Sizes")
                    # Top entities
                    top_df = pd.DataFrame([
                        {"Cluster": k, "Top Entities": ", ".join(v[:5])}
                        for k, v in result["top_entities_by_cluster"].items()
                    ])
                    top_df.to_excel(xlw, sheet_name="Top Entities", index=False)
            except Exception:
                pass
        
        return result

    def extract_repository_links(
        self,
        use_text_mining: bool = True,
        use_datacite: bool = False,
        use_paperswithcode: bool = False,
        text_columns: list[str] | None = None,
        max_api_requests: int = 50,
    ) -> dict:
        """
        Extract data/code repository links from publications.
        
        Combines multiple methods:
        - Text mining: Extract URLs from abstracts/titles
        - DataCite API: Find related datasets
        - Papers With Code API: Find code repositories
        
        Parameters
        ----------
        use_text_mining : bool, default=True
            Extract links from text columns using regex patterns.
        use_datacite : bool, default=False
            Query DataCite API for related datasets (slower).
        use_paperswithcode : bool, default=False
            Query Papers With Code API for code links (slower).
        text_columns : list of str, optional
            Columns to search for text mining.
        max_api_requests : int, default=50
            Maximum API requests per external source.
        
        Returns
        -------
        dict with keys:
            - stats: Summary statistics
            - links_df: DataFrame with all extracted links
        
        Side Effects
        ------------
        Adds columns to self.df:
            - Repository Links, Has Code Link, Has Data Link
            - Code Repositories, Data Repositories, N Repository Links
        """
        self.df, stats = utilsbib.enrich_with_repository_links(
            self.df,
            use_text_mining=use_text_mining,
            use_datacite=use_datacite,
            use_paperswithcode=use_paperswithcode,
            doi_column=self.mapping.get("DOI", "DOI"),
            title_column=self.mapping.get("Title", "Title"),
            text_columns=text_columns,
            max_api_requests=max_api_requests,
        )
        
        # Store results
        self.repository_links_stats = stats
        
        # Create links DataFrame
        links_data = []
        for idx, row in self.df.iterrows():
            links_str = row.get("Repository Links", "")
            if links_str:
                for url in links_str.split("; "):
                    if url:
                        # Determine type from URL
                        link_type = "unknown"
                        repo_name = "Unknown"
                        for repo_key, repo_info in utilsbib.REPOSITORY_PATTERNS.items():
                            if re.search(repo_info["pattern"], url, re.IGNORECASE):
                                link_type = repo_info["type"]
                                repo_name = repo_info["name"]
                                break
                        
                        links_data.append({
                            "Doc ID": row.get("Doc ID", idx),
                            "Title": row.get(self.mapping.get("Title", "Title"), "")[:80],
                            "URL": url,
                            "Repository": repo_name,
                            "Type": link_type,
                        })
        
        self.repository_links_df = pd.DataFrame(links_data)
        
        # Save to file if res_folder is set
        if getattr(self, "res_folder", None):
            try:
                out_path = os.path.join(self.res_folder, "tables", "repository_links.xlsx")
                with pd.ExcelWriter(out_path, engine="xlsxwriter") as xlw:
                    self.repository_links_df.to_excel(xlw, sheet_name="Links", index=False)
                    
                    # Summary
                    summary_df = pd.DataFrame([stats])
                    summary_df.to_excel(xlw, sheet_name="Summary", index=False)
            except Exception:
                pass
        
        return {
            "stats": stats,
            "links_df": self.repository_links_df,
        }

    # topic modelling

    def get_topics(
        self,
        v: str='Processed Abstract',
        *,
        model_type: str='LDA',
        n_topics: int | None=None,
        max_topics: int=10,
        max_features: int=5000,
        stop_words: str | list[str] | None='english',
        weight_col: str | None=None,
        save_attrs: bool=True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run `utilsbib.topic_modeling` and return `(df_out, topics_df)` — minimal, no helpers.

        What this does
        --------------
        1) Picks the text column (`v`, else "Abstract").
        2) Calls the *existing* `utilsbib.topic_modeling` with only its core arguments.
        3) Remembers what was computed and passes it forward unchanged:
           - `df_out` keeps ALL original columns + a hard "Topic" assignment added by the modeler.
           - If the modeler already produced per-document topic weights (e.g., "Topic 1 Weight", ...),
             they remain in `df_out` and can be used downstream without extra work.
           - `topics_df` is kept as returned; if it uses "Word", it is renamed to "Term" for consistency.
        4) Optionally exposes a stable per-document weight alias:
           - If `weight_col` is provided and exists, creates "Doc Weight" = numeric(weight_col, NaN→1).

        Assumptions
        -----------
        - "Doc ID" already exists in `self.df`.
        - `utilsbib.topic_modeling` returns `(df_out, topics_df)`.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            df_out : all original columns (+ "Topic", and any topic-weight cols if the modeler produced them)
            topics_df : columns ["Topic", "Term", "Weight"] (renamed from "Word" if needed)
        """
        if "Doc ID" not in self.df.columns:
            raise KeyError('Expected "Doc ID" in self.df.columns.')

        col = v if v in self.df.columns else ("Abstract" if "Abstract" in self.df.columns else None)
        if col is None:
            raise KeyError(f'Neither "{v}" nor "Abstract" found in self.df.columns.')

        # Call ONLY with the base args the legacy/new function definitely supports.
        df_out, topics_df = utilsbib.topic_modeling(
            df=self.df,
            text_column=col,
            model_type=model_type,
            n_topics=n_topics,
            max_topics=max_topics,
            max_features=max_features,
            stop_words=stop_words,
        )

        # Normalize topics_df column name if older code returns "Word"
        if isinstance(topics_df, pd.DataFrame) and "Term" not in topics_df.columns and "Word" in topics_df.columns:
            topics_df = topics_df.rename(columns={"Word": "Term"})

        # Optional: stable alias for a per-document weight already present in df_out
        if weight_col is not None and weight_col in df_out.columns:
            df_out["Doc Weight"] = pd.to_numeric(df_out[weight_col], errors="coerce").fillna(1.0)

        if save_attrs:
            self.topic_assignment_df = df_out.copy()
            self.topics_df = topics_df
            # If the modeler produced per-topic weight columns, remember their names for later use
            self.topic_weight_cols = [c for c in df_out.columns if c.startswith("Topic ") and c.endswith(" Weight")]

        self._save_table(self.topics_df, "topics")

        return df_out, topics_df

    def get_topics_extended(
        self,
        text_column: str = 'Processed Abstract',
        *,
        model_type: str = 'LDA',
        n_topics: int | None = None,
        max_topics: int = 15,
        max_features: int = 5000,
        stop_words: str | list[str] | None = 'english',
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        auto_select_topics: bool = True,
    ) -> dict:
        """
        Extended topic modeling with coherence scores and additional analytics.
        
        Parameters
        ----------
        text_column : str, default='Processed Abstract'
            Column containing text to model.
        model_type : str, default='LDA'
            Model type: 'LDA', 'NMF', or 'LSA'.
        n_topics : int or None
            Number of topics. If None and auto_select_topics=True, auto-selects.
        max_topics : int, default=15
            Maximum topics to try for auto-selection.
        max_features : int, default=5000
            Maximum vocabulary size.
        stop_words : str or list, default='english'
            Stop words to remove.
        ngram_range : tuple, default=(1, 2)
            N-gram range for vectorization.
        min_df : int, default=2
            Minimum document frequency.
        max_df : float, default=0.95
            Maximum document frequency.
        auto_select_topics : bool, default=True
            Auto-select optimal number of topics.
        
        Returns
        -------
        dict with comprehensive topic modeling results including:
            - df_out, topics_df, model, vectorizer
            - coherence_scores, topic_coherence, topic_stats
            - topic_similarity, topic_trends
        """
        # Find text column
        col = text_column if text_column in self.df.columns else None
        if col is None:
            for fallback in ["Abstract", "Processed Title", "Title"]:
                if fallback in self.df.columns:
                    col = fallback
                    break
        
        if col is None:
            raise KeyError(f"No suitable text column found. Tried: {text_column}, Abstract, Processed Title, Title")
        
        # Call extended topic modeling
        result = utilsbib.topic_modeling_extended(
            df=self.df,
            text_column=col,
            model_type=model_type,
            n_topics=n_topics,
            max_topics=max_topics,
            max_features=max_features,
            stop_words=stop_words,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            auto_select_topics=auto_select_topics,
        )
        
        # Store results
        self.topic_result = result
        self.df = result["df_out"]
        self.topics_df = result["topics_df"]
        self.topic_assignment_df = result["df_out"].copy()
        self.topic_weight_cols = [c for c in result["df_out"].columns 
                                  if c.startswith("Topic ") and c.endswith(" Weight")]
        
        # Compute additional analytics
        if result["topic_term_matrix"] is not None:
            # Topic similarity
            result["topic_similarity"] = utilsbib.compute_topic_similarity(
                result["topic_term_matrix"], method='cosine'
            )
            
            # Topic trends over time
            year_col = self.mapping.get("Year", "Year")
            if year_col in self.df.columns:
                result["topic_trends"] = utilsbib.compute_topic_trends(
                    self.df, year_column=year_col, normalize=True
                )
            else:
                result["topic_trends"] = pd.DataFrame()
        
        # Save to files
        if getattr(self, "res_folder", None):
            try:
                out_path = os.path.join(self.res_folder, "tables", "topics_extended.xlsx")
                with pd.ExcelWriter(out_path, engine="xlsxwriter") as xlw:
                    result["topics_df"].to_excel(xlw, sheet_name="Topic Terms", index=False)
                    result["topic_stats"].to_excel(xlw, sheet_name="Topic Stats", index=False)
                    if "topic_similarity" in result and result["topic_similarity"] is not None:
                        result["topic_similarity"].to_excel(xlw, sheet_name="Similarity")
                    if "topic_trends" in result and not result["topic_trends"].empty:
                        result["topic_trends"].to_excel(xlw, sheet_name="Trends")
                    
                    # Coherence scores
                    if result["coherence_scores"]:
                        coh_df = pd.DataFrame([
                            {"K": k, "Coherence": v} 
                            for k, v in result["coherence_scores"].items()
                        ])
                        coh_df.to_excel(xlw, sheet_name="Coherence Scores", index=False)
            except Exception:
                pass
        
        return result

    def plot_topic_wordclouds(
        self,
        n_cols: int = 3,
        max_words: int = 30,
        figsize_per_topic: tuple[float, float] = (4, 3),
        colormap: str = 'viridis',
        save_path: str | None = None,
        dpi: int = 600,
    ):
        """
        Plot word clouds for each topic.
        
        Requires get_topics() or get_topics_extended() to have been called first.
        
        Parameters
        ----------
        n_cols : int, default=3
            Number of columns in grid.
        max_words : int, default=30
            Maximum words per cloud.
        figsize_per_topic : tuple
            Size per topic subplot.
        colormap : str
            Colormap for word colors.
        save_path : str, optional
            Path to save figure.
        dpi : int
            DPI for saving.
        
        Returns
        -------
        matplotlib.Figure
        """
        from biblium import plotbib
        
        if not hasattr(self, 'topics_df') or self.topics_df is None:
            raise ValueError("Run get_topics() or get_topics_extended() first")
        
        return plotbib.plot_topic_wordclouds(
            self.topics_df,
            n_cols=n_cols,
            max_words=max_words,
            figsize_per_topic=figsize_per_topic,
            colormap=colormap,
            save_path=save_path,
            dpi=dpi,
        )
    
    def plot_comparative_topic_wordclouds(
        self,
        reference_topic: str | None = None,
        n_cols: int = 3,
        max_words: int = 30,
        figsize_per_topic: tuple[float, float] = (4, 3),
        save_path: str | None = None,
        dpi: int = 600,
    ):
        """
        Plot comparative word clouds showing word associations between topics.
        
        For each comparison topic, words are colored by their association:
        - Blue: more associated with reference topic  
        - Red: more associated with comparison topic
        
        Requires get_topics_extended() to have been called first.
        
        Parameters
        ----------
        reference_topic : str, optional
            Reference topic for comparison. If None, uses topic with most documents.
        n_cols : int, default=3
            Number of columns in grid.
        max_words : int, default=30
            Maximum words per cloud.
        figsize_per_topic : tuple
            Size per topic subplot.
        save_path : str, optional
            Path to save figure.
        dpi : int
            DPI for saving.
        
        Returns
        -------
        matplotlib.Figure
        """
        from biblium import plotbib
        
        if not hasattr(self, 'topic_result') or self.topic_result is None:
            raise ValueError("Run get_topics_extended() first")
        
        result = self.topic_result
        
        # Default reference topic: one with most documents
        if reference_topic is None:
            topic_stats = result.get("topic_stats", pd.DataFrame())
            if not topic_stats.empty and "Dominant Documents" in topic_stats.columns:
                reference_topic = topic_stats.loc[
                    topic_stats["Dominant Documents"].idxmax(), "Topic"
                ]
        
        return plotbib.plot_comparative_topic_wordclouds(
            topics_df=result["topics_df"],
            topic_term_matrix=result.get("topic_term_matrix"),
            feature_names=result.get("feature_names"),
            reference_topic=reference_topic,
            n_cols=n_cols,
            max_words=max_words,
            figsize_per_topic=figsize_per_topic,
            save_path=save_path,
            dpi=dpi,
        )

    # Sequential Topic Modeling

    def sequential_topic_modeling(
        self,
        text_column: str = 'Processed Abstract',
        time_column: str = 'Year',
        n_topics: int = 5,
        model_type: str = 'LDA',
        max_features: int = 5000,
        stop_words: str | list[str] | None = 'english',
        min_df: int = 2,
        max_df: float = 0.95,
    ) -> dict:
        """
        Sequential Topic Modeling - fit separate models per time period.
        
        Parameters
        ----------
        text_column : str, default='Processed Abstract'
            Column containing text.
        time_column : str, default='Year'
            Column containing time information.
        n_topics : int, default=5
            Number of topics per period.
        model_type : str, default='LDA'
            Model type: 'LDA', 'NMF', or 'LSA'.
        max_features : int, default=5000
            Maximum vocabulary size.
        stop_words : str or list, default='english'
            Stop words to remove.
        min_df : int, default=2
            Minimum document frequency.
        max_df : float, default=0.95
            Maximum document frequency.
        
        Returns
        -------
        dict
            Sequential topic modeling results.
        """
        # Find text column
        col = text_column if text_column in self.df.columns else None
        if col is None:
            for fallback in ["Abstract", "Processed Title", "Title"]:
                if fallback in self.df.columns:
                    col = fallback
                    break
        
        if col is None:
            raise KeyError(f"No text column found.")
        
        # Find time column
        time_col = self.mapping.get("Year", time_column)
        if time_col not in self.df.columns:
            time_col = time_column
        
        result = utilsbib.sequential_topic_modeling(
            df=self.df,
            text_column=col,
            time_column=time_col,
            n_topics=n_topics,
            model_type=model_type,
            max_features=max_features,
            stop_words=stop_words,
            min_df=min_df,
            max_df=max_df,
        )
        
        self.stm_result = result
        self.df = result["df_out"]
        
        # Save results
        if getattr(self, "res_folder", None):
            try:
                out_path = os.path.join(self.res_folder, "tables", "sequential_topics.xlsx")
                with pd.ExcelWriter(out_path, engine="xlsxwriter") as xlw:
                    result["topic_evolution"].to_excel(xlw, sheet_name="Topic Evolution", index=False)
                    result["topic_prevalence"].to_excel(xlw, sheet_name="Prevalence", index=False)
                    result["period_stats"].to_excel(xlw, sheet_name="Period Stats", index=False)
                    
                    # Period topics
                    for period, topics_df in result["period_topics"].items():
                        sheet_name = f"Period_{period}"[:31]
                        topics_df.to_excel(xlw, sheet_name=sheet_name, index=False)
            except Exception:
                pass
        
        return result

    def dynamic_topic_modeling(
        self,
        text_column: str = 'Processed Abstract',
        time_column: str = 'Year',
        n_topics: int = 5,
        n_time_slices: int = 5,
        model_type: str = 'LDA',
        max_features: int = 5000,
        stop_words: str | list[str] | None = 'english',
        min_df: int = 2,
        max_df: float = 0.95,
        chain_variance: float = 0.1,
    ) -> dict:
        """
        Dynamic Topic Modeling - track topic evolution with temporal smoothing.
        
        Parameters
        ----------
        text_column : str, default='Processed Abstract'
            Column containing text.
        time_column : str, default='Year'
            Column containing time information.
        n_topics : int, default=5
            Number of topics.
        n_time_slices : int, default=5
            Number of time slices.
        model_type : str, default='LDA'
            Model type: 'LDA', 'NMF', or 'LSA'.
        max_features : int, default=5000
            Maximum vocabulary size.
        stop_words : str or list, default='english'
            Stop words to remove.
        min_df : int, default=2
            Minimum document frequency.
        max_df : float, default=0.95
            Maximum document frequency.
        chain_variance : float, default=0.1
            Temporal smoothing (lower = smoother).
        
        Returns
        -------
        dict
            Dynamic topic modeling results.
        """
        # Find text column
        col = text_column if text_column in self.df.columns else None
        if col is None:
            for fallback in ["Abstract", "Processed Title", "Title"]:
                if fallback in self.df.columns:
                    col = fallback
                    break
        
        if col is None:
            raise KeyError(f"No text column found.")
        
        # Find time column
        time_col = self.mapping.get("Year", time_column)
        if time_col not in self.df.columns:
            time_col = time_column
        
        result = utilsbib.dynamic_topic_modeling(
            df=self.df,
            text_column=col,
            time_column=time_col,
            n_topics=n_topics,
            n_time_slices=n_time_slices,
            model_type=model_type,
            max_features=max_features,
            stop_words=stop_words,
            min_df=min_df,
            max_df=max_df,
            chain_variance=chain_variance,
        )
        
        self.dtm_result = result
        self.df = result["df_out"]
        
        # Save results
        if getattr(self, "res_folder", None):
            try:
                out_path = os.path.join(self.res_folder, "tables", "dynamic_topics.xlsx")
                with pd.ExcelWriter(out_path, engine="xlsxwriter") as xlw:
                    result["topic_prevalence_evolution"].to_excel(xlw, sheet_name="Prevalence", index=False)
                    result["time_slice_info"].to_excel(xlw, sheet_name="Time Slices", index=False)
                    result["global_topics"].to_excel(xlw, sheet_name="Global Topics", index=False)
                    
                    # Topic word evolution
                    for topic, evo_df in result["topic_word_evolution"].items():
                        sheet_name = topic[:31]
                        evo_df.to_excel(xlw, sheet_name=sheet_name, index=False)
            except Exception:
                pass
        
        return result

    def plot_stm_prevalence(
        self,
        stacked: bool = True,
        figsize: tuple[float, float] = (12, 6),
        save_path: str | None = None,
        dpi: int = 600,
    ):
        """Plot sequential topic model prevalence over time."""
        from biblium import plotbib
        
        if not hasattr(self, 'stm_result') or self.stm_result is None:
            raise ValueError("Run sequential_topic_modeling() first.")
        
        return plotbib.plot_topic_prevalence_evolution(
            self.stm_result["topic_prevalence"],
            time_col='Period',
            figsize=figsize,
            stacked=stacked,
            save_path=save_path,
            dpi=dpi,
        )

    def plot_dtm_prevalence(
        self,
        stacked: bool = True,
        figsize: tuple[float, float] = (12, 6),
        save_path: str | None = None,
        dpi: int = 600,
    ):
        """Plot dynamic topic model prevalence over time."""
        from biblium import plotbib
        
        if not hasattr(self, 'dtm_result') or self.dtm_result is None:
            raise ValueError("Run dynamic_topic_modeling() first.")
        
        return plotbib.plot_topic_prevalence_evolution(
            self.dtm_result["topic_prevalence_evolution"],
            time_col='Time_Slice',
            figsize=figsize,
            stacked=stacked,
            save_path=save_path,
            dpi=dpi,
        )

    def plot_dtm_streams(
        self,
        figsize: tuple[float, float] = (14, 6),
        save_path: str | None = None,
        dpi: int = 600,
    ):
        """Plot dynamic topics as stream graph."""
        from biblium import plotbib
        
        if not hasattr(self, 'dtm_result') or self.dtm_result is None:
            raise ValueError("Run dynamic_topic_modeling() first.")
        
        return plotbib.plot_dtm_topic_streams(
            self.dtm_result["topic_prevalence_evolution"],
            figsize=figsize,
            save_path=save_path,
            dpi=dpi,
        )

    # Term Evolution Analysis

    def compute_term_evolution(
        self,
        text_column: str = 'Processed Abstract',
        time_column: str = 'Year',
        top_n_terms: int = 20,
        min_df: int = 2,
        max_df: float = 0.95,
        stop_words: str | list | None = 'english',
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        Compute term frequency evolution over time.
        
        Parameters
        ----------
        text_column : str, default='Processed Abstract'
            Column containing text.
        time_column : str, default='Year'
            Column containing time information.
        top_n_terms : int, default=20
            Number of top terms to track.
        min_df : int, default=2
            Minimum document frequency.
        max_df : float, default=0.95
            Maximum document frequency.
        stop_words : str or list, default='english'
            Stop words.
        normalize : bool, default=True
            Normalize frequencies within each period.
        
        Returns
        -------
        pd.DataFrame
            Term evolution data.
        """
        # Find text column
        col = text_column if text_column in self.df.columns else None
        if col is None:
            for fallback in ["Abstract", "Processed Title", "Title"]:
                if fallback in self.df.columns:
                    col = fallback
                    break
        
        if col is None:
            raise KeyError("No text column found.")
        
        # Find time column
        time_col = self.mapping.get("Year", time_column)
        if time_col not in self.df.columns:
            time_col = time_column
        
        result = utilsbib.compute_term_evolution(
            df=self.df,
            text_column=col,
            time_column=time_col,
            top_n_terms=top_n_terms,
            min_df=min_df,
            max_df=max_df,
            stop_words=stop_words,
            normalize=normalize,
        )
        
        self.term_evolution = result
        return result

    def compute_term_trends(
        self,
        text_column: str = 'Processed Abstract',
        time_column: str = 'Year',
        terms: list[str] | None = None,
        top_n_terms: int = 10,
        min_df: int = 2,
        max_df: float = 0.95,
        stop_words: str | list | None = 'english',
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        Compute trends for specific terms over time.
        
        Parameters
        ----------
        text_column : str, default='Processed Abstract'
            Column containing text.
        time_column : str, default='Year'
            Column containing time information.
        terms : list of str, optional
            Specific terms to track. If None, uses top terms.
        top_n_terms : int, default=10
            Number of top terms if terms not specified.
        min_df : int, default=2
            Minimum document frequency.
        max_df : float, default=0.95
            Maximum document frequency.
        stop_words : str or list, default='english'
            Stop words.
        normalize : bool, default=True
            Normalize to proportion.
        
        Returns
        -------
        pd.DataFrame
            Term trends with periods as index, terms as columns.
        """
        # Find text column
        col = text_column if text_column in self.df.columns else None
        if col is None:
            for fallback in ["Abstract", "Processed Title", "Title"]:
                if fallback in self.df.columns:
                    col = fallback
                    break
        
        if col is None:
            raise KeyError("No text column found.")
        
        # Find time column
        time_col = self.mapping.get("Year", time_column)
        if time_col not in self.df.columns:
            time_col = time_column
        
        result = utilsbib.compute_term_trends(
            df=self.df,
            text_column=col,
            time_column=time_col,
            terms=terms,
            top_n_terms=top_n_terms,
            min_df=min_df,
            max_df=max_df,
            stop_words=stop_words,
            normalize=normalize,
        )
        
        self.term_trends = result
        return result

    def plot_term_evolution(
        self,
        top_n: int = 15,
        plot_type: str = 'heatmap',
        figsize: tuple[float, float] = (14, 8),
        save_path: str | None = None,
        dpi: int = 600,
    ):
        """
        Plot term frequency evolution.
        
        Parameters
        ----------
        top_n : int, default=15
            Number of top terms to display.
        plot_type : str, default='heatmap'
            Type of plot: 'heatmap', 'line', or 'area'.
        figsize : tuple
            Figure size.
        save_path : str, optional
            Path to save figure.
        dpi : int
            DPI for saving.
        """
        from biblium import plotbib
        
        if not hasattr(self, 'term_evolution') or self.term_evolution is None or self.term_evolution.empty:
            raise ValueError("Run compute_term_evolution() first.")
        
        return plotbib.plot_term_evolution(
            self.term_evolution,
            top_n=top_n,
            plot_type=plot_type,
            figsize=figsize,
            save_path=save_path,
            dpi=dpi,
        )

    def plot_term_trends(
        self,
        figsize: tuple[float, float] = (12, 6),
        show_markers: bool = True,
        save_path: str | None = None,
        dpi: int = 600,
    ):
        """
        Plot term trends over time.
        
        Parameters
        ----------
        figsize : tuple
            Figure size.
        show_markers : bool, default=True
            Show markers on lines.
        save_path : str, optional
            Path to save figure.
        dpi : int
            DPI for saving.
        """
        from biblium import plotbib
        
        if not hasattr(self, 'term_trends') or self.term_trends is None or self.term_trends.empty:
            raise ValueError("Run compute_term_trends() first.")
        
        return plotbib.plot_term_trends(
            self.term_trends,
            figsize=figsize,
            show_markers=show_markers,
            save_path=save_path,
            dpi=dpi,
        )

    def plot_term_bump_chart(
        self,
        top_n: int = 10,
        figsize: tuple[float, float] = (14, 8),
        save_path: str | None = None,
        dpi: int = 600,
    ):
        """
        Plot term rank evolution as bump chart.
        
        Parameters
        ----------
        top_n : int, default=10
            Number of terms to track.
        figsize : tuple
            Figure size.
        save_path : str, optional
            Path to save figure.
        dpi : int
            DPI for saving.
        """
        from biblium import plotbib
        
        if not hasattr(self, 'term_evolution') or self.term_evolution is None or self.term_evolution.empty:
            raise ValueError("Run compute_term_evolution() first.")
        
        return plotbib.plot_term_bump_chart(
            self.term_evolution,
            top_n=top_n,
            figsize=figsize,
            save_path=save_path,
            dpi=dpi,
        )

    # semantic interdisciplinarity

    def get_semantic_interdisciplinarity(
        self,
        vrs=[('Processed Abstract', 'text')],
    ):
        """
        Estimate semantic interdisciplinarity based on processed text fields.
        
        For each text variable specified in `vrs` this method applies
        `utilsbib.compute_semantic_interdisciplinarity` and stores the
                    resulting score in a new column named "Semantic id <VariableName>".
        
        Parameters
        ----------
        vrs : list[tuple[str, str]], optional
            List of (column_name, mode) pairs describing which text columns to
            analyse and how to interpret them. The default uses
            ("Processed Abstract", "text").
        """
        for v, m in vrs:
            self.df[f"Semantic id {v}"] = self.df[v].apply(lambda x: utilsbib.semantic_interdisciplinarity(x, mode=m, sep=self.default_separator))

    # sentiment analysis

    def get_sentiment(
        self,
        v='Processed Abstract',
        sentiment_threshold=0.05,
        top_words=10,
        correlate=['Year', 'Cited by'],
    ):
        """
        Perform sentiment analysis on a text field.
        
        Delegates to `utilsbib.analyze_sentiment` to compute a sentiment score
        for the selected text column and stores aggregate statistics in
        `self.sentiment_stats_df`. If `correlate` is non-empty, a simple
        correlation matrix between the sentiment score and the listed numeric
        variables is stored as `self.sentiment_correlations`.
        
        Parameters
        ----------
        v : str, default "Processed Abstract"
            Column containing text to analyse.
        sentiment_threshold : float, default 0.05
            Threshold used internally to classify texts as positive/negative.
        top_words : int, default 10
            Number of most influential words per sentiment class to report.
        correlate : list[str], optional
            Names of numeric columns to correlate with the sentiment score.
        """
        self.df, self.sentiment_stats_df = utilsbib.analyze_sentiment(self.df, v, sentiment_threshold=sentiment_threshold, top_words=top_words)
        self._save_table(self.sentiment_stats_df, "sentiment analysis stats")
        if len(correlate) > 0:
            self.sentiment_correlations = self.df[["Sentiment Score"] + correlate].corr()

    # LLM

    def llm_describe_df(
        self,
        df_att: str,
        *,
        prompt: str | None=None,
        prompt_template: str | None=None,
        attr_suffix: str='_desc',
        save_attr: bool=True,
        **kwargs,
    ) -> str:
        """
        Describe a dataframe-like attribute using either OpenAI or Hugging Face's
        OpenAI-compatible router (serverless; no local install).

        How it routes
        -------------
        - provider="openai": uses OpenAI Chat Completions (needs OPENAI_API_KEY or openai_api_key=...).
        - provider in {"huggingface","hf","hf-inference","hf_router"}:
            Uses OpenAI client with base_url="https://router.huggingface.co/v1"
            and your HF token. Model *must* include provider suffix ':hf-inference',
            e.g., "HuggingFaceTB/SmolLM3-3B:hf-inference".
            (This is the recommended way per HF docs.)  # noqa

        Tokens args
        -----------
        - OpenAI route expects "max_tokens".
        - HF router also uses OpenAI-compatible "max_tokens".
          If you pass "max_new_tokens", it’ll be mapped to "max_tokens".

        Parameters
        ----------
        df_att : str
            Name of the attribute on `self` holding the table (e.g., a pandas DataFrame).
        prompt : str | None
            Direct prompt; supports "{table_md}" and "{name}" placeholders.
        prompt_template : str | None
            Used if `prompt` is None; also supports placeholders.
        attr_suffix : str
            If `save_attr=True`, stores result as `self.<df_att><attr_suffix>`.
        save_attr : bool
            Save the generated text on `self` when True.
        **kwargs
            provider: "openai" | "huggingface" | "hf" | "hf-inference" | "hf_router"
            model, fallback_models: model names (append ':hf-inference' if using HF router)
            gen_kwargs: dict (use "max_tokens"; "max_new_tokens" is auto-mapped)
            openai_api_key: str (optional; else from env OPENAI_API_KEY)
            hf_token: str (optional; else from env HF_TOKEN or HUGGINGFACEHUB_API_TOKEN)

        Returns
        -------
        str
            The generated description.

        Raises
        ------
        AttributeError
            If `df_att` does not exist on `self`.
        RuntimeError
            If all attempts fail, with reasons listed.
        """
        if not hasattr(self, df_att):
            raise AttributeError(f"Attribute \"{df_att}\" not found on object.")

        table = getattr(self, df_att)

        # Build markdown view
        try:
            import pandas as pd  # type: ignore
            table_md = table.to_markdown(index=False) if isinstance(table, pd.DataFrame) else str(table)
        except Exception:
            table_md = str(table)

        # Final prompt
        if prompt is not None:
            final_prompt = prompt.replace("{table_md}", table_md).replace("{name}", str(df_att))
        else:
            tmpl = prompt_template or "Provide a concise, factual description of {name}:\n\n{table_md}"
            final_prompt = tmpl.replace("{table_md}", table_md).replace("{name}", str(df_att))

        # Config
        import os
        provider = kwargs.pop("provider", None)
        model = kwargs.pop("model", None)
        fallbacks = kwargs.pop("fallback_models", []) or []
        gen = dict(kwargs.pop("gen_kwargs", {}) or {})

        openai_api_key = kwargs.pop("openai_api_key", None) or os.environ.get("OPENAI_API_KEY")
        hf_token = (
            kwargs.pop("hf_token", None)
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        )

        # Provider default
        if provider is None:
            provider = "openai" if openai_api_key else "huggingface"

        # Models to try
        if provider in {"huggingface", "hf", "hf-inference", "hf_router"}:
            # Known-good small LLMs on hf-inference (check Supported Models page).
            defaults = ["HuggingFaceTB/SmolLM3-3B:hf-inference", "katanemo/Arch-Router-1.5B:hf-inference"]
            models_to_try = [m for m in [model, *fallbacks] if m] or defaults
            # Ensure ':hf-inference' suffix is present
            models_to_try = [m if ":" in m else f"{m}:hf-inference" for m in models_to_try]
        else:
            models_to_try = [m for m in [model, *fallbacks] if m] or ["gpt-4o-mini", "gpt-4.1-mini"]

        errors: list[str] = []

        # Backends
        def _run_openai(
            model_id: str,
            base_url: str | None,
            api_key: str,
        ) -> str:
            """
            Run inference using OpenAI-compatible API.
            
            Parameters
            ----------
            model_id : str
                Model identifier.
            base_url : str or None
                Base URL for API endpoint.
            api_key : str
                API key for authentication.
                
            Returns
            -------
            str
                Generated text response.
            """
            from openai import OpenAI  # type: ignore
            # Normalize tokens for OpenAI-compatible
            okw = dict(gen)
            if "max_tokens" not in okw and "max_new_tokens" in okw:
                okw["max_tokens"] = okw.pop("max_new_tokens")
            client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": final_prompt}],
                **okw,
            )
            return resp.choices[0].message.content or ""

        for m in models_to_try:
            try:
                if provider in {"huggingface", "hf", "hf-inference", "hf_router"}:
                    if not hf_token:
                        raise RuntimeError("HF token not provided (set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN).")
                    # OpenAI-compatible router call (HF docs show this exact pattern)
                    text = _run_openai(m, "https://router.huggingface.co/v1", hf_token)
                elif provider == "openai":
                    if not openai_api_key:
                        raise RuntimeError("OPENAI_API_KEY not provided.")
                    text = _run_openai(m, None, openai_api_key)
                else:
                    raise RuntimeError(f"Unsupported provider: {provider}")
                if save_attr:
                    setattr(self, f"{df_att}{attr_suffix}", text)
                return text
            except Exception as e:
                errors.append(f"{provider}:{m}: {type(e).__name__}: {e}")

        raise RuntimeError("All candidate models failed. " + " | ".join(errors))

    def llm_describe_dfs(
        self,
        dfs: list[str] | None=None,
        *,
        prompt: str | None=None,
        prompt_template: str | None=None,
        attr_suffix: str='_desc',
        save_attr: bool=True,
        **kwargs,
    ) -> dict[str, str]:
        """
        Batch-generate LLM descriptions for multiple dataframe-like attributes, delegating to
        `self.llm_describe_df` (which supports OpenAI *or* Hugging Face router with HF_TOKEN, no local install).

        Behavior
        --------
        - If `dfs` is None, auto-discovers attributes that end with "_df" (and, if pandas is available,
          also keeps only those that are actually DataFrames). Excludes "show_df".
        - For each attribute, calls `self.llm_describe_df` with the same prompt/template and kwargs.
        - Returns a dict {attr_name -> description}; per-item errors are captured as "[ERROR] ...".

        Provider/Model Notes (inherits from llm_describe_df)
        ----------------------------------------------------
        - OpenAI route uses Chat Completions with "max_tokens".
        - Hugging Face router route uses the OpenAI-compatible endpoint; models should include the
          provider suffix ":hf-inference" (e.g., "HuggingFaceTB/SmolLM3-3B:hf-inference").
          This function will auto-append the suffix if `provider` looks like Hugging Face and the
          model/fallbacks lack a suffix. (llm_describe_df also normalizes, so this is safe.)

        Parameters
        ----------
        dfs : list[str] | None
            Attribute names to describe. If None, auto-discovers as noted above.
        prompt : str | None
            Direct prompt with optional "{table_md}" and "{name}" placeholders.
        prompt_template : str | None
            Used when `prompt` is None; also supports placeholders.
        attr_suffix : str
            If `save_attr=True`, each result is stored as `self.<attr><attr_suffix>`.
        save_attr : bool
            Whether to store the generated text on `self`.
        **kwargs
            Forwarded to `llm_describe_df` (e.g., provider, model, fallback_models,
            gen_kwargs, openai_api_key, hf_token, etc.).

        Returns
        -------
        dict[str, str]
            Mapping from attribute name to its generated description, or "[ERROR] <message>" per failure.
        """
        # Auto-discover DF-like attributes
        if dfs is None:
            all_names = [n for n in dir(self) if n.endswith("_df") and n not in {"show_df"}]
            try:
                import pandas as pd  # type: ignore
                discovered = []
                for n in all_names:
                    try:
                        if isinstance(getattr(self, n), pd.DataFrame):
                            discovered.append(n)
                    except Exception:
                        # If any attribute access fails, skip it quietly
                        continue
                dfs = discovered or all_names
            except Exception:
                dfs = all_names

        # Normalize provider/model suffixes for HF router without mutating caller kwargs
        prov = kwargs.get("provider")
        prov = {"hf": "huggingface", "hf-inference": "huggingface", "hf_router": "huggingface",
                "huggingface_hub": "huggingface"}.get(prov, prov)
        def _with_hf_suffix(
            name: str,
        ) -> str:
            """Internal helper function."""
            return name if (":" in name) else f"{name}:hf-inference"

        norm_kwargs = dict(kwargs)
        if prov == "huggingface":
            if "model" in norm_kwargs and isinstance(norm_kwargs["model"], str):
                norm_kwargs["model"] = _with_hf_suffix(norm_kwargs["model"])
            if "fallback_models" in norm_kwargs and isinstance(norm_kwargs["fallback_models"], (list, tuple)):
                norm_kwargs["fallback_models"] = [
                    _with_hf_suffix(m) if isinstance(m, str) and ":" not in m else m
                    for m in norm_kwargs["fallback_models"]
                ]

        # Run
        results: dict[str, str] = {}
        for name in dfs:
            try:
                results[name] = self.llm_describe_df(
                    name,
                    prompt=prompt,
                    prompt_template=prompt_template,
                    attr_suffix=attr_suffix,
                    save_attr=save_attr,
                    **norm_kwargs,
                )
            except Exception as e:
                results[name] = f"[ERROR] {e}"
        return results

    def llm_summarize_abstracts(
        self,
        abstracts,
        *,
        provider: str | None=None,
        model: str | None=None,
        fallback_models: list[str] | None=None,
        hf_token: str | None=None,
        openai_api_key: str | None=None,
        gen_kwargs: dict | None=None,
        separator: str='\n\n---\n\n',
        max_chunk_chars: int=12000,
        chunk_overlap_chars: int=600,
        save_attr: bool=False,
        attr_name: str='abstracts_desc',
    ) -> str:
        """
        Summarize one or many abstracts via OpenAI or Hugging Face router (no local install).

        Input handling
        --------------
        - Accepts: str | list[str] | pandas.Series | numpy array-like.
        - Cleans: drops None/NaN/empty, casts to str, strips.
        - Fixes your error: joins only after ensuring we have an *iterable of strings*.

        Routing
        -------
        - If `openai_api_key` (or env OPENAI_API_KEY) is set → use OpenAI Chat.
        - Else if `hf_token` (or env HF_TOKEN/HUGGINGFACEHUB_API_TOKEN) → use HF router
          with OpenAI-compatible client at base_url="https://router.huggingface.co/v1".
          **Models must include ':hf-inference'** (this function adds it if missing).

        Generation args
        ---------------
        - Uses OpenAI-compatible "max_tokens". If you pass "max_new_tokens", it’s mapped.
        - `gen_kwargs` is forwarded to the chat completion call.

        Chunking
        --------
        - If text > `max_chunk_chars`, it’s chunked with `chunk_overlap_chars`,
          summarized per chunk, then a final summary is generated from the chunk summaries.

        Returns
        -------
        str
            The final summary. Optionally saved to `self.<attr_name>`.

        Examples
        --------
        # Hugging Face router (HF_TOKEN only, no local install)
        # NOTE: ':hf-inference' is auto-appended if missing.
        # ba.llm_summarize_abstracts(list(ba.df["Abstract"])[15:17], hf_token=HF_TOKEN,
        #     model="HuggingFaceTB/SmolLM3-3B", gen_kwargs={"max_tokens": 300})

        # OpenAI route
        # ba.llm_summarize_abstracts(ba.df["Abstract"].dropna().head(20).tolist(),
        #     openai_api_key=OPENAI_KEY, model="gpt-4o-mini", gen_kwargs={"max_tokens": 300})
        """
        import os
        from typing import Iterable

        # -------- normalize abstracts to a clean list[str] --------
        def _to_list_of_str(
            x,
        ) -> list[str]:
            """Internal helper function."""
            try:
                import pandas as pd  # type: ignore
                import numpy as np   # type: ignore
            except Exception:
                pd = np = None

            # If already a string, wrap
            if isinstance(x, str):
                seq = [x]
            # Pandas Series
            elif (pd is not None) and isinstance(x, pd.Series):
                seq = x.tolist()
            # Numpy array
            elif (np is not None) and isinstance(getattr(x, "__class__", object), type) and hasattr(x, "shape"):
                try:
                    import numpy as np  # type: ignore
                    if isinstance(x, np.ndarray):
                        seq = x.tolist()
                    else:
                        seq = list(x)  # generic array-like
                except Exception:
                    seq = list(x) if isinstance(x, Iterable) else [str(x)]
            # Generic iterable (list, tuple, etc.)
            elif isinstance(x, Iterable):
                seq = list(x)
            else:
                # Non-iterable fallback
                seq = [str(x)]

            # Clean: drop None/NaN/empties, cast to str, strip
            cleaned: list[str] = []
            for itm in seq:
                if itm is None:
                    continue
                s = str(itm)
                if s.lower() == "nan":
                    continue
                s = s.strip()
                if s:
                    cleaned.append(s)
            return cleaned

        items = _to_list_of_str(abstracts)
        if not items:
            raise ValueError("No valid abstracts to summarize after cleaning.")

        # Join in manageable chunks
        big_text = separator.join(items)

        def _chunk_text(
            txt: str,
        ) -> list[str]:
            """Internal helper function."""
            if len(txt) <= max_chunk_chars:
                return [txt]
            chunks = []
            start = 0
            while start < len(txt):
                end = min(start + max_chunk_chars, len(txt))
                chunks.append(txt[start:end])
                if end == len(txt):
                    break
                start = max(0, end - chunk_overlap_chars)
            return chunks

        chunks = _chunk_text(big_text)

        # -------- routing config --------
        provider = provider or None
        gen = dict(gen_kwargs or {})
        # Normalize token arg for OpenAI-compatible endpoints
        if "max_tokens" not in gen and "max_new_tokens" in gen:
            gen["max_tokens"] = gen.pop("max_new_tokens")

        openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

        if provider is None:
            provider = "openai" if openai_api_key else "huggingface"

        # Model lists
        fallback_models = fallback_models or []
        if provider.lower() in {"huggingface", "hf", "hf-inference", "hf_router"}:
            # Default small, serverless-friendly
            defaults = ["HuggingFaceTB/SmolLM3-3B", "katanemo/Arch-Router-1.5B"]
            models = [m for m in [model, *fallback_models] if m] or defaults
            # Ensure router provider suffix
            models = [m if ":" in m else f"{m}:hf-inference" for m in models]
            base_url = "https://router.huggingface.co/v1"
            api_key = hf_token
        elif provider.lower() == "openai":
            models = [m for m in [model, *fallback_models] if m] or ["gpt-4o-mini", "gpt-4.1-mini"]
            base_url = None
            api_key = openai_api_key
        else:
            raise RuntimeError(f"Unsupported provider: {provider}")

        if not api_key:
            raise RuntimeError(f"Missing API key for provider '{provider}'.")

        # -------- call function (OpenAI-compatible client) --------
        def _chat_complete(
            prompt_text: str,
            model_id: str,
        ) -> str:
            """Internal helper function."""
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt_text}],
                **gen,
            )
            return resp.choices[0].message.content or ""

        # Prompts
        per_chunk_prompt = (
            "You are a precise scientific summarizer. Summarize the following abstracts into "
            "a compact, readable overview in bullet points. Keep facts, avoid fluff.\n\n"
            "{text}"
        )
        final_prompt = (
            "Combine the following chunk summaries into one concise summary (≤ 10 bullets). "
            "Remove redundancy, keep key findings, scope, methods, and limitations.\n\n{text}"
        )

        # Try models in order
        errors: list[str] = []
        for m in models:
            try:
                # Summarize each chunk
                chunk_summaries = []
                for ch in chunks:
                    p = per_chunk_prompt.replace("{text}", ch)
                    chunk_summaries.append(_chat_complete(p, m))
                # If multiple chunks, synthesize a final summary
                if len(chunk_summaries) > 1:
                    stitched = separator.join(chunk_summaries)
                    out = _chat_complete(final_prompt.replace("{text}", stitched), m)
                else:
                    out = chunk_summaries[0]
                if save_attr:
                    setattr(self, attr_name, out)
                return out
            except Exception as e:
                errors.append(f"{provider}:{m}: {type(e).__name__}: {e}")

        raise RuntimeError("All candidate models failed. " + " | ".join(errors))

    def llm_summarize_ngrams(
        self,
        ngrams,
        *,
        provider: str | None=None,
        model: str | None=None,
        fallback_models: list[str] | None=None,
        hf_token: str | None=None,
        openai_api_key: str | None=None,
        gen_kwargs: dict | None=None,
        lower: bool=True,
        strip: bool=True,
        stopwords: set[str] | list[str] | None=None,
        max_items: int=500,
        min_len: int=1,
        separator: str='\n',
        max_chunk_chars: int=12000,
        chunk_overlap_chars: int=600,
        save_attr: bool=False,
        attr_name: str='ngrams_desc',
    ) -> str:
        """
        Summarize a list of terms/phrases (n-grams) with an LLM — no local install required.

        Input
        -----
        - Accepts: list[str] | tuple[str] | pandas.Series | numpy array-like | dict[str,int] | str.
          Duplicates imply frequency; dict values are treated as counts.
        - Cleans: optional lower/strip; optional stopword removal; filters by `min_len`.

        Routing
        -------
        - If `openai_api_key` (or env OPENAI_API_KEY) is set → OpenAI Chat Completions.
        - Else if `hf_token` (or env HF_TOKEN/HUGGINGFACEHUB_API_TOKEN) → Hugging Face router
          at base_url="https://router.huggingface.co/v1" using an OpenAI-compatible client.
          **Models must include ':hf-inference'**; the suffix is auto-added if missing.

        Generation args
        ---------------
        - Uses OpenAI-compatible "max_tokens". If you pass "max_new_tokens", it is mapped to "max_tokens".
        - `gen_kwargs` is forwarded to the chat completion call.

        Chunking
        --------
        - If the prompt would exceed `max_chunk_chars`, the term list is processed in chunks;
          each chunk is summarized, then a final synthesis is produced from chunk summaries.

        Returns
        -------
        str
            The summary text. Optionally saved to `self.<attr_name>` when `save_attr=True`.

        Examples
        --------
        # Hugging Face router (HF_TOKEN only, no local install)
        # ':hf-inference' is auto-appended if missing.
        # ba.llm_summarize_ngrams(ba.df["Author Keywords"].dropna().str.split("; ").explode(),
        #                         provider="huggingface", model="HuggingFaceTB/SmolLM3-3B",
        #                         hf_token=HF_TOKEN, gen_kwargs={"max_tokens": 300, "temperature": 0.2})

        # OpenAI route
        # ba.llm_summarize_ngrams(my_terms_list, provider="openai", model="gpt-4o-mini",
        #                         openai_api_key=OPENAI_KEY, gen_kwargs={"max_tokens": 300})
        """
        import os
        from collections import Counter
        from typing import Iterable

        # ---------- normalize to Counter[str:int] ----------
        def _clean(
            s: str,
        ) -> str:
            """Internal helper function."""
            if strip:
                s = s.strip()
            if lower:
                s = s.lower()
            return s

        def _to_counter(
            x,
        ) -> Counter:
            """Internal helper function."""
            try:
                import pandas as pd  # type: ignore
                import numpy as np   # type: ignore
            except Exception:
                pd = np = None

            # dict[str,int]
            if isinstance(x, dict):
                c = Counter()
                for k, v in x.items():
                    if k is None:
                        continue
                    ks = _clean(str(k))
                    if not ks or len(ks) < min_len:
                        continue
                    if stopwords and ks in set(stopwords):
                        continue
                    try:
                        c[ks] += int(v)
                    except Exception:
                        c[ks] += 1
                return c

            # scalar string → single item
            if isinstance(x, str):
                xs = [_clean(x)]
            # pandas Series
            elif (pd is not None) and isinstance(x, pd.Series):
                xs = [ _clean(str(t)) for t in x.dropna().astype(str).tolist() ]
            # numpy array
            elif (np is not None) and isinstance(getattr(x, "__class__", object), type) and hasattr(x, "shape"):
                try:
                    import numpy as np  # type: ignore
                    if isinstance(x, np.ndarray):
                        xs = [ _clean(str(t)) for t in x.flatten().tolist() ]
                    else:
                        xs = [ _clean(str(t)) for t in list(x) ]
                except Exception:
                    xs = [ _clean(str(t)) for t in list(x) ] if isinstance(x, Iterable) else [_clean(str(x))]
            # generic iterable
            elif isinstance(x, Iterable):
                xs = [ _clean(str(t)) for t in x ]
            else:
                xs = [_clean(str(x))]

            sw = set(stopwords) if stopwords else None
            xs = [t for t in xs if t and len(t) >= min_len and (not sw or t not in sw)]
            return Counter(xs)

        counts = _to_counter(ngrams)
        if not counts:
            raise ValueError("No valid terms to summarize after cleaning.")

        # keep top N by frequency
        most_common = counts.most_common(max_items)
        # Prepare a compact lines block: "term\tcount"
        lines = "\n".join(f"{t}\t{c}" for t, c in most_common)

        # ---------- prompt assembly & chunking ----------
        per_chunk_prompt = (
            "You are a bibliometrics analyst. Group the following terms (with counts) into concise themes. "
            "Output 6-12 bullet points. For each theme, give a short label and 3-6 representative terms. "
            "Prefer specificity, merge synonyms/variants, avoid redundancy.\n\n"
            "TERMS (one per line as 'term<TAB>count'):\n{terms}"
        )
        final_prompt = (
            "Combine and deduplicate the bullet-point themes below into a single, clean list of 6-12 bullets. "
            "Keep theme labels terse; keep only the strongest 3-6 example terms per theme.\n\n{chunk_summaries}"
        )

        def _chunk_text(
            txt: str,
        ) -> list[str]:
            """Internal helper function."""
            if len(txt) <= max_chunk_chars:
                return [txt]
            chunks = []
            start = 0
            while start < len(txt):
                end = min(start + max_chunk_chars, len(txt))
                chunks.append(txt[start:end])
                if end == len(txt):
                    break
                start = max(0, end - chunk_overlap_chars)
            return chunks

        chunks = _chunk_text(lines)

        # ---------- routing (OpenAI-compatible client) ----------
        provider = provider or None
        gen = dict(gen_kwargs or {})
        if "max_tokens" not in gen and "max_new_tokens" in gen:
            gen["max_tokens"] = gen.pop("max_new_tokens")

        openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

        if provider is None:
            provider = "openai" if openai_api_key else "huggingface"

        fallback_models = fallback_models or []
        if provider.lower() in {"huggingface", "hf", "hf-inference", "hf_router"}:
            # Small, router-served defaults; suffix added if missing
            defaults = ["HuggingFaceTB/SmolLM3-3B", "katanemo/Arch-Router-1.5B"]
            models = [m for m in [model, *fallback_models] if m] or defaults
            models = [m if ":" in m else f"{m}:hf-inference" for m in models]
            base_url = "https://router.huggingface.co/v1"
            api_key = hf_token
        elif provider.lower() == "openai":
            models = [m for m in [model, *fallback_models] if m] or ["gpt-4o-mini", "gpt-4.1-mini"]
            base_url = None
            api_key = openai_api_key
        else:
            raise RuntimeError(f"Unsupported provider: {provider}")

        if not api_key:
            raise RuntimeError(f"Missing API key for provider \"{provider}\".")

        def _chat_complete(
            prompt_text: str,
            model_id: str,
        ) -> str:
            """Internal helper function."""
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt_text}],
                **gen,
            )
            return resp.choices[0].message.content or ""

        errors: list[str] = []
        for m in models:
            try:
                # summarize each chunk
                chunk_summaries = []
                for ch in chunks:
                    p = per_chunk_prompt.replace("{terms}", ch)
                    chunk_summaries.append(_chat_complete(p, m))
                # final synthesis if needed
                if len(chunk_summaries) > 1:
                    stitched = separator.join(chunk_summaries)
                    out = _chat_complete(final_prompt.replace("{chunk_summaries}", stitched), m)
                else:
                    out = chunk_summaries[0]
                if save_attr:
                    setattr(self, attr_name, out)
                return out
            except Exception as e:
                errors.append(f"{provider}:{m}: {type(e).__name__}: {e}")

        raise RuntimeError("All candidate models failed. " + " | ".join(errors))

    
    # =========================================================================
    # SLEEPING BEAUTY ANALYSIS METHODS
    # =========================================================================
    # Prerequisites: import utilsbib, plotbib
    # Add these methods to your existing BiblioAnalysis class
    # Requires: self.db == "oa" (OpenAlex database only)
    # Attributes used: self.df, self.db, self.sleeping_beauties, self.all_metrics, 
    #                  self.storytellers, self.sb_summary
    # =========================================================================

    def compute_sb_metrics(self, current_year: int = 2025) -> pd.DataFrame:
        """
        Calculate Sleeping Beauty metrics for all papers.
        
        Args:
            current_year: Current year for analysis
        
        Returns:
            DataFrame with all papers and their SB metrics
        """
        if self.db != "oa":
            raise ValueError("Sleeping Beauty analysis is only available for OpenAlex database (self.db must be 'oa')")
        
        print("Calculating Sleeping Beauty metrics for all papers...")
        self.all_metrics = utilsbib.extract_all_papers_with_metrics(
            self.df, 
            current_year=current_year
        )
        print(f"   Calculated metrics for {len(self.all_metrics)} papers")
        return self.all_metrics

    def extract_sleeping_beauties(
        self,
        min_beauty_coefficient: float = 30.0,
        min_sleep_years: int = 3,
        min_total_citations: int = 30,
        min_awakening_intensity: float = 1.5,
        current_year: int = 2025
    ) -> pd.DataFrame:
        """
        Extract Sleeping Beauties based on configured thresholds.
        
        A Sleeping Beauty is a paper that:
        1. Has a high Beauty Coefficient (large deviation from linear citation growth)
        2. Remained dormant for several years before awakening
        3. Eventually achieved significant citation counts
        4. Had a dramatic awakening (sharp increase in citations)
        
        Args:
            min_beauty_coefficient: Minimum B coefficient threshold
            min_sleep_years: Minimum sleep duration threshold
            min_total_citations: Minimum total citations threshold
            min_awakening_intensity: Minimum awakening intensity threshold
            current_year: Current year for analysis
        
        Returns:
            DataFrame of identified Sleeping Beauties
        """
        if self.db != "oa":
            raise ValueError("Sleeping Beauty analysis is only available for OpenAlex database (self.db must be 'oa')")
        
        print(f"Extracting Sleeping Beauties...")
        print(f"   Criteria: B >= {min_beauty_coefficient}, "
              f"Sleep >= {min_sleep_years} years, "
              f"Citations >= {min_total_citations}, "
              f"Intensity >= {min_awakening_intensity}")
        
        self.sleeping_beauties = utilsbib.extract_sleeping_beauties(
            self.df,
            min_beauty_coefficient=min_beauty_coefficient,
            min_sleep_years=min_sleep_years,
            min_total_citations=min_total_citations,
            min_awakening_intensity=min_awakening_intensity,
            current_year=current_year
        )
        print(f"   Found {len(self.sleeping_beauties)} Sleeping Beauties")
        return self.sleeping_beauties

    def find_storytellers(self, min_sb_citations: int = 1) -> pd.DataFrame:
        """
        Identify Storytellers who cite multiple Sleeping Beauties.
        
        Storytellers are researchers who recognize the value of dormant papers
        and help bring them back to attention through their citations.
        
        Args:
            min_sb_citations: Minimum number of SBs an author must cite
        
        Returns:
            DataFrame of Storytellers
        """
        if self.sleeping_beauties is None:
            raise ValueError("Run extract_sleeping_beauties() first")
        
        print("Identifying Storytellers...")
        self.storytellers = utilsbib.identify_storytellers(
            self.df,
            self.sleeping_beauties,
            min_sb_citations=min_sb_citations
        )
        print(f"   Found {len(self.storytellers)} potential Storytellers")
        return self.storytellers

    def find_princes(
        self, 
        sleeping_beauty_id: str, 
        awakening_year: int
    ) -> List[Dict[str, Any]]:
        """
        Find potential Princes for a specific Sleeping Beauty.
        
        A Prince is typically a highly influential paper published around the 
        awakening year that cites the Sleeping Beauty, bringing it renewed attention.
        
        Args:
            sleeping_beauty_id: OpenAlex ID of the Sleeping Beauty
            awakening_year: Year when the SB was awakened
        
        Returns:
            List of potential Prince candidates
        """
        return utilsbib.identify_potential_princes(
            self.df, 
            sleeping_beauty_id, 
            awakening_year
        )

    def run_sb_analysis(
        self,
        min_beauty_coefficient: float = 30.0,
        min_sleep_years: int = 3,
        min_total_citations: int = 30,
        min_awakening_intensity: float = 1.5,
        current_year: int = 2025
    ) -> Dict[str, Any]:
        """
        Run the complete Sleeping Beauty analysis pipeline.
        
        This method orchestrates all SB analysis steps:
        1. Compute metrics for all papers
        2. Extract Sleeping Beauties
        3. Identify Storytellers
        4. Generate summary statistics
        
        Args:
            min_beauty_coefficient: Minimum B coefficient threshold
            min_sleep_years: Minimum sleep duration threshold
            min_total_citations: Minimum total citations threshold
            min_awakening_intensity: Minimum awakening intensity threshold
            current_year: Current year for analysis
        
        Returns:
            Dictionary with summary of analysis results
        """
        if self.db != "oa":
            raise ValueError("Sleeping Beauty analysis is only available for OpenAlex database (self.db must be 'oa')")
        
        print("=" * 60)
        print("SLEEPING BEAUTY ANALYSIS")
        print("=" * 60)
        
        # Step 1: Calculate metrics for all papers
        print("\n1. Calculating metrics...")
        self.compute_sb_metrics(current_year=current_year)
        
        # Step 2: Extract Sleeping Beauties
        print("\n2. Extracting Sleeping Beauties...")
        self.extract_sleeping_beauties(
            min_beauty_coefficient=min_beauty_coefficient,
            min_sleep_years=min_sleep_years,
            min_total_citations=min_total_citations,
            min_awakening_intensity=min_awakening_intensity,
            current_year=current_year
        )
        
        # Step 3: Identify Storytellers
        print("\n3. Identifying Storytellers...")
        self.find_storytellers()
        
        # Step 4: Generate summary
        print("\n4. Generating summary...")
        self.sb_summary = {
            "total_papers": len(self.df),
            "papers_with_metrics": len(self.all_metrics),
            "num_sleeping_beauties": len(self.sleeping_beauties),
            "num_storytellers": len(self.storytellers),
            "avg_beauty_coefficient": (
                self.sleeping_beauties["beauty_coefficient"].mean() 
                if len(self.sleeping_beauties) > 0 else 0
            ),
            "avg_sleep_duration": (
                self.sleeping_beauties["sleep_duration"].mean() 
                if len(self.sleeping_beauties) > 0 else 0
            ),
            "max_beauty_coefficient": (
                self.sleeping_beauties["beauty_coefficient"].max() 
                if len(self.sleeping_beauties) > 0 else 0
            ),
            "max_sleep_duration": (
                self.sleeping_beauties["sleep_duration"].max() 
                if len(self.sleeping_beauties) > 0 else 0
            ),
        }
        
        print("\n" + "=" * 60)
        print("SLEEPING BEAUTY ANALYSIS COMPLETE")
        print("=" * 60)
        
        return self.sb_summary

    def get_sleeping_beauty_result(self, index: int = 0) -> "utilsbib.SleepingBeautyResult":
        """
        Get a SleepingBeautyResult object for a specific paper.
        
        Args:
            index: Index in the sleeping_beauties DataFrame
        
        Returns:
            SleepingBeautyResult object
        """
        if self.sleeping_beauties is None or len(self.sleeping_beauties) == 0:
            raise ValueError("No Sleeping Beauties found. Run extract_sleeping_beauties() first.")
        
        row = self.sleeping_beauties.iloc[index]
        return utilsbib.SleepingBeautyResult(**row.to_dict())

    def compute_bursts(self, 
                       keyword_col: str = "Processed Author Keywords", 
                       year_col: str = "Year",
                       top_n: int = 50,
                       s: float = 2.0,
                       gamma: float = 1.0,
                       min_duration: int = 0) -> pd.DataFrame:
        """
        Compute Kleinberg bursts for the top N keywords.
        
        Parameters
        ----------
        keyword_col : str
            Column containing lists of keywords (or semicolon-separated strings).
        year_col : str
            Column containing the publication year.
        top_n : int
            Number of most frequent keywords to analyze.
        s, gamma : float
            Kleinberg algorithm parameters (Scaling, Cost).
            
        Returns
        -------
        pd.DataFrame
            Columns: [Keyword, Start, End, Weight, Duration]
        """
        # 1. Prepare Data
        # Ensure keywords are list type if they are strings
        temp_df = self.df[[year_col, keyword_col]].dropna().copy()
        
        if isinstance(temp_df[keyword_col].iloc[0], str):
            # Assuming semi-colon separated
            temp_df[keyword_col] = temp_df[keyword_col].str.split(r";\s*")
            
        # Explode to get one row per keyword instance
        exploded = temp_df.explode(keyword_col)
        
        # 2. Filter for Top N Keywords
        top_keywords = exploded[keyword_col].value_counts().head(top_n).index.tolist()
        exploded = exploded[exploded[keyword_col].isin(top_keywords)]
        
        # 3. Calculate Bursts
        all_bursts = []
        
        for kw, group in exploded.groupby(keyword_col):
            years = group[year_col].sort_values().astype(float).tolist()
            # Add small noise to years to allow calculation on yearly data
            # (Kleinberg assumes continuous time)
            years_noisy = [y + np.random.uniform(0, 0.01) for y in years]
            
            bursts = utilsbib.kleinberg_burst_detection(years_noisy, s=s, gamma=gamma)
            
            for b in bursts:
                # Round back to nearest integer year
                start_yr = int(round(b['start']))
                end_yr = int(round(b['end']))
                
                if (end_yr - start_yr) >= min_duration:
                    all_bursts.append({
                        "Keyword": kw,
                        "Start": start_yr,
                        "End": end_yr,
                        "Weight": b['weight'],
                        "Duration": end_yr - start_yr
                    })
                    
        if not all_bursts:
            print("No bursts detected with current parameters.")
            return pd.DataFrame()
            
        burst_df = pd.DataFrame(all_bursts)
        return burst_df.sort_values("Start", ascending=True)


    def save_to_file(
        self,
        file_name='biblio.pkl',
        exclude_dataset=True,
    ):
        """
        Serialize the current analysis object to disk using pickle.
        
        The object is written to the results folder as a binary .pkl file so
        that it can be reloaded in a later Python session. Optionally the main
        dataframe is temporarily removed before pickling to keep the file
        size smaller.
        
        Parameters
        ----------
        file_name : str, default "biblio.pkl"
            File name (without path) used in `self.res_folder`.
        exclude_dataset : bool, default True
            If True, `self.df` is temporarily set to None before pickling and
            restored afterwards.
        """
        with open(os.path.join(self.res_folder, file_name), "wb") as file:
            import pickle
            if exclude_dataset:
                df0 = self.df
                self.df = None
            pickle.dump(self,  file)
            self.set_data(df0)
        print(f"Analysis saved to {file_name}")

    def set_data(
        self,
        df,
    ):
        """
        Replace the main dataframe attached to the analysis object.
        
        This is a simple helper used after reloading a pickled object or when
        experimenting with alternative filtered views of the same dataset.
        
        Parameters
        ----------
        df : pandas.DataFrame
            New dataframe to assign to `self.df`.
        """
        self.df = df

    def show_data(
        self,
        sample=True,
        n=20,
    ):
        """
        Export the working dataset (or a sample) to Excel and open it.
        
        Writes the current `self.df` (or a random sample of it) to an Excel
        file in a "sample data" subfolder of the results directory and then
        opens the file with the default system application.
        
        Parameters
        ----------
        sample : bool, default True
            If True, export only a random sample of `n` rows; otherwise export
            the full dataset.
        n : int, default 20
            Number of rows to sample when `sample=True`.
        """
        utilsbib.make_folders([os.path.join(self.res_folder, "sample data")])
        f_name = os.path.join(self.res_folder, "sample data", "working data.xlsx")
        if sample:
            df = self.df.sample(n=n)
            f_name = os.path.join(self.res_folder, "sample data", "working data sample.xlsx")
        else:
            df = self.df
            f_name = os.path.join(self.res_folder, "sample data", "working data.xlsx")
        df.to_excel(f_name, index=False)
        os.startfile(f_name)

    def show_df(
        self,
        att,
    ):
        """
        Export any DataFrame attribute to Excel and open it.
        
        Looks up a DataFrame stored as an attribute on `self`, writes it to an
        Excel file in a temporary subfolder of the results directory and opens
        the file with the default system application.
        
        Parameters
        ----------
        att : str
            Name of the attribute on `self` holding the DataFrame.
        """
        utilsbib.make_folders([os.path.join(self.res_folder, "tmp")])
        f_name = os.path.join(self.res_folder, "tmp", f"{att}.xlsx")
        df = getattr(self, att)
        df.to_excel(f_name, index=False)
        os.startfile(f_name)

    # =========================================================================
    # DISTRIBUTION FITTING AND ANALYSIS
    # =========================================================================
    
    def fit_citation_distributions(
        self,
        column: str = "Cited by",
        distributions: list = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Fit multiple probability distributions to citation data.
        
        Compares how well different theoretical distributions fit the observed
        citation counts. Useful for understanding citation patterns and
        comparing with theoretical models (e.g., preferential attachment).
        
        Parameters
        ----------
        column : str
            Column containing citation counts.
        distributions : list
            List of distributions to fit. If None, fits all available:
            ["lognormal", "exponential", "power_law", "poisson", 
             "negative_binomial", "weibull", "gamma"]
        verbose : bool
            Print results.
        
        Returns
        -------
        pd.DataFrame
            Fit results with parameters, AIC, BIC, and KS test statistics.
        """
        from scipy import stats
        import numpy as np
        
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        data = pd.to_numeric(self.df[column], errors="coerce").dropna()
        data = data[data > 0].values  # Most distributions need positive values
        
        if len(data) < 10:
            raise ValueError("Not enough data points for distribution fitting")
        
        if distributions is None:
            distributions = ["lognormal", "exponential", "power_law", "poisson", 
                           "negative_binomial", "weibull", "gamma"]
        
        results = []
        n = len(data)
        
        for dist_name in distributions:
            try:
                if dist_name == "lognormal":
                    # Log-normal distribution
                    shape, loc, scale = stats.lognorm.fit(data, floc=0)
                    params = {"shape": shape, "loc": loc, "scale": scale}
                    log_likelihood = np.sum(stats.lognorm.logpdf(data, shape, loc, scale))
                    ks_stat, ks_pval = stats.kstest(data, "lognorm", args=(shape, loc, scale))
                    k = 3  # number of parameters
                    
                elif dist_name == "exponential":
                    # Exponential distribution
                    loc, scale = stats.expon.fit(data)
                    params = {"loc": loc, "scale": scale}
                    log_likelihood = np.sum(stats.expon.logpdf(data, loc, scale))
                    ks_stat, ks_pval = stats.kstest(data, "expon", args=(loc, scale))
                    k = 2
                    
                elif dist_name == "power_law":
                    # Power law (Pareto) distribution
                    # Fit using MLE for alpha: alpha = 1 + n / sum(log(x/xmin))
                    xmin = data.min()
                    alpha = 1 + n / np.sum(np.log(data / xmin))
                    params = {"alpha": alpha, "xmin": xmin}
                    # Log-likelihood for power law
                    log_likelihood = n * np.log(alpha - 1) - n * np.log(xmin) - alpha * np.sum(np.log(data / xmin))
                    # KS test against Pareto
                    b, loc, scale = stats.pareto.fit(data, floc=0)
                    ks_stat, ks_pval = stats.kstest(data, "pareto", args=(b, loc, scale))
                    k = 2
                    
                elif dist_name == "poisson":
                    # Poisson distribution (for discrete counts)
                    lambda_param = data.mean()
                    params = {"lambda": lambda_param}
                    data_int = data.astype(int)
                    log_likelihood = np.sum(stats.poisson.logpmf(data_int, lambda_param))
                    # KS test (approximate for discrete)
                    ks_stat, ks_pval = stats.kstest(data_int, stats.poisson(lambda_param).cdf)
                    k = 1
                    
                elif dist_name == "negative_binomial":
                    # Negative binomial (better for overdispersed counts)
                    mean_val = data.mean()
                    var_val = data.var()
                    if var_val > mean_val:  # Overdispersed
                        p = mean_val / var_val
                        r = mean_val * p / (1 - p)
                        params = {"r": r, "p": p}
                        data_int = data.astype(int)
                        log_likelihood = np.sum(stats.nbinom.logpmf(data_int, r, p))
                        ks_stat, ks_pval = stats.kstest(data_int, stats.nbinom(r, p).cdf)
                    else:
                        # Fall back to Poisson
                        params = {"r": mean_val, "p": 0.5}
                        log_likelihood = -np.inf
                        ks_stat, ks_pval = 1.0, 0.0
                    k = 2
                    
                elif dist_name == "weibull":
                    # Weibull distribution
                    c, loc, scale = stats.weibull_min.fit(data, floc=0)
                    params = {"c": c, "loc": loc, "scale": scale}
                    log_likelihood = np.sum(stats.weibull_min.logpdf(data, c, loc, scale))
                    ks_stat, ks_pval = stats.kstest(data, "weibull_min", args=(c, loc, scale))
                    k = 3
                    
                elif dist_name == "gamma":
                    # Gamma distribution
                    a, loc, scale = stats.gamma.fit(data, floc=0)
                    params = {"a": a, "loc": loc, "scale": scale}
                    log_likelihood = np.sum(stats.gamma.logpdf(data, a, loc, scale))
                    ks_stat, ks_pval = stats.kstest(data, "gamma", args=(a, loc, scale))
                    k = 3
                
                else:
                    continue
                
                # Compute AIC and BIC
                aic = 2 * k - 2 * log_likelihood
                bic = k * np.log(n) - 2 * log_likelihood
                
                results.append({
                    "Distribution": dist_name,
                    "Parameters": str(params),
                    "Log Likelihood": log_likelihood,
                    "AIC": aic,
                    "BIC": bic,
                    "KS Statistic": ks_stat,
                    "KS P-value": ks_pval,
                    "N": n,
                })
                
            except Exception as e:
                if verbose:
                    print(f"Could not fit {dist_name}: {e}")
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            # Rank by AIC (lower is better)
            results_df = results_df.sort_values("AIC")
            results_df["AIC Rank"] = range(1, len(results_df) + 1)
            
            # Best fit
            best = results_df.iloc[0]
            
            if verbose:
                print("="*60)
                print("DISTRIBUTION FITTING RESULTS")
                print("="*60)
                print(f"Data: {column} (n={n})")
                print(f"Mean: {data.mean():.2f}, Median: {np.median(data):.2f}")
                print(f"Std: {data.std():.2f}, Skewness: {stats.skew(data):.2f}")
                print(f"\nBest fit: {best['Distribution']}")
                print(f"  AIC: {best['AIC']:.2f}")
                print(f"  KS p-value: {best['KS P-value']:.4f}")
                print(f"\nAll results (sorted by AIC):")
                for _, row in results_df.iterrows():
                    print(f"  {row['Distribution']}: AIC={row['AIC']:.1f}, KS p={row['KS P-value']:.4f}")
        
        self.distribution_fit_results = results_df
        return results_df
    
    def fit_productivity_distributions(
        self,
        column: str = "Authors",
        distributions: list = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Fit distributions to author productivity data.
        
        Analyzes how papers are distributed among authors - relates to
        Lotka's Law and scientific productivity patterns.
        
        Parameters
        ----------
        column : str
            Column containing author names (will count papers per author).
        distributions : list
            Distributions to fit.
        verbose : bool
            Print results.
        
        Returns
        -------
        pd.DataFrame
            Fit results.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        # Count papers per author
        sep = self.default_separator
        all_authors = []
        for val in self.df[column].dropna():
            authors = [a.strip() for a in str(val).split(sep) if a.strip()]
            all_authors.extend(authors)
        
        author_counts = pd.Series(all_authors).value_counts()
        
        # Use the counts as data
        data = author_counts.values
        
        if verbose:
            print(f"Analyzing productivity of {len(author_counts)} unique authors")
            print(f"Total authorships: {len(all_authors)}")
        
        # Create temporary column and fit
        temp_df = pd.DataFrame({"_productivity": data})
        original_df = self.df
        self.df = temp_df
        
        try:
            results = self.fit_citation_distributions("_productivity", distributions, verbose)
        finally:
            self.df = original_df
        
        self.productivity_fit_results = results
        return results
    
    def fit_growth_model(
        self,
        model_type: str = "auto",
        forecast_years: int = 5,
        year_col: str = None,
        min_year: int = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Fit bibliometric growth models to publication data.
        
        Fits exponential, logistic, power law, or linear models to annual 
        publication counts. Models the growth pattern of the research field
        and provides forecasts.
        
        Parameters
        ----------
        model_type : str
            Model type: "exponential", "logistic", "power", "linear", "auto".
            - exponential: y = a * exp(b * t) - constant rate of growth
            - logistic: y = L / (1 + exp(-k*(t-t0))) - S-curve with carrying capacity
            - power: y = a * (t+1)^b - polynomial growth
            - linear: y = a*t + b - constant absolute growth
            - auto: select best model by AIC
        forecast_years : int
            Number of years to forecast beyond the data.
        year_col : str, optional
            Year column name. If None, auto-detected.
        min_year : int, optional
            Minimum year to include. If None, uses earliest year >= 1900.
        verbose : bool
            Print results.
        
        Returns
        -------
        dict
            Dictionary containing:
            - model_type: str - The fitted model type
            - parameters: dict - Model parameters
            - r_squared: float - Coefficient of determination
            - aic: float - Akaike Information Criterion
            - growth_rate: float - Annual growth rate
            - doubling_time: float or None - Time to double
            - prediction_df: pd.DataFrame - Historical + forecast data
            - comparison_df: pd.DataFrame - Comparison of all models
        
        Examples
        --------
        >>> # Fit best model automatically
        >>> result = bib.fit_growth_model()
        >>> print(f"Best model: {result['model_type']}")
        >>> print(f"R²: {result['r_squared']:.3f}")
        
        >>> # Fit specific model with forecast
        >>> result = bib.fit_growth_model(model_type="logistic", forecast_years=10)
        >>> print(result['prediction_df'])
        """
        # Get year column
        if year_col is None:
            year_col = self._get_column(["Year", "Publication Year", "PY"])
        
        result = utilsbib.fit_growth_model(
            self.df,
            year_col=year_col,
            model_type=model_type,
            forecast_years=forecast_years,
            min_year=min_year,
        )
        
        if verbose:
            print(f"\n📈 Growth Model Analysis")
            print(f"=" * 50)
            print(f"Selected model: {result['model_type'].title()}")
            print(f"R²: {result['r_squared']:.4f}")
            print(f"AIC: {result['aic']:.2f}")
            print(f"Growth rate: {result['growth_rate']:.4f}")
            if result['doubling_time']:
                print(f"Doubling time: {result['doubling_time']:.1f} years")
            
            print(f"\nParameters:")
            for name, val in result['parameters'].items():
                print(f"  {name}: {val:.4f}")
            
            print(f"\nModel comparison (by AIC):")
            print(result['comparison_df'].to_string(index=False))
            
            # Show forecast
            forecast = result['prediction_df'][result['prediction_df']['Is Forecast']]
            if len(forecast) > 0:
                print(f"\nForecast ({len(forecast)} years):")
                print(forecast[['Year', 'Fitted']].to_string(index=False))
        
        self.growth_model_result = result
        return result
    
    def fit_life_cycle_model(
        self,
        forecast_years: int = 10,
        year_col: str = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Fit a life cycle (logistic S-curve) model to cumulative publications.
        
        Analyzes the maturity of a research field by fitting a logistic growth
        model to cumulative publication data. Estimates carrying capacity
        (saturation level) and identifies the current growth phase.
        
        Parameters
        ----------
        forecast_years : int
            Years to forecast.
        year_col : str, optional
            Year column. If None, auto-detected.
        verbose : bool
            Print results.
        
        Returns
        -------
        dict
            Dictionary containing:
            - saturation_k: float - Carrying capacity (max total pubs)
            - peak_year: float - Inflection point year
            - growth_rate: float - Intrinsic growth rate
            - growth_duration: float - Years from 10% to 90% saturation
            - current_phase: str - "emerging", "growth", "maturity", "saturation"
            - progress: float - Current progress to saturation (0-1)
            - r_squared: float - Model fit quality
            - prediction_df: pd.DataFrame - Historical + forecast
        
        Examples
        --------
        >>> result = bib.fit_life_cycle_model()
        >>> print(f"Phase: {result['current_phase']}")
        >>> print(f"Progress: {result['progress']*100:.1f}%")
        """
        if year_col is None:
            year_col = self._get_column(["Year", "Publication Year", "PY"])
        
        result = utilsbib.fit_life_cycle_model(
            self.df,
            year_col=year_col,
            forecast_years=forecast_years,
        )
        
        if verbose:
            print(f"\n🔄 Life Cycle Analysis")
            print(f"=" * 50)
            print(f"Current phase: {result['current_phase'].upper()}")
            print(f"Progress to saturation: {result['progress']*100:.1f}%")
            print(f"R²: {result['r_squared']:.4f}")
            
            print(f"\nModel parameters:")
            print(f"  Saturation (K): {result['saturation_k']:,.0f} publications")
            print(f"  Peak year (Tm): {result['peak_year']:.1f}")
            print(f"  Growth rate (r): {result['growth_rate']:.4f}")
            print(f"  Growth duration: {result['growth_duration']:.1f} years (10% → 90%)")
            print(f"  Peak annual rate: {result['peak_annual']:,.0f} pubs/year")
            
            # Forecast summary
            forecast = result['prediction_df'][result['prediction_df']['Is Forecast']]
            if len(forecast) > 0:
                last_forecast = forecast.iloc[-1]
                print(f"\nForecast to {int(last_forecast['Year'])}:")
                print(f"  Cumulative: {last_forecast['Fitted Cumulative']:,.0f}")
                print(f"  Annual: {last_forecast['Fitted Annual']:,.0f}")
        
        self.life_cycle_result = result
        return result
    
    def analyze_citation_distribution(
        self,
        citations_col: str = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze citation distribution characteristics.
        
        Computes comprehensive statistics about the citation distribution including
        basic stats, percentiles, inequality measures (Gini, h-index), and 
        distribution fitting.
        
        Parameters
        ----------
        citations_col : str, optional
            Column containing citation counts. If None, auto-detected.
        verbose : bool
            Print summary results.
        
        Returns
        -------
        dict
            Dictionary containing:
            - n_papers: Total number of papers
            - basic_stats: Mean, median, std, max, min, sum
            - percentiles: 25th, 50th, 75th, 90th, 95th, 99th percentiles
            - uncited: Count and proportion of uncited papers
            - highly_cited: Threshold and count for top 10%
            - h_index: Hirsch index
            - g_index: Egghe's g-index
            - gini_coefficient: Citation inequality measure
            - skewness: Distribution skewness
            - kurtosis: Distribution kurtosis
            - distribution_fit: Log-normal fit parameters
            - citation_classes: Papers grouped by citation class
            - class_distribution: Count per citation class
        
        Examples
        --------
        >>> result = bib.analyze_citation_distribution()
        >>> print(f"H-index: {result['h_index']}")
        >>> print(f"Gini: {result['gini_coefficient']:.3f}")
        >>> print(f"Uncited: {result['uncited']['percentage']:.1f}%")
        """
        from biblium.utilsbib_modules.stats import analyze_citation_distribution
        
        # Get citations column
        if citations_col is None:
            citations_col = self._get_column([
                "Cited by", "Citations", "cited_by", "TC", 
                "Times Cited", "Citation Count"
            ])
        
        result = analyze_citation_distribution(self.df, citations_col=citations_col)
        
        if verbose:
            print(f"\n📊 Citation Distribution Analysis")
            print(f"=" * 50)
            print(f"Papers analyzed: {result['n_papers']:,}")
            print(f"\nBasic Statistics:")
            print(f"  Mean citations: {result['basic_stats']['mean']:.2f}")
            print(f"  Median citations: {result['basic_stats']['median']:.1f}")
            print(f"  Std deviation: {result['basic_stats']['std']:.2f}")
            print(f"  Max citations: {result['basic_stats']['max']:,}")
            print(f"  Total citations: {result['basic_stats']['sum']:,}")
            
            print(f"\nDistribution Metrics:")
            print(f"  H-index: {result['h_index']}")
            print(f"  G-index: {result['g_index']}")
            print(f"  Gini coefficient: {result['gini_coefficient']:.3f}")
            print(f"  Skewness: {result['skewness']:.2f}")
            
            print(f"\nUncited Papers:")
            print(f"  Count: {result['uncited']['count']:,} ({result['uncited']['percentage']:.1f}%)")
            
            print(f"\nHighly Cited (Top 10%):")
            print(f"  Threshold: ≥{result['highly_cited']['threshold']:.0f} citations")
            print(f"  Count: {result['highly_cited']['count']:,}")
            
            print(f"\nPercentiles:")
            for p, v in result['percentiles'].items():
                print(f"  {p}th: {v:.0f}")
            
            print(f"\nCitation Classes:")
            for cls, data in result['class_distribution'].items():
                print(f"  {cls}: {data['count']:,} ({data['percentage']:.1f}%)")
        
        self.citation_distribution_result = result
        return result
    
    def analyze_collaboration(
        self,
        authors_col: str = None,
        year_col: str = None,
        sep: str = "; ",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze collaboration patterns in bibliographic data.
        
        Computes collaboration metrics including Collaboration Index,
        Degree of Collaboration, and Collaboration Coefficient, along
        with author count distributions and temporal trends.
        
        Parameters
        ----------
        authors_col : str, optional
            Column containing author names. If None, auto-detected.
        year_col : str, optional
            Column containing publication year. If None, auto-detected.
        sep : str
            Separator between author names.
        verbose : bool
            Print summary results.
        
        Returns
        -------
        dict
            Dictionary containing:
            - collaboration_index: Mean authors per paper
            - degree_of_collaboration: Proportion of multi-authored papers
            - collaboration_coefficient: Modified collaboration measure
            - single_author_papers: Count of single-author papers
            - multi_author_papers: Count of multi-author papers
            - author_distribution: Distribution of author counts
            - temporal_trend: Collaboration metrics over time
            - collaboration_types: Papers by team size category
        
        Examples
        --------
        >>> result = bib.analyze_collaboration()
        >>> print(f"CI: {result['collaboration_index']:.2f}")
        >>> print(f"DC: {result['degree_of_collaboration']:.2%}")
        """
        from biblium.utilsbib_modules.stats import analyze_collaboration
        
        # Get authors column
        if authors_col is None:
            authors_col = self._get_column([
                "Authors", "Author", "authors", "AU", "Authors or Inventors"
            ])
        
        # Get year column
        if year_col is None:
            year_col = self._get_column(["Year", "Publication Year", "PY"])
        
        result = analyze_collaboration(
            self.df,
            authors_col=authors_col,
            year_col=year_col,
            sep=sep,
        )
        
        if verbose:
            print(f"\n👥 Collaboration Analysis")
            print(f"=" * 50)
            print(f"Papers analyzed: {result['n_papers']:,}")
            
            print(f"\nCollaboration Metrics:")
            print(f"  Collaboration Index (CI): {result['collaboration_index']:.2f}")
            print(f"  Degree of Collaboration (DC): {result['degree_of_collaboration']:.3f} ({result['degree_of_collaboration']*100:.1f}%)")
            print(f"  Collaboration Coefficient (CC): {result['collaboration_coefficient']:.3f}")
            
            print(f"\nAuthorship:")
            print(f"  Single-author papers: {result['single_author_papers']:,} ({result['single_author_papers']/result['n_papers']*100:.1f}%)")
            print(f"  Multi-author papers: {result['multi_author_papers']:,} ({result['multi_author_papers']/result['n_papers']*100:.1f}%)")
            print(f"  Max authors on a paper: {result['max_authors']}")
            
            print(f"\nAuthor Statistics:")
            print(f"  Mean: {result['basic_stats']['mean']:.2f}")
            print(f"  Median: {result['basic_stats']['median']:.1f}")
            print(f"  Mode: {result['basic_stats']['mode']}")
            print(f"  Std: {result['basic_stats']['std']:.2f}")
            
            print(f"\nCollaboration Types:")
            for _, row in result['collaboration_types'].iterrows():
                print(f"  {row['Type']}: {row['Count']:,} ({row['Percentage']:.1f}%)")
        
        self.collaboration_result = result
        return result
    
    def analyze_distribution_over_time(
        self,
        column: str = "Cited by",
        year_column: str = "Year",
        window_size: int = 5,
        distribution: str = "lognormal",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Analyze how distribution parameters change over time.
        
        Fits the specified distribution to data in each time window
        and tracks parameter evolution.
        
        Parameters
        ----------
        column : str
            Column to analyze.
        year_column : str
            Year column.
        window_size : int
            Years per window.
        distribution : str
            Distribution to fit in each window.
        verbose : bool
            Print results.
        
        Returns
        -------
        pd.DataFrame
            Parameters by time period.
        """
        from scipy import stats
        import numpy as np
        
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")
        if year_column not in self.df.columns:
            raise ValueError(f"Year column '{year_column}' not found")
        
        df_clean = self.df[[column, year_column]].copy()
        df_clean[year_column] = pd.to_numeric(df_clean[year_column], errors="coerce")
        df_clean[column] = pd.to_numeric(df_clean[column], errors="coerce")
        df_clean = df_clean.dropna(subset=[column, year_column])
        
        if len(df_clean) == 0:
            raise ValueError("No valid data after removing NaN values")
        
        min_year = int(df_clean[year_column].min())
        max_year = int(df_clean[year_column].max())
        
        results = []
        
        for start in range(min_year, max_year, window_size):
            end = min(start + window_size - 1, max_year)
            mask = (df_clean[year_column] >= start) & (df_clean[year_column] <= end)
            data = df_clean.loc[mask, column].values
            data = data[data > 0]
            
            if len(data) < 10:
                continue
            
            period = f"{start}-{end}"
            
            try:
                if distribution == "lognormal":
                    shape, loc, scale = stats.lognorm.fit(data, floc=0)
                    results.append({
                        "Period": period,
                        "N": len(data),
                        "Mean": data.mean(),
                        "Median": np.median(data),
                        "Shape": shape,
                        "Scale": scale,
                    })
                elif distribution == "power_law":
                    xmin = data.min()
                    alpha = 1 + len(data) / np.sum(np.log(data / xmin))
                    results.append({
                        "Period": period,
                        "N": len(data),
                        "Mean": data.mean(),
                        "Median": np.median(data),
                        "Alpha": alpha,
                        "Xmin": xmin,
                    })
                elif distribution == "exponential":
                    loc, scale = stats.expon.fit(data)
                    results.append({
                        "Period": period,
                        "N": len(data),
                        "Mean": data.mean(),
                        "Median": np.median(data),
                        "Rate": 1/scale,
                        "Scale": scale,
                    })
            except Exception as e:
                if verbose:
                    print(f"Could not fit {period}: {e}")
        
        results_df = pd.DataFrame(results)
        
        if verbose and len(results_df) > 0:
            print("="*60)
            print(f"TEMPORAL DISTRIBUTION ANALYSIS ({distribution})")
            print("="*60)
            print(results_df.to_string(index=False))
        
        self.temporal_distribution_results = results_df
        return results_df

    # =========================================================================
    # EXPORT METHODS
    # =========================================================================
    
    def export_all_tables(
        self,
        output_path: str = "biblium_tables.xlsx",
        include_stats: bool = True,
    ) -> str:
        """
        Export all computed tables to a single Excel workbook.
        
        Parameters
        ----------
        output_path : str
            Path for the output Excel file.
        include_stats : bool, default True
            If True, also compute and include stats tables.
            
        Returns
        -------
        str
            Path to the created file.
        """
        from pathlib import Path
        
        # Ensure counts are computed
        self.count_sources()
        self.count_authors()
        self.count_author_keywords()
        self.count_document_types()
        self.get_production()
        
        if include_stats:
            try:
                self.get_sources_stats()
            except:
                pass
            try:
                self.get_authors_stats()
            except:
                pass
            try:
                self.get_author_keywords_stats()
            except:
                pass
        
        # Collect all tables
        tables = {}
        
        # Main data summary
        summary_data = {
            "Metric": ["Total Documents", "Time Span", "Sources", "Authors", "Keywords"],
            "Value": [
                self.n,
                f"{self.df[self._get_column('Year')].min()}-{self.df[self._get_column('Year')].max()}" if self._get_column("Year", required=False) else "N/A",
                len(self.sources_counts_df) if hasattr(self, "sources_counts_df") and self.sources_counts_df is not None else 0,
                len(self.authors_counts_df) if hasattr(self, "authors_counts_df") and self.authors_counts_df is not None else 0,
                len(self.author_keywords_counts_df) if hasattr(self, "author_keywords_counts_df") and self.author_keywords_counts_df is not None else 0,
            ]
        }
        tables["Summary"] = pd.DataFrame(summary_data)
        
        # Count tables
        if hasattr(self, "sources_counts_df") and self.sources_counts_df is not None:
            tables["Sources"] = self.sources_counts_df
        if hasattr(self, "authors_counts_df") and self.authors_counts_df is not None:
            tables["Authors"] = self.authors_counts_df
        if hasattr(self, "author_keywords_counts_df") and self.author_keywords_counts_df is not None:
            tables["Keywords"] = self.author_keywords_counts_df
        if hasattr(self, "document_types_counts_df") and self.document_types_counts_df is not None:
            tables["Document Types"] = self.document_types_counts_df
        if hasattr(self, "production_df") and self.production_df is not None:
            tables["Production"] = self.production_df
        
        # Stats tables
        if hasattr(self, "sources_stats_df") and self.sources_stats_df is not None:
            tables["Sources Stats"] = self.sources_stats_df
        if hasattr(self, "authors_stats_df") and self.authors_stats_df is not None:
            tables["Authors Stats"] = self.authors_stats_df
        if hasattr(self, "author_keywords_stats_df") and self.author_keywords_stats_df is not None:
            tables["Keywords Stats"] = self.author_keywords_stats_df
        
        # Country data if available
        if hasattr(self, "ca_country_counts_df") and self.ca_country_counts_df is not None:
            tables["Countries"] = self.ca_country_counts_df
        
        # Write to Excel
        output_path = Path(output_path)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for sheet_name, df in tables.items():
                # Truncate sheet name to 31 chars (Excel limit)
                sheet_name = sheet_name[:31]
                # Round float columns to 2 decimal places
                df_export = df.copy()
                for col in df_export.select_dtypes(include=['float64', 'float32']).columns:
                    df_export[col] = df_export[col].round(2)
                df_export.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"✓ All tables exported to: {output_path}")
        return str(output_path)
    
    def export_network_gephi(
        self,
        network: nx.Graph,
        output_path: str = "network.gexf",
        node_attributes: Optional[List[str]] = None,
    ) -> str:
        """
        Export a NetworkX graph to Gephi GEXF format.
        
        Parameters
        ----------
        network : nx.Graph
            NetworkX graph to export.
        output_path : str
            Path for the output file.
        node_attributes : list of str, optional
            Node attributes to include.
            
        Returns
        -------
        str
            Path to the created file.
        """
        nx.write_gexf(network, output_path)
        print(f"✓ Network exported to Gephi format: {output_path}")
        return output_path
    
    def export_network_pajek(
        self,
        network: nx.Graph,
        output_base: str = "network",
        partition_attr: Optional[str] = None,
        vector_attrs: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Export a NetworkX graph to Pajek format (.net) with optional
        partition (.clu) and vector (.vec) files.
        
        Parameters
        ----------
        network : nx.Graph
            NetworkX graph to export.
        output_base : str
            Base name for output files (without extension).
        partition_attr : str, optional
            Node attribute to use for partition file (.clu).
            Creates integer class assignments.
        vector_attrs : list of str, optional
            Node attributes to export as vector files (.vec).
            
        Returns
        -------
        dict
            Dictionary mapping file type to path.
        """
        from pathlib import Path
        
        output_base = Path(output_base)
        files_created = {}
        
        # Create node mapping (Pajek uses 1-indexed)
        nodes = list(network.nodes())
        node_to_idx = {n: i + 1 for i, n in enumerate(nodes)}
        
        # Write .net file (network)
        net_path = str(output_base) + ".net"
        with open(net_path, "w", encoding="utf-8") as f:
            # Vertices
            f.write(f"*Vertices {len(nodes)}\n")
            for i, node in enumerate(nodes, 1):
                # Escape quotes in node labels
                label = str(node).replace('"', "'")
                f.write(f'{i} "{label}"\n')
            
            # Edges/Arcs
            if network.is_directed():
                f.write("*Arcs\n")
            else:
                f.write("*Edges\n")
            
            for u, v, data in network.edges(data=True):
                weight = data.get("weight", 1)
                f.write(f"{node_to_idx[u]} {node_to_idx[v]} {weight}\n")
        
        files_created["net"] = net_path
        print(f"✓ Network exported to Pajek format: {net_path}")
        
        # Write .clu file (partition) if requested
        if partition_attr:
            # Get unique values and create integer mapping
            values = [network.nodes[n].get(partition_attr, 0) for n in nodes]
            unique_vals = sorted(set(values))
            val_to_class = {v: i + 1 for i, v in enumerate(unique_vals)}
            
            clu_path = str(output_base) + ".clu"
            with open(clu_path, "w", encoding="utf-8") as f:
                f.write(f"*Vertices {len(nodes)}\n")
                for node in nodes:
                    val = network.nodes[node].get(partition_attr, 0)
                    f.write(f"{val_to_class[val]}\n")
            
            files_created["clu"] = clu_path
            print(f"✓ Partition file exported: {clu_path}")
        
        # Write .vec files (vectors) if requested
        if vector_attrs:
            for attr in vector_attrs:
                vec_path = str(output_base) + f"_{attr}.vec"
                with open(vec_path, "w", encoding="utf-8") as f:
                    f.write(f"*Vertices {len(nodes)}\n")
                    for node in nodes:
                        val = network.nodes[node].get(attr, 0)
                        try:
                            val = float(val)
                        except (ValueError, TypeError):
                            val = 0.0
                        # Check if attribute is likely a year or count (integer-like)
                        if attr in ["documents", "citations", "frequency", "degree", "count"]:
                            f.write(f"{int(val)}\n")
                        else:
                            f.write(f"{val:.2f}\n")
                
                files_created[f"vec_{attr}"] = vec_path
                print(f"✓ Vector file exported: {vec_path}")
        
        return files_created
    
    def export_network_vosviewer(
        self,
        network: nx.Graph,
        output_path: str = "vosviewer_network.txt",
        map_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Export a NetworkX graph to VOSviewer format.
        
        Creates a network file and optionally a map file with coordinates.
        
        Parameters
        ----------
        network : nx.Graph
            NetworkX graph to export.
        output_path : str
            Path for the network file.
        map_path : str, optional
            Path for the map file with coordinates. If None, auto-generated.
            
        Returns
        -------
        dict
            Dictionary mapping file type to path.
        """
        from pathlib import Path
        
        files_created = {}
        nodes = list(network.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        
        # Network file (tab-separated: source, target, weight)
        with open(output_path, "w", encoding="utf-8") as f:
            for u, v, data in network.edges(data=True):
                weight = data.get("weight", 1)
                # VOSviewer uses node labels directly
                f.write(f"{u}\t{v}\t{weight}\n")
        
        files_created["network"] = output_path
        print(f"✓ Network exported to VOSviewer format: {output_path}")
        
        # Map file (tab-separated: id, label, x, y, cluster, weight)
        if map_path is None:
            map_path = str(Path(output_path).with_suffix("")) + "_map.txt"
        
        # Calculate layout
        try:
            pos = nx.spring_layout(network, seed=42)
        except:
            pos = {n: (0, 0) for n in nodes}
        
        # Get clusters if available
        try:
            from networkx.algorithms import community
            communities = list(community.greedy_modularity_communities(network))
            node_cluster = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    node_cluster[node] = i + 1
        except:
            node_cluster = {n: 1 for n in nodes}
        
        with open(map_path, "w", encoding="utf-8") as f:
            # Header
            f.write("id\tlabel\tx\ty\tcluster\tweight\n")
            
            for i, node in enumerate(nodes):
                x, y = pos.get(node, (0, 0))
                cluster = node_cluster.get(node, 1)
                weight = network.degree(node)
                # Escape tabs in labels
                label = str(node).replace("\t", " ")
                f.write(f"{i}\t{label}\t{x:.4f}\t{y:.4f}\t{cluster}\t{weight}\n")
        
        files_created["map"] = map_path
        print(f"✓ Map file exported: {map_path}")
        
        return files_created
    
    def build_coauthorship_network(
        self,
        top_n: int = 50,
        min_collabs: int = 1,
    ) -> nx.Graph:
        """
        Build a co-authorship network from the dataset.
        
        Parameters
        ----------
        top_n : int, default 50
            Number of top authors to include.
        min_collabs : int, default 1
            Minimum collaborations to create an edge.
            
        Returns
        -------
        nx.Graph
            Co-authorship network.
        """
        from collections import Counter, defaultdict
        
        # Support multiple possible author column names
        auth_col = self._get_column(["Authors", "Authors or Inventors", "Author", "AU"])
        cite_col = self._get_column(["Cited by", "Times Cited, All Databases", "TC", "Citations"], required=False)
        sep = self.default_separator
        
        # Count author documents and citations
        author_docs = Counter()
        author_cites = defaultdict(int)
        coauthorship = defaultdict(int)
        
        for idx, row in self.df.iterrows():
            if pd.isna(row[auth_col]):
                continue
            
            authors = [a.strip() for a in str(row[auth_col]).split(sep) if a.strip()]
            
            for auth in authors:
                author_docs[auth] += 1
                if cite_col and cite_col in self.df.columns and pd.notna(row[cite_col]):
                    author_cites[auth] += int(row[cite_col])
            
            # Co-authorships
            for i, a1 in enumerate(authors):
                for a2 in authors[i+1:]:
                    pair = tuple(sorted([a1, a2]))
                    coauthorship[pair] += 1
        
        # Get top authors
        top_authors = set(a for a, _ in author_docs.most_common(top_n))
        
        # Build network
        G = nx.Graph()
        
        for auth in top_authors:
            G.add_node(
                auth,
                documents=author_docs[auth],
                citations=author_cites[auth],
                label=auth,
            )
        
        for (a1, a2), count in coauthorship.items():
            if a1 in top_authors and a2 in top_authors and count >= min_collabs:
                G.add_edge(a1, a2, weight=count)
        
        # Add community detection
        try:
            from networkx.algorithms import community
            communities = list(community.greedy_modularity_communities(G))
            for i, comm in enumerate(communities):
                for node in comm:
                    G.nodes[node]["community"] = i
        except:
            pass
        
        self.coauthorship_network = G
        return G
    
    def build_keyword_cooccurrence_network(
        self,
        top_n: int = 50,
        min_cooccur: int = 2,
    ) -> nx.Graph:
        """
        Build a keyword co-occurrence network from the dataset.
        
        Parameters
        ----------
        top_n : int, default 50
            Number of top keywords to include.
        min_cooccur : int, default 2
            Minimum co-occurrences to create an edge.
            
        Returns
        -------
        nx.Graph
            Keyword co-occurrence network.
        """
        from collections import Counter, defaultdict
        
        kw_col = self._get_column("Author Keywords", required=False)
        if not kw_col or kw_col not in self.df.columns:
            kw_col = self._get_column("Index Keywords", required=False)
        
        if not kw_col or kw_col not in self.df.columns:
            raise ValueError("No keyword column found in dataset")
        
        year_col = self._get_column(["Year", "Publication Year", "PY"], required=False)
        sep = self.default_separator
        
        # Count keywords
        keyword_freq = Counter()
        keyword_years = defaultdict(list)
        cooccurrence = defaultdict(int)
        
        for idx, row in self.df.iterrows():
            if pd.isna(row[kw_col]):
                continue
            
            keywords = [k.strip().lower() for k in str(row[kw_col]).split(sep) if k.strip()]
            
            for kw in keywords:
                keyword_freq[kw] += 1
                if year_col and year_col in self.df.columns and pd.notna(row[year_col]):
                    # Convert year to numeric
                    try:
                        year_val = int(float(str(row[year_col])))
                        keyword_years[kw].append(year_val)
                    except (ValueError, TypeError):
                        pass
            
            # Co-occurrences
            for i, k1 in enumerate(keywords):
                for k2 in keywords[i+1:]:
                    if k1 != k2:
                        pair = tuple(sorted([k1, k2]))
                        cooccurrence[pair] += 1
        
        # Get top keywords
        top_keywords = set(k for k, _ in keyword_freq.most_common(top_n))
        
        # Build network
        G = nx.Graph()
        
        for kw in top_keywords:
            avg_year = np.mean(keyword_years[kw]) if keyword_years[kw] else 2020
            G.add_node(
                kw,
                frequency=keyword_freq[kw],
                avg_year=round(avg_year, 1),
                label=kw,
            )
        
        for (k1, k2), count in cooccurrence.items():
            if k1 in top_keywords and k2 in top_keywords and count >= min_cooccur:
                G.add_edge(k1, k2, weight=count)
        
        # Add community detection
        try:
            from networkx.algorithms import community
            communities = list(community.greedy_modularity_communities(G))
            for i, comm in enumerate(communities):
                for node in comm:
                    G.nodes[node]["community"] = i
        except:
            pass
        
        self.keyword_network = G
        return G
    
    def build_cocitation_network(
        self,
        top_n: int = 50,
        min_cocitations: int = 2,
    ) -> nx.Graph:
        """
        Build a co-citation network from references.
        
        Parameters
        ----------
        top_n : int, default 50
            Number of top cited references to include.
        min_cocitations : int, default 2
            Minimum co-citations to create an edge.
            
        Returns
        -------
        nx.Graph
            Co-citation network.
        """
        from collections import Counter, defaultdict
        
        # Find reference column
        ref_col = None
        for col_name in ["References", "Cited References", "Cited references", "Reference"]:
            if col_name in self.df.columns:
                ref_col = col_name
                break
        
        if not ref_col:
            raise ValueError("No reference column found in dataset")
        
        sep = self.default_separator
        
        # Count references and co-citations
        ref_counts = Counter()
        cocitation = defaultdict(int)
        
        for idx, row in self.df.iterrows():
            if pd.isna(row[ref_col]):
                continue
            
            refs = [r.strip() for r in str(row[ref_col]).split(sep) if r.strip()]
            
            for ref in refs:
                ref_counts[ref] += 1
            
            # Co-citations
            for i, r1 in enumerate(refs):
                for r2 in refs[i+1:]:
                    if r1 != r2:
                        pair = tuple(sorted([r1, r2]))
                        cocitation[pair] += 1
        
        # Get top references
        top_refs = set(r for r, _ in ref_counts.most_common(top_n))
        
        # Build network
        G = nx.Graph()
        
        for ref in top_refs:
            # Shorten reference for display
            short_ref = ref[:50] + "..." if len(ref) > 50 else ref
            G.add_node(
                ref,
                citations=ref_counts[ref],
                label=short_ref,
            )
        
        for (r1, r2), count in cocitation.items():
            if r1 in top_refs and r2 in top_refs and count >= min_cocitations:
                G.add_edge(r1, r2, weight=count)
        
        self.cocitation_network = G
        return G
    
    def export_all_networks(
        self,
        output_dir: str = "networks",
        formats: List[str] = ["gexf", "pajek", "vosviewer"],
        top_n: int = 50,
    ) -> Dict[str, Dict[str, str]]:
        """
        Build and export all available networks in multiple formats.
        
        Parameters
        ----------
        output_dir : str
            Directory for output files.
        formats : list of str
            Export formats: "gexf", "pajek", "vosviewer"
        top_n : int
            Number of top items per network.
            
        Returns
        -------
        dict
            Nested dict mapping network type -> format -> filepath.
        """
        from pathlib import Path
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Co-authorship network
        try:
            G = self.build_coauthorship_network(top_n=top_n)
            if len(G.nodes()) > 0:
                results["coauthorship"] = {}
                base = output_dir / "coauthorship"
                
                if "gexf" in formats:
                    results["coauthorship"]["gexf"] = self.export_network_gephi(
                        G, str(base) + ".gexf"
                    )
                
                if "pajek" in formats:
                    pajek_files = self.export_network_pajek(
                        G, str(base),
                        partition_attr="community",
                        vector_attrs=["documents", "citations"]
                    )
                    results["coauthorship"].update(pajek_files)
                
                if "vosviewer" in formats:
                    vos_files = self.export_network_vosviewer(
                        G, str(base) + "_vosviewer.txt"
                    )
                    results["coauthorship"].update(vos_files)
        except Exception as e:
            print(f"Could not build coauthorship network: {e}")
        
        # Keyword co-occurrence network
        try:
            G = self.build_keyword_cooccurrence_network(top_n=top_n)
            if len(G.nodes()) > 0:
                results["keywords"] = {}
                base = output_dir / "keywords"
                
                if "gexf" in formats:
                    results["keywords"]["gexf"] = self.export_network_gephi(
                        G, str(base) + ".gexf"
                    )
                
                if "pajek" in formats:
                    pajek_files = self.export_network_pajek(
                        G, str(base),
                        partition_attr="community",
                        vector_attrs=["frequency", "avg_year"]
                    )
                    results["keywords"].update(pajek_files)
                
                if "vosviewer" in formats:
                    vos_files = self.export_network_vosviewer(
                        G, str(base) + "_vosviewer.txt"
                    )
                    results["keywords"].update(vos_files)
        except Exception as e:
            print(f"Could not build keyword network: {e}")
        
        # Co-citation network
        try:
            G = self.build_cocitation_network(top_n=top_n)
            if len(G.nodes()) > 0:
                results["cocitation"] = {}
                base = output_dir / "cocitation"
                
                if "gexf" in formats:
                    results["cocitation"]["gexf"] = self.export_network_gephi(
                        G, str(base) + ".gexf"
                    )
                
                if "pajek" in formats:
                    pajek_files = self.export_network_pajek(
                        G, str(base),
                        vector_attrs=["citations"]
                    )
                    results["cocitation"].update(pajek_files)
                
                if "vosviewer" in formats:
                    vos_files = self.export_network_vosviewer(
                        G, str(base) + "_vosviewer.txt"
                    )
                    results["cocitation"].update(vos_files)
        except Exception as e:
            print(f"Could not build cocitation network: {e}")
        
        print(f"\n✓ All networks exported to: {output_dir}")
        return results
    
    def export_pdf_report(
        self,
        output_path: str = "bibliometric_report.pdf",
        title: str = "Bibliometric Analysis Report",
        include_charts: bool = True,
    ) -> str:
        """
        Export a comprehensive PDF report with tables and charts.
        
        Parameters
        ----------
        output_path : str
            Path for the output PDF file.
        title : str
            Report title.
        include_charts : bool, default True
            If True, include matplotlib charts in the report.
            
        Returns
        -------
        str
            Path to the created file.
        """
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, Image, KeepTogether
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from io import BytesIO
        from datetime import datetime
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Ensure data is computed
        self.count_sources()
        self.count_authors()
        self.count_author_keywords()
        self.count_document_types()
        self.get_production()
        
        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
        )
        
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor("#1F4E79"),
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubheading',
            parent=styles['Heading2'],
            fontSize=12,
            spaceBefore=15,
            spaceAfter=8,
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6,
        )
        
        story = []
        
        # Title
        story.append(Paragraph(title, title_style))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ParagraphStyle('Date', parent=styles['Normal'], alignment=TA_CENTER, textColor=colors.grey)
        ))
        story.append(Spacer(1, 30))
        
        # Summary section
        story.append(Paragraph("Executive Summary", heading_style))
        
        year_col = self._get_column("Year", required=False)
        cite_col = self._get_column("Cited by", required=False)
        
        year_range = "N/A"
        if year_col and year_col in self.df.columns:
            year_range = f"{int(self.df[year_col].min())}-{int(self.df[year_col].max())}"
        
        total_cites = 0
        if cite_col and cite_col in self.df.columns:
            total_cites = int(self.df[cite_col].sum())
        
        summary_data = [
            ["Metric", "Value"],
            ["Total Documents", f"{self.n:,}"],
            ["Time Span", year_range],
            ["Total Citations", f"{total_cites:,}"],
            ["Sources", f"{len(self.sources_counts_df):,}" if hasattr(self, "sources_counts_df") else "N/A"],
            ["Authors", f"{len(self.authors_counts_df):,}" if hasattr(self, "authors_counts_df") else "N/A"],
            ["Keywords", f"{len(self.author_keywords_counts_df):,}" if hasattr(self, "author_keywords_counts_df") else "N/A"],
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#F5F5F5")),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
            ('ROWHEIGHTS', (0, 0), (-1, -1), 25),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Helper to create data tables
        def create_data_table(df, title, max_rows=15):
            story.append(Paragraph(title, subheading_style))
            
            # Prepare data
            df_display = df.head(max_rows).copy()
            
            # Truncate long strings
            for col in df_display.columns:
                if df_display[col].dtype == object:
                    df_display[col] = df_display[col].apply(
                        lambda x: str(x)[:40] + "..." if len(str(x)) > 40 else str(x)
                    )
            
            # Convert to list format
            data = [list(df_display.columns)] + df_display.values.tolist()
            
            # Calculate column widths
            n_cols = len(df_display.columns)
            col_width = min(1.5*inch, (6.5*inch) / n_cols)
            
            table = Table(data, colWidths=[col_width] * n_cols)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('TOPPADDING', (0, 1), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F8F8")]),
            ]))
            
            story.append(table)
            story.append(Spacer(1, 15))
        
        # Helper to create charts
        def add_chart(fig, width=5*inch, height=3*inch):
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img = Image(buf, width=width, height=height)
            story.append(img)
            plt.close(fig)
            story.append(Spacer(1, 10))
        
        # Production chart
        if include_charts and hasattr(self, "production_df") and self.production_df is not None:
            story.append(Paragraph("Scientific Production", heading_style))
            
            prod_df = self.production_df
            year_col_prod = "Year" if "Year" in prod_df.columns else prod_df.columns[0]
            count_col = next((c for c in prod_df.columns if "number" in c.lower() or "documents" in c.lower()), prod_df.columns[1])
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(prod_df[year_col_prod], prod_df[count_col], color="#1F4E79")
            ax.set_xlabel("Year")
            ax.set_ylabel("Documents")
            ax.set_title("Annual Scientific Production")
            plt.tight_layout()
            add_chart(fig, width=6*inch, height=3*inch)
        
        story.append(PageBreak())
        
        # Sources section
        if hasattr(self, "sources_counts_df") and self.sources_counts_df is not None:
            story.append(Paragraph("Top Sources", heading_style))
            create_data_table(self.sources_counts_df, "Sources by Document Count")
            
            if include_charts:
                df = self.sources_counts_df.head(10)
                name_col = "Source" if "Source" in df.columns else df.columns[0]
                count_col = next((c for c in df.columns if "number" in c.lower()), df.columns[-1])
                
                fig, ax = plt.subplots(figsize=(8, 4))
                y_pos = range(len(df))
                ax.barh(y_pos, df[count_col], color="#1F4E79")
                ax.set_yticks(y_pos)
                ax.set_yticklabels([s[:30] for s in df[name_col]], fontsize=8)
                ax.invert_yaxis()
                ax.set_xlabel("Documents")
                ax.set_title("Top 10 Sources")
                plt.tight_layout()
                add_chart(fig, width=6*inch, height=3.5*inch)
        
        story.append(PageBreak())
        
        # Authors section
        if hasattr(self, "authors_counts_df") and self.authors_counts_df is not None:
            story.append(Paragraph("Top Authors", heading_style))
            create_data_table(self.authors_counts_df, "Authors by Document Count")
            
            if include_charts:
                df = self.authors_counts_df.head(10)
                name_col = "Author" if "Author" in df.columns else df.columns[0]
                count_col = next((c for c in df.columns if "number" in c.lower()), df.columns[-1])
                
                fig, ax = plt.subplots(figsize=(8, 4))
                y_pos = range(len(df))
                ax.barh(y_pos, df[count_col], color="#2E86AB")
                ax.set_yticks(y_pos)
                ax.set_yticklabels([s[:25] for s in df[name_col]], fontsize=8)
                ax.invert_yaxis()
                ax.set_xlabel("Documents")
                ax.set_title("Top 10 Authors")
                plt.tight_layout()
                add_chart(fig, width=6*inch, height=3.5*inch)
        
        story.append(PageBreak())
        
        # Keywords section
        if hasattr(self, "author_keywords_counts_df") and self.author_keywords_counts_df is not None:
            story.append(Paragraph("Top Keywords", heading_style))
            create_data_table(self.author_keywords_counts_df, "Keywords by Frequency")
            
            if include_charts:
                df = self.author_keywords_counts_df.head(15)
                name_col = "Keyword" if "Keyword" in df.columns else df.columns[0]
                count_col = next((c for c in df.columns if "number" in c.lower()), df.columns[-1])
                
                fig, ax = plt.subplots(figsize=(8, 5))
                y_pos = range(len(df))
                ax.barh(y_pos, df[count_col], color="#A23B72")
                ax.set_yticks(y_pos)
                ax.set_yticklabels([s[:30] for s in df[name_col]], fontsize=8)
                ax.invert_yaxis()
                ax.set_xlabel("Frequency")
                ax.set_title("Top 15 Keywords")
                plt.tight_layout()
                add_chart(fig, width=6*inch, height=4*inch)
        
        # Citation analysis
        if cite_col and cite_col in self.df.columns:
            story.append(PageBreak())
            story.append(Paragraph("Citation Analysis", heading_style))
            
            citations = self.df[cite_col].fillna(0)
            
            # H-index calculation
            sorted_cites = sorted(citations, reverse=True)
            h_index = 0
            for i, c in enumerate(sorted_cites, 1):
                if c >= i:
                    h_index = i
                else:
                    break
            
            cite_stats = [
                ["Metric", "Value"],
                ["Total Citations", f"{int(citations.sum()):,}"],
                ["Average Citations", f"{citations.mean():.2f}"],
                ["Median Citations", f"{citations.median():.1f}"],
                ["Max Citations", f"{int(citations.max()):,}"],
                ["H-index", str(h_index)],
                ["Uncited Papers", f"{(citations == 0).sum()} ({100*(citations == 0).sum()/len(citations):.1f}%)"],
            ]
            
            cite_table = Table(cite_stats, colWidths=[2.5*inch, 2*inch])
            cite_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.white),
                ('ROWHEIGHTS', (0, 0), (-1, -1), 22),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#F5F5F5")),
            ]))
            story.append(cite_table)
            story.append(Spacer(1, 20))
            
            # Top cited papers
            if self._get_column("Title", required=False):
                title_col = self._get_column("Title")
                top_cited = self.df.nlargest(10, cite_col)[[title_col, cite_col]].copy()
                top_cited[title_col] = top_cited[title_col].apply(lambda x: str(x)[:60])
                top_cited.columns = ["Title", "Citations"]
                
                story.append(Paragraph("Most Cited Documents", subheading_style))
                
                data = [list(top_cited.columns)] + top_cited.values.tolist()
                top_table = Table(data, colWidths=[4.5*inch, 1.5*inch])
                top_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                    ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F8F8")]),
                ]))
                story.append(top_table)
        
        # Build PDF
        doc.build(story)
        
        print(f"✓ PDF report exported to: {output_path}")
        return output_path

    # =========================================================================
    # DATA QUALITY METHODS
    # =========================================================================
    
    def disambiguate_authors(
        self,
        similarity_threshold: float = 0.85,
        method: str = "hybrid",
        use_affiliations: bool = True,
        apply_to_df: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Disambiguate author names in the dataset.
        
        Groups author name variants that likely refer to the same person
        based on name similarity, affiliations, and ORCID identifiers.
        
        Parameters
        ----------
        similarity_threshold : float
            Minimum similarity score to consider names as variants (0-1).
        method : str
            Similarity method: "exact", "initials", "fuzzy", "hybrid"
        use_affiliations : bool
            If True, use affiliations to help disambiguation.
        apply_to_df : bool
            If True, add "Disambiguated Authors" column to self.df.
        verbose : bool
            If True, print progress information.
            
        Returns
        -------
        pd.DataFrame
            Report of author variants found.
        """
        from biblium.utilsbib_modules.data_quality import (
            disambiguate_authors as _disambiguate,
            apply_author_disambiguation,
            get_author_variants_report,
        )
        
        auth_col = self._get_column(["Authors", "Authors or Inventors", "Author", "AU"])
        aff_col = self._get_column("Affiliations", required=False)
        orcid_col = self._get_column("ORCID numbers", required=False)
        
        clusters, mapping = _disambiguate(
            self.df,
            author_column=auth_col,
            separator=self.default_separator,
            affiliation_column=aff_col,
            orcid_column=orcid_col,
            similarity_threshold=similarity_threshold,
            method=method,
            use_affiliations=use_affiliations,
            verbose=verbose,
        )
        
        self.author_clusters = clusters
        self.author_mapping = mapping
        
        if apply_to_df:
            self.df = apply_author_disambiguation(
                self.df, mapping,
                author_column=auth_col,
                separator=self.default_separator,
            )
        
        report = get_author_variants_report(clusters, min_variants=2)
        self.author_variants_report = report
        self._save_table(report, "author_variants")
        
        return report
    
    def detect_duplicates(
        self,
        similarity_threshold: float = 0.90,
        remove: bool = False,
        keep: str = "most_cited",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Detect potential duplicate documents in the dataset.
        
        Parameters
        ----------
        similarity_threshold : float
            Minimum title similarity to consider as duplicate.
        remove : bool
            If True, remove duplicates from self.df.
        keep : str
            Which duplicate to keep: "first", "last", "most_cited"
        verbose : bool
            If True, print progress.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with duplicate pairs and similarity scores.
        """
        from biblium.utilsbib_modules.data_quality import (
            detect_duplicate_documents,
            remove_duplicates,
        )
        
        title_col = self._get_column(["Title", "TI", "Document Title"])
        doi_col = self._get_column("DOI", required=False)
        year_col = self._get_column(["Year", "Publication Year", "PY"], required=False)
        auth_col = self._get_column(["Authors", "Authors or Inventors", "Author", "AU"], required=False)
        cite_col = self._get_column(["Cited by", "Times Cited, All Databases", "TC"], required=False)
        
        duplicates_df, duplicate_groups = detect_duplicate_documents(
            self.df,
            title_column=title_col,
            doi_column=doi_col,
            year_column=year_col,
            author_column=auth_col,
            similarity_threshold=similarity_threshold,
            verbose=verbose,
        )
        
        self.duplicates_df = duplicates_df
        self.duplicate_groups = duplicate_groups
        
        if remove and duplicate_groups:
            n_before = len(self.df)
            self.df = remove_duplicates(
                self.df, duplicate_groups,
                keep=keep,
                citation_column=cite_col,
            )
            self.n = len(self.df)
            if verbose:
                print(f"Removed {n_before - self.n} duplicates. New count: {self.n}")
        
        self._save_table(duplicates_df, "duplicates")
        return duplicates_df
    
    def analyze_data_quality(
        self,
        verbose: bool = True,
    ) -> Tuple[float, pd.DataFrame]:
        """
        Analyze overall data quality of the dataset.
        
        Parameters
        ----------
        verbose : bool
            If True, print summary.
            
        Returns
        -------
        tuple
            (quality_score, details_df)
        """
        from biblium.utilsbib_modules.data_quality import (
            analyze_missing_data,
            get_data_quality_score,
        )
        
        # Missing data analysis
        missing_report = analyze_missing_data(self.df)
        self.missing_data_report = missing_report
        
        # Quality score
        score, details = get_data_quality_score(self.df)
        self.data_quality_score = score
        self.data_quality_details = details
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"DATA QUALITY REPORT")
            print(f"{'='*50}")
            print(f"\nOverall Quality Score: {score}/100")
            print(f"\nMissing Data Summary (top issues):")
            top_missing = missing_report[missing_report["% Missing"] > 0].head(5)
            if len(top_missing) > 0:
                for _, row in top_missing.iterrows():
                    print(f"  {row['Column']}: {row['% Missing']:.1f}% missing")
            else:
                print("  No missing data in key columns!")
            print(f"\nKey Column Details:")
            print(details[["Column", "Completeness %"]].to_string(index=False))
        
        self._save_table(missing_report, "missing_data")
        self._save_table(details, "quality_details")
        
        return score, details

    # =========================================================================
    # ADVANCED CITATION ANALYSIS
    # =========================================================================
    
    def compute_normalized_citations(
        self,
        field_col: Optional[str] = None,
        method: str = "mean",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Compute field-normalized citation impact (FNCI) for all documents.
        
        Similar to Scopus FWCI, this normalizes citations by comparing to
        the average of similar documents (same field and year).
        
        FNCI > 1: Above average impact
        FNCI = 1: Average impact
        FNCI < 1: Below average impact
        
        Parameters
        ----------
        field_col : str, optional
            Column for field classification. If None, normalizes by year only.
            Common options: 'Source title', 'Index Keywords', 'Field', 'Area'
        method : str
            'mean' or 'median' for computing expected citations.
        verbose : bool
            If True, print summary statistics.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with FNCI columns added.
        """
        from biblium.utilsbib_modules.citations import (
            compute_field_normalized_citations,
            compute_citation_classes,
            get_citation_summary,
        )
        
        cite_col = self._get_column("Cited by")
        year_col = self._get_column("Year")
        
        # Try to find a field column if not specified
        if field_col is None:
            for candidate in ["Field", "Area", "Source title", "Primary Topic"]:
                if candidate in self.df.columns:
                    field_col = candidate
                    break
        
        self.df = compute_field_normalized_citations(
            self.df,
            citations_col=cite_col,
            year_col=year_col,
            field_col=field_col,
            method=method,
        )
        
        self.df = compute_citation_classes(self.df)
        
        if verbose:
            summary = get_citation_summary(self.df, cite_col, "FNCI")
            print(f"\nCitation Analysis Summary:")
            print(f"  Total citations: {summary['total_citations']:,}")
            print(f"  Mean citations: {summary['mean_citations']:.2f}")
            print(f"  H-index: {summary['h_index']}")
            print(f"  Uncited papers: {summary['uncited_papers']} ({summary['uncited_percentage']:.1f}%)")
            if 'mean_fnci' in summary:
                print(f"  Mean FNCI: {summary['mean_fnci']:.3f}")
                print(f"  Papers above average (FNCI≥1): {summary['papers_above_average']}")
        
        return self.df[["Title", "Year", "Cited by", "Expected Citations", "FNCI", "FNCI Percentile", "Citation Class"]]
    
    def detect_self_citations(
        self,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Detect self-citations in the dataset.
        
        A self-citation occurs when a paper cites another paper by the same author(s).
        
        Parameters
        ----------
        verbose : bool
            If True, print summary statistics.
            
        Returns
        -------
        pd.DataFrame
            Summary of self-citation patterns.
        """
        from biblium.utilsbib_modules.citations import (
            detect_self_citations,
            get_self_citation_summary,
        )
        
        auth_col = self._get_column(["Authors", "Authors or Inventors", "Author", "AU"])
        ref_col = self._get_column("References", required=False)
        
        if ref_col is None:
            print("Warning: References column not found. Cannot detect self-citations.")
            return pd.DataFrame()
        
        self.df = detect_self_citations(
            self.df,
            authors_col=auth_col,
            references_col=ref_col,
            separator=self.default_separator,
        )
        
        if verbose:
            summary = get_self_citation_summary(self.df)
            print(f"\nSelf-Citation Analysis:")
            print(f"  Total self-citations: {summary['total_self_citations']}")
            print(f"  Papers with self-citations: {summary['papers_with_self_citations']} ({summary['percentage_with_self_citations']:.1f}%)")
            print(f"  Mean self-citation rate: {summary['mean_self_citation_rate']:.1f}%")
        
        # Create summary table
        summary_df = self.df[self.df["Self Citations Count"] > 0][
            ["Title", "Authors", "Self Citations Count", "Self Citation Rate", "Self Citing Authors"]
        ].sort_values("Self Citation Rate", ascending=False)
        
        self._save_table(summary_df, "self_citations")
        
        return summary_df
    
    def build_doi_citation_network(
        self,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Build a citation network using DOI-based reference matching.
        
        This matches references to documents within the dataset using DOIs,
        creating an internal citation network.
        
        Parameters
        ----------
        verbose : bool
            If True, print network statistics.
            
        Returns
        -------
        tuple
            (citation_network_df, stats_dict)
        """
        from biblium.utilsbib_modules.citations import build_citation_network_from_dois
        
        doi_col = self._get_column("DOI", required=False)
        ref_col = self._get_column("References", required=False)
        
        if doi_col is None or ref_col is None:
            print("Warning: DOI or References column not found.")
            return pd.DataFrame(), {}
        
        citation_network, stats = build_citation_network_from_dois(
            self.df,
            doi_col=doi_col,
            references_col=ref_col,
            separator=self.default_separator,
        )
        
        self.doi_citation_network = citation_network
        self.doi_citation_stats = stats
        
        if verbose:
            print(f"\nDOI-Based Citation Network:")
            print(f"  Documents with DOI: {stats['documents_with_doi']}")
            print(f"  References with DOI: {stats['total_references_with_doi']}")
            print(f"  Matched within dataset: {stats['matched_references']}")
            print(f"  Citation edges: {stats['citation_edges']}")
            print(f"  Mean internal citation rate: {stats['mean_internal_citation_rate']:.1f}%")
        
        self._save_table(citation_network, "doi_citation_network")
        
        return citation_network, stats
    
    def compute_citation_velocity(
        self,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Compute citation velocity metrics.
        
        Citation velocity measures how quickly papers accumulate citations.
        
        Parameters
        ----------
        verbose : bool
            If True, print summary.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with velocity metrics.
        """
        from biblium.utilsbib_modules.citations import compute_citation_velocity
        
        cite_col = self._get_column("Cited by")
        year_col = self._get_column("Year")
        
        self.df = compute_citation_velocity(
            self.df,
            citations_col=cite_col,
            year_col=year_col,
        )
        
        if verbose:
            velocity_counts = self.df["Citation Velocity Class"].value_counts()
            print(f"\nCitation Velocity Distribution:")
            for cls, count in velocity_counts.items():
                pct = 100 * count / len(self.df)
                print(f"  {cls}: {count} ({pct:.1f}%)")
        
        return self.df[["Title", "Year", "Cited by", "Age", "Citations per Year", "Citation Velocity Class"]]