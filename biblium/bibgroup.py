# -*- coding: utf-8 -*-
"""
BiblioGroup - Group-based bibliometric analysis.

This module provides the BiblioGroup class for comparative analysis across
multiple groups defined by a grouping variable (e.g., year, country, topic).
"""

from __future__ import annotations

import os
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

import pandas as pd

from biblium import utilsbib
from biblium.base import BiblioBase
from biblium.bibstats import BiblioStats

# Import mixins
from biblium.bibgroup_modules.counting import GroupCountingMixin
from biblium.bibgroup_modules.stats import GroupStatsMixin
from biblium.bibgroup_modules.associations import GroupAssociationsMixin
from biblium.bibgroup_modules.analysis import GroupAnalysisMixin


def initialize_biblio_common(*args: Any, **kwargs: Any) -> BiblioStats:
    """
    Initialize a temporary BiblioStats instance.

    This helper bypasses normal construction and directly calls
    BiblioStats.__init__ on a freshly allocated instance. It is used by
    BiblioGroup to reuse the full initialization logic of BiblioStats
    without subclassing.

    Parameters
    ----------
    *args, **kwargs :
        Arguments forwarded to BiblioStats.__init__.

    Returns
    -------
    BiblioStats
        A fully initialized temporary analysis object.
    """
    temp = BiblioStats.__new__(BiblioStats)
    BiblioStats.__init__(temp, *args, **kwargs)
    return temp


class BiblioGroup(
    BiblioBase,
    GroupCountingMixin,
    GroupStatsMixin,
    GroupAssociationsMixin,
    GroupAnalysisMixin,
):
    """
    Group-based bibliometric analysis.

    This class enables comparative analysis across multiple groups defined
    by a grouping variable (e.g., year, country, topic).

    Attributes
    ----------
    df : pd.DataFrame
        The main bibliometric dataset.
    n : int
        Number of documents in the dataset.
    groups : Dict[str, BiblioStats]
        Dictionary mapping group names to BiblioStats objects.
    group_matrix : pd.DataFrame
        Binary matrix indicating group membership.
    """

    # Class-level type hints
    df: pd.DataFrame
    n: int
    db: str
    groups: Dict[str, BiblioStats]
    group_matrix: pd.DataFrame
    res_folder: Optional[str]
    default_separator: str

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _save_table(
        self,
        df: pd.DataFrame,
        name: str,
        subfolder: str = "tables",
    ) -> None:
        """Save DataFrame to Excel if res_folder is set."""
        if self.res_folder is not None and df is not None:
            utilsbib.to_excel_fancy(
                df,
                f_name=os.path.join(self.res_folder, subfolder, f"{name}.xlsx"),
                autofit=getattr(self, "autofit", False),
                conditional_formatting=getattr(self, "cond_formatting", False),
            )

    def _save_plot(
        self,
        filename_base: str,
        subfolder: str = "plots",
    ) -> None:
        """Save current matplotlib figure if res_folder is set."""
        if self.res_folder is not None:
            path = os.path.join(self.res_folder, subfolder, filename_base)
            utilsbib.save_plot(path, dpi=getattr(self, "dpi", 600))

    def _get_column(
        self,
        candidates: Union[str, List[str]],
        required: bool = True,
    ) -> Optional[str]:
        """Find the first available column from candidates."""
        if isinstance(candidates, str):
            candidates = [candidates]
        for col in candidates:
            if col in self.df.columns:
                return col
        if required:
            raise ValueError(f"None of the columns found: {candidates}")
        return None

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        f_name: Optional[str] = None,
        db: str = "",
        df: Optional[pd.DataFrame] = None,
        group_desc: Optional[Any] = None,
        res_folder: Optional[str] = "results-groups",
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
        group_colors: Optional[Union[Dict[str, str], List[str]]] = None,
        merge_data_with_group: bool = True,
        extra_stopwords: Optional[List[str]] = None,
        specific_stopword_categories: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Create a BiblioGroup wrapper around a bibliographic dataset.

        The constructor first builds a temporary BiblioStats instance to handle
        all standard preprocessing and bookkeeping. It then constructs group
        membership indicators (``self.group_matrix``) and, for each group,
        creates a dedicated BiblioStats object in ``self.groups``.

        Parameters
        ----------
        f_name : str or path-like, optional
            Path to an input file; passed to BiblioStats when ``df`` is not
            provided.
        db : str, default ""
            Name of the source database (for example "scopus" or "wos").
        df : pandas.DataFrame, optional
            Pre-loaded dataframe. When provided, ``f_name`` is ignored.
        group_desc : any, optional
            Group description passed to :meth:`build_groups` and ultimately to
            ``utilsbib.generate_group_matrix``.
        res_folder : str, default "results-groups"
            Base folder for outputs of this BiblioGroup instance.
        output_lang : str, default "en"
            Language code for user-facing messages.
        preprocess_level : int, default 0
            Preprocessing level forwarded to BiblioStats.
        exclude_list_kw, synonyms_kw, lemmatize_kw, default_keywords :
            Options governing how keyword fields are normalized.
        lang_of_docs : str, default "en"
            Language of the documents; forwarded to BiblioStats.
        fancy_output : bool, default False
            Whether to enable more stylised Excel output.
        label_docs : bool, default True
            If True, BiblioStats adds a ``Doc ID`` label column.
        group_colors : dict or sequence, optional
            Explicit mapping from group name to color, or a sequence of colors
            from which a mapping is constructed.
        merge_data_with_group : bool, default True
            If True, group membership columns from ``group_matrix`` are
            concatenated to ``self.df``.
        extra_stopwords : iterable of str or None, optional
            Additional stopwords applied to all text fields.
        specific_stopword_categories : iterable of str or None, optional
            Category names from the "specific" sheet of the stopword file.
        **kwargs :
            Additional options forwarded to :meth:`build_groups`.
        """
        # Create and initialize a temp BiblioStats-like object
        temp = initialize_biblio_common(
            f_name=f_name,
            db=db,
            df=df,
            res_folder=res_folder,
            output_lang=output_lang,
            preprocess_level=preprocess_level,
            exclude_list_kw=exclude_list_kw,
            synonyms_kw=synonyms_kw,
            lemmatize_kw=lemmatize_kw,
            default_keywords=default_keywords,
            combine_with_index_keywords=combine_with_index_keywords,
            concept_df=concept_df,
            concept_column=concept_column,
            asjc_map_df=asjc_map_df,
            lang_of_docs=lang_of_docs,
            fancy_output=fancy_output,
            label_docs=label_docs,
            dpi=dpi,
            cmap=cmap,
            cmap_disc=cmap_disc,
            default_color=default_color,
            extra_stopwords=extra_stopwords,
            specific_stopword_categories=specific_stopword_categories,
        )

        # Copy all attributes
        self.__dict__.update(temp.__dict__)

        # Group-specific additions
        self.group_desc = group_desc

        self.build_groups(**kwargs)
        self.groups, self.group_df = {}, {}
        for group_name in self.group_matrix.columns:
            mask = self.group_matrix[group_name].astype(bool)
            self.group_df[group_name] = self.df[mask]
            self.groups[group_name] = BiblioStats(
                df=self.group_df[group_name],
                db=self.db,
                preprocess_level=0,
                label_docs=False,
                res_folder=None,
            )

        if group_colors is None:
            default_colors = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
                "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
                "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
                "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363",
            ]
            self.group_colors = {
                group: default_colors[i % len(default_colors)]
                for i, group in enumerate(self.groups.keys())
            }
        else:
            self.group_colors = group_colors

        if merge_data_with_group:
            self.df = pd.concat(
                [self.df, pd.DataFrame(self.group_matrix)],
                axis=1,
            )

    # =========================================================================
    # CORE METHODS
    # =========================================================================

    def filter_dataframe(self, *args: Any, **kwargs: Any) -> None:
        """
        Reuse the filtering logic from BiblioStats.

        This is a thin wrapper that forwards all arguments to
        :meth:`BiblioStats.filter_dataframe`, operating on this
        BiblioGroup instance.

        Notes
        -----
        - This modifies ``self.df`` in-place.
        - Group-related attributes such as ``self.group_matrix``,
          ``self.groups`` and ``self.group_df`` are *not* updated here.
        """
        BiblioStats.filter_dataframe(self, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """
        Delegate missing ``count_*`` methods to the underlying BiblioStats API.

        If an attribute name starting with ``"count_"`` is not found on
        BiblioGroup, this method looks it up on :class:`BiblioStats` and binds
        the function to the current instance.

        Parameters
        ----------
        name : str
            Attribute name being requested.

        Returns
        -------
        callable
            The bound counting method from BiblioStats, if available.

        Raises
        ------
        AttributeError
            If no matching method exists on BiblioStats.
        """
        attr = getattr(BiblioStats, name, None)
        if callable(attr) and name.startswith("count_"):
            return attr.__get__(self, self.__class__)
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    def __dir__(self) -> List[str]:
        """
        Extend attribute completion with inherited ``count_*`` methods.

        The returned list includes the usual attributes plus any public
        methods on :class:`BiblioStats` whose names start with ``"count_"``.
        """
        inherited = [n for n in dir(BiblioStats) if n.startswith("count_")]
        return sorted(set(super().__dir__()) | set(inherited))

    def build_groups(self, **kwargs: Any) -> None:
        """
        Generate group_matrix using the stored group_desc and additional arguments.

        For example: cutpoints, n_periods, year_range, text_column, etc.

        Raises
        ------
        ValueError
            If no dataframe or group descriptor is available.
        """
        if self.df is None:
            raise ValueError("No dataframe (self.df) to build group matrix from.")
        if self.group_desc is None:
            raise ValueError("No group descriptor (self.group_desc) provided.")

        self.group_matrix = utilsbib.generate_group_matrix(
            df=self.df,
            group_desc=self.group_desc,
            sep=self.default_separator,
            **kwargs,
        )

    # =========================================================================
    # ALIASES
    # =========================================================================

    def count_countries(self, **kwargs: Any) -> pd.DataFrame:
        """Alias for count_ca_countries."""
        return self.count_ca_countries(**kwargs)

    def count_keywords(self, **kwargs: Any) -> pd.DataFrame:
        """Alias for count_author_keywords."""
        return self.count_author_keywords(**kwargs)
