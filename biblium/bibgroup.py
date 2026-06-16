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
from biblium.bibgroup_modules.year_trend import GroupYearTrendMixin
from biblium.bibgroup_modules.content_analysis import GroupContentAnalysisMixin
from biblium.bibgroup_modules.field_dynamics import GroupFieldDynamicsMixin
from biblium.bibgroup_modules.pairs import GroupPairsMixin


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
    GroupYearTrendMixin,
    GroupContentAnalysisMixin,
    GroupFieldDynamicsMixin,
    GroupPairsMixin,
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
            from biblium import plotbib
            path = os.path.join(self.res_folder, subfolder, filename_base)
            plotbib.save_plot(path, dpi=getattr(self, "dpi", 600))

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
    # SUBGROUP PREVALENCE COMPARISON
    # =========================================================================

    def _resolve_subgroup_mask(self, subgroup: Any) -> "np.ndarray":
        """Resolve a subgroup selector into a boolean mask aligned
        positionally with the rows of ``self.group_matrix``.

        Accepts a callable (applied to ``self.df``), a boolean mask/Series,
        or a sequence of ``self.df`` index labels / positional indices.
        """
        import numpy as np
        if getattr(self, "group_matrix", None) is None:
            raise ValueError(
                "group_matrix is not available; call build_groups() first.")
        n = len(self.group_matrix)
        if callable(subgroup):
            subgroup = subgroup(self.df)
        arr = np.asarray(subgroup)
        if arr.dtype == bool:
            if len(arr) != n:
                raise ValueError(
                    f"Boolean subgroup mask has length {len(arr)}, "
                    f"expected {n}.")
            return arr
        mask = np.zeros(n, dtype=bool)
        labels = list(arr)
        pos = self.df.index.get_indexer(labels)
        if (pos >= 0).all():
            mask[pos] = True
        else:
            mask[np.asarray(labels, dtype=int)] = True
        return mask

    def compare_subgroup_prevalence(
        self,
        subgroup: Any,
        subgroup_label: str = "Subgroup",
        sort: bool = True,
        save: bool = True,
        filename: str = "subgroup_prevalence",
    ) -> pd.DataFrame:
        """Compare how prevalent each group is within a subset of documents
        versus the whole corpus.

        For every group in ``self.group_matrix`` this reports the number
        and share of documents belonging to the group, computed once for
        the supplied subset of documents and once for the entire corpus,
        together with the lift (subset share divided by corpus share). A
        lift above 1 means the group is over-represented in the subset.

        Parameters
        ----------
        subgroup : callable, boolean mask, or sequence of index labels
            Selects the documents forming the subset. A callable is
            applied to ``self.df`` and must return a boolean mask.
        subgroup_label : str, default "Subgroup"
            Label used for the subset in the returned table.
        sort : bool, default True
            Sort groups by descending subset share.
        save : bool, default True
            Save the resulting table to ``res_folder`` when available.
        filename : str, default "subgroup_prevalence"
            Base file name used when saving.

        Returns
        -------
        pd.DataFrame
            Columns: ``group``, ``subgroup_documents``, ``subgroup_pct``,
            ``corpus_documents``, ``corpus_pct``, ``lift``.
        """
        import numpy as np
        gm = self.group_matrix
        mask = self._resolve_subgroup_mask(subgroup)
        n = len(gm)
        sub_n = int(mask.sum())
        if sub_n == 0:
            raise ValueError("The subgroup selects zero documents.")
        members = gm.to_numpy().astype(bool)
        corpus_docs = members.sum(axis=0)
        sub_docs = members[mask].sum(axis=0)
        corpus_pct = 100.0 * corpus_docs / n
        sub_pct = 100.0 * sub_docs / sub_n
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.where(corpus_pct > 0, sub_pct / corpus_pct, np.nan)
        out = pd.DataFrame({
            "group": list(gm.columns),
            "subgroup_documents": sub_docs.astype(int),
            "subgroup_pct": sub_pct.round(2),
            "corpus_documents": corpus_docs.astype(int),
            "corpus_pct": corpus_pct.round(2),
            "lift": np.round(lift, 2),
        })
        out.attrs["subgroup_label"] = subgroup_label
        out.attrs["subgroup_n"] = sub_n
        out.attrs["corpus_n"] = n
        if sort:
            out = out.sort_values(
                "subgroup_pct", ascending=False).reset_index(drop=True)
        if save:
            self._save_table(out, filename)
        return out

    def plot_subgroup_prevalence(
        self,
        subgroup: Any,
        subgroup_label: str = "Subgroup",
        top_n: Optional[int] = 12,
        title: Optional[str] = None,
        sort: bool = True,
        subgroup_color: str = "#B5121B",
        corpus_color: str = "#9BB7D4",
        filename: str = "subgroup_prevalence",
        figsize: tuple = (11, 6.2),
        show: bool = False,
    ) -> pd.DataFrame:
        """Plot a grouped horizontal bar chart comparing group prevalence
        in a subset of documents against the whole corpus.

        This visualises :meth:`compare_subgroup_prevalence`: for each
        group, one bar shows the share of subset documents belonging to it
        and a second bar shows the share across the whole corpus. Groups
        are ordered by descending subset share and the chart is drawn
        without grid lines.

        Parameters
        ----------
        subgroup : callable, boolean mask, or sequence of index labels
            Passed through to :meth:`compare_subgroup_prevalence`.
        subgroup_label : str, default "Subgroup"
            Legend label for the subset bars.
        top_n : int or None, default 12
            Show only the ``top_n`` groups with the highest subset share.
        title : str, optional
            Plot title; a sensible default is generated when omitted.
        sort : bool, default True
            Sort groups by descending subset share.
        subgroup_color, corpus_color : str
            Bar colours for the subset and the whole corpus.
        filename : str, default "subgroup_prevalence"
            Base file name used when saving to ``res_folder``.
        figsize : tuple, default (11, 6.2)
            Figure size in inches.
        show : bool, default False
            Display the figure interactively.

        Returns
        -------
        pd.DataFrame
            The comparison table from :meth:`compare_subgroup_prevalence`.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        comp = self.compare_subgroup_prevalence(
            subgroup, subgroup_label=subgroup_label, sort=sort,
            save=True, filename=filename)
        sub_n = comp.attrs.get("subgroup_n", 0)
        corpus_n = comp.attrs.get("corpus_n", 0)
        data = comp.head(top_n) if top_n else comp
        data = data.iloc[::-1]
        y = np.arange(len(data))
        bar_h = 0.38
        fig, ax = plt.subplots(figsize=figsize, dpi=getattr(self, "dpi", 200))
        ax.barh(y + bar_h / 2, data["subgroup_pct"], height=bar_h,
                color=subgroup_color,
                label=f"{subgroup_label} (n={sub_n:,})")
        ax.barh(y - bar_h / 2, data["corpus_pct"], height=bar_h,
                color=corpus_color, label=f"Whole corpus (n={corpus_n:,})")
        ax.set_yticks(y)
        ax.set_yticklabels(data["group"], fontsize=11)
        ax.set_xlabel("Share of documents (%)", fontsize=12)
        if title is None:
            title = f"Group prevalence: {subgroup_label} vs. whole corpus"
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.grid(False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=11)
        xmax = float(max(data["subgroup_pct"].max(), data["corpus_pct"].max()))
        for yi, (sp_v, co_v) in enumerate(
                zip(data["subgroup_pct"], data["corpus_pct"])):
            ax.text(sp_v + xmax * 0.012, yi + bar_h / 2, f"{sp_v:.0f}%",
                    va="center", fontsize=9, color="#333333")
            ax.text(co_v + xmax * 0.012, yi - bar_h / 2, f"{co_v:.0f}%",
                    va="center", fontsize=9, color="#333333")
        ax.set_xlim(0, xmax * 1.12)
        ax.legend(fontsize=10, loc="lower right", frameon=False)
        fig.tight_layout()
        self._save_plot(filename)
        if show:
            plt.show()
        plt.close(fig)
        return comp

    # =========================================================================
    # ALIASES
    # =========================================================================

    def count_countries(self, **kwargs: Any) -> pd.DataFrame:
        """Alias for count_ca_countries."""
        return self.count_ca_countries(**kwargs)

    def count_keywords(self, **kwargs: Any) -> pd.DataFrame:
        """Alias for count_author_keywords."""
        return self.count_author_keywords(**kwargs)
