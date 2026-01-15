# -*- coding: utf-8 -*-
"""
Group association methods for BiblioGroup.

This module provides mixin methods for associating groups with various entities.
"""

from __future__ import annotations

import os
import re
import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import pandas as pd

from biblium import utilsbib

if TYPE_CHECKING:
    from biblium.bibgroup import BiblioGroup


class GroupAssociationsMixin:
    """Mixin providing associate_* methods for BiblioGroup."""

    # Type hints for attributes from BiblioGroup
    df: pd.DataFrame
    group_matrix: pd.DataFrame
    default_separator: str
    res_folder: Optional[str]

    def associate_items(
        self: "BiblioGroup",
        *,
        domain_key: str,
        item_col: str,
        count_type: str = "single",
        value_type: str = "string",
        item_column_name: str = "Item",
        translated_column_name: Optional[str] = None,
        include_stats: Tuple[str, ...] = ("diversity", "correspondence", "chi2", "svd", "log-ratio"),
        clean_zeros: bool = True,
        to_self: bool = False,
        top_n: int = 0,
        min_freq: int = 5,
        items_included: Optional[Union[List[str], str]] = None,
        items_excluded: Optional[Union[List[str], str]] = None,
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> "BiblioGroup":
        """
        Build item indicators and relate them to groups.

        Workflow:
        1. Count items via utilsbib.count_occurrences.
        2. Select items by items_included/top_n/min_freq and apply items_excluded.
        3. Build indicators via utilsbib.match_items_and_compute_binary_indicators.
        4. Relate groups to items using self.group_matrix.
        5. If filename provided, save associations to Excel.

        Parameters
        ----------
        domain_key : str
            Key to identify this association domain.
        item_col : str
            Column name containing the items.
        count_type : str, default "single"
            Type of counting ("single", "list", "text").
        value_type : str, default "string"
            Type of values ("string", "list", "text").
        item_column_name : str, default "Item"
            Label for the item column in output.
        translated_column_name : str, optional
            Optional translated column name.
        include_stats : tuple, default ("diversity", "correspondence", "chi2", "svd", "log-ratio")
            Statistics to include in associations.
        clean_zeros : bool, default True
            Remove zero-count items.
        to_self : bool, default False
            Include self-associations.
        top_n : int, default 0
            Number of top items (0 = use min_freq).
        min_freq : int, default 5
            Minimum frequency threshold.
        items_included : list or str, optional
            Items to include (list or regex pattern).
        items_excluded : list or str, optional
            Items to exclude (list or regex pattern).
        filename : str, optional
            Output filename for associations.
        **kwargs :
            Additional options.

        Returns
        -------
        BiblioGroup
            Self for method chaining.
        """
        if not hasattr(self, "group_matrix") or self.group_matrix is None:
            raise AttributeError("self.group_matrix is missing. Build groups first.")

        # Extract sep for count_occurrences
        sep_for_count = kwargs.pop("sep", None)
        if sep_for_count is None:
            sep_for_count = getattr(self, "default_separator", "; ")

        # 1) Count items
        counts_df = utilsbib.count_occurrences(
            self.df,
            item_col,
            count_type=count_type,
            item_column_name=item_column_name,
            translated_column_name=translated_column_name,
            sep=sep_for_count,
            **kwargs,
        )

        # Helper functions
        def _select_from_counts_by_regex(pattern: str) -> List[str]:
            mask = counts_df[item_column_name].astype(str).str.contains(
                pattern, regex=True, na=False
            )
            return counts_df.loc[mask, item_column_name].astype(str).tolist()

        def _dedup_preserve(seq):
            return list(dict.fromkeys(seq))

        # 2) Select items
        if items_included is not None:
            if isinstance(items_included, str):
                selected = _select_from_counts_by_regex(items_included)
            else:
                selected = [str(x) for x in items_included]
        else:
            if top_n and top_n > 0:
                selected = (
                    counts_df.nlargest(top_n, "Number of documents")[item_column_name]
                    .dropna().astype(str).tolist()
                )
            else:
                selected = (
                    counts_df.loc[
                        counts_df["Number of documents"] >= int(min_freq),
                        item_column_name
                    ]
                    .dropna().astype(str).tolist()
                )

        # Apply exclusion
        if items_excluded is not None and selected:
            if isinstance(items_excluded, str):
                pat = re.compile(items_excluded)
                selected = [it for it in selected if not pat.search(str(it))]
            else:
                excl_set = {str(x) for x in items_excluded}
                selected = [it for it in selected if str(it) not in excl_set]

        items_of_interest = _dedup_preserve(map(str, selected))

        # Edge case: nothing selected
        if len(items_of_interest) == 0:
            warnings.warn(
                f"No items selected for '{domain_key}' (min_freq={min_freq}, top_n={top_n}). "
                f"Total items counted: {len(counts_df)}. "
                f"Try lowering min_freq or using top_n parameter."
            )
            setattr(self, f"{domain_key}_associations", None)
            setattr(self, f"{domain_key}_contingency", pd.DataFrame())
            if filename:
                utilsbib._save_associations_xlsx(None, filename)
            return self

        # 3) Build indicators
        _, indicators_dict = utilsbib.match_items_and_compute_binary_indicators(
            df=self.df,
            col=item_col,
            items_of_interest=items_of_interest,
            value_type=value_type,
        )
        binary_mat = indicators_dict["binary"]

        # 4) Relate groups to items
        G = (
            self.group_matrix.copy()
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .astype("float64")
        )
        X = (
            binary_mat.copy()
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .astype("float64")
        )
        common_idx = G.index.intersection(X.index)
        if len(common_idx) == 0:
            raise ValueError(
                "No shared document index between group_matrix and the binary indicators."
            )
        G = G.loc[common_idx]
        X = X.loc[common_idx]

        relate_groups_to = getattr(self, "relate_groups_to", None)
        if callable(relate_groups_to):
            assoc, cont = relate_groups_to(
                domain_key,
                other_matrix=X,
                include_stats=include_stats,
                clean_zeros=clean_zeros,
                to_self=to_self,
            )
        else:
            assoc, cont = utilsbib.relate_concepts_general(
                "group",
                domain_key,
                known_matrices={"group": G},
                custom_matrices={domain_key: X},
                include_stats=include_stats,
                clean_zeros=clean_zeros,
                to_self=to_self,
            )

        setattr(self, f"{domain_key}_associations", assoc)
        setattr(self, f"{domain_key}_contingency", cont)

        # 5) Save if requested
        if filename:
            utilsbib._save_associations_xlsx(assoc, filename)

        return self

    def _resolve_filename(
        self: "BiblioGroup",
        filename: Optional[str],
        subfolder: str = "relations",
    ) -> Optional[str]:
        """Resolve filename with result folder path."""
        if filename is not None and getattr(self, "res_folder", None):
            return os.path.join(self.res_folder, subfolder, filename)
        return filename

    def associate_sources(
        self: "BiblioGroup",
        *,
        include_stats: Tuple[str, ...] = ("diversity", "correspondence", "chi2", "svd", "log-ratio"),
        clean_zeros: bool = True,
        to_self: bool = False,
        top_n: int = 0,
        min_freq: int = 5,
        items_included: Optional[Union[List[str], str]] = None,
        items_excluded: Optional[Union[List[str], str]] = None,
        filename: Optional[str] = "associated sources",
        **kwargs: Any,
    ) -> "BiblioGroup":
        """
        Relate fixed groups to journal sources.

        Parameters
        ----------
        include_stats : tuple
            Statistics to compute.
        clean_zeros : bool, default True
            Remove zero-count items.
        to_self : bool, default False
            Include self-associations.
        top_n : int, default 0
            Number of top items.
        min_freq : int, default 5
            Minimum frequency threshold.
        items_included, items_excluded : list or str, optional
            Filters for item selection.
        filename : str, optional
            Output filename.
        **kwargs :
            Additional options.

        Returns
        -------
        BiblioGroup
            Self for method chaining.
        """
        return self.associate_items(
            domain_key="sources",
            item_col="Source title",
            count_type="single",
            value_type="string",
            item_column_name="Source",
            translated_column_name="Abbreviated Source Title",
            include_stats=include_stats,
            clean_zeros=clean_zeros,
            to_self=to_self,
            top_n=top_n,
            min_freq=min_freq,
            items_included=items_included,
            items_excluded=items_excluded,
            filename=self._resolve_filename(filename),
            **kwargs,
        )

    def associate_author_keywords(
        self: "BiblioGroup",
        *,
        include_stats: Tuple[str, ...] = ("diversity", "correspondence", "chi2", "svd", "log-ratio"),
        clean_zeros: bool = True,
        to_self: bool = False,
        top_n: int = 0,
        min_freq: int = 5,
        items_included: Optional[Union[List[str], str]] = None,
        items_excluded: Optional[Union[List[str], str]] = None,
        sep: Optional[str] = None,
        filename: Optional[str] = "associated keywords",
        **kwargs: Any,
    ) -> "BiblioGroup":
        """
        Relate fixed groups to Author Keywords.

        Prefers processed keywords if available.
        """
        col = (
            "Processed Author Keywords"
            if "Processed Author Keywords" in self.df.columns
            else "Author Keywords"
        )
        if sep is None:
            sep = getattr(self, "default_separator", "; ")

        return self.associate_items(
            domain_key="author_keywords",
            item_col=col,
            count_type="list",
            value_type="list",
            item_column_name="Keyword",
            translated_column_name=None,
            include_stats=include_stats,
            clean_zeros=clean_zeros,
            to_self=to_self,
            top_n=top_n,
            min_freq=min_freq,
            items_included=items_included,
            items_excluded=items_excluded,
            sep=sep,
            filename=self._resolve_filename(filename),
            **kwargs,
        )

    def associate_index_keywords(
        self: "BiblioGroup",
        *,
        include_stats: Tuple[str, ...] = ("diversity", "correspondence", "chi2", "svd", "log-ratio"),
        clean_zeros: bool = True,
        to_self: bool = False,
        top_n: int = 0,
        min_freq: int = 5,
        items_included: Optional[Union[List[str], str]] = None,
        items_excluded: Optional[Union[List[str], str]] = None,
        sep: Optional[str] = None,
        filename: Optional[str] = "associated index keywords",
        **kwargs: Any,
    ) -> "BiblioGroup":
        """
        Relate fixed groups to Index Keywords.

        Prefers processed keywords if available.
        """
        col = (
            "Processed Index Keywords"
            if "Processed Index Keywords" in self.df.columns
            else "Index Keywords"
        )
        if sep is None:
            sep = getattr(self, "default_separator", "; ")

        return self.associate_items(
            domain_key="index_keywords",
            item_col=col,
            count_type="list",
            value_type="list",
            item_column_name="Keyword",
            translated_column_name=None,
            include_stats=include_stats,
            clean_zeros=clean_zeros,
            to_self=to_self,
            top_n=top_n,
            min_freq=min_freq,
            items_included=items_included,
            items_excluded=items_excluded,
            sep=sep,
            filename=self._resolve_filename(filename),
            **kwargs,
        )

    def associate_abstract_words(
        self: "BiblioGroup",
        *,
        include_stats: Tuple[str, ...] = ("diversity", "correspondence", "chi2", "svd", "log-ratio"),
        clean_zeros: bool = True,
        to_self: bool = False,
        top_n: int = 0,
        min_freq: int = 5,
        items_included: Optional[Union[List[str], str]] = None,
        items_excluded: Optional[Union[List[str], str]] = None,
        sep: Optional[str] = None,
        filename: Optional[str] = "associated abstract words",
        **kwargs: Any,
    ) -> "BiblioGroup":
        """
        Relate fixed groups to words from Abstracts.

        Prefers processed abstracts if available.
        """
        col = (
            "Processed Abstract"
            if "Processed Abstract" in self.df.columns
            else "Abstract"
        )

        extra = {}
        if sep is not None:
            extra["sep"] = sep

        return self.associate_items(
            domain_key="abstract_words",
            item_col=col,
            count_type="text",
            value_type="text",
            item_column_name="Word/Phrase",
            translated_column_name=None,
            include_stats=include_stats,
            clean_zeros=clean_zeros,
            to_self=to_self,
            top_n=top_n,
            min_freq=min_freq,
            items_included=items_included,
            items_excluded=items_excluded,
            filename=self._resolve_filename(filename),
            **extra,
            **kwargs,
        )

    def associate_title_words(
        self: "BiblioGroup",
        *,
        include_stats: Tuple[str, ...] = ("diversity", "correspondence", "chi2", "svd", "log-ratio"),
        clean_zeros: bool = True,
        to_self: bool = False,
        top_n: int = 0,
        min_freq: int = 5,
        items_included: Optional[Union[List[str], str]] = None,
        items_excluded: Optional[Union[List[str], str]] = None,
        sep: Optional[str] = None,
        filename: Optional[str] = "associated title words",
        **kwargs: Any,
    ) -> "BiblioGroup":
        """
        Relate fixed groups to words from Titles.

        Prefers processed titles if available.
        """
        col = (
            "Processed Title"
            if "Processed Title" in self.df.columns
            else "Title"
        )

        extra = {}
        if sep is not None:
            extra["sep"] = sep

        return self.associate_items(
            domain_key="title_words",
            item_col=col,
            count_type="text",
            value_type="list",
            item_column_name="Word/Phrase",
            translated_column_name=None,
            include_stats=include_stats,
            clean_zeros=clean_zeros,
            to_self=to_self,
            top_n=top_n,
            min_freq=min_freq,
            items_included=items_included,
            items_excluded=items_excluded,
            filename=self._resolve_filename(filename),
            **extra,
            **kwargs,
        )

    def associate_authors(
        self: "BiblioGroup",
        *,
        include_stats: Tuple[str, ...] = ("diversity", "correspondence", "chi2", "svd", "log-ratio"),
        clean_zeros: bool = True,
        to_self: bool = False,
        top_n: int = 0,
        min_freq: int = 5,
        items_included: Optional[Union[List[str], str]] = None,
        items_excluded: Optional[Union[List[str], str]] = None,
        sep: Optional[str] = None,
        filename: Optional[str] = "associated authors",
        **kwargs: Any,
    ) -> "BiblioGroup":
        """
        Relate fixed groups to Authors.

        Prefers full names if available.
        """
        col = (
            "Author full names"
            if "Author full names" in self.df.columns
            else "Authors"
        )
        if sep is None:
            sep = getattr(self, "default_separator", "; ")

        return self.associate_items(
            domain_key="authors",
            item_col=col,
            count_type="list",
            value_type="list",
            item_column_name="Author",
            translated_column_name=None,
            include_stats=include_stats,
            clean_zeros=clean_zeros,
            to_self=to_self,
            top_n=top_n,
            min_freq=min_freq,
            items_included=items_included,
            items_excluded=items_excluded,
            sep=sep,
            filename=self._resolve_filename(filename),
            **kwargs,
        )

    def associate_countries(
        self: "BiblioGroup",
        *,
        include_stats: Tuple[str, ...] = ("diversity", "correspondence", "chi2", "svd", "log-ratio"),
        clean_zeros: bool = True,
        to_self: bool = False,
        top_n: int = 0,
        min_freq: int = 5,
        items_included: Optional[Union[List[str], str]] = None,
        items_excluded: Optional[Union[List[str], str]] = None,
        sep: Optional[str] = None,
        filename: Optional[str] = "associated countries",
        **kwargs: Any,
    ) -> "BiblioGroup":
        """
        Relate fixed groups to Countries.

        Uses 'CA Country' column if available, otherwise falls back to
        extracting countries from affiliations.
        """
        # Determine country column
        if "CA Country" in self.df.columns:
            col = "CA Country"
            count_type = "single"
            value_type = "string"
        elif "Countries of Authors" in self.df.columns:
            col = "Countries of Authors"
            count_type = "list"
            value_type = "list"
        else:
            # Try to create CA Country column
            self.df = utilsbib.add_ca_country_df(self.df, getattr(self, "db", ""))
            if "CA Country" in self.df.columns:
                col = "CA Country"
                count_type = "single"
                value_type = "string"
            else:
                raise KeyError(
                    "No country column found. Available columns with 'country' or 'affil': "
                    f"{[c for c in self.df.columns if 'country' in c.lower() or 'affil' in c.lower()]}"
                )

        if sep is None:
            sep = getattr(self, "default_separator", "; ")

        return self.associate_items(
            domain_key="countries",
            item_col=col,
            count_type=count_type,
            value_type=value_type,
            item_column_name="Country",
            translated_column_name=None,
            include_stats=include_stats,
            clean_zeros=clean_zeros,
            to_self=to_self,
            top_n=top_n,
            min_freq=min_freq,
            items_included=items_included,
            items_excluded=items_excluded,
            sep=sep,
            filename=self._resolve_filename(filename),
            **kwargs,
        )

    def associate_affiliations(
        self: "BiblioGroup",
        *,
        include_stats: Tuple[str, ...] = ("diversity", "correspondence", "chi2", "svd", "log-ratio"),
        clean_zeros: bool = True,
        to_self: bool = False,
        top_n: int = 0,
        min_freq: int = 5,
        items_included: Optional[Union[List[str], str]] = None,
        items_excluded: Optional[Union[List[str], str]] = None,
        sep: Optional[str] = None,
        filename: Optional[str] = "associated affiliations",
        **kwargs: Any,
    ) -> "BiblioGroup":
        """Relate fixed groups to Affiliations."""
        if sep is None:
            sep = getattr(self, "default_separator", "; ")

        return self.associate_items(
            domain_key="affiliations",
            item_col="Affiliations",
            count_type="list",
            value_type="list",
            item_column_name="Affiliation",
            translated_column_name=None,
            include_stats=include_stats,
            clean_zeros=clean_zeros,
            to_self=to_self,
            top_n=top_n,
            min_freq=min_freq,
            items_included=items_included,
            items_excluded=items_excluded,
            sep=sep,
            filename=self._resolve_filename(filename),
            **kwargs,
        )

    def associate_references(
        self: "BiblioGroup",
        *,
        include_stats: Tuple[str, ...] = ("diversity", "correspondence", "chi2", "svd", "log-ratio"),
        clean_zeros: bool = True,
        to_self: bool = False,
        top_n: int = 0,
        min_freq: int = 5,
        items_included: Optional[Union[List[str], str]] = None,
        items_excluded: Optional[Union[List[str], str]] = None,
        sep: Optional[str] = None,
        filename: Optional[str] = "associated references",
        **kwargs: Any,
    ) -> "BiblioGroup":
        """Relate fixed groups to References."""
        col = (
            "References"
            if "References" in self.df.columns
            else "Cited References"
        )
        if sep is None:
            sep = getattr(self, "default_separator", "; ")

        return self.associate_items(
            domain_key="references",
            item_col=col,
            count_type="list",
            value_type="list",
            item_column_name="Reference",
            translated_column_name=None,
            include_stats=include_stats,
            clean_zeros=clean_zeros,
            to_self=to_self,
            top_n=top_n,
            min_freq=min_freq,
            items_included=items_included,
            items_excluded=items_excluded,
            sep=sep,
            filename=self._resolve_filename(filename),
            **kwargs,
        )
