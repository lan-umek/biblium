# -*- coding: utf-8 -*-
"""
Group counting methods for BiblioGroup.

This module provides mixin methods for counting entities across groups.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from biblium import utilsbib

if TYPE_CHECKING:
    from biblium.bibgroup import BiblioGroup


class GroupCountingMixin:
    """Mixin providing group_count_* methods for BiblioGroup."""

    # Type hints for attributes from BiblioGroup
    groups: dict
    group_matrix: pd.DataFrame

    def _group_count_entity(
        self: "BiblioGroup",
        count_method_name: str,
        attr_name: str,
        merge_type: Literal["all items", "shared items"] = "all items",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Generic helper for counting entities across groups.

        Parameters
        ----------
        count_method_name : str
            Name of the count method to call on each group's BiblioStats.
        attr_name : str
            Attribute name to store the result on self.
        merge_type : {"all items", "shared items"}, default "all items"
            Strategy for merging items across groups.
        **kwargs :
            Additional options forwarded to the counting method.

        Returns
        -------
        pd.DataFrame
            The aggregated counts across groups.
        """
        result = utilsbib.count_occurrences_across_groups(
            self.groups,
            self.group_matrix,
            count_method_name,
            merge_type=merge_type,
            **kwargs,
        )
        setattr(self, attr_name, result)
        return result

    def group_count_sources(
        self: "BiblioGroup",
        merge_type: Literal["all items", "shared items"] = "all items",
    ) -> pd.DataFrame:
        """
        Count source occurrences across groups.

        Uses ``utilsbib.count_occurrences_across_groups`` and
        ``BiblioStats.count_sources`` to build a table of source counts per
        group. The result is stored as ``group_sources_counts_df``.

        Parameters
        ----------
        merge_type : str, default "all items"
            Strategy for merging items across groups.

        Returns
        -------
        pd.DataFrame
            Source counts per group.
        """
        return self._group_count_entity(
            "count_sources", "group_sources_counts_df", merge_type
        )

    def group_count_author_keywords(
        self: "BiblioGroup",
        merge_type: Literal["all items", "shared items"] = "all items",
    ) -> pd.DataFrame:
        """
        Count author keywords across groups.

        Delegates to ``utilsbib.count_occurrences_across_groups`` with the
        ``count_author_keywords`` method of each nested BiblioStats object.
        The aggregated table is stored as ``group_author_keywords_counts_df``.

        Parameters
        ----------
        merge_type : str, default "all items"
            Strategy for merging items across groups.

        Returns
        -------
        pd.DataFrame
            Author keyword counts per group.
        """
        return self._group_count_entity(
            "count_author_keywords", "group_author_keywords_counts_df", merge_type
        )

    def group_count_index_keywords(
        self: "BiblioGroup",
        merge_type: Literal["all items", "shared items"] = "all items",
    ) -> pd.DataFrame:
        """
        Count index keywords across groups.

        Uses ``utilsbib.count_occurrences_across_groups`` together with
        ``count_index_keywords`` from the nested BiblioStats objects and stores
        the result as ``group_index_keywords_counts_df``.

        Parameters
        ----------
        merge_type : str, default "all items"
            Strategy for merging items across groups.

        Returns
        -------
        pd.DataFrame
            Index keyword counts per group.
        """
        return self._group_count_entity(
            "count_index_keywords", "group_index_keywords_counts_df", merge_type
        )

    def group_count_all_countries(
        self: "BiblioGroup",
        merge_type: Literal["all items", "shared items"] = "all items",
    ) -> pd.DataFrame:
        """
        Count all collaborating countries across groups.

        Aggregates the results of ``count_all_countries`` for each group using
        ``utilsbib.count_occurrences_across_groups`` and stores the table as
        ``group_all_countries_counts_df``.

        Parameters
        ----------
        merge_type : str, default "all items"
            Strategy for merging items across groups.

        Returns
        -------
        pd.DataFrame
            Country counts per group.
        """
        return self._group_count_entity(
            "count_all_countries", "group_all_countries_counts_df", merge_type
        )

    def group_count_keywords(
        self: "BiblioGroup",
        merge_type: Literal["all items", "shared items"] = "all items",
    ) -> None:
        """
        Convenience wrapper to count both author and index keywords across groups.

        Calls :meth:`group_count_author_keywords` and
        :meth:`group_count_index_keywords` with the same ``merge_type``.

        Parameters
        ----------
        merge_type : str, default "all items"
            Strategy for merging items across groups.
        """
        self.group_count_author_keywords(merge_type=merge_type)
        self.group_count_index_keywords(merge_type=merge_type)

    def group_count_ca_countries(
        self: "BiblioGroup",
        merge_type: Literal["all items", "shared items"] = "all items",
    ) -> pd.DataFrame:
        """
        Count corresponding-author countries across groups.

        Uses ``utilsbib.count_occurrences_across_groups`` with
        ``count_ca_countries`` from each nested BiblioStats object and stores
        the result as ``group_ca_countries_counts_df``.

        Parameters
        ----------
        merge_type : str, default "all items"
            Strategy for merging items across groups.

        Returns
        -------
        pd.DataFrame
            CA country counts per group.
        """
        return self._group_count_entity(
            "count_ca_countries", "group_ca_countries_counts_df", merge_type
        )

    def group_count_authors(
        self: "BiblioGroup",
        merge_type: Literal["all items", "shared items"] = "all items",
    ) -> pd.DataFrame:
        """
        Count authors across groups.

        Aggregates author counts obtained from ``count_authors`` on each
        nested BiblioStats object using
        ``utilsbib.count_occurrences_across_groups``. The result is stored as
        ``group_authors_counts_df``.

        Parameters
        ----------
        merge_type : str, default "all items"
            Strategy for merging items across groups.

        Returns
        -------
        pd.DataFrame
            Author counts per group.
        """
        return self._group_count_entity(
            "count_authors", "group_authors_counts_df", merge_type
        )

    def group_count_affiliations(
        self: "BiblioGroup",
        merge_type: Literal["all items", "shared items"] = "all items",
    ) -> pd.DataFrame:
        """
        Count affiliations across groups.

        Uses ``utilsbib.count_occurrences_across_groups`` together with
        ``count_affiliations`` from each group-level BiblioStats object and
        stores the result as ``group_affiliations_counts_df``.

        Parameters
        ----------
        merge_type : str, default "all items"
            Strategy for merging items across groups.

        Returns
        -------
        pd.DataFrame
            Affiliation counts per group.
        """
        return self._group_count_entity(
            "count_affiliations", "group_affiliations_counts_df", merge_type
        )

    def group_count_references(
        self: "BiblioGroup",
        merge_type: Literal["all items", "shared items"] = "all items",
    ) -> pd.DataFrame:
        """
        Count cited references across groups.

        Aggregates the results of ``count_references`` via
        ``utilsbib.count_occurrences_across_groups`` and stores the table as
        ``group_references_counts_df``.

        Parameters
        ----------
        merge_type : str, default "all items"
            Strategy for merging items across groups.

        Returns
        -------
        pd.DataFrame
            Reference counts per group.
        """
        return self._group_count_entity(
            "count_references", "group_references_counts_df", merge_type
        )

    def group_count_ngrams_abstract(
        self: "BiblioGroup",
        merge_type: Literal["all items", "shared items"] = "all items",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Count n-grams from abstracts across groups.

        Calls ``count_ngrams_abstract`` on the nested BiblioStats objects
        through ``utilsbib.count_occurrences_across_groups`` and stores the
        aggregated table as ``group_words_abs_counts_df``.

        Parameters
        ----------
        merge_type : str, default "all items"
            Strategy for merging items across groups.
        **kwargs :
            Additional options forwarded to the underlying counting method.

        Returns
        -------
        pd.DataFrame
            N-gram counts from abstracts per group.
        """
        return self._group_count_entity(
            "count_ngrams_abstract",
            "group_words_abs_counts_df",
            merge_type,
            **kwargs,
        )

    def group_count_ngrams_title(
        self: "BiblioGroup",
        merge_type: Literal["all items", "shared items"] = "all items",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Count n-grams from titles across groups.

        Uses ``count_ngrams_title`` via
        ``utilsbib.count_occurrences_across_groups`` and stores the aggregated
        table as ``group_words_tit_counts_df``.

        Parameters
        ----------
        merge_type : str, default "all items"
            Strategy for merging items across groups.
        **kwargs :
            Additional options forwarded to the underlying counting method.

        Returns
        -------
        pd.DataFrame
            N-gram counts from titles per group.
        """
        return self._group_count_entity(
            "count_ngrams_title",
            "group_words_tit_counts_df",
            merge_type,
            **kwargs,
        )

    def group_count_ngrams(
        self: "BiblioGroup",
        merge_type: Literal["all items", "shared items"] = "all items",
        **kwargs: Any,
    ) -> None:
        """
        Convenience wrapper to count n-grams in both abstracts and titles.

        Calls :meth:`group_count_ngrams_abstract` and
        :meth:`group_count_ngrams_title` with the same arguments.

        Parameters
        ----------
        merge_type : str, default "all items"
            Strategy for merging items across groups.
        **kwargs :
            Additional options forwarded to the underlying methods.
        """
        self.group_count_ngrams_abstract(merge_type=merge_type, **kwargs)
        self.group_count_ngrams_title(merge_type=merge_type, **kwargs)

    # Aliases
    def group_count_countries(self: "BiblioGroup", **kwargs: Any) -> pd.DataFrame:
        """Alias for group_count_ca_countries."""
        return self.group_count_ca_countries(**kwargs)
