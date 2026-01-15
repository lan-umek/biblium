# -*- coding: utf-8 -*-
"""
Group statistics methods for BiblioGroup.

This module provides mixin methods for computing statistics across groups.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Literal, Optional, Union

import pandas as pd

from biblium import utilsbib

if TYPE_CHECKING:
    from biblium.bibgroup import BiblioGroup


class GroupStatsMixin:
    """Mixin providing get_group_*_stats methods for BiblioGroup."""

    # Type hints for attributes from BiblioGroup
    df: pd.DataFrame
    group_matrix: pd.DataFrame
    default_separator: str

    def _get_group_entity_stats(
        self: "BiblioGroup",
        entity_col: Union[str, List[str]],
        entity_label: str,
        count_method_name: str,
        attr_name: str,
        indicators_attr_name: str,
        save_name: str,
        value_type: str = "list",
        items_of_interest: Optional[List[str]] = None,
        exclude_items: Optional[List[str]] = None,
        top_n: int = 100,
        output_format: Literal["wide", "long"] = "wide",
        indicators: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Generic helper for computing group-level entity statistics.

        Parameters
        ----------
        entity_col : str or list of str
            Column name(s) to use for the entity.
        entity_label : str
            Label for the entity type (e.g., "Source", "Keyword").
        count_method_name : str
            Name of the count method attribute on self.
        attr_name : str
            Attribute name to store the result DataFrame.
        indicators_attr_name : str
            Attribute name to store indicator matrices.
        save_name : str
            Base filename for Excel export.
        value_type : str, default "list"
            Type of values in the column.
        items_of_interest : list of str, optional
            Items to include.
        exclude_items : list of str, optional
            Items to exclude.
        top_n : int, default 100
            Number of top items to keep.
        output_format : {"wide", "long"}, default "wide"
            Shape of the returned statistics.
        indicators : bool, default False
            Whether to compute indicator matrices.
        **kwargs :
            Additional options forwarded to utilsbib.group_entity_stats.

        Returns
        -------
        pd.DataFrame
            Group-level statistics.
        """
        # Resolve column name if multiple candidates provided
        if isinstance(entity_col, list):
            col = None
            for c in entity_col:
                if c in self.df.columns:
                    col = c
                    break
            if col is None:
                raise ValueError(f"None of the columns found: {entity_col}")
            entity_col = col

        # Get the count method
        count_method = getattr(self, count_method_name, None)
        if count_method is None:
            raise AttributeError(f"No count method '{count_method_name}' found")

        stats, inds = utilsbib.group_entity_stats(
            df=self.df,
            group_matrix=self.group_matrix,
            entity_col=entity_col,
            entity_label=entity_label,
            items_of_interest=items_of_interest,
            exclude_items=exclude_items,
            top_n=top_n,
            count_method=count_method,
            output_format=output_format,
            indicators=indicators,
            value_type=value_type,
            sep=self.default_separator,
            **kwargs,
        )

        setattr(self, attr_name, stats)
        if indicators:
            setattr(self, indicators_attr_name, inds)

        # Save to Excel
        if hasattr(self, "_save_table"):
            if isinstance(stats, pd.DataFrame) and not stats.empty:
                self._save_table(stats.reset_index(), save_name)

        return stats

    def get_group_sources_stats(
        self: "BiblioGroup",
        items_of_interest: Optional[List[str]] = None,
        exclude_items: Optional[List[str]] = None,
        top_n: int = 100,
        output_format: Literal["wide", "long"] = "wide",
        indicators: bool = False,
        filename: str = "group_sources_stats",
        **extra_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute group-level statistics for source titles.

        Uses ``utilsbib.group_entity_stats`` with ``count_sources`` to derive
        performance indicators for each source within each group.

        Parameters
        ----------
        items_of_interest, exclude_items : sequence or None, optional
            Optional filters to include or exclude selected sources.
        top_n : int, default 100
            Number of top sources to keep per group.
        output_format : {"wide", "long"}, default "wide"
            Shape of the returned statistics.
        indicators : bool, default False
            If True, also return binary indicator matrices.
        filename : str, default "group_sources_stats"
            Name of the Excel file to write.
        **extra_kwargs :
            Additional options forwarded to ``utilsbib.group_entity_stats``.

        Returns
        -------
        pd.DataFrame
            Source statistics per group.
        """
        return self._get_group_entity_stats(
            entity_col="Source title",
            entity_label="Source",
            count_method_name="count_sources",
            attr_name="group_sources_stats_df",
            indicators_attr_name="group_sources_indicators",
            save_name=filename,
            value_type="string",
            items_of_interest=items_of_interest,
            exclude_items=exclude_items,
            top_n=top_n,
            output_format=output_format,
            indicators=indicators,
            **extra_kwargs,
        )

    def get_group_author_keywords_stats(
        self: "BiblioGroup",
        items_of_interest: Optional[List[str]] = None,
        exclude_items: Optional[List[str]] = None,
        top_n: int = 100,
        output_format: Literal["wide", "long"] = "wide",
        indicators: bool = False,
        filename: str = "group_author_keywords_stats",
        **extra_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute group-level performance statistics for author keywords.

        Parameters
        ----------
        items_of_interest, exclude_items : sequence of str or None, optional
            Optional lists of keywords to include or exclude.
        top_n : int, default 100
            Number of top keywords to keep.
        output_format : {"wide", "long"}, default "wide"
            Shape of the returned statistics.
        indicators : bool, default False
            If True, also compute indicator matrices.
        filename : str, default "group_author_keywords_stats"
            Name of the Excel file to write.
        **extra_kwargs :
            Additional options forwarded to ``utilsbib.group_entity_stats``.

        Returns
        -------
        pd.DataFrame
            Author keyword statistics per group.
        """
        return self._get_group_entity_stats(
            entity_col=["Processed Author Keywords", "Author Keywords"],
            entity_label="Keyword",
            count_method_name="count_author_keywords",
            attr_name="group_author_keywords_stats_df",
            indicators_attr_name="group_author_keywords_indicators",
            save_name=filename,
            value_type="list",
            items_of_interest=items_of_interest,
            exclude_items=exclude_items,
            top_n=top_n,
            output_format=output_format,
            indicators=indicators,
            **extra_kwargs,
        )

    def get_group_index_keywords_stats(
        self: "BiblioGroup",
        items_of_interest: Optional[List[str]] = None,
        exclude_items: Optional[List[str]] = None,
        top_n: int = 100,
        output_format: Literal["wide", "long"] = "wide",
        indicators: bool = False,
        filename: str = "group_index_keywords_stats",
        **extra_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute group-level statistics for index keywords.

        Parameters
        ----------
        items_of_interest, exclude_items : sequence or None, optional
            Optional filters to include or exclude selected keywords.
        top_n : int, default 100
            Number of top keywords to keep per group.
        output_format : {"wide", "long"}, default "wide"
            Shape of the returned statistics.
        indicators : bool, default False
            If True, also return binary indicator matrices.
        filename : str, default "group_index_keywords_stats"
            Name of the Excel file to write.
        **extra_kwargs :
            Additional options forwarded to ``utilsbib.group_entity_stats``.

        Returns
        -------
        pd.DataFrame
            Index keyword statistics per group.
        """
        return self._get_group_entity_stats(
            entity_col=["Processed Index Keywords", "Index Keywords"],
            entity_label="Keyword",
            count_method_name="count_index_keywords",
            attr_name="group_index_keywords_stats_df",
            indicators_attr_name="group_index_keywords_indicators",
            save_name=filename,
            value_type="list",
            items_of_interest=items_of_interest,
            exclude_items=exclude_items,
            top_n=top_n,
            output_format=output_format,
            indicators=indicators,
            **extra_kwargs,
        )

    def get_group_keywords_stats(
        self: "BiblioGroup",
        items_of_interest: Optional[List[str]] = None,
        exclude_items: Optional[List[str]] = None,
        top_n: int = 100,
        output_format: Literal["wide", "long"] = "wide",
        indicators: bool = False,
        **extra_kwargs: Any,
    ) -> None:
        """
        Convenience wrapper to compute stats for both author and index keywords.

        Calls :meth:`get_group_author_keywords_stats` and
        :meth:`get_group_index_keywords_stats` with the same arguments.

        Parameters
        ----------
        items_of_interest, exclude_items : sequence or None, optional
            Optional filters for keyword selection.
        top_n : int, default 100
            Number of top keywords to keep per group.
        output_format : {"wide", "long"}, default "wide"
            Shape of the returned statistics.
        indicators : bool, default False
            If True, also compute indicator matrices.
        **extra_kwargs :
            Additional options forwarded to the underlying methods.
        """
        self.get_group_author_keywords_stats(
            items_of_interest=items_of_interest,
            exclude_items=exclude_items,
            top_n=top_n,
            output_format=output_format,
            indicators=indicators,
            **extra_kwargs,
        )
        self.get_group_index_keywords_stats(
            items_of_interest=items_of_interest,
            exclude_items=exclude_items,
            top_n=top_n,
            output_format=output_format,
            indicators=indicators,
            **extra_kwargs,
        )

    def get_group_authors_stats(
        self: "BiblioGroup",
        items_of_interest: Optional[List[str]] = None,
        exclude_items: Optional[List[str]] = None,
        top_n: int = 100,
        output_format: Literal["wide", "long"] = "wide",
        indicators: bool = False,
        filename: str = "group_authors_stats",
        **extra_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute group-level statistics for authors.

        Parameters
        ----------
        items_of_interest, exclude_items : sequence or None, optional
            Optional filters for author selection.
        top_n : int, default 100
            Number of top authors to keep per group.
        output_format : {"wide", "long"}, default "wide"
            Shape of the returned statistics.
        indicators : bool, default False
            If True, also compute indicator matrices.
        filename : str, default "group_authors_stats"
            Name of the Excel file to write.
        **extra_kwargs :
            Additional options forwarded to ``utilsbib.group_entity_stats``.

        Returns
        -------
        pd.DataFrame
            Author statistics per group.
        """
        return self._get_group_entity_stats(
            entity_col="Authors",
            entity_label="Author",
            count_method_name="count_authors",
            attr_name="group_authors_stats_df",
            indicators_attr_name="group_authors_indicators",
            save_name=filename,
            value_type="list",
            items_of_interest=items_of_interest,
            exclude_items=exclude_items,
            top_n=top_n,
            output_format=output_format,
            indicators=indicators,
            **extra_kwargs,
        )

    def get_group_affiliations_stats(
        self: "BiblioGroup",
        items_of_interest: Optional[List[str]] = None,
        exclude_items: Optional[List[str]] = None,
        top_n: int = 100,
        output_format: Literal["wide", "long"] = "wide",
        indicators: bool = False,
        filename: str = "group_affiliations_stats",
        **extra_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute group-level statistics for affiliations.

        Parameters
        ----------
        items_of_interest, exclude_items : sequence or None, optional
            Optional filters for selecting affiliations.
        top_n : int, default 100
            Number of top affiliations to keep per group.
        output_format : {"wide", "long"}, default "wide"
            Shape of the returned statistics.
        indicators : bool, default False
            If True, also compute indicator matrices.
        filename : str, default "group_affiliations_stats"
            Name of the Excel file to write.
        **extra_kwargs :
            Additional options forwarded to ``utilsbib.group_entity_stats``.

        Returns
        -------
        pd.DataFrame
            Affiliation statistics per group.
        """
        return self._get_group_entity_stats(
            entity_col="Affiliations",
            entity_label="Affiliation",
            count_method_name="count_affiliations",
            attr_name="group_affiliations_stats_df",
            indicators_attr_name="group_affiliations_indicators",
            save_name=filename,
            value_type="list",
            items_of_interest=items_of_interest,
            exclude_items=exclude_items,
            top_n=top_n,
            output_format=output_format,
            indicators=indicators,
            **extra_kwargs,
        )

    def get_group_references_stats(
        self: "BiblioGroup",
        items_of_interest: Optional[List[str]] = None,
        exclude_items: Optional[List[str]] = None,
        top_n: int = 100,
        output_format: Literal["wide", "long"] = "wide",
        indicators: bool = False,
        filename: str = "group_references_stats",
        **extra_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute group-level statistics for cited references.

        Parameters
        ----------
        items_of_interest, exclude_items : sequence or None, optional
            Optional filters for reference selection.
        top_n : int, default 100
            Number of top references to keep per group.
        output_format : {"wide", "long"}, default "wide"
            Shape of the returned statistics.
        indicators : bool, default False
            If True, also compute indicator matrices.
        filename : str, default "group_references_stats"
            Name of the Excel file to write.
        **extra_kwargs :
            Additional options forwarded to ``utilsbib.group_entity_stats``.

        Returns
        -------
        pd.DataFrame
            Reference statistics per group.
        """
        return self._get_group_entity_stats(
            entity_col=["References", "Cited References"],
            entity_label="Reference",
            count_method_name="count_references",
            attr_name="group_references_stats_df",
            indicators_attr_name="group_references_indicators",
            save_name=filename,
            value_type="list",
            items_of_interest=items_of_interest,
            exclude_items=exclude_items,
            top_n=top_n,
            output_format=output_format,
            indicators=indicators,
            **extra_kwargs,
        )

    def get_group_all_countries_stats(
        self: "BiblioGroup",
        items_of_interest: Optional[List[str]] = None,
        exclude_items: Optional[List[str]] = None,
        top_n: int = 100,
        output_format: Literal["wide", "long"] = "wide",
        indicators: bool = False,
        filename: str = "group_all_countries_stats",
        **extra_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute group-level statistics for all collaborating countries.

        Parameters
        ----------
        items_of_interest, exclude_items : sequence or None, optional
            Optional filters for country selection.
        top_n : int, default 100
            Number of top countries to keep per group.
        output_format : {"wide", "long"}, default "wide"
            Shape of the returned statistics.
        indicators : bool, default False
            If True, also compute indicator matrices.
        filename : str, default "group_all_countries_stats"
            Name of the Excel file to write.
        **extra_kwargs :
            Additional options forwarded to ``utilsbib.group_entity_stats``.

        Returns
        -------
        pd.DataFrame
            Country statistics per group.
        """
        return self._get_group_entity_stats(
            entity_col=["Countries of Authors", "All Countries", "Country"],
            entity_label="Country",
            count_method_name="count_all_countries",
            attr_name="group_all_countries_stats_df",
            indicators_attr_name="group_all_countries_indicators",
            save_name=filename,
            value_type="list",
            items_of_interest=items_of_interest,
            exclude_items=exclude_items,
            top_n=top_n,
            output_format=output_format,
            indicators=indicators,
            **extra_kwargs,
        )

    def get_group_ca_countries_stats(
        self: "BiblioGroup",
        items_of_interest: Optional[List[str]] = None,
        exclude_items: Optional[List[str]] = None,
        top_n: int = 100,
        output_format: Literal["wide", "long"] = "wide",
        indicators: bool = False,
        filename: str = "group_ca_countries_stats",
        **extra_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute group-level statistics for corresponding-author countries.

        Parameters
        ----------
        items_of_interest, exclude_items : sequence or None, optional
            Optional filters for country selection.
        top_n : int, default 100
            Number of top countries to keep per group.
        output_format : {"wide", "long"}, default "wide"
            Shape of the returned statistics.
        indicators : bool, default False
            If True, also compute indicator matrices.
        filename : str, default "group_ca_countries_stats"
            Name of the Excel file to write.
        **extra_kwargs :
            Additional options forwarded to ``utilsbib.group_entity_stats``.

        Returns
        -------
        pd.DataFrame
            CA country statistics per group.
        """
        return self._get_group_entity_stats(
            entity_col="CA Country",
            entity_label="Country",
            count_method_name="count_ca_countries",
            attr_name="group_ca_countries_stats_df",
            indicators_attr_name="group_ca_countries_indicators",
            save_name=filename,
            value_type="string",
            items_of_interest=items_of_interest,
            exclude_items=exclude_items,
            top_n=top_n,
            output_format=output_format,
            indicators=indicators,
            **extra_kwargs,
        )

    def get_group_ngrams_abstract_stats(
        self: "BiblioGroup",
        items_of_interest: Optional[List[str]] = None,
        exclude_items: Optional[List[str]] = None,
        top_n: int = 100,
        output_format: Literal["wide", "long"] = "wide",
        indicators: bool = False,
        filename: str = "group_ngrams_abstract_stats",
        **extra_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute group-level statistics for n-grams in abstracts.

        Parameters
        ----------
        items_of_interest, exclude_items : sequence or None, optional
            Optional filters for n-gram selection.
        top_n : int, default 100
            Number of top n-grams to keep per group.
        output_format : {"wide", "long"}, default "wide"
            Shape of the returned statistics.
        indicators : bool, default False
            If True, also compute indicator matrices.
        filename : str, default "group_ngrams_abstract_stats"
            Name of the Excel file to write.
        **extra_kwargs :
            Additional options forwarded to ``utilsbib.group_entity_stats``.

        Returns
        -------
        pd.DataFrame
            N-gram statistics from abstracts per group.
        """
        return self._get_group_entity_stats(
            entity_col=["Processed Abstract", "Abstract"],
            entity_label="Term",
            count_method_name="count_ngrams_abstract",
            attr_name="group_words_abs_stats_df",
            indicators_attr_name="group_words_abs_indicators",
            save_name=filename,
            value_type="text",
            items_of_interest=items_of_interest,
            exclude_items=exclude_items,
            top_n=top_n,
            output_format=output_format,
            indicators=indicators,
            **extra_kwargs,
        )

    def get_group_ngrams_title_stats(
        self: "BiblioGroup",
        items_of_interest: Optional[List[str]] = None,
        exclude_items: Optional[List[str]] = None,
        top_n: int = 100,
        output_format: Literal["wide", "long"] = "wide",
        indicators: bool = False,
        filename: str = "group_ngrams_title_stats",
        **extra_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute group-level statistics for n-grams in titles.

        Parameters
        ----------
        items_of_interest, exclude_items : sequence or None, optional
            Optional filters for n-gram selection.
        top_n : int, default 100
            Number of top n-grams to keep per group.
        output_format : {"wide", "long"}, default "wide"
            Shape of the returned statistics.
        indicators : bool, default False
            If True, also compute indicator matrices.
        filename : str, default "group_ngrams_title_stats"
            Name of the Excel file to write.
        **extra_kwargs :
            Additional options forwarded to ``utilsbib.group_entity_stats``.

        Returns
        -------
        pd.DataFrame
            N-gram statistics from titles per group.
        """
        return self._get_group_entity_stats(
            entity_col=["Processed Title", "Title"],
            entity_label="Term",
            count_method_name="count_ngrams_title",
            attr_name="group_words_tit_stats_df",
            indicators_attr_name="group_words_tit_indicators",
            save_name=filename,
            value_type="text",
            items_of_interest=items_of_interest,
            exclude_items=exclude_items,
            top_n=top_n,
            output_format=output_format,
            indicators=indicators,
            **extra_kwargs,
        )

    def get_group_ngrams_stats(
        self: "BiblioGroup",
        items_of_interest: Optional[List[str]] = None,
        exclude_items: Optional[List[str]] = None,
        top_n: int = 100,
        output_format: Literal["wide", "long"] = "wide",
        indicators: bool = False,
        **extra_kwargs: Any,
    ) -> None:
        """
        Convenience wrapper to compute n-gram stats for both abstracts and titles.

        Calls :meth:`get_group_ngrams_abstract_stats` and
        :meth:`get_group_ngrams_title_stats` with the same arguments.
        """
        self.get_group_ngrams_abstract_stats(
            items_of_interest=items_of_interest,
            exclude_items=exclude_items,
            top_n=top_n,
            output_format=output_format,
            indicators=indicators,
            **extra_kwargs,
        )
        self.get_group_ngrams_title_stats(
            items_of_interest=items_of_interest,
            exclude_items=exclude_items,
            top_n=top_n,
            output_format=output_format,
            indicators=indicators,
            **extra_kwargs,
        )

    def get_group_fields_stats(
        self: "BiblioGroup",
        items_of_interest: Optional[List[str]] = None,
        exclude_items: Optional[List[str]] = None,
        top_n: int = 100,
        output_format: Literal["wide", "long"] = "wide",
        indicators: bool = False,
        filename: str = "group_fields_stats",
        **extra_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute group-level statistics for subject fields.

        Parameters
        ----------
        items_of_interest, exclude_items : sequence or None, optional
            Optional filters for field selection.
        top_n : int, default 100
            Number of top fields to keep per group.
        output_format : {"wide", "long"}, default "wide"
            Shape of the returned statistics.
        indicators : bool, default False
            If True, also compute indicator matrices.
        filename : str, default "group_fields_stats"
            Name of the Excel file to write.
        **extra_kwargs :
            Additional options forwarded to ``utilsbib.group_entity_stats``.

        Returns
        -------
        pd.DataFrame
            Field statistics per group.
        """
        return self._get_group_entity_stats(
            entity_col="Field",
            entity_label="Field",
            count_method_name="count_fields",
            attr_name="group_fields_stats_df",
            indicators_attr_name="group_fields_indicators",
            save_name=filename,
            value_type="list",
            items_of_interest=items_of_interest,
            exclude_items=exclude_items,
            top_n=top_n,
            output_format=output_format,
            indicators=indicators,
            **extra_kwargs,
        )

    def get_group_areas_stats(
        self: "BiblioGroup",
        items_of_interest: Optional[List[str]] = None,
        exclude_items: Optional[List[str]] = None,
        top_n: int = 100,
        output_format: Literal["wide", "long"] = "wide",
        indicators: bool = False,
        filename: str = "group_areas_stats",
        **extra_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute group-level statistics for subject areas.

        Parameters
        ----------
        items_of_interest, exclude_items : sequence or None, optional
            Optional filters for area selection.
        top_n : int, default 100
            Number of top areas to keep per group.
        output_format : {"wide", "long"}, default "wide"
            Shape of the returned statistics.
        indicators : bool, default False
            If True, also compute indicator matrices.
        filename : str, default "group_areas_stats"
            Name of the Excel file to write.
        **extra_kwargs :
            Additional options forwarded to ``utilsbib.group_entity_stats``.

        Returns
        -------
        pd.DataFrame
            Area statistics per group.
        """
        return self._get_group_entity_stats(
            entity_col="Area",
            entity_label="Area",
            count_method_name="count_areas",
            attr_name="group_areas_stats_df",
            indicators_attr_name="group_areas_indicators",
            save_name=filename,
            value_type="list",
            items_of_interest=items_of_interest,
            exclude_items=exclude_items,
            top_n=top_n,
            output_format=output_format,
            indicators=indicators,
            **extra_kwargs,
        )

    def get_group_sciences_stats(
        self: "BiblioGroup",
        items_of_interest: Optional[List[str]] = None,
        exclude_items: Optional[List[str]] = None,
        top_n: int = 100,
        output_format: Literal["wide", "long"] = "wide",
        indicators: bool = False,
        filename: str = "group_sciences_stats",
        **extra_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Compute group-level statistics for broad sciences.

        Parameters
        ----------
        items_of_interest, exclude_items : sequence or None, optional
            Optional filters for science selection.
        top_n : int, default 100
            Number of top sciences to keep per group.
        output_format : {"wide", "long"}, default "wide"
            Shape of the returned statistics.
        indicators : bool, default False
            If True, also compute indicator matrices.
        filename : str, default "group_sciences_stats"
            Name of the Excel file to write.
        **extra_kwargs :
            Additional options forwarded to ``utilsbib.group_entity_stats``.

        Returns
        -------
        pd.DataFrame
            Science statistics per group.
        """
        return self._get_group_entity_stats(
            entity_col="Science",
            entity_label="Science",
            count_method_name="count_sciences",
            attr_name="group_sciences_stats_df",
            indicators_attr_name="group_sciences_indicators",
            save_name=filename,
            value_type="list",
            items_of_interest=items_of_interest,
            exclude_items=exclude_items,
            top_n=top_n,
            output_format=output_format,
            indicators=indicators,
            **extra_kwargs,
        )
