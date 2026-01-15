# -*- coding: utf-8 -*-
"""
Counting utilities - entity counting, occurrence analysis, binary indicators.

This module contains:
- count_occurrences: Count items in single values, lists, or text
- match_items_and_compute_binary_indicators: Create binary indicator columns
- get_entity_stats: Compute comprehensive entity statistics
"""

from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def count_occurrences(
    df: pd.DataFrame,
    column_name: str,
    count_type: str = "single",
    ngram_range: tuple = (1, 1),
    item_column_name: str = "Item",
    rename_dict: Optional[Dict[str, str]] = None,
    translated_column_name: str = "Translated Item",
    sep: str = "; ",
    token_pattern: str = r"(?u)[^\s]+",
) -> pd.DataFrame:
    """
    Process a DataFrame column and return a DataFrame with counts,
    proportions, and ranks.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    column_name : str
        Name of the column to process.
    count_type : {"single", "list", "text"}
        Type of processing:
        - "single": each cell is a single item.
        - "list": each cell is a delimited list of items (separated by ``sep``).
        - "text": free text; n-grams are extracted with CountVectorizer.
    ngram_range : tuple
        N-gram range for text processing.
    item_column_name : str
        Name of the item column in the output.
    rename_dict : dict, optional
        If provided, adds a column with translated/renamed items.
    translated_column_name : str
        Name of the translated column if ``rename_dict`` is provided.
    sep : str
        Separator used for "list" count_type.
    token_pattern : str
        Regex pattern for tokenization in "text" mode.

    Returns
    -------
    DataFrame
        Processed counts sorted in descending order with columns:
        - "Rank" (1 = most frequent)
        - "Percentrank" (0–1, top item = 1)
    """

    def _balance_closing_parenthesis(s: str) -> str:
        """If there are more '(' than ')', append ')' to balance."""
        if s.count("(") > s.count(")"):
            return s + ")"
        return s

    def _add_rank_columns(df_out: pd.DataFrame) -> pd.DataFrame:
        """Add Rank and Percentrank columns based on Number of documents."""
        if df_out.empty:
            df_out["Rank"] = []
            df_out["Percentrank"] = []
            return df_out

        df_out = df_out.sort_values(
            by="Number of documents",
            ascending=False,
        ).reset_index(drop=True)

        n = len(df_out)
        df_out["Rank"] = np.arange(1, n + 1)
        if n == 1:
            df_out["Percentrank"] = 1.0
        else:
            df_out["Percentrank"] = (n - df_out["Rank"]) / (n - 1)
        return df_out

    total_rows = len(df)

    # Remove NaNs and trim outer whitespace
    data = df[column_name].dropna().astype(str).str.strip()
    data = data[data != ""]

    if count_type == "single":
        counts = Counter(data)

    elif count_type == "list":
        split_series = data.str.split(sep, regex=False)
        split_series = split_series.dropna().tolist()
        
        cleaned_lists: List[List[str]] = []
        for items in split_series:
            cleaned: List[str] = []
            for item in items:
                s = item.strip()
                if not s:
                    continue
                s = _balance_closing_parenthesis(s)
                cleaned.append(s)
            cleaned_lists.append(cleaned)

        flattened = list(chain.from_iterable(cleaned_lists))
        counts = Counter(flattened)

        fractional_counts: Counter = Counter()
        for items in cleaned_lists:
            unique_items = set(items)
            weight = 1 / len(unique_items) if unique_items else 0
            for item in unique_items:
                fractional_counts[item] += weight

    elif count_type == "text":
        if data.empty:
            columns = [
                item_column_name,
                "Number of documents",
                "Proportion of documents",
                "Percentage of documents",
                "Number of occurrences",
                "Rank",
                "Percentrank",
            ]
            if rename_dict is not None:
                columns.insert(1, translated_column_name)
            return pd.DataFrame(columns=columns)

        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            token_pattern=token_pattern,
        )
        term_matrix = vectorizer.fit_transform(data)
        terms = vectorizer.get_feature_names_out()

        doc_counts = (term_matrix > 0).sum(axis=0).A1
        total_counts = term_matrix.sum(axis=0).A1

        result_df = pd.DataFrame({
            item_column_name: terms,
            "Number of documents": doc_counts,
            "Proportion of documents": doc_counts / total_rows,
            "Percentage of documents": (doc_counts / total_rows) * 100,
            "Number of occurrences": total_counts,
        })

        if rename_dict is not None:
            result_df.insert(
                1,
                translated_column_name,
                result_df[item_column_name].map(rename_dict).fillna(""),
            )

        result_df = _add_rank_columns(result_df)
        return result_df

    else:
        raise ValueError('Invalid count_type. Choose from "single", "list", or "text".')

    # "single" and "list" branches share this aggregation
    result_data: Dict[str, list] = {
        item_column_name: list(counts.keys()),
        "Number of documents": list(counts.values()),
        "Proportion of documents": [v / total_rows for v in counts.values()],
        "Percentage of documents": [(v / total_rows) * 100 for v in counts.values()],
    }

    if count_type == "list":
        result_data["Fractional number of documents"] = [
            fractional_counts[item] for item in counts.keys()
        ]

    result_df = pd.DataFrame(result_data)

    if rename_dict is not None:
        result_df.insert(
            1,
            translated_column_name,
            result_df[item_column_name].map(rename_dict).fillna(""),
        )

    result_df = _add_rank_columns(result_df)
    return result_df


def match_items_and_compute_binary_indicators(
    df: pd.DataFrame,
    col: str,
    items_of_interest: List[str],
    value_type: str = "string",
    separator: Optional[str] = None,
    case_sensitive: bool = False,
    prefix: str = "",
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Create binary indicator columns for items of interest.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    col : str
        Column to search for items.
    items_of_interest : list
        Items to create indicators for.
    value_type : {"string", "list"}
        Whether the column contains single values or lists.
    separator : str, optional
        Separator for list values.
    case_sensitive : bool
        Whether matching is case-sensitive.
    prefix : str
        Prefix for indicator column names.

    Returns
    -------
    tuple
        (DataFrame with indicators added, dict with "binary" DataFrame)
    """
    df = df.copy()
    binary_cols = {}

    for item in items_of_interest:
        col_name = f"{prefix}{item}" if prefix else item
        
        if value_type == "string":
            if case_sensitive:
                df[col_name] = (df[col] == item).astype(int)
            else:
                df[col_name] = df[col].str.lower().eq(item.lower()).astype(int)
        
        elif value_type == "list":
            if separator is None:
                separator = "; "
            
            def check_item(val, item=item, sep=separator, cs=case_sensitive):
                if pd.isna(val):
                    return 0
                items = str(val).split(sep)
                items = [i.strip() for i in items]
                if cs:
                    return 1 if item in items else 0
                else:
                    items_lower = [i.lower() for i in items]
                    return 1 if item.lower() in items_lower else 0
            
            df[col_name] = df[col].apply(check_item)
        
        binary_cols[col_name] = df[col_name]

    binary_df = pd.DataFrame(binary_cols)
    
    return df, {"binary": binary_df}


def get_entity_stats(
    df: pd.DataFrame,
    entity_col: str,
    entity_label: str,
    count_method: Optional[Callable] = None,
    items_of_interest: Optional[List[str]] = None,
    exclude_items: Optional[List[str]] = None,
    top_n: int = 20,
    value_type: str = "list",
    sep: str = "; ",
    indicators: bool = False,
    **kwargs
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Compute comprehensive statistics for an entity type.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    entity_col : str
        Column containing entity values.
    entity_label : str
        Label for the entity in output.
    count_method : callable, optional
        Method to get counts DataFrame.
    items_of_interest : list, optional
        Specific items to analyze.
    exclude_items : list, optional
        Items to exclude from analysis.
    top_n : int
        Number of top items to include.
    value_type : {"list", "string", "text"}
        Type of values in the column.
    sep : str
        Separator for list values.
    indicators : bool
        Whether to compute binary indicators.
    **kwargs
        Additional arguments.

    Returns
    -------
    tuple
        (stats DataFrame, indicators DataFrame or None)
    """
    # Get counts if method provided
    if count_method is not None:
        counts_df = count_method(**kwargs)
    else:
        counts_df = count_occurrences(
            df, entity_col,
            count_type=value_type if value_type != "string" else "single",
            item_column_name=entity_label,
            sep=sep,
        )

    # Determine items of interest
    if items_of_interest is None:
        # Get item column (first column)
        item_col = counts_df.columns[0]
        items_of_interest = counts_df[item_col].head(top_n).tolist()

    # Exclude items if specified
    if exclude_items:
        items_of_interest = [i for i in items_of_interest if i not in exclude_items]

    # Filter to items of interest
    item_col = counts_df.columns[0]
    stats_df = counts_df[counts_df[item_col].isin(items_of_interest)].copy()
    stats_df = stats_df.head(top_n)

    # Compute indicators if requested
    indicators_df = None
    if indicators:
        _, ind_dict = match_items_and_compute_binary_indicators(
            df, entity_col, items_of_interest,
            value_type=value_type,
            separator=sep,
        )
        indicators_df = ind_dict.get("binary")

    return stats_df, indicators_df
