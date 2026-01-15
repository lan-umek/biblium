# -*- coding: utf-8 -*-
"""
Group utilities - group-based analysis and comparisons.

This module contains:
- generate_group_matrix: Create group membership indicators
- count_occurrences_across_groups: Count items by group
- group_entity_stats: Compute group-level statistics
- Group comparison functions
"""

from __future__ import annotations

from functools import reduce
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import pandas as pd
import numpy as np


def generate_group_matrix(
    df: pd.DataFrame,
    group_col: str,
    sep: str = "; ",
    value_type: str = "single",
) -> pd.DataFrame:
    """
    Generate a binary group membership matrix from a column.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    group_col : str
        Column containing group assignments.
    sep : str
        Separator for multi-valued group assignments.
    value_type : {"single", "list"}
        Whether each row has one group or multiple.

    Returns
    -------
    DataFrame
        Binary matrix with groups as columns.
    """
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in DataFrame")
    
    # Get all unique groups
    if value_type == "list":
        all_groups = set()
        for val in df[group_col].dropna():
            groups = [g.strip() for g in str(val).split(sep) if g.strip()]
            all_groups.update(groups)
        all_groups = sorted(all_groups)
    else:
        all_groups = sorted(df[group_col].dropna().unique())
    
    # Create binary matrix
    matrix = pd.DataFrame(0, index=df.index, columns=all_groups)
    
    for idx, val in df[group_col].items():
        if pd.isna(val):
            continue
        
        if value_type == "list":
            groups = [g.strip() for g in str(val).split(sep) if g.strip()]
        else:
            groups = [val]
        
        for g in groups:
            if g in matrix.columns:
                matrix.loc[idx, g] = 1
    
    return matrix


def merge_group_performances(
    group_dfs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge performance DataFrames from different groups.

    Parameters
    ----------
    group_dfs : dict
        Dictionary mapping group names to performance DataFrames.

    Returns
    -------
    DataFrame
        Merged DataFrame with columns for each group.
    """
    merged_df = None
    
    for group_name, df in group_dfs.items():
        temp_df = df.copy()
        temp_df = temp_df.rename(columns={"Value": group_name})
        
        if merged_df is None:
            merged_df = temp_df
        else:
            merge_cols = [c for c in ["Variable", "Indicator"] if c in temp_df.columns]
            if merge_cols:
                merged_df = pd.merge(merged_df, temp_df, on=merge_cols, how="outer")
            else:
                merged_df = pd.concat([merged_df, temp_df], axis=1)
    
    return merged_df


def count_occurrences_across_groups(
    groups: Dict[str, Any],
    group_matrix: pd.DataFrame,
    count_func_name: str,
    merge_type: str = "all items",
    **kwargs,
) -> pd.DataFrame:
    """
    Count item occurrences across multiple groups.

    Parameters
    ----------
    groups : dict
        Mapping of group names to group objects with count methods.
    group_matrix : DataFrame
        Group membership matrix (groups as columns).
    count_func_name : str
        Name of the counting method to call on each group.
    merge_type : {"all items", "shared items"}
        How to merge results across groups.
    **kwargs
        Additional arguments passed to the counting function.

    Returns
    -------
    DataFrame
        Merged counts with group-labeled columns and combined statistics.
    """
    how_map = {"all items": "outer", "shared items": "inner"}
    how = how_map.get(merge_type)
    if how is None:
        raise ValueError('merge_type must be "all items" or "shared items"')
    
    if group_matrix is None or len(getattr(group_matrix, "columns", [])) == 0:
        raise ValueError("group_matrix must have at least one column (group).")
    
    # Run the counting function per group
    dfs = []
    group_names = list(group_matrix.columns)
    
    for g in group_names:
        if g not in groups:
            raise KeyError(f'Group "{g}" not found in the provided groups mapping.')
        if not hasattr(groups[g], count_func_name):
            raise AttributeError(f'Group "{g}" has no method "{count_func_name}".')
        
        df = getattr(groups[g], count_func_name)(**kwargs)
        if not isinstance(df, pd.DataFrame) or df.shape[1] < 2:
            raise ValueError(f'Counting method for group "{g}" must return a DataFrame with at least 2 columns.')
        
        key_col = df.columns[0]
        df = df.rename(columns={c: f"{c} ({g})" for c in df.columns[1:]})
        dfs.append(df)
    
    # Ensure all key columns match
    first_key = dfs[0].columns[0]
    if any(d.columns[0] != first_key for d in dfs[1:]):
        raise ValueError("All group DataFrames must share the same first (key) column.")
    
    # Merge all group DataFrames
    merged = reduce(lambda l, r: pd.merge(l, r, on=first_key, how=how), dfs)
    
    # Identify rank columns vs count columns
    rank_cols = [col for col in merged.columns if "rank" in col.lower()]
    non_rank_cols = [col for col in merged.columns[1:] if col not in rank_cols]
    
    # Fill NaN: 0 for counts, leave NaN for ranks
    for col in non_rank_cols:
        merged[col] = merged[col].fillna(0)
    
    # Numeric conversion
    for col in merged.columns[1:]:
        try:
            merged[col] = pd.to_numeric(merged[col])
        except (ValueError, TypeError):
            pass
    
    # Compute combined statistics
    base_cols = set()
    for col in merged.columns[1:]:
        if "(" in col and ")" in col:
            base_name = col.split("(")[0].strip()
            base_cols.add(base_name)
    
    # Collect new columns to avoid fragmentation
    new_cols = {}
    
    for base_col in base_cols:
        if "rank" in base_col.lower():
            continue
        
        metric_cols = [col for col in merged.columns if col.startswith(f"{base_col} (")]
        
        if metric_cols:
            if "OCC" in base_col or "Num" in base_col or base_col.startswith("N"):
                new_cols[f"{base_col} (Combined)"] = merged[metric_cols].sum(axis=1)
    
    # Recalculate proportions/percentages
    count_col_name = None
    for base_col in base_cols:
        if "OCC" in base_col or "Num" in base_col or (base_col.startswith("N") and "rank" not in base_col.lower()):
            count_col_name = f"{base_col} (Combined)"
            break
    
    if count_col_name and count_col_name in new_cols:
        total_combined = new_cols[count_col_name].sum()
        
        for base_col in base_cols:
            if "Proportion" in base_col:
                new_cols[f"{base_col} (Combined)"] = new_cols[count_col_name] / total_combined if total_combined > 0 else 0
            elif "Percentage" in base_col or "%" in base_col:
                new_cols[f"{base_col} (Combined)"] = (new_cols[count_col_name] / total_combined * 100) if total_combined > 0 else 0
        
        # Compute combined ranks
        count_series = new_cols[count_col_name]
        ranks = count_series.rank(method="first", ascending=False).astype(int)
        new_cols["Rank (Combined)"] = ranks
        
        max_rank = len(merged)
        new_cols["PercentRank (Combined)"] = ((max_rank - ranks + 1) / max_rank * 100) if max_rank > 0 else 0
    
    # Add all new columns at once
    if new_cols:
        new_cols_df = pd.DataFrame(new_cols, index=merged.index)
        merged = pd.concat([merged, new_cols_df], axis=1)
    
    # Reorder columns
    key_col = merged.columns[0]
    ordered_cols = [key_col]
    
    for g in group_names:
        group_cols = [col for col in merged.columns if f"({g})" in col]
        ordered_cols.extend(sorted(group_cols))
    
    combined_cols = [col for col in merged.columns if "(Combined)" in col]
    ordered_cols.extend(sorted(combined_cols))
    
    merged = merged[ordered_cols]
    
    return merged


def compute_group_intersections(
    group_matrix: pd.DataFrame,
    include_ids: bool = False,
    id_column: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute all unique intersections from a group membership matrix.

    Parameters
    ----------
    group_matrix : DataFrame
        Binary group membership matrix.
    include_ids : bool
        Whether to include document IDs in results.
    id_column : Series, optional
        Document IDs if include_ids is True.

    Returns
    -------
    DataFrame
        Intersection statistics.
    """
    from itertools import combinations
    
    groups = list(group_matrix.columns)
    results = []
    
    # Single groups
    for g in groups:
        mask = group_matrix[g] == 1
        count = mask.sum()
        result = {"Groups": g, "Count": count}
        if include_ids and id_column is not None:
            result["IDs"] = list(id_column[mask])
        results.append(result)
    
    # Pairwise intersections
    for g1, g2 in combinations(groups, 2):
        mask = (group_matrix[g1] == 1) & (group_matrix[g2] == 1)
        count = mask.sum()
        if count > 0:
            result = {"Groups": f"{g1} & {g2}", "Count": count}
            if include_ids and id_column is not None:
                result["IDs"] = list(id_column[mask])
            results.append(result)
    
    return pd.DataFrame(results)


def compare_continuous_by_binary_groups(
    df: pd.DataFrame,
    continuous_col: str,
    group_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare a continuous variable across binary groups.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    continuous_col : str
        Column with continuous values.
    group_matrix : DataFrame
        Binary group membership matrix.

    Returns
    -------
    DataFrame
        Comparison statistics for each group.
    """
    results = []
    
    values = pd.to_numeric(df[continuous_col], errors="coerce")
    
    for group in group_matrix.columns:
        mask = group_matrix[group] == 1
        group_values = values[mask].dropna()
        non_group_values = values[~mask].dropna()
        
        result = {
            "Group": group,
            "N": len(group_values),
            "Mean": group_values.mean(),
            "Std": group_values.std(),
            "Median": group_values.median(),
            "Non-Group Mean": non_group_values.mean(),
        }
        
        # Statistical test
        if len(group_values) > 1 and len(non_group_values) > 1:
            from scipy.stats import mannwhitneyu
            try:
                stat, pval = mannwhitneyu(group_values, non_group_values, alternative="two-sided")
                result["U-statistic"] = stat
                result["p-value"] = pval
            except Exception:
                pass
        
        results.append(result)
    
    return pd.DataFrame(results)


def group_entity_stats(
    df: pd.DataFrame,
    group_matrix: pd.DataFrame,
    entity_col: str,
    entity_label: str,
    items_of_interest: Optional[List[str]] = None,
    exclude_items: Optional[List[str]] = None,
    top_n: int = 20,
    count_method: Optional[Callable] = None,
    output_format: str = "wide",
    indicators: bool = False,
    value_type: str = "list",
    sep: str = "; ",
    **extra_kwargs,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Compute group-level entity statistics.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    group_matrix : DataFrame
        Binary group membership matrix.
    entity_col : str
        Column containing entity values.
    entity_label : str
        Label for the entity type.
    items_of_interest : list, optional
        Specific items to analyze.
    exclude_items : list, optional
        Items to exclude.
    top_n : int
        Number of top items per group.
    count_method : callable, optional
        Custom counting method.
    output_format : {"wide", "long"}
        Output format.
    indicators : bool
        Whether to compute binary indicators.
    value_type : str
        Type of values in entity column.
    sep : str
        Separator for list values.
    **extra_kwargs
        Additional arguments.

    Returns
    -------
    tuple
        (stats DataFrame, indicators DataFrame or None)
    """
    from biblium.utilsbib_modules.counting import count_occurrences, match_items_and_compute_binary_indicators
    
    groups = list(group_matrix.columns)
    all_results = []
    
    for group in groups:
        mask = group_matrix[group] == 1
        group_df = df[mask]
        
        if count_method is not None:
            counts = count_method()
        else:
            counts = count_occurrences(
                group_df, entity_col,
                count_type=value_type if value_type != "string" else "single",
                item_column_name=entity_label,
                sep=sep,
            )
        
        counts["Group"] = group
        all_results.append(counts)
    
    # Combine results
    combined = pd.concat(all_results, ignore_index=True)
    
    # Filter to items of interest
    if items_of_interest is not None:
        combined = combined[combined[entity_label].isin(items_of_interest)]
    
    if exclude_items:
        combined = combined[~combined[entity_label].isin(exclude_items)]
    
    # Format output
    if output_format == "wide":
        # Pivot to wide format
        stats = combined.pivot_table(
            index=entity_label,
            columns="Group",
            values="Number of documents",
            fill_value=0
        )
    else:
        stats = combined
    
    # Compute indicators if requested
    indicators_df = None
    if indicators and items_of_interest:
        _, ind_dict = match_items_and_compute_binary_indicators(
            df, entity_col, items_of_interest,
            value_type=value_type,
            separator=sep,
        )
        indicators_df = ind_dict.get("binary")
    
    return stats, indicators_df
