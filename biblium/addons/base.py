# -*- coding: utf-8 -*-
"""
Biblium Addon Base Classes

This module provides abstract base classes that enforce a consistent
interface for all addon modules. New addons should inherit from these
classes to ensure compatibility with the Biblium ecosystem.

@author: Claude (Anthropic) for Lan.Umek
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from biblium.bibstats import BiblioStats


# =============================================================================
# ABSTRACT RESULT CLASS
# =============================================================================

@dataclass
class AddonResult(ABC):
    """
    Abstract base class for addon analysis results.
    
    All addon result classes should inherit from this class to ensure
    consistent interface for accessing and exporting results.
    
    Required methods to implement:
    - to_dict(): Convert results to dictionary
    - to_dataframe(): Convert main results to DataFrame
    - summary(): Return a text summary
    
    Optional methods:
    - to_excel(): Export to Excel file
    - plot(): Generate visualization
    
    Example
    -------
    @dataclass
    class MyAnalysisResult(AddonResult):
        data: pd.DataFrame
        metrics: Dict[str, float]
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                "data": self.data.to_dict(),
                "metrics": self.metrics
            }
        
        def to_dataframe(self) -> pd.DataFrame:
            return self.data
        
        def summary(self) -> str:
            return f"Analysis with {len(self.data)} records"
    """
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to a dictionary.
        
        Returns
        -------
        dict
            Dictionary representation of all results.
        """
        pass
    
    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert main results to a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Primary results as a DataFrame.
        """
        pass
    
    @abstractmethod
    def summary(self) -> str:
        """
        Return a human-readable summary of results.
        
        Returns
        -------
        str
            Text summary of the analysis.
        """
        pass
    
    def to_excel(self, filepath: str, **kwargs) -> None:
        """
        Export results to an Excel file.
        
        Parameters
        ----------
        filepath : str
            Output file path.
        **kwargs
            Additional arguments for pandas ExcelWriter.
        """
        df = self.to_dataframe()
        if df is not None and len(df) > 0:
            df.to_excel(filepath, index=False, **kwargs)
        else:
            raise ValueError("No data available to export")
    
    def __repr__(self) -> str:
        return self.summary()
    
    def __str__(self) -> str:
        return self.summary()


# =============================================================================
# ABSTRACT ANALYZER CLASS
# =============================================================================

class AddonAnalyzer(ABC):
    """
    Abstract base class for addon analyzers.
    
    Provides a consistent interface for running analyses on bibliometric data.
    Analyzers can work with either a pandas DataFrame directly or with a
    BiblioStats object.
    
    Required methods to implement:
    - analyze(): Run the analysis and return results
    - validate_input(): Check if input data is valid
    
    Optional methods:
    - preprocess(): Prepare data before analysis
    - get_required_columns(): List required DataFrame columns
    
    Attributes
    ----------
    name : str
        Name of the analyzer.
    version : str
        Version of the analyzer.
    description : str
        Brief description of what the analyzer does.
    
    Example
    -------
    class MyAnalyzer(AddonAnalyzer):
        name = "My Analyzer"
        version = "1.0.0"
        description = "Analyzes something interesting"
        
        def get_required_columns(self) -> List[str]:
            return ["Title", "Abstract", "Year"]
        
        def validate_input(self, df: pd.DataFrame) -> Tuple[bool, str]:
            missing = [c for c in self.get_required_columns() if c not in df.columns]
            if missing:
                return False, f"Missing columns: {missing}"
            return True, "OK"
        
        def analyze(self, df: pd.DataFrame, **kwargs) -> MyAnalysisResult:
            valid, msg = self.validate_input(df)
            if not valid:
                raise ValueError(msg)
            # ... perform analysis ...
            return MyAnalysisResult(data=result_df, metrics=metrics)
    """
    
    # Class attributes - override in subclasses
    name: str = "Base Analyzer"
    version: str = "1.0.0"
    description: str = "Abstract base analyzer"
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the analyzer.
        
        Parameters
        ----------
        verbose : bool
            Whether to print progress messages.
        """
        self.verbose = verbose
    
    @abstractmethod
    def analyze(
        self,
        data: Union[pd.DataFrame, "BiblioStats"],
        **kwargs
    ) -> AddonResult:
        """
        Run the analysis.
        
        Parameters
        ----------
        data : pd.DataFrame or BiblioStats
            Input data to analyze.
        **kwargs
            Additional analysis parameters.
        
        Returns
        -------
        AddonResult
            Analysis results.
        """
        pass
    
    @abstractmethod
    def validate_input(
        self,
        df: pd.DataFrame
    ) -> Tuple[bool, str]:
        """
        Validate that input data has required columns/format.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to validate.
        
        Returns
        -------
        tuple
            (is_valid: bool, message: str)
        """
        pass
    
    def get_required_columns(self) -> List[str]:
        """
        Get list of required DataFrame columns.
        
        Returns
        -------
        list
            Column names required for analysis.
        """
        return []
    
    def get_optional_columns(self) -> List[str]:
        """
        Get list of optional DataFrame columns.
        
        Returns
        -------
        list
            Column names that enhance analysis if present.
        """
        return []
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data before analysis.
        
        Override this method to add custom preprocessing steps.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Preprocessed DataFrame.
        """
        return df.copy()
    
    def _get_dataframe(
        self,
        data: Union[pd.DataFrame, "BiblioStats"]
    ) -> pd.DataFrame:
        """
        Extract DataFrame from input (handles both DataFrame and BiblioStats).
        
        Parameters
        ----------
        data : pd.DataFrame or BiblioStats
            Input data.
        
        Returns
        -------
        pd.DataFrame
            Extracted DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            return data
        elif hasattr(data, 'df'):
            return data.df
        else:
            raise TypeError(f"Expected DataFrame or BiblioStats, got {type(data)}")
    
    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{self.name}] {message}")
    
    def run(
        self,
        data: Union[pd.DataFrame, "BiblioStats"],
        **kwargs
    ) -> AddonResult:
        """
        Run analysis with validation and preprocessing.
        
        This is the recommended entry point that handles:
        1. Input extraction
        2. Validation
        3. Preprocessing
        4. Analysis
        
        Parameters
        ----------
        data : pd.DataFrame or BiblioStats
            Input data.
        **kwargs
            Additional parameters.
        
        Returns
        -------
        AddonResult
            Analysis results.
        """
        # Extract DataFrame
        df = self._get_dataframe(data)
        
        # Validate
        is_valid, message = self.validate_input(df)
        if not is_valid:
            raise ValueError(f"Validation failed: {message}")
        
        self._log(f"Starting analysis on {len(df)} records")
        
        # Preprocess
        df_processed = self.preprocess(df)
        
        # Run analysis
        result = self.analyze(df_processed, **kwargs)
        
        self._log("Analysis complete")
        
        return result
    
    def __repr__(self) -> str:
        return f"{self.name} v{self.version}: {self.description}"


# =============================================================================
# MIXIN CLASSES FOR COMMON FUNCTIONALITY
# =============================================================================

class VisualizationMixin:
    """
    Mixin class providing visualization capabilities.
    
    Add this to result classes that need plotting functionality.
    """
    
    def get_available_plots(self) -> List[str]:
        """
        Get list of available plot types.
        
        Returns
        -------
        list
            Names of available plot methods.
        """
        return [m for m in dir(self) if m.startswith('plot_')]
    
    def plot(self, plot_type: str = None, **kwargs):
        """
        Generate a visualization.
        
        Parameters
        ----------
        plot_type : str
            Type of plot to generate. If None, uses default.
        **kwargs
            Additional plotting parameters.
        """
        if plot_type is None:
            available = self.get_available_plots()
            if available:
                plot_type = available[0].replace('plot_', '')
            else:
                raise ValueError("No plots available")
        
        method_name = f"plot_{plot_type}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(**kwargs)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}. Available: {self.get_available_plots()}")


class TemporalMixin:
    """
    Mixin class for temporal analysis capabilities.
    
    Add this to analyzers that work with time-series data.
    """
    
    def get_time_range(self, df: pd.DataFrame, year_col: str = "Year") -> Tuple[int, int]:
        """
        Get the time range of the data.
        
        Returns
        -------
        tuple
            (min_year, max_year)
        """
        years = pd.to_numeric(df[year_col], errors='coerce').dropna()
        return int(years.min()), int(years.max())
    
    def create_time_windows(
        self,
        df: pd.DataFrame,
        year_col: str = "Year",
        window_size: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data into time windows.
        
        Returns
        -------
        dict
            Mapping of period labels to DataFrames.
        """
        min_year, max_year = self.get_time_range(df, year_col)
        windows = {}
        
        for start in range(min_year, max_year + 1, window_size):
            end = min(start + window_size - 1, max_year)
            mask = (df[year_col] >= start) & (df[year_col] <= end)
            windows[f"{start}-{end}"] = df[mask].copy()
        
        return windows


class NetworkMixin:
    """
    Mixin class for network analysis capabilities.
    
    Add this to analyzers that build and analyze networks.
    """
    
    def build_cooccurrence_matrix(
        self,
        df: pd.DataFrame,
        column: str,
        sep: str = "; "
    ) -> pd.DataFrame:
        """
        Build a co-occurrence matrix from a column with multiple values.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        column : str
            Column containing lists of items (separated by sep).
        sep : str
            Separator for splitting values.
        
        Returns
        -------
        pd.DataFrame
            Co-occurrence matrix.
        """
        from collections import Counter
        from itertools import combinations
        
        # Extract all items
        all_items = set()
        for val in df[column].dropna():
            items = [i.strip() for i in str(val).split(sep) if i.strip()]
            all_items.update(items)
        
        # Count co-occurrences
        cooc = Counter()
        for val in df[column].dropna():
            items = [i.strip() for i in str(val).split(sep) if i.strip()]
            for a, b in combinations(sorted(set(items)), 2):
                cooc[(a, b)] += 1
        
        # Build matrix
        items = sorted(all_items)
        matrix = pd.DataFrame(0, index=items, columns=items)
        
        for (a, b), count in cooc.items():
            matrix.loc[a, b] = count
            matrix.loc[b, a] = count
        
        return matrix


# =============================================================================
# REGISTRATION AND DISCOVERY
# =============================================================================

# Registry of all available addons
_ADDON_REGISTRY: Dict[str, type] = {}


def register_addon(cls: type) -> type:
    """
    Decorator to register an addon analyzer.
    
    Example
    -------
    @register_addon
    class MyAnalyzer(AddonAnalyzer):
        name = "my_analyzer"
        ...
    """
    if hasattr(cls, 'name'):
        _ADDON_REGISTRY[cls.name] = cls
    return cls


def get_registered_addons() -> Dict[str, type]:
    """
    Get all registered addon analyzers.
    
    Returns
    -------
    dict
        Mapping of addon names to classes.
    """
    return _ADDON_REGISTRY.copy()


def get_addon(name: str) -> Optional[type]:
    """
    Get a registered addon by name.
    
    Parameters
    ----------
    name : str
        Addon name.
    
    Returns
    -------
    type or None
        Addon class if found, None otherwise.
    """
    return _ADDON_REGISTRY.get(name)


def list_addons() -> List[str]:
    """
    List all registered addon names.
    
    Returns
    -------
    list
        List of addon names.
    """
    return list(_ADDON_REGISTRY.keys())


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_columns(
    df: pd.DataFrame,
    required: List[str],
    optional: List[str] = None
) -> Tuple[bool, str, List[str]]:
    """
    Validate that DataFrame has required columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required : list
        Required column names.
    optional : list
        Optional column names.
    
    Returns
    -------
    tuple
        (is_valid, message, available_optional)
    """
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        return False, f"Missing required columns: {missing}", []
    
    available_optional = []
    if optional:
        available_optional = [c for c in optional if c in df.columns]
    
    return True, "OK", available_optional


def find_column(
    df: pd.DataFrame,
    candidates: List[str],
    required: bool = True
) -> Optional[str]:
    """
    Find the first matching column from a list of candidates.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to search.
    candidates : list
        List of column name candidates (in priority order).
    required : bool
        Whether to raise error if not found.
    
    Returns
    -------
    str or None
        Found column name.
    
    Raises
    ------
    ValueError
        If required=True and no column found.
    """
    for col in candidates:
        if col in df.columns:
            return col
    
    if required:
        raise ValueError(f"None of the columns found: {candidates}")
    
    return None


# =============================================================================
# EXAMPLE IMPLEMENTATION
# =============================================================================

@dataclass
class ExampleResult(AddonResult, VisualizationMixin):
    """Example result class showing proper implementation."""
    
    data: pd.DataFrame
    metrics: Dict[str, float] = field(default_factory=dict)
    n_records: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": self.data.to_dict() if self.data is not None else {},
            "metrics": self.metrics,
            "n_records": self.n_records,
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        return self.data
    
    def summary(self) -> str:
        metric_str = ", ".join(f"{k}={v:.2f}" for k, v in self.metrics.items())
        return f"ExampleResult: {self.n_records} records, metrics: {metric_str}"
    
    def plot_bar(self, **kwargs):
        """Example plot method."""
        import matplotlib.pyplot as plt
        if self.data is not None and len(self.data) > 0:
            self.data.plot(kind='bar', **kwargs)
            plt.tight_layout()
            return plt.gcf()


class ExampleAnalyzer(AddonAnalyzer):
    """Example analyzer class showing proper implementation."""
    
    name = "example_analyzer"
    version = "1.0.0"
    description = "Example analyzer for demonstration"
    
    def get_required_columns(self) -> List[str]:
        return ["Year"]
    
    def get_optional_columns(self) -> List[str]:
        return ["Cited by", "Title"]
    
    def validate_input(self, df: pd.DataFrame) -> Tuple[bool, str]:
        return validate_columns(df, self.get_required_columns())[:2]
    
    def analyze(
        self,
        data: Union[pd.DataFrame, "BiblioStats"],
        **kwargs
    ) -> ExampleResult:
        df = self._get_dataframe(data)
        
        # Example analysis: count by year
        year_counts = df["Year"].value_counts().sort_index()
        result_df = year_counts.reset_index()
        result_df.columns = ["Year", "Count"]
        
        metrics = {
            "total": len(df),
            "years_covered": len(year_counts),
            "avg_per_year": len(df) / len(year_counts) if len(year_counts) > 0 else 0,
        }
        
        return ExampleResult(
            data=result_df,
            metrics=metrics,
            n_records=len(df)
        )
