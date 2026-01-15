# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 11:54:37 2025

@author: Lan.Umek

Main entry points for bibliometric analysis with tiered execution modes.

Enhanced with:
- Streamlined report workflow
- Progress feedback
- Lazy evaluation
- Better error messages
- Caching layer
- Fluent API support
- Data validation
- Interactive features
"""

import os
import sys
import warnings
from typing import Optional, List, Dict, Any, Callable, Tuple, Union
from biblium.bibstats import BiblioStats
from biblium.bibplot import BiblioPlot, BiblioGroupPlot
from biblium.bibgroup import BiblioGroup
from biblium.bibclass import BiblioGroupClassifier
from biblium import reportbib

# Import enhancements
from biblium.enhancements import (
    # Error handling
    BibliumError,
    ColumnNotFoundError,
    InsufficientDataError,
    suggest_column,
    safe_get_column,
    # Caching
    AnalysisCache,
    CacheConfig,
    cached_analysis,
    compute_data_hash,
    # Validation
    ValidationResult,
    validate_bibliometric_data,
    # Lazy evaluation
    LazyResult,
    # Fluent API
    FluentMixin,
    # Summary and introspection
    DatasetSummary,
    get_available_analyses,
    what_can_i_do,
    make_repr,
    make_repr_html,
    # Export presets
    EXPORT_PRESETS,
    get_export_presets,
    list_export_presets,
    # Plugin system
    register_analysis,
    get_registered_analyses,
    run_custom_analysis,
    # Config
    ProjectConfig,
    load_project_config,
)


def _safe_execute(func: Callable, name: str, errors: List[str], verbose: bool = True, **kwargs) -> bool:
    """
    Safely execute a function with error handling.
    
    Parameters
    ----------
    func : Callable
        Function to execute.
    name : str
        Name of the function for logging.
    errors : list
        List to append errors to.
    verbose : bool
        Whether to print progress.
    **kwargs
        Arguments to pass to the function.
    
    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    try:
        if verbose:
            print(f"  → {name}...", end=" ", flush=True)
        func(**kwargs)
        if verbose:
            print("✓")
        return True
    except Exception as e:
        error_msg = f"{name}: {type(e).__name__}: {str(e)[:100]}"
        errors.append(error_msg)
        if verbose:
            print(f"✗ ({type(e).__name__})")
        return False


class BiblioAnalysis(FluentMixin, BiblioPlot):
    """
    Main class for comprehensive bibliometric analysis.
    
    Inherits from BiblioPlot and provides tiered execution modes:
    - basic: Core counts and main information
    - extended: Adds performance stats, laws, and visualizations
    - full: Adds advanced analytics (networks, co-occurrence, etc.)
    - full+: Adds LLM-based analyses and experimental features
    
    Enhanced Features
    -----------------
    - Fluent/chainable API: ba.filter(...).count_all().export()
    - Lazy evaluation: ba.sources returns lazy result
    - Caching: Results cached to disk for faster repeated analysis
    - Validation: ba.validate() checks data quality
    - Introspection: ba.what_can_i_do() shows available analyses
    - Verbose tracking: Shows computed attributes when verbose=True
    
    Examples
    --------
    >>> # Basic usage with verbose output
    >>> ba = BiblioAnalysis("data.csv", db="scopus", verbose=True)
    >>> ba.count_sources()
    ✓ count_sources completed
      → New: self.sources_counts_df (DataFrame: 150 rows × 4 cols)
    >>> 
    >>> # Fluent API
    >>> (ba
    ...     .filter(year_range=(2020, 2024))
    ...     .count_all()
    ...     .export("report.docx"))
    >>> 
    >>> # Generate report with automatic preparation
    >>> ba.generate_report(level="basic", formats=["docx", "xlsx"])
    """
    
    def __init__(
        self,
        f_name: Optional[str] = None,
        db: str = "",
        df: Optional[Any] = None,
        res_folder: Optional[str] = "results",
        output_lang: str = "en",
        cache_dir: Optional[str] = None,
        validate: bool = False,
        config_file: Optional[str] = None,
        verbose: bool = True,
        **kwargs
    ):
        """
        Initialize BiblioAnalysis.
        
        Parameters
        ----------
        f_name : str, optional
            Path to the data file.
        db : str
            Database type: "scopus", "wos", "oa", etc.
        df : DataFrame, optional
            Pre-loaded DataFrame instead of file.
        res_folder : str, optional
            Folder for results. None to disable saving.
        output_lang : str, default "en"
            Output language.
        cache_dir : str, optional
            Directory for caching results. None to disable.
        validate : bool, default False
            Run data validation on initialization.
        config_file : str, optional
            Path to project config file (biblium.yaml).
        verbose : bool, default True
            Print information about computed attributes.
            Shows what results are stored and where to access them.
        """
        # Load project config if specified
        if config_file:
            self._project_config = load_project_config(config_file)
            if not db:
                db = self._project_config.database
        else:
            self._project_config = None
        
        # Initialize parent
        super().__init__(
            f_name=f_name,
            db=db,
            df=df,
            res_folder=res_folder,
            output_lang=output_lang,
            **kwargs
        )
        
        # Setup caching
        if cache_dir:
            self._cache = AnalysisCache(cache_dir=cache_dir, enabled=True)
            self._data_hash = compute_data_hash(self.df)
        else:
            self._cache = AnalysisCache(enabled=False)
            self._data_hash = None
        
        # Setup verbose tracking
        self._verbose = verbose
        self._tracked_attrs = {}
        if verbose:
            self._snapshot_attrs()
        
        # Run validation if requested
        if validate:
            result = self.validate(verbose=verbose)
            if not result.valid:
                warnings.warn(f"Data validation failed: {result.errors}")
    
    # =========================================================================
    # VERBOSE TRACKING
    # =========================================================================
    
    def _snapshot_attrs(self):
        """Take a snapshot of current attributes for comparison."""
        self._tracked_attrs = {
            k: id(v) for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
    
    def _report_new_attrs(self, method_name: str):
        """Report any new or changed attributes after a method call."""
        if not self._verbose:
            return
        
        current_attrs = {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
        
        new_attrs = []
        changed_attrs = []
        
        for k, v in current_attrs.items():
            if k not in self._tracked_attrs:
                new_attrs.append((k, v))
            elif id(v) != self._tracked_attrs.get(k):
                changed_attrs.append((k, v))
        
        if new_attrs or changed_attrs:
            print(f"✓ {method_name} completed")
            
            for attr_name, attr_value in new_attrs + changed_attrs:
                attr_desc = self._describe_attribute(attr_value)
                prefix = "→ New" if (attr_name, attr_value) in new_attrs else "→ Updated"
                print(f"  {prefix}: self.{attr_name} {attr_desc}")
        
        # Update snapshot
        self._snapshot_attrs()
    
    @staticmethod
    def _describe_attribute(value: Any) -> str:
        """Create a short description of an attribute value."""
        if value is None:
            return "(None)"
        
        type_name = type(value).__name__
        
        # NetworkX graph check
        if hasattr(value, 'number_of_nodes') and hasattr(value, 'number_of_edges'):
            try:
                n_nodes = value.number_of_nodes()
                n_edges = value.number_of_edges()
                return f"(Graph: {n_nodes} nodes, {n_edges} edges)"
            except:
                pass
        
        if hasattr(value, 'shape'):  # DataFrame, ndarray
            shape = value.shape
            if len(shape) == 2:
                return f"(DataFrame: {shape[0]} rows × {shape[1]} cols)"
            else:
                return f"(array: shape {shape})"
        elif isinstance(value, dict):
            return f"(dict: {len(value)} items)"
        elif isinstance(value, (list, tuple)):
            return f"({type_name}: {len(value)} items)"
        elif isinstance(value, str):
            if len(value) > 50:
                return f'(str: "{value[:50]}...")'
            return f'(str: "{value}")'
        elif isinstance(value, (int, float)):
            return f"({type_name}: {value})"
        else:
            return f"({type_name})"
    
    def __setattr__(self, name: str, value: Any):
        """Track attribute changes for verbose reporting."""
        super().__setattr__(name, value)
        
        # Don't track private attributes or during initialization
        if name.startswith('_') or not hasattr(self, '_verbose'):
            return
        
        # Report if verbose and this is a new/changed public attribute
        if getattr(self, '_verbose', False) and not name.startswith('_'):
            # Only report DataFrames, dicts, NetworkX graphs, and other result types
            is_reportable = (
                hasattr(value, 'shape') or 
                isinstance(value, dict) or
                (hasattr(value, 'nodes') and hasattr(value, 'edges'))  # NetworkX graph
            )
            if is_reportable:
                attr_desc = self._describe_attribute(value)
                print(f"  → Stored: self.{name} {attr_desc}")
    
    # =========================================================================
    # REPR AND DISPLAY
    # =========================================================================
    
    def __repr__(self) -> str:
        """Nice string representation."""
        return make_repr(self)
    
    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        return make_repr_html(self)
    
    # =========================================================================
    # SUMMARY AND INTROSPECTION
    # =========================================================================
    
    def summary(self, verbose: bool = True) -> DatasetSummary:
        """
        Get a comprehensive summary of the dataset.
        
        Parameters
        ----------
        verbose : bool, default True
            Print the summary.
        
        Returns
        -------
        DatasetSummary
            Summary object with key statistics.
        
        Examples
        --------
        >>> ba = BiblioAnalysis("data.csv", db="scopus")
        >>> summary = ba.summary()
        >>> print(summary.n_documents)
        """
        # Year range
        year_col = self.mapping.get("Year", "Year")
        if year_col in self.df.columns:
            years = self.df[year_col].dropna()
            year_range = (int(years.min()), int(years.max())) if len(years) > 0 else (0, 0)
        else:
            year_range = (0, 0)
        
        # Top sources
        source_col = self.mapping.get("Source title", "Source title")
        if source_col in self.df.columns:
            top_sources = self.df[source_col].value_counts().head(10).items()
            top_sources = [(str(k), int(v)) for k, v in top_sources]
        else:
            top_sources = []
        
        # Top authors
        author_col = self.mapping.get("Authors", "Authors")
        if author_col in self.df.columns:
            # Simple split for quick summary
            all_authors = self.df[author_col].dropna().str.split(self.default_separator).explode()
            top_authors = all_authors.value_counts().head(10).items()
            top_authors = [(str(k).strip(), int(v)) for k, v in top_authors]
        else:
            top_authors = []
        
        # Top keywords
        kw_col = self.mapping.get("Author Keywords", "Author Keywords")
        if kw_col in self.df.columns:
            all_kw = self.df[kw_col].dropna().str.split(self.default_separator).explode()
            top_keywords = all_kw.value_counts().head(10).items()
            top_keywords = [(str(k).strip(), int(v)) for k, v in top_keywords]
        else:
            top_keywords = []
        
        # Missing rates
        key_cols = ["Title", "Authors", "Year", "Abstract", "DOI", "Cited by"]
        missing_rates = {}
        for col in key_cols:
            mapped = self.mapping.get(col, col)
            if mapped in self.df.columns:
                missing_rates[col] = round(self.df[mapped].isna().mean() * 100, 1)
        
        # Available analyses
        analyses = get_available_analyses(self)
        available = [a["method"] for a in analyses if a["available"]]
        
        # Completed analyses
        completed = []
        result_attrs = [
            "sources_counts_df", "authors_counts_df", "author_keywords_counts_df",
            "index_keywords_counts_df", "affiliations_counts_df", "references_counts_df",
        ]
        for attr in result_attrs:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                completed.append(attr.replace("_counts_df", ""))
        
        summary = DatasetSummary(
            n_documents=self.n,
            n_columns=len(self.df.columns),
            database=self.db,
            year_range=year_range,
            top_sources=list(top_sources),
            top_authors=list(top_authors),
            top_keywords=list(top_keywords),
            missing_rates=missing_rates,
            analyses_available=available,
            analyses_completed=completed,
        )
        
        if verbose:
            print(summary)
        
        return summary
    
    def what_can_i_do(self) -> List[str]:
        """
        List available analyses based on the dataset columns.
        
        Returns
        -------
        list of str
            Method names that can be called.
        
        Examples
        --------
        >>> ba = BiblioAnalysis("data.csv", db="scopus")
        >>> methods = ba.what_can_i_do()
        """
        return what_can_i_do(self, verbose=True)
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    def validate(self, strict: bool = False, verbose: bool = True) -> ValidationResult:
        """
        Validate the dataset for common issues.
        
        Parameters
        ----------
        strict : bool, default False
            Treat warnings as errors.
        verbose : bool, default True
            Print validation results.
        
        Returns
        -------
        ValidationResult
            Validation results with errors, warnings, and info.
        
        Examples
        --------
        >>> ba = BiblioAnalysis("data.csv", db="scopus")
        >>> result = ba.validate()
        >>> if not result.valid:
        ...     print("Issues found:", result.errors)
        """
        result = validate_bibliometric_data(self.df, database=self.db, strict=strict)
        
        if verbose:
            print(result)
        
        return result
    
    # =========================================================================
    # LAZY PROPERTIES
    # =========================================================================
    
    @property
    def sources(self) -> LazyResult:
        """Lazy access to source counts."""
        return LazyResult(self, "count_sources", "sources_counts_df")
    
    @property
    def authors(self) -> LazyResult:
        """Lazy access to author counts."""
        return LazyResult(self, "count_authors", "authors_counts_df")
    
    @property  
    def keywords(self) -> LazyResult:
        """Lazy access to keyword counts."""
        return LazyResult(self, "count_author_keywords", "author_keywords_counts_df")
    
    @property
    def affiliations(self) -> LazyResult:
        """Lazy access to affiliation counts."""
        return LazyResult(self, "count_affiliations", "affiliations_counts_df")
    
    @property
    def references(self) -> LazyResult:
        """Lazy access to reference counts."""
        return LazyResult(self, "count_references", "references_counts_df")
    
    # =========================================================================
    # STREAMLINED REPORT WORKFLOW
    # =========================================================================
    
    def generate_report(
        self,
        output: str = "bibliometric_report",
        level: str = "basic",
        formats: List[str] = None,
        prepare: bool = True,
        verbose: bool = True,
    ) -> Dict[str, str]:
        """
        Generate reports with automatic data preparation.
        
        This is the recommended way to generate reports. It:
        1. Runs all necessary analyses for the report level
        2. Generates reports in the specified formats
        
        Parameters
        ----------
        output : str, default "bibliometric_report"
            Output path or base name for reports.
        level : str, default "basic"
            Report level: "basic", "standard", "extended", "full".
        formats : list, optional
            Output formats: ["xlsx", "docx", "pptx", "tex"].
            Default is ["docx", "xlsx"].
        prepare : bool, default True
            Automatically run prepare_for_report() first.
        verbose : bool, default True
            Print progress messages.
        
        Returns
        -------
        dict
            Mapping of format to output file path.
        
        Examples
        --------
        >>> ba = BiblioAnalysis("data.csv", db="scopus")
        >>> ba.generate_report(level="basic", formats=["docx"])
        """
        formats = formats or ["docx", "xlsx"]
        
        # Prepare data if needed
        if prepare:
            self.prepare_for_report(level=level, verbose=verbose)
        
        # Determine output directory and base name
        if os.path.dirname(output):
            output_dir = os.path.dirname(output)
            base_name = os.path.splitext(os.path.basename(output))[0]
        else:
            output_dir = os.path.join(self.res_folder or "results", "reports")
            base_name = output
        
        # Generate reports
        return self.generate_all_reports(
            output_dir=output_dir,
            base_name=base_name,
            formats=formats,
            template_sheet=level,
        )
    
    # =========================================================================
    # EXPORT WITH PRESETS
    # =========================================================================
    
    def export(
        self,
        output: str = "report",
        preset: str = None,
        formats: List[str] = None,
        level: str = "basic",
        **kwargs
    ) -> "BiblioAnalysis":
        """
        Export reports using presets or custom options.
        
        Parameters
        ----------
        output : str
            Output path or base name.
        preset : str, optional
            Preset name: "journal_submission", "thesis_appendix", 
            "presentation", "quick_overview", "full_analysis".
        formats : list, optional
            Override preset formats.
        level : str, default "basic"
            Override preset level.
        
        Returns
        -------
        BiblioAnalysis
            Self for method chaining.
        
        Examples
        --------
        >>> ba.export(preset="journal_submission")
        >>> ba.export(preset="thesis_appendix", output="thesis/appendix")
        """
        if preset:
            if preset not in EXPORT_PRESETS:
                available = list(EXPORT_PRESETS.keys())
                raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
            
            config = EXPORT_PRESETS[preset]
            formats = formats or config["formats"]
            level = config["level"]
            # Apply any preset options
            for k, v in config.get("options", {}).items():
                kwargs.setdefault(k, v)
        
        formats = formats or ["docx"]
        
        # Run the export
        self.generate_report(
            output=output,
            level=level,
            formats=formats,
            prepare=True,
        )
        
        return self
    
    @staticmethod
    def list_presets() -> List[str]:
        """List available export presets."""
        return list_export_presets(verbose=True)
    
    # =========================================================================
    # PLUGIN SYSTEM
    # =========================================================================
    
    def run_analysis(self, name: str, **kwargs) -> Any:
        """
        Run a registered custom analysis.
        
        Parameters
        ----------
        name : str
            Name of the registered analysis.
        **kwargs
            Arguments for the analysis.
        
        Returns
        -------
        Any
            Analysis result.
        
        Examples
        --------
        >>> @register_analysis("my_analysis")
        ... def my_analysis(ba, top_n=10):
        ...     return ba.df.head(top_n)
        >>> 
        >>> ba.run_analysis("my_analysis", top_n=5)
        """
        return run_custom_analysis(self, name, **kwargs)
    
    @staticmethod
    def list_custom_analyses() -> Dict[str, Callable]:
        """List all registered custom analyses."""
        return get_registered_analyses()
    
    # =========================================================================
    # DIVERSITY ANALYSIS
    # =========================================================================
    
    def compute_diversity(
        self,
        entities: List[str] = None,
        fetch_benchmark: bool = False,
        year_range: Tuple[int, int] = None,
    ):
        """
        Compute diversity indices (Shannon, Simpson, Gini) for bibliometric entities.
        
        Parameters
        ----------
        entities : list, optional
            Entities to analyze. If None, auto-detects available entities.
            Options: Sources, Authors, Countries, Affiliations, Author Keywords,
                    Index Keywords, Subject Areas, Document Types, SDGs, Years
        fetch_benchmark : bool, default False
            Fetch global diversity from OpenAlex for comparison.
        year_range : tuple, optional
            (start_year, end_year) for benchmark filtering.
            
        Returns
        -------
        DiversityAnalysisResult
            Results with indices for each entity.
            
        Examples
        --------
        >>> result = ba.compute_diversity()
        >>> print(result.summary())
        >>> 
        >>> # With benchmark
        >>> result = ba.compute_diversity(fetch_benchmark=True)
        >>> result.to_comparison_dataframe()
        >>> 
        >>> # Specific entities
        >>> result = ba.compute_diversity(entities=["Sources", "Countries", "SDGs"])
        """
        from biblium.diversity import compute_research_diversity_with_benchmark
        
        result = compute_research_diversity_with_benchmark(
            self.df,
            entities=entities,
            separator=self.default_separator,
            fetch_benchmark=fetch_benchmark,
            year_range=year_range,
        )
        
        # Store result
        self.diversity_result = result
        
        if self._verbose:
            print(f"✓ compute_diversity completed")
            print(f"  → self.diversity_result (DiversityAnalysisResult: {len(result.results)} entities)")
        
        return result
    
    def plot_diversity_radar(
        self,
        result=None,
        title: str = "Research Diversity Profile",
        show_benchmark: bool = True,
        filename: str = None,
    ):
        """
        Create radar plot of diversity indices across entities.
        
        Parameters
        ----------
        result : DiversityAnalysisResult, optional
            Result from compute_diversity(). If None, uses stored result.
        title : str
            Plot title.
        show_benchmark : bool
            Show benchmark lines if available.
        filename : str, optional
            Save plot to file.
            
        Returns
        -------
        tuple
            (fig, ax)
        """
        from biblium.diversity import plot_diversity_radar
        
        if result is None:
            if not hasattr(self, 'diversity_result'):
                raise ValueError("No diversity result. Run compute_diversity() first.")
            result = self.diversity_result
        
        return plot_diversity_radar(
            result,
            title=title,
            show_benchmark=show_benchmark,
            filename=filename,
        )
    
    def plot_diversity_bars(
        self,
        result=None,
        title: str = "Research Diversity Indices",
        show_benchmark: bool = True,
        filename: str = None,
    ):
        """
        Create grouped bar chart of diversity indices.
        
        Parameters
        ----------
        result : DiversityAnalysisResult, optional
            Result from compute_diversity(). If None, uses stored result.
        title : str
            Plot title.
        show_benchmark : bool
            Show benchmark markers if available.
        filename : str, optional
            Save plot to file.
            
        Returns
        -------
        tuple
            (fig, ax)
        """
        from biblium.diversity import plot_diversity_bars
        
        if result is None:
            if not hasattr(self, 'diversity_result'):
                raise ValueError("No diversity result. Run compute_diversity() first.")
            result = self.diversity_result
        
        return plot_diversity_bars(
            result,
            title=title,
            show_benchmark=show_benchmark,
            filename=filename,
        )
    
    # =========================================================================
    # TEMPORAL DIVERSITY ANALYSIS
    # =========================================================================
    
    def compute_temporal_diversity(
        self,
        entities: List[str] = None,
        year_col: str = "Year",
        min_items_per_year: int = 5,
    ):
        """
        Compute diversity indices over time.
        
        Parameters
        ----------
        entities : list, optional
            Entities to analyze. If None, auto-detects available.
        year_col : str
            Column containing publication year.
        min_items_per_year : int
            Minimum items required per year.
            
        Returns
        -------
        TemporalDiversityAnalysisResult
            Time series of diversity indices.
            
        Example
        -------
        >>> result = ba.compute_temporal_diversity()
        >>> result.to_dataframe("Sources")
        >>> ba.plot_temporal_diversity()
        """
        from biblium.diversity import compute_temporal_diversity_multi
        
        result = compute_temporal_diversity_multi(
            self.df,
            entities=entities,
            year_col=year_col,
            separator=self.default_separator,
            min_items_per_year=min_items_per_year,
        )
        
        self.temporal_diversity_result = result
        
        if self._verbose:
            print(f"✓ compute_temporal_diversity completed")
            print(f"  → self.temporal_diversity_result ({len(result.results)} entities, years {result.year_range[0]}-{result.year_range[1]})")
        
        return result
    
    def plot_temporal_diversity(
        self,
        result=None,
        entities: List[str] = None,
        index: str = "shannon",
        title: str = None,
        filename: str = None,
    ):
        """
        Plot diversity indices over time.
        
        Parameters
        ----------
        result : TemporalDiversityAnalysisResult, optional
            Result from compute_temporal_diversity().
        entities : list, optional
            Entities to plot.
        index : str
            Which index: 'shannon', 'simpson', 'gini', or 'all'
        title : str, optional
            Plot title.
        filename : str, optional
            Save plot to file.
            
        Returns
        -------
        tuple
            (fig, ax)
        """
        from biblium.diversity import plot_temporal_diversity
        
        if result is None:
            if not hasattr(self, 'temporal_diversity_result'):
                raise ValueError("No temporal diversity result. Run compute_temporal_diversity() first.")
            result = self.temporal_diversity_result
        
        return plot_temporal_diversity(
            result,
            entities=entities,
            index=index,
            title=title,
            filename=filename,
        )
    
    def plot_temporal_diversity_heatmap(
        self,
        result=None,
        index: str = "shannon",
        title: str = None,
        filename: str = None,
    ):
        """
        Create heatmap of diversity over time.
        
        Parameters
        ----------
        result : TemporalDiversityAnalysisResult, optional
            Result from compute_temporal_diversity().
        index : str
            Which index: 'shannon', 'simpson', 'gini'
        title : str, optional
            Plot title.
        filename : str, optional
            Save plot to file.
            
        Returns
        -------
        tuple
            (fig, ax)
        """
        from biblium.diversity import plot_temporal_diversity_heatmap
        
        if result is None:
            if not hasattr(self, 'temporal_diversity_result'):
                raise ValueError("No temporal diversity result. Run compute_temporal_diversity() first.")
            result = self.temporal_diversity_result
        
        return plot_temporal_diversity_heatmap(
            result,
            index=index,
            title=title,
            filename=filename,
        )
    
    # =========================================================================
    # CITATION PATTERN CLASSIFICATION
    # =========================================================================
    
    def analyze_citation_patterns(
        self,
        use_openalex: bool = True,
        max_papers: int = 500,
        min_age: int = 3,
    ):
        """
        Classify papers by citation trajectory pattern.
        
        Patterns include:
        - Evergreen: Sustained citations over many years
        - Flash-in-the-pan: Quick burst then decline
        - Delayed Recognition: Initially ignored, later discovered
        - Sleeping Beauty: Extreme delayed recognition
        - Normal: Typical citation decay
        
        Parameters
        ----------
        use_openalex : bool
            Fetch actual citation history from OpenAlex API.
            If False or unavailable, estimates from total citations.
        max_papers : int
            Maximum papers to fetch from OpenAlex (API limit consideration).
        min_age : int
            Minimum paper age (years) to analyze.
            
        Returns
        -------
        CitationPatternResult
            Classification results with pattern distribution.
            
        Example
        -------
        >>> result = ba.analyze_citation_patterns(use_openalex=True)
        >>> print(result.summary())
        >>> ba.plot_citation_patterns()
        
        Notes
        -----
        ⚠️ For accurate classification, OpenAlex data is strongly recommended.
        Estimation from total citations provides only rough approximations.
        """
        from biblium.citation_patterns import analyze_citation_patterns
        
        result = analyze_citation_patterns(
            self.df,
            use_openalex=use_openalex,
            max_papers=max_papers,
            min_age=min_age,
            verbose=self._verbose,
        )
        
        self.citation_pattern_result = result
        
        if self._verbose:
            print(f"✓ analyze_citation_patterns completed")
            print(f"  → self.citation_pattern_result ({result.n_analyzed} papers, source: {result.data_source})")
            if result.data_source == "estimated":
                print(f"  ⚠️ WARNING: Results are estimated. Use use_openalex=True for accuracy.")
        
        return result
    
    def plot_citation_patterns(
        self,
        result=None,
        plot_type: str = "distribution",
        pattern: str = None,
        metric: str = "half_life",
        title: str = None,
        filename: str = None,
    ):
        """
        Plot citation pattern analysis results.
        
        Parameters
        ----------
        result : CitationPatternResult, optional
            Result from analyze_citation_patterns().
        plot_type : str
            Type of plot:
            - 'distribution': Bar chart of pattern counts
            - 'by_year': Stacked bar by publication year
            - 'trajectories': Example trajectories for a pattern
            - 'metrics': Box plot comparing metrics
            - 'scatter': Scatter plot by metrics
        pattern : str, optional
            For 'trajectories' plot, which pattern to show examples for.
        metric : str
            For 'metrics' plot: 'half_life', 'years_to_peak', 'early_pct', 'decay_rate'
        title : str, optional
            Plot title.
        filename : str, optional
            Save plot to file.
            
        Returns
        -------
        tuple
            (fig, ax)
        """
        from biblium.citation_patterns import (
            plot_pattern_distribution,
            plot_pattern_by_year,
            plot_trajectory_examples,
            plot_metrics_comparison,
            plot_pattern_scatter,
        )
        
        if result is None:
            if not hasattr(self, 'citation_pattern_result'):
                raise ValueError("No citation pattern result. Run analyze_citation_patterns() first.")
            result = self.citation_pattern_result
        
        if plot_type == "distribution":
            return plot_pattern_distribution(result, title=title, filename=filename)
        elif plot_type == "by_year":
            return plot_pattern_by_year(result, title=title, filename=filename)
        elif plot_type == "trajectories":
            return plot_trajectory_examples(result, pattern=pattern, title=title, filename=filename)
        elif plot_type == "metrics":
            return plot_metrics_comparison(result, metric=metric, title=title, filename=filename)
        elif plot_type == "scatter":
            return plot_pattern_scatter(result, title=title, filename=filename)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
    
    # =========================================================================
    # CITATION VELOCITY & MOMENTUM
    # =========================================================================
    
    def analyze_citation_velocity(
        self,
        use_openalex: bool = False,
        max_papers: int = 500,
        recent_window: int = 3,
        min_age: int = 2,
    ):
        """
        Analyze citation velocity and momentum for papers.
        
        Velocity measures how fast papers are accumulating citations.
        Momentum measures whether that rate is accelerating or decelerating.
        
        Parameters
        ----------
        use_openalex : bool
            Fetch citation history from OpenAlex API if not in dataset.
            Disabled by default - only needed for non-OpenAlex datasets.
        max_papers : int
            Maximum papers to fetch from API.
        recent_window : int
            Years to consider as "recent" for velocity calculation.
        min_age : int
            Minimum paper age to analyze.
            
        Returns
        -------
        CitationVelocityResult
            Velocity and momentum analysis results.
            
        Example
        -------
        >>> result = ba.analyze_citation_velocity()
        >>> print(result.summary())
        >>> ba.plot_citation_velocity(plot_type="trend")
        
        Notes
        -----
        This analysis requires yearly citation counts. For datasets without
        counts_by_year, enable use_openalex=True to fetch from API.
        """
        from biblium.citation_velocity import analyze_citation_velocity
        
        result = analyze_citation_velocity(
            self.df,
            use_openalex=use_openalex,
            max_papers=max_papers,
            recent_window=recent_window,
            min_age=min_age,
            verbose=self._verbose,
        )
        
        self.citation_velocity_result = result
        
        if self._verbose:
            print(f"✓ analyze_citation_velocity completed")
            print(f"  → self.citation_velocity_result ({result.n_analyzed} documents)")
        
        return result
    
    def plot_citation_velocity(
        self,
        result=None,
        plot_type: str = "trend",
        n: int = 15,
        selection: str = "top_velocity",
        title: str = None,
        filename: str = None,
    ):
        """
        Plot citation velocity analysis results.
        
        Parameters
        ----------
        result : CitationVelocityResult, optional
            Result from analyze_citation_velocity().
        plot_type : str
            Type of plot:
            - 'trend': Distribution of velocity trends
            - 'velocity': Histogram of current velocities
            - 'vs_age': Velocity vs paper age scatter
            - 'momentum': Distribution of momentum values
            - 'top_accelerating': Top accelerating papers
            - 'top_velocity': Highest velocity papers
            - 'trajectories': Citation trajectories
        n : int
            Number of papers for top lists.
        selection : str
            For trajectories: 'top_velocity', 'top_accelerating', 'rising_stars'
        title : str, optional
            Plot title.
        filename : str, optional
            Save plot to file.
            
        Returns
        -------
        tuple
            (fig, ax)
        """
        from biblium.citation_velocity import (
            plot_trend_distribution,
            plot_velocity_distribution,
            plot_velocity_vs_age,
            plot_momentum_distribution,
            plot_top_accelerating,
            plot_top_velocity,
            plot_velocity_trajectories,
        )
        
        if result is None:
            if not hasattr(self, 'citation_velocity_result'):
                raise ValueError("No velocity result. Run analyze_citation_velocity() first.")
            result = self.citation_velocity_result
        
        if plot_type == "trend":
            return plot_trend_distribution(result, title=title, filename=filename)
        elif plot_type == "velocity":
            return plot_velocity_distribution(result, title=title, filename=filename)
        elif plot_type == "vs_age":
            return plot_velocity_vs_age(result, title=title, filename=filename)
        elif plot_type == "momentum":
            return plot_momentum_distribution(result, title=title, filename=filename)
        elif plot_type == "top_accelerating":
            return plot_top_accelerating(result, n=n, title=title, filename=filename)
        elif plot_type == "top_velocity":
            return plot_top_velocity(result, n=n, title=title, filename=filename)
        elif plot_type == "trajectories":
            return plot_velocity_trajectories(result, n_examples=n, selection=selection, 
                                              title=title, filename=filename)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
    
    # =========================================================================
    # CACHE MANAGEMENT
    # =========================================================================
    
    def clear_cache(self, method: str = None) -> int:
        """
        Clear cached results.
        
        Parameters
        ----------
        method : str, optional
            Clear only this method's cache. None clears all.
        
        Returns
        -------
        int
            Number of entries cleared.
        """
        if self._cache:
            return self._cache.clear(method)
        return 0
    
    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self._cache:
            return self._cache.stats()
        return {"enabled": False}

    def analyze_life_cycle(
        self,
        forecast_years: int = 50,
        target_years: Optional[List[int]] = None,
        plot: bool = True,
        save_path: Optional[str] = None,
        verbose: bool = True,
    ) -> "LifeCycleResult":
        """
        Analyze the life cycle of scientific production using logistic growth model.
        
        This implements BiblioShiny-style life cycle analysis, fitting a logistic
        (S-curve) model to cumulative publication data to estimate saturation,
        peak year, growth duration, and current phase.
        
        Parameters
        ----------
        forecast_years : int, default=50
            Number of years to forecast into the future.
        target_years : List[int], optional
            Specific years for projections (e.g., [2025, 2030]).
        plot : bool, default=True
            Generate and save the life cycle plot.
        save_path : str, optional
            Path to save the plot. Default: results/plots/life_cycle.png
        verbose : bool, default=True
            Print progress and results.
        
        Returns
        -------
        LifeCycleResult
            Comprehensive life cycle analysis with:
            - saturation_k: Carrying capacity
            - peak_year_tm: Inflection point year
            - peak_annual: Maximum annual rate
            - growth_duration_delta_t: Time from 10% to 90% of K
            - milestone_years: When 10%, 50%, 90%, 99% of K reached
            - current_phase: emergence/rapid_growth/maturity/saturation
            - projections: Forecasts for target years
            - forecast_df: Full DataFrame with fitted values
        
        Examples
        --------
        >>> ba = BiblioAnalysis("data.csv", db="scopus")
        >>> result = ba.analyze_life_cycle()
        >>> print(f"Saturation: {result.saturation_k:,.0f} publications")
        >>> print(f"Peak year: {result.peak_year_tm:.1f}")
        >>> print(f"Phase: {result.current_phase}")
        
        See Also
        --------
        biblium.addons.advanced_statistics.analyze_life_cycle : Underlying function
        biblium.addons.advanced_statistics.plot_life_cycle : Plot function
        """
        from biblium.addons.advanced_statistics import analyze_life_cycle, plot_life_cycle
        
        # Find year column
        year_col = "Year"
        for col in ["Year", "PY", "Publication Year"]:
            if col in self.df.columns:
                year_col = col
                break
        
        # Run analysis
        result = analyze_life_cycle(
            self.df,
            year_col=year_col,
            forecast_years=forecast_years,
            target_years=target_years,
            verbose=verbose,
        )
        
        # Store result as attribute
        self.life_cycle_result = result
        
        # Generate plot if requested
        if plot:
            if save_path is None and self.res_folder:
                save_path = os.path.join(self.res_folder, "plots", "life_cycle.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            if save_path:
                plot_life_cycle(result, save_path=save_path)
        
        return result

    def run_bibliometric_analysis(
        self, 
        top_n: int = 20, 
        mode: str = "full",
        verbose: bool = True,
        save_tables: bool = True,
        generate_plots: bool = True,
        skip_errors: bool = True,
    ) -> Dict[str, Any]:
        """
        Run comprehensive bibliometric analysis with tiered execution.
        
        Parameters
        ----------
        top_n : int, default 20
            Number of top items to include in counts and stats.
        mode : str, default "full"
            Execution mode: "basic", "extended", "full", or "full+".
        verbose : bool, default True
            Print progress messages.
        save_tables : bool, default True
            Save result tables to Excel files.
        generate_plots : bool, default True
            Generate and save plots.
        skip_errors : bool, default True
            Continue execution if individual functions fail.
        
        Returns
        -------
        dict
            Summary of execution with keys:
            - 'mode': Execution mode used
            - 'completed': List of successfully completed functions
            - 'errors': List of errors encountered
            - 'n_documents': Number of documents analyzed
        """
        errors = []
        completed = []
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"BIBLIOMETRIC ANALYSIS - Mode: {mode.upper()}")
            print(f"{'='*60}")
            print(f"Documents: {self.n}")
            print(f"Database: {self.db}")
            print()
        
        # =====================================================================
        # BASIC MODE: Core counts and main information
        # =====================================================================
        if mode in ["basic", "extended", "full", "full+"]:
            if verbose:
                print("▶ BASIC ANALYSIS")
                print("-" * 40)
            
            # Main information
            if _safe_execute(self.get_main_info, "Main Information", errors, verbose):
                completed.append("get_main_info")
            
            # Scientific production
            if _safe_execute(self.get_production, "Scientific Production", errors, verbose):
                completed.append("get_production")
            
            # Core counts
            counts = [
                ("count_sources", "Sources"),
                ("count_authors", "Authors"),
                ("count_author_keywords", "Author Keywords"),
                ("count_index_keywords", "Index Keywords"),
                ("count_document_types", "Document Types"),
                ("count_ca_countries", "CA Countries"),
                ("count_all_countries", "All Countries"),
                ("count_affiliations", "Affiliations"),
                ("count_references", "References"),
            ]
            
            for method_name, display_name in counts:
                method = getattr(self, method_name, None)
                if method:
                    if _safe_execute(method, display_name, errors, verbose, top_n=top_n):
                        completed.append(method_name)
            
            # Top cited documents
            if _safe_execute(self.get_top_cited_documents, "Top Cited Documents", errors, verbose, top_n=top_n):
                completed.append("get_top_cited_documents")
            
            if verbose:
                print()
        
        # =====================================================================
        # EXTENDED MODE: Performance stats, laws, and visualizations
        # =====================================================================
        if mode in ["extended", "full", "full+"]:
            if verbose:
                print("▶ EXTENDED ANALYSIS")
                print("-" * 40)
            
            # Performance statistics
            stats = [
                ("get_sources_stats", "Sources Stats"),
                ("get_authors_stats", "Authors Stats"),
                ("get_author_keywords_stats", "Keywords Stats"),
                ("get_all_countries_stats", "Countries Stats"),
                ("get_affiliations_stats", "Affiliations Stats"),
            ]
            
            for method_name, display_name in stats:
                method = getattr(self, method_name, None)
                if method:
                    if _safe_execute(method, display_name, errors, verbose, top_n=top_n):
                        completed.append(method_name)
            
            # Bibliometric laws
            laws = [
                ("compute_bradford_law", "Bradford's Law"),
                ("compute_lotka_law", "Lotka's Law"),
                ("compute_zipf_law", "Zipf's Law"),
            ]
            
            for method_name, display_name in laws:
                method = getattr(self, method_name, None)
                if method:
                    if _safe_execute(method, display_name, errors, verbose):
                        completed.append(method_name)
            
            # Generate basic plots
            if generate_plots:
                plots = [
                    ("sources_bar", "Sources Bar Plot"),
                    ("authors_bar", "Authors Bar Plot"),
                    ("keywords_bar", "Keywords Bar Plot"),
                    ("countries_bar", "Countries Bar Plot"),
                    ("production_line", "Production Line Plot"),
                    ("citations_histogram", "Citations Histogram"),
                    ("keywords_wordcloud", "Keywords Wordcloud"),
                ]
                
                for method_name, display_name in plots:
                    try:
                        if verbose:
                            print(f"  → {display_name}...", end=" ", flush=True)
                        plot_method = getattr(self.plot, method_name, None)
                        if plot_method:
                            plot_method(top_n=min(top_n, 20))
                            completed.append(f"plot.{method_name}")
                            if verbose:
                                print("✓")
                    except Exception as e:
                        errors.append(f"plot.{method_name}: {type(e).__name__}")
                        if verbose:
                            print(f"✗ ({type(e).__name__})")
            
            if verbose:
                print()
        
        # =====================================================================
        # FULL MODE: Advanced analytics
        # =====================================================================
        if mode in ["full", "full+"]:
            if verbose:
                print("▶ FULL ANALYSIS")
                print("-" * 40)
            
            # Co-occurrence analysis
            cooccurrence = [
                ("get_author_keyword_cooccurrence", "Keyword Co-occurrence"),
                ("get_coauthorship", "Co-authorship"),
            ]
            
            for method_name, display_name in cooccurrence:
                method = getattr(self, method_name, None)
                if method:
                    if _safe_execute(method, display_name, errors, verbose, top_n=min(top_n, 50)):
                        completed.append(method_name)
            
            # Country collaboration
            if _safe_execute(self.get_country_collaboration, "Country Collaboration", errors, verbose):
                completed.append("get_country_collaboration")
            
            # Reference spectrogram
            if _safe_execute(self.compute_reference_spectrogram, "Reference Spectrogram", errors, verbose):
                completed.append("compute_reference_spectrogram")
            
            # Network exports
            networks = [
                ("build_coauthorship_network", "Coauthorship Network"),
                ("build_keyword_cooccurrence_network", "Keyword Network"),
            ]
            
            for method_name, display_name in networks:
                method = getattr(self, method_name, None)
                if method:
                    if _safe_execute(method, display_name, errors, verbose):
                        completed.append(method_name)
            
            # Advanced plots
            if generate_plots:
                adv_plots = [
                    ("sources_treemap", "Sources Treemap"),
                    ("cooccurrence_heatmap", "Cooccurrence Heatmap"),
                    ("keyword_network", "Keyword Network"),
                ]
                
                for method_name, display_name in adv_plots:
                    try:
                        if verbose:
                            print(f"  → {display_name}...", end=" ", flush=True)
                        plot_method = getattr(self.plot, method_name, None)
                        if plot_method:
                            plot_method()
                            completed.append(f"plot.{method_name}")
                            if verbose:
                                print("✓")
                    except Exception as e:
                        errors.append(f"plot.{method_name}: {type(e).__name__}")
                        if verbose:
                            print(f"✗ ({type(e).__name__})")
            
            if verbose:
                print()
        
        # =====================================================================
        # FULL+ MODE: Experimental and LLM-based features
        # =====================================================================
        if mode == "full+":
            if verbose:
                print("▶ ADVANCED ANALYSIS (FULL+)")
                print("-" * 40)
            
            # Sleeping beauty analysis
            if _safe_execute(self.run_sb_analysis, "Sleeping Beauty Analysis", errors, verbose):
                completed.append("run_sb_analysis")
            
            # Clustering
            try:
                if verbose:
                    print(f"  → Document Clustering...", end=" ", flush=True)
                self.cluster_documents(n_clusters=5)
                completed.append("cluster_documents")
                if verbose:
                    print("✓")
            except Exception as e:
                errors.append(f"cluster_documents: {type(e).__name__}")
                if verbose:
                    print(f"✗ ({type(e).__name__})")
            
            # Topic modeling
            try:
                if verbose:
                    print(f"  → Topic Modeling...", end=" ", flush=True)
                self.get_topics(n_topics=5)
                completed.append("get_topics")
                if verbose:
                    print("✓")
            except Exception as e:
                errors.append(f"get_topics: {type(e).__name__}")
                if verbose:
                    print(f"✗ ({type(e).__name__})")
            
            # Sentiment analysis
            if _safe_execute(self.get_sentiment, "Sentiment Analysis", errors, verbose):
                completed.append("get_sentiment")
            
            # Distribution fitting
            try:
                if verbose:
                    print(f"  → Citation Distribution Fitting...", end=" ", flush=True)
                self.fit_citation_distributions()
                completed.append("fit_citation_distributions")
                if verbose:
                    print("✓")
            except Exception as e:
                errors.append(f"fit_citation_distributions: {type(e).__name__}")
                if verbose:
                    print(f"✗ ({type(e).__name__})")
            
            if verbose:
                print()
        
        # =====================================================================
        # Summary
        # =====================================================================
        if verbose:
            print(f"{'='*60}")
            print(f"ANALYSIS COMPLETE")
            print(f"{'='*60}")
            print(f"Completed: {len(completed)} functions")
            print(f"Errors: {len(errors)} functions")
            
            if errors:
                print(f"\nErrors encountered:")
                for err in errors[:10]:
                    print(f"  • {err}")
                if len(errors) > 10:
                    print(f"  ... and {len(errors) - 10} more")
            print()
        
        return {
            "mode": mode,
            "completed": completed,
            "errors": errors,
            "n_documents": self.n,
            "n_completed": len(completed),
            "n_errors": len(errors),
        }
    
    # Report generation methods
    def save_report_to_excel(self, output_path: str = os.path.join("results", "reports", "bibliometric_report.xlsx"), **kwargs):
        """Generate Excel report from template."""
        return reportbib.save_excel_report_from_template(self, output_path=output_path, results_base=self.res_folder, **kwargs)
        
    def save_report_to_word(self, output_path: str = os.path.join("results", "reports", "bibliometric_report.docx"), use_simple_mode: bool = False, **kwargs):
        """
        Generate Word report from template.
        
        Parameters
        ----------
        output_path : str
            Output file path
        use_simple_mode : bool, default False
            If True, bypasses template system and directly adds all found tables/plots.
            Use this if template-based reports are empty.
        **kwargs
            Additional arguments passed to save_word_report_from_template
        """
        return reportbib.save_word_report_from_template(self, output_path=output_path, results_base=self.res_folder, use_simple_mode=use_simple_mode, **kwargs)
        
    def save_report_to_pptx(self, output_path: str = os.path.join("results", "reports", "bibliometric_report.pptx"), **kwargs):
        """Generate PowerPoint report from template."""
        return reportbib.save_powerpoint_report_from_template(self, output_path=output_path, results_base=self.res_folder, **kwargs)
        
    def save_report_to_tex(self, output_path: str = os.path.join("results", "reports", "bibliometric_report.tex"), **kwargs):
        """Generate LaTeX report from template."""
        return reportbib.save_tex_report_from_template(self, output_path=output_path, results_base=self.res_folder, **kwargs)
    
    def generate_all_reports(
        self,
        output_dir: str = "results/reports",
        base_name: str = "bibliometric_report",
        formats: List[str] = None,
        template_sheet: str = "basic",
        **kwargs
    ) -> Dict[str, str]:
        """
        Generate reports in multiple formats.
        
        Parameters
        ----------
        output_dir : str
            Output directory for reports.
        base_name : str
            Base filename for reports.
        formats : list
            List of formats: ["xlsx", "docx", "pptx", "tex"]
        template_sheet : str, default "basic"
            Template sheet to use: "basic", "standard", "extended", "full", or "all".
        **kwargs
            Additional arguments passed to report functions.
        
        Returns
        -------
        dict
            Mapping of format to output path.
        """
        if formats is None:
            formats = ["xlsx", "docx", "pptx", "tex"]
        
        results = {}
        
        for fmt in formats:
            try:
                if fmt == "xlsx":
                    path = self.save_report_to_excel(
                        output_path=os.path.join(output_dir, f"{base_name}.xlsx"),
                        template_sheet=template_sheet,
                        **kwargs
                    )
                elif fmt == "docx":
                    path = self.save_report_to_word(
                        output_path=os.path.join(output_dir, f"{base_name}.docx"),
                        template_sheet=template_sheet,
                        **kwargs
                    )
                elif fmt == "pptx":
                    path = self.save_report_to_pptx(
                        output_path=os.path.join(output_dir, f"{base_name}.pptx"),
                        template_sheet=template_sheet,
                        **kwargs
                    )
                elif fmt == "tex":
                    path = self.save_report_to_tex(
                        output_path=os.path.join(output_dir, f"{base_name}.tex"),
                        template_sheet=template_sheet,
                        **kwargs
                    )
                else:
                    continue
                
                results[fmt] = str(path)
                print(f"✓ Generated {fmt.upper()}: {path}")
            except Exception as e:
                results[fmt] = f"Error: {e}"
                print(f"✗ Failed {fmt.upper()}: {e}")
        
        return results

    def prepare_for_report(
        self,
        level: str = "basic",
        verbose: bool = True,
        generate_plots: bool = True,
    ) -> Dict[str, Any]:
        """
        Automatically run all analyses needed for the selected report level.
        
        This method examines the report template and runs the necessary
        analysis methods to populate all required data attributes.
        
        Parameters
        ----------
        level : str, default "basic"
            Report level: "basic", "standard", "extended", "full", or "all".
            - basic: Core counts and main information
            - standard: Adds performance stats and visualizations
            - extended: Adds production over time and co-occurrence
            - full/all: All available analyses
        verbose : bool, default True
            Print progress messages.
        generate_plots : bool, default True
            Generate plot files for the report.
            
        Returns
        -------
        dict
            Summary with 'completed', 'errors', and 'coverage' keys.
            
        Examples
        --------
        >>> ba = BiblioAnalysis("data.csv", db="scopus")
        >>> ba.prepare_for_report(level="basic")
        >>> ba.generate_all_reports()
        """
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            
        errors = []
        completed = []
        
        # Define which methods are needed for each level
        # These methods populate the data attributes expected by the report template
        level_methods = {
            "basic": [
                # Main information - generates descriptives_df, performances_df, 
                # production_df, time_series_stats_df
                ("get_main_info", {}),
                # Entity counts - generates *_counts_df attributes
                ("count_sources", {}),
                ("count_authors", {}),
                ("count_author_keywords", {}),
                ("count_document_types", {}),
                ("count_ca_countries", {}),
            ],
            "standard": [
                # Includes all from basic plus:
                # Performance stats - generates *_stats_df attributes
                ("get_sources_stats", {}),
                ("get_authors_stats", {}),
                ("get_author_keywords_stats", {}),
                ("count_all_countries", {}),
                ("get_ca_countries_stats", {}),
                ("get_all_countries_stats", {}),
            ],
            "extended": [
                # Includes all from standard plus:
                ("count_references", {}),
                ("count_affiliations", {}),
                ("count_index_keywords", {}),
                # Production over time - generates *_production_over_time_df
                ("get_production_over_time", {"var": "sources", "top_n": 10}),
                ("get_production_over_time", {"var": "authors", "top_n": 10}),
                ("get_production_over_time", {"var": "author_keywords", "top_n": 10}),
                ("get_production_over_time", {"var": "all_countries", "top_n": 10}),
                # Co-occurrence
                ("get_author_keywords_cooccurrence", {}),
            ],
            "full": [
                # Includes all from extended plus:
                # Bibliometric laws
                ("compute_lotka", {}),
                ("compute_bradford", {}),
                ("compute_zipf", {}),
                # Citation analysis
                ("get_top_cited_documents", {"mode": "both"}),
                # Advanced
                ("get_sleeping_beauties", {}),
            ],
        }
        
        # Group analysis methods (only applicable if groups are set up)
        group_methods = [
            # Group overview
            ("get_group_intersections", {}),
            ("get_main_info", {}),  # Uses group-aware version if BiblioGroupAnalysis
            ("get_scientific_production", {}),
            ("compare_continuous_vars", {}),
            # Group citations
            ("get_group_top_cited_documents", {"mode": "both", "top_n": 10}),
            # Group associations (these generate *_associations and *_contingency)
            ("associate_author_keywords", {"min_freq": 5}),
            ("associate_index_keywords", {"min_freq": 5}),
            ("associate_sources", {"min_freq": 3}),
            ("associate_authors", {"min_freq": 3}),
            ("associate_countries", {"min_freq": 3}),
            ("associate_references", {"min_freq": 3}),
            ("associate_affiliations", {"min_freq": 3}),
            # Group counts
            ("group_count_sources", {}),
            ("group_count_author_keywords", {}),
            ("group_count_index_keywords", {}),
            ("group_count_authors", {}),
            ("group_count_all_countries", {}),
            ("group_count_ca_countries", {}),
            ("group_count_affiliations", {}),
            ("group_count_references", {}),
            # Group stats
            ("get_group_sources_stats", {}),
            ("get_group_author_keywords_stats", {}),
            ("get_group_authors_stats", {}),
            ("get_group_all_countries_stats", {}),
        ]
        
        # Build the list of methods to run
        methods_to_run = []
        levels_order = ["basic", "standard", "extended", "full"]
        
        target_level = level.lower()
        
        # Handle special levels
        if target_level == "groups":
            # Only run group methods (requires BiblioGroupAnalysis)
            if hasattr(self, "group_matrix") and self.group_matrix is not None:
                methods_to_run = group_methods
            else:
                if verbose:
                    print("Warning: 'groups' level requires BiblioGroupAnalysis with groups set up.")
                    print("         Use BiblioGroupAnalysis and call build_groups() first.")
        elif target_level == "all":
            # Run full + groups
            for lvl in levels_order:
                if lvl in level_methods:
                    methods_to_run.extend(level_methods[lvl])
            # Add group methods if groups are available
            if hasattr(self, "group_matrix") and self.group_matrix is not None:
                methods_to_run.extend(group_methods)
        else:
            # Standard progression: basic -> standard -> extended -> full
            for lvl in levels_order:
                if lvl in level_methods:
                    methods_to_run.extend(level_methods[lvl])
                if lvl == target_level:
                    break
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"PREPARING DATA FOR '{level.upper()}' REPORT")
            print(f"{'='*60}")
            print(f"Running {len(methods_to_run)} analysis methods...")
            print()
        
        # Create iterator with or without tqdm
        if use_tqdm and verbose:
            iterator = tqdm(methods_to_run, desc="Preparing report data")
        else:
            iterator = methods_to_run
        
        for method_name, kwargs in iterator:
            try:
                if hasattr(self, method_name):
                    method = getattr(self, method_name)
                    if callable(method):
                        if verbose and not use_tqdm:
                            print(f"  → {method_name}...", end=" ", flush=True)
                        method(**kwargs)
                        completed.append(method_name)
                        if verbose and not use_tqdm:
                            print("✓")
            except Exception as e:
                error_msg = f"{method_name}: {type(e).__name__}: {str(e)[:50]}"
                errors.append(error_msg)
                if verbose and not use_tqdm:
                    print(f"✗ ({type(e).__name__})")
        
        # Generate plots if requested
        if generate_plots:
            plot_methods = [
                # Basic plots
                ("plot_scientific_production", {}),
            ]
            
            if target_level in ["standard", "extended", "full"]:
                plot_methods.extend([
                    ("plot_sources_bar", {"top_n": 10}),
                    ("plot_authors_bar", {"top_n": 10}),
                    ("plot_keywords_bar", {"top_n": 10}),
                    ("plot_countries_bar", {"top_n": 10}),
                    ("plot_document_types_donut", {}),
                    ("plot_citations_histogram", {}),
                ])
            
            if target_level in ["extended", "full"]:
                plot_methods.extend([
                    ("plot_keywords_wordcloud", {}),
                    ("plot_sources_bubble", {}),
                    ("plot_authors_bubble", {}),
                ])
            
            if target_level == "full":
                plot_methods.extend([
                    ("plot_lotka", {}),
                    ("plot_bradford", {}),
                    ("plot_zipf", {}),
                ])
            
            if verbose:
                print(f"\nGenerating {len(plot_methods)} plots...")
            
            for method_name, kwargs in plot_methods:
                try:
                    if hasattr(self, method_name):
                        method = getattr(self, method_name)
                        if callable(method):
                            if verbose:
                                print(f"  → {method_name}...", end=" ", flush=True)
                            method(**kwargs)
                            completed.append(method_name)
                            if verbose:
                                print("✓")
                except Exception as e:
                    errors.append(f"{method_name}: {type(e).__name__}")
                    if verbose:
                        print(f"✗")
        
        # Check coverage
        status = reportbib.check_report_data_availability(
            self, 
            template_sheet=level if level != "all" else "full",
            verbose=False
        )
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"PREPARATION COMPLETE")
            print(f"{'='*60}")
            print(f"Completed: {len(completed)} methods")
            print(f"Errors: {len(errors)}")
            print(f"Report coverage: {status['summary']['coverage_pct']}%")
            if errors:
                print(f"\nErrors encountered:")
                for err in errors[:5]:
                    print(f"  • {err}")
            print()
        
        return {
            "completed": completed,
            "errors": errors,
            "coverage": status["summary"],
        }

    def preview_report(
        self,
        level: str = "basic",
    ) -> str:
        """
        Generate a text preview of what will be in the report.
        
        Parameters
        ----------
        level : str, default "basic"
            Report level to preview.
            
        Returns
        -------
        str
            Text summary of report contents.
        """
        status = reportbib.check_report_data_availability(
            self,
            template_sheet=level if level != "all" else "full",
            verbose=False
        )
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"REPORT PREVIEW - Level: {level.upper()}")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Coverage: {status['summary']['coverage_pct']}%")
        lines.append(f"Available items: {status['summary']['available_items']}/{status['summary']['total_items']}")
        lines.append("")
        
        if status['available']:
            lines.append("TABLES TO BE INCLUDED:")
            lines.append("-" * 40)
            for item in status['available']:
                if item['type'] == 'table':
                    lines.append(f"  ✓ {item['name']} ({item['rows']} rows)")
            
            lines.append("")
            lines.append("PLOTS TO BE INCLUDED:")
            lines.append("-" * 40)
            for item in status['available']:
                if item['type'] == 'plot':
                    lines.append(f"  ✓ {item['name']}")
        
        if status['missing_data']:
            lines.append("")
            lines.append("MISSING TABLES (will be skipped):")
            lines.append("-" * 40)
            for item in status['missing_data'][:10]:
                lines.append(f"  ✗ {item['name']} - {item['reason']}")
            if len(status['missing_data']) > 10:
                lines.append(f"  ... and {len(status['missing_data']) - 10} more")
        
        if status['missing_plots']:
            lines.append("")
            lines.append("MISSING PLOTS (will be skipped):")
            lines.append("-" * 40)
            for item in status['missing_plots'][:10]:
                lines.append(f"  ✗ {item['name']}")
            if len(status['missing_plots']) > 10:
                lines.append(f"  ... and {len(status['missing_plots']) - 10} more")
        
        lines.append("")
        lines.append("=" * 60)
        
        preview_text = "\n".join(lines)
        print(preview_text)
        return preview_text

    def check_report_data(self, level: str = "basic", verbose: bool = True) -> Dict[str, Any]:
        """
        Check which data is available for the report.
        
        Convenience wrapper around check_report_data_availability.
        
        Parameters
        ----------
        level : str, default "basic"
            Report level to check.
        verbose : bool, default True
            Print detailed information.
            
        Returns
        -------
        dict
            Data availability status.
        """
        return reportbib.check_report_data_availability(
            self,
            template_sheet=level if level != "all" else "full",
            verbose=verbose
        )


class BiblioGroupAnalysis(BiblioGroupClassifier, BiblioGroupPlot, BiblioGroup):
    """
    Class for group-based bibliometric analysis.
    
    Analyzes overlapping bibliographic subgroups and computes
    statistical associations using measures like Jaccard index,
    Yule's Q, and Fisher's exact test.
    
    Parameters
    ----------
    verbose : bool, default True
        Print information about computed attributes.
        Shows what results are stored and where to access them.
    
    Examples
    --------
    >>> bg = BiblioGroupAnalysis("data.csv", db="scopus", group_desc=dg, verbose=True)
    >>> bg.associate_countries()
    → Stored: self.countries_associations (DataFrame: 50 rows × 12 cols)
    → Stored: self.countries_contingency (DataFrame: 6 rows × 50 cols)
    """
    
    def __init__(self, *args, verbose: bool = True, **kwargs):
        """
        Initialize BiblioGroupAnalysis with verbose tracking.
        
        Parameters
        ----------
        *args : 
            Positional arguments passed to parent class.
        verbose : bool, default True
            Print information about computed attributes.
        **kwargs :
            Keyword arguments passed to parent class.
        """
        # Initialize tracking before parent init
        object.__setattr__(self, '_verbose', verbose)
        object.__setattr__(self, '_tracked_attrs', {})
        object.__setattr__(self, '_initializing', True)
        
        # Call parent init
        super().__init__(*args, **kwargs)
        
        # Finished initializing - now track changes
        object.__setattr__(self, '_initializing', False)
        if verbose:
            self._snapshot_attrs()
    
    def _snapshot_attrs(self):
        """Take a snapshot of current attributes for comparison."""
        self._tracked_attrs = {
            k: id(v) for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
    
    @staticmethod
    def _describe_attribute(value: Any) -> str:
        """Create a short description of an attribute value."""
        if value is None:
            return "(None)"
        
        type_name = type(value).__name__
        
        # NetworkX graph check
        if hasattr(value, 'number_of_nodes') and hasattr(value, 'number_of_edges'):
            try:
                n_nodes = value.number_of_nodes()
                n_edges = value.number_of_edges()
                return f"(Graph: {n_nodes} nodes, {n_edges} edges)"
            except:
                pass
        
        if hasattr(value, 'shape'):  # DataFrame, ndarray
            shape = value.shape
            if len(shape) == 2:
                return f"(DataFrame: {shape[0]} rows × {shape[1]} cols)"
            else:
                return f"(array: shape {shape})"
        elif isinstance(value, dict):
            return f"(dict: {len(value)} items)"
        elif isinstance(value, (list, tuple)):
            return f"({type_name}: {len(value)} items)"
        elif isinstance(value, str):
            if len(value) > 50:
                return f'(str: "{value[:50]}...")'
            return f'(str: "{value}")'
        elif isinstance(value, (int, float)):
            return f"({type_name}: {value})"
        else:
            return f"({type_name})"
    
    def __setattr__(self, name: str, value: Any):
        """Track attribute changes for verbose reporting."""
        super().__setattr__(name, value)
        
        # Don't track private attributes or during initialization
        if name.startswith('_'):
            return
        if getattr(self, '_initializing', True):
            return
        
        # Report if verbose and this is a result-type attribute
        if getattr(self, '_verbose', False):
            # Only report DataFrames, dicts with data, and other result types
            if hasattr(value, 'shape') or (isinstance(value, dict) and len(value) > 0):
                attr_desc = self._describe_attribute(value)
                print(f"  → Stored: self.{name} {attr_desc}")
    
    def compute_all(
        self,
        verbose: bool = True,
        skip_errors: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute all group analysis metrics.
        
        Parameters
        ----------
        verbose : bool, default True
            Print progress messages.
        skip_errors : bool, default True
            Continue execution if individual functions fail.
        
        Returns
        -------
        dict
            Summary of execution with completed functions and errors.
        """
        errors = []
        completed = []
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"GROUP ANALYSIS")
            print(f"{'='*60}")
            print(f"Number of groups: {len(self.groups) if hasattr(self, 'groups') else 'Not built'}")
            print()
        
        # Build groups if not already built
        if not hasattr(self, 'groups') or not self.groups:
            try:
                if verbose:
                    print("  → Building groups...", end=" ", flush=True)
                self.build_groups()
                completed.append("build_groups")
                if verbose:
                    print("✓")
            except Exception as e:
                errors.append(f"build_groups: {type(e).__name__}: {str(e)[:50]}")
                if verbose:
                    print(f"✗ ({type(e).__name__})")
                if not skip_errors:
                    raise
        
        # Count keywords per group
        try:
            if verbose:
                print("  → Counting keywords...", end=" ", flush=True)
            self.count_keywords()
            completed.append("count_keywords")
            if verbose:
                print("✓")
        except Exception as e:
            errors.append(f"count_keywords: {type(e).__name__}")
            if verbose:
                print(f"✗ ({type(e).__name__})")
        
        # Count countries per group
        try:
            if verbose:
                print("  → Counting countries...", end=" ", flush=True)
            self.count_countries()
            completed.append("count_countries")
            if verbose:
                print("✓")
        except Exception as e:
            errors.append(f"count_countries: {type(e).__name__}")
            if verbose:
                print(f"✗ ({type(e).__name__})")
        
        # Compute associations if classifier methods available
        if hasattr(self, 'compute_associations'):
            try:
                if verbose:
                    print("  → Computing associations...", end=" ", flush=True)
                self.compute_associations()
                completed.append("compute_associations")
                if verbose:
                    print("✓")
            except Exception as e:
                errors.append(f"compute_associations: {type(e).__name__}")
                if verbose:
                    print(f"✗ ({type(e).__name__})")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"GROUP ANALYSIS COMPLETE")
            print(f"Completed: {len(completed)} functions")
            print(f"Errors: {len(errors)} functions")
            print()
        
        return {
            "completed": completed,
            "errors": errors,
            "n_groups": len(self.groups) if hasattr(self, 'groups') else 0,
        }
    
    # =========================================================================
    # INTROSPECTION AND SUMMARY
    # =========================================================================
    
    def __repr__(self) -> str:
        """Nice string representation."""
        lines = [
            f"BiblioGroupAnalysis(",
            f"  documents={self.n:,},",
            f"  database='{self.db}',",
        ]
        
        if hasattr(self, 'groups') and self.groups:
            lines.append(f"  groups={len(self.groups)},")
            group_names = list(self.groups.keys())[:3]
            lines.append(f"  group_names={group_names}{'...' if len(self.groups) > 3 else ''},")
        else:
            lines.append(f"  groups=not_built,")
        
        lines.append(")")
        return "\n".join(lines)
    
    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        n_groups = len(self.groups) if hasattr(self, 'groups') and self.groups else 0
        groups_status = f"{n_groups} groups" if n_groups > 0 else "Not built"
        
        html = f'''
        <div style="font-family: sans-serif; padding: 15px; background: #f0f8ff; border-radius: 8px; border: 1px solid #b0d4f1;">
            <h3 style="margin-top: 0; color: #2c5282;">📊 BiblioGroupAnalysis</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #bee3f8;"><strong>Documents</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #bee3f8;">{self.n:,}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #bee3f8;"><strong>Database</strong></td>
                    <td style="padding: 8px; border-bottom: 1px solid #bee3f8;">{self.db}</td>
                </tr>
                <tr>
                    <td style="padding: 8px;"><strong>Groups</strong></td>
                    <td style="padding: 8px;">{groups_status}</td>
                </tr>
            </table>
            <p style="margin-bottom: 0; color: #4a5568; font-size: 0.9em;">
                Use <code>.summary()</code> for details or <code>.what_can_i_do()</code> to see available analyses.
            </p>
        </div>
        '''
        return html
    
    def what_can_i_do(self) -> List[str]:
        """
        List available group analyses based on the current state.
        
        Returns
        -------
        list of str
            Method names that can be called.
        
        Examples
        --------
        >>> bg = BiblioGroupAnalysis(ba, group_var="SDG")
        >>> bg.what_can_i_do()
        """
        return what_can_i_do(self, verbose=True)
    
    def summary(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Get a summary of the group analysis state.
        
        Parameters
        ----------
        verbose : bool, default True
            Print the summary.
        
        Returns
        -------
        dict
            Summary with group information.
        """
        summary_data = {
            "n_documents": self.n,
            "database": self.db,
            "groups_built": hasattr(self, 'groups') and bool(self.groups),
            "n_groups": len(self.groups) if hasattr(self, 'groups') and self.groups else 0,
            "group_names": list(self.groups.keys()) if hasattr(self, 'groups') and self.groups else [],
            "group_sizes": {},
            "analyses_completed": [],
        }
        
        # Get group sizes
        if hasattr(self, 'groups') and self.groups:
            for name, docs in self.groups.items():
                summary_data["group_sizes"][name] = len(docs) if hasattr(docs, '__len__') else 0
        
        # Check completed analyses
        if hasattr(self, 'associations_df') and self.associations_df is not None:
            summary_data["analyses_completed"].append("associations")
        if hasattr(self, 'group_keywords_df') and self.group_keywords_df is not None:
            summary_data["analyses_completed"].append("keywords")
        if hasattr(self, 'group_countries_df') and self.group_countries_df is not None:
            summary_data["analyses_completed"].append("countries")
        
        if verbose:
            print("\n" + "=" * 60)
            print("GROUP ANALYSIS SUMMARY")
            print("=" * 60)
            print(f"Documents: {summary_data['n_documents']:,}")
            print(f"Database: {summary_data['database']}")
            print(f"Groups built: {'Yes' if summary_data['groups_built'] else 'No'}")
            
            if summary_data['groups_built']:
                print(f"Number of groups: {summary_data['n_groups']}")
                print("\nGroup sizes:")
                for name, size in list(summary_data['group_sizes'].items())[:10]:
                    print(f"  • {name}: {size} documents")
                if len(summary_data['group_sizes']) > 10:
                    print(f"  ... and {len(summary_data['group_sizes']) - 10} more groups")
            
            if summary_data['analyses_completed']:
                print(f"\nAnalyses completed: {', '.join(summary_data['analyses_completed'])}")
            
            print("=" * 60)
        
        return summary_data
    
    # Report generation methods for group analysis
    def save_group_report_to_excel(
        self,
        output_path: str = os.path.join("results", "reports", "group_report.xlsx"),
        **kwargs
    ):
        """Generate Excel report for group analysis."""
        return reportbib.save_group_excel_report(self, output_path=output_path, **kwargs)
    
    def save_group_report_to_word(
        self,
        output_path: str = os.path.join("results", "reports", "group_report.docx"),
        **kwargs
    ):
        """Generate Word report for group analysis."""
        return reportbib.save_group_word_report(self, output_path=output_path, **kwargs)
    
    def save_group_report_to_pptx(
        self,
        output_path: str = os.path.join("results", "reports", "group_report.pptx"),
        **kwargs
    ):
        """Generate PowerPoint report for group analysis."""
        return reportbib.save_group_pptx_report(self, output_path=output_path, **kwargs)
    
    def save_group_report_to_tex(
        self,
        output_path: str = os.path.join("results", "reports", "group_report.tex"),
        **kwargs
    ):
        """Generate LaTeX report for group analysis."""
        return reportbib.save_group_tex_report(self, output_path=output_path, **kwargs)
    
    # =========================================================================
    # GROUP DIVERSITY ANALYSIS
    # =========================================================================
    
    def compute_group_diversity(
        self,
        entities: List[str] = None,
    ):
        """
        Compute diversity indices for each group.
        
        Parameters
        ----------
        entities : list, optional
            Entities to analyze. If None, auto-detects available.
            
        Returns
        -------
        GroupDiversityAnalysisResult
            Diversity comparison across groups.
            
        Example
        -------
        >>> bg = BiblioGroupAnalysis(...)
        >>> bg.build_groups()
        >>> result = bg.compute_group_diversity()
        >>> print(result.summary())
        >>> bg.plot_group_diversity()
        """
        from biblium.diversity import compute_group_diversity_from_bib_group
        
        if not hasattr(self, 'groups') or not self.groups:
            raise ValueError("Groups not built. Call build_groups() first.")
        
        result = compute_group_diversity_from_bib_group(
            self,
            entities=entities,
            separator=getattr(self, 'default_separator', '; '),
        )
        
        self.group_diversity_result = result
        
        if self._verbose:
            print(f"✓ compute_group_diversity completed")
            print(f"  → self.group_diversity_result ({len(result.group_results)} groups, {len(result.entities)} entities)")
        
        return result
    
    def plot_group_diversity(
        self,
        result=None,
        index: str = "shannon",
        plot_type: str = "bar",
        entity: str = None,
        title: str = None,
        filename: str = None,
    ):
        """
        Plot group diversity comparison.
        
        Parameters
        ----------
        result : GroupDiversityAnalysisResult, optional
            Result from compute_group_diversity().
        index : str
            Which index: 'shannon', 'simpson', 'gini'
        plot_type : str
            'bar', 'radar', or 'heatmap'
        entity : str, optional
            For radar plot, specific entity to compare.
        title : str, optional
            Plot title.
        filename : str, optional
            Save plot to file.
            
        Returns
        -------
        tuple
            (fig, ax)
        """
        from biblium.diversity import (
            plot_group_diversity_comparison,
            plot_group_diversity_radar,
            plot_group_diversity_heatmap,
        )
        
        if result is None:
            if not hasattr(self, 'group_diversity_result'):
                raise ValueError("No group diversity result. Run compute_group_diversity() first.")
            result = self.group_diversity_result
        
        if plot_type == "bar":
            return plot_group_diversity_comparison(
                result, index=index, title=title, filename=filename
            )
        elif plot_type == "radar":
            return plot_group_diversity_radar(
                result, entity=entity, title=title, filename=filename
            )
        elif plot_type == "heatmap":
            return plot_group_diversity_heatmap(
                result, index=index, title=title, filename=filename
            )
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}. Use 'bar', 'radar', or 'heatmap'")