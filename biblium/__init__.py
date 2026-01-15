"""Biblium - Comprehensive Bibliometric Analysis Library"""

__version__ = "2.12.0"

# Logging setup (import first for other modules to use)
from biblium import logging_config
from biblium.logging_config import get_logger, set_level

# Base module with shared initialization
from biblium import base
from biblium.base import BiblioBase, SEPARATOR_MAP

from biblium import utilsbib
from biblium import readbib
from biblium import plotbib
from biblium import reportbib
from biblium.reportbib import check_report_data_availability, validate_template, generate_plots_from_template, debug_report_generation
from biblium import sdg_identifier

# Configuration system
from biblium import config
from biblium.config import (
    BibliumConfig,
    PlotConfig,
    ReportConfig as ReportConfigSettings,
    get_config,
    set_config,
    get_plot_config,
    set_plot_config,
    get_report_config,
    set_report_config,
    apply_plot_style as apply_global_plot_style,
    plot_config,
    report_config,
)

# Enhancements module
from biblium import enhancements
from biblium.enhancements import (
    # Error handling
    BibliumError,
    ColumnNotFoundError,
    InsufficientDataError,
    # Caching
    AnalysisCache,
    CacheConfig,
    # Validation
    ValidationResult,
    validate_bibliometric_data,
    # Lazy evaluation
    LazyResult,
    # Summary
    DatasetSummary,
    get_available_analyses,
    get_available_group_analyses,
    # Export presets
    EXPORT_PRESETS,
    get_export_presets,
    list_export_presets,
    # Plugin system
    register_analysis,
    get_registered_analyses,
    # Config file
    ProjectConfig,
    load_project_config,
)

# Deduplication
from biblium import dedup
from biblium.dedup import deduplicate, detect_duplicates

# Reference Benchmark / Representation
from biblium import representation
from biblium.representation import (
    compute_relative_representation,
    plot_relative_representation,
    plot_distribution_comparison,
    chi_square_test,
    ChiSquareResult,
    ReferenceBenchmarkResult,
    get_reference_warning,
    OPENALEX_WARNING,
    SUPPORTED_REFERENCES,
    fetch_openalex_yearly_counts,
    fetch_openalex_sdg_counts,
    fetch_openalex_country_counts,
)

# Research Diversity Indices
from biblium import diversity
from biblium.diversity import (
    # Data structures
    DiversityResult,
    DiversityAnalysisResult,
    TemporalDiversityResult,
    TemporalDiversityAnalysisResult,
    GroupDiversityResult,
    GroupDiversityAnalysisResult,
    # Core functions
    compute_shannon_index,
    compute_simpson_index,
    compute_gini_index,
    compute_diversity_indices,
    compute_research_diversity,
    compute_research_diversity_with_benchmark,
    fetch_openalex_diversity_benchmark,
    # Temporal diversity
    compute_temporal_diversity,
    compute_temporal_diversity_multi,
    plot_temporal_diversity,
    plot_temporal_diversity_heatmap,
    # Group diversity
    compute_group_diversity,
    compute_group_diversity_from_bib_group,
    plot_group_diversity_comparison,
    plot_group_diversity_radar,
    plot_group_diversity_heatmap,
    # Visualization
    plot_diversity_radar,
    plot_diversity_bars,
    interpret_diversity,
    list_available_entities,
)

# Citation Pattern Classification
try:
    from biblium import citation_patterns
    from biblium.citation_patterns import (
        CitationPattern,
        CitationTrajectory,
        CitationPatternResult,
        analyze_citation_patterns,
        fetch_citation_history_openalex,
        plot_pattern_distribution,
        plot_pattern_by_year,
        plot_trajectory_examples,
        plot_metrics_comparison,
        plot_pattern_scatter,
    )
    HAS_CITATION_PATTERNS = True
except ImportError:
    citation_patterns = None
    HAS_CITATION_PATTERNS = False

# Citation Velocity & Momentum
try:
    from biblium import citation_velocity
    from biblium.citation_velocity import (
        VelocityTrend,
        CitationVelocityMetrics,
        CitationVelocityResult,
        analyze_citation_velocity,
        plot_trend_distribution,
        plot_velocity_distribution,
        plot_velocity_vs_age,
        plot_momentum_distribution,
        plot_top_accelerating,
        plot_top_velocity,
        plot_velocity_trajectories,
    )
    HAS_CITATION_VELOCITY = True
except ImportError:
    citation_velocity = None
    HAS_CITATION_VELOCITY = False

# Reference Diversity
try:
    from biblium import reference_diversity
    from biblium.reference_diversity import (
        DiversityLevel,
        ReferenceDiversityMetrics,
        ReferenceDiversityResult,
        analyze_reference_diversity,
        compute_shannon_entropy,
        compute_simpson_diversity,
        compute_rao_stirling,
        plot_diversity_distribution,
        plot_source_diversity,
        plot_field_diversity,
        plot_reference_age_distribution,
        plot_diversity_by_year,
        get_top_diverse_papers,
        add_diversity_to_dataframe,
    )
    HAS_REFERENCE_DIVERSITY = True
except ImportError:
    reference_diversity = None
    HAS_REFERENCE_DIVERSITY = False

# Concept Extraction
try:
    from biblium import concept_extraction
    from biblium.concept_extraction import (
        ConceptInfo,
        ConceptExtractionResult,
        analyze_concepts,
        extract_ngrams_tfidf,
        compute_cooccurrence_matrix,
        compute_temporal_trends,
        plot_top_concepts,
        plot_concept_cooccurrence,
        plot_temporal_trends,
    )
    HAS_CONCEPT_EXTRACTION = True
except ImportError:
    concept_extraction = None
    HAS_CONCEPT_EXTRACTION = False

# Compare Means (Statistical Analysis)
try:
    from biblium import compare_means as compare_means_module
    from biblium.compare_means import (
        CompareMeansResult,
        GroupDescriptives,
        StatisticalTest,
        PostHocResult,
        compare_means,
        compute_descriptives,
        test_normality,
        test_homogeneity,
        independent_t_test,
        mann_whitney_u,
        one_way_anova,
        kruskal_wallis,
        tukey_hsd,
        get_numeric_columns,
        get_categorical_columns,
    )
    HAS_COMPARE_MEANS = True
except ImportError:
    compare_means_module = None
    HAS_COMPARE_MEANS = False

# Crosstabs (Contingency Table Analysis)
try:
    from biblium import crosstabs as crosstabs_module
    from biblium.crosstabs import (
        CrosstabResult,
        ChiSquaredResult,
        FisherResult,
        EffectSize,
        compute_crosstab,
        format_crosstab_table,
    )
    HAS_CROSSTABS = True
except ImportError:
    crosstabs_module = None
    HAS_CROSSTABS = False

# Correlation Analysis
try:
    from biblium import correlation as correlation_module
    from biblium.correlation import (
        CorrelationResult,
        CorrelationPair,
        compute_correlation,
        get_correlation_interpretation,
        plot_correlation_matrix,
        plot_scatter_matrix,
    )
    HAS_CORRELATION = True
except ImportError:
    correlation_module = None
    HAS_CORRELATION = False

# Main Path Analysis
try:
    from biblium import main_path as main_path_module
    from biblium.main_path import (
        MainPathResult,
        compute_main_path_analysis,
        compute_traversal_weights,
        find_global_main_path,
        find_forward_main_path,
        find_backward_main_path,
        find_key_routes,
        plot_main_path,
    )
    HAS_MAIN_PATH = True
except ImportError:
    main_path_module = None
    HAS_MAIN_PATH = False

# OpenAlex API
try:
    from biblium import openalex_api
    from biblium.openalex_api import (
        OpenAlexClient,
        OpenAlexWork,
        OpenAlexAuthor,
        OpenAlexInstitution,
        search_openalex,
        download_openalex_dataset,
    )
    HAS_OPENALEX = True
except ImportError:
    openalex_api = None
    OpenAlexClient = None
    HAS_OPENALEX = False

# Color scheme module
try:
    from biblium import colors
    from biblium.colors import (
        PRIMARY_PALETTE, CATEGORICAL_PALETTE, SDG_COLORS,
        get_color, get_colors, get_sdg_color, apply_plot_style
    )
except ImportError:
    colors = None

# Batch processing module
try:
    from biblium import batch
    from biblium.batch import (
        chunk_dataframe, batch_apply, batch_aggregate,
        batch_count_column, batch_text_process,
        read_large_csv, BatchProcessor,
        suggest_chunk_size, estimate_dataframe_memory,
    )
except ImportError:
    batch = None

# Report generation module
try:
    from biblium import reports
    from biblium.reports import (
        ReportGenerator,
        ReportConfig,
        ReportLevel,
        ReportTemplate,
    )
    HAS_REPORTS = True
except ImportError:
    reports = None
    ReportGenerator = None
    ReportConfig = None
    HAS_REPORTS = False

# Dashboard module
try:
    from biblium import dashboard
    from biblium.dashboard import Dashboard, DashboardConfig
    HAS_DASHBOARD = True
except ImportError:
    dashboard = None
    Dashboard = None
    DashboardConfig = None
    HAS_DASHBOARD = False

# LLM utilities module
try:
    from biblium import llm_utils
    from biblium.llm_utils import (
        # Config
        LLMConfig,
        get_default_config as get_llm_config,
        set_default_config as set_llm_config,
        # Cache
        LLMCache,
        get_cache as get_llm_cache,
        # Main functions
        invoke_llm,
        invoke_llm_async,
        invoke_llm_batch,
        invoke_llm_batch_async,
        # Bibliometric functions
        llm_summarize_abstracts,
        llm_describe_table,
        llm_extract_keywords,
        llm_classify_methodology,
        llm_identify_research_gaps,
        llm_batch_classify,
        # Convenience
        quick_summarize,
        quick_describe,
        # Providers
        get_provider as get_llm_provider,
    )
    HAS_LLM = True
except ImportError:
    llm_utils = None
    HAS_LLM = False

from biblium.bibstats import BiblioStats
from biblium.bibgroup import BiblioGroup
from biblium.bibplot import BiblioPlot, BiblioGroupPlot
from biblium.bibclass import BiblioGroupClassifier
from biblium.biblium_main import BiblioAnalysis, BiblioGroupAnalysis

# Tkinter GUI (optional)
try:
    from biblium.bibtkinter_app import BibliumApp, main as run_gui
    HAS_GUI = True
except ImportError:
    HAS_GUI = False
    BibliumApp = None
    run_gui = None

__all__ = [
    # Main classes
    "BiblioAnalysis", "BiblioGroupAnalysis",
    "BiblioStats", "BiblioGroup", "BiblioPlot", "BiblioGroupPlot", "BiblioGroupClassifier",
    # Modules
    "utilsbib", "readbib", "plotbib", "reportbib", "sdg_identifier", 
    "colors", "batch", "config", "logging_config", "dedup", "openalex_api",
    "reports", "dashboard", "enhancements", "representation", "diversity",
    # Configuration
    "BibliumConfig", "PlotConfig", "ReportConfigSettings",
    "get_config", "set_config", "get_plot_config", "set_plot_config",
    "get_report_config", "set_report_config", "apply_global_plot_style",
    "plot_config", "report_config",
    # Report utilities
    "check_report_data_availability", "validate_template",
    # Enhancements - Errors
    "BibliumError", "ColumnNotFoundError", "InsufficientDataError",
    # Enhancements - Caching
    "AnalysisCache", "CacheConfig",
    # Enhancements - Validation
    "ValidationResult", "validate_bibliometric_data",
    # Enhancements - Lazy
    "LazyResult",
    # Enhancements - Summary
    "DatasetSummary",
    "get_available_analyses",
    "get_available_group_analyses",
    # Enhancements - Presets
    "EXPORT_PRESETS", "get_export_presets", "list_export_presets",
    # Enhancements - Plugin
    "register_analysis", "get_registered_analyses",
    # Enhancements - Config
    "ProjectConfig", "load_project_config",
    # Logging
    "get_logger", "set_level", 
    # Deduplication
    "deduplicate", "detect_duplicates",
    # Reference Benchmark
    "compute_relative_representation", "plot_relative_representation",
    "plot_distribution_comparison", "chi_square_test",
    "ChiSquareResult", "ReferenceBenchmarkResult",
    "get_reference_warning", "OPENALEX_WARNING", "SUPPORTED_REFERENCES",
    "fetch_openalex_yearly_counts", "fetch_openalex_sdg_counts", "fetch_openalex_country_counts",
    # Diversity Indices
    "DiversityResult", "DiversityAnalysisResult",
    "TemporalDiversityResult", "TemporalDiversityAnalysisResult",
    "GroupDiversityResult", "GroupDiversityAnalysisResult",
    "compute_shannon_index", "compute_simpson_index", "compute_gini_index",
    "compute_diversity_indices", "compute_research_diversity",
    "compute_research_diversity_with_benchmark", "fetch_openalex_diversity_benchmark",
    "compute_temporal_diversity", "compute_temporal_diversity_multi",
    "plot_temporal_diversity", "plot_temporal_diversity_heatmap",
    "compute_group_diversity", "compute_group_diversity_from_bib_group",
    "plot_group_diversity_comparison", "plot_group_diversity_radar", "plot_group_diversity_heatmap",
    "plot_diversity_radar", "plot_diversity_bars", "interpret_diversity", "list_available_entities",
    # Citation Patterns
    "CitationPattern", "CitationTrajectory", "CitationPatternResult",
    "analyze_citation_patterns", "fetch_citation_history_openalex",
    "plot_pattern_distribution", "plot_pattern_by_year", "plot_trajectory_examples",
    "plot_metrics_comparison", "plot_pattern_scatter", "HAS_CITATION_PATTERNS",
    # Citation Velocity
    "VelocityTrend", "CitationVelocityMetrics", "CitationVelocityResult",
    "analyze_citation_velocity", "plot_trend_distribution", "plot_velocity_distribution",
    "plot_velocity_vs_age", "plot_momentum_distribution", "plot_top_accelerating",
    "plot_top_velocity", "plot_velocity_trajectories", "HAS_CITATION_VELOCITY",
    # OpenAlex
    "OpenAlexClient", "OpenAlexWork", "OpenAlexAuthor", "OpenAlexInstitution",
    "search_openalex", "download_openalex_dataset", "HAS_OPENALEX",
    # Reports module
    "ReportGenerator", "ReportConfig", "ReportLevel", "ReportTemplate", "HAS_REPORTS",
    # Dashboard
    "Dashboard", "DashboardConfig", "HAS_DASHBOARD",
    # GUI
    "BibliumApp", "run_gui", "HAS_GUI",
    # Colors
    "PRIMARY_PALETTE", "CATEGORICAL_PALETTE", "SDG_COLORS",
    "get_color", "get_colors", "get_sdg_color", "apply_plot_style",
    # Batch processing
    "chunk_dataframe", "batch_apply", "batch_aggregate",
    "batch_count_column", "batch_text_process", "BatchProcessor",
]
