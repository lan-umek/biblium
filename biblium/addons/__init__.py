# -*- coding: utf-8 -*-
"""
Biblium Addons - Extended functionality for bibliometric analysis

Available addons:
- comparative_analysis: Dataset comparison, benchmarking, radar charts
- impact_metrics: Disruption Index, novelty measures, citation patterns
- text_mining_nlp: NER, claim extraction, sentiment analysis
- conceptual_drift: Topic evolution, concept emergence/decline
- dynamic_topic_models: Temporal topic modeling
- temporal_networks: Network evolution, community dynamics
- predictive_analytics: Citation prediction, trend forecasting
- open_science: OA analysis, data sharing metrics
- altmetrics: Social media metrics, attention scores
- advanced_statistics: Statistical tests, regression, meta-analysis
- geographic_analysis: Choropleth maps, geocoding
- sdg_drift: SDG temporal analysis
- sdg_networks: SDG co-occurrence networks
- research_gaps: Research gap identification
- methodology_classifier: Research method classification

Base classes for creating new addons:
- AddonResult: Base class for analysis results
- AddonAnalyzer: Base class for analyzers
- VisualizationMixin: Add plotting capabilities
- TemporalMixin: Add time-series analysis
- NetworkMixin: Add network analysis

Usage:
    from biblium.addons import impact_metrics
    results = impact_metrics.compute_disruption_index(df)
    
    from biblium.addons import geographic_analysis
    geo = geographic_analysis.run_geographic_analysis(df)
    
    # Creating new addons:
    from biblium.addons.base import AddonResult, AddonAnalyzer
    
    class MyResult(AddonResult):
        ...
    
    class MyAnalyzer(AddonAnalyzer):
        ...
"""

__version__ = "1.1.0"

# Base classes for addon development
from biblium.addons.base import (
    AddonResult,
    AddonAnalyzer,
    VisualizationMixin,
    TemporalMixin,
    NetworkMixin,
    register_addon,
    get_registered_addons,
    get_addon,
    list_addons,
    validate_columns,
    find_column,
)

# Direct imports - each addon is loaded only when accessed
# Users should import specific modules they need

__all__ = [
    # Base classes
    "AddonResult",
    "AddonAnalyzer",
    "VisualizationMixin",
    "TemporalMixin",
    "NetworkMixin",
    "register_addon",
    "get_registered_addons",
    "get_addon",
    "list_addons",
    "validate_columns",
    "find_column",
    # Addon modules
    "core_utils",
    "comparative_analysis",
    "impact_metrics", 
    "text_mining_nlp",
    "conceptual_drift",
    "dynamic_topic_models",
    "temporal_networks",
    "predictive_analytics",
    "open_science",
    "altmetrics",
    "advanced_statistics",
    "geographic_analysis",
    "sdg_drift",
    "sdg_networks",
    "research_gaps",
    "methodology_classifier",
]
