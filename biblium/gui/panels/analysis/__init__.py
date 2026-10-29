# -*- coding: utf-8 -*-
"""Analysis panels package."""

from biblium.gui.panels.analysis.counts_panel import CountsPanel
from biblium.gui.panels.analysis.overview_panel import OverviewPanel
from biblium.gui.panels.analysis.trends_panel import TrendsPanel
from biblium.gui.panels.analysis.laws_panel import LawsPanel
from biblium.gui.panels.analysis.statistics_panel import StatisticsPanel
from biblium.gui.panels.analysis.top_cited_panel import TopCitedPanel
from biblium.gui.panels.analysis.life_cycle_panel import LifeCyclePanel
from biblium.gui.panels.analysis.kfields_panel import KFieldsPanel
from biblium.gui.panels.analysis.production_panel import ProductionOverTimePanel
from biblium.gui.panels.analysis.trend_topics_panel import TrendTopicsPanel
from biblium.gui.panels.analysis.top_items_timeline_panel import TopItemsTimelinePanel
from biblium.gui.panels.analysis.relationship_panel import RelationshipPanel
from biblium.gui.panels.analysis.factorial_panel import FactorialPanel
from biblium.gui.panels.analysis.distribution_panel import DistributionPanel
from biblium.gui.panels.analysis.collaboration_panel import CollaborationPanel
from biblium.gui.panels.analysis.altmetrics_panel import AltmetricsPanel
from biblium.gui.panels.analysis.novelty_panel import NoveltyPanel
from biblium.gui.panels.analysis.sentiment_panel import SentimentPanel
from biblium.gui.panels.analysis.race_bar_panel import RaceBarPanel
from biblium.gui.panels.analysis.sleeping_beauty_panel import SleepingBeautyPanel
from biblium.gui.panels.analysis.clustering_panel import DocumentClusteringPanel
from biblium.gui.panels.analysis.entity_clustering_panel import EntityClusteringPanel
from biblium.gui.panels.analysis.repository_links_panel import RepositoryLinksPanel
from biblium.gui.panels.analysis.topic_modeling_panel import TopicModelingPanel
from biblium.gui.panels.analysis.dynamic_topics_panel import DynamicTopicsPanel
from biblium.gui.panels.analysis.reference_benchmark_panel import ReferenceBenchmarkPanel
from biblium.gui.panels.analysis.diversity_panel import DiversityIndicesPanel
from biblium.gui.panels.analysis.temporal_diversity_panel import TemporalDiversityPanel
from biblium.gui.panels.analysis.citation_patterns_panel import CitationPatternsPanel
from biblium.gui.panels.analysis.citation_velocity_panel import CitationVelocityPanel

__all__ = [
    "CountsPanel", 
    "OverviewPanel", 
    "TrendsPanel", 
    "LawsPanel", 
    "StatisticsPanel", 
    "TopCitedPanel", 
    "LifeCyclePanel", 
    "KFieldsPanel", 
    "ProductionOverTimePanel", 
    "TrendTopicsPanel", 
    "TopItemsTimelinePanel",
    "RelationshipPanel",
    "FactorialPanel",
    "DistributionPanel",
    "CollaborationPanel",
    "AltmetricsPanel",
    "NoveltyPanel",
    "SentimentPanel",
    "RaceBarPanel",
    "SleepingBeautyPanel",
    "DocumentClusteringPanel",
    "EntityClusteringPanel",
    "RepositoryLinksPanel",
    "TopicModelingPanel",
    "DynamicTopicsPanel",
    "ReferenceBenchmarkPanel",
    "DiversityIndicesPanel",
    "TemporalDiversityPanel",
    "CitationPatternsPanel",
    "CitationVelocityPanel",
]
