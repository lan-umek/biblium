# -*- coding: utf-8 -*-
"""
Biblium GUI Application
=======================
Main application window and entry point.

@author: Lan.Umek
@version: 2.5.0
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import threading
import traceback
from typing import Optional

from biblium.gui.config import (
    LAYOUT, FONTS, get_theme, 
    DATABASE_OPTIONS, PREPROCESS_LEVELS, SHORTCUTS,
    AppState,
)
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.core.state import StateManager
from biblium.gui.core.threading import TaskManager, BackgroundTask
from biblium.gui.components.sidebar import Sidebar
from biblium.gui.components.toolbar import Toolbar
from biblium.gui.components.statusbar import StatusBar
from biblium.gui.components.workspace import Workspace

# Import panels
from biblium.gui.panels.base import BasePanel

# Try to import biblium
try:
    from biblium import BiblioAnalysis, BiblioPlot, BiblioGroupAnalysis
    HAS_BIBLIUM = True
except ImportError:
    HAS_BIBLIUM = False
    BiblioAnalysis = None
    BiblioPlot = None


class BibliumApp(tk.Tk):
    """
    Main Biblium GUI Application.
    
    A modern desktop application for bibliometric analysis.
    
    Usage:
        app = BibliumApp()
        app.mainloop()
    """
    
    def __init__(self, theme: str = "light"):
        super().__init__()
        
        self.theme_name = theme
        self.theme = get_theme(theme)
        
        # State
        self.state = StateManager()
        self.task_manager = TaskManager()
        
        # Biblium instances
        self.bib: Optional[BiblioAnalysis] = None
        self.bib_group = None  # BiblioGroupAnalysis instance
        self.original_df = None
        self.dataset_path = None
        
        # Setup window
        self._setup_window()
        self._setup_styles()
        self._create_layout()
        self._setup_menu()
        self._setup_shortcuts()
        self._register_panels()
        
        # Subscribe to events
        self._subscribe_events()
        
        # Load saved AI settings
        self._load_ai_settings()
    
    def _setup_window(self):
        """Configure the main window."""
        self.title("Biblium - Bibliometric Analysis")
        self.geometry(f"{LAYOUT.default_window_width}x{LAYOUT.default_window_height}")
        self.minsize(LAYOUT.min_window_width, LAYOUT.min_window_height)
        self.configure(bg=self.theme["bg_secondary"])
        
        # Icon (if available)
        try:
            # self.iconbitmap("icon.ico")
            pass
        except:
            pass
        
        # Handle close
        self.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _load_ai_settings(self):
        """Load saved AI settings and apply to DataTable."""
        try:
            from biblium.gui.widgets.tables import DataTable
            
            settings = self.state.settings
            api_key = settings.ai_api_key
            provider = settings.ai_provider
            model = settings.ai_model
            custom_prompt = settings.ai_custom_prompt
            
            DataTable.set_llm_settings(
                enabled=bool(api_key),
                provider=provider,
                model=model,
                api_key=api_key,
                custom_prompt=custom_prompt,
            )
        except Exception as e:
            print(f"Warning: Could not load AI settings: {e}")
    
    def _setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Notebook
        style.configure(
            "TNotebook",
            background=self.theme["bg_secondary"],
            borderwidth=0,
        )
        style.configure(
            "TNotebook.Tab",
            padding=[16, 8],
            font=FONTS.get_font("body"),
            background=self.theme["bg_secondary"],
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.theme["bg_card"])],
            foreground=[("selected", self.theme["accent_primary"])],
        )
        
        # Treeview
        style.configure(
            "Treeview",
            font=FONTS.get_font("body"),
            rowheight=26,
            background=self.theme["bg_card"],
            fieldbackground=self.theme["bg_card"],
        )
        style.configure(
            "Treeview.Heading",
            font=FONTS.get_font("body"),
            background=self.theme["table_header"],
        )
        style.map(
            "Treeview",
            background=[("selected", self.theme["table_row_selected"])],
        )
        
        # Scrollbars
        style.configure(
            "Vertical.TScrollbar",
            background=self.theme["bg_secondary"],
            troughcolor=self.theme["bg_card"],
        )
        style.configure(
            "Horizontal.TScrollbar",
            background=self.theme["bg_secondary"],
            troughcolor=self.theme["bg_card"],
        )
        
        # PanedWindow
        style.configure(
            "TPanedwindow",
            background=self.theme["bg_secondary"],
        )
    
    def _create_layout(self):
        """Create the main application layout."""
        # Main container
        self.main_container = tk.Frame(self, bg=self.theme["bg_secondary"])
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Sidebar
        self.sidebar = Sidebar(
            self.main_container,
            theme=self.theme_name,
            on_navigate=self._on_navigate,
        )
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        # Right side container
        right_container = tk.Frame(self.main_container, bg=self.theme["bg_secondary"])
        right_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Toolbar
        self.toolbar = Toolbar(
            right_container,
            theme=self.theme_name,
            on_open=self._open_file,
            on_save=self._save_results,
            on_export=self._export,
            on_run=self._run_current,
        )
        self.toolbar.pack(fill=tk.X)
        
        # Separator
        ttk.Separator(right_container, orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Workspace
        self.workspace = Workspace(right_container, theme=self.theme_name)
        self.workspace._state = self.state  # Share state with workspace
        self.workspace.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.statusbar = StatusBar(self, theme=self.theme_name)
        self.statusbar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def _setup_menu(self):
        """Create the menu bar."""
        menubar = tk.Menu(self)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Dataset...", command=self._open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Results", command=self._save_results, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Export...", command=self._export, accelerator="Ctrl+E")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Preferences...", command=self._show_settings)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Toggle Sidebar", command=self._toggle_sidebar)
        view_menu.add_separator()
        view_menu.add_command(label="Light Theme", command=lambda: self._set_theme("light"))
        view_menu.add_command(label="Dark Theme", command=lambda: self._set_theme("dark"))
        menubar.add_cascade(label="View", menu=view_menu)
        
        # Analysis menu - mirrors sidebar structure
        analysis_menu = tk.Menu(menubar, tearoff=0)
        analysis_menu.add_command(label="Run Current Analysis", command=self._run_current, accelerator="F5")
        analysis_menu.add_command(label="Refresh Panel", command=self._refresh_current, accelerator="Shift+F5")
        analysis_menu.add_separator()
        
        # ANALYSIS submenu
        analysis_sub = tk.Menu(analysis_menu, tearoff=0)
        analysis_sub.add_command(label="Overview", command=lambda: self._on_navigate("overview"))
        analysis_sub.add_command(label="Counts", command=lambda: self._on_navigate("counts"))
        analysis_sub.add_command(label="Statistics", command=lambda: self._on_navigate("statistics"))
        analysis_sub.add_command(label="Laws", command=lambda: self._on_navigate("laws"))
        analysis_sub.add_command(label="Top Cited", command=lambda: self._on_navigate("top_cited"))
        analysis_sub.add_command(label="Citation Distribution", command=lambda: self._on_navigate("citation_distribution"))
        analysis_sub.add_command(label="Collaboration Metrics", command=lambda: self._on_navigate("collaboration"))
        analysis_sub.add_command(label="Altmetrics", command=lambda: self._on_navigate("altmetrics"))
        analysis_sub.add_command(label="Novelty Analysis", command=lambda: self._on_navigate("novelty"))
        analysis_sub.add_command(label="Sentiment Analysis", command=lambda: self._on_navigate("sentiment"))
        analysis_sub.add_command(label="K-Fields Plot", command=lambda: self._on_navigate("kfields"))
        analysis_sub.add_command(label="Relationships", command=lambda: self._on_navigate("relationships"))
        analysis_menu.add_cascade(label="Analysis", menu=analysis_sub)
        
        # TEMPORAL ANALYSIS submenu
        temporal_menu = tk.Menu(analysis_menu, tearoff=0)
        temporal_menu.add_command(label="Scientific Production", command=lambda: self._on_navigate("trends"))
        temporal_menu.add_command(label="Entity Over Time", command=lambda: self._on_navigate("production"))
        temporal_menu.add_command(label="Top Items Timeline", command=lambda: self._on_navigate("top_items_timeline"))
        temporal_menu.add_command(label="Trend Topics", command=lambda: self._on_navigate("trend_topics"))
        temporal_menu.add_command(label="Growth Models", command=lambda: self._on_navigate("growth_models"))
        temporal_menu.add_command(label="Life Cycle", command=lambda: self._on_navigate("life_cycle"))
        analysis_menu.add_cascade(label="Temporal Analysis", menu=temporal_menu)
        
        # VISUALIZATION submenu
        viz_menu = tk.Menu(analysis_menu, tearoff=0)
        viz_menu.add_command(label="Word Cloud", command=lambda: self._on_navigate("wordcloud"))
        viz_menu.add_command(label="Treemap", command=lambda: self._on_navigate("treemap"))
        viz_menu.add_command(label="Distribution", command=lambda: self._on_navigate("distribution"))
        viz_menu.add_command(label="Geographic", command=lambda: self._on_navigate("geographic"))
        viz_menu.add_command(label="Race Bar Animation", command=lambda: self._on_navigate("race_bar"))
        analysis_menu.add_cascade(label="Visualization", menu=viz_menu)
        
        # FACTORIAL submenu
        factorial_menu = tk.Menu(analysis_menu, tearoff=0)
        factorial_menu.add_command(label="Factorial Analysis", command=lambda: self._on_navigate("factorial"))
        analysis_menu.add_cascade(label="Factorial", menu=factorial_menu)
        
        # NETWORKS submenu
        networks_menu = tk.Menu(analysis_menu, tearoff=0)
        networks_menu.add_command(label="Co-occurrence Networks", command=lambda: self._on_navigate("network"))
        networks_menu.add_command(label="Citation Network", command=lambda: self._on_navigate("citation_network"))
        networks_menu.add_command(label="Thematic Map", command=lambda: self._on_navigate("thematic_map"))
        networks_menu.add_command(label="Historiograph", command=lambda: self._on_navigate("historiograph"))
        analysis_menu.add_cascade(label="Networks", menu=networks_menu)
        
        # MAPPING submenu
        mapping_menu = tk.Menu(analysis_menu, tearoff=0)
        mapping_menu.add_command(label="Topic Modeling", command=lambda: self._on_navigate("topic_modeling"))
        mapping_menu.add_command(label="Dynamic Topics", command=lambda: self._on_navigate("dynamic_topics"))
        analysis_menu.add_cascade(label="Mapping", menu=mapping_menu)
        
        # CLUSTERING submenu
        clustering_menu = tk.Menu(analysis_menu, tearoff=0)
        clustering_menu.add_command(label="Document Clustering", command=lambda: self._on_navigate("document_clustering"))
        clustering_menu.add_command(label="Entity Clustering", command=lambda: self._on_navigate("entity_clustering"))
        analysis_menu.add_cascade(label="Clustering", menu=clustering_menu)
        
        # ADVANCED submenu
        advanced_menu = tk.Menu(analysis_menu, tearoff=0)
        advanced_menu.add_command(label="Concept Builder", command=lambda: self._on_navigate("concept_builder"))
        advanced_menu.add_command(label="PA Concepts", command=lambda: self._on_navigate("pa_concepts"))
        advanced_menu.add_command(label="My Concepts", command=lambda: self._on_navigate("my_concepts"))
        advanced_menu.add_command(label="SDG Identifier", command=lambda: self._on_navigate("sdg"))
        advanced_menu.add_command(label="Sleeping Beauty", command=lambda: self._on_navigate("sleeping_beauty"))
        advanced_menu.add_command(label="Disruption Index", command=lambda: self._on_navigate("disruption_index"))
        advanced_menu.add_command(label="Repository Links", command=lambda: self._on_navigate("repository_links"))
        advanced_menu.add_command(label="Research Fronts", command=lambda: self._on_navigate("fronts"))
        analysis_menu.add_cascade(label="Advanced", menu=advanced_menu)
        
        # GROUPS submenu
        groups_menu = tk.Menu(analysis_menu, tearoff=0)
        groups_menu.add_command(label="Setup Groups", command=lambda: self._on_navigate("group_setup"))
        groups_menu.add_command(label="Group Counts", command=lambda: self._on_navigate("group_counts"))
        groups_menu.add_command(label="Group Statistics", command=lambda: self._on_navigate("group_stats"))
        groups_menu.add_command(label="Compare Groups", command=lambda: self._on_navigate("group_compare"))
        groups_menu.add_command(label="Intersections", command=lambda: self._on_navigate("group_intersections"))
        groups_menu.add_command(label="Associations", command=lambda: self._on_navigate("group_associations"))
        groups_menu.add_separator()
        groups_menu.add_command(label="Classification (ML)", command=lambda: self._on_navigate("group_classification"))
        groups_menu.add_command(label="Logistic Regression", command=lambda: self._on_navigate("group_logistic"))
        analysis_menu.add_cascade(label="Groups", menu=groups_menu)
        
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Generate Report...", command=self._generate_report, accelerator="Ctrl+G")
        analysis_menu.add_command(label="Custom Report Builder...", command=lambda: self._on_navigate("custom_report"))
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Documentation", command=self._show_docs)
        help_menu.add_command(label="Keyboard Shortcuts", command=self._show_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="About Biblium", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.config(menu=menubar)
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        self.bind("<Control-o>", lambda e: self._open_file())
        self.bind("<Control-s>", lambda e: self._save_results())
        self.bind("<Control-e>", lambda e: self._export())
        self.bind("<Control-r>", lambda e: self._run_current())
        self.bind("<Control-g>", lambda e: self._generate_report())
        self.bind("<F1>", lambda e: self._show_docs())
        self.bind("<F5>", lambda e: self._run_current())  # F5 runs current analysis
        self.bind("<Shift-F5>", lambda e: self._refresh_current())  # Shift+F5 refreshes
    
    def _register_panels(self):
        """Register analysis panels."""
        # Data panels
        from biblium.gui.panels.data.load_panel import LoadDataPanel
        from biblium.gui.panels.data.view_panel import ViewDataPanel
        from biblium.gui.panels.data.filter_panel import FilterPanel
        from biblium.gui.panels.data.api_panel import APIDataPanel
        
        self.workspace.register_panel("load", LoadDataPanel)
        self.workspace.register_panel("api_data", APIDataPanel)
        self.workspace.register_panel("view", ViewDataPanel)
        self.workspace.register_panel("filter", FilterPanel)
        self.workspace.register_panel("quality", ViewDataPanel)  # Placeholder
        
        # Analysis panels
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
        from biblium.gui.panels.analysis.wordcloud_panel import WordCloudPanel
        from biblium.gui.panels.analysis.treemap_panel import TreemapPanel
        from biblium.gui.panels.analysis.relationship_panel import RelationshipPanel
        from biblium.gui.panels.analysis.distribution_panel import DistributionPanel
        from biblium.gui.panels.analysis.growth_models_panel import GrowthModelsPanel
        from biblium.gui.panels.analysis.citation_distribution_panel import CitationDistributionPanel
        from biblium.gui.panels.analysis.collaboration_panel import CollaborationPanel
        from biblium.gui.panels.analysis.altmetrics_panel import AltmetricsPanel
        from biblium.gui.panels.analysis.novelty_panel import NoveltyPanel
        from biblium.gui.panels.analysis.sentiment_panel import SentimentPanel
        from biblium.gui.panels.analysis.reference_benchmark_panel import ReferenceBenchmarkPanel
        
        self.workspace.register_panel("overview", OverviewPanel)
        self.workspace.register_panel("counts", CountsPanel)
        self.workspace.register_panel("statistics", StatisticsPanel)
        self.workspace.register_panel("trends", TrendsPanel)
        self.workspace.register_panel("laws", LawsPanel)
        self.workspace.register_panel("top_cited", TopCitedPanel)
        self.workspace.register_panel("citation_distribution", CitationDistributionPanel)
        self.workspace.register_panel("collaboration", CollaborationPanel)
        self.workspace.register_panel("reference_benchmark", ReferenceBenchmarkPanel)
        
        from biblium.gui.panels.analysis.diversity_panel import DiversityIndicesPanel
        self.workspace.register_panel("diversity_indices", DiversityIndicesPanel)
        
        self.workspace.register_panel("altmetrics", AltmetricsPanel)
        self.workspace.register_panel("novelty", NoveltyPanel)
        self.workspace.register_panel("sentiment", SentimentPanel)
        self.workspace.register_panel("kfields", KFieldsPanel)
        self.workspace.register_panel("relationships", RelationshipPanel)
        
        # Temporal panels
        self.workspace.register_panel("life_cycle", LifeCyclePanel)
        self.workspace.register_panel("production", ProductionOverTimePanel)
        self.workspace.register_panel("trend_topics", TrendTopicsPanel)
        self.workspace.register_panel("top_items_timeline", TopItemsTimelinePanel)
        self.workspace.register_panel("growth_models", GrowthModelsPanel)
        
        from biblium.gui.panels.analysis.temporal_diversity_panel import TemporalDiversityPanel
        self.workspace.register_panel("temporal_diversity", TemporalDiversityPanel)
        
        # Visualization panels
        self.workspace.register_panel("wordcloud", WordCloudPanel)
        self.workspace.register_panel("treemap", TreemapPanel)
        self.workspace.register_panel("distribution", DistributionPanel)
        
        from biblium.gui.panels.analysis.geographic_panel import GeographicPanel
        self.workspace.register_panel("geographic", GeographicPanel)
        
        from biblium.gui.panels.analysis.race_bar_panel import RaceBarPanel
        self.workspace.register_panel("race_bar", RaceBarPanel)
        
        # Factorial panels
        from biblium.gui.panels.analysis.factorial_panel import FactorialPanel
        
        self.workspace.register_panel("factorial", FactorialPanel)
        
        # Network panels
        from biblium.gui.panels.networks.network_panel import NetworkPanel
        from biblium.gui.panels.networks.thematic_map_panel import ThematicMapPanel as NetworkThematicMapPanel
        from biblium.gui.panels.networks.citation_network_panel import CitationNetworkPanel
        from biblium.gui.panels.networks.historiograph_panel import HistoriographPanel
        
        self.workspace.register_panel("network", NetworkPanel)
        self.workspace.register_panel("citation_network", CitationNetworkPanel)
        self.workspace.register_panel("thematic_map", NetworkThematicMapPanel)
        self.workspace.register_panel("historiograph", HistoriographPanel)
        
        # Mapping panels
        from biblium.gui.panels.mapping import ThematicMapPanel
        from biblium.gui.panels.analysis.topic_modeling_panel import TopicModelingPanel
        from biblium.gui.panels.analysis.dynamic_topics_panel import DynamicTopicsPanel
        
        self.workspace.register_panel("topic_modeling", TopicModelingPanel)
        self.workspace.register_panel("topics", TopicModelingPanel)  # Alias for backward compatibility
        self.workspace.register_panel("dynamic_topics", DynamicTopicsPanel)
        
        # Advanced panels
        from biblium.gui.panels.advanced import DisruptionPanel, ResearchFrontsPanel
        from biblium.gui.panels.analysis.sdg_panel import SDGPanel
        from biblium.gui.panels.analysis.disruption_panel import DisruptionIndexPanel
        from biblium.gui.panels.analysis.concept_builder_panel import ConceptBuilderPanel
        from biblium.gui.panels.analysis.pa_concepts_panel import PAConceptsPanel
        from biblium.gui.panels.analysis.my_concepts_panel import MyConceptsPanel
        from biblium.gui.panels.analysis.sleeping_beauty_panel import SleepingBeautyPanel
        from biblium.gui.panels.analysis.clustering_panel import DocumentClusteringPanel
        from biblium.gui.panels.analysis.entity_clustering_panel import EntityClusteringPanel
        from biblium.gui.panels.analysis.repository_links_panel import RepositoryLinksPanel
        
        self.workspace.register_panel("concept_builder", ConceptBuilderPanel)
        self.workspace.register_panel("pa_concepts", PAConceptsPanel)
        self.workspace.register_panel("my_concepts", MyConceptsPanel)
        self.workspace.register_panel("sdg", SDGPanel)
        self.workspace.register_panel("sleeping_beauty", SleepingBeautyPanel)
        self.workspace.register_panel("disruption", DisruptionPanel)  # Legacy
        self.workspace.register_panel("disruption_index", DisruptionIndexPanel)
        self.workspace.register_panel("document_clustering", DocumentClusteringPanel)
        self.workspace.register_panel("entity_clustering", EntityClusteringPanel)
        self.workspace.register_panel("repository_links", RepositoryLinksPanel)
        self.workspace.register_panel("fronts", ResearchFrontsPanel)
        
        from biblium.gui.panels.analysis.citation_patterns_panel import CitationPatternsPanel
        self.workspace.register_panel("citation_patterns", CitationPatternsPanel)
        
        from biblium.gui.panels.analysis.citation_velocity_panel import CitationVelocityPanel
        self.workspace.register_panel("citation_velocity", CitationVelocityPanel)
        
        from biblium.gui.panels.analysis.reference_diversity_panel import ReferenceDiversityPanel
        self.workspace.register_panel("reference_diversity", ReferenceDiversityPanel)
        
        from biblium.gui.panels.analysis.concept_extraction_panel import ConceptExtractionPanel
        self.workspace.register_panel("concept_extraction", ConceptExtractionPanel)
        
        from biblium.gui.panels.analysis.compare_means_panel import CompareMeansPanel
        self.workspace.register_panel("compare_means", CompareMeansPanel)
        
        from biblium.gui.panels.analysis.crosstabs_panel import CrosstabsPanel
        self.workspace.register_panel("crosstabs", CrosstabsPanel)
        
        from biblium.gui.panels.analysis.correlation_panel import CorrelationPanel
        self.workspace.register_panel("correlation", CorrelationPanel)
        
        # Group Analysis panels
        from biblium.gui.panels.groups import (
            GroupSetupPanel,
            GroupCountsPanel,
            GroupStatsPanel,
            GroupComparePanel,
            GroupAssociationsPanel,
            GroupVisualizationsPanel,
            GroupIntersectionsPanel,
            GroupClassificationPanel,
            GroupLogisticPanel,
        )
        
        self.workspace.register_panel("group_setup", GroupSetupPanel)
        self.workspace.register_panel("group_counts", GroupCountsPanel)
        self.workspace.register_panel("group_stats", GroupStatsPanel)
        self.workspace.register_panel("group_compare", GroupComparePanel)
        self.workspace.register_panel("group_intersections", GroupIntersectionsPanel)
        self.workspace.register_panel("group_associations", GroupAssociationsPanel)
        self.workspace.register_panel("group_visualizations", GroupVisualizationsPanel)
        self.workspace.register_panel("group_classification", GroupClassificationPanel)
        self.workspace.register_panel("group_logistic", GroupLogisticPanel)
        
        from biblium.gui.panels.groups.group_diversity_panel import GroupDiversityPanel
        self.workspace.register_panel("group_diversity", GroupDiversityPanel)
        
        # Legacy compatibility
        self.workspace.register_panel("group_overlap", GroupComparePanel)
        
        # Report panels
        from biblium.gui.panels.reports import ReportBuilderPanel, CustomReportPanel
        
        self.workspace.register_panel("report_builder", ReportBuilderPanel)
        self.workspace.register_panel("custom_report", CustomReportPanel)
        
        # Settings panel
        from biblium.gui.panels.settings_panel import SettingsPanel
        self.workspace.register_panel("settings", SettingsPanel)
    
    def _subscribe_events(self):
        """Subscribe to application events."""
        event_bus.subscribe(EventBus.DATASET_LOADED, self._on_dataset_loaded)
        event_bus.subscribe(EventBus.ERROR_OCCURRED, self._on_error)
        event_bus.subscribe("open_settings", lambda data: self._on_navigate("settings"))
        event_bus.subscribe("open_help", lambda data: self._show_docs())
        event_bus.subscribe("group_created", self._on_group_created)
    
    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================
    
    def _on_navigate(self, panel_id: str):
        """Handle sidebar navigation."""
        self.workspace.open_panel(panel_id, bib=self.bib)
    
    def _on_dataset_loaded(self, data):
        """Handle dataset loaded event."""
        if isinstance(data, dict):
            self.bib = data.get("bib")
            self.original_df = data.get("original_df")
            self.dataset_path = data.get("path")
            
            # Update workspace's shared bib - THIS IS THE KEY
            if self.bib:
                self.workspace.bib = self.bib
            
            # Update state
            if self.bib:
                self.state.update_data(
                    loaded=True,
                    path=self.dataset_path,
                    database=self.bib.db,
                    n_documents=self.bib.n,
                    columns=list(self.bib.df.columns),
                )
    
    def _on_error(self, data):
        """Handle error event."""
        message = data.get("message", str(data)) if isinstance(data, dict) else str(data)
        messagebox.showerror("Error", message)
    
    def _on_group_created(self, data):
        """Handle group creation event."""
        if isinstance(data, dict):
            self.bib_group = data.get("bib_group")
            
            # Update workspace's shared bib_group
            if self.bib_group:
                self.workspace.bib_group = self.bib_group
                
                # Show success message
                n_groups = len(self.bib_group.groups) if hasattr(self.bib_group, 'groups') else 0
                self.statusbar.set_status(f"✅ Created {n_groups} groups")
    
    def _on_close(self):
        """Handle window close."""
        if self.task_manager.has_running_tasks():
            if not messagebox.askyesno(
                "Tasks Running",
                "There are tasks still running. Are you sure you want to exit?"
            ):
                return
            self.task_manager.cancel_all()
        
        self.state.save_ui_state()
        self.destroy()
    
    # =========================================================================
    # ACTIONS
    # =========================================================================
    
    def _open_file(self):
        """Open file dialog to load dataset."""
        self._on_navigate("load")
    
    def _save_results(self):
        """Save current results."""
        if not self.bib:
            messagebox.showwarning("No Data", "No data to save.")
            return
        
        # For now, just show info
        messagebox.showinfo("Save", "Results are automatically saved to the results folder.")
    
    def _export(self):
        """Export data/results."""
        if not self.bib:
            messagebox.showwarning("No Data", "No data to export.")
            return
        
        filetypes = [
            ("Excel Files", "*.xlsx"),
            ("CSV Files", "*.csv"),
            ("All Files", "*.*"),
        ]
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=filetypes,
            title="Export Data",
        )
        
        if filename:
            try:
                if filename.endswith(".xlsx"):
                    self.bib.df.to_excel(filename, index=False)
                else:
                    self.bib.df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Data exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
    
    def _run_current(self):
        """Run analysis on current panel."""
        panel = self.workspace.get_active_panel()
        if panel and hasattr(panel, "_run_clicked"):
            panel._run_clicked()
        elif panel and hasattr(panel, "_run_analysis"):
            panel._run_analysis()
        elif panel and hasattr(panel, "_primary_action") and panel._primary_action:
            panel._primary_action()
    
    def _refresh_current(self):
        """Refresh current panel (reload data/update display)."""
        panel = self.workspace.get_active_panel()
        if panel:
            if hasattr(panel, "refresh"):
                panel.refresh()
            elif hasattr(panel, "_update_variable_lists"):
                panel._update_variable_lists()
            elif hasattr(panel, "on_data_loaded") and self.bib:
                panel.on_data_loaded(self.bib)
    
    def _count_all(self):
        """Run count all entities."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        def do_count(progress_cb, cancel_check):
            self.bib.count_all()
            return True
        
        def on_complete(result):
            event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": "Count All"})
            messagebox.showinfo("Complete", "All entity counts completed.")
        
        event_bus.emit(EventBus.ANALYSIS_STARTED, {"name": "Count All"})
        
        task = BackgroundTask(
            func=do_count,
            on_complete=on_complete,
            on_error=lambda e: event_bus.emit(EventBus.ERROR_OCCURRED, {"message": str(e)}),
        )
        task.start()
    
    def _open_counts(self, entity_type: str):
        """Open counts panel with specific entity pre-selected."""
        self._on_navigate("counts")
        # The panel will use its default, user can change entity type there
    
    def _generate_report(self):
        """Generate report."""
        self._on_navigate("report_builder")
    
    def _toggle_sidebar(self):
        """Toggle sidebar visibility."""
        if self.sidebar.winfo_viewable():
            self.sidebar.pack_forget()
        else:
            self.sidebar.pack(side=tk.LEFT, fill=tk.Y, before=self.sidebar.master.winfo_children()[1])
    
    def _set_theme(self, theme_name: str):
        """Change application theme."""
        # This would require rebuilding widgets - placeholder for now
        messagebox.showinfo("Theme", f"Theme switching to '{theme_name}' requires restart.")
    
    def _show_settings(self):
        """Show settings dialog."""
        settings_window = tk.Toplevel(self)
        settings_window.title("Settings")
        settings_window.geometry("550x650")
        settings_window.transient(self)
        settings_window.grab_set()
        
        # Center on parent
        settings_window.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 550) // 2
        y = self.winfo_y() + (self.winfo_height() - 650) // 2
        settings_window.geometry(f"+{x}+{y}")
        
        # Scrollable frame
        canvas = tk.Canvas(settings_window, bg=self.theme["bg_card"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(settings_window, orient="vertical", command=canvas.yview)
        main_frame = tk.Frame(canvas, bg=self.theme["bg_card"], padx=20, pady=20)
        
        main_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=main_frame, anchor="nw", width=520)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        tk.Label(
            main_frame, text="⚙️ Settings",
            font=("Segoe UI", 16, "bold"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(anchor=tk.W, pady=(0, 16))
        
        # ============ AI Settings Section ============
        tk.Label(
            main_frame, text="🤖 AI Analysis Settings",
            font=FONTS.get_font("body", bold=True),
            bg=self.theme["bg_card"], fg=self.theme["accent_primary"],
        ).pack(anchor=tk.W, pady=(8, 4))
        
        tk.Label(
            main_frame, text="Configure AI for automatic table and plot descriptions",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"], fg=self.theme["text_muted"],
        ).pack(anchor=tk.W, pady=(0, 8))
        
        # AI Provider
        ai_provider_frame = tk.Frame(main_frame, bg=self.theme["bg_card"])
        ai_provider_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            ai_provider_frame, text="Provider:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_secondary"], width=15, anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        self._ai_provider_var = tk.StringVar(value=self.state.data.get("ai_provider", "openai"))
        ai_provider_combo = ttk.Combobox(
            ai_provider_frame, textvariable=self._ai_provider_var,
            values=["openai", "anthropic", "google", "huggingface"], state="readonly", width=20,
        )
        ai_provider_combo.pack(side=tk.LEFT)
        
        # AI Model
        ai_model_frame = tk.Frame(main_frame, bg=self.theme["bg_card"])
        ai_model_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            ai_model_frame, text="Model:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_secondary"], width=15, anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        self._ai_model_var = tk.StringVar(value=self.state.data.get("ai_model", "gpt-4o-mini"))
        ai_model_entry = tk.Entry(
            ai_model_frame, textvariable=self._ai_model_var,
            font=FONTS.get_font("body"), width=25,
        )
        ai_model_entry.pack(side=tk.LEFT)
        
        # AI API Key
        ai_key_frame = tk.Frame(main_frame, bg=self.theme["bg_card"])
        ai_key_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            ai_key_frame, text="API Key:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_secondary"], width=15, anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        self._ai_key_var = tk.StringVar(value=self.state.data.get("ai_api_key", ""))
        ai_key_entry = tk.Entry(
            ai_key_frame, textvariable=self._ai_key_var,
            font=FONTS.get_font("body"), width=25, show="*",
        )
        ai_key_entry.pack(side=tk.LEFT)
        
        # Show/hide key button
        def toggle_key_visibility():
            if ai_key_entry.cget('show') == '*':
                ai_key_entry.config(show='')
                show_key_btn.config(text='🙈')
            else:
                ai_key_entry.config(show='*')
                show_key_btn.config(text='👁')
        
        show_key_btn = tk.Button(
            ai_key_frame, text="👁", font=("Segoe UI", 9),
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
            relief=tk.FLAT, command=toggle_key_visibility, width=3,
        )
        show_key_btn.pack(side=tk.LEFT, padx=(4, 0))
        
        # Custom Prompt
        tk.Label(
            main_frame, text="Custom Prompt (optional):",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
        ).pack(anchor=tk.W, pady=(8, 2))
        
        tk.Label(
            main_frame, text="Use {data} placeholder for table/plot data. Leave empty for default.",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"], fg=self.theme["text_muted"],
        ).pack(anchor=tk.W, pady=(0, 4))
        
        self._ai_prompt_text = tk.Text(
            main_frame, font=FONTS.get_font("body"),
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
            height=4, width=50, wrap=tk.WORD,
        )
        self._ai_prompt_text.pack(fill=tk.X, pady=(0, 8))
        self._ai_prompt_text.insert("1.0", self.state.data.get("ai_custom_prompt", ""))
        
        # Separator
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)
        
        # ============ Theme section ============
        tk.Label(
            main_frame, text="Appearance",
            font=FONTS.get_font("body", bold=True),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(anchor=tk.W, pady=(8, 4))
        
        theme_frame = tk.Frame(main_frame, bg=self.theme["bg_card"])
        theme_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            theme_frame, text="Theme:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_secondary"], width=15, anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        self._settings_theme_var = tk.StringVar(value=self.theme_name)
        theme_combo = ttk.Combobox(
            theme_frame, textvariable=self._settings_theme_var,
            values=["light", "dark"], state="readonly", width=20,
        )
        theme_combo.pack(side=tk.LEFT)
        
        # ============ Default folder section ============
        tk.Label(
            main_frame, text="Paths",
            font=FONTS.get_font("body", bold=True),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(anchor=tk.W, pady=(16, 4))
        
        folder_frame = tk.Frame(main_frame, bg=self.theme["bg_card"])
        folder_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            folder_frame, text="Results Folder:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_secondary"], width=15, anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        self._settings_results_var = tk.StringVar(value=self.state.data.get("results_folder", "results"))
        tk.Entry(
            folder_frame, textvariable=self._settings_results_var,
            font=FONTS.get_font("body"), width=25,
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        ttk.Button(folder_frame, text="Browse", command=lambda: self._browse_settings_folder()).pack(side=tk.LEFT)
        
        # ============ Analysis defaults section ============
        tk.Label(
            main_frame, text="Analysis Defaults",
            font=FONTS.get_font("body", bold=True),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(anchor=tk.W, pady=(16, 4))
        
        topn_frame = tk.Frame(main_frame, bg=self.theme["bg_card"])
        topn_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            topn_frame, text="Default Top N:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_secondary"], width=15, anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        self._settings_topn_var = tk.StringVar(value=str(self.state.data.get("default_top_n", 50)))
        tk.Spinbox(
            topn_frame, textvariable=self._settings_topn_var,
            from_=5, to=500, width=10, font=FONTS.get_font("body"),
        ).pack(side=tk.LEFT)
        
        # Plot settings
        dpi_frame = tk.Frame(main_frame, bg=self.theme["bg_card"])
        dpi_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            dpi_frame, text="Default Plot DPI:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_secondary"], width=15, anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        self._settings_dpi_var = tk.StringVar(value=str(self.state.data.get("default_dpi", 600)))
        tk.Spinbox(
            dpi_frame, textvariable=self._settings_dpi_var,
            from_=72, to=600, width=10, font=FONTS.get_font("body"),
        ).pack(side=tk.LEFT)
        
        # ============ Buttons ============
        btn_frame = tk.Frame(main_frame, bg=self.theme["bg_card"])
        btn_frame.pack(fill=tk.X, pady=(24, 0))
        
        ttk.Button(
            btn_frame, text="Save",
            command=lambda: self._save_settings(settings_window),
        ).pack(side=tk.RIGHT, padx=(8, 0))
        
        ttk.Button(
            btn_frame, text="Cancel",
            command=lambda: self._close_settings(settings_window, canvas),
        ).pack(side=tk.RIGHT)
    
    def _close_settings(self, window, canvas):
        """Close settings and unbind mousewheel."""
        canvas.unbind_all("<MouseWheel>")
        window.destroy()
    
    def _browse_settings_folder(self):
        """Browse for results folder in settings."""
        folder = filedialog.askdirectory(title="Select Results Folder")
        if folder:
            self._settings_results_var.set(folder)
    
    def _save_settings(self, window):
        """Save settings and close dialog."""
        # Save general settings to state
        self.state.data["results_folder"] = self._settings_results_var.get()
        self.state.data["default_top_n"] = int(self._settings_topn_var.get())
        self.state.data["default_dpi"] = int(self._settings_dpi_var.get())
        
        # Save AI settings to state
        self.state.data["ai_provider"] = self._ai_provider_var.get()
        self.state.data["ai_model"] = self._ai_model_var.get()
        self.state.data["ai_api_key"] = self._ai_key_var.get()
        self.state.data["ai_custom_prompt"] = self._ai_prompt_text.get("1.0", tk.END).strip()
        
        # Apply AI settings to DataTable class immediately
        from biblium.gui.widgets.tables import DataTable
        api_key = self._ai_key_var.get()
        DataTable.set_llm_settings(
            enabled=bool(api_key),
            provider=self._ai_provider_var.get(),
            model=self._ai_model_var.get(),
            api_key=api_key,
            custom_prompt=self._ai_prompt_text.get("1.0", tk.END).strip(),
        )
        
        # Check if theme changed
        new_theme = self._settings_theme_var.get()
        if new_theme != self.theme_name:
            self.state.data["theme"] = new_theme
            messagebox.showinfo("Theme Changed", "Please restart the application for theme changes to take effect.")
        
        self.state.save_ui_state()
        
        # Unbind mousewheel before closing
        try:
            window.children.get('!canvas', window).unbind_all("<MouseWheel>")
        except:
            pass
        
        window.destroy()
        messagebox.showinfo("Settings", "Settings saved successfully.\n\nAI settings are now active for all panels.")
    
    def _show_docs(self):
        """Show documentation."""
        import webbrowser
        webbrowser.open("https://github.com/lan-umek/biblium")
    
    def _show_shortcuts(self):
        """Show keyboard shortcuts."""
        shortcuts = """
Keyboard Shortcuts
==================

Ctrl+O      Open Dataset
Ctrl+S      Save Results
Ctrl+E      Export
Ctrl+R / F5 Run Current Analysis
Shift+F5    Refresh Panel
Ctrl+G      Generate Report
F1          Help / Documentation
Escape      Cancel Operation
        """
        messagebox.showinfo("Keyboard Shortcuts", shortcuts.strip())
    
    def _show_about(self):
        """Show about dialog."""
        about = """
Biblium 2.5
Bibliometric Analysis Platform

A comprehensive Python library and GUI application
for bibliometric analysis of scientific literature.

Author: Lan Umek
        """
        messagebox.showinfo("About Biblium", about.strip())
    
    def _refresh(self):
        """Refresh current view."""
        panel = self.workspace.get_active_panel()
        if panel and hasattr(panel, "refresh"):
            panel.refresh()


def main():
    """Application entry point."""
    app = BibliumApp()
    app.mainloop()


if __name__ == "__main__":
    main()
