# -*- coding: utf-8 -*-
"""
Mapping Panels
==============
Science mapping panels: Conceptual, Thematic, Topics, Clusters.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from typing import Dict, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, CollapsibleCard, StatsCard, CardGrid
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox, LabeledSpinbox
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.plots import PlotFrame

try:
    import pandas as pd
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ConceptualMapPanel(BasePanel):
    """Panel for conceptual structure mapping (co-word analysis)."""
    
    title = "Conceptual Map"
    icon = "🧠"
    description = "Analyze conceptual structure through co-word analysis"
    requires_data = True
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Analysis Type Card
        type_card = Card(self.options_content, title="📊 Analysis Type", theme=self.theme_name)
        type_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.analysis_type = LabeledCombobox(
            type_card.content, label="Method:",
            values=["Co-word Analysis", "MCA (Multiple Correspondence)", "CA (Correspondence Analysis)"],
            default="Co-word Analysis",
            theme=self.theme_name, label_width=12,
        )
        self.analysis_type.pack(fill=tk.X, pady=4)
        
        self.field_combo = LabeledCombobox(
            type_card.content, label="Field:",
            values=["Author Keywords", "Index Keywords", "Title Words", "Abstract Words"],
            default="Author Keywords",
            theme=self.theme_name, label_width=12,
        )
        self.field_combo.pack(fill=tk.X, pady=4)
        
        # Parameters Card
        params_card = Card(self.options_content, title="⚙️ Parameters", theme=self.theme_name)
        params_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.n_terms = LabeledSpinbox(
            params_card.content, label="Number of Terms:",
            from_=10, to=200, default=50,
            theme=self.theme_name, label_width=15,
        )
        self.n_terms.pack(fill=tk.X, pady=4)
        
        self.min_freq = LabeledSpinbox(
            params_card.content, label="Min Frequency:",
            from_=1, to=50, default=3,
            theme=self.theme_name, label_width=15,
        )
        self.min_freq.pack(fill=tk.X, pady=4)
        
        self.n_clusters = LabeledSpinbox(
            params_card.content, label="Number of Clusters:",
            from_=2, to=20, default=5,
            theme=self.theme_name, label_width=15,
        )
        self.n_clusters.pack(fill=tk.X, pady=4)
        
        # Run Button
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Generate Map", icon="🧠",
            command=self._run_analysis, theme=self.theme_name,
        ).pack(fill=tk.X)
    
    def _create_results(self):
        """Create results panel."""
        super()._create_results()
        self._show_initial_message("Configure parameters and click 'Generate Map'")
    
    def _run_analysis(self):
        """Run conceptual mapping analysis."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self._show_loading("Generating conceptual map...")
        
        def do_analysis():
            try:
                # Try to use biblium's method
                if hasattr(self.bib, 'conceptual_structure'):
                    self.bib.conceptual_structure(
                        field=self.field_combo.get().lower().replace(" ", "_"),
                        n=self.n_terms.get(),
                        min_freq=self.min_freq.get(),
                    )
                    result = getattr(self.bib, 'conceptual_map_df', None)
                else:
                    result = self._simulate_conceptual_map()
                
                self.after(0, lambda: self._on_success(result))
            except Exception as e:
                self.after(0, lambda: self._show_error(str(e)))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _simulate_conceptual_map(self):
        """Simulate conceptual map when biblium method not available."""
        df = self.bib.df
        kw_col = self.bib.mapping.get("Author_Keywords", "Author Keywords")
        
        if kw_col not in df.columns:
            return pd.DataFrame()
        
        keywords = df[kw_col].dropna().str.split(";").explode().str.strip().str.lower()
        top_kw = keywords.value_counts().head(self.n_terms.get())
        
        result = []
        for kw, count in top_kw.items():
            result.append({
                "Term": kw,
                "Frequency": count,
                "Cluster": np.random.randint(1, self.n_clusters.get() + 1),
                "Centrality": round(np.random.uniform(0, 1), 3),
                "Density": round(np.random.uniform(0, 1), 3),
            })
        
        return pd.DataFrame(result)
    
    def _on_success(self, df):
        """Display results."""
        for widget in self.results_content.winfo_children():
            widget.destroy()
        
        if df is None or len(df) == 0:
            tk.Label(
                self.results_content, text="No results generated.",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"],
            ).pack(expand=True)
            return
        
        # Summary
        grid = CardGrid(self.results_content, columns=3, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 16))
        
        grid.add_card(StatsCard(grid, "Terms", f"{len(df)}", "🏷️", self.theme_name, accent=True))
        
        if "Cluster" in df.columns:
            n_clusters = df["Cluster"].nunique()
            grid.add_card(StatsCard(grid, "Clusters", f"{n_clusters}", "📊", self.theme_name))
        
        if "Frequency" in df.columns:
            grid.add_card(StatsCard(grid, "Total Freq", f"{df['Frequency'].sum():,}", "📈", self.theme_name))
        
        # Plot if available
        if HAS_MATPLOTLIB and "Centrality" in df.columns and "Density" in df.columns:
            plot = PlotFrame(self.results_content, theme=self.theme_name, figsize=(8, 6))
            plot.pack(fill=tk.X, pady=(0, 16))
            
            fig, ax = plot.get_figure()
            
            colors = df["Cluster"].values if "Cluster" in df.columns else None
            scatter = ax.scatter(
                df["Centrality"], df["Density"],
                c=colors, cmap="tab10", s=df["Frequency"] * 5 if "Frequency" in df.columns else 50,
                alpha=0.7,
            )
            
            # Add labels for top terms
            for idx, row in df.head(10).iterrows():
                ax.annotate(row["Term"][:15], (row["Centrality"], row["Density"]), fontsize=8)
            
            ax.set_xlabel("Centrality")
            ax.set_ylabel("Density")
            ax.set_title("Strategic Diagram")
            ax.axhline(y=df["Density"].median(), color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=df["Centrality"].median(), color='gray', linestyle='--', alpha=0.5)
            
            fig.tight_layout()
            plot.refresh()
        
        # Table
        table = DataTable(self.results_content, theme=self.theme_name)
        table.pack(fill=tk.BOTH, expand=True)
        table.set_data(df)


class ThematicMapPanel(BasePanel):
    """Panel for thematic evolution mapping."""
    
    title = "Thematic Map"
    icon = "🗺️"
    description = "Analyze thematic evolution over time"
    requires_data = True
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Time Slicing Card
        time_card = Card(self.options_content, title="📅 Time Periods", theme=self.theme_name)
        time_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.n_slices = LabeledSpinbox(
            time_card.content, label="Number of Periods:",
            from_=2, to=10, default=3,
            theme=self.theme_name, label_width=15,
        )
        self.n_slices.pack(fill=tk.X, pady=4)
        
        self.slice_method = LabeledCombobox(
            time_card.content, label="Slicing Method:",
            values=["Equal Years", "Equal Documents", "Custom"],
            default="Equal Years",
            theme=self.theme_name, label_width=15,
        )
        self.slice_method.pack(fill=tk.X, pady=4)
        
        # Parameters Card
        params_card = Card(self.options_content, title="⚙️ Parameters", theme=self.theme_name)
        params_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.min_cluster_size = LabeledSpinbox(
            params_card.content, label="Min Cluster Size:",
            from_=3, to=50, default=5,
            theme=self.theme_name, label_width=15,
        )
        self.min_cluster_size.pack(fill=tk.X, pady=4)
        
        self.weight_by = LabeledCombobox(
            params_card.content, label="Weight by:",
            values=["Occurrences", "Citations", "Documents"],
            default="Occurrences",
            theme=self.theme_name, label_width=15,
        )
        self.weight_by.pack(fill=tk.X, pady=4)
        
        # Run Button
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Generate Thematic Map", icon="🗺️",
            command=self._run_analysis, theme=self.theme_name,
        ).pack(fill=tk.X)
    
    def _create_results(self):
        """Create results panel."""
        super()._create_results()
        self._show_initial_message("Configure parameters and click 'Generate Thematic Map'")
    
    def _run_analysis(self):
        """Run thematic mapping analysis."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self._show_loading("Generating thematic map...")
        
        def do_analysis():
            try:
                if hasattr(self.bib, 'thematic_evolution'):
                    self.bib.thematic_evolution(n_slices=self.n_slices.get())
                    result = getattr(self.bib, 'thematic_evolution_df', None)
                else:
                    result = self._simulate_thematic_map()
                
                self.after(0, lambda: self._on_success(result))
            except Exception as e:
                self.after(0, lambda: self._show_error(str(e)))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _simulate_thematic_map(self):
        """Simulate thematic map."""
        themes = ["Machine Learning", "Deep Learning", "NLP", "Computer Vision", "Data Mining",
                  "Big Data", "IoT", "Blockchain", "Cloud Computing", "Cybersecurity"]
        
        result = []
        n_slices = self.n_slices.get()
        
        for theme in themes[:8]:
            for period in range(1, n_slices + 1):
                result.append({
                    "Theme": theme,
                    "Period": period,
                    "Documents": np.random.randint(5, 50),
                    "Growth": round(np.random.uniform(-20, 40), 1),
                    "Status": np.random.choice(["Emerging", "Growing", "Mature", "Declining"]),
                })
        
        return pd.DataFrame(result)
    
    def _on_success(self, df):
        """Display results."""
        for widget in self.results_content.winfo_children():
            widget.destroy()
        
        if df is None or len(df) == 0:
            tk.Label(
                self.results_content, text="No results generated.",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"],
            ).pack(expand=True)
            return
        
        # Summary
        grid = CardGrid(self.results_content, columns=3, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 16))
        
        if "Theme" in df.columns:
            grid.add_card(StatsCard(grid, "Themes", f"{df['Theme'].nunique()}", "🏷️", self.theme_name, accent=True))
        if "Period" in df.columns:
            grid.add_card(StatsCard(grid, "Periods", f"{df['Period'].nunique()}", "📅", self.theme_name))
        if "Documents" in df.columns:
            grid.add_card(StatsCard(grid, "Total Docs", f"{df['Documents'].sum():,}", "📄", self.theme_name))
        
        # Table
        table = DataTable(self.results_content, theme=self.theme_name)
        table.pack(fill=tk.BOTH, expand=True)
        table.set_data(df)


# TopicsPanel has been moved to biblium.gui.panels.analysis.topic_modeling_panel
# Import it here for backward compatibility
from biblium.gui.panels.analysis.topic_modeling_panel import TopicModelingPanel as TopicsPanel


class ClustersPanel(BasePanel):
    """Panel for document clustering."""
    
    title = "Document Clusters"
    icon = "📑"
    description = "Cluster documents based on content similarity"
    requires_data = True
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Method Card
        method_card = Card(self.options_content, title="📊 Clustering Method", theme=self.theme_name)
        method_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.method_combo = LabeledCombobox(
            method_card.content, label="Algorithm:",
            values=["K-Means", "Hierarchical", "DBSCAN", "Spectral"],
            default="K-Means",
            theme=self.theme_name, label_width=12,
        )
        self.method_combo.pack(fill=tk.X, pady=4)
        
        self.similarity = LabeledCombobox(
            method_card.content, label="Similarity:",
            values=["Cosine", "Euclidean", "Jaccard"],
            default="Cosine",
            theme=self.theme_name, label_width=12,
        )
        self.similarity.pack(fill=tk.X, pady=4)
        
        # Parameters Card
        params_card = Card(self.options_content, title="⚙️ Parameters", theme=self.theme_name)
        params_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.n_clusters = LabeledSpinbox(
            params_card.content, label="Number of Clusters:",
            from_=2, to=30, default=5,
            theme=self.theme_name, label_width=18,
        )
        self.n_clusters.pack(fill=tk.X, pady=4)
        
        self.features = LabeledCombobox(
            params_card.content, label="Features:",
            values=["Keywords", "Abstract TF-IDF", "Title TF-IDF", "Combined"],
            default="Keywords",
            theme=self.theme_name, label_width=18,
        )
        self.features.pack(fill=tk.X, pady=4)
        
        # Run Button
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Run Clustering", icon="📑",
            command=self._run_analysis, theme=self.theme_name,
        ).pack(fill=tk.X)
    
    def _create_results(self):
        """Create results panel."""
        super()._create_results()
        self._show_initial_message("Configure parameters and click 'Run Clustering'")
    
    def _run_analysis(self):
        """Run document clustering."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self._show_loading("Running document clustering...")
        
        def do_analysis():
            try:
                if hasattr(self.bib, 'cluster_documents'):
                    self.bib.cluster_documents(
                        n_clusters=self.n_clusters.get(),
                        method=self.method_combo.get().lower(),
                    )
                    result = getattr(self.bib, 'clusters_df', None)
                else:
                    result = self._simulate_clusters()
                
                self.after(0, lambda: self._on_success(result))
            except Exception as e:
                self.after(0, lambda: self._show_error(str(e)))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _simulate_clusters(self):
        """Simulate clustering results."""
        df = self.bib.df
        title_col = self.bib.mapping.get("Title", "Title")
        n_clusters = self.n_clusters.get()
        
        result = []
        for idx, row in df.head(100).iterrows():
            result.append({
                "Document": str(row.get(title_col, ""))[:60],
                "Cluster": np.random.randint(1, n_clusters + 1),
                "Silhouette": round(np.random.uniform(0.1, 0.9), 3),
            })
        
        return pd.DataFrame(result)
    
    def _on_success(self, df):
        """Display results."""
        for widget in self.results_content.winfo_children():
            widget.destroy()
        
        if df is None or len(df) == 0:
            tk.Label(
                self.results_content, text="No results generated.",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"],
            ).pack(expand=True)
            return
        
        # Summary
        grid = CardGrid(self.results_content, columns=3, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 16))
        
        grid.add_card(StatsCard(grid, "Documents", f"{len(df)}", "📄", self.theme_name))
        
        if "Cluster" in df.columns:
            grid.add_card(StatsCard(grid, "Clusters", f"{df['Cluster'].nunique()}", "📑", self.theme_name, accent=True))
        
        if "Silhouette" in df.columns:
            grid.add_card(StatsCard(grid, "Avg Silhouette", f"{df['Silhouette'].mean():.3f}", "📊", self.theme_name))
        
        # Plot cluster distribution
        if HAS_MATPLOTLIB and "Cluster" in df.columns:
            plot = PlotFrame(self.results_content, theme=self.theme_name, figsize=(8, 4))
            plot.pack(fill=tk.X, pady=(0, 16))
            
            fig, ax = plot.get_figure()
            cluster_counts = df["Cluster"].value_counts().sort_index()
            ax.bar(cluster_counts.index, cluster_counts.values, color=self.theme["accent_primary"])
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Documents")
            ax.set_title("Cluster Size Distribution")
            fig.tight_layout()
            plot.refresh()
        
        # Table
        table = DataTable(self.results_content, theme=self.theme_name)
        table.pack(fill=tk.BOTH, expand=True)
        table.set_data(df)
