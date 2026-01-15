# -*- coding: utf-8 -*-
"""
Factorial Analysis Panel
========================
Conceptual structure analysis using MCA, CA, PCA with clustering.
Uses biblium's plotbib functions for Word Map, Topic Dendrogram, etc.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from typing import Dict, Optional, List, Tuple
import io
import os

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledSpinbox, LabeledCheckbox, LabeledEntry

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image, ImageTk
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class FactorialPanel(BasePanel):
    """Panel for factorial/conceptual structure analysis using biblium."""
    
    title = "Factorial Analysis"
    icon = "📐"
    description = "MCA, CA, PCA with clustering and visualization"
    requires_data = True
    
    # Field options for analysis
    FIELD_OPTIONS = [
        "Author Keywords",
        "Index Keywords",
        "Title",
        "Abstract",
        "Authors",
        "Sources",
        "Countries",
        "Affiliations",
        "References",
    ]
    
    # Dimensionality reduction methods
    DR_METHODS = ["MCA", "CA", "PCA", "SVD"]
    
    # Clustering methods
    CLUSTER_METHODS = ["kmeans", "hierarchical", "spectral", "dbscan"]
    
    # DTM methods
    DTM_METHODS = ["count", "tfidf"]
    
    # Term selection methods
    # Note: chi2 and mutual_info require a target variable (y) for supervised selection,
    # which is not available in the GUI. Only frequency-based selection is supported.
    TERM_SELECTION = ["frequency"]
    
    # Visualization types
    VIS_TYPES = [
        "Word Map",
        "Topic Dendrogram",
        "Cluster Heatmap",
        "Terms by Cluster",
    ]
    
    # Linkage methods for dendrogram
    LINKAGE_METHODS = ["ward", "complete", "average", "single"]
    
    # Distance metrics
    DISTANCE_METRICS = ["euclidean", "cosine", "manhattan", "correlation"]
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._current_fig = None
        self._photo_image = None
        self._conceptual_result = None
        self._last_image_bytes = None
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._run_analysis
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Visualization Selection (at top)
        vis_card = Card(self.options_content, title="📈 Visualization", theme=self.theme_name)
        vis_card.pack(fill=tk.X, padx=8, pady=4)
        
        self.vis_combo = LabeledCombobox(
            vis_card.content, label="Plot Type:", values=self.VIS_TYPES,
            default="Word Map", theme=self.theme_name, label_width=10)
        self.vis_combo.pack(fill=tk.X, pady=2)
        self.vis_combo.bind("<<ComboboxSelected>>", self._on_vis_change)
        
        # Dendrogram-specific options (hidden by default)
        self.dendro_frame = tk.Frame(vis_card.content, bg=self.theme["bg_card"])
        
        self.linkage_combo = LabeledCombobox(
            self.dendro_frame, label="Linkage:", values=self.LINKAGE_METHODS,
            default="ward", theme=self.theme_name, label_width=10)
        self.linkage_combo.pack(fill=tk.X, pady=2)
        
        self.metric_combo = LabeledCombobox(
            self.dendro_frame, label="Metric:", values=self.DISTANCE_METRICS,
            default="euclidean", theme=self.theme_name, label_width=10)
        self.metric_combo.pack(fill=tk.X, pady=2)
        
        # Field Selection
        field_card = Card(self.options_content, title="📊 Field Selection", theme=self.theme_name)
        field_card.pack(fill=tk.X, padx=8, pady=4)
        
        self.field_combo = LabeledCombobox(
            field_card.content, label="Field:", values=self.FIELD_OPTIONS,
            default="Author Keywords", theme=self.theme_name, label_width=10)
        self.field_combo.pack(fill=tk.X, pady=2)
        
        self.n_terms_spin = LabeledSpinbox(
            field_card.content, label="N Terms:", from_=20, to=500, default=100,
            theme=self.theme_name, label_width=10)
        self.n_terms_spin.pack(fill=tk.X, pady=2)
        
        self.min_df_spin = LabeledSpinbox(
            field_card.content, label="Min Doc Freq:", from_=1, to=50, default=2,
            theme=self.theme_name, label_width=10)
        self.min_df_spin.pack(fill=tk.X, pady=2)
        
        # Analysis Settings
        analysis_card = Card(self.options_content, title="⚙️ Analysis Settings", theme=self.theme_name)
        analysis_card.pack(fill=tk.X, padx=8, pady=4)
        
        self.dr_method_combo = LabeledCombobox(
            analysis_card.content, label="DR Method:", values=self.DR_METHODS,
            default="MCA", theme=self.theme_name, label_width=10)
        self.dr_method_combo.pack(fill=tk.X, pady=2)
        
        self.n_components_spin = LabeledSpinbox(
            analysis_card.content, label="Components:", from_=2, to=10, default=2,
            theme=self.theme_name, label_width=10)
        self.n_components_spin.pack(fill=tk.X, pady=2)
        
        self.dtm_method_combo = LabeledCombobox(
            analysis_card.content, label="DTM Method:", values=self.DTM_METHODS,
            default="count", theme=self.theme_name, label_width=10)
        self.dtm_method_combo.pack(fill=tk.X, pady=2)
        
        # Note: Term selection is fixed to "frequency" as chi2/mutual_info require
        # supervised labels (y) which are not available in this interface
        
        # Clustering Settings
        cluster_card = Card(self.options_content, title="🔬 Clustering", theme=self.theme_name)
        cluster_card.pack(fill=tk.X, padx=8, pady=4)
        
        self.cluster_method_combo = LabeledCombobox(
            cluster_card.content, label="Method:", values=self.CLUSTER_METHODS,
            default="kmeans", theme=self.theme_name, label_width=10)
        self.cluster_method_combo.pack(fill=tk.X, pady=2)
        
        self.n_clusters_spin = LabeledSpinbox(
            cluster_card.content, label="N Clusters:", from_=2, to=20, default=5,
            theme=self.theme_name, label_width=10)
        self.n_clusters_spin.pack(fill=tk.X, pady=2)
        
        # Appearance
        appear_card = Card(self.options_content, title="🎨 Appearance", theme=self.theme_name)
        appear_card.pack(fill=tk.X, padx=8, pady=4)
        
        self.colormap_combo = LabeledCombobox(
            appear_card.content, label="Colormap:",
            values=["tab10", "viridis", "plasma", "Set1", "Set2", "Paired", "coolwarm", "Spectral"],
            default="tab10", theme=self.theme_name, label_width=10)
        self.colormap_combo.pack(fill=tk.X, pady=2)
        
        self.marker_size_spin = LabeledSpinbox(
            appear_card.content, label="Marker Size:", from_=20, to=200, default=50,
            theme=self.theme_name, label_width=10)
        self.marker_size_spin.pack(fill=tk.X, pady=2)
        
        self.show_legend_cb = LabeledCheckbox(
            appear_card.content, label="Show legend", default=True, theme=self.theme_name)
        self.show_legend_cb.pack(fill=tk.X, pady=2)
        
        # Run Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=8)
        
        ActionButton(btn_frame, text="Run Analysis", icon="📐",
            command=self._run_analysis, theme=self.theme_name).pack(fill=tk.X)
        
        ThemedButton(btn_frame, text="Export Results", style="secondary",
            command=self._export_results, theme=self.theme_name).pack(fill=tk.X, pady=(4, 0))
    
    def _on_vis_change(self, event=None):
        """Update UI based on visualization type."""
        vis_type = self.vis_combo.get()
        if vis_type == "Topic Dendrogram":
            self.dendro_frame.pack(fill=tk.X, pady=2)
        else:
            self.dendro_frame.pack_forget()
    
    def _create_results(self):
        """Create results panel."""
        self.results_card = tk.Frame(
            self.results_frame, bg=self.theme["bg_card"],
            highlightbackground=self.theme["border"], highlightthickness=1)
        self.results_card.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.results_notebook = ttk.Notebook(self.results_card)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Plot tab
        self.plot_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.plot_frame, text="📊 Plot")
        
        self.image_label = tk.Label(self.plot_frame, bg=self.theme["bg_card"])
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Clusters tab
        self.clusters_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.clusters_frame, text="📋 Clusters")
        
        self.clusters_tree_frame = tk.Frame(self.clusters_frame, bg=self.theme["bg_card"])
        self.clusters_tree_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Embeddings tab
        self.embed_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.embed_frame, text="📉 Embeddings")
        
        # Info tab
        info_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(info_frame, text="ℹ️ Info")
        self._create_info_content(info_frame)
        
        self.embed_tree_frame = tk.Frame(self.embed_frame, bg=self.theme["bg_card"])
        self.embed_tree_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Note: ScaledImageFrame handles its own resize events
    
    def _on_plot_resize(self, event=None):
        # Deprecated - ScaledImageFrame handles resize internally
        pass
    
    def _show_placeholder(self, message: str):
        """Show a placeholder message in the plot frame."""
        # Clear all widgets in plot_frame
        for widget in self.plot_frame.winfo_children():
            try:
                widget.destroy()
            except Exception:
                pass
        
        # Create a new label for the message
        self.image_label = tk.Label(
            self.plot_frame, 
            text=message, 
            font=FONTS.get_font("body"), 
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"]
        )
        self.image_label.pack(expand=True)
    
    def _run_analysis(self):
        """Run factorial analysis using biblium's implementation."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        event_bus.emit(EventBus.ANALYSIS_STARTED, {"name": self.title})
        self._is_running = True
        
        vis_type = self.vis_combo.get()
        field = self.field_combo.get()
        self._show_placeholder(f"⏳ Running {vis_type} on {field}...")
        
        # Get parameters
        params = {
            'n_terms': self.n_terms_spin.get(),
            'min_df': self.min_df_spin.get(),
            'dr_method': self.dr_method_combo.get(),
            'n_components': self.n_components_spin.get(),
            'dtm_method': self.dtm_method_combo.get(),
            'term_selection': 'frequency',  # Fixed: chi2/mutual_info require supervised labels
            'cluster_method': self.cluster_method_combo.get(),
            'n_clusters': self.n_clusters_spin.get(),
            'colormap': self.colormap_combo.get(),
            'marker_size': self.marker_size_spin.get(),
            'show_legend': self.show_legend_cb.get(),
            'linkage_method': self.linkage_combo.get(),
            'metric': self.metric_combo.get(),
            'vis_type': vis_type,
            'field': field,
        }
        
        def do_analysis():
            try:
                from biblium import utilsbib
                
                print(f"[DEBUG] Factorial Analysis: field={field}, dr={params['dr_method']}, clusters={params['n_clusters']}, vis={vis_type}")
                
                # Run conceptual structure analysis using biblium
                result = utilsbib.conceptual_structure_analysis(
                    self.bib.df,
                    field=field,
                    dr_method=params['dr_method'],
                    cluster_method=params['cluster_method'],
                    n_clusters=params['n_clusters'],
                    n_terms=params['n_terms'],
                    n_components=params['n_components'],
                    dtm_method=params['dtm_method'],
                    term_selection=params['term_selection'],
                    min_df=params['min_df'],
                    compute_metrics=True,
                    keyword_separator=getattr(self.bib, 'default_separator', '; '),
                )
                
                self._conceptual_result = result
                
                # Generate visualization
                fig = self._create_visualization(result, params)
                
                if fig:
                    self._current_fig = fig
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
                    buf.seek(0)
                    image_bytes = buf.getvalue()
                    buf.close()
                    plt.close(fig)
                    self.after(0, lambda: self._show_results(image_bytes, result))
                else:
                    self.after(0, lambda: self._on_error("Failed to create visualization"))
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: self._on_error(msg))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _create_visualization(self, result, params):
        """Create visualization using biblium's plotting style."""
        from biblium import utilsbib
        from scipy.cluster.hierarchy import linkage, dendrogram
        import seaborn as sns
        
        plt.ioff()
        
        embeddings = result.get("term_embeddings")
        terms = result.get("terms", [])
        labels = result.get("term_labels", [])
        
        if embeddings is None or len(embeddings) == 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, "No embeddings computed", ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            return fig
        
        # Balance parentheses in terms (following biblium's convention)
        terms = [utilsbib._balance_closing_parenthesis(str(t)) for t in terms]
        
        vis_type = params['vis_type']
        field = params['field']
        
        if vis_type == "Word Map":
            return self._plot_word_map(embeddings, terms, labels, params, field)
        elif vis_type == "Topic Dendrogram":
            return self._plot_dendrogram(embeddings, terms, params, field)
        elif vis_type == "Cluster Heatmap":
            return self._plot_cluster_heatmap(embeddings, terms, labels, params, field)
        elif vis_type == "Terms by Cluster":
            return self._plot_terms_by_cluster(terms, labels, params, field)
        
        return None
    
    def _plot_word_map(self, embeddings, terms, labels, params, field):
        """
        Plot word map following biblium's plot_word_map implementation.
        Reference: biblium/plotbib.py lines 4787-4924
        """
        embeddings = np.asarray(embeddings)
        labels = np.asarray(labels)
        
        # Ensure 2D for scatter (following biblium)
        if embeddings.ndim == 1:
            x = embeddings
            y = np.zeros_like(x)
            embeddings = np.vstack((x, y)).T
        elif embeddings.ndim == 2 and embeddings.shape[1] == 1:
            x = embeddings[:, 0]
            y = np.zeros_like(x)
            embeddings = np.vstack((x, y)).T
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        scatter = ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=labels,
            cmap=params['colormap'],
            s=params['marker_size'],
        )
        
        # Add term labels (following biblium)
        for i, term in enumerate(terms):
            ax.text(
                embeddings[i, 0],
                embeddings[i, 1],
                term,
                fontsize=8,
            )
        
        ax.set_title(f"Word Map - {field}", fontsize=12)
        ax.set_xlabel("Dim 1", fontsize=10)
        ax.set_ylabel("Dim 2", fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        
        # No grid - following biblium's plot_word_map
        ax.grid(False)
        
        if params['show_legend']:
            handles, label_values = scatter.legend_elements()
            legend = ax.legend(handles, label_values, title="Cluster", fontsize=8)
            legend.get_title().set_fontsize(10)
        
        fig.tight_layout()
        return fig
    
    def _plot_dendrogram(self, embeddings, terms, params, field):
        """
        Plot dendrogram following biblium's plot_topic_dendrogram implementation.
        Reference: biblium/plotbib.py lines 4928-5056
        """
        from scipy.cluster.hierarchy import linkage, dendrogram
        
        emb = np.atleast_2d(embeddings)
        
        if emb.shape[0] < 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, "Not enough terms for dendrogram", ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            return fig
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Hierarchical clustering (following biblium)
        Z = linkage(emb, method=params['linkage_method'], metric=params['metric'])
        
        # Plot dendrogram (following biblium)
        dendrogram(
            Z,
            labels=terms,
            leaf_rotation=90,
            leaf_font_size=8,
            ax=ax,
        )
        
        ax.set_title(f"Topic Dendrogram - {field}", fontsize=12)
        ax.set_xlabel("Terms", fontsize=10)
        ax.set_ylabel("Distance", fontsize=10)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        
        # Remove gridlines - following biblium's plot_topic_dendrogram
        ax.grid(False)
        ax.set_facecolor("white")
        
        fig.tight_layout()
        return fig
    
    def _plot_cluster_heatmap(self, embeddings, terms, labels, params, field):
        """Plot cluster membership heatmap."""
        import seaborn as sns
        
        # Create DataFrame with embeddings
        n_dims = min(embeddings.shape[1], 5)
        df = pd.DataFrame(
            embeddings[:, :n_dims],
            index=terms,
            columns=[f"Dim {i+1}" for i in range(n_dims)]
        )
        
        # Sort by cluster
        sort_idx = np.argsort(labels)
        df = df.iloc[sort_idx]
        
        if df.shape[0] < 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, "Not enough terms for heatmap", ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            return fig
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(df, cmap=params['colormap'], ax=ax, 
                   yticklabels=True, xticklabels=True,
                   cbar_kws={'label': 'Embedding Value'})
        
        ax.set_title(f"Cluster Heatmap - {field}", fontsize=12)
        ax.set_xlabel("Dimensions", fontsize=10)
        ax.set_ylabel("Terms", fontsize=10)
        
        fig.tight_layout()
        return fig
    
    def _plot_terms_by_cluster(self, terms, labels, params, field):
        """Plot terms grouped by cluster."""
        # Group terms by cluster
        cluster_terms = {}
        for term, lbl in zip(terms, labels):
            if lbl not in cluster_terms:
                cluster_terms[lbl] = []
            cluster_terms[lbl].append(term)
        
        n_clusters = len(cluster_terms)
        
        if n_clusters == 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, "No cluster data", ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            return fig
        
        # Create figure with subplots
        cols = min(n_clusters, 4)
        rows = (n_clusters + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 6 * rows))
        
        if n_clusters == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        cmap = plt.cm.get_cmap(params['colormap'])
        
        for i, (cluster_id, cluster_terms_list) in enumerate(sorted(cluster_terms.items())):
            row, col = i // cols, i % cols
            ax = axes[row, col]
            
            # Show top 15 terms per cluster
            top_terms = cluster_terms_list[:15]
            y_pos = range(len(top_terms))
            
            color = cmap(i / max(1, n_clusters - 1))
            ax.barh(y_pos, [1] * len(top_terms), color=color, alpha=0.7, edgecolor='white')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_terms, fontsize=9)
            ax.invert_yaxis()
            ax.set_title(f"Cluster {cluster_id}\n({len(cluster_terms_list)} terms)", fontsize=10)
            ax.set_xlim(0, 1.2)
            ax.set_xticks([])
            ax.grid(False)
        
        # Hide unused subplots
        for i in range(n_clusters, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].set_visible(False)
        
        fig.suptitle(f"Terms by Cluster - {field}", fontsize=14, y=1.02)
        fig.tight_layout()
        return fig
    
    def _display_image(self, img_bytes):
        # Clear existing widgets in plot_frame
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        try:
            from biblium.gui.widgets.plots import ScaledImageFrame
            
            buf = io.BytesIO(img_bytes)
            img = Image.open(buf)
            
            scaled_frame = ScaledImageFrame(
                self.plot_frame, 
                theme=self.theme_name,
                maintain_aspect=True,
                max_scale=1.5
            )
            scaled_frame.pack(fill=tk.BOTH, expand=True)
            scaled_frame.set_image(img)
            
            self._scaled_frame = scaled_frame
        except Exception as e:
            # Fallback to original method
            buf = io.BytesIO(img_bytes)
            img = Image.open(buf)
            self._photo_image = ImageTk.PhotoImage(img)
            self.image_label = tk.Label(self.plot_frame, image=self._photo_image, bg=self.theme["bg_card"])
            self.image_label.pack(fill=tk.BOTH, expand=True)
    
    def _show_results(self, img_bytes, result):
        self._is_running = False
        self._last_image_bytes = img_bytes
        event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": self.title})
        
        self._display_image(img_bytes)
        self._update_clusters_table(result)
        self._update_embeddings_table(result)
    
    def _update_clusters_table(self, result):
        """Update clusters table."""
        for w in self.clusters_tree_frame.winfo_children():
            w.destroy()
        
        terms = result.get("terms", [])
        labels = result.get("term_labels", [])
        
        print(f"[DEBUG] _update_clusters_table: terms={len(terms) if terms else 0}, labels={len(labels) if hasattr(labels, '__len__') else 'N/A'}")
        
        if terms is None or len(terms) == 0:
            tk.Label(self.clusters_tree_frame, text="No cluster data available", 
                    bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(expand=True)
            return
        
        if labels is None or (hasattr(labels, '__len__') and len(labels) == 0):
            tk.Label(self.clusters_tree_frame, text="No cluster labels available", 
                    bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(expand=True)
            return
        
        # Create DataFrame
        import numpy as np
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        
        df = pd.DataFrame({"Term": terms, "Cluster": labels})
        df = df.sort_values(["Cluster", "Term"])
        
        # Use pack layout with a container frame
        container = tk.Frame(self.clusters_tree_frame, bg=self.theme["bg_card"])
        container.pack(fill=tk.BOTH, expand=True)
        
        cols = list(df.columns)
        tree = ttk.Treeview(container, columns=cols, show='headings', height=15)
        
        vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        for col in cols:
            tree.column(col, width=150, anchor='w')
            tree.heading(col, text=col)
        
        for _, row in df.iterrows():
            tree.insert("", "end", values=list(row))
        
        # Use pack layout
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    def _update_embeddings_table(self, result):
        """Update embeddings table."""
        for w in self.embed_tree_frame.winfo_children():
            w.destroy()
        
        embeddings = result.get("term_embeddings")
        terms = result.get("terms", [])
        
        print(f"[DEBUG] _update_embeddings_table: terms={len(terms) if terms else 0}, embeddings shape={embeddings.shape if embeddings is not None else 'None'}")
        
        if embeddings is None or len(embeddings) == 0:
            tk.Label(self.embed_tree_frame, text="No embedding data available", 
                    bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(expand=True)
            return
        
        if terms is None or len(terms) == 0:
            tk.Label(self.embed_tree_frame, text="No term data available", 
                    bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(expand=True)
            return
        
        # Use pack layout with a container frame
        container = tk.Frame(self.embed_tree_frame, bg=self.theme["bg_card"])
        container.pack(fill=tk.BOTH, expand=True)
        
        # Create DataFrame
        n_dims = min(embeddings.shape[1], 10)
        cols = ["Term"] + [f"Dim {i+1}" for i in range(n_dims)]
        
        tree = ttk.Treeview(container, columns=cols, show='headings', height=15)
        
        vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        tree.column("Term", width=150, anchor='w')
        tree.heading("Term", text="Term")
        for i in range(n_dims):
            col = f"Dim {i+1}"
            tree.column(col, width=80, anchor='center')
            tree.heading(col, text=col)
        
        for i, term in enumerate(terms):
            vals = [str(term)] + [f"{embeddings[i, j]:.4f}" for j in range(n_dims)]
            tree.insert("", "end", values=vals)
        
        # Use pack layout
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    def _on_error(self, msg):
        self._is_running = False
        event_bus.emit(EventBus.ERROR_OCCURRED, {"message": msg})
        self._show_placeholder(f"❌ Error: {msg}")
        messagebox.showerror("Error", msg)
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
FACTORIAL ANALYSIS
══════════════════

Dimensionality reduction and conceptual mapping.

METHODS
───────
• MCA (Multiple Correspondence Analysis)
  - For categorical data
  - Keywords, subject areas
  - Most common in bibliometrics
  
• CA (Correspondence Analysis)
  - Two-way contingency tables
  - Row-column associations
  
• PCA (Principal Component Analysis)
  - For numeric data
  - Creates linear combinations
  
• SVD (Singular Value Decomposition)
  - Matrix factorization
  - Document-term matrices

OUTPUT
──────
• Eigenvalues: Variance per component
• Explained variance %
• Loadings: Variable contributions
• Scores: Entity positions
• 2D/3D visualization
• Clustering on factors

VISUALIZATION
─────────────
• Factorial map (2D plot)
• Scree plot (variance)
• Dendrogram (clustering)
• Contribution plots

INTERPRETATION
──────────────
• Close points = similar
• Axes = underlying dimensions
• Clusters = conceptual groups
"""
        text_widget = tk.Text(
            parent, font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            wrap=tk.WORD, padx=16, pady=16, relief=tk.FLAT,
        )
        text_widget.insert("1.0", info_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Add right-click context menu for copy
        def show_context_menu(event):
            menu = tk.Menu(text_widget, tearoff=0)
            menu.add_command(label="Copy Selected", command=lambda: copy_selected(text_widget))
            menu.add_command(label="Copy All", command=lambda: copy_all(text_widget))
            menu.tk_popup(event.x_root, event.y_root)
        
        def copy_selected(widget):
            try:
                widget.config(state=tk.NORMAL)
                selected = widget.get(tk.SEL_FIRST, tk.SEL_LAST)
                widget.config(state=tk.DISABLED)
                widget.clipboard_clear()
                widget.clipboard_append(selected)
            except tk.TclError:
                pass  # No selection
        
        def copy_all(widget):
            widget.config(state=tk.NORMAL)
            content = widget.get("1.0", tk.END)
            widget.config(state=tk.DISABLED)
            widget.clipboard_clear()
            widget.clipboard_append(content.strip())
        
        text_widget.bind("<Button-3>", show_context_menu)  # Right-click
        text_widget.bind("<Control-c>", lambda e: copy_selected(text_widget))

    def _export_results(self):
        if self._conceptual_result is None:
            messagebox.showwarning("No Results", "Run analysis first.")
            return
        
        fp = filedialog.asksaveasfilename(defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("All", "*.*")])
        
        if fp:
            try:
                result = self._conceptual_result
                terms = result.get("terms", [])
                labels = result.get("term_labels", [])
                embeddings = result.get("term_embeddings")
                
                with pd.ExcelWriter(fp) as writer:
                    # Clusters sheet
                    if terms and labels:
                        df_clusters = pd.DataFrame({"Term": terms, "Cluster": labels})
                        df_clusters.to_excel(writer, sheet_name='Clusters', index=False)
                    
                    # Embeddings sheet
                    if embeddings is not None:
                        n_dims = embeddings.shape[1]
                        embed_data = {"Term": terms}
                        for i in range(n_dims):
                            embed_data[f"Dim {i+1}"] = embeddings[:, i]
                        df_embed = pd.DataFrame(embed_data)
                        df_embed.to_excel(writer, sheet_name='Embeddings', index=False)
                
                messagebox.showinfo("Exported", f"Saved to {fp}")
                
                # Save plot
                if self._current_fig:
                    plot_path = fp.rsplit('.', 1)[0] + '_plot.png'
                    self._current_fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                    
            except Exception as e:
                messagebox.showerror("Export Error", str(e))
