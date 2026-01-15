"""
Entity Clustering Panel
=======================

Cluster entities (authors, sources, keywords, etc.) based on co-occurrence patterns.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import threading

from biblium.gui.panels.base import BasePanel
from biblium.gui.config import FONTS, get_theme
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.plots import PlotFrame
from biblium.gui.widgets.forms import LabeledEntry, LabeledCombobox, LabeledSpinbox
from biblium.gui.widgets.cards import Card, StatsCard, CardGrid
from biblium.gui.widgets.buttons import ActionButton

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class EntityClusteringPanel(BasePanel):
    """
    Panel for clustering entities (authors, keywords, sources, etc.).
    
    Uses biblium's cluster_entities method which clusters based on
    co-occurrence patterns or document presence.
    """
    
    title = "Entity Clustering"
    icon = "🏷️"
    description = "Cluster authors, keywords, sources by co-occurrence"
    requires_data = True
    
    METHODS = ["kmeans", "hierarchical", "spectral"]
    FEATURES = ["cooccurrence", "documents"]
    SCORERS = ["silhouette", "calinski"]
    
    # Proper entity columns for clustering
    ENTITY_COLUMNS = [
        "Authors",
        "Countries", 
        "Affiliations",
        "Author Keywords", "Processed Author Keywords",
        "Index Keywords", "Processed Index Keywords",
        "Words from Abstract",
        "Words from Title",
        "Funding Sponsor",
    ]
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._cluster_result = None
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Entity Selection Card
        entity_card = Card(self.options_content, title="🏷️ Entity Selection", theme=self.theme_name)
        entity_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.entity_combo = LabeledCombobox(
            entity_card.content, label="Entity Column:",
            values=self.ENTITY_COLUMNS, default="Author Keywords",
            theme=self.theme_name, label_width=14,
            tooltip="Select the column containing entities to cluster"
        )
        self.entity_combo.pack(fill=tk.X, pady=4)
        
        self.min_occur_spin = LabeledSpinbox(
            entity_card.content, label="Min Occurrences:",
            from_=1, to=100, default=2,
            theme=self.theme_name, label_width=14,
            tooltip="Minimum times an entity must appear to be included"
        )
        self.min_occur_spin.pack(fill=tk.X, pady=4)
        
        # Method Card
        method_card = Card(self.options_content, title="📊 Clustering Method", theme=self.theme_name)
        method_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.method_combo = LabeledCombobox(
            method_card.content, label="Algorithm:",
            values=self.METHODS, default="kmeans",
            theme=self.theme_name, label_width=14,
            tooltip="kmeans: fast, auto-k selection\nhierarchical: dendogram-based\nspectral: graph-based"
        )
        self.method_combo.pack(fill=tk.X, pady=4)
        
        self.features_combo = LabeledCombobox(
            method_card.content, label="Features:",
            values=self.FEATURES, default="cooccurrence",
            theme=self.theme_name, label_width=14,
            tooltip="cooccurrence: entities appearing together\ndocuments: which docs contain each entity"
        )
        self.features_combo.pack(fill=tk.X, pady=4)
        
        # Parameters Card
        params_card = Card(self.options_content, title="⚙️ Parameters", theme=self.theme_name)
        params_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Auto k selection
        self.auto_k_var = tk.BooleanVar(value=True)
        auto_k_frame = tk.Frame(params_card.content, bg=self.theme["bg_card"])
        auto_k_frame.pack(fill=tk.X, pady=4)
        
        tk.Checkbutton(
            auto_k_frame, text="Auto-select k (kmeans only)",
            variable=self.auto_k_var, bg=self.theme["bg_card"],
            fg=self.theme["text_primary"], font=FONTS.get_font("body"),
            activebackground=self.theme["bg_card"],
            command=self._on_auto_k_change
        ).pack(side=tk.LEFT)
        
        # Number of clusters
        self.n_clusters_spin = LabeledSpinbox(
            params_card.content, label="Number of Clusters:",
            from_=2, to=30, default=5,
            theme=self.theme_name, label_width=18,
        )
        self.n_clusters_spin.pack(fill=tk.X, pady=4)
        
        # K range for auto selection
        k_range_frame = tk.Frame(params_card.content, bg=self.theme["bg_card"])
        k_range_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            k_range_frame, text="K Range (auto):",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"], width=18, anchor="w"
        ).pack(side=tk.LEFT)
        
        self.k_min_spin = tk.Spinbox(
            k_range_frame, from_=2, to=20, width=5,
            font=FONTS.get_font("body")
        )
        self.k_min_spin.delete(0, tk.END)
        self.k_min_spin.insert(0, "2")
        self.k_min_spin.pack(side=tk.LEFT, padx=2)
        
        tk.Label(
            k_range_frame, text="to",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"]
        ).pack(side=tk.LEFT, padx=4)
        
        self.k_max_spin = tk.Spinbox(
            k_range_frame, from_=3, to=30, width=5,
            font=FONTS.get_font("body")
        )
        self.k_max_spin.delete(0, tk.END)
        self.k_max_spin.insert(0, "10")
        self.k_max_spin.pack(side=tk.LEFT, padx=2)
        
        # Scorer for auto-k
        self.scorer_combo = LabeledCombobox(
            params_card.content, label="Scorer (auto-k):",
            values=self.SCORERS, default="silhouette",
            theme=self.theme_name, label_width=18,
        )
        self.scorer_combo.pack(fill=tk.X, pady=4)
        
        # Run Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Run Clustering", icon="🏷️",
            command=self._run_analysis, theme=self.theme_name,
        ).pack(fill=tk.X)
        
        # Export
        export_card = Card(self.options_content, title="📥 Export", theme=self.theme_name)
        export_card.pack(fill=tk.X, padx=8, pady=8)
        
        tk.Button(
            export_card.content, text="Export Results to Excel",
            command=self._export_results,
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
            font=FONTS.get_font("body"), relief=tk.FLAT, cursor="hand2"
        ).pack(fill=tk.X, pady=4)
        
        # Initialize UI state
        self._on_auto_k_change()
    
    def _on_auto_k_change(self):
        """Handle auto-k checkbox change."""
        auto_k = self.auto_k_var.get()
        method = self.method_combo.get()
        
        if method == "kmeans" and auto_k:
            self.n_clusters_spin.spinbox.config(state="disabled")
        else:
            self.n_clusters_spin.spinbox.config(state="normal")
    
    def _create_results(self):
        """Create results panel."""
        super()._create_results()
        
        # Create notebook with permanent Info tab
        self.results_notebook = ttk.Notebook(self.results_content)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        self.results_tab = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.results_tab, text="📊 Results")
        
        # Info tab (always visible)
        self.info_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.info_frame, text="ℹ️ Info")
        self._create_info_content(self.info_frame)
        
        # Show placeholder in results tab
        self._show_results_placeholder()
    
    def _show_results_placeholder(self):
        """Show placeholder in results tab."""
        # Check if results_tab still exists
        if not hasattr(self, 'results_tab'):
            return
        try:
            if not self.results_tab.winfo_exists():
                return
        except Exception:
            return
        
        for widget in self.results_tab.winfo_children():
            widget.destroy()
        tk.Label(
            self.results_tab,
            text="Click 'Run' to see results here.\nSee Info tab for documentation.",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)
    
    def _show_loading(self, message: str = "Processing..."):
        """Show loading state inside the results tab, preserving notebook structure."""
        self._stop_active_spinners()
        
        # Only clear the results_tab content, not the whole notebook
        if hasattr(self, 'results_tab') and self._widget_exists(self.results_tab):
            try:
                for widget in self.results_tab.winfo_children():
                    try:
                        widget.destroy()
                    except:
                        pass
            except tk.TclError:
                pass
            
            # Create loading indicator inside results_tab
            try:
                from biblium.gui.widgets.progress import LoadingSpinner
                
                loading_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
                loading_frame.pack(expand=True)
                
                spinner = LoadingSpinner(loading_frame, size=32, theme=self.theme_name)
                spinner.pack()
                spinner.start()
                
                # Track active spinner for cleanup
                if not hasattr(self, '_active_spinners'):
                    self._active_spinners = []
                self._active_spinners.append(spinner)
                
                tk.Label(
                    loading_frame,
                    text=message,
                    font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"],
                    fg=self.theme["text_muted"],
                ).pack(pady=(12, 0))
                
                # Switch to results tab to show loading
                if hasattr(self, 'results_notebook'):
                    self.results_notebook.select(0)
            except tk.TclError:
                pass
        else:
            # Fallback to base behavior if notebook structure is missing
            super()._show_loading(message)
    
    def _widget_exists(self, widget) -> bool:
        """Check if a tkinter widget still exists."""
        if widget is None:
            return False
        try:
            return widget.winfo_exists()
        except (tk.TclError, AttributeError):
            return False

    def _show_placeholder(self):
        """Show detailed placeholder with instructions."""
        # Check if results_tab still exists
        if not hasattr(self, 'results_tab'):
            return
        try:
            if not self.results_tab.winfo_exists():
                return
        except Exception:
            return
        
        for widget in self.results_tab.winfo_children():
            widget.destroy()
        
        msg = (
            "🔗 Entity Clustering\n\n"
            "Cluster similar entities together.\n\n"
            "Features:\n"
            "• Author similarity clustering\n"
            "• Journal/source grouping\n"
            "• Keyword co-occurrence clusters\n"
            "• Visual cluster maps\n"
            "\n"
            "Identifies research communities and keyword groups.\n\n"
            "Steps:\n"
            "1. Load a dataset\n"
            "2. Select entity type\n"
            "3. Choose similarity metric\n"
            "4. Click 'Cluster'\n"
        )
        
        tk.Label(
            self.results_tab,
            text=msg,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)

    def _run_analysis(self):
        """Run entity clustering."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        entity_column = self.entity_combo.get()
        
        self._show_loading(f"Preparing {entity_column}...")
        
        def do_analysis():
            try:
                actual_column = entity_column
                
                # Compute derived entities if needed
                if entity_column == "Words from Abstract":
                    actual_column = self._compute_words_column("Abstract", "Processed Abstract")
                elif entity_column == "Words from Title":
                    actual_column = self._compute_words_column("Title", "Processed Title")
                elif entity_column == "Countries":
                    actual_column = self._ensure_countries_column()
                
                if actual_column is None:
                    self.after(0, lambda: self._show_error(f"Could not compute {entity_column}"))
                    return
                
                # Check if column exists
                if actual_column not in self.bib.df.columns:
                    self.after(0, lambda: self._show_error(f"Column '{actual_column}' not found in dataset."))
                    return
                
                print(f"[DEBUG] Starting entity clustering for {actual_column}")
                
                # Build parameters
                method = self.method_combo.get()
                params = {
                    "entity_column": actual_column,
                    "method": method,
                    "features": self.features_combo.get(),
                    "min_occurrence": self.min_occur_spin.get(),
                    "scorer": self.scorer_combo.get(),
                }
                
                # Handle n_clusters
                if method == "kmeans" and self.auto_k_var.get():
                    params["n_clusters"] = None
                    k_min = int(self.k_min_spin.get())
                    k_max = int(self.k_max_spin.get())
                    params["k_range"] = range(k_min, k_max + 1)
                else:
                    params["n_clusters"] = self.n_clusters_spin.get()
                
                print(f"[DEBUG] Entity clustering params: {params}")
                
                # Call biblium's cluster_entities
                if hasattr(self.bib, 'cluster_entities'):
                    result = self.bib.cluster_entities(**params)
                else:
                    # Fallback to utilsbib directly
                    from biblium.utilsbib import cluster_entities
                    result = cluster_entities(self.bib.df, sep=self.bib.default_separator, **params)
                
                print(f"[DEBUG] Entity clustering complete. {len(result['clusters_df'])} entities clustered.")
                
                self._cluster_result = result
                self.after(0, lambda: self._on_success(result))
                
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[DEBUG] Entity clustering error: {e}")
                print(f"[DEBUG] Traceback: {tb}")
                self.after(0, lambda: self._show_error(f"{e}\n\n{tb}"))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _compute_words_column(self, raw_col: str, processed_col: str) -> Optional[str]:
        """
        Compute a words column from text field for clustering.
        Creates a column with words separated by the default separator.
        """
        # Prefer processed column if available
        if processed_col in self.bib.df.columns:
            source_col = processed_col
        elif raw_col in self.bib.df.columns:
            source_col = raw_col
        else:
            return None
        
        target_col = f"Words from {raw_col}"
        
        # Check if already computed
        if target_col in self.bib.df.columns:
            return target_col
        
        print(f"[DEBUG] Computing {target_col} from {source_col}")
        
        sep = self.bib.default_separator
        
        def extract_words(text):
            if pd.isna(text) or not isinstance(text, str):
                return ""
            # Split on whitespace and filter
            words = text.lower().split()
            # Filter: keep words with 3+ chars, only letters
            words = [w for w in words if len(w) >= 3 and w.isalpha()]
            return sep.join(words)
        
        self.bib.df[target_col] = self.bib.df[source_col].apply(extract_words)
        return target_col
    
    def _ensure_countries_column(self) -> Optional[str]:
        """
        Ensure Countries column exists, computing it if necessary.
        """
        # Check various possible country column names
        country_cols = ["Countries", "Country", "CA Country", "Corresponding Author Country"]
        
        for col in country_cols:
            if col in self.bib.df.columns:
                # Check if it has data
                non_empty = self.bib.df[col].dropna()
                if len(non_empty) > 0:
                    return col
        
        # Try to compute from Affiliations
        if "Affiliations" in self.bib.df.columns:
            print("[DEBUG] Computing Countries from Affiliations")
            try:
                from biblium import utilsbib
                # Try to add countries
                if hasattr(utilsbib, 'add_ca_country_df'):
                    self.bib.df = utilsbib.add_ca_country_df(self.bib.df, self.bib.db)
                    # Check what columns were added
                    for col in country_cols:
                        if col in self.bib.df.columns:
                            return col
                
                # Try extract_countries_from_affiliations
                if hasattr(utilsbib, 'extract_countries_from_affiliations'):
                    self.bib.df, _ = utilsbib.extract_countries_from_affiliations(
                        self.bib.df, aff_column="Affiliations"
                    )
                    for col in country_cols:
                        if col in self.bib.df.columns:
                            return col
            except Exception as e:
                print(f"[DEBUG] Error computing countries: {e}")
        
        return None

    def _on_success(self, result):
        """Display clustering results."""
        self._stop_active_spinners()
        
        # Only clear results_tab content, not the whole notebook
        if hasattr(self, 'results_tab') and self._widget_exists(self.results_tab):
            for widget in self.results_tab.winfo_children():
                try:
                    widget.destroy()
                except:
                    pass
        else:
            # Notebook was destroyed, try to recreate it
            self._safe_clear_results()
            self.results_notebook = ttk.Notebook(self.results_content)
            self.results_notebook.pack(fill=tk.BOTH, expand=True)
            self.results_tab = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
            self.results_notebook.add(self.results_tab, text="📊 Results")
        
        clusters_df = result.get("clusters_df")
        
        if clusters_df is None or len(clusters_df) == 0:
            tk.Label(
                self.results_tab, text="Clustering did not produce results.",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"],
            ).pack(expand=True, pady=50)
            return
        
        # Create notebook for results
        notebook = ttk.Notebook(self.results_tab)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Summary
        summary_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(summary_frame, text="Summary")
        self._create_summary_tab(summary_frame, result)
        
        # Tab 2: Cluster Distribution
        if HAS_MATPLOTLIB:
            dist_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(dist_frame, text="Distribution")
            self._create_distribution_plot(dist_frame, result)
        
        # Tab 3: All Entities
        entities_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(entities_frame, text="Entities")
        self._create_entities_tab(entities_frame, result)
        
        # Tab 4: Top Entities by Cluster
        top_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(top_frame, text="Top by Cluster")
        
        # Info tab
        info_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(info_frame, text="ℹ️ Info")
        self._create_info_content(info_frame)
        self._create_top_entities_tab(top_frame, result)
        
        n_clusters = result.get("n_clusters", "?")
        n_entities = len(clusters_df)
        messagebox.showinfo("Complete", f"Clustered {n_entities} entities into {n_clusters} clusters.")
    
    def _create_summary_tab(self, parent, result):
        """Create summary statistics tab."""
        clusters_df = result["clusters_df"]
        
        # Title
        entity_col = result.get("entity_column", "Entities")
        tk.Label(
            parent, text=f"Clustering Results: {entity_col}",
            font=FONTS.get_font("heading2"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"]
        ).pack(pady=(16, 8))
        
        # Stats cards
        n_clusters = result.get("n_clusters", clusters_df["Cluster"].nunique())
        n_entities = len(clusters_df)
        sil_score = result.get("silhouette_score")
        
        grid = CardGrid(parent, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X, padx=16, pady=8)
        
        grid.add_card(StatsCard(grid, "Entities", f"{n_entities:,}", "🏷️", self.theme_name))
        grid.add_card(StatsCard(grid, "Clusters", f"{n_clusters}", "📊", self.theme_name, accent=True))
        grid.add_card(StatsCard(grid, "Avg Size", f"{n_entities / n_clusters:.1f}", "📈", self.theme_name))
        
        if sil_score is not None:
            grid.add_card(StatsCard(grid, "Silhouette", f"{sil_score:.3f}", "✓", self.theme_name))
        else:
            largest = result.get("cluster_sizes", pd.Series()).max() if "cluster_sizes" in result else "?"
            grid.add_card(StatsCard(grid, "Largest", f"{largest}", "📊", self.theme_name))
        
        # Method info
        info_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        info_frame.pack(fill=tk.X, padx=16, pady=8)
        
        method = result.get("method", "kmeans")
        tk.Label(
            info_frame, text=f"Method: {method.upper()}",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_secondary"]
        ).pack(side=tk.LEFT)
        
        # Cluster sizes table
        tk.Label(
            parent, text="Cluster Sizes",
            font=FONTS.get_font("heading3"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"]
        ).pack(pady=(16, 8))
        
        cluster_sizes = result.get("cluster_sizes", clusters_df["Cluster"].value_counts().sort_index())
        sizes_df = pd.DataFrame({
            "Cluster": cluster_sizes.index,
            "Count": cluster_sizes.values,
            "Percentage": (cluster_sizes.values / n_entities * 100).round(1)
        })
        
        table = DataTable(parent, theme=self.theme_name)
        table.pack(fill=tk.X, padx=16, pady=8)
        table.set_data(sizes_df)
    
    def _create_distribution_plot(self, parent, result):
        """Create cluster distribution plot."""
        plot_frame = PlotFrame(parent, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        cluster_sizes = result.get("cluster_sizes", 
                                   result["clusters_df"]["Cluster"].value_counts().sort_index())
        
        fig, ax = plot_frame.get_figure()
        ax.bar(cluster_sizes.index.astype(str), cluster_sizes.values, color="#3b82f6", edgecolor="white")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of Entities")
        
        entity_col = result.get("entity_column", "Entities")
        ax.set_title(f"{entity_col} Distribution Across {len(cluster_sizes)} Clusters")
        
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        plot_frame.refresh()
    
    def _create_entities_tab(self, parent, result):
        """Create entities table tab."""
        clusters_df = result["clusters_df"]
        
        # Cluster filter
        control_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        control_frame.pack(fill=tk.X, padx=8, pady=8)
        
        tk.Label(
            control_frame, text="Filter by Cluster:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"]
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        clusters = sorted(clusters_df["Cluster"].unique())
        cluster_options = ["All"] + [str(c) for c in clusters]
        
        self._entity_filter_var = tk.StringVar(value="All")
        cluster_combo = ttk.Combobox(
            control_frame, textvariable=self._entity_filter_var,
            values=cluster_options, state="readonly", width=10
        )
        cluster_combo.pack(side=tk.LEFT, padx=4)
        
        # Table
        table_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        self._entities_table = DataTable(table_frame, theme=self.theme_name)
        self._entities_table.pack(fill=tk.BOTH, expand=True)
        self._entities_df = clusters_df
        
        # Initial display
        self._update_entities_table()
        
        # Bind filter change
        cluster_combo.bind("<<ComboboxSelected>>", lambda e: self._update_entities_table())
    
    def _update_entities_table(self):
        """Update entities table based on filter."""
        if not hasattr(self, '_entities_df'):
            return
        
        filter_val = self._entity_filter_var.get()
        
        if filter_val == "All":
            filtered = self._entities_df
        else:
            cluster_num = int(filter_val)
            filtered = self._entities_df[self._entities_df["Cluster"] == cluster_num]
        
        self._entities_table.set_data(filtered)
    
    def _create_top_entities_tab(self, parent, result):
        """Create top entities by cluster tab."""
        top_by_cluster = result.get("top_entities_by_cluster", {})
        
        if not top_by_cluster:
            tk.Label(
                parent, text="No top entities data available.",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"]
            ).pack(expand=True, pady=50)
            return
        
        # Create scrollable frame
        canvas = tk.Canvas(parent, bg=self.theme["bg_card"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.theme["bg_card"])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Display top entities for each cluster
        entity_col = result.get("entity_column", "Entities")
        
        for cluster_id in sorted(top_by_cluster.keys()):
            entities = top_by_cluster[cluster_id]
            
            # Cluster header
            header = tk.Frame(scrollable_frame, bg=self.theme["accent_primary"])
            header.pack(fill=tk.X, padx=16, pady=(16, 4))
            
            tk.Label(
                header, text=f"Cluster {cluster_id}",
                font=FONTS.get_font("heading3"),
                bg=self.theme["accent_primary"], fg="white",
                padx=8, pady=4
            ).pack(side=tk.LEFT)
            
            tk.Label(
                header, text=f"({len(entities)} shown)",
                font=FONTS.get_font("body"),
                bg=self.theme["accent_primary"], fg="white",
                padx=8, pady=4
            ).pack(side=tk.RIGHT)
            
            # Entities list
            entities_text = "\n".join([f"  • {e}" for e in entities[:10]])
            
            tk.Label(
                scrollable_frame, text=entities_text,
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                justify=tk.LEFT, anchor="w", padx=16
            ).pack(fill=tk.X, pady=4)
    
    def _export_results(self):
        """Export clustering results to Excel."""
        if self._cluster_result is None:
            messagebox.showwarning("No Results", "Run clustering first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")],
            title="Export Entity Clustering Results"
        )
        
        if not filename:
            return
        
        try:
            result = self._cluster_result
            
            if filename.endswith(".csv"):
                result["clusters_df"].to_csv(filename, index=False)
            else:
                with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
                    result["clusters_df"].to_excel(writer, sheet_name="Entities", index=False)
                    
                    # Cluster sizes
                    sizes = result.get("cluster_sizes", 
                                       result["clusters_df"]["Cluster"].value_counts().sort_index())
                    sizes_df = pd.DataFrame({"Cluster": sizes.index, "Count": sizes.values})
                    sizes_df.to_excel(writer, sheet_name="Cluster Sizes", index=False)
                    
                    # Top entities
                    top_data = []
                    for cluster_id, entities in result.get("top_entities_by_cluster", {}).items():
                        for rank, entity in enumerate(entities, 1):
                            top_data.append({"Cluster": cluster_id, "Rank": rank, "Entity": entity})
                    if top_data:
                        pd.DataFrame(top_data).to_excel(writer, sheet_name="Top Entities", index=False)
            
            messagebox.showinfo("Success", f"Results exported to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {e}")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
ENTITY CLUSTERING
═════════════════

Cluster similar entities (authors, keywords, etc.).

ENTITY TYPES
────────────
• Authors: Research communities
• Keywords: Topic clusters
• Sources: Journal families
• Countries: Regional groups
• Institutions: Collaboration groups

METHODS
───────
• Hierarchical clustering
• K-means on co-occurrence
• Community detection

SIMILARITY MEASURES
───────────────────
• Co-occurrence frequency
• Jaccard similarity
• Cosine similarity
• Association strength

OUTPUT
──────
• Cluster assignments
• Dendrogram
• Cluster characteristics
• Top entities per cluster

VISUALIZATION
─────────────
• Dendrogram tree
• 2D cluster map
• Heatmap of similarities
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

    def set_bib(self, bib):
        """Set the bibliometric data object."""
        self.bib = bib
        self._cluster_result = None
        
        # Update entity column options based on available columns
        if bib and hasattr(bib, 'df'):
            available_cols = []
            
            # Check for direct columns
            direct_cols = [
                "Authors", "Affiliations", "Funding Sponsor",
                "Author Keywords", "Processed Author Keywords",
                "Index Keywords", "Processed Index Keywords",
            ]
            
            for col in direct_cols:
                if col in bib.df.columns:
                    # Check if column has data
                    non_empty = bib.df[col].dropna()
                    if len(non_empty) > 0:
                        available_cols.append(col)
            
            # Check for Countries (various possible names)
            country_cols = ["Countries", "Country", "CA Country", "Corresponding Author Country"]
            country_found = False
            for col in country_cols:
                if col in bib.df.columns and not country_found:
                    non_empty = bib.df[col].dropna()
                    if len(non_empty) > 0:
                        available_cols.append(col)
                        country_found = True
            
            # Add computable columns if source exists
            if not country_found and "Affiliations" in bib.df.columns:
                available_cols.append("Countries")  # Can be computed
            
            if "Abstract" in bib.df.columns or "Processed Abstract" in bib.df.columns:
                available_cols.append("Words from Abstract")
            
            if "Title" in bib.df.columns or "Processed Title" in bib.df.columns:
                available_cols.append("Words from Title")
            
            if available_cols:
                self.entity_combo.combobox['values'] = available_cols
                # Set default
                if "Author Keywords" in available_cols:
                    self.entity_combo.set("Author Keywords")
                elif "Processed Author Keywords" in available_cols:
                    self.entity_combo.set("Processed Author Keywords")
                elif "Authors" in available_cols:
                    self.entity_combo.set("Authors")
                elif available_cols:
                    self.entity_combo.set(available_cols[0])
