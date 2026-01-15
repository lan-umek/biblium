"""
Document Clustering Panel
=========================

Cluster documents using biblium's cluster_documents method.

Supports:
- K-Means clustering with auto or manual k selection
- Hierarchical clustering
- Coupling-based clustering (bibliographic coupling)
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


class DocumentClusteringPanel(BasePanel):
    """
    Panel for document clustering analysis.
    
    Uses biblium's cluster_documents method which supports:
    - K-Means: Vector space clustering with optional auto-k selection
    - Hierarchical: Agglomerative clustering 
    - Coupling: Bibliographic coupling based clustering
    """
    
    title = "Document Clustering"
    icon = "🔬"
    description = "Cluster documents by content similarity or bibliographic coupling"
    requires_data = True
    
    METHODS = ["kmeans", "hierarchical", "coupling"]
    TEXT_FIELDS = [
        "Abstract", "Processed Abstract", "Title", "Processed Title",
        "Keywords", "Author Keywords", "Processed Author Keywords",
        "Index Keywords", "Processed Index Keywords",
        "Combined Text", "Processed Combined Text"
    ]
    VECTORIZERS = ["tfidf", "count"]
    SCORERS = ["silhouette", "calinski"]
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._cluster_result = None
        self._matrix_df = None
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Method Card
        method_card = Card(self.options_content, title="📊 Clustering Method", theme=self.theme_name)
        method_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.method_combo = LabeledCombobox(
            method_card.content, label="Algorithm:",
            values=self.METHODS, default="kmeans",
            theme=self.theme_name, label_width=14,
            tooltip="kmeans: fast, good for spherical clusters\nhierarchical: dendogram-based\ncoupling: uses references/citations"
        )
        self.method_combo.pack(fill=tk.X, pady=4)
        self.method_combo.combobox.bind("<<ComboboxSelected>>", self._on_method_change)
        
        # Text Field (for kmeans/hierarchical)
        self.text_field_combo = LabeledCombobox(
            method_card.content, label="Text Field:",
            values=self.TEXT_FIELDS, default="Abstract",
            theme=self.theme_name, label_width=14,
            tooltip="Field to use for document vectorization"
        )
        self.text_field_combo.pack(fill=tk.X, pady=4)
        
        # Vectorizer
        self.vectorizer_combo = LabeledCombobox(
            method_card.content, label="Vectorizer:",
            values=self.VECTORIZERS, default="tfidf",
            theme=self.theme_name, label_width=14,
            tooltip="TF-IDF recommended for most cases"
        )
        self.vectorizer_combo.pack(fill=tk.X, pady=4)
        
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
            tooltip="For hierarchical: required\nFor kmeans: used if auto-k disabled\nFor coupling: auto if not set"
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
            tooltip="Metric for selecting optimal k"
        )
        self.scorer_combo.pack(fill=tk.X, pady=4)
        
        # Coupling fields (for coupling method)
        coupling_card = Card(self.options_content, title="🔗 Coupling Options", theme=self.theme_name)
        coupling_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.coupling_field_combo = LabeledCombobox(
            coupling_card.content, label="Coupling Field:",
            values=["References", "Cited by", "Authors", "Keywords"],
            default="References",
            theme=self.theme_name, label_width=14,
            tooltip="Field to use for bibliographic coupling"
        )
        self.coupling_field_combo.pack(fill=tk.X, pady=4)
        
        # Run Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Run Clustering", icon="🔬",
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
        self._on_method_change()
        self._on_auto_k_change()
    
    def _on_method_change(self, event=None):
        """Handle method selection change."""
        method = self.method_combo.get()
        
        # Show/hide relevant options based on method
        is_coupling = method == "coupling"
        is_kmeans = method == "kmeans"
        
        # Text field and vectorizer only for vector methods
        # (coupling uses bibliographic data)
    
    def _on_auto_k_change(self):
        """Handle auto-k checkbox change."""
        auto_k = self.auto_k_var.get()
        method = self.method_combo.get()
        
        # Only enable auto-k for kmeans
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
            "🔬 Document Clustering\n\n"
            "Group similar documents using machine learning.\n\n"
            "Features:\n"
            "• Hierarchical clustering with dendrograms\n"
            "• K-means clustering\n"
            "• DBSCAN density-based clustering\n"
            "• Silhouette scores for quality\n"
            "• 2D/3D visualization\n"
            "\n"
            "Identifies research themes without predefined categories.\n\n"
            "Steps:\n"
            "1. Load a dataset\n"
            "2. Select clustering algorithm\n"
            "3. Set number of clusters\n"
            "4. Choose features\n"
            "5. Click 'Run Clustering'\n"
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
        """Run document clustering."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        method = self.method_combo.get()
        text_field = self.text_field_combo.get()
        
        # Check if text field exists for vector methods
        if method in ["kmeans", "hierarchical"]:
            # Try to find the text field
            available_fields = self.bib.df.columns.tolist()
            if text_field not in available_fields:
                # Try alternatives
                alternatives = ["Processed Abstract", "Abstract", "Title"]
                found = False
                for alt in alternatives:
                    if alt in available_fields:
                        text_field = alt
                        found = True
                        break
                if not found:
                    messagebox.showerror("Error", f"Text field '{text_field}' not found in dataset.")
                    return
        
        self._show_loading(f"Running {method} clustering...")
        
        def do_analysis():
            try:
                print(f"[DEBUG] Starting clustering with method={method}")
                
                # Build parameters
                params = {
                    "text_field": text_field,
                    "method": method,
                    "vectorizer": self.vectorizer_combo.get(),
                }
                
                # Handle n_clusters
                if method == "kmeans" and self.auto_k_var.get():
                    params["n_clusters"] = None
                    k_min = int(self.k_min_spin.get())
                    k_max = int(self.k_max_spin.get())
                    params["k_range"] = range(k_min, k_max + 1)
                    params["scorer"] = self.scorer_combo.get()
                elif method == "hierarchical":
                    params["n_clusters"] = self.n_clusters_spin.get()
                else:
                    n_clusters = self.n_clusters_spin.get()
                    if n_clusters and n_clusters > 0:
                        params["n_clusters"] = n_clusters
                
                # Coupling specific
                if method == "coupling":
                    coupling_field = self.coupling_field_combo.get()
                    params["coupling_fields"] = [coupling_field]
                    # Remove text-related params
                    params.pop("text_field", None)
                    params.pop("vectorizer", None)
                
                print(f"[DEBUG] Clustering params: {params}")
                
                # Call biblium's cluster_documents
                if hasattr(self.bib, 'cluster_documents'):
                    # Use the bib method which saves results as attributes
                    self.bib.cluster_documents(**params)
                    
                    # Get results from attributes
                    cluster_col = f"{method}_cluster"
                    if cluster_col in self.bib.df.columns:
                        result_df = self.bib.df.copy()
                        matrix_df = getattr(self.bib, 'cluster_matrix_df', None)
                    else:
                        raise ValueError(f"Clustering did not produce expected column: {cluster_col}")
                else:
                    # Fallback to utilsbib directly
                    from biblium.utilsbib import cluster_documents
                    result_df, matrix_df = cluster_documents(self.bib.df, **params)
                
                print(f"[DEBUG] Clustering complete. Result shape: {result_df.shape}")
                
                self._cluster_result = result_df
                self._matrix_df = matrix_df
                
                self.after(0, lambda: self._on_success(result_df, method))
                
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[DEBUG] Clustering error: {e}")
                print(f"[DEBUG] Traceback: {tb}")
                self.after(0, lambda: self._show_error(f"{e}\n\n{tb}"))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _on_success(self, df, method):
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
        
        cluster_col = f"{method}_cluster"
        
        if df is None or cluster_col not in df.columns:
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
        self._create_summary_tab(summary_frame, df, cluster_col)
        
        # Tab 2: Cluster Distribution
        if HAS_MATPLOTLIB:
            dist_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(dist_frame, text="Distribution")
            self._create_distribution_plot(dist_frame, df, cluster_col)
        
        # Tab 3: Documents by Cluster
        docs_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(docs_frame, text="Documents")
        self._create_documents_tab(docs_frame, df, cluster_col)
        
        # Tab 4: Cluster Profiles (top terms per cluster if available)
        if self._matrix_df is not None:
            profile_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(profile_frame, text="Profiles")

            

            # Info tab

            info_frame = tk.Frame(notebook, bg=self.theme["bg_card"])

            notebook.add(info_frame, text="ℹ️ Info")

            self._create_info_content(info_frame)
            self._create_profiles_tab(profile_frame, df, cluster_col)
        
        messagebox.showinfo("Complete", f"Clustering complete: {df[cluster_col].nunique()} clusters found.")
    
    def _create_summary_tab(self, parent, df, cluster_col):
        """Create summary statistics tab."""
        # Title
        tk.Label(
            parent, text="Clustering Results Summary",
            font=FONTS.get_font("heading2"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"]
        ).pack(pady=(16, 8))
        
        # Stats cards
        cluster_counts = df[cluster_col].value_counts().sort_index()
        n_clusters = len(cluster_counts)
        
        grid = CardGrid(parent, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X, padx=16, pady=8)
        
        grid.add_card(StatsCard(grid, "Documents", f"{len(df):,}", "📄", self.theme_name))
        grid.add_card(StatsCard(grid, "Clusters", f"{n_clusters}", "🔬", self.theme_name, accent=True))
        grid.add_card(StatsCard(grid, "Avg Size", f"{len(df) / n_clusters:.1f}", "📊", self.theme_name))
        grid.add_card(StatsCard(grid, "Largest", f"{cluster_counts.max():,}", "📈", self.theme_name))
        
        # Cluster sizes table
        tk.Label(
            parent, text="Cluster Sizes",
            font=FONTS.get_font("heading3"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"]
        ).pack(pady=(16, 8))
        
        sizes_df = pd.DataFrame({
            "Cluster": cluster_counts.index,
            "Count": cluster_counts.values,
            "Percentage": (cluster_counts.values / len(df) * 100).round(1)
        })
        
        table = DataTable(parent, theme=self.theme_name)
        table.pack(fill=tk.X, padx=16, pady=8)
        table.set_data(sizes_df)
    
    def _create_distribution_plot(self, parent, df, cluster_col):
        """Create cluster distribution plot."""
        plot_frame = PlotFrame(parent, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        cluster_counts = df[cluster_col].value_counts().sort_index()
        
        fig, ax = plot_frame.get_figure()
        ax.bar(cluster_counts.index.astype(str), cluster_counts.values, color="#3b82f6", edgecolor="white")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of Documents")
        ax.set_title(f"Document Distribution Across {len(cluster_counts)} Clusters")
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        plot_frame.refresh()
    
    def _create_documents_tab(self, parent, df, cluster_col):
        """Create documents by cluster tab."""
        # Cluster selector
        control_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        control_frame.pack(fill=tk.X, padx=8, pady=8)
        
        tk.Label(
            control_frame, text="Select Cluster:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"]
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        clusters = sorted(df[cluster_col].unique())
        cluster_options = ["All"] + [str(c) for c in clusters]
        
        self._cluster_filter_var = tk.StringVar(value="All")
        cluster_combo = ttk.Combobox(
            control_frame, textvariable=self._cluster_filter_var,
            values=cluster_options, state="readonly", width=10
        )
        cluster_combo.pack(side=tk.LEFT, padx=4)
        
        # Table frame
        table_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Get display columns
        title_col = self.bib.mapping.get("Title", "Title") if self.bib else "Title"
        year_col = self.bib.mapping.get("Year", "Year") if self.bib else "Year"
        
        display_cols = [cluster_col]
        for col in [title_col, year_col, "Authors", "Cited by"]:
            if col in df.columns:
                display_cols.append(col)
        
        self._docs_table = DataTable(table_frame, theme=self.theme_name)
        self._docs_table.pack(fill=tk.BOTH, expand=True)
        self._docs_df = df
        self._docs_display_cols = display_cols
        self._cluster_col = cluster_col
        
        # Initial display
        self._update_docs_table()
        
        # Bind selection change
        cluster_combo.bind("<<ComboboxSelected>>", lambda e: self._update_docs_table())
    
    def _update_docs_table(self):
        """Update documents table based on cluster filter."""
        if not hasattr(self, '_docs_df'):
            return
        
        filter_val = self._cluster_filter_var.get()
        
        if filter_val == "All":
            filtered = self._docs_df
        else:
            cluster_num = int(filter_val)
            filtered = self._docs_df[self._docs_df[self._cluster_col] == cluster_num]
        
        display_df = filtered[self._docs_display_cols].copy()
        self._docs_table.set_data(display_df)
    
    def _create_profiles_tab(self, parent, df, cluster_col):
        """Create cluster profiles tab showing top terms per cluster."""
        if self._matrix_df is None:
            tk.Label(
                parent, text="Cluster profiles not available (no term matrix)",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"]
            ).pack(expand=True, pady=50)
            return
        
        tk.Label(
            parent, text="Top Terms by Cluster",
            font=FONTS.get_font("heading2"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"]
        ).pack(pady=(16, 8))
        
        # Calculate top terms per cluster
        try:
            matrix = self._matrix_df
            clusters = sorted(df[cluster_col].unique())
            
            profiles = []
            for cluster_id in clusters:
                cluster_docs = df[df[cluster_col] == cluster_id].index
                
                # Get mean term weights for this cluster
                if hasattr(matrix, 'sparse'):
                    # Sparse DataFrame
                    cluster_matrix = matrix.loc[cluster_docs]
                    mean_weights = cluster_matrix.mean(axis=0)
                else:
                    cluster_matrix = matrix.loc[cluster_docs]
                    mean_weights = cluster_matrix.mean(axis=0)
                
                # Get top 10 terms
                top_terms = mean_weights.nlargest(10)
                
                for term, weight in top_terms.items():
                    profiles.append({
                        "Cluster": cluster_id,
                        "Term": str(term),
                        "Weight": round(float(weight), 4)
                    })
            
            profiles_df = pd.DataFrame(profiles)
            
            table = DataTable(parent, theme=self.theme_name)
            table.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)
            table.set_data(profiles_df)
            
        except Exception as e:
            tk.Label(
                parent, text=f"Could not generate profiles: {e}",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"]
            ).pack(expand=True, pady=50)
    
    def _export_results(self):
        """Export clustering results to Excel."""
        if self._cluster_result is None:
            messagebox.showwarning("No Results", "Run clustering first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")],
            title="Export Clustering Results"
        )
        
        if not filename:
            return
        
        try:
            if filename.endswith(".csv"):
                self._cluster_result.to_csv(filename, index=False)
            else:
                with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
                    self._cluster_result.to_excel(writer, sheet_name="Documents", index=False)
                    
                    # Add cluster sizes
                    method = self.method_combo.get()
                    cluster_col = f"{method}_cluster"
                    if cluster_col in self._cluster_result.columns:
                        sizes = self._cluster_result[cluster_col].value_counts().sort_index()
                        sizes_df = pd.DataFrame({"Cluster": sizes.index, "Count": sizes.values})
                        sizes_df.to_excel(writer, sheet_name="Cluster Sizes", index=False)
                    
                    # Add matrix if available
                    if self._matrix_df is not None and self._matrix_df.shape[1] < 1000:
                        try:
                            self._matrix_df.to_excel(writer, sheet_name="Term Matrix")
                        except:
                            pass
            
            messagebox.showinfo("Success", f"Results exported to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {e}")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
DOCUMENT CLUSTERING
═══════════════════

Group similar documents using machine learning.

CLUSTERING METHODS
──────────────────
• K-Means
  - Partitions into K clusters
  - Uses text vector similarity
  - Fast, scalable
  - Requires specifying K
  
• Hierarchical
  - Agglomerative clustering
  - Creates dendrogram tree
  - Cut at desired level
  - No K required upfront
  
• Coupling-based
  - Uses bibliographic coupling
  - Documents sharing references
  - Citation-based similarity
  - Good for citation analysis

AUTO-K SELECTION
────────────────
• Elbow method: Plot inertia vs K
• Silhouette analysis: Cluster quality
• Calinski-Harabasz: Variance ratio

QUALITY METRICS
───────────────
• Silhouette Score (-1 to +1)
  - >0.7: Excellent clustering
  - 0.5-0.7: Good clustering
  - 0.25-0.5: Fair clustering
  - <0.25: Poor clustering
  
• Calinski-Harabasz Index
  - Higher = better defined clusters

FEATURES USED
─────────────
• TF-IDF of abstracts/titles
• Keywords
• Reference patterns (coupling)

OUTPUT
──────
• Cluster assignments
• Cluster sizes
• Top terms per cluster
• 2D visualization (PCA/t-SNE)
• Dendrogram (hierarchical)
• Quality metrics
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
        self._matrix_df = None
        
        # Update text field options based on available columns
        if bib and hasattr(bib, 'df'):
            available_fields = []
            for field in self.TEXT_FIELDS:
                if field in bib.df.columns:
                    available_fields.append(field)
            
            if available_fields:
                self.text_field_combo.combobox['values'] = available_fields
                self.text_field_combo.set(available_fields[0])
