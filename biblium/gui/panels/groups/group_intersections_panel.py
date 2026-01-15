# -*- coding: utf-8 -*-
"""
Group Intersections Panel
=========================
Panel for analyzing intersections between document groups.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from typing import Dict, List, Optional, Any

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, CollapsibleCard, StatsCard, CardGrid
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox, LabeledEntry
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.tooltips import ToolTip
from biblium.gui.widgets.plots import PlotFrame

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class GroupIntersectionsPanel(BasePanel):
    """
    Panel for analyzing group intersections.
    
    Shows which documents belong to multiple groups and computes
    overlap statistics between groups.
    """
    
    title = "Group Intersections"
    icon = "🔀"
    description = "Analyze document overlap between groups"
    requires_data = True
    
    @property
    def bib_group(self):
        """Get BiblioGroup instance."""
        if hasattr(self, '_bib_group') and self._bib_group is not None:
            return self._bib_group
        try:
            workspace = self.master.master
            if hasattr(workspace, 'bib_group'):
                return workspace.bib_group
            if hasattr(workspace, '_shared_bib_group'):
                return workspace._shared_bib_group
        except:
            pass
        return None
    
    @bib_group.setter
    def bib_group(self, value):
        self._bib_group = value
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        if not self._check_groups():
            return
        
        # Analysis Options Card
        options_card = Card(
            self.options_content,
            title="📋 Analysis Options",
            theme=self.theme_name
        )
        options_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Include document IDs
        self.include_ids_var = tk.BooleanVar(value=False)
        LabeledCheckbox(
            options_card.content,
            label="Include document IDs in results",
            variable=self.include_ids_var,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
        
        # ID column selection
        id_cols = self._get_id_columns()
        self.id_column_var = tk.StringVar(value=id_cols[0] if id_cols else "")
        
        self.id_column_combo = LabeledCombobox(
            options_card.content,
            label="ID Column:",
            values=id_cols,
            variable=self.id_column_var,
            theme=self.theme_name,
            label_width=12
        )
        self.id_column_combo.pack(fill=tk.X, pady=4)
        
        # Visualization Options Card
        viz_card = Card(
            self.options_content,
            title="📊 Visualization Type",
            theme=self.theme_name
        )
        viz_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Visualization type
        self.viz_type_var = tk.StringVar(value="venn")
        
        viz_types = [
            ("🔵 Venn Diagram", "venn"),
            ("📊 UpSet Plot", "upset"),
            ("🗺️ Heatmap", "heatmap"),
            ("🕸️ Network", "network"),
            ("🌳 Dendrogram", "dendrogram"),
        ]
        
        for text, value in viz_types:
            rb = tk.Radiobutton(
                viz_card.content,
                text=text,
                variable=self.viz_type_var,
                value=value,
                bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
                selectcolor=self.theme["bg_secondary"],
                activebackground=self.theme["bg_card"],
                activeforeground=self.theme["text_primary"],
                font=FONTS.get_font("body"),
            )
            rb.pack(anchor=tk.W, pady=2)
        
        # Network-specific options
        network_frame = tk.Frame(viz_card.content, bg=self.theme["bg_card"])
        network_frame.pack(fill=tk.X, pady=(8, 4))
        
        tk.Label(
            network_frame,
            text="Network/Heatmap Options:",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(anchor=tk.W)
        
        # Similarity method
        self.similarity_var = tk.StringVar(value="jaccard")
        sim_frame = tk.Frame(network_frame, bg=self.theme["bg_card"])
        sim_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(
            sim_frame,
            text="Similarity:",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
            width=10,
            anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        sim_combo = ttk.Combobox(
            sim_frame,
            textvariable=self.similarity_var,
            values=["jaccard", "count", "dice", "overlap"],
            state="readonly",
            width=12,
        )
        sim_combo.pack(side=tk.LEFT, padx=4)
        
        # Threshold
        self.threshold_var = tk.DoubleVar(value=0.1)
        thresh_frame = tk.Frame(network_frame, bg=self.theme["bg_card"])
        thresh_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(
            thresh_frame,
            text="Threshold:",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
            width=10,
            anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        thresh_entry = tk.Entry(
            thresh_frame,
            textvariable=self.threshold_var,
            width=8,
        )
        thresh_entry.pack(side=tk.LEFT, padx=4)
        
        # Action buttons
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame,
            text="Analyze Intersections",
            icon="🔀",
            command=self._run_analysis,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
        
        # Export button
        ThemedButton(
            btn_frame,
            text="Export Results",
            command=self._export_results,
            theme=self.theme_name,
            style="secondary",
        ).pack(fill=tk.X, pady=4)
    
    def _get_id_columns(self) -> List[str]:
        """Get potential ID columns from the dataframe."""
        if not self.bib or self.bib.df is None:
            return []
        
        # Common ID column names
        id_candidates = ["Doc ID", "EID", "DOI", "Title", "index"]
        available = [c for c in id_candidates if c in self.bib.df.columns]
        
        # Add other string columns
        for col in self.bib.df.columns:
            if col not in available and self.bib.df[col].dtype == object:
                available.append(col)
        
        return available[:10]  # Limit to 10 options
    
    def _check_groups(self) -> bool:
        """Check if groups are available."""
        if not self.bib:
            self._show_no_data_message("Please load a dataset first.")
            return False
        
        if not self.bib_group:
            self._show_no_groups_message()
            return False
        
        return True
    
    def _show_no_data_message(self, message: str):
        """Show no data message."""
        tk.Label(
            self.options_content,
            text=f"📂 {message}\n\nGo to DATA → Load Dataset",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)
    
    def _show_no_groups_message(self):
        """Show message when no groups are defined."""
        frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        tk.Label(
            frame,
            text="⚠️ No Groups Defined",
            font=FONTS.get_font("subheading"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_primary"],
        ).pack(pady=(0, 8))
        
        tk.Label(
            frame,
            text="Please create document groups first.\n\n"
                 "Go to GROUPS → Setup Groups to define your groups.",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(pady=(0, 16))
        
        ActionButton(
            frame,
            text="Go to Group Setup",
            icon="⚙️",
            command=lambda: event_bus.emit(EventBus.PANEL_CHANGED, {"panel": "group_setup"}),
            theme=self.theme_name,
        ).pack()
    
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
    
    def _show_loading(self, message: str = "Processing..."):
        """Show loading state within the results tab (not destroying notebook)."""
        from biblium.gui.widgets.progress import LoadingSpinner
        
        self._stop_active_spinners()
        
        try:
            if hasattr(self, 'results_tab') and self.results_tab.winfo_exists():
                for widget in self.results_tab.winfo_children():
                    try:
                        widget.destroy()
                    except:
                        pass
                
                loading_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
                loading_frame.pack(expand=True)
                
                spinner = LoadingSpinner(loading_frame, size=32, theme=self.theme_name)
                spinner.pack()
                spinner.start()
                
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
        except tk.TclError:
            pass
    
    def _show_error(self, message: str):
        """Show error message within the results tab."""
        self._stop_active_spinners()
        
        try:
            if not hasattr(self, 'results_tab') or not self.results_tab.winfo_exists():
                return
                
            for widget in self.results_tab.winfo_children():
                try:
                    widget.destroy()
                except:
                    pass
            
            error_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
            error_frame.pack(expand=True)
            
            tk.Label(error_frame, text="❌", font=("Segoe UI", 32), bg=self.theme["bg_card"]).pack()
            tk.Label(error_frame, text="Error", font=FONTS.get_font("heading", bold=True),
                    bg=self.theme["bg_card"], fg=self.theme["danger"]).pack(pady=(8, 4))
            tk.Label(error_frame, text=message, font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"], fg=self.theme["text_secondary"], wraplength=400).pack()
        except tk.TclError:
            pass
    
    def _show_initial_message(self, message: str = None):
        """Show initial message within the results tab."""
        self._stop_active_spinners()
        
        if message is None:
            message = "Click 'Run' to see results here.\nSee Info tab for documentation."
        
        try:
            if not hasattr(self, 'results_tab') or not self.results_tab.winfo_exists():
                return
                
            for widget in self.results_tab.winfo_children():
                try:
                    widget.destroy()
                except:
                    pass
            
            tk.Label(self.results_tab, text=message, font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"], fg=self.theme["text_muted"], justify=tk.CENTER).pack(expand=True)
        except tk.TclError:
            pass
    
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
            "🔀 Group Intersections\n\n"
            "Analyze overlaps between groups.\n\n"
            "Features:\n"
            "• Venn diagrams\n"
            "• Intersection counts\n"
            "• Jaccard similarity\n"
            "• Unique element identification\n"
            "\n"
            "Shows shared vs unique characteristics.\n\n"
            "Steps:\n"
            "1. Set up groups in Group Setup\n"
            "2. Select groups to compare\n"
            "3. Choose intersection metric\n"
            "4. Click 'Analyze'\n"
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
        """Run the intersections analysis."""
        if not self._check_groups():
            return
        
        viz_type = self.viz_type_var.get()
        self._show_loading(f"Computing group intersections...")
        
        def do_analysis():
            try:
                include_ids = self.include_ids_var.get()
                id_column = self.id_column_var.get()
                
                # Use biblium's implementation to compute intersections
                method = getattr(self.bib_group, "get_group_intersections", None)
                
                if method:
                    result = method(
                        include_ids=include_ids,
                        id_column=id_column if id_column else "Doc ID",
                    )
                else:
                    # Fallback to utilsbib
                    from biblium import utilsbib
                    id_col_data = None
                    if include_ids and id_column and id_column in self.bib_group.df.columns:
                        id_col_data = self.bib_group.df[id_column]
                    
                    result = utilsbib.compute_group_intersections(
                        self.bib_group.group_matrix,
                        include_ids=include_ids,
                        id_column=id_col_data,
                    )
                
                # Store result as attribute
                self.bib_group.group_intersections_df = result
                self.current_result = result
                
                self.after(0, lambda: self._display_results(result, viz_type))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: self._show_error(msg))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _display_results(self, df: pd.DataFrame, viz_type: str):
        """Display intersection results with tabs."""
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
        
        if df is None or df.empty:
            self._show_initial_message("No intersections found.")
            return
        
        # Summary cards
        grid = CardGrid(self.results_tab, columns=3, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 16))
        
        # Total intersections
        n_intersections = len(df)
        grid.add_card(StatsCard(grid, "Intersections", str(n_intersections), "🔀", self.theme_name, accent=True))
        
        # Total documents in any intersection
        total_docs = df["Size"].sum()
        grid.add_card(StatsCard(grid, "Documents", f"{int(total_docs):,}", "📄", self.theme_name))
        
        # Largest intersection
        max_size = df["Size"].max()
        grid.add_card(StatsCard(grid, "Largest", str(int(max_size)), "📊", self.theme_name))
        
        # Create tabbed interface (inside results_tab)
        inner_notebook = ttk.Notebook(self.results_tab)
        inner_notebook.pack(fill=tk.BOTH, expand=True, pady=8)
        
        # --- Table Tab (always available) ---
        table_frame = tk.Frame(inner_notebook, bg=self.theme["bg_card"])
        inner_notebook.add(table_frame, text="📋 Table")
        self._display_table(df, table_frame)
        
        # --- Visualization Tab ---
        viz_frame = tk.Frame(inner_notebook, bg=self.theme["bg_card"])
        viz_labels = {
            "venn": "🔵 Venn",
            "upset": "📊 UpSet",
            "heatmap": "🗺️ Heatmap",
            "network": "🕸️ Network",
        }
        inner_notebook.add(viz_frame, text=viz_labels.get(viz_type, "📊 Chart"))
        
        # Info tab
        info_frame = tk.Frame(inner_notebook, bg=self.theme["bg_card"])
        inner_notebook.add(info_frame, text="ℹ️ Info")
        self._create_info_content(info_frame)
        
        self._display_visualization(viz_type, viz_frame)
    
    def _display_table(self, df: pd.DataFrame, parent: tk.Frame):
        """Display results as table."""
        tk.Label(
            parent,
            text="Group Intersections",
            font=FONTS.get_font("subheading"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(fill=tk.X, pady=(8, 4), padx=8)
        
        # Convert Groups tuple to string for display
        display_df = df.copy()
        display_df["Groups"] = display_df["Groups"].apply(lambda x: " ∩ ".join(x) if isinstance(x, tuple) else str(x))
        
        # If ID column exists and is a list, convert to string
        if "ID" in display_df.columns:
            display_df["ID"] = display_df["ID"].apply(
                lambda x: ", ".join(str(i) for i in x[:5]) + ("..." if len(x) > 5 else "") 
                if isinstance(x, list) else str(x)
            )
        
        table = DataTable(parent, theme=self.theme_name)
        table.pack(fill=tk.BOTH, expand=True, pady=8, padx=8)
        table.set_data(display_df)
        
        # Info about multi-group membership
        multi_group = df[df["Groups"].apply(lambda x: len(x) > 1 if isinstance(x, tuple) else False)]
        if len(multi_group) > 0:
            multi_docs = multi_group["Size"].sum()
            tk.Label(
                parent,
                text=f"ℹ️ {int(multi_docs)} documents belong to multiple groups",
                font=FONTS.get_font("small"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
            ).pack(fill=tk.X, pady=4, padx=8)
    
    def _display_visualization(self, viz_type: str, parent: tk.Frame):
        """Display the selected visualization."""
        if viz_type == "venn":
            self._display_venn(parent)
        elif viz_type == "upset":
            self._display_upset(parent)
        elif viz_type == "heatmap":
            self._display_heatmap(parent)
        elif viz_type == "network":
            self._display_network(parent)
        elif viz_type == "dendrogram":
            self._display_dendrogram(parent)
    
    def _display_venn(self, parent: tk.Frame):
        """Display Venn diagram using biblium's plotbib."""
        if not HAS_MATPLOTLIB:
            self._show_viz_error(parent, "Matplotlib is required for visualizations.")
            return
        
        n_groups = len(self.bib_group.group_matrix.columns)
        if n_groups < 2 or n_groups > 6:
            self._show_viz_error(parent, f"Venn diagram requires 2-6 groups. You have {n_groups} groups.")
            return
        
        try:
            from venn import venn
            
            # Create plot frame
            plot_frame = PlotFrame(parent, theme=self.theme_name, figsize=(8, 8), show_ai_button=True)
            plot_frame.pack(fill=tk.BOTH, expand=True, pady=8, padx=8)
            
            fig, ax = plot_frame.get_figure()
            ax.clear()
            
            # Get group colors
            group_colors = getattr(self.bib_group, 'group_colors', None)
            
            # Prepare sets - use actual document indices
            group_matrix = self.bib_group.group_matrix
            sets_raw = {}
            for col in group_matrix.columns:
                mask = group_matrix[col] == 1
                sets_raw[col] = set(group_matrix.index[mask].tolist())
            
            # Add counts to labels
            labeled_sets = {f"{col} (n={len(val)})": val for col, val in sets_raw.items()}
            
            # Draw venn diagram
            venn(labeled_sets, ax=ax)
            
            # Apply colors if available
            if group_colors and ax.patches:
                for i, (patch, col) in enumerate(zip(ax.patches, group_matrix.columns)):
                    if patch is not None:
                        color = group_colors.get(col, None)
                        if color:
                            patch.set_facecolor(color)
                            patch.set_alpha(0.5)
                            patch.set_edgecolor('black')
                            patch.set_linewidth(1.0)
            
            # Adjust legend size
            legend = ax.get_legend()
            if legend:
                # Make legend text smaller
                for text in legend.get_texts():
                    text.set_fontsize(7)
                # Make legend markers smaller - handle both old and new matplotlib API
                handles = getattr(legend, 'legend_handles', None) or getattr(legend, 'legendHandles', None)
                if handles:
                    for handle in handles:
                        if hasattr(handle, 'set_sizes'):
                            handle.set_sizes([20])
                        elif hasattr(handle, '_sizes'):
                            handle._sizes = [20]
                # Make legend frame smaller and position it
                legend.set_bbox_to_anchor((1.0, 1.0))
                legend.set_loc('upper left')
                # Reduce spacing
                legend._fontsize = 7
                if hasattr(legend, 'handlelength'):
                    legend.handlelength = 1.0
                if hasattr(legend, 'handletextpad'):
                    legend.handletextpad = 0.3
            
            ax.set_title("Group Overlap - Venn Diagram", fontsize=10)
            fig.tight_layout()
            plot_frame.canvas.draw()
            
        except ImportError as e:
            self._show_viz_error(parent, f"The 'venn' package is required. Install via: pip install venn\n{e}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._show_viz_error(parent, f"Error creating Venn diagram: {e}")
    
    def _display_upset(self, parent: tk.Frame):
        """Display UpSet plot using biblium's plotbib."""
        if not HAS_MATPLOTLIB:
            self._show_viz_error(parent, "Matplotlib is required for visualizations.")
            return
        
        try:
            from biblium import plotbib
            from upsetplot import UpSet, from_indicators
            
            # Create plot frame
            plot_frame = PlotFrame(parent, theme=self.theme_name, figsize=(10, 6), show_ai_button=True)
            plot_frame.pack(fill=tk.BOTH, expand=True, pady=8, padx=8)
            
            fig, ax = plot_frame.get_figure()
            fig.clear()
            
            # Prepare data
            data = from_indicators(
                self.bib_group.group_matrix.columns.tolist(),
                self.bib_group.group_matrix.astype(bool)
            )
            
            # Create UpSet plot
            upset = UpSet(data, show_counts=True, sort_categories_by="cardinality")
            upset.plot(fig=fig)
            
            # Apply colors if available
            group_colors = getattr(self.bib_group, 'group_colors', None)
            if group_colors and len(fig.axes) > 0:
                ax_cat = fig.axes[0]
                for bar, col in zip(ax_cat.patches, self.bib_group.group_matrix.columns):
                    color = group_colors.get(col, None)
                    if color:
                        bar.set_facecolor(color)
            
            fig.suptitle("Group Intersections - UpSet Plot")
            plot_frame.canvas.draw()
            
        except ImportError:
            self._show_viz_error(parent, "The 'upsetplot' package is required. Install via: pip install upsetplot")
        except Exception as e:
            self._show_viz_error(parent, f"Error creating UpSet plot: {e}")
    
    def _display_heatmap(self, parent: tk.Frame):
        """Display similarity heatmap using biblium's utilsbib."""
        if not HAS_MATPLOTLIB:
            self._show_viz_error(parent, "Matplotlib is required for visualizations.")
            return
        
        method = self.similarity_var.get()
        method_labels = {
            "jaccard": "Jaccard Index",
            "count": "Shared Documents",
            "dice": "Dice Coefficient",
            "overlap": "Overlap Coefficient"
        }
        
        try:
            from biblium import utilsbib
            
            # Compute similarity matrix using biblium
            matrices = utilsbib.compute_group_similarity_matrices(
                self.bib_group.group_matrix, 
                methods=[method if method in ['jaccard', 'count'] else 'jaccard']
            )
            
            sim_matrix = matrices.get(method if method in ['jaccard', 'count'] else 'jaccard')
            if sim_matrix is None:
                self._show_viz_error(parent, "Could not compute similarity matrix.")
                return
            
            # Create plot frame
            plot_frame = PlotFrame(parent, theme=self.theme_name, figsize=(8, 6), show_ai_button=True)
            plot_frame.pack(fill=tk.BOTH, expand=True, pady=8, padx=8)
            
            fig, ax = plot_frame.get_figure()
            ax.clear()
            
            # Create heatmap without gridlines
            im = ax.imshow(sim_matrix.values, cmap='YlOrRd', aspect='auto', 
                          vmin=0 if method != 'count' else None, 
                          vmax=1 if method != 'count' else None)
            
            # Add labels
            ax.set_xticks(range(len(sim_matrix.columns)))
            ax.set_yticks(range(len(sim_matrix.index)))
            ax.set_xticklabels(sim_matrix.columns, rotation=45, ha='right')
            ax.set_yticklabels(sim_matrix.index)
            
            # Remove gridlines
            ax.grid(False)
            ax.set_axisbelow(False)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(method_labels.get(method, 'Similarity'))
            
            # Add values in cells
            for i in range(len(sim_matrix.index)):
                for j in range(len(sim_matrix.columns)):
                    val = sim_matrix.iloc[i, j]
                    if method == 'count':
                        text = f'{int(val)}'
                        color = 'white' if val > sim_matrix.values.max() * 0.5 else 'black'
                    else:
                        text = f'{val:.2f}'
                        color = 'white' if val > 0.5 else 'black'
                    ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)
            
            ax.set_title(f"Group Similarity - {method_labels.get(method, method)}")
            fig.tight_layout()
            plot_frame.canvas.draw()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._show_viz_error(parent, f"Error creating heatmap: {e}")
    
    def _display_network(self, parent: tk.Frame):
        """Display intersection network."""
        if not HAS_MATPLOTLIB:
            self._show_viz_error(parent, "Matplotlib is required for visualizations.")
            return
        
        method = self.similarity_var.get()
        threshold = self.threshold_var.get()
        
        try:
            import networkx as nx
            
            # Create plot frame
            plot_frame = PlotFrame(parent, theme=self.theme_name, figsize=(10, 8), show_ai_button=True)
            plot_frame.pack(fill=tk.BOTH, expand=True, pady=8, padx=8)
            
            fig, ax = plot_frame.get_figure()
            ax.clear()
            
            # Get group colors
            group_colors = getattr(self.bib_group, 'group_colors', None)
            
            # Create network directly in our axes
            self._create_network_in_ax(ax, method, threshold, group_colors)
            
            fig.tight_layout()
            plot_frame.canvas.draw()
            
        except ImportError:
            self._show_viz_error(parent, "NetworkX is required. Install via: pip install networkx")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._show_viz_error(parent, f"Error creating network: {e}")
    
    def _create_network_in_ax(self, ax, method: str, threshold: float, group_colors: dict = None):
        """Create network visualization in given axes."""
        import networkx as nx
        
        group_matrix = self.bib_group.group_matrix
        groups = group_matrix.columns.tolist()
        n_groups = len(groups)
        
        # Compute group sizes (as integers)
        group_sizes = {g: int(group_matrix[g].sum()) for g in groups}
        
        # Compute pairwise similarities
        similarity_matrix = pd.DataFrame(index=groups, columns=groups, dtype=float)
        
        for i, g1 in enumerate(groups):
            set1 = set(group_matrix.index[group_matrix[g1] == 1])
            for j, g2 in enumerate(groups):
                if i == j:
                    similarity_matrix.loc[g1, g2] = 1.0
                    continue
                
                set2 = set(group_matrix.index[group_matrix[g2] == 1])
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                
                if method == "jaccard":
                    sim = intersection / union if union > 0 else 0
                elif method == "count":
                    sim = intersection
                elif method == "dice":
                    sim = 2 * intersection / (len(set1) + len(set2)) if (len(set1) + len(set2)) > 0 else 0
                elif method == "overlap":
                    sim = intersection / min(len(set1), len(set2)) if min(len(set1), len(set2)) > 0 else 0
                else:
                    sim = intersection / union if union > 0 else 0
                
                similarity_matrix.loc[g1, g2] = sim
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for g in groups:
            G.add_node(g, size=group_sizes[g])
        
        # Add edges above threshold
        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups):
                if i < j:
                    sim = similarity_matrix.loc[g1, g2]
                    if sim > threshold:
                        G.add_edge(g1, g2, weight=sim)
        
        # Layout - increase k for better spacing
        if n_groups > 1:
            pos = nx.spring_layout(G, k=3.0/np.sqrt(n_groups), iterations=100, seed=42)
        else:
            pos = {g: (0.5, 0.5) for g in groups}
        
        # Node colors
        if group_colors:
            node_colors = [group_colors.get(g, "#1f77b4") for g in G.nodes()]
        else:
            cmap = plt.cm.tab10
            node_colors = [cmap(i % 10) for i in range(len(G.nodes()))]
        
        # Node sizes - make them larger
        max_size = max(group_sizes.values()) if group_sizes else 1
        node_sizes = [1500 * (group_sizes[g] / max_size + 0.3) for g in G.nodes()]
        
        # Edge widths
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        if edge_weights:
            if method == "count":
                max_weight = max(edge_weights) if edge_weights else 1
                edge_widths = [5.0 * (w / max_weight) for w in edge_weights]
            else:
                edge_widths = [5.0 * w for w in edge_weights]
        else:
            edge_widths = []
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                               node_size=node_sizes, alpha=0.8)
        
        # Draw edges
        if G.edges():
            nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, 
                                   alpha=0.6, edge_color="gray")
        
        # Labels with group names and sizes
        labels = {g: f"{g}\n(n={group_sizes[g]})" for g in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10, font_weight='bold')
        
        # Edge labels
        if G.edges():
            edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=8, alpha=0.8)
        
        method_labels = {
            "jaccard": "Jaccard Similarity",
            "count": "Shared Documents", 
            "dice": "Dice Coefficient",
            "overlap": "Overlap Coefficient"
        }
        ax.set_title(f"Group Intersection Network ({method_labels.get(method, method)})")
        ax.axis("off")
    
    def _display_dendrogram(self, parent: tk.Frame):
        """Display hierarchical clustering dendrogram of groups using biblium's style."""
        if not HAS_MATPLOTLIB:
            self._show_viz_error(parent, "Matplotlib is required for visualizations.")
            return
        
        try:
            from scipy.cluster.hierarchy import dendrogram, linkage
            from scipy.spatial.distance import pdist
            
            # Create plot frame
            plot_frame = PlotFrame(parent, theme=self.theme_name, figsize=(10, 6), show_ai_button=True)
            plot_frame.pack(fill=tk.BOTH, expand=True, pady=8, padx=8)
            
            fig, ax = plot_frame.get_figure()
            ax.clear()
            
            # Get group matrix
            group_matrix = self.bib_group.group_matrix
            groups = list(group_matrix.columns)
            n_groups = len(groups)
            
            if n_groups < 2:
                self._show_viz_error(parent, "Need at least 2 groups for dendrogram.")
                return
            
            # Get group colors
            group_colors = getattr(self.bib_group, 'group_colors', None) or {}
            
            # Ensure labels are strings
            group_matrix.columns = group_matrix.columns.astype(str)
            
            # Compute pairwise distances between columns (groups) - using jaccard
            dist_matrix = pdist(group_matrix.T.values, metric='jaccard')
            Z = linkage(dist_matrix, method='average')
            
            # Plot dendrogram
            dendro = dendrogram(
                Z, 
                labels=group_matrix.columns.tolist(), 
                leaf_rotation=45, 
                leaf_font_size=9, 
                ax=ax,
                above_threshold_color='gray'
            )
            
            # Color the labels according to group colors
            xlbls = ax.get_xmajorticklabels()
            for lbl in xlbls:
                group_name = lbl.get_text()
                color = group_colors.get(group_name, 'black')
                lbl.set_color(color)
                lbl.set_fontweight('bold')
            
            # Style - no gridlines, clean look
            ax.set_facecolor('white')
            ax.grid(False)
            ax.set_ylabel("Jaccard Distance", fontsize=10)
            ax.set_title("Group Similarity Dendrogram", fontsize=11)
            
            # Remove spines except left and bottom
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            fig.tight_layout()
            plot_frame.canvas.draw()
            
        except ImportError as e:
            self._show_viz_error(parent, f"scipy is required for dendrogram: {e}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._show_viz_error(parent, f"Error creating dendrogram: {e}")
    
    def _show_viz_error(self, parent: tk.Frame, message: str):
        """Show visualization error message."""
        tk.Label(
            parent,
            text=f"⚠️ {message}",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme.get("warning", "#f59e0b"),
            wraplength=400,
            justify=tk.CENTER,
        ).pack(expand=True, pady=20)
    
    def _export_results(self):
        """Export results to file."""
        if not hasattr(self, 'current_result') or self.current_result is None:
            messagebox.showwarning("No Results", "Please compute intersections first.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Intersections",
            defaultextension=".xlsx",
            filetypes=[
                ("Excel Files", "*.xlsx"),
                ("CSV Files", "*.csv"),
            ]
        )
        
        if filename:
            try:
                # Convert for export
                export_df = self.current_result.copy()
                export_df["Groups"] = export_df["Groups"].apply(
                    lambda x: " | ".join(x) if isinstance(x, tuple) else str(x)
                )
                if "ID" in export_df.columns:
                    export_df["ID"] = export_df["ID"].apply(
                        lambda x: "; ".join(str(i) for i in x) if isinstance(x, list) else str(x)
                    )
                
                if filename.endswith('.csv'):
                    export_df.to_csv(filename, index=False)
                else:
                    export_df.to_excel(filename, index=False)
                
                messagebox.showinfo("Success", f"Results exported to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
GROUP INTERSECTIONS
═══════════════════

Analyze overlaps between groups.

METRICS
───────
• Intersection: |A ∩ B|
  Documents in both groups
  
• Union: |A ∪ B|
  Documents in either group
  
• Jaccard Index: |A ∩ B| / |A ∪ B|
  Similarity measure (0-1)
  
• Overlap Coefficient
  |A ∩ B| / min(|A|, |B|)

VISUALIZATION
─────────────
• Venn diagram (2-3 groups)
• UpSet plot (many groups)
• Heatmap (pairwise)

JACCARD INTERPRETATION
──────────────────────
• 0: No overlap (disjoint)
• 0.1-0.3: Low overlap
• 0.3-0.6: Moderate
• 0.6+: High overlap
• 1: Identical sets

OUTPUT
──────
• Intersection counts
• Unique to each group
• Shared across all
• Pairwise similarity matrix
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

    def refresh(self):
        """Refresh panel when bib_group changes."""
        for widget in self.options_content.winfo_children():
            widget.destroy()
        self._create_options()
        
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
        self._show_initial_message()
