# -*- coding: utf-8 -*-
"""
Counts Panel
============
Panel for counting entities with statistics and visualizations.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from typing import Dict, List, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme, ENTITY_TYPES
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, CollapsibleCard, StatsCard, CardGrid
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox, LabeledSpinbox
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.plots import PlotFrame

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class CountsPanel(BasePanel):
    """Panel for counting entities and computing statistics."""
    
    title = "Entity Counts"
    icon = "📊"
    description = "Count and analyze entities like sources, authors, keywords, and more"
    requires_data = True
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._current_entity = None
        self._counts_df = None
        self._stats_df = None
        
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._run_single_count  # Set primary action for toolbar Run button
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Entity Selection Card
        entity_card = Card(self.options_content, title="📊 Entity Selection", theme=self.theme_name)
        entity_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Entity type dropdown
        entity_names = [ENTITY_TYPES[k]["label"] for k in ENTITY_TYPES]
        self.entity_combo = LabeledCombobox(
            entity_card.content,
            label="Entity Type:",
            values=entity_names,
            default="Sources",
            theme=self.theme_name,
            label_width=15,
        )
        self.entity_combo.pack(fill=tk.X, pady=4)
        
        # Entity checkboxes for batch counting
        batch_card = CollapsibleCard(
            self.options_content,
            title="Batch Count Multiple Entities",
            collapsed=True,
            theme=self.theme_name,
        )
        batch_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.entity_checkboxes = {}
        for key, config in ENTITY_TYPES.items():
            cb = LabeledCheckbox(
                batch_card.content,
                label=f"{config['icon']} {config['label']}",
                default=key in ["sources", "authors", "author_keywords"],
                theme=self.theme_name,
            )
            cb.pack(fill=tk.X, pady=1)
            self.entity_checkboxes[key] = cb
        
        # Count Options Card
        options_card = Card(self.options_content, title="⚙️ Options", theme=self.theme_name)
        options_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.top_n_spin = LabeledSpinbox(
            options_card.content,
            label="Top N:",
            from_=5,
            to=500,
            default=50,
            theme=self.theme_name,
            label_width=15,
        )
        self.top_n_spin.pack(fill=tk.X, pady=4)
        
        self.min_count_spin = LabeledSpinbox(
            options_card.content,
            label="Minimum Count:",
            from_=1,
            to=100,
            default=1,
            theme=self.theme_name,
            label_width=15,
        )
        self.min_count_spin.pack(fill=tk.X, pady=4)
        
        self.generate_plot_cb = LabeledCheckbox(
            options_card.content,
            label="Generate bar plot",
            default=True,
            theme=self.theme_name,
        )
        self.generate_plot_cb.pack(fill=tk.X, pady=4)
        
        # Advanced Options
        advanced_card = CollapsibleCard(
            self.options_content,
            title="Advanced Options",
            collapsed=True,
            theme=self.theme_name,
        )
        advanced_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.normalize_cb = LabeledCheckbox(
            advanced_card.content,
            label="Normalize counts (fractions)",
            default=False,
            theme=self.theme_name,
        )
        self.normalize_cb.pack(fill=tk.X, pady=2)
        
        self.cumulative_cb = LabeledCheckbox(
            advanced_card.content,
            label="Show cumulative percentage",
            default=True,
            theme=self.theme_name,
        )
        self.cumulative_cb.pack(fill=tk.X, pady=2)
        
        self.growth_cb = LabeledCheckbox(
            advanced_card.content,
            label="Compute growth rates",
            default=False,
            theme=self.theme_name,
        )
        self.growth_cb.pack(fill=tk.X, pady=2)
        
        # N-gram Options Card
        ngram_card = CollapsibleCard(
            self.options_content,
            title="📝 N-gram Options",
            collapsed=True,
            theme=self.theme_name,
        )
        ngram_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.ngram_min_spin = LabeledSpinbox(
            ngram_card.content,
            label="Min n-gram:",
            from_=1,
            to=5,
            default=1,
            theme=self.theme_name,
            label_width=12,
        )
        self.ngram_min_spin.pack(fill=tk.X, pady=4)
        
        self.ngram_max_spin = LabeledSpinbox(
            ngram_card.content,
            label="Max n-gram:",
            from_=1,
            to=5,
            default=2,
            theme=self.theme_name,
            label_width=12,
        )
        self.ngram_max_spin.pack(fill=tk.X, pady=4)
        
        ngram_note = tk.Label(
            ngram_card.content,
            text="N-gram range: 1=unigrams, 2=bigrams, 3=trigrams.\nOnly applies to N-gram entity types.",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            wraplength=250,
            justify=tk.LEFT,
        )
        ngram_note.pack(fill=tk.X, pady=(4, 0))
        
        # Run buttons
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        self.run_btn = ActionButton(
            btn_frame,
            text="Count Selected Entity",
            icon="▶",
            command=self._run_single_count,
            theme=self.theme_name,
        )
        self.run_btn.pack(fill=tk.X)
        
        ThemedButton(
            btn_frame,
            text="Count All Checked",
            style="secondary",
            command=self._run_batch_count,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=(8, 0))
    
    def _create_results(self):
        """Create the results panel with tabs for table and plot."""
        # Results container
        self.results_card = tk.Frame(
            self.results_frame,
            bg=self.theme["bg_card"],
            highlightbackground=self.theme["border"],
            highlightthickness=1,
        )
        self.results_card.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        # Header
        header = tk.Frame(self.results_card, bg=self.theme["bg_card"])
        header.pack(fill=tk.X)
        
        tk.Label(
            header,
            text="Results",
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            padx=12,
            pady=8,
        ).pack(side=tk.LEFT)
        
        # Export button
        ThemedButton(
            header,
            text="📥 Export",
            style="ghost",
            size="small",
            command=self._export_results,
            theme=self.theme_name,
        ).pack(side=tk.RIGHT, padx=8, pady=4)
        
        ttk.Separator(self.results_card, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12)
        
        # Notebook for tabs
        self.results_notebook = ttk.Notebook(self.results_card)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Table tab
        self.table_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.table_frame, text="  📊 Table  ")
        
        # Plot tab
        self.plot_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.plot_frame, text="  📈 Bar Chart  ")
        
        # Info tab
        info_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(info_frame, text="ℹ️ Info")
        self._create_info_content(info_frame)
        
        # Initial message
        self._show_table_message("Select an entity type and click 'Count' to see results")
    
    def _show_table_message(self, message: str):
        """Show message in table tab."""
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.table_frame,
            text=message,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(expand=True)
    
    def _run_single_count(self):
        """Count single selected entity."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        # Get selected entity
        selected_label = self.entity_combo.get()
        entity_key = None
        for key, config in ENTITY_TYPES.items():
            if config["label"] == selected_label:
                entity_key = key
                break
        
        if not entity_key:
            messagebox.showwarning("Error", "Invalid entity type selected.")
            return
        
        self._current_entity = entity_key
        self._show_loading_in_tabs()
        
        options = {
            "top_n": self.top_n_spin.get(),
            "min_count": self.min_count_spin.get(),
            "generate_plot": self.generate_plot_cb.get(),
            "ngram_range": (self.ngram_min_spin.get(), self.ngram_max_spin.get()),
        }
        
        def do_count():
            try:
                config = ENTITY_TYPES[entity_key]
                method_name = config["method"]
                
                # Call the count method
                method = getattr(self.bib, method_name, None)
                if method:
                    # Pass ngram_range for ngram methods
                    if config.get("has_ngram_options"):
                        ngram_range = options.get("ngram_range", (1, 2))
                        print(f"DEBUG Counts: Calling {method_name} with ngram_range={ngram_range}")
                        method(ngram_range=ngram_range)
                    else:
                        method()
                
                # Get results
                counts_df = getattr(self.bib, config["result_attr"], None)
                
                self.after(0, lambda df=counts_df: self._on_count_success(df, None, entity_key, options))
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                error_msg = str(e)
                self.after(0, lambda msg=error_msg, t=tb: self._on_count_error(msg, t))
        
        threading.Thread(target=do_count, daemon=True).start()
    
    def _run_batch_count(self):
        """Count all checked entities."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        # Get checked entities
        checked = [key for key, cb in self.entity_checkboxes.items() if cb.get()]
        
        if not checked:
            messagebox.showwarning("Nothing Selected", "Please check at least one entity type.")
            return
        
        self._show_loading_in_tabs()
        
        # Get ngram_range for ngram entities
        ngram_range = (self.ngram_min_spin.get(), self.ngram_max_spin.get())
        
        def do_batch():
            try:
                results = {}
                for entity_key in checked:
                    config = ENTITY_TYPES[entity_key]
                    method_name = config["method"]
                    method = getattr(self.bib, method_name, None)
                    if method:
                        # Pass ngram_range for ngram methods
                        if config.get("has_ngram_options"):
                            method(ngram_range=ngram_range)
                        else:
                            method()
                    results[entity_key] = getattr(self.bib, config["result_attr"], None)
                
                self.after(0, lambda: self._on_batch_success(results))
            except Exception as e:
                self.after(0, lambda: self._on_count_error(str(e), ""))
        
        threading.Thread(target=do_batch, daemon=True).start()
    
    def _show_loading_in_tabs(self):
        """Show loading in all tabs."""
        # Stop any active spinners first
        if hasattr(self, '_tab_spinners'):
            for spinner in self._tab_spinners:
                try:
                    spinner.stop()
                except:
                    pass
        self._tab_spinners = []
        
        for frame in [self.table_frame, self.plot_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
            
            from biblium.gui.widgets.progress import LoadingSpinner
            spinner = LoadingSpinner(frame, theme=self.theme_name)
            spinner.pack(expand=True)
            spinner.start()
            self._tab_spinners.append(spinner)
    
    def _stop_tab_spinners(self):
        """Stop all tab spinners."""
        if hasattr(self, '_tab_spinners'):
            for spinner in self._tab_spinners:
                try:
                    spinner.stop()
                except:
                    pass
            self._tab_spinners.clear()
    
    def _on_count_success(self, counts_df, stats_df, entity_key, options):
        """Handle successful count."""
        self._stop_tab_spinners()
        
        self._counts_df = counts_df
        self._stats_df = stats_df
        
        config = ENTITY_TYPES[entity_key]
        
        # Show table
        self._show_counts_table(counts_df, options["top_n"])
        
        # Show bar plot
        if options["generate_plot"] and counts_df is not None:
            self._show_bar_plot(counts_df, config["label"], options["top_n"])
        else:
            self._show_plot_message("Enable 'Generate bar plot' option to see visualization.")
        
        # Emit event
        event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": f"Count {config['label']}"})
        
        event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": f"Count {config['label']}"})
    
    def _generate_stats_from_counts(self, counts_df, entity_label: str):
        """Generate statistics from counts DataFrame."""
        try:
            if counts_df is None or len(counts_df) == 0:
                return None
            
            # Get column names
            cols = counts_df.columns.tolist()
            if len(cols) < 2:
                return None
            
            name_col = cols[0]
            count_col = cols[1]
            
            # Create stats DataFrame
            stats_data = []
            
            for idx, row in counts_df.iterrows():
                name = row[name_col]
                count = row[count_col]
                
                stat_row = {
                    name_col: name,
                    "Count": count,
                    "Rank": idx + 1 if isinstance(idx, int) else len(stats_data) + 1,
                }
                
                # Add percentage
                total = counts_df[count_col].sum()
                if total > 0:
                    stat_row["Percentage"] = round(count / total * 100, 2)
                
                # Add cumulative percentage
                cumsum = counts_df[count_col].iloc[:len(stats_data)+1].sum()
                if total > 0:
                    stat_row["Cumulative %"] = round(cumsum / total * 100, 2)
                
                # Add any other numeric columns from original df
                for col in cols[2:]:
                    if col in row and pd.notna(row[col]):
                        try:
                            stat_row[col] = row[col]
                        except:
                            pass
                
                stats_data.append(stat_row)
            
            return pd.DataFrame(stats_data)
        except Exception as e:
            print(f"Error generating stats: {e}")
            return None
    
    def _show_plot_message(self, message: str):
        """Show message in plot tab."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.plot_frame,
            text=message,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(expand=True)
    
    def _on_batch_success(self, results):
        """Handle batch count success."""
        # Show summary
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        
        summary_text = "Batch Count Results\n" + "=" * 40 + "\n\n"
        for entity_key, df in results.items():
            config = ENTITY_TYPES[entity_key]
            count = len(df) if df is not None else 0
            summary_text += f"{config['icon']} {config['label']}: {count:,} unique items\n"
        
        tk.Label(
            self.table_frame,
            text=summary_text,
            font=FONTS.get_font("mono"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            justify=tk.LEFT,
        ).pack(expand=True)
        
        messagebox.showinfo("Complete", f"Counted {len(results)} entity types.")
    
    def _on_count_error(self, error: str, tb: str = ""):
        """Handle count error."""
        self._show_table_message(f"Error: {error}")
        event_bus.emit(EventBus.ERROR_OCCURRED, {"message": error})
        print(f"Count error: {error}\n{tb}")
    
    def _show_counts_table(self, df, top_n: int):
        """Show counts in table tab."""
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        
        if df is None or len(df) == 0:
            self._show_table_message("No results found.")
            return
        
        # Summary stats
        summary = tk.Frame(self.table_frame, bg=self.theme["bg_card"])
        summary.pack(fill=tk.X, pady=(0, 8))
        
        grid = CardGrid(summary, columns=3, theme=self.theme_name)
        grid.pack(fill=tk.X)
        
        # Calculate total safely
        try:
            if len(df.columns) > 1:
                total = pd.to_numeric(df.iloc[:, 1], errors='coerce').sum()
                total = int(total) if pd.notna(total) else len(df)
            else:
                total = len(df)
        except:
            total = len(df)
        
        grid.add_card(StatsCard(grid, "Total Items", f"{len(df):,}", "📊", self.theme_name))
        grid.add_card(StatsCard(grid, "Total Count", f"{total:,}", "📈", self.theme_name))
        grid.add_card(StatsCard(grid, "Showing", f"Top {min(top_n, len(df))}", "👁️", self.theme_name))
        
        # Table
        table = DataTable(self.table_frame, theme=self.theme_name, max_rows=top_n)
        table.pack(fill=tk.BOTH, expand=True)
        table.set_data(df.head(top_n))
    
    def _show_bar_plot(self, df, entity_label: str, top_n: int):
        """Show horizontal bar plot in plot tab."""
        # Clean up existing plots properly
        for widget in self.plot_frame.winfo_children():
            if hasattr(widget, 'figure'):
                try:
                    import matplotlib.pyplot as plt
                    plt.close(widget.figure)
                except:
                    pass
            widget.destroy()
        
        if df is None or len(df) == 0:
            return
        
        if not HAS_MATPLOTLIB:
            tk.Label(
                self.plot_frame,
                text="Matplotlib required for plots",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
            ).pack(expand=True)
            return
        
        # Limit to top 20 for readability
        n_items = min(top_n, 20)
        plot_df = df.head(n_items)
        
        if len(plot_df.columns) < 2:
            return
        
        # Get labels from first column
        labels = plot_df.iloc[:, 0].astype(str).tolist()
        
        # Find the count column - look for "Number of documents" or second column
        values = None
        for col in plot_df.columns:
            if 'number' in col.lower() or 'count' in col.lower() or 'document' in col.lower():
                try:
                    values = pd.to_numeric(plot_df[col], errors='coerce').fillna(0).astype(int).tolist()
                    break
                except:
                    pass
        
        # Fallback to second column if no count column found
        if values is None:
            try:
                values = pd.to_numeric(plot_df.iloc[:, 1], errors='coerce').fillna(0).astype(int).tolist()
            except:
                values = [0] * len(labels)
        
        # Wrap long labels instead of truncating
        import textwrap
        wrapped_labels = [textwrap.fill(str(l), width=25) for l in labels]
        
        # Reverse for proper display (highest at top)
        wrapped_labels = wrapped_labels[::-1]
        values = values[::-1]
        
        # Calculate figure height based on number of items
        fig_height = max(5, min(12, n_items * 0.5))
        
        plot_info = {
            "type": "horizontal bar chart",
            "title": f"Top {n_items} {entity_label}",
            "x_label": "Number of Documents",
            "y_label": entity_label,
            "data_summary": f"{n_items} items shown. Top: {labels[0] if labels else 'N/A'}",
        }
        
        plot = PlotFrame(self.plot_frame, theme=self.theme_name, figsize=(8, fig_height),
                         show_ai_button=True, plot_info=plot_info)
        plot.pack(fill=tk.BOTH, expand=True)
        
        fig, ax = plot.get_figure()
        
        # Create horizontal bar chart
        y_pos = range(len(wrapped_labels))
        bars = ax.barh(y_pos, values, color=self.theme["accent_primary"], height=0.7)
        
        # Set labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(wrapped_labels, fontsize=8)
        ax.set_xlabel("Number of Documents")
        ax.set_title(f"Top {n_items} {entity_label}")
        
        # Remove gridlines
        ax.grid(False)
        
        # Force integer ticks on x-axis
        max_val = max(values) if values else 1
        ax.set_xlim(0, max_val * 1.15)
        
        # Set integer ticks
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add value labels on bars
        for i, (bar, v) in enumerate(zip(bars, values)):
            ax.text(v + max_val * 0.02, i, f"{int(v)}", va='center', fontsize=8)
        
        fig.tight_layout()
        plot.refresh()
    
    def _show_stats_table(self, df, top_n: int):
        """Show statistics table."""
        for widget in self.stats_frame.winfo_children():
            widget.destroy()
        
        if df is None or len(df) == 0:
            self._show_stats_message("No statistics available.")
            return
        
        table = DataTable(self.stats_frame, theme=self.theme_name, max_rows=top_n)
        table.pack(fill=tk.BOTH, expand=True)
        table.set_data(df.head(top_n))
    
    def _show_stats_message(self, message: str):
        """Show message in stats tab."""
        for widget in self.stats_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.stats_frame,
            text=message,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(expand=True)
    
    def _show_scatter_plot(self, stats_df, entity_label: str):
        """Show scatter plot of statistics (e.g., h-index vs citations)."""
        for widget in self.scatter_frame.winfo_children():
            widget.destroy()
        
        if stats_df is None or len(stats_df) == 0:
            self._show_scatter_message("No statistics available for scatter plot.")
            return
        
        if not HAS_MATPLOTLIB:
            self._show_scatter_message("Matplotlib required for plots")
            return
        
        # Find numeric columns for scatter plot
        numeric_cols = stats_df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            self._show_scatter_message("Need at least 2 numeric columns for scatter plot.")
            return
        
        # Controls
        controls = tk.Frame(self.scatter_frame, bg=self.theme["bg_card"])
        controls.pack(fill=tk.X, padx=8, pady=8)
        
        x_combo = LabeledCombobox(
            controls, label="X-axis:", values=numeric_cols,
            default=numeric_cols[0], theme=self.theme_name, label_width=8,
        )
        x_combo.pack(side=tk.LEFT, padx=(0, 16))
        
        y_combo = LabeledCombobox(
            controls, label="Y-axis:", values=numeric_cols,
            default=numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0],
            theme=self.theme_name, label_width=8,
        )
        y_combo.pack(side=tk.LEFT, padx=(0, 16))
        
        # Plot container
        plot_container = tk.Frame(self.scatter_frame, bg=self.theme["bg_card"])
        plot_container.pack(fill=tk.BOTH, expand=True)
        
        def update_plot():
            for w in plot_container.winfo_children():
                w.destroy()
            
            x_col = x_combo.get()
            y_col = y_combo.get()
            
            plot_info = {
                "type": "scatter plot",
                "title": f"{entity_label}: {x_col} vs {y_col}",
                "x_label": x_col,
                "y_label": y_col,
                "data_summary": f"{len(stats_df)} data points",
            }
            
            plot = PlotFrame(plot_container, theme=self.theme_name, figsize=(10, 6),
                             show_ai_button=True, plot_info=plot_info)
            plot.pack(fill=tk.BOTH, expand=True)
            
            fig, ax = plot.get_figure()
            
            x_data = pd.to_numeric(stats_df[x_col], errors='coerce').dropna()
            y_data = pd.to_numeric(stats_df[y_col], errors='coerce').dropna()
            
            # Align indices
            common_idx = x_data.index.intersection(y_data.index)
            x_data = x_data.loc[common_idx]
            y_data = y_data.loc[common_idx]
            
            ax.scatter(x_data, y_data, alpha=0.6, color=self.theme["accent_primary"])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{entity_label}: {x_col} vs {y_col}")
            
            fig.tight_layout()
            plot.refresh()
        
        ThemedButton(
            controls, text="Update Plot", style="primary", size="small",
            command=update_plot, theme=self.theme_name,
        ).pack(side=tk.LEFT)
        
        # Initial plot
        update_plot()
    
    def _show_scatter_message(self, message: str):
        """Show message in scatter tab."""
        for widget in self.scatter_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.scatter_frame,
            text=message,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(expand=True)
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
FREQUENCY COUNTS
════════════════

Count and rank bibliometric entities.

ENTITY TYPES
────────────
• Authors: Most productive researchers
• Sources: Core journals/venues
• Author Keywords: Topic terms
• Index Keywords: Database terms
• Countries: Geographic distribution
• Affiliations: Institutions
• References: Most cited works

COUNT OPTIONS
─────────────
• Top N: Number of items to display
• Minimum frequency: Filter threshold
• Fractional counting: Split credit
• Full counting: Full credit each

OUTPUT COLUMNS
──────────────
• Entity name
• Frequency (count)
• Percentage of total
• Cumulative percentage
• Rank

VISUALIZATION
─────────────
• Bar chart: Top N entities
• Table: Full ranking data

LOTKA'S LAW CONNECTION
──────────────────────
Author counts reveal productivity patterns:
• Few highly productive authors
• Many single-paper authors
• Approximately 1/n² distribution

BRADFORD'S LAW CONNECTION
─────────────────────────
Source counts show journal scatter:
• Core journals (few, many papers)
• Peripheral (many, few papers)

EXPORT
──────
• Excel/CSV with full data
• Include percentages
• Ranked listing
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
        """Export current results."""
        if self._counts_df is None:
            messagebox.showwarning("No Results", "No results to export. Run a count first.")
            return
        
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx"), ("CSV Files", "*.csv")],
            title="Export Results",
        )
        
        if filename:
            try:
                if filename.endswith(".xlsx"):
                    with pd.ExcelWriter(filename) as writer:
                        self._counts_df.to_excel(writer, sheet_name="Counts", index=False)
                        if self._stats_df is not None:
                            self._stats_df.to_excel(writer, sheet_name="Statistics", index=False)
                else:
                    self._counts_df.to_csv(filename, index=False)
                
                messagebox.showinfo("Success", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
