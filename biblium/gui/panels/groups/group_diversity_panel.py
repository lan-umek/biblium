# -*- coding: utf-8 -*-
"""
Group Diversity Panel
=====================
Compare diversity indices across bibliographic groups.

@author: Lan.Umek
@version: 2.9.0
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from typing import Dict, List, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox
from biblium.gui.widgets.plots import add_plot_context_menu, make_canvas_resizable

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class GroupDiversityPanel(BasePanel):
    """
    Panel for comparing diversity indices across groups.
    """
    
    title = "Group Diversity"
    icon = "📊"
    description = "Compare diversity indices across bibliographic groups"
    requires_data = True
    requires_groups = True
    
    def __init__(self, parent, theme: str = "light", bib=None, bib_group=None, **kwargs):
        self._result = None
        self._current_fig = None
        self._canvas = None
        self._entity_vars = {}
        self.bib_group = bib_group
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._run_analysis
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Info Card
        info_card = Card(self.options_content, title="ℹ️ Group Analysis", theme=self.theme_name)
        info_card.pack(fill=tk.X, padx=8, pady=8)
        
        info_label = tk.Label(
            info_card.content,
            text="Compare diversity indices across\nthe groups defined in your analysis.\n\nRequires groups to be created first\n(Data → Group Settings).",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
            justify=tk.LEFT,
        )
        info_label.pack(fill=tk.X, pady=(0, 8))
        
        # Entity Selection Card
        entity_card = Card(self.options_content, title="📋 Entities to Compare", theme=self.theme_name)
        entity_card.pack(fill=tk.X, padx=8, pady=8)
        
        self._entity_checkboxes_frame = tk.Frame(entity_card.content, bg=self.theme["bg_card"])
        self._entity_checkboxes_frame.pack(fill=tk.X)
        
        default_entities = [
            "Sources", "Authors", "Countries", "Affiliations",
            "Author Keywords", "Index Keywords", "Subject Areas",
            "Document Types", "SDGs"
        ]
        
        for entity in default_entities:
            var = tk.BooleanVar(value=entity in ["Sources", "Countries", "Author Keywords"])
            self._entity_vars[entity] = var
            
            cb = tk.Checkbutton(
                self._entity_checkboxes_frame,
                text=entity,
                variable=var,
                bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
                activebackground=self.theme["bg_card"],
                selectcolor=self.theme["bg_card"],
            )
            cb.pack(anchor=tk.W, pady=1)
        
        # Visualization Card
        viz_card = Card(self.options_content, title="📊 Visualization", theme=self.theme_name)
        viz_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.index_var = LabeledCombobox(
            viz_card.content, label="Index:",
            values=["Shannon", "Simpson", "Gini"],
            default="Shannon",
            theme=self.theme_name, label_width=12,
        )
        self.index_var.pack(fill=tk.X, pady=4)
        
        self.plot_type = LabeledCombobox(
            viz_card.content, label="Plot Type:",
            values=["Bar Chart", "Radar Chart", "Heatmap"],
            default="Bar Chart",
            theme=self.theme_name, label_width=12,
        )
        self.plot_type.pack(fill=tk.X, pady=4)
        
        # Run button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_primary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        self.run_btn = ActionButton(
            btn_frame, text="Compare Group Diversity",
            command=self._run_analysis,
            icon="▶",
            theme=self.theme_name,
        )
        self.run_btn.pack(fill=tk.X)
        
        # Export buttons
        self.export_plot_btn = ThemedButton(
            btn_frame, text="Export Plot",
            command=self._export_plot,
            style="secondary",
            icon="🖼️",
            theme=self.theme_name,
        )
        self.export_plot_btn.pack(fill=tk.X, pady=(8, 0))
        
        self.export_data_btn = ThemedButton(
            btn_frame, text="Export Data",
            command=self._export_data,
            style="secondary",
            icon="📊",
            theme=self.theme_name,
        )
        self.export_data_btn.pack(fill=tk.X, pady=(8, 0))
    
    def _create_results(self):
        """Create the results panel."""
        self.notebook = ttk.Notebook(self.results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Plot tab
        self.plot_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.plot_frame, text="📊 Visualization")
        
        # Data tab
        self.data_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.data_frame, text="📋 Data")
        
        # Comparison tab
        self.comparison_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.comparison_frame, text="⚖️ Comparison")
        
        # Summary tab
        self.summary_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.summary_frame, text="📝 Summary")

        

        # Info tab

        info_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])

        self.notebook.add(info_frame, text="ℹ️ Info")

        self._create_info_content(info_frame)
        
        self._show_placeholder()
    
    def _show_placeholder(self):
        """Show placeholder message."""
        if self._current_fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._current_fig)
            except:
                pass
            self._current_fig = None
        
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Check if groups are available
        if self.bib_group is None or not hasattr(self.bib_group, 'groups') or not self.bib_group.groups:
            msg = "No groups available.\n\nPlease create groups first:\n1. Go to Data → Group Settings\n2. Define your groups\n3. Return here to compare diversity"
        else:
            n_groups = len(self.bib_group.groups)
            msg = f"{n_groups} groups available.\n\nSelect entities and click 'Compare Group Diversity'"
        
        tk.Label(
            self.plot_frame,
            text=msg,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
            justify=tk.CENTER,
        ).pack(expand=True)
    
    def _get_selected_entities(self) -> List[str]:
        """Get list of selected entities."""
        return [entity for entity, var in self._entity_vars.items() if var.get()]
    
    def _run_analysis(self):
        """Run the group diversity analysis."""
        if self.bib_group is None:
            messagebox.showwarning("No Groups", "Please create groups first (Data → Group Settings).")
            return
        
        if not hasattr(self.bib_group, 'groups') or not self.bib_group.groups:
            messagebox.showwarning("No Groups", "No groups found. Please build groups first.")
            return
        
        selected_entities = self._get_selected_entities()
        if not selected_entities:
            messagebox.showwarning("No Entities", "Please select at least one entity.")
            return
        
        self.run_btn.config(state=tk.DISABLED, text="⏳ Computing...")
        
        # Capture values
        index_map = {"Shannon": "shannon", "Simpson": "simpson", "Gini": "gini"}
        index = index_map.get(self.index_var.get(), "shannon")
        plot_type = self.plot_type.get()
        
        def do_analysis():
            try:
                from biblium.diversity import compute_group_diversity_from_bib_group
                
                result = compute_group_diversity_from_bib_group(
                    self.bib_group,
                    entities=selected_entities,
                )
                
                self._result = result
                
                self.after(0, lambda: self._create_plot(result, index, plot_type))
                self.after(0, self._on_analysis_success)
                
            except Exception as exc:
                import traceback
                error_info = (str(exc), traceback.format_exc())
                self.after(0, lambda ei=error_info: self._on_analysis_error(ei[0], ei[1]))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _create_plot(self, result, index: str, plot_type: str):
        """Create the visualization."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from biblium.diversity import (
            plot_group_diversity_comparison,
            plot_group_diversity_radar,
            plot_group_diversity_heatmap,
        )
        
        # Clean up previous
        if self._current_fig is not None:
            try:
                plt.close(self._current_fig)
            except:
                pass
        
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        try:
            if plot_type == "Radar Chart":
                fig, ax = plot_group_diversity_radar(result)
            elif plot_type == "Heatmap":
                fig, ax = plot_group_diversity_heatmap(result, index=index)
            else:  # Bar Chart
                fig, ax = plot_group_diversity_comparison(result, index=index)
            
            self._current_fig = fig
            
            # Create canvas with proper scaling
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)
            add_plot_context_menu(canvas_widget, fig)
            self._canvas = canvas
            
            # Bind resize to update figure size
            self._resize_after_id = None
            def on_resize(event):
                if event.width > 100 and event.height > 100:
                    # Cancel pending resize
                    if self._resize_after_id:
                        canvas_widget.after_cancel(self._resize_after_id)
                    
                    def do_resize():
                        try:
                            dpi = fig.get_dpi()
                            fig.set_size_inches(event.width / dpi, event.height / dpi)
                            fig.tight_layout(pad=1.5, rect=[0.12, 0.12, 0.95, 0.95])
                            canvas.draw_idle()
                        except:
                            pass
                    
                    # Debounce resize
                    self._resize_after_id = canvas_widget.after(100, do_resize)
            
            canvas_widget.bind("<Configure>", on_resize)
            
        except Exception as e:
            tk.Label(
                self.plot_frame,
                text=f"Error creating plot: {e}",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_secondary"],
            ).pack(expand=True)
            import traceback
            traceback.print_exc()
    
    def _on_analysis_success(self):
        """Handle successful analysis."""
        self.run_btn.config(state=tk.NORMAL, text="▶ Compare Group Diversity")
        self._update_data_table()
        self._update_comparison_table()
        self._update_summary()
        self.notebook.select(0)
        
        event_bus.emit("status_update", {
            "message": f"Group diversity analysis complete.",
            "level": "success"
        })
    
    def _on_analysis_error(self, error_msg: str, traceback_str: str):
        """Handle analysis error."""
        self.run_btn.config(state=tk.NORMAL, text="▶ Compare Group Diversity")
        messagebox.showerror("Analysis Error", f"Error:\n{error_msg}")
        print(f"Error details:\n{traceback_str}")
    
    def _update_data_table(self):
        """Update the data table."""
        for widget in self.data_frame.winfo_children():
            widget.destroy()
        
        if self._result is None:
            return
        
        df = self._result.to_dataframe()
        
        columns = list(df.columns)
        tree = ttk.Treeview(self.data_frame, columns=columns, show="headings", height=15)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, minwidth=60)
        
        for _, row in df.iterrows():
            values = []
            for v in row:
                if isinstance(v, float):
                    values.append(f"{v:.3f}")
                else:
                    values.append(str(v)[:30])
            tree.insert("", tk.END, values=values)
        
        vsb = ttk.Scrollbar(self.data_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(self.data_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        self.data_frame.grid_rowconfigure(0, weight=1)
        self.data_frame.grid_columnconfigure(0, weight=1)
    
    def _update_comparison_table(self):
        """Update the comparison table."""
        for widget in self.comparison_frame.winfo_children():
            widget.destroy()
        
        if self._result is None:
            return
        
        # Create comparison tables for each index
        for idx_name, idx_key in [("Shannon", "shannon"), ("Simpson", "simpson"), ("Gini", "gini")]:
            df = self._result.to_comparison_dataframe(idx_key)
            
            label = tk.Label(
                self.comparison_frame,
                text=f"\n{idx_name} Index Comparison",
                font=FONTS.get_font("heading"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
            )
            label.pack(anchor=tk.W, padx=8)
            
            columns = list(df.columns)
            tree = ttk.Treeview(self.comparison_frame, columns=columns, show="headings", height=6)
            
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=100, minwidth=60)
            
            for _, row in df.iterrows():
                values = [f"{v:.3f}" if isinstance(v, float) and not pd.isna(v) else str(v) for v in row]
                tree.insert("", tk.END, values=values)
            
            tree.pack(fill=tk.X, padx=8, pady=4)
    
    def _update_summary(self):
        """Update the summary tab."""
        for widget in self.summary_frame.winfo_children():
            widget.destroy()
        
        if self._result is None:
            return
        
        summary_text = self._result.summary()
        
        text_widget = tk.Text(
            self.summary_frame,
            font=("Consolas", 10),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            wrap=tk.WORD,
            padx=16, pady=16,
            relief=tk.FLAT,
        )
        text_widget.insert("1.0", summary_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True)
    
    def _export_plot(self):
        """Export the current plot."""
        if self._current_fig is None:
            messagebox.showinfo("No Plot", "Please run analysis first.")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Plot",
            filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf")],
            defaultextension=".png",
        )
        if filepath:
            self._current_fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            messagebox.showinfo("Saved", f"Plot saved to:\n{filepath}")
    
    def _export_data(self):
        """Export the results data."""
        if self._result is None:
            messagebox.showinfo("No Data", "Please run analysis first.")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Data",
            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv")],
            defaultextension=".xlsx",
        )
        if filepath:
            df = self._result.to_dataframe()
            if filepath.endswith('.csv'):
                df.to_csv(filepath, index=False)
            else:
                df.to_excel(filepath, index=False)
            messagebox.showinfo("Saved", f"Data saved to:\n{filepath}")
    
    def update_bib(self, bib):
        """Update the BiblioAnalysis instance."""
        self.bib = bib
        if self._current_fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._current_fig)
            except:
                pass
        self._result = None
        self._current_fig = None
        self._canvas = None
        self._show_placeholder()
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
GROUP DIVERSITY
═══════════════

Compare diversity metrics across groups.

METRICS PER GROUP
─────────────────
• Shannon Index (H')
• Simpson Index (1-D)
• Richness (S)
• Evenness (J)
• Gini Index

COMPARISON TABLE
────────────────
         Group A   Group B
H'        2.89      2.21
1-D       0.91      0.85
S          45        32
J         0.76      0.68

INTERPRETATION
──────────────
Higher values indicate:
• More entities represented
• More even distribution
• Less concentration

APPLICATIONS
────────────
• Compare topic breadth
• Author concentration
• Journal diversity
• Geographic spread
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

    def update_bib_group(self, bib_group):
        """Update the BiblioGroupAnalysis instance."""
        self.bib_group = bib_group
        self._show_placeholder()
