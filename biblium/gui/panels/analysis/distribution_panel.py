# -*- coding: utf-8 -*-
"""
Distribution Panel
==================
Box plot and violin plot visualizations for continuous variables.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import pandas as pd
import numpy as np

from biblium.gui.panels.base import BasePanel
from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.widgets.buttons import ActionButton
from biblium.gui.widgets.cards import Card
from biblium.gui.widgets.plots import ScaledImageFrame


class DistributionPanel(BasePanel):
    """Panel for box plot and violin plot visualizations."""
    
    title = "Distribution Analysis"
    description = "Visualize distributions with box plots and violin plots"
    icon = "📊"
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        # Initialize instance variables BEFORE calling super().__init__
        self._image_bytes = None
        self._thread_params = None
        
        # Call parent - this will call _create_options() and _create_results()
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        
        # Set primary action for toolbar
        self._primary_action = self._run_analysis
    
    def _create_options(self):
        """Create options panel - called by BasePanel.__init__."""
        # Title
        tk.Label(
            self.options_content,
            text="📊 Distribution Analysis",
            font=FONTS.get_font("title"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_primary"],
        ).pack(anchor=tk.W, padx=8, pady=(8, 4))
        
        tk.Label(
            self.options_content,
            text="Box plot and violin plot visualizations",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_muted"],
        ).pack(anchor=tk.W, padx=8, pady=(0, 12))
        
        # Variable Selection Card
        var_card = Card(self.options_content, title="📋 Variables", theme=self.theme_name)
        var_card.pack(fill=tk.X, padx=4, pady=4)
        
        # Numeric variable
        tk.Label(
            var_card.content, text="Numeric Variable:",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_primary"]
        ).pack(anchor=tk.W, pady=(4, 2))
        
        self.numeric_var = tk.StringVar(value="")
        self.numeric_combo = ttk.Combobox(
            var_card.content, textvariable=self.numeric_var, state="readonly", width=28
        )
        self.numeric_combo.pack(fill=tk.X, pady=(0, 8))
        
        # Group by variable
        tk.Label(
            var_card.content, text="Group By:",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_primary"]
        ).pack(anchor=tk.W, pady=(4, 2))
        
        self.grouping_var = tk.StringVar(value="")
        self.grouping_combo = ttk.Combobox(
            var_card.content, textvariable=self.grouping_var, state="readonly", width=28
        )
        self.grouping_combo.pack(fill=tk.X, pady=(0, 4))
        
        # Plot Type Card
        plot_card = Card(self.options_content, title="📈 Plot Type", theme=self.theme_name)
        plot_card.pack(fill=tk.X, padx=4, pady=4)
        
        self.plot_type_var = tk.StringVar(value="boxplot")
        
        tk.Radiobutton(
            plot_card.content, text="📦 Box Plot", variable=self.plot_type_var, value="boxplot",
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            activebackground=self.theme["bg_card"],
            font=FONTS.get_font("body"),
        ).pack(anchor=tk.W, pady=2)
        
        tk.Radiobutton(
            plot_card.content, text="🎻 Violin Plot", variable=self.plot_type_var, value="violin",
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            activebackground=self.theme["bg_card"],
            font=FONTS.get_font("body"),
        ).pack(anchor=tk.W, pady=2)
        
        # Options Card
        opt_card = Card(self.options_content, title="⚙️ Options", theme=self.theme_name)
        opt_card.pack(fill=tk.X, padx=4, pady=4)
        
        # Max groups
        max_frame = tk.Frame(opt_card.content, bg=self.theme["bg_card"])
        max_frame.pack(fill=tk.X, pady=2)
        tk.Label(
            max_frame, text="Max Groups:", width=14, anchor=tk.W,
            font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_primary"]
        ).pack(side=tk.LEFT)
        self.max_groups_var = tk.IntVar(value=10)
        tk.Spinbox(
            max_frame, from_=3, to=20, textvariable=self.max_groups_var, width=8
        ).pack(side=tk.LEFT)
        
        # Min group size
        min_frame = tk.Frame(opt_card.content, bg=self.theme["bg_card"])
        min_frame.pack(fill=tk.X, pady=2)
        tk.Label(
            min_frame, text="Min Group Size:", width=14, anchor=tk.W,
            font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_primary"]
        ).pack(side=tk.LEFT)
        self.min_size_var = tk.IntVar(value=5)
        tk.Spinbox(
            min_frame, from_=1, to=100, textvariable=self.min_size_var, width=8
        ).pack(side=tk.LEFT)
        
        # Checkboxes
        self.show_counts_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            opt_card.content, text="Show sample sizes (n=X)", variable=self.show_counts_var,
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            activebackground=self.theme["bg_card"],
            font=FONTS.get_font("body"),
        ).pack(anchor=tk.W, pady=2)
        
        self.order_by_size_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            opt_card.content, text="Order groups by size", variable=self.order_by_size_var,
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            activebackground=self.theme["bg_card"],
            font=FONTS.get_font("body"),
        ).pack(anchor=tk.W, pady=2)
        
        self.stat_test_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            opt_card.content, text="Show Kruskal-Wallis test", variable=self.stat_test_var,
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            activebackground=self.theme["bg_card"],
            font=FONTS.get_font("body"),
        ).pack(anchor=tk.W, pady=2)
        
        # Run Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=4, pady=12)
        
        self.run_btn = ActionButton(
            btn_frame, text="Generate Plot", icon="▶",
            command=self._run_analysis, theme=self.theme_name,
        )
        self.run_btn.pack(fill=tk.X)
        
        # Export buttons
        export_frame = tk.Frame(btn_frame, bg=self.theme["bg_secondary"])
        export_frame.pack(fill=tk.X, pady=(8, 0))
        
        tk.Button(
            export_frame, text="Export PNG", command=lambda: self._export_plot("png"),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        tk.Button(
            export_frame, text="Export PDF", command=lambda: self._export_plot("pdf"),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
        # Update variable lists if data is loaded
        self._update_variable_lists()
    
    def _create_results(self):
        """Create results panel - called by BasePanel.__init__."""
        # Results header
        tk.Label(
            self.results_frame,
            text="Results",
            font=FONTS.get_font("heading"),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_primary"],
        ).pack(anchor=tk.W, padx=16, pady=(16, 8))
        
        # Create notebook for results and info
        self.results_notebook = ttk.Notebook(self.results_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))
        
        # Plot tab
        self.plot_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.plot_frame, text="📊 Plot")
        
        # Info tab
        self.info_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.info_frame, text="ℹ️ Info")
        self._create_info_content(self.info_frame)
        
        # Show placeholder
        self._show_placeholder()
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
DISTRIBUTION ANALYSIS
═════════════════════

Analyze statistical distributions of numeric variables.

PLOT TYPES
──────────
• Histogram: Frequency distribution
• Density (KDE): Smoothed curve
• Box Plot: Quartiles and outliers
• Violin Plot: Full distribution shape

STATISTICS SHOWN
────────────────
• Mean: Average value
• Median: Middle value (robust)
• Std Dev: Spread measure
• Skewness: Asymmetry
• Kurtosis: Tail heaviness

GROUPING
────────
• Split by categorical variable
• Compare distributions
• Side-by-side plots

INTERPRETATION
──────────────
• Skewness > 0: Right tail (citations)
• Skewness < 0: Left tail
• High kurtosis: Heavy tails, outliers
• Mean >> Median: Skewed distribution

COMMON PATTERNS
───────────────
• Citations: Highly right-skewed
• Author count: Moderate skew
• References: Field-dependent

OPTIONS
───────
• Log scale: For skewed data
• Bins: Histogram resolution
• Bandwidth: KDE smoothness
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
                pass
        
        def copy_all(widget):
            widget.config(state=tk.NORMAL)
            content = widget.get("1.0", tk.END)
            widget.config(state=tk.DISABLED)
            widget.clipboard_clear()
            widget.clipboard_append(content.strip())
        
        text_widget.bind("<Button-3>", show_context_menu)
        text_widget.bind("<Control-c>", lambda e: copy_selected(text_widget))
    
    def _update_variable_lists(self):
        """Update combobox values based on loaded data."""
        if self.bib is None:
            numeric_vars = ["Cited by", "Year"]
            grouping_vars = ["Document Type", "Source"]
        else:
            # Find numeric columns
            numeric_vars = []
            for col in self.bib.df.columns:
                if pd.api.types.is_numeric_dtype(self.bib.df[col]):
                    if 'unnamed' not in col.lower() and 'index' not in col.lower():
                        numeric_vars.append(col)
            
            # Find categorical columns
            grouping_vars = []
            for col in self.bib.df.columns:
                if self.bib.df[col].dtype == 'object':
                    nunique = self.bib.df[col].nunique()
                    if 2 <= nunique <= 500:
                        grouping_vars.append(col)
        
        self.numeric_combo['values'] = numeric_vars
        if numeric_vars:
            self.numeric_var.set(numeric_vars[0])
        
        self.grouping_combo['values'] = grouping_vars
        if grouping_vars:
            self.grouping_var.set(grouping_vars[0])
    
    def _show_placeholder(self):
        """Show placeholder message."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.plot_frame,
            text="Select variables and click 'Generate Plot'\nto visualize distributions",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)
    
    def _run_analysis(self):
        """Run distribution analysis."""
        if self.bib is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        if self._is_running:
            return
        
        # Get values from UI
        numeric_col = self.numeric_var.get()
        grouping_col = self.grouping_var.get()
        plot_type = self.plot_type_var.get()
        max_groups = self.max_groups_var.get()
        min_size = self.min_size_var.get()
        show_counts = self.show_counts_var.get()
        order_by_size = self.order_by_size_var.get()
        stat_test = self.stat_test_var.get()
        
        print(f"DEBUG: numeric={numeric_col}, grouping={grouping_col}, stat_test={stat_test}")
        
        if not numeric_col or not grouping_col:
            messagebox.showwarning("Missing Selection", "Please select both variables.")
            return
        
        self._is_running = True
        self.run_btn.config(state=tk.DISABLED)
        
        # Show loading
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        tk.Label(
            self.plot_frame, text="⏳ Generating plot...",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"],
        ).pack(expand=True)
        
        # Store parameters
        self._thread_params = {
            'numeric_col': numeric_col,
            'grouping_col': grouping_col,
            'plot_type': plot_type,
            'max_groups': max_groups,
            'min_size': min_size,
            'show_counts': show_counts,
            'order_by_size': order_by_size,
            'stat_test': stat_test,
        }
        
        def do_analysis():
            try:
                from biblium import plotbib
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import io
                
                plt.close('all')
                
                p = self._thread_params
                
                if p['plot_type'] == "boxplot":
                    plotbib.plot_boxplot(
                        df=self.bib.df,
                        value_column=p['numeric_col'],
                        group_by=p['grouping_col'],
                        max_groups=p['max_groups'],
                        min_group_size=p['min_size'],
                        show_counts=p['show_counts'],
                        order_by_size=p['order_by_size'],
                        stat_test=p['stat_test'],
                        title=f"Distribution of {p['numeric_col']} by {p['grouping_col']}",
                        show=False,
                        figsize=(10, 6),
                    )
                else:
                    plotbib.plot_violinplot(
                        df=self.bib.df,
                        value_column=p['numeric_col'],
                        group_by=p['grouping_col'],
                        max_groups=p['max_groups'],
                        min_group_size=p['min_size'],
                        show_counts=p['show_counts'],
                        order_by_size=p['order_by_size'],
                        stat_test=p['stat_test'],
                        title=f"Distribution of {p['numeric_col']} by {p['grouping_col']}",
                        show=False,
                        figsize=(10, 6),
                    )
                
                fig = plt.gcf()
                print(f"DEBUG: Figure texts = {[t.get_text() for t in fig.texts]}")
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                           facecolor='white', edgecolor='none', pad_inches=0.2)
                buf.seek(0)
                self._image_bytes = buf.getvalue()
                buf.close()
                plt.close(fig)
                
                self.after(0, self._on_success)
                
            except Exception as e:
                import traceback
                print(f"ERROR: {traceback.format_exc()}")
                self.after(0, lambda: self._on_error(str(e)))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _on_success(self):
        """Handle successful analysis."""
        self._is_running = False
        self.run_btn.config(state=tk.NORMAL)
        self._display_plot()
    
    def _on_error(self, error_msg: str):
        """Handle analysis error."""
        self._is_running = False
        self.run_btn.config(state=tk.NORMAL)
        
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.plot_frame, text=f"❌ Error: {error_msg}",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["danger"],
            wraplength=400,
        ).pack(expand=True, pady=20)
    
    def _display_plot(self):
        """Display the plot from saved bytes."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        if self._image_bytes is None:
            self._show_placeholder()
            return
        
        try:
            from PIL import Image
            import io
            
            buf = io.BytesIO(self._image_bytes)
            pil_image = Image.open(buf)
            pil_image = pil_image.copy()
            buf.close()
            
            scaled_frame = ScaledImageFrame(
                self.plot_frame, theme=self.theme_name,
                maintain_aspect=True, max_scale=1.5
            )
            scaled_frame.pack(fill=tk.BOTH, expand=True)
            scaled_frame.set_image(pil_image)
            
            # Right-click menu
            def show_menu(event):
                menu = tk.Menu(scaled_frame, tearoff=0)
                menu.add_command(label="📄 Add to Report", command=self._add_plot_to_report)
                menu.add_separator()
                menu.add_command(label="💾 Save as PNG...", command=lambda: self._export_plot("png"))
                menu.add_command(label="💾 Save as PDF...", command=lambda: self._export_plot("pdf"))
                menu.tk_popup(event.x_root, event.y_root)
            scaled_frame.bind("<Button-3>", show_menu)
            
        except Exception as e:
            import traceback
            print(f"Display error: {traceback.format_exc()}")
            tk.Label(
                self.plot_frame, text=f"Display error: {e}",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["danger"],
            ).pack(expand=True)
    
    def _add_plot_to_report(self):
        """Add current plot to report queue."""
        if self._image_bytes is None:
            messagebox.showinfo("No Plot", "No plot to add to report.")
            return
        
        try:
            from biblium.gui.core.state import report_queue
            
            plot_title = "Distribution Analysis"
            
            report_queue.add_plot(
                figure_or_bytes=self._image_bytes,
                title=plot_title,
                source_panel=self.title,
            )
            
            messagebox.showinfo(
                "Added to Report",
                f"Plot '{plot_title}' has been added to the report queue.\n\n"
                f"Items in queue: {len(report_queue)}\n\n"
                "Go to Reports → Report Builder to generate your report."
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add to report: {e}")
    
    def _export_plot(self, fmt: str):
        """Export current plot."""
        if self._image_bytes is None:
            messagebox.showwarning("No Plot", "Generate a plot first.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=f".{fmt}",
            filetypes=[(f"{fmt.upper()} files", f"*.{fmt}")],
        )
        
        if filepath:
            try:
                if fmt == "png":
                    with open(filepath, 'wb') as f:
                        f.write(self._image_bytes)
                else:
                    from PIL import Image
                    import io
                    buf = io.BytesIO(self._image_bytes)
                    img = Image.open(buf)
                    img.save(filepath, fmt.upper())
                    buf.close()
                messagebox.showinfo("Saved", f"Plot saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")
    
    def on_data_loaded(self, bib):
        """Handle data loaded event."""
        self.bib = bib
        self._image_bytes = None
        self._update_variable_lists()
        self._show_placeholder()
    
    def on_data_cleared(self):
        """Handle data cleared event."""
        self.bib = None
        self._image_bytes = None
        self._show_placeholder()
