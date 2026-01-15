# -*- coding: utf-8 -*-
"""
Correlation Panel - Correlation Analysis
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import io

from biblium.gui.config import FONTS, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.buttons import ActionButton
from biblium.gui.widgets.plots import add_plot_context_menu, make_canvas_resizable

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
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class CorrelationPanel(BasePanel):
    """Panel for correlation analysis - SPSS-like interface."""
    
    title = "Correlation"
    icon = "📈"
    requires_data = True
    
    def __init__(self, parent, theme="light", **kwargs):
        self._result = None
        self._current_fig = None
        super().__init__(parent, theme=theme, **kwargs)
        event_bus.subscribe(EventBus.DATASET_LOADED, self._handle_data_loaded)
    
    def _handle_data_loaded(self, data):
        bib = data.get("bib") if isinstance(data, dict) else data
        self.on_data_loaded(bib)
    
    def _create_options(self):
        self._add_title()
        
        # Variable selection frame
        var_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        var_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        
        # Left: Available variables
        left_frame = tk.Frame(var_frame, bg=self.theme["bg_secondary"])
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tk.Label(left_frame, text="Numeric Variables:", font=FONTS.get_font("body", bold=True),
                bg=self.theme["bg_secondary"], fg=self.theme["text_primary"]).pack(anchor="w")
        
        avail_frame = tk.Frame(left_frame)
        avail_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        
        self.avail_listbox = tk.Listbox(avail_frame, height=10, width=20, font=FONTS.get_font("small"),
                                        selectmode=tk.EXTENDED, exportselection=False, bg="white")
        self.avail_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        avail_sb = ttk.Scrollbar(avail_frame, command=self.avail_listbox.yview)
        avail_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.avail_listbox.config(yscrollcommand=avail_sb.set)
        
        # Middle: Arrow buttons
        mid_frame = tk.Frame(var_frame, bg=self.theme["bg_secondary"])
        mid_frame.pack(side=tk.LEFT, padx=6, pady=10)
        
        tk.Label(mid_frame, text="Variables:", font=FONTS.get_font("small"),
                bg=self.theme["bg_secondary"], fg=self.theme["text_muted"]).pack()
        tk.Button(mid_frame, text="▶", width=3, command=self._add_variables).pack(pady=2)
        tk.Button(mid_frame, text="◀", width=3, command=self._remove_variables).pack(pady=2)
        tk.Button(mid_frame, text="All ▶", width=5, command=self._add_all_variables).pack(pady=(10,2))
        tk.Button(mid_frame, text="◀ Clear", width=5, command=self._clear_variables).pack(pady=2)
        
        # Right: Selected variables
        right_frame = tk.Frame(var_frame, bg=self.theme["bg_secondary"])
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tk.Label(right_frame, text="Variables to Correlate:", font=FONTS.get_font("body", bold=True),
                bg=self.theme["bg_secondary"], fg=self.theme["text_primary"]).pack(anchor="w")
        
        sel_frame = tk.Frame(right_frame)
        sel_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        
        self.sel_listbox = tk.Listbox(sel_frame, height=10, width=20, font=FONTS.get_font("small"),
                                      selectmode=tk.EXTENDED, exportselection=False, bg="white")
        self.sel_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sel_sb = ttk.Scrollbar(sel_frame, command=self.sel_listbox.yview)
        sel_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.sel_listbox.config(yscrollcommand=sel_sb.set)
        
        self.var_count_label = tk.Label(right_frame, text="Selected: 0", font=FONTS.get_font("small"),
                                        bg=self.theme["bg_secondary"], fg=self.theme["text_muted"])
        self.var_count_label.pack(anchor="w")
        
        # Options frame
        opt_frame = tk.LabelFrame(self.options_content, text="Options", font=FONTS.get_font("body"),
                                  bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        opt_frame.pack(fill=tk.X, padx=8, pady=8)
        
        # Correlation method
        method_frame = tk.Frame(opt_frame, bg=self.theme["bg_secondary"])
        method_frame.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(method_frame, text="Coefficient:", font=FONTS.get_font("body"),
                bg=self.theme["bg_secondary"], fg=self.theme["text_primary"]).pack(side=tk.LEFT)
        self.method_var = tk.StringVar(value="pearson")
        ttk.Combobox(method_frame, textvariable=self.method_var, 
                    values=["pearson", "spearman", "kendall"],
                    state="readonly", width=12).pack(side=tk.LEFT, padx=8)
        
        # Options checkboxes
        self.show_pvalues_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_frame, text="Show p-values", variable=self.show_pvalues_var,
                      font=FONTS.get_font("body"), bg=self.theme["bg_secondary"],
                      selectcolor="white").pack(anchor="w", padx=8, pady=2)
        
        self.show_n_var = tk.BooleanVar(value=False)
        tk.Checkbutton(opt_frame, text="Show sample sizes", variable=self.show_n_var,
                      font=FONTS.get_font("body"), bg=self.theme["bg_secondary"],
                      selectcolor="white").pack(anchor="w", padx=8, pady=2)
        
        self.flag_sig_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_frame, text="Flag significant correlations", variable=self.flag_sig_var,
                      font=FONTS.get_font("body"), bg=self.theme["bg_secondary"],
                      selectcolor="white").pack(anchor="w", padx=8, pady=2)
        
        # Alpha
        alpha_frame = tk.Frame(opt_frame, bg=self.theme["bg_secondary"])
        alpha_frame.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(alpha_frame, text="Alpha:", font=FONTS.get_font("body"),
                bg=self.theme["bg_secondary"], fg=self.theme["text_primary"]).pack(side=tk.LEFT)
        self.alpha_var = tk.StringVar(value="0.05")
        ttk.Combobox(alpha_frame, textvariable=self.alpha_var, values=["0.01","0.05","0.10"],
                    state="readonly", width=6).pack(side=tk.LEFT, padx=8)
        
        # Plot options
        plot_frame = tk.LabelFrame(self.options_content, text="Visualization", font=FONTS.get_font("body"),
                                   bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        plot_frame.pack(fill=tk.X, padx=8, pady=4)
        
        self.plot_type_var = tk.StringVar(value="heatmap")
        tk.Radiobutton(plot_frame, text="Correlation Heatmap", variable=self.plot_type_var,
                      value="heatmap", font=FONTS.get_font("body"), bg=self.theme["bg_secondary"],
                      selectcolor="white").pack(anchor="w", padx=8, pady=2)
        tk.Radiobutton(plot_frame, text="Scatter Matrix", variable=self.plot_type_var,
                      value="scatter", font=FONTS.get_font("body"), bg=self.theme["bg_secondary"],
                      selectcolor="white").pack(anchor="w", padx=8, pady=2)
        
        # Buttons
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=8)
        
        ActionButton(btn_frame, text="OK - Run Analysis", icon="▶", command=self._run_analysis,
                    theme=self.theme_name).pack(fill=tk.X)
        ActionButton(btn_frame, text="Reset", icon="🔄", command=self._reset_selection,
                    theme=self.theme_name).pack(fill=tk.X, pady=(4,0))
        ActionButton(btn_frame, text="Export Results", icon="📊", command=self._export_results,
                    theme=self.theme_name).pack(fill=tk.X, pady=(4,0))
        
        # Populate if data already loaded
        if hasattr(self, 'bib') and self.bib is not None and hasattr(self.bib, 'df'):
            self._populate_variables()
    
    def _add_variables(self):
        selection = self.avail_listbox.curselection()
        if not selection:
            return
        for idx in reversed(selection):
            var = self.avail_listbox.get(idx)
            self.sel_listbox.insert(tk.END, var)
            self.avail_listbox.delete(idx)
        self._update_var_count()
    
    def _remove_variables(self):
        selection = self.sel_listbox.curselection()
        if not selection:
            return
        for idx in reversed(selection):
            var = self.sel_listbox.get(idx)
            self.avail_listbox.insert(tk.END, var)
            self.sel_listbox.delete(idx)
        self._update_var_count()
    
    def _add_all_variables(self):
        while self.avail_listbox.size() > 0:
            var = self.avail_listbox.get(0)
            self.sel_listbox.insert(tk.END, var)
            self.avail_listbox.delete(0)
        self._update_var_count()
    
    def _clear_variables(self):
        while self.sel_listbox.size() > 0:
            var = self.sel_listbox.get(0)
            self.avail_listbox.insert(tk.END, var)
            self.sel_listbox.delete(0)
        self._update_var_count()
    
    def _update_var_count(self):
        count = self.sel_listbox.size()
        self.var_count_label.config(text=f"Selected: {count}")
    
    def _reset_selection(self):
        self._clear_variables()
        self._result = None
        self._current_fig = None
        self._show_placeholder()
    
    def _create_results(self):
        self.results_card = tk.Frame(self.results_frame, bg=self.theme["bg_card"],
                                     highlightbackground=self.theme["border"], highlightthickness=1)
        self.results_card.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        self.notebook = ttk.Notebook(self.results_card)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        self.matrix_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.matrix_frame, text="📊 Correlation Matrix")
        
        self.pvalues_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.pvalues_frame, text="📋 P-values")
        
        self.plot_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.plot_frame, text="📈 Visualization")
        
        self.pairs_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.pairs_frame, text="🔗 Pairs")
        
        self.summary_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.summary_frame, text="📝 Summary")

        

        # Info tab

        info_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])

        self.notebook.add(info_frame, text="ℹ️ Info")

        self._create_info_content(info_frame)
        
        self._show_placeholder()
    
    def _show_placeholder(self):
        for f in [self.matrix_frame, self.pvalues_frame, self.plot_frame, self.pairs_frame, self.summary_frame]:
            for w in f.winfo_children():
                w.destroy()
        tk.Label(self.matrix_frame, 
                text="Select numeric variables to correlate\nthen click 'OK - Run Analysis'\n\n"
                     "Coefficients:\n• Pearson - linear relationship\n• Spearman - monotonic (rank-based)\n• Kendall - concordance",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"],
                justify=tk.CENTER).pack(expand=True)
    
    def _run_analysis(self):
        if not self.bib:
            messagebox.showwarning("No Data", "Load a dataset first.")
            return
        if self.sel_listbox.size() < 2:
            messagebox.showwarning("Required", "Select at least 2 variables.")
            return
        
        variables = [self.sel_listbox.get(i) for i in range(self.sel_listbox.size())]
        method = self.method_var.get()
        alpha = float(self.alpha_var.get())
        
        try:
            from biblium.correlation import compute_correlation
            
            result = compute_correlation(self.bib.df, variables=variables, method=method, 
                                        alpha=alpha, verbose=True)
            self._result = result
            
            self._update_matrix()
            self._update_pvalues()
            self._update_plot()
            self._update_pairs()
            self._update_summary()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Analysis failed:\n{e}")
    
    def _update_matrix(self):
        for w in self.matrix_frame.winfo_children():
            w.destroy()
        if not self._result:
            return
        
        r = self._result
        
        tk.Label(self.matrix_frame, text=f"{r.method_name} Correlation Matrix",
                font=FONTS.get_font("heading", bold=True), bg=self.theme["bg_card"],
                fg=self.theme["text_primary"]).pack(pady=(8,4), anchor="w", padx=8)
        
        tk.Label(self.matrix_frame, text=f"Variables: {r.n_vars}  |  Significant pairs: {r.n_significant}/{r.n_total_pairs}",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                fg=self.theme["text_secondary"]).pack(anchor="w", padx=8)
        
        # Create table
        tree_frame = tk.Frame(self.matrix_frame, bg=self.theme["bg_card"])
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        cols = ["Variable"] + r.variables
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=min(15, r.n_vars+1))
        
        for c in cols:
            tree.heading(c, text=str(c))
            tree.column(c, width=90, minwidth=70, anchor="center")
        tree.column("Variable", anchor="w", width=120)
        
        alpha = float(self.alpha_var.get())
        
        for var in r.variables:
            values = [var]
            for var2 in r.variables:
                corr = r.corr_matrix.loc[var, var2]
                p = r.p_matrix.loc[var, var2]
                
                if var == var2:
                    text = "1.000"
                elif np.isnan(corr):
                    text = "-"
                else:
                    text = f"{corr:.3f}"
                    if self.flag_sig_var.get() and p < alpha:
                        text += " *"
                values.append(text)
            tree.insert("", tk.END, values=values)
        
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        if self.flag_sig_var.get():
            tk.Label(self.matrix_frame, text=f"* p < {alpha}",
                    font=FONTS.get_font("small"), bg=self.theme["bg_card"],
                    fg=self.theme["text_muted"]).pack(anchor="w", padx=8)
    
    def _update_pvalues(self):
        for w in self.pvalues_frame.winfo_children():
            w.destroy()
        if not self._result:
            return
        
        r = self._result
        
        tk.Label(self.pvalues_frame, text="P-value Matrix",
                font=FONTS.get_font("heading", bold=True), bg=self.theme["bg_card"],
                fg=self.theme["text_primary"]).pack(pady=(8,4), anchor="w", padx=8)
        
        # Create table
        tree_frame = tk.Frame(self.pvalues_frame, bg=self.theme["bg_card"])
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        cols = ["Variable"] + r.variables
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=min(15, r.n_vars+1))
        
        for c in cols:
            tree.heading(c, text=str(c))
            tree.column(c, width=90, minwidth=70, anchor="center")
        tree.column("Variable", anchor="w", width=120)
        
        alpha = float(self.alpha_var.get())
        
        for var in r.variables:
            values = [var]
            for var2 in r.variables:
                p = r.p_matrix.loc[var, var2]
                
                if var == var2:
                    text = "-"
                elif np.isnan(p):
                    text = "-"
                else:
                    text = f"{p:.4f}"
                values.append(text)
            tree.insert("", tk.END, values=values)
        
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
    
    def _update_plot(self):
        for w in self.plot_frame.winfo_children():
            w.destroy()
        if not self._result or not HAS_MATPLOTLIB:
            if not HAS_MATPLOTLIB:
                tk.Label(self.plot_frame, text="Matplotlib not available",
                        font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                        fg=self.theme["text_muted"]).pack(expand=True)
            return
        
        r = self._result
        plot_type = self.plot_type_var.get()
        
        try:
            if plot_type == "heatmap":
                from biblium.correlation import plot_correlation_matrix
                fig = plot_correlation_matrix(r, show_values=True, 
                                             show_significance=self.flag_sig_var.get())
            else:  # scatter
                from biblium.correlation import plot_scatter_matrix
                fig = plot_scatter_matrix(self.bib.df, r.variables)
            
            self._current_fig = fig
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            _canvas_widget = canvas.get_tk_widget()

            _canvas_widget.pack(fill=tk.BOTH, expand=True)
            add_plot_context_menu(canvas_widget, fig)

            add_plot_context_menu(_canvas_widget, fig)
            
            # Export button
            btn_frame = tk.Frame(self.plot_frame, bg=self.theme["bg_card"])
            btn_frame.pack(fill=tk.X, padx=8, pady=4)
            tk.Button(btn_frame, text="Save Plot", command=self._save_plot).pack(side=tk.LEFT)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            tk.Label(self.plot_frame, text=f"Plot error: {e}",
                    font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                    fg="red").pack(expand=True)
    
    def _update_pairs(self):
        for w in self.pairs_frame.winfo_children():
            w.destroy()
        if not self._result:
            return
        
        r = self._result
        
        tk.Label(self.pairs_frame, text="Pairwise Correlations (sorted by |r|)",
                font=FONTS.get_font("heading", bold=True), bg=self.theme["bg_card"],
                fg=self.theme["text_primary"]).pack(pady=(8,4), anchor="w", padx=8)
        
        # Create table
        tree_frame = tk.Frame(self.pairs_frame, bg=self.theme["bg_card"])
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        cols = ("Variable 1", "Variable 2", "r", "p-value", "N", "Interpretation", "Sig.")
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=15)
        
        for c in cols:
            tree.heading(c, text=c)
            if c in ["Variable 1", "Variable 2"]:
                tree.column(c, width=120, minwidth=100, anchor="w")
            elif c == "Interpretation":
                tree.column(c, width=100, minwidth=80, anchor="center")
            else:
                tree.column(c, width=80, minwidth=60, anchor="center")
        
        from biblium.correlation import get_correlation_interpretation
        
        for p in r.pairs:
            interp = get_correlation_interpretation(p.correlation)
            sig = "Yes *" if p.is_significant else "No"
            tree.insert("", tk.END, values=(
                p.var1, p.var2, f"{p.correlation:.4f}", f"{p.p_value:.4f}",
                p.n, interp, sig
            ))
        
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _update_summary(self):
        for w in self.summary_frame.winfo_children():
            w.destroy()
        if not self._result:
            return
        
        r = self._result
        
        from biblium.correlation import get_correlation_interpretation
        
        txt = "CORRELATION ANALYSIS REPORT\n" + "="*50 + "\n\n"
        txt += f"Method: {r.method_name}\n"
        txt += f"Variables: {r.n_vars}\n"
        txt += f"Alpha: {r.alpha}\n"
        txt += f"Significant pairs: {r.n_significant}/{r.n_total_pairs}\n\n"
        
        txt += "CORRELATION MATRIX\n" + "-"*30 + "\n"
        txt += r.corr_matrix.round(3).to_string() + "\n\n"
        
        txt += "P-VALUE MATRIX\n" + "-"*30 + "\n"
        txt += r.p_matrix.round(4).to_string() + "\n\n"
        
        txt += "SIGNIFICANT CORRELATIONS\n" + "-"*30 + "\n"
        sig_pairs = [p for p in r.pairs if p.is_significant]
        if sig_pairs:
            for p in sig_pairs:
                interp = get_correlation_interpretation(p.correlation)
                txt += f"  {p.var1} × {p.var2}: r = {p.correlation:.4f}, p = {p.p_value:.4f} ({interp})\n"
        else:
            txt += "  No significant correlations found.\n"
        
        txt += "\nINTERPRETATION GUIDE\n" + "-"*30 + "\n"
        txt += "  |r| < 0.1: Negligible\n"
        txt += "  |r| < 0.3: Weak\n"
        txt += "  |r| < 0.5: Moderate\n"
        txt += "  |r| < 0.7: Strong\n"
        txt += "  |r| >= 0.7: Very strong\n"
        
        text_w = tk.Text(self.summary_frame, font=FONTS.get_font("code"), bg=self.theme["bg_card"],
                        fg=self.theme["text_primary"], wrap=tk.WORD)
        text_w.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        text_w.insert("1.0", txt)
        text_w.config(state=tk.DISABLED)
    
    def _save_plot(self):
        if not self._current_fig:
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")]
        )
        if filepath:
            self._current_fig.savefig(filepath, dpi=150, bbox_inches='tight')
            messagebox.showinfo("Saved", f"Plot saved to:\n{filepath}")
    
    def _export_results(self):
        if not self._result:
            messagebox.showwarning("No Results", "Run analysis first.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                filetypes=[("Excel","*.xlsx"),("Text","*.txt")])
        if not filepath:
            return
        try:
            r = self._result
            if filepath.endswith('.xlsx'):
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    r.corr_matrix.to_excel(writer, sheet_name='Correlations')
                    r.p_matrix.to_excel(writer, sheet_name='P-values')
                    r.n_matrix.to_excel(writer, sheet_name='Sample_Sizes')
                    
                    # Pairs
                    pairs_data = [{
                        'Variable_1': p.var1, 'Variable_2': p.var2,
                        'Correlation': p.correlation, 'P-value': p.p_value,
                        'N': p.n, 'Significant': p.is_significant
                    } for p in r.pairs]
                    pd.DataFrame(pairs_data).to_excel(writer, sheet_name='Pairs', index=False)
            else:
                with open(filepath, 'w') as f:
                    for w in self.summary_frame.winfo_children():
                        if isinstance(w, tk.Text):
                            f.write(w.get("1.0", tk.END))
            messagebox.showinfo("Exported", f"Saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{e}")
    
    def _populate_variables(self):
        self.avail_listbox.delete(0, tk.END)
        self.sel_listbox.delete(0, tk.END)
        if self.bib is not None and hasattr(self.bib, 'df'):
            # Only numeric columns
            for col in self.bib.df.columns:
                if pd.api.types.is_numeric_dtype(self.bib.df[col]):
                    self.avail_listbox.insert(tk.END, col)
        self._update_var_count()
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
CORRELATION ANALYSIS
════════════════════

Examine relationships between numeric variables.

CORRELATION METHODS
───────────────────
• Pearson (r)
  - Linear relationships
  - Assumes normality
  - Sensitive to outliers
  
• Spearman (ρ)
  - Rank-based
  - Monotonic relationships
  - Robust to outliers
  
• Kendall (τ)
  - Concordance-based
  - More robust for small n
  - Conservative estimates

INTERPRETING STRENGTH
─────────────────────
|r| = 0.0-0.1: Negligible
|r| = 0.1-0.3: Weak
|r| = 0.3-0.5: Moderate
|r| = 0.5-0.7: Strong
|r| = 0.7-1.0: Very strong

OUTPUT
──────
• Correlation matrix
• Heatmap visualization
• Scatter matrix
• Significance flags (p < 0.05)

COMMON CORRELATIONS
───────────────────
• Citations × Authors: Often positive
• Citations × References: Moderate
• Age × Citations: Complex relationship
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

    def on_data_loaded(self, bib):
        if isinstance(bib, dict):
            bib = bib.get("bib", bib)
        self.bib = bib
        self._result = None
        self._current_fig = None
        self._populate_variables()
        self._show_placeholder()
