# -*- coding: utf-8 -*-
"""
Crosstabs Panel - Contingency Table Analysis
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from biblium.gui.config import FONTS, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.buttons import ActionButton

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class CrosstabsPanel(BasePanel):
    """Panel for contingency table analysis - SPSS-like interface."""
    
    title = "Crosstabs"
    icon = "📋"
    requires_data = True
    
    def __init__(self, parent, theme="light", **kwargs):
        self._result = None
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
        
        tk.Label(left_frame, text="Available Variables:", font=FONTS.get_font("body", bold=True),
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
        
        tk.Label(mid_frame, text="Row:", font=FONTS.get_font("small"),
                bg=self.theme["bg_secondary"], fg=self.theme["text_muted"]).pack()
        tk.Button(mid_frame, text="▶", width=3, command=self._add_row).pack(pady=2)
        tk.Button(mid_frame, text="◀", width=3, command=self._remove_row).pack(pady=2)
        
        tk.Label(mid_frame, text="Column:", font=FONTS.get_font("small"),
                bg=self.theme["bg_secondary"], fg=self.theme["text_muted"]).pack(pady=(12,0))
        tk.Button(mid_frame, text="▶", width=3, command=self._add_col).pack(pady=2)
        tk.Button(mid_frame, text="◀", width=3, command=self._remove_col).pack(pady=2)
        
        # Right: Selected variables
        right_frame = tk.Frame(var_frame, bg=self.theme["bg_secondary"])
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tk.Label(right_frame, text="Row Variable:", font=FONTS.get_font("body", bold=True),
                bg=self.theme["bg_secondary"], fg=self.theme["text_primary"]).pack(anchor="w")
        self.row_listbox = tk.Listbox(right_frame, height=2, width=20, font=FONTS.get_font("small"),
                                      selectmode=tk.SINGLE, exportselection=False, bg="white")
        self.row_listbox.pack(fill=tk.X, pady=4)
        
        self.row_info_label = tk.Label(right_frame, text="", font=FONTS.get_font("small"),
                                       bg=self.theme["bg_secondary"], fg=self.theme["text_muted"])
        self.row_info_label.pack(anchor="w")
        
        tk.Label(right_frame, text="Column Variable:", font=FONTS.get_font("body", bold=True),
                bg=self.theme["bg_secondary"], fg=self.theme["text_primary"]).pack(anchor="w", pady=(8,0))
        self.col_listbox = tk.Listbox(right_frame, height=2, width=20, font=FONTS.get_font("small"),
                                      selectmode=tk.SINGLE, exportselection=False, bg="white")
        self.col_listbox.pack(fill=tk.X, pady=4)
        
        self.col_info_label = tk.Label(right_frame, text="", font=FONTS.get_font("small"),
                                       bg=self.theme["bg_secondary"], fg=self.theme["text_muted"])
        self.col_info_label.pack(anchor="w")
        
        # Options frame
        opt_frame = tk.LabelFrame(self.options_content, text="Options", font=FONTS.get_font("body"),
                                  bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        opt_frame.pack(fill=tk.X, padx=8, pady=8)
        
        # Statistics checkboxes
        self.chi_squared_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_frame, text="Chi-squared test", variable=self.chi_squared_var,
                      font=FONTS.get_font("body"), bg=self.theme["bg_secondary"],
                      selectcolor="white").pack(anchor="w", padx=8, pady=2)
        
        self.fisher_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_frame, text="Fisher's exact test (2×2)", variable=self.fisher_var,
                      font=FONTS.get_font("body"), bg=self.theme["bg_secondary"],
                      selectcolor="white").pack(anchor="w", padx=8, pady=2)
        
        self.effect_size_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_frame, text="Effect size (Cramér's V, Phi)", variable=self.effect_size_var,
                      font=FONTS.get_font("body"), bg=self.theme["bg_secondary"],
                      selectcolor="white").pack(anchor="w", padx=8, pady=2)
        
        self.expected_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_frame, text="Expected frequencies", variable=self.expected_var,
                      font=FONTS.get_font("body"), bg=self.theme["bg_secondary"],
                      selectcolor="white").pack(anchor="w", padx=8, pady=2)
        
        self.residuals_var = tk.BooleanVar(value=False)
        tk.Checkbutton(opt_frame, text="Standardized residuals", variable=self.residuals_var,
                      font=FONTS.get_font("body"), bg=self.theme["bg_secondary"],
                      selectcolor="white").pack(anchor="w", padx=8, pady=2)
        
        # Percentages
        pct_frame = tk.Frame(opt_frame, bg=self.theme["bg_secondary"])
        pct_frame.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(pct_frame, text="Show percentages:", font=FONTS.get_font("body"),
                bg=self.theme["bg_secondary"], fg=self.theme["text_primary"]).pack(side=tk.LEFT)
        self.pct_var = tk.StringVar(value="Row")
        ttk.Combobox(pct_frame, textvariable=self.pct_var, 
                    values=["None", "Row", "Column", "Total"],
                    state="readonly", width=10).pack(side=tk.LEFT, padx=8)
        
        # Alpha
        alpha_frame = tk.Frame(opt_frame, bg=self.theme["bg_secondary"])
        alpha_frame.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(alpha_frame, text="Alpha:", font=FONTS.get_font("body"),
                bg=self.theme["bg_secondary"], fg=self.theme["text_primary"]).pack(side=tk.LEFT)
        self.alpha_var = tk.StringVar(value="0.05")
        ttk.Combobox(alpha_frame, textvariable=self.alpha_var, values=["0.01","0.05","0.10"],
                    state="readonly", width=6).pack(side=tk.LEFT, padx=8)
        
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
    
    def _add_row(self):
        selection = self.avail_listbox.curselection()
        if not selection:
            return
        if self.row_listbox.size() > 0:
            messagebox.showinfo("Info", "Only one row variable allowed. Remove current first.")
            return
        idx = selection[0]
        var = self.avail_listbox.get(idx)
        self.row_listbox.insert(tk.END, var)
        self.avail_listbox.delete(idx)
        self._update_var_info()
    
    def _remove_row(self):
        if self.row_listbox.size() > 0:
            var = self.row_listbox.get(0)
            self.avail_listbox.insert(tk.END, var)
            self.row_listbox.delete(0)
            self.row_info_label.config(text="")
    
    def _add_col(self):
        selection = self.avail_listbox.curselection()
        if not selection:
            return
        if self.col_listbox.size() > 0:
            messagebox.showinfo("Info", "Only one column variable allowed. Remove current first.")
            return
        idx = selection[0]
        var = self.avail_listbox.get(idx)
        self.col_listbox.insert(tk.END, var)
        self.avail_listbox.delete(idx)
        self._update_var_info()
    
    def _remove_col(self):
        if self.col_listbox.size() > 0:
            var = self.col_listbox.get(0)
            self.avail_listbox.insert(tk.END, var)
            self.col_listbox.delete(0)
            self.col_info_label.config(text="")
    
    def _update_var_info(self):
        if self.row_listbox.size() > 0 and self.bib:
            var = self.row_listbox.get(0)
            n = self.bib.df[var].nunique()
            vals = self.bib.df[var].value_counts()
            info = f"Categories: {n} | " + ", ".join([f"{k}" for k in vals.head(4).index])
            if n > 4:
                info += "..."
            self.row_info_label.config(text=info)
        
        if self.col_listbox.size() > 0 and self.bib:
            var = self.col_listbox.get(0)
            n = self.bib.df[var].nunique()
            vals = self.bib.df[var].value_counts()
            info = f"Categories: {n} | " + ", ".join([f"{k}" for k in vals.head(4).index])
            if n > 4:
                info += "..."
            self.col_info_label.config(text=info)
    
    def _reset_selection(self):
        self._remove_row()
        self._remove_col()
        self._result = None
        self._show_placeholder()
    
    def _create_results(self):
        self.results_card = tk.Frame(self.results_frame, bg=self.theme["bg_card"],
                                     highlightbackground=self.theme["border"], highlightthickness=1)
        self.results_card.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        self.notebook = ttk.Notebook(self.results_card)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        self.crosstab_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.crosstab_frame, text="📋 Crosstab")
        
        self.stats_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.stats_frame, text="🧪 Statistics")
        
        self.expected_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.expected_frame, text="📊 Expected")
        
        self.summary_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.summary_frame, text="📝 Summary")

        

        # Info tab

        info_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])

        self.notebook.add(info_frame, text="ℹ️ Info")

        self._create_info_content(info_frame)
        
        self._show_placeholder()
    
    def _show_placeholder(self):
        for f in [self.crosstab_frame, self.stats_frame, self.expected_frame, self.summary_frame]:
            for w in f.winfo_children():
                w.destroy()
        tk.Label(self.crosstab_frame, 
                text="Select Row and Column variables\nthen click 'OK - Run Analysis'\n\nAnalyzes association between two categorical variables.",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"],
                justify=tk.CENTER).pack(expand=True)
    
    def _run_analysis(self):
        if not self.bib:
            messagebox.showwarning("No Data", "Load a dataset first.")
            return
        if self.row_listbox.size() == 0:
            messagebox.showwarning("Required", "Select a row variable.")
            return
        if self.col_listbox.size() == 0:
            messagebox.showwarning("Required", "Select a column variable.")
            return
        
        row_var = self.row_listbox.get(0)
        col_var = self.col_listbox.get(0)
        alpha = float(self.alpha_var.get())
        
        try:
            from biblium.crosstabs import compute_crosstab
            
            result = compute_crosstab(self.bib.df, row_var, col_var, alpha=alpha, verbose=True)
            self._result = result
            
            self._update_crosstab()
            self._update_statistics()
            self._update_expected()
            self._update_summary()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Analysis failed:\n{e}")
    
    def _update_crosstab(self):
        for w in self.crosstab_frame.winfo_children():
            w.destroy()
        if not self._result:
            return
        
        r = self._result
        
        tk.Label(self.crosstab_frame, text=f"Crosstab: {r.row_var} × {r.col_var}",
                font=FONTS.get_font("heading", bold=True), bg=self.theme["bg_card"],
                fg=self.theme["text_primary"]).pack(pady=(8,4), anchor="w", padx=8)
        
        tk.Label(self.crosstab_frame, text=f"N = {r.n_total}  |  {r.n_rows} × {r.n_cols} table",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                fg=self.theme["text_secondary"]).pack(anchor="w", padx=8)
        
        # Create table
        tree_frame = tk.Frame(self.crosstab_frame, bg=self.theme["bg_card"])
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        observed = r.observed
        cols = [r.row_var] + list(observed.columns)
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=min(15, len(observed)+1))
        
        for c in cols:
            tree.heading(c, text=str(c))
            tree.column(c, width=100, minwidth=80, anchor="center")
        tree.column(cols[0], anchor="w", width=120)
        
        # Get percentages based on selection
        pct_type = self.pct_var.get().lower()
        
        for idx in observed.index:
            values = [str(idx)]
            for col in observed.columns:
                count = observed.loc[idx, col]
                pct_value = None
                
                # Try to get percentage, handle missing indices gracefully
                try:
                    if pct_type == "row" and r.row_pct is not None and idx in r.row_pct.index and col in r.row_pct.columns:
                        pct_value = r.row_pct.loc[idx, col]
                    elif pct_type == "column" and r.col_pct is not None and idx in r.col_pct.index and col in r.col_pct.columns:
                        pct_value = r.col_pct.loc[idx, col]
                    elif pct_type == "total" and r.total_pct is not None and idx in r.total_pct.index and col in r.total_pct.columns:
                        pct_value = r.total_pct.loc[idx, col]
                except (KeyError, TypeError):
                    pct_value = None
                
                if pct_value is not None:
                    values.append(f"{count} ({pct_value:.1f}%)")
                else:
                    values.append(str(count))
            
            tag = 'total' if idx == 'Total' else ''
            tree.insert("", tk.END, values=values, tags=(tag,))
        
        tree.tag_configure('total', background='#e8f4f8')
        
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
    
    def _update_statistics(self):
        for w in self.stats_frame.winfo_children():
            w.destroy()
        if not self._result:
            return
        
        r = self._result
        
        canvas = tk.Canvas(self.stats_frame, bg=self.theme["bg_card"], highlightthickness=0)
        sb = ttk.Scrollbar(self.stats_frame, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=self.theme["bg_card"])
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        
        tk.Label(inner, text="Statistical Tests", font=FONTS.get_font("heading", bold=True),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(pady=(8,4), anchor="w", padx=8)
        
        # Chi-squared test
        if self.chi_squared_var.get():
            chi_frame = tk.LabelFrame(inner, text="Chi-Squared Test", font=FONTS.get_font("body"),
                                      bg=self.theme["bg_card"], fg=self.theme["text_primary"])
            chi_frame.pack(fill=tk.X, padx=8, pady=4)
            
            c = r.chi_squared
            chi_text = f"χ²({c.df}) = {c.statistic:.4f}\np-value = {c.p_value:.4f}\n"
            chi_text += f"Result: {'Significant *' if c.is_significant else 'Not significant'}"
            
            chi_color = "green" if c.is_significant else self.theme["text_primary"]
            tk.Label(chi_frame, text=chi_text, font=FONTS.get_font("code"),
                    bg=self.theme["bg_card"], fg=chi_color, justify=tk.LEFT).pack(padx=8, pady=4, anchor="w")
            
            if c.warning:
                tk.Label(chi_frame, text=f"⚠️ {c.warning}", font=FONTS.get_font("small"),
                        bg=self.theme["bg_card"], fg="orange", wraplength=400,
                        justify=tk.LEFT).pack(padx=8, pady=4, anchor="w")
        
        # Fisher's exact test (2x2 only)
        if self.fisher_var.get() and r.is_2x2 and r.fisher:
            fish_frame = tk.LabelFrame(inner, text="Fisher's Exact Test (2×2)", font=FONTS.get_font("body"),
                                       bg=self.theme["bg_card"], fg=self.theme["text_primary"])
            fish_frame.pack(fill=tk.X, padx=8, pady=4)
            
            f = r.fisher
            fish_text = f"p-value = {f.p_value:.4f}\n"
            fish_text += f"Odds Ratio = {f.odds_ratio:.3f}"
            if f.ci_lower > 0 and f.ci_upper > 0:
                fish_text += f"\n95% CI: [{f.ci_lower:.3f}, {f.ci_upper:.3f}]"
            fish_text += f"\nResult: {'Significant *' if f.is_significant else 'Not significant'}"
            
            fish_color = "green" if f.is_significant else self.theme["text_primary"]
            tk.Label(fish_frame, text=fish_text, font=FONTS.get_font("code"),
                    bg=self.theme["bg_card"], fg=fish_color, justify=tk.LEFT).pack(padx=8, pady=4, anchor="w")
        
        # Effect sizes
        if self.effect_size_var.get():
            eff_frame = tk.LabelFrame(inner, text="Effect Size", font=FONTS.get_font("body"),
                                      bg=self.theme["bg_card"], fg=self.theme["text_primary"])
            eff_frame.pack(fill=tk.X, padx=8, pady=4)
            
            e = r.effect_size
            eff_text = f"Cramér's V = {e.cramers_v:.4f} ({e.cramers_v_interpretation})\n"
            if r.is_2x2:
                eff_text += f"Phi (φ) = {e.phi:.4f} ({e.phi_interpretation})\n"
            eff_text += f"Contingency Coefficient = {e.contingency_coef:.4f}"
            
            tk.Label(eff_frame, text=eff_text, font=FONTS.get_font("code"),
                    bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                    justify=tk.LEFT).pack(padx=8, pady=4, anchor="w")
    
    def _update_expected(self):
        for w in self.expected_frame.winfo_children():
            w.destroy()
        if not self._result:
            return
        
        r = self._result
        
        canvas = tk.Canvas(self.expected_frame, bg=self.theme["bg_card"], highlightthickness=0)
        sb = ttk.Scrollbar(self.expected_frame, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=self.theme["bg_card"])
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        
        # Expected frequencies
        if self.expected_var.get() and r.expected is not None:
            tk.Label(inner, text="Expected Frequencies", font=FONTS.get_font("heading", bold=True),
                    bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(pady=(8,4), anchor="w", padx=8)
            
            exp_frame = tk.Frame(inner, bg=self.theme["bg_card"])
            exp_frame.pack(fill=tk.X, padx=8, pady=4)
            
            expected = r.expected
            cols = [r.row_var] + list(expected.columns)
            tree = ttk.Treeview(exp_frame, columns=cols, show="headings", height=min(10, len(expected)+1))
            
            for c in cols:
                tree.heading(c, text=str(c))
                tree.column(c, width=100, anchor="center")
            tree.column(cols[0], anchor="w", width=120)
            
            for idx in expected.index:
                values = [str(idx)] + [f"{expected.loc[idx, col]:.2f}" for col in expected.columns]
                tree.insert("", tk.END, values=values)
            
            tree.pack(fill=tk.X)
            
            # Check for cells < 5
            c = r.chi_squared
            tk.Label(inner, text=f"Min expected: {c.min_expected:.2f}  |  Cells < 5: {c.cells_below_5}",
                    font=FONTS.get_font("small"), bg=self.theme["bg_card"],
                    fg="orange" if c.cells_below_5 > 0 else self.theme["text_secondary"]).pack(anchor="w", padx=8, pady=4)
        
        # Standardized residuals
        if self.residuals_var.get() and r.residuals is not None:
            tk.Label(inner, text="Standardized Residuals", font=FONTS.get_font("heading", bold=True),
                    bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(pady=(16,4), anchor="w", padx=8)
            
            tk.Label(inner, text="Values > |1.96| indicate significant deviation from expected",
                    font=FONTS.get_font("small"), bg=self.theme["bg_card"],
                    fg=self.theme["text_muted"]).pack(anchor="w", padx=8)
            
            res_frame = tk.Frame(inner, bg=self.theme["bg_card"])
            res_frame.pack(fill=tk.X, padx=8, pady=4)
            
            residuals = r.residuals
            cols = [r.row_var] + list(residuals.columns)
            tree = ttk.Treeview(res_frame, columns=cols, show="headings", height=min(10, len(residuals)+1))
            
            for c in cols:
                tree.heading(c, text=str(c))
                tree.column(c, width=100, anchor="center")
            tree.column(cols[0], anchor="w", width=120)
            
            for idx in residuals.index:
                values = [str(idx)] + [f"{residuals.loc[idx, col]:.3f}" for col in residuals.columns]
                tree.insert("", tk.END, values=values)
            
            tree.pack(fill=tk.X)
    
    def _update_summary(self):
        for w in self.summary_frame.winfo_children():
            w.destroy()
        if not self._result:
            return
        
        r = self._result
        
        txt = "CROSSTABS ANALYSIS REPORT\n" + "="*50 + "\n\n"
        txt += f"Row Variable: {r.row_var}\n"
        txt += f"Column Variable: {r.col_var}\n"
        txt += f"Table Size: {r.n_rows} × {r.n_cols}\n"
        txt += f"Total N: {r.n_total}\n\n"
        
        txt += "OBSERVED FREQUENCIES\n" + "-"*30 + "\n"
        txt += r.observed.to_string() + "\n\n"
        
        txt += "STATISTICAL TESTS\n" + "-"*30 + "\n"
        c = r.chi_squared
        txt += f"Chi-Squared: χ²({c.df}) = {c.statistic:.4f}, p = {c.p_value:.4f}\n"
        
        if r.is_2x2 and r.fisher:
            f = r.fisher
            txt += f"Fisher's Exact: p = {f.p_value:.4f}, OR = {f.odds_ratio:.3f}\n"
        
        txt += f"\nEFFECT SIZE\n" + "-"*30 + "\n"
        e = r.effect_size
        txt += f"Cramér's V = {e.cramers_v:.4f} ({e.cramers_v_interpretation})\n"
        if r.is_2x2:
            txt += f"Phi = {e.phi:.4f} ({e.phi_interpretation})\n"
        
        txt += f"\nINTERPRETATION\n" + "-"*30 + "\n"
        txt += r.interpretation + "\n"
        
        text_w = tk.Text(self.summary_frame, font=FONTS.get_font("code"), bg=self.theme["bg_card"],
                        fg=self.theme["text_primary"], wrap=tk.WORD)
        text_w.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        text_w.insert("1.0", txt)
        text_w.config(state=tk.DISABLED)
    
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
                    r.observed.to_excel(writer, sheet_name='Observed')
                    if r.expected is not None:
                        r.expected.to_excel(writer, sheet_name='Expected')
                    if r.row_pct is not None:
                        r.row_pct.to_excel(writer, sheet_name='Row_Pct')
                    if r.col_pct is not None:
                        r.col_pct.to_excel(writer, sheet_name='Col_Pct')
                    if r.residuals is not None:
                        r.residuals.to_excel(writer, sheet_name='Residuals')
                    
                    # Statistics summary
                    stats_data = {
                        'Test': ['Chi-Squared', 'df', 'p-value', "Cramér's V"],
                        'Value': [r.chi_squared.statistic, r.chi_squared.df, 
                                 r.chi_squared.p_value, r.effect_size.cramers_v]
                    }
                    if r.is_2x2 and r.fisher:
                        stats_data['Test'].extend(['Fisher p-value', 'Odds Ratio', 'Phi'])
                        stats_data['Value'].extend([r.fisher.p_value, r.fisher.odds_ratio, r.effect_size.phi])
                    pd.DataFrame(stats_data).to_excel(writer, sheet_name='Statistics', index=False)
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
        self.row_listbox.delete(0, tk.END)
        self.col_listbox.delete(0, tk.END)
        self.row_info_label.config(text="")
        self.col_info_label.config(text="")
        if self.bib is not None and hasattr(self.bib, 'df'):
            for col in self.bib.df.columns:
                self.avail_listbox.insert(tk.END, col)
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
CROSSTABULATION
═══════════════

Analyze relationships between categorical variables.

OUTPUT TABLE
────────────
• Observed frequencies
• Expected frequencies (if independent)
• Row percentages
• Column percentages
• Total percentages

STATISTICAL TESTS
─────────────────
• Chi-Square (χ²)
  Tests independence
  H0: Variables are independent
  p < 0.05: Significant association
  
• Fisher's Exact Test
  For 2×2 tables
  Better for small samples
  
• Cramér's V
  Effect size (0 to 1)
  0.1 small, 0.3 medium, 0.5 large

ASSUMPTIONS
───────────
• Expected freq ≥ 5 (most cells)
• Use Fisher for small samples
• Independent observations

USE CASES
─────────
• Document type × Source type
• Country × Collaboration type
• Subject × Open access status
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
        self._populate_variables()
        self._show_placeholder()
