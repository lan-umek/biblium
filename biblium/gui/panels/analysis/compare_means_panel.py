# -*- coding: utf-8 -*-
"""
Compare Means Panel - SPSS-like interface
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


class CompareMeansPanel(BasePanel):
    """Panel for comparing means - SPSS-like interface."""
    
    title = "Compare Means"
    icon = "📊"
    requires_data = True
    
    def __init__(self, parent, theme="light", **kwargs):
        self._result = None
        self._all_results = []
        super().__init__(parent, theme=theme, **kwargs)
        
        # Subscribe to data loaded event
        event_bus.subscribe(EventBus.DATASET_LOADED, self._handle_data_loaded)
    
    def _handle_data_loaded(self, data):
        """Handle DATA_LOADED event."""
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
        
        self.avail_listbox = tk.Listbox(avail_frame, height=12, width=20, font=FONTS.get_font("small"),
                                        selectmode=tk.EXTENDED, exportselection=False, bg="white")
        self.avail_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        avail_sb = ttk.Scrollbar(avail_frame, command=self.avail_listbox.yview)
        avail_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.avail_listbox.config(yscrollcommand=avail_sb.set)
        
        # Middle: Arrow buttons
        mid_frame = tk.Frame(var_frame, bg=self.theme["bg_secondary"])
        mid_frame.pack(side=tk.LEFT, padx=6, pady=10)
        
        tk.Label(mid_frame, text="Dependent:", font=FONTS.get_font("small"),
                bg=self.theme["bg_secondary"], fg=self.theme["text_muted"]).pack()
        tk.Button(mid_frame, text="▶", width=3, command=self._add_dependent).pack(pady=2)
        tk.Button(mid_frame, text="◀", width=3, command=self._remove_dependent).pack(pady=2)
        
        tk.Label(mid_frame, text="Factor:", font=FONTS.get_font("small"),
                bg=self.theme["bg_secondary"], fg=self.theme["text_muted"]).pack(pady=(12,0))
        tk.Button(mid_frame, text="▶", width=3, command=self._add_grouping).pack(pady=2)
        tk.Button(mid_frame, text="◀", width=3, command=self._remove_grouping).pack(pady=2)
        
        # Right: Selected variables
        right_frame = tk.Frame(var_frame, bg=self.theme["bg_secondary"])
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tk.Label(right_frame, text="Dependent List:", font=FONTS.get_font("body", bold=True),
                bg=self.theme["bg_secondary"], fg=self.theme["text_primary"]).pack(anchor="w")
        self.dep_listbox = tk.Listbox(right_frame, height=4, width=20, font=FONTS.get_font("small"),
                                      selectmode=tk.EXTENDED, exportselection=False, bg="white")
        self.dep_listbox.pack(fill=tk.X, pady=4)
        
        tk.Label(right_frame, text="Factor:", font=FONTS.get_font("body", bold=True),
                bg=self.theme["bg_secondary"], fg=self.theme["text_primary"]).pack(anchor="w", pady=(8,0))
        self.grp_listbox = tk.Listbox(right_frame, height=2, width=20, font=FONTS.get_font("small"),
                                      selectmode=tk.SINGLE, exportselection=False, bg="white")
        self.grp_listbox.pack(fill=tk.X, pady=4)
        self.grp_listbox.bind('<<ListboxSelect>>', self._on_group_selected)
        
        self.group_info_label = tk.Label(right_frame, text="", font=FONTS.get_font("small"),
                                         bg=self.theme["bg_secondary"], fg=self.theme["text_muted"])
        self.group_info_label.pack(anchor="w")
        
        # Options frame
        opt_frame = tk.LabelFrame(self.options_content, text="Options", font=FONTS.get_font("body"),
                                  bg=self.theme["bg_secondary"], fg=self.theme["text_primary"])
        opt_frame.pack(fill=tk.X, padx=8, pady=8)
        
        self.descriptives_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_frame, text="Descriptives", variable=self.descriptives_var,
                      font=FONTS.get_font("body"), bg=self.theme["bg_secondary"],
                      selectcolor="white").pack(anchor="w", padx=8, pady=2)
        
        self.homogeneity_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_frame, text="Homogeneity of variance", variable=self.homogeneity_var,
                      font=FONTS.get_font("body"), bg=self.theme["bg_secondary"],
                      selectcolor="white").pack(anchor="w", padx=8, pady=2)
        
        self.normality_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_frame, text="Normality tests", variable=self.normality_var,
                      font=FONTS.get_font("body"), bg=self.theme["bg_secondary"],
                      selectcolor="white").pack(anchor="w", padx=8, pady=2)
        
        self.posthoc_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_frame, text="Post-hoc tests (3+ groups)", variable=self.posthoc_var,
                      font=FONTS.get_font("body"), bg=self.theme["bg_secondary"],
                      selectcolor="white").pack(anchor="w", padx=8, pady=2)
        
        self.nonparam_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_frame, text="Non-parametric alternatives", variable=self.nonparam_var,
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
    
    def _add_dependent(self):
        selection = self.avail_listbox.curselection()
        if not selection:
            return
        for idx in reversed(selection):
            var = self.avail_listbox.get(idx)
            if self.bib and var in self.bib.df.columns:
                if pd.api.types.is_numeric_dtype(self.bib.df[var]):
                    self.dep_listbox.insert(tk.END, var)
                    self.avail_listbox.delete(idx)
                else:
                    messagebox.showwarning("Invalid", f"'{var}' must be numeric")
    
    def _remove_dependent(self):
        selection = self.dep_listbox.curselection()
        for idx in reversed(selection if selection else range(self.dep_listbox.size())):
            var = self.dep_listbox.get(idx)
            self.avail_listbox.insert(tk.END, var)
            self.dep_listbox.delete(idx)
            if not selection:
                break
    
    def _add_grouping(self):
        selection = self.avail_listbox.curselection()
        if not selection:
            return
        if self.grp_listbox.size() > 0:
            messagebox.showinfo("Info", "Only one factor allowed. Remove current first.")
            return
        idx = selection[0]
        var = self.avail_listbox.get(idx)
        if self.bib and var in self.bib.df.columns:
            n = self.bib.df[var].nunique()
            if n < 2:
                messagebox.showwarning("Invalid", f"'{var}' needs at least 2 groups")
                return
        self.grp_listbox.insert(tk.END, var)
        self.avail_listbox.delete(idx)
        self._on_group_selected(None)
    
    def _remove_grouping(self):
        if self.grp_listbox.size() > 0:
            var = self.grp_listbox.get(0)
            self.avail_listbox.insert(tk.END, var)
            self.grp_listbox.delete(0)
            self.group_info_label.config(text="")
    
    def _reset_selection(self):
        while self.dep_listbox.size() > 0:
            self._remove_dependent()
        self._remove_grouping()
        self._result = None
        self._all_results = []
        self._show_placeholder()
    
    def _on_group_selected(self, event):
        if self.grp_listbox.size() == 0 or not self.bib:
            self.group_info_label.config(text="")
            return
        col = self.grp_listbox.get(0)
        n = self.bib.df[col].nunique()
        counts = self.bib.df[col].value_counts()
        info = f"Groups: {n} | " + ", ".join([f"{k}({v})" for k,v in counts.head(4).items()])
        if n > 4:
            info += "..."
        self.group_info_label.config(text=info)
    
    def _create_results(self):
        self.results_card = tk.Frame(self.results_frame, bg=self.theme["bg_card"],
                                     highlightbackground=self.theme["border"], highlightthickness=1)
        self.results_card.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        self.notebook = ttk.Notebook(self.results_card)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        self.desc_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.desc_frame, text="📊 Descriptives")
        
        self.tests_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.tests_frame, text="🧪 Tests")
        
        self.assume_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.assume_frame, text="✓ Assumptions")
        
        self.posthoc_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.posthoc_frame, text="🔍 Post-hoc")
        
        self.summary_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.summary_frame, text="📝 Summary")

        

        # Info tab

        info_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])

        self.notebook.add(info_frame, text="ℹ️ Info")

        self._create_info_content(info_frame)
        
        self._show_placeholder()
    
    def _show_placeholder(self):
        for f in [self.desc_frame, self.tests_frame, self.assume_frame, self.posthoc_frame, self.summary_frame]:
            for w in f.winfo_children():
                w.destroy()
        tk.Label(self.desc_frame, text="Move variables using arrow buttons:\n\n1. Select dependent variable(s)\n2. Select factor (grouping)\n3. Click 'OK - Run Analysis'",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"],
                justify=tk.CENTER).pack(expand=True)
    
    def _run_analysis(self):
        if not self.bib:
            messagebox.showwarning("No Data", "Load a dataset first.")
            return
        if self.dep_listbox.size() == 0:
            messagebox.showwarning("Required", "Select dependent variable(s).")
            return
        if self.grp_listbox.size() == 0:
            messagebox.showwarning("Required", "Select a factor (grouping variable).")
            return
        
        dep_vars = [self.dep_listbox.get(i) for i in range(self.dep_listbox.size())]
        grp_var = self.grp_listbox.get(0)
        alpha = float(self.alpha_var.get())
        
        try:
            from biblium.compare_means import compare_means
            self._all_results = []
            for dep_var in dep_vars:
                result = compare_means(self.bib.df, dep_var, grp_var, alpha=alpha, verbose=True)
                self._all_results.append(result)
            
            self._result = self._all_results[0] if len(self._all_results) == 1 else self._all_results
            self._update_descriptives()
            self._update_tests()
            self._update_assumptions()
            self._update_posthoc()
            self._update_summary()
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Analysis failed:\n{e}")
    
    def _update_descriptives(self):
        for w in self.desc_frame.winfo_children():
            w.destroy()
        if not self._all_results:
            return
        
        canvas = tk.Canvas(self.desc_frame, bg=self.theme["bg_card"], highlightthickness=0)
        sb = ttk.Scrollbar(self.desc_frame, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=self.theme["bg_card"])
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        
        for r in self._all_results:
            tk.Label(inner, text=f"Descriptives: {r.dependent_var} by {r.grouping_var}",
                    font=FONTS.get_font("heading", bold=True), bg=self.theme["bg_card"],
                    fg=self.theme["text_primary"]).pack(pady=(8,4), anchor="w", padx=8)
            
            tree_frame = tk.Frame(inner, bg=self.theme["bg_card"])
            tree_frame.pack(fill=tk.X, padx=8, pady=4)
            
            cols = ("Group", "N", "Mean", "SD", "SE", "95% CI", "Median", "Min", "Max")
            tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=min(8, len(r.group_descriptives)+2))
            for c in cols:
                tree.heading(c, text=c)
                if c == "Group":
                    tree.column(c, width=120, minwidth=80, anchor="w")
                elif c == "95% CI":
                    tree.column(c, width=130, minwidth=100, anchor="center")
                elif c == "N":
                    tree.column(c, width=50, minwidth=40, anchor="center")
                else:
                    tree.column(c, width=75, minwidth=60, anchor="center")
            
            o = r.overall_descriptives
            tree.insert("", tk.END, values=("Overall", o.n, f"{o.mean:.3f}", f"{o.std:.3f}", f"{o.se:.3f}",
                       f"[{o.ci_lower:.2f}, {o.ci_upper:.2f}]", f"{o.median:.3f}", f"{o.min_val:.3f}", f"{o.max_val:.3f}"), tags=('overall',))
            for g in r.group_descriptives:
                tree.insert("", tk.END, values=(g.group_name, g.n, f"{g.mean:.3f}", f"{g.std:.3f}", f"{g.se:.3f}",
                           f"[{g.ci_lower:.2f}, {g.ci_upper:.2f}]", f"{g.median:.3f}", f"{g.min_val:.3f}", f"{g.max_val:.3f}"))
            tree.tag_configure('overall', background='#e8f4f8')
            
            hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
            tree.configure(xscrollcommand=hsb.set)
            tree.pack(fill=tk.X, expand=True)
            hsb.pack(fill=tk.X)
            
            ttk.Separator(inner, orient='horizontal').pack(fill=tk.X, padx=8, pady=8)
    
    def _update_tests(self):
        for w in self.tests_frame.winfo_children():
            w.destroy()
        if not self._all_results:
            return
        
        canvas = tk.Canvas(self.tests_frame, bg=self.theme["bg_card"], highlightthickness=0)
        sb = ttk.Scrollbar(self.tests_frame, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=self.theme["bg_card"])
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        
        for r in self._all_results:
            tk.Label(inner, text=f"Tests: {r.dependent_var}", font=FONTS.get_font("heading", bold=True),
                    bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(pady=(8,4), anchor="w", padx=8)
            
            # Create frame for table with horizontal scroll
            tree_frame = tk.Frame(inner, bg=self.theme["bg_card"])
            tree_frame.pack(fill=tk.X, padx=8, pady=4)
            
            cols = ("Test", "Statistic", "df", "p-value", "Sig.", "Effect Size")
            tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=3)
            for c in cols:
                tree.heading(c, text=c)
                if c == "Test":
                    tree.column(c, width=120, minwidth=100, anchor="w")
                elif c == "Effect Size":
                    tree.column(c, width=180, minwidth=150, anchor="w")
                elif c == "Statistic":
                    tree.column(c, width=100, minwidth=80, anchor="center")
                else:
                    tree.column(c, width=70, minwidth=50, anchor="center")
            
            t = r.parametric_test
            df_str = f"{t.df:.0f}" if t.df2 == 0 else f"{t.df:.0f}, {t.df2:.0f}"
            effect_str = f"{t.effect_size_name} = {t.effect_size:.3f} ({t.effect_size_interpretation})"
            tree.insert("", tk.END, values=(t.test_name, f"{t.statistic:.4f}", df_str, f"{t.p_value:.4f}",
                       "Yes *" if t.is_significant else "No", effect_str))
            
            t = r.nonparametric_test
            effect_str = f"{t.effect_size_name} = {t.effect_size:.3f} ({t.effect_size_interpretation})"
            tree.insert("", tk.END, values=(t.test_name, f"{t.statistic:.4f}", f"{t.df:.0f}" if t.df>0 else "-",
                       f"{t.p_value:.4f}", "Yes *" if t.is_significant else "No", effect_str))
            
            # Add horizontal scrollbar
            hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
            tree.configure(xscrollcommand=hsb.set)
            tree.pack(fill=tk.X, expand=True)
            hsb.pack(fill=tk.X)
            
            color = "green" if r.assumptions_met else "orange"
            tk.Label(inner, text=f"Recommended: {r.recommended_test}", font=FONTS.get_font("body", bold=True),
                    bg=self.theme["bg_card"], fg=color).pack(anchor="w", padx=8, pady=4)
            ttk.Separator(inner, orient='horizontal').pack(fill=tk.X, padx=8, pady=8)
    
    def _update_assumptions(self):
        for w in self.assume_frame.winfo_children():
            w.destroy()
        if not self._all_results:
            return
        
        canvas = tk.Canvas(self.assume_frame, bg=self.theme["bg_card"], highlightthickness=0)
        sb = ttk.Scrollbar(self.assume_frame, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=self.theme["bg_card"])
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        
        for r in self._all_results:
            tk.Label(inner, text=f"Assumptions: {r.dependent_var}", font=FONTS.get_font("heading", bold=True),
                    bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(pady=(8,4), anchor="w", padx=8)
            
            if self.normality_var.get():
                tk.Label(inner, text="Normality (Shapiro-Wilk):", font=FONTS.get_font("body", bold=True),
                        bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(anchor="w", padx=8)
                
                tree_frame = tk.Frame(inner, bg=self.theme["bg_card"])
                tree_frame.pack(fill=tk.X, padx=8, pady=4)
                
                cols = ("Group", "W Statistic", "p-value", "Result")
                tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=min(5, len(r.normality_tests)))
                for c in cols:
                    tree.heading(c, text=c)
                    if c == "Group":
                        tree.column(c, width=150, minwidth=100, anchor="w")
                    elif c == "Result":
                        tree.column(c, width=120, minwidth=100, anchor="center")
                    else:
                        tree.column(c, width=100, minwidth=80, anchor="center")
                for grp, test in r.normality_tests.items():
                    res = "✓ Normal" if test.is_normal else "✗ Not Normal"
                    tree.insert("", tk.END, values=(grp, f"{test.statistic:.4f}" if not np.isnan(test.statistic) else "N/A",
                               f"{test.p_value:.4f}" if not np.isnan(test.p_value) else "N/A", res))
                tree.pack(fill=tk.X, expand=True)
            
            if self.homogeneity_var.get():
                h = r.homogeneity_test
                hcolor = "green" if h.is_homogeneous else "red"
                tk.Label(inner, text=f"Levene's Test: F={h.statistic:.4f}, p={h.p_value:.4f} → {'✓ Homogeneous' if h.is_homogeneous else '✗ Not homogeneous'}",
                        font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=hcolor).pack(anchor="w", padx=8, pady=4)
            
            ocolor = "green" if r.assumptions_met else "orange"
            tk.Label(inner, text="✓ Assumptions met" if r.assumptions_met else "✗ Assumptions violated",
                    font=FONTS.get_font("heading", bold=True), bg=self.theme["bg_card"], fg=ocolor).pack(anchor="w", padx=8, pady=8)
            ttk.Separator(inner, orient='horizontal').pack(fill=tk.X, padx=8, pady=8)
    
    def _update_posthoc(self):
        for w in self.posthoc_frame.winfo_children():
            w.destroy()
        if not self._all_results:
            return
        
        r = self._all_results[0]
        if r.n_groups < 3:
            tk.Label(self.posthoc_frame, text="Post-hoc tests require 3+ groups.\nYour analysis has only 2 groups.",
                    font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"],
                    justify=tk.CENTER).pack(expand=True)
            return
        
        if not r.post_hoc_results:
            tk.Label(self.posthoc_frame, text="No significant main effect.\nPost-hoc tests not performed.",
                    font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"],
                    justify=tk.CENTER).pack(expand=True)
            return
        
        tk.Label(self.posthoc_frame, text=f"Post-hoc: {r.post_hoc_method}", font=FONTS.get_font("heading", bold=True),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(pady=(8,4))
        
        cols = ("Comparison", "Mean Diff", "p-value", "p-adjusted", "Significant")
        tree_frame = tk.Frame(self.posthoc_frame, bg=self.theme["bg_card"])
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=15)
        for c in cols:
            tree.heading(c, text=c)
            if c == "Comparison":
                tree.column(c, width=180, minwidth=150, anchor="w")
            elif c == "Significant":
                tree.column(c, width=100, minwidth=80, anchor="center")
            else:
                tree.column(c, width=100, minwidth=80, anchor="center")
        for ph in r.post_hoc_results:
            tree.insert("", tk.END, values=(f"{ph.group1} vs {ph.group2}", f"{ph.mean_diff:.3f}",
                       f"{ph.p_value:.4f}", f"{ph.p_adjusted:.4f}", "Yes *" if ph.is_significant else "No"))
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
    
    def _update_summary(self):
        for w in self.summary_frame.winfo_children():
            w.destroy()
        if not self._all_results:
            return
        
        txt = "COMPARE MEANS REPORT\n" + "="*50 + "\n\n"
        for r in self._all_results:
            txt += f"VARIABLE: {r.dependent_var}\nFactor: {r.grouping_var} ({r.n_groups} groups, N={r.n_total})\n\n"
            txt += "DESCRIPTIVES:\n"
            for g in r.group_descriptives:
                txt += f"  {g.group_name}: N={g.n}, M={g.mean:.3f}, SD={g.std:.3f}\n"
            txt += f"\nTESTS:\n  Parametric: {r.parametric_test.test_name}, p={r.parametric_test.p_value:.4f}\n"
            txt += f"  Non-parametric: {r.nonparametric_test.test_name}, p={r.nonparametric_test.p_value:.4f}\n"
            txt += f"\nAssumptions: {'Met' if r.assumptions_met else 'Violated'}\nRecommended: {r.recommended_test}\n"
            txt += f"\n{r.interpretation}\n"
            if r.post_hoc_results:
                txt += f"\nPOST-HOC ({r.post_hoc_method}):\n"
                for ph in r.post_hoc_results:
                    txt += f"  {ph.group1} vs {ph.group2}: diff={ph.mean_diff:.3f}, p_adj={ph.p_adjusted:.4f} {'*' if ph.is_significant else ''}\n"
            txt += "\n" + "-"*50 + "\n\n"
        
        text_w = tk.Text(self.summary_frame, font=FONTS.get_font("code"), bg=self.theme["bg_card"],
                        fg=self.theme["text_primary"], wrap=tk.WORD)
        text_w.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        text_w.insert("1.0", txt)
        text_w.config(state=tk.DISABLED)
    
    def _export_results(self):
        if not self._all_results:
            messagebox.showwarning("No Results", "Run analysis first.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                filetypes=[("Excel","*.xlsx"),("Text","*.txt")])
        if not filepath:
            return
        try:
            if filepath.endswith('.xlsx'):
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    for r in self._all_results:
                        desc = [{'Group':g.group_name,'N':g.n,'Mean':g.mean,'SD':g.std,'SE':g.se} for g in [r.overall_descriptives]+r.group_descriptives]
                        pd.DataFrame(desc).to_excel(writer, sheet_name=f'{r.dependent_var[:20]}_Desc', index=False)
            else:
                with open(filepath, 'w') as f:
                    for w in self.summary_frame.winfo_children():
                        if isinstance(w, tk.Text):
                            f.write(w.get("1.0", tk.END))
            messagebox.showinfo("Exported", f"Saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{e}")
    
    def _populate_variables(self):
        """Populate available variables listbox from bib.df."""
        self.avail_listbox.delete(0, tk.END)
        self.dep_listbox.delete(0, tk.END)
        self.grp_listbox.delete(0, tk.END)
        self.group_info_label.config(text="")
        
        if self.bib is not None and hasattr(self.bib, 'df'):
            for col in self.bib.df.columns:
                self.avail_listbox.insert(tk.END, col)
            print(f"Compare Means: Loaded {len(self.bib.df.columns)} variables")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
COMPARE MEANS
═════════════

Statistical comparison of means between groups.

PARAMETRIC TESTS
────────────────
• Independent t-test
  Two groups, normal data, equal variance
  
• Welch's t-test
  Two groups, unequal variance
  
• One-way ANOVA
  3+ groups, normal data

NON-PARAMETRIC TESTS
────────────────────
• Mann-Whitney U
  Two groups, non-normal
  
• Kruskal-Wallis
  3+ groups, non-normal

EFFECT SIZES
────────────
• Cohen's d (two groups)
  0.2 = small
  0.5 = medium
  0.8 = large
  
• Eta-squared η² (ANOVA)
  0.01 = small
  0.06 = medium
  0.14 = large

POST-HOC TESTS
──────────────
• Tukey HSD: All pairwise
• Bonferroni: Conservative

OUTPUT
──────
• Test statistics
• p-values with significance
• Effect sizes with interpretation
• Descriptive statistics per group
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
        # Handle dict format from event
        if isinstance(bib, dict):
            bib = bib.get("bib", bib)
        
        self.bib = bib
        self._result = None
        self._all_results = []
        self._populate_variables()
        self._show_placeholder()
