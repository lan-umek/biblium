# -*- coding: utf-8 -*-
"""
Group Logistic Regression Panel
================================
GUI panel for statistical logistic regression analysis of group membership
with coefficients, odds ratios, p-values, and significance indicators.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, StatsCard, CardGrid
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox, LabeledEntry
from biblium.gui.widgets.buttons import ActionButton
from biblium.gui.widgets.plots import PlotFrame
from biblium.gui.widgets.tables import DataTable
from biblium.gui.config import FONTS, get_theme
from biblium.gui.core.events import EventBus

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

event_bus = EventBus()


class GroupLogisticPanel(BasePanel):
    """
    Panel for statistical logistic regression analysis of group membership.
    
    Uses BiblioGroupClassifier.logistic_regression_analysis() to fit logistic
    regression models with statsmodels and display coefficients, odds ratios,
    p-values, and significance indicators.
    
    Features:
    - Text-based features (keywords, abstracts) or custom design matrix
    - Configurable predictors (top_n, regex filters, items of interest)
    - Per-group logistic regression models
    - Coefficients with standard errors and confidence intervals
    - Odds ratios with interpretation
    - P-values with significance highlighting
    - Direction indicators (↑/↓ arrows for significant effects)
    - Model statistics (AIC, BIC, pseudo R²)
    - Export to Excel with formatting
    """
    
    title = "📉 Logistic Regression"
    description = "Statistical logistic regression with coefficients and p-values"
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self.bib = bib
        self._bib_group = None
        self._results = None
        self._group_vars = {}  # Group name -> BooleanVar
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
    
    @property
    def bib_group(self):
        """Get the BiblioGroup object from workspace or local storage."""
        if hasattr(self, '_bib_group') and self._bib_group is not None:
            return self._bib_group
        
        # Try to get from workspace
        workspace = self.master
        if hasattr(workspace, 'bib_group'):
            return workspace.bib_group
        if hasattr(workspace, '_shared_bib_group'):
            return workspace._shared_bib_group
        
        return None
    
    @bib_group.setter
    def bib_group(self, value):
        self._bib_group = value
    
    def _create_options(self):
        """Create options panel."""
        super()._create_options()
        
        # Dependent Variables Card (Groups)
        dv_card = Card(self.options_content, title="📊 Dependent Variables (Groups)", theme=self.theme_name)
        dv_card.pack(fill=tk.X, padx=8, pady=8)
        
        tk.Label(dv_card.content, text="Select groups to analyze:", 
                font=FONTS.get_font("small"), bg=self.theme["bg_card"],
                fg=self.theme["text_muted"]).pack(anchor=tk.W)
        
        # Scrollable frame for group checkboxes
        self.groups_frame = tk.Frame(dv_card.content, bg=self.theme["bg_card"])
        self.groups_frame.pack(fill=tk.X, pady=4)
        
        # Buttons for select all/none
        btn_frame = tk.Frame(dv_card.content, bg=self.theme["bg_card"])
        btn_frame.pack(fill=tk.X, pady=4)
        tk.Button(btn_frame, text="Select All", command=self._select_all_groups,
                 font=FONTS.get_font("small")).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Select None", command=self._select_no_groups,
                 font=FONTS.get_font("small")).pack(side=tk.LEFT, padx=2)
        
        # Independent Variables Card
        iv_card = Card(self.options_content, title="📝 Independent Variables", theme=self.theme_name)
        iv_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Text column for term extraction
        self.text_col_var = tk.StringVar(value="Author Keywords")
        text_combo = LabeledCombobox(
            iv_card.content, label="Text column:",
            values=["Author Keywords", "Index Keywords", "Abstract", "Title"],
            variable=self.text_col_var, theme=self.theme_name,
        )
        text_combo.pack(fill=tk.X, pady=4)
        self.text_combo = text_combo
        
        # Top N terms
        topn_frame = tk.Frame(iv_card.content, bg=self.theme["bg_card"])
        topn_frame.pack(fill=tk.X, pady=4)
        tk.Label(topn_frame, text="Top N terms:", font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(side=tk.LEFT)
        self.topn_var = tk.StringVar(value="50")
        tk.Entry(topn_frame, textvariable=self.topn_var,
                font=FONTS.get_font("body"), width=8).pack(side=tk.LEFT, padx=8)
        tk.Label(topn_frame, text="(0 = all)", font=FONTS.get_font("small"),
                bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(side=tk.LEFT)
        
        # Include/Exclude regex
        tk.Label(iv_card.content, text="Filter terms (optional):", 
                font=FONTS.get_font("small"), bg=self.theme["bg_card"],
                fg=self.theme["text_muted"]).pack(anchor=tk.W, pady=(8, 0))
        
        inc_frame = tk.Frame(iv_card.content, bg=self.theme["bg_card"])
        inc_frame.pack(fill=tk.X, pady=2)
        tk.Label(inc_frame, text="Include regex:", font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"], width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.include_var = tk.StringVar(value="")
        tk.Entry(inc_frame, textvariable=self.include_var,
                font=FONTS.get_font("body")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        exc_frame = tk.Frame(iv_card.content, bg=self.theme["bg_card"])
        exc_frame.pack(fill=tk.X, pady=2)
        tk.Label(exc_frame, text="Exclude regex:", font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"], width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.exclude_var = tk.StringVar(value="")
        tk.Entry(exc_frame, textvariable=self.exclude_var,
                font=FONTS.get_font("body")).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Items of interest (specific terms)
        tk.Label(iv_card.content, text="Specific terms (optional, one per line):", 
                font=FONTS.get_font("small"), bg=self.theme["bg_card"],
                fg=self.theme["text_muted"]).pack(anchor=tk.W, pady=(8, 0))
        
        self.items_text = tk.Text(iv_card.content, height=4, font=FONTS.get_font("body"))
        self.items_text.pack(fill=tk.X, pady=4)
        
        # Significance Levels Card
        sig_card = Card(self.options_content, title="📊 Display Options", theme=self.theme_name)
        sig_card.pack(fill=tk.X, padx=8, pady=8)
        
        tk.Label(sig_card.content, text="Significance thresholds used for highlighting:", 
                font=FONTS.get_font("small"), bg=self.theme["bg_card"],
                fg=self.theme["text_muted"]).pack(anchor=tk.W)
        
        sig_frame = tk.Frame(sig_card.content, bg=self.theme["bg_card"])
        sig_frame.pack(fill=tk.X, pady=4)
        
        # Color legend
        legend_items = [
            ("p ≤ 0.001", "#006400", "↑↑↑ / ↓↓↓"),
            ("p ≤ 0.01", "#228B22", "↑↑ / ↓↓"),
            ("p ≤ 0.05", "#66CDAA", "↑ / ↓"),
            ("p ≤ 0.10", "#98FB98", ""),
        ]
        
        for threshold, color, arrows in legend_items:
            row = tk.Frame(sig_frame, bg=self.theme["bg_card"])
            row.pack(fill=tk.X, pady=1)
            color_box = tk.Label(row, text="   ", bg=color, width=3)
            color_box.pack(side=tk.LEFT, padx=4)
            tk.Label(row, text=f"{threshold}", font=FONTS.get_font("small"),
                    bg=self.theme["bg_card"], fg=self.theme["text_primary"], width=10).pack(side=tk.LEFT)
            if arrows:
                tk.Label(row, text=arrows, font=FONTS.get_font("small"),
                        bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(side=tk.LEFT)
        
        # Action Buttons
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Run Analysis", icon="▶️",
            command=self._run_analysis, theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
        
        ActionButton(
            btn_frame, text="Export Results", icon="💾",
            command=self._export_results, theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
    
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

    def _select_all_groups(self):
        """Select all groups."""
        for var in self._group_vars.values():
            var.set(True)
    
    def _select_no_groups(self):
        """Deselect all groups."""
        for var in self._group_vars.values():
            var.set(False)
    
    def _populate_groups(self):
        """Populate group checkboxes from group_matrix."""
        for widget in self.groups_frame.winfo_children():
            widget.destroy()
        self._group_vars.clear()
        
        # Check for bib_group first, then fall back to bib.group_matrix
        group_matrix = None
        if self.bib_group is not None and hasattr(self.bib_group, 'group_matrix') and self.bib_group.group_matrix is not None:
            group_matrix = self.bib_group.group_matrix
        elif self.bib is not None and hasattr(self.bib, 'group_matrix') and self.bib.group_matrix is not None:
            group_matrix = self.bib.group_matrix
        
        if group_matrix is None or len(group_matrix.columns) == 0:
            tk.Label(self.groups_frame, text="No groups defined. Set up groups first.",
                    font=FONTS.get_font("small"), bg=self.theme["bg_card"],
                    fg=self.theme["text_muted"]).pack(anchor=tk.W)
            return
        
        for col in group_matrix.columns:
            var = tk.BooleanVar(value=True)
            self._group_vars[col] = var
            cb = tk.Checkbutton(self.groups_frame, text=col, variable=var,
                               bg=self.theme["bg_card"], font=FONTS.get_font("body"))
            cb.pack(anchor=tk.W)
    
    def _populate_columns(self):
        """Populate column dropdowns."""
        # Get dataframe from bib_group or bib
        df = None
        if self.bib_group is not None and hasattr(self.bib_group, 'df'):
            df = self.bib_group.df
        elif self.bib is not None and hasattr(self.bib, 'df'):
            df = self.bib.df
        
        if df is None:
            return
        
        cols = list(df.columns)
        
        # Text columns - prefer keyword columns
        text_cols = []
        for c in cols:
            c_lower = c.lower()
            if any(kw in c_lower for kw in ['keyword', 'abstract', 'title', 'description']):
                text_cols.append(c)
        
        if not text_cols:
            text_cols = cols[:10]
        
        self.text_combo.combobox['values'] = text_cols
        
        # Set default to "Processed Abstract" if available
        if 'Processed Abstract' in text_cols:
            self.text_col_var.set('Processed Abstract')
        else:
            # Fall back to keyword column if available
            for col in text_cols:
                if 'keyword' in col.lower():
                    self.text_col_var.set(col)
                    break
    
    def _get_selected_groups(self) -> List[str]:
        """Get list of selected group names."""
        return [name for name, var in self._group_vars.items() if var.get()]
    
    def _run_analysis(self):
        """Run the logistic regression analysis."""
        # Get dataframe and group_matrix from bib_group or bib
        df = None
        group_matrix = None
        
        if self.bib_group is not None:
            if hasattr(self.bib_group, 'df'):
                df = self.bib_group.df
            if hasattr(self.bib_group, 'group_matrix'):
                group_matrix = self.bib_group.group_matrix
        
        if df is None and self.bib is not None and hasattr(self.bib, 'df'):
            df = self.bib.df
        if group_matrix is None and self.bib is not None and hasattr(self.bib, 'group_matrix'):
            group_matrix = self.bib.group_matrix
        
        if df is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        if group_matrix is None or len(group_matrix.columns) == 0:
            messagebox.showwarning("No Groups", "Please set up groups first (use Group Setup panel).")
            return
        
        selected_groups = self._get_selected_groups()
        if not selected_groups:
            messagebox.showwarning("No Groups Selected", "Please select at least one group to analyze.")
            return
        
        self._show_loading("Running logistic regression analysis...")
        self.update_idletasks()
        
        try:
            import statsmodels.api as sm
            from sklearn.feature_extraction.text import CountVectorizer
            import re
            
            # Get settings
            text_column = self.text_col_var.get()
            
            try:
                top_n = int(self.topn_var.get())
            except ValueError:
                top_n = 50
            
            include_regex = self.include_var.get().strip() or None
            exclude_regex = self.exclude_var.get().strip() or None
            
            # Get items of interest
            items_text = self.items_text.get("1.0", tk.END).strip()
            items_of_interest = None
            if items_text:
                items_of_interest = [item.strip() for item in items_text.split('\n') if item.strip()]
            
            # Build term matrix from text column
            if text_column not in df.columns:
                self._show_error(f"Column '{text_column}' not found in data.")
                return
            
            texts = df[text_column].fillna("").astype(str)
            
            # Check if it's a keyword column (use special tokenizer)
            is_keyword_col = "keyword" in text_column.lower()
            
            if is_keyword_col:
                # Get separator from bib_group if available
                sep = "; "
                if self.bib_group is not None and hasattr(self.bib_group, 'default_separator'):
                    sep = self.bib_group.default_separator
                
                def keyword_tokenizer(text, _sep=sep):
                    if not text:
                        return []
                    return [t.strip() for t in str(text).split(_sep) if t.strip()]
                
                vec = CountVectorizer(
                    tokenizer=keyword_tokenizer,
                    token_pattern=None,
                    max_features=max(500, top_n * 2) if top_n > 0 else 1000,
                    binary=True,
                )
            else:
                vec = CountVectorizer(
                    max_features=max(500, top_n * 2) if top_n > 0 else 1000,
                    stop_words="english",
                    binary=True,
                )
            
            Xc = vec.fit_transform(texts)
            vocab = np.array(vec.get_feature_names_out())
            doc_counts = (Xc > 0).sum(axis=0).A1
            
            items_df = pd.DataFrame({"item": vocab, "doc_count": doc_counts})
            
            # Apply filters
            if include_regex:
                items_df = items_df[items_df["item"].str.contains(include_regex, regex=True, na=False)]
            if exclude_regex:
                items_df = items_df[~items_df["item"].str.contains(exclude_regex, regex=True, na=False)]
            
            # Select items
            if items_of_interest is not None:
                items_set = set(items_df["item"])
                selected_items = [it for it in items_of_interest if it in items_set]
                if not selected_items:
                    self._show_error("No items of interest found in vocabulary after filtering.")
                    return
                items = selected_items
            else:
                items_df = items_df.sort_values("doc_count", ascending=False)
                if top_n > 0:
                    items = items_df.head(top_n)["item"].tolist()
                else:
                    items = items_df["item"].tolist()
            
            if not items:
                self._show_error("No predictors remain after filtering.")
                return
            
            # Build design matrix
            X_dense = Xc.toarray()
            X_all = pd.DataFrame(X_dense, columns=vocab, index=df.index)
            X_design = X_all[items]
            X_design = sm.add_constant(X_design, has_constant="add")
            
            # Store for export
            self._df = df
            self._group_matrix = group_matrix
            
            # Fit logistic regression for each group
            results = {}
            
            for grp in selected_groups:
                if grp not in group_matrix.columns:
                    continue
                
                y = group_matrix[grp].values
                
                # Skip if only one class
                if len(np.unique(y)) < 2:
                    print(f"Skipping logistic regression for group '{grp}': only one class")
                    continue
                
                try:
                    model = sm.Logit(y, X_design).fit(disp=False)
                    coef_table = model.summary2().tables[1]
                    results[grp] = {"model": model, "summary": coef_table}
                except Exception as e:
                    print(f"Logit failed for group '{grp}': {e}")
            
            if not results:
                self._show_error("No results generated. Check that groups have variation (not all 0 or all 1).")
                return
            
            self._results = results
            self._display_results(results)
            
        except Exception as e:
            self._show_error(str(e))
            import traceback
            traceback.print_exc()
    
    def _display_results(self, results: Dict):
        """Display logistic regression results."""
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
        
        # Summary stats
        grid = CardGrid(self.results_tab, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 8))
        
        n_groups = len(results)
        total_sig = 0
        total_predictors = 0
        avg_pseudo_r2 = 0
        
        for grp, data in results.items():
            model = data["model"]
            coef_df = data["summary"]
            
            # Count significant predictors (excluding const)
            if "P>|z|" in coef_df.columns:
                p_vals = pd.to_numeric(coef_df["P>|z|"], errors="coerce")
                # Exclude const
                if "const" in coef_df.index:
                    p_vals = p_vals.drop("const", errors="ignore")
                sig_count = (p_vals <= 0.05).sum()
                total_sig += sig_count
                total_predictors += len(p_vals)
            
            avg_pseudo_r2 += model.prsquared
        
        avg_pseudo_r2 /= n_groups if n_groups > 0 else 1
        
        grid.add_card(StatsCard(grid, "Groups", str(n_groups), "📊", self.theme_name))
        grid.add_card(StatsCard(grid, "Predictors", str(total_predictors // n_groups if n_groups else 0), "📝", self.theme_name))
        grid.add_card(StatsCard(grid, "Significant (p≤0.05)", str(total_sig), "✓", self.theme_name))
        grid.add_card(StatsCard(grid, "Avg Pseudo R²", f"{avg_pseudo_r2:.3f}", "📈", self.theme_name))
        
        # Create notebook with one tab per group + summary
        notebook = ttk.Notebook(self.results_tab)
        notebook.pack(fill=tk.BOTH, expand=True, pady=8)
        
        # Summary tab
        summary_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(summary_frame, text="Summary")
        
        summary_rows = []
        for grp, data in results.items():
            model = data["model"]
            coef_df = data["summary"]
            
            # Count significant at different levels
            p_vals = pd.to_numeric(coef_df["P>|z|"], errors="coerce")
            if "const" in coef_df.index:
                p_vals_no_const = p_vals.drop("const", errors="ignore")
            else:
                p_vals_no_const = p_vals
            
            summary_rows.append({
                "Group": grp,
                "N_obs": model.nobs,
                "N_predictors": len(p_vals_no_const),
                "Sig_0.10": (p_vals_no_const <= 0.10).sum(),
                "Sig_0.05": (p_vals_no_const <= 0.05).sum(),
                "Sig_0.01": (p_vals_no_const <= 0.01).sum(),
                "Sig_0.001": (p_vals_no_const <= 0.001).sum(),
                "AIC": round(model.aic, 2),
                "BIC": round(model.bic, 2),
                "Pseudo_R2": round(model.prsquared, 4),
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_table = DataTable(summary_frame, theme=self.theme_name)
        summary_table.pack(fill=tk.BOTH, expand=True)
        summary_table.set_data(summary_df)
        
        # Per-group tabs
        for grp, data in results.items():
            grp_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(grp_frame, text=grp[:15])

            

            # Info tab

            info_frame = tk.Frame(notebook, bg=self.theme["bg_card"])

            notebook.add(info_frame, text="ℹ️ Info")

            self._create_info_content(info_frame)
            
            self._create_group_tab(grp_frame, grp, data)
        
        # Store for export
        self._results_dict = results
    
    def _create_group_tab(self, parent, group_name: str, data: Dict):
        """Create a tab for a single group's results."""
        model = data["model"]
        coef_df = data["summary"].copy()
        
        # Add odds ratios and direction
        coef_df["OR"] = np.exp(coef_df["Coef."])
        
        # Add direction arrows
        if "Coef." in coef_df.columns and "P>|z|" in coef_df.columns:
            coef_vals = pd.to_numeric(coef_df["Coef."], errors="coerce")
            p_vals = pd.to_numeric(coef_df["P>|z|"], errors="coerce")
            
            direction = []
            for idx in coef_df.index:
                coef = coef_vals.get(idx, 0)
                p = p_vals.get(idx, 1)
                
                if idx == "const" or pd.isna(p):
                    direction.append("")
                elif p <= 0.001:
                    direction.append("↑↑↑" if coef > 0 else "↓↓↓")
                elif p <= 0.01:
                    direction.append("↑↑" if coef > 0 else "↓↓")
                elif p <= 0.05:
                    direction.append("↑" if coef > 0 else "↓")
                else:
                    direction.append("")
            
            coef_df["Direction"] = direction
        
        # Model stats at top
        stats_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        stats_frame.pack(fill=tk.X, pady=8, padx=8)
        
        stats_text = f"N = {int(model.nobs)}  |  AIC = {model.aic:.2f}  |  BIC = {model.bic:.2f}  |  Pseudo R² = {model.prsquared:.4f}"
        tk.Label(stats_frame, text=stats_text, font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(anchor=tk.W)
        
        # Coefficient table
        # Reorder columns for display
        display_cols = ["Coef.", "Std.Err.", "z", "P>|z|", "[0.025", "0.975]", "OR", "Direction"]
        display_cols = [c for c in display_cols if c in coef_df.columns]
        
        coef_display = coef_df[display_cols].copy()
        coef_display = coef_display.reset_index()
        coef_display.columns = ["Term"] + list(coef_display.columns[1:])
        
        # Round numeric columns
        for col in coef_display.columns:
            if col not in ["Term", "Direction"]:
                coef_display[col] = pd.to_numeric(coef_display[col], errors="coerce").round(4)
        
        # Create table with highlighting
        table_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Create treeview with custom colors
        tree = ttk.Treeview(table_frame, columns=list(coef_display.columns), show="headings", height=20)
        
        for col in coef_display.columns:
            tree.heading(col, text=col)
            width = 80 if col not in ["Term"] else 150
            tree.column(col, width=width, anchor=tk.CENTER if col != "Term" else tk.W)
        
        # Configure tags for highlighting
        tree.tag_configure("sig_001", background="#006400", foreground="white")
        tree.tag_configure("sig_01", background="#228B22", foreground="white")
        tree.tag_configure("sig_05", background="#66CDAA", foreground="black")
        tree.tag_configure("sig_10", background="#98FB98", foreground="black")
        
        # Insert data with tags
        for _, row in coef_display.iterrows():
            p_val = row.get("P>|z|", 1)
            
            if pd.notna(p_val):
                if p_val <= 0.001:
                    tag = "sig_001"
                elif p_val <= 0.01:
                    tag = "sig_01"
                elif p_val <= 0.05:
                    tag = "sig_05"
                elif p_val <= 0.10:
                    tag = "sig_10"
                else:
                    tag = ""
            else:
                tag = ""
            
            values = [str(v) if pd.notna(v) else "" for v in row]
            tree.insert("", tk.END, values=values, tags=(tag,) if tag else ())
        
        # Scrollbars
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
    
    def _export_results(self):
        """Export results to Excel."""
        if self._results is None:
            messagebox.showwarning("No Results", "Please run analysis first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Save Logistic Regression Results"
        )
        
        if not filename:
            return
        
        try:
            results = self._results
            
            with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
                workbook = writer.book
                
                # p-value formats
                p_formats = {
                    0.001: workbook.add_format({"bg_color": "#006400", "font_color": "#FFFFFF"}),
                    0.01: workbook.add_format({"bg_color": "#228B22", "font_color": "#FFFFFF"}),
                    0.05: workbook.add_format({"bg_color": "#66CDAA", "font_color": "#000000"}),
                    0.1: workbook.add_format({"bg_color": "#98FB98", "font_color": "#000000"}),
                }
                
                summary_rows = []
                
                for grp, data in results.items():
                    coef_df = data["summary"].copy()
                    model = data["model"]
                    
                    # Add odds ratios
                    coef_df["OR"] = np.exp(coef_df["Coef."])
                    
                    # Add direction arrows
                    if "Coef." in coef_df.columns and "P>|z|" in coef_df.columns:
                        coef_vals = pd.to_numeric(coef_df["Coef."], errors="coerce")
                        p_vals = pd.to_numeric(coef_df["P>|z|"], errors="coerce")
                        
                        direction = np.full(len(coef_df), "", dtype=object)
                        pos = coef_vals > 0
                        neg = coef_vals < 0
                        
                        band1 = (p_vals <= 0.05) & (p_vals > 0.01)
                        band2 = (p_vals <= 0.01) & (p_vals > 0.001)
                        band3 = p_vals <= 0.001
                        
                        direction[band1 & pos] = "↑"
                        direction[band1 & neg] = "↓"
                        direction[band2 & pos] = "↑↑"
                        direction[band2 & neg] = "↓↓"
                        direction[band3 & pos] = "↑↑↑"
                        direction[band3 & neg] = "↓↓↓"
                        
                        if "const" in coef_df.index:
                            direction[coef_df.index == "const"] = ""
                        
                        coef_df["Direction"] = direction
                    
                    # Write coefficients sheet
                    sheet_name = f"coeff {grp}"[:31]
                    coef_df.to_excel(writer, sheet_name=sheet_name, index=True)
                    
                    # Highlight p-values
                    ws = writer.sheets[sheet_name]
                    if "P>|z|" in coef_df.columns:
                        p_col_idx = coef_df.columns.get_loc("P>|z|") + 1
                        for row_idx, p_val in enumerate(coef_df["P>|z|"], start=1):
                            if pd.isna(p_val):
                                continue
                            for threshold, fmt in p_formats.items():
                                if p_val <= threshold:
                                    ws.write(row_idx, p_col_idx, p_val, fmt)
                                    break
                    
                    # Write statistics sheet
                    stats_df = pd.DataFrame({
                        "AIC": [model.aic],
                        "BIC": [model.bic],
                        "Pseudo R-squared": [model.prsquared],
                    })
                    stats_df.to_excel(writer, sheet_name=f"stats {grp}"[:31], index=False)
                    
                    # Summary row
                    p_vals_no_const = pd.to_numeric(coef_df["P>|z|"], errors="coerce")
                    if "const" in coef_df.index:
                        p_vals_no_const = p_vals_no_const.drop("const", errors="ignore")
                    
                    summary_rows.append({
                        "Group": grp,
                        "N_obs": int(model.nobs),
                        "N_predictors": len(p_vals_no_const),
                        "Sig_0.10": int((p_vals_no_const <= 0.10).sum()),
                        "Sig_0.05": int((p_vals_no_const <= 0.05).sum()),
                        "Sig_0.01": int((p_vals_no_const <= 0.01).sum()),
                        "Sig_0.001": int((p_vals_no_const <= 0.001).sum()),
                        "AIC": round(model.aic, 2),
                        "BIC": round(model.bic, 2),
                        "Pseudo_R2": round(model.prsquared, 4),
                    })
                
                # Write summary sheet
                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
            
            messagebox.showinfo("Success", f"Results saved to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {e}")
            import traceback
            traceback.print_exc()
    
    def _show_loading(self, message: str = "Processing..."):
        """Show loading indicator."""
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
        frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
        frame.pack(expand=True)
        tk.Label(frame, text="⏳", font=("Segoe UI", 32), bg=self.theme["bg_card"],
                fg=self.theme["text_primary"]).pack(pady=(20, 10))
        tk.Label(frame, text=message, font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                fg=self.theme["text_muted"], justify=tk.CENTER).pack()
    
    def _show_error(self, message: str):
        """Show error message."""
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
        error_color = self.theme.get("error", self.theme.get("danger", "#e74c3c"))
        tk.Label(self.results_tab, text=f"❌ Error\n\n{message}", font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=error_color, justify=tk.CENTER,
                wraplength=400).pack(expand=True)
    
    def set_bib(self, bib):
        """Set the bibliometric data object."""
        self.bib = bib
        self._populate_groups()
        self._populate_columns()
    
    def set_bib_group(self, bib_group):
        """Set the BiblioGroup object."""
        self._bib_group = bib_group
        self._populate_groups()
        self._populate_columns()
    
    def refresh(self):
        """Refresh the panel (called when switching to this panel)."""
        self._populate_groups()
        self._populate_columns()
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
LOGISTIC REGRESSION
═══════════════════

Statistical model for predicting group membership.

MODEL OUTPUT
────────────
• Coefficients (β): Effect direction and strength
• Standard Errors: Coefficient precision
• Odds Ratios (OR): Multiplicative effect
• P-values: Statistical significance
• Confidence Intervals: Range of plausible values
• Direction Indicators: ↑ positive, ↓ negative

INTERPRETING COEFFICIENTS
─────────────────────────
• Positive β (↑): Increases group probability
• Negative β (↓): Decreases group probability
• Magnitude: Effect strength

ODDS RATIOS
───────────
• OR = exp(β)
• OR = 1.0: No effect
• OR > 1.0: Higher odds of group membership
• OR < 1.0: Lower odds of group membership
• OR = 2.0: 2× higher odds

SIGNIFICANCE LEVELS
───────────────────
• *** p < 0.001
• ** p < 0.01
• * p < 0.05
• . p < 0.10

MODEL FIT STATISTICS
────────────────────
• Pseudo R²: Variance explained
• Log-likelihood: Model fit
• AIC/BIC: Model comparison

FEATURE IMPORTANCE
──────────────────
Identifies which terms distinguish groups:
• Significant positive: Characteristic of group
• Significant negative: Absent from group

USE CASES
─────────
• Identify distinguishing features
• Predict group membership
• Compare group characteristics
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

    def destroy(self):
        """Clean up resources."""
        try:
            plt.close('all')
        except:
            pass
        super().destroy()
