# -*- coding: utf-8 -*-
"""
Group Classification Panel
==========================
GUI panel for ML-based classification of group membership with multiple models
and comprehensive performance metrics (accuracy, precision, recall, F1, AUC).
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


class GroupClassificationPanel(BasePanel):
    """
    Panel for ML-based classification of group membership.
    
    Uses BiblioGroupClassifier.classify_groups() to train and evaluate
    multiple classifiers on predicting group membership.
    
    Features:
    - Multiple classifier selection (Logistic, Random Forest, GBM, SVM, Naive Bayes)
    - Multiple evaluation methods (cross-validation, leave-one-out, train-test split)
    - Performance metrics: Accuracy, Precision, Recall, F1, AUC
    - Configurable dependent variables (group columns)
    - Feature selection (text columns or numeric features)
    - Results visualization and export
    """
    
    title = "🎯 Group Classification"
    description = "ML-based classification of group membership with performance metrics"
    
    # Available classifiers
    CLASSIFIERS = {
        "Logistic Regression": "Logistic",
        "Random Forest": "RandomForest",
        "Gradient Boosting": "GBM",
        "Naive Bayes": "NaiveBayes",
        "Support Vector Machine": "SVM",
    }
    
    # Evaluation methods
    EVAL_METHODS = {
        "Cross-Validation (5-fold)": "cross_validation",
        "Leave-One-Out": "leave_one_out",
        "Train-Test Split (80/20)": "train_test",
    }
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self.bib = bib
        self._bib_group = None
        self._results = None
        self._group_vars = {}  # Group name -> BooleanVar
        self._classifier_vars = {}  # Classifier name -> BooleanVar
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
        
        # Info label
        tk.Label(dv_card.content, text="Select groups to predict:", 
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
        
        # Feature Selection Card
        feat_card = Card(self.options_content, title="📝 Features (Independent Variables)", theme=self.theme_name)
        feat_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Feature type
        self.feature_type_var = tk.StringVar(value="text")
        tk.Label(feat_card.content, text="Feature type:", font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(anchor=tk.W)
        
        type_frame = tk.Frame(feat_card.content, bg=self.theme["bg_card"])
        type_frame.pack(fill=tk.X, pady=4)
        tk.Radiobutton(type_frame, text="Text (TF-IDF)", variable=self.feature_type_var,
                      value="text", bg=self.theme["bg_card"], 
                      command=self._on_feature_type_change).pack(anchor=tk.W)
        tk.Radiobutton(type_frame, text="Numeric columns", variable=self.feature_type_var,
                      value="numeric", bg=self.theme["bg_card"],
                      command=self._on_feature_type_change).pack(anchor=tk.W)
        
        # Text column selection
        self.text_col_frame = tk.Frame(feat_card.content, bg=self.theme["bg_card"])
        self.text_col_frame.pack(fill=tk.X, pady=4)
        
        self.text_col_var = tk.StringVar(value="Abstract")
        self.text_combo = LabeledCombobox(
            self.text_col_frame, label="Text column:",
            values=["Abstract", "Title", "Author Keywords"],
            variable=self.text_col_var, theme=self.theme_name,
        )
        self.text_combo.pack(fill=tk.X)
        
        # Max TF-IDF features
        tfidf_frame = tk.Frame(self.text_col_frame, bg=self.theme["bg_card"])
        tfidf_frame.pack(fill=tk.X, pady=4)
        tk.Label(tfidf_frame, text="Max features:", font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"]).pack(side=tk.LEFT)
        self.max_features_var = tk.StringVar(value="500")
        tk.Entry(tfidf_frame, textvariable=self.max_features_var,
                font=FONTS.get_font("body"), width=8).pack(side=tk.LEFT, padx=8)
        
        # Numeric columns selection (hidden initially)
        self.numeric_col_frame = tk.Frame(feat_card.content, bg=self.theme["bg_card"])
        # Will be populated when data is loaded
        
        tk.Label(self.numeric_col_frame, text="Select numeric columns:", 
                font=FONTS.get_font("small"), bg=self.theme["bg_card"],
                fg=self.theme["text_muted"]).pack(anchor=tk.W)
        self.numeric_cols_listbox = tk.Listbox(
            self.numeric_col_frame, selectmode=tk.MULTIPLE, height=6,
            font=FONTS.get_font("body")
        )
        self.numeric_cols_listbox.pack(fill=tk.X, pady=4)
        
        # Classifiers Card
        clf_card = Card(self.options_content, title="🤖 Classifiers", theme=self.theme_name)
        clf_card.pack(fill=tk.X, padx=8, pady=8)
        
        for name in self.CLASSIFIERS.keys():
            var = tk.BooleanVar(value=True)
            self._classifier_vars[name] = var
            cb = tk.Checkbutton(clf_card.content, text=name, variable=var,
                               bg=self.theme["bg_card"], font=FONTS.get_font("body"))
            cb.pack(anchor=tk.W)
        
        # Evaluation Method Card
        eval_card = Card(self.options_content, title="📏 Evaluation Method", theme=self.theme_name)
        eval_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.eval_method_var = tk.StringVar(value="Cross-Validation (5-fold)")
        for name in self.EVAL_METHODS.keys():
            tk.Radiobutton(eval_card.content, text=name, variable=self.eval_method_var,
                          value=name, bg=self.theme["bg_card"],
                          font=FONTS.get_font("body")).pack(anchor=tk.W)
        
        # Options Card
        opt_card = Card(self.options_content, title="⚙️ Options", theme=self.theme_name)
        opt_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.multilabel_var = tk.BooleanVar(value=False)
        LabeledCheckbox(
            opt_card.content, label="Multilabel mode (joint prediction)",
            variable=self.multilabel_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        # Action Buttons
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Run Classification", icon="▶️",
            command=self._run_classification, theme=self.theme_name,
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
            "🤖 Classification (ML)\n\n"
            "Build ML models to classify documents.\n\n"
            "Features:\n"
            "• Multiple classifiers (SVM, RF)\n"
            "• Cross-validation evaluation\n"
            "• Feature importance ranking\n"
            "• Confusion matrix\n"
            "\n"
            "Learn to predict group membership.\n\n"
            "Steps:\n"
            "1. Set up groups in Group Setup\n"
            "2. Select features\n"
            "3. Choose classifier\n"
            "4. Click 'Train & Evaluate'\n"
        )
        
        tk.Label(
            self.results_tab,
            text=msg,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)

    def _on_feature_type_change(self):
        """Handle feature type change."""
        if self.feature_type_var.get() == "text":
            self.text_col_frame.pack(fill=tk.X, pady=4)
            self.numeric_col_frame.pack_forget()
        else:
            self.text_col_frame.pack_forget()
            self.numeric_col_frame.pack(fill=tk.X, pady=4)
    
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
        # Clear existing
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
        
        # Text columns
        text_cols = [c for c in cols if any(kw in c.lower() for kw in 
                    ['abstract', 'title', 'keyword', 'description', 'text'])]
        if not text_cols:
            text_cols = cols[:10]
        self.text_combo.combobox['values'] = text_cols
        
        # Set default to "Processed Abstract" if available, otherwise first text column
        if 'Processed Abstract' in text_cols:
            self.text_col_var.set('Processed Abstract')
        elif text_cols:
            self.text_col_var.set(text_cols[0])
        
        # Numeric columns
        self.numeric_cols_listbox.delete(0, tk.END)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Get group_matrix to exclude its columns
        group_matrix = None
        if self.bib_group is not None and hasattr(self.bib_group, 'group_matrix'):
            group_matrix = self.bib_group.group_matrix
        elif self.bib is not None and hasattr(self.bib, 'group_matrix'):
            group_matrix = self.bib.group_matrix
        
        for col in numeric_cols:
            # Exclude group_matrix columns
            if group_matrix is not None and col in group_matrix.columns:
                continue
            self.numeric_cols_listbox.insert(tk.END, col)
    
    def _get_selected_groups(self) -> List[str]:
        """Get list of selected group names."""
        return [name for name, var in self._group_vars.items() if var.get()]
    
    def _get_selected_classifiers(self) -> Dict[str, Any]:
        """Get dictionary of selected classifiers."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.svm import SVC
        
        clf_map = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Naive Bayes": MultinomialNB(),
            "Support Vector Machine": SVC(probability=True),
        }
        
        selected = {}
        for name, var in self._classifier_vars.items():
            if var.get():
                key = self.CLASSIFIERS[name]
                selected[key] = clf_map[name]
        
        return selected
    
    def _run_classification(self):
        """Run the classification analysis."""
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
            messagebox.showwarning("No Groups Selected", "Please select at least one group to classify.")
            return
        
        selected_classifiers = self._get_selected_classifiers()
        if not selected_classifiers:
            messagebox.showwarning("No Classifiers", "Please select at least one classifier.")
            return
        
        self._show_loading("Running classification analysis...")
        self.update_idletasks()
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.model_selection import cross_val_score, LeaveOneOut, train_test_split
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            from sklearn.base import clone
            
            # Get feature settings
            feature_type = self.feature_type_var.get()
            
            # Build design matrix X
            if feature_type == "text":
                text_col = self.text_col_var.get()
                try:
                    max_features = int(self.max_features_var.get())
                except ValueError:
                    max_features = 500
                
                # Build TF-IDF features
                text_series = df[text_col].fillna("").astype(str)
                vectorizer = TfidfVectorizer(max_features=max_features)
                X = vectorizer.fit_transform(text_series).toarray()
            else:
                # Numeric features
                selected_idx = self.numeric_cols_listbox.curselection()
                if not selected_idx:
                    messagebox.showwarning("No Features", "Please select numeric feature columns.")
                    return
                
                feature_cols = [self.numeric_cols_listbox.get(i) for i in selected_idx]
                X = df[feature_cols].fillna(0).values
            
            # Get evaluation method
            eval_method = self.EVAL_METHODS[self.eval_method_var.get()]
            
            # Run classification for each group
            results = {}
            
            for grp in selected_groups:
                if grp not in group_matrix.columns:
                    continue
                    
                y = group_matrix[grp].values
                
                # Skip if only one class
                if len(np.unique(y)) < 2:
                    print(f"Skipping group '{grp}': only one class present")
                    continue
                
                group_results = {}
                
                for clf_name, clf in selected_classifiers.items():
                    try:
                        model = clone(clf)
                        
                        if eval_method == "cross_validation":
                            # 5-fold cross-validation
                            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                            accuracy = cv_scores.mean()
                            
                            # For other metrics, we need predictions
                            model.fit(X, y)
                            y_pred = model.predict(X)
                            
                        elif eval_method == "leave_one_out":
                            loo = LeaveOneOut()
                            y_pred_list = []
                            y_true_list = []
                            for train_idx, test_idx in loo.split(X):
                                m = clone(clf)
                                m.fit(X[train_idx], y[train_idx])
                                y_pred_list.append(m.predict(X[test_idx])[0])
                                y_true_list.append(y[test_idx][0])
                            y_pred = np.array(y_pred_list)
                            y_true = np.array(y_true_list)
                            accuracy = accuracy_score(y_true, y_pred)
                            model.fit(X, y)  # Fit on full data for AUC
                            
                        else:  # train_test
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
                            )
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                        
                        # Calculate metrics
                        if eval_method == "leave_one_out":
                            precision = precision_score(y_true, y_pred, zero_division=0)
                            recall = recall_score(y_true, y_pred, zero_division=0)
                            f1 = f1_score(y_true, y_pred, zero_division=0)
                        elif eval_method == "train_test":
                            precision = precision_score(y_test, y_pred, zero_division=0)
                            recall = recall_score(y_test, y_pred, zero_division=0)
                            f1 = f1_score(y_test, y_pred, zero_division=0)
                        else:
                            precision = precision_score(y, y_pred, zero_division=0)
                            recall = recall_score(y, y_pred, zero_division=0)
                            f1 = f1_score(y, y_pred, zero_division=0)
                        
                        # AUC
                        try:
                            if hasattr(model, 'predict_proba'):
                                if eval_method == "train_test":
                                    proba = model.predict_proba(X_test)[:, 1]
                                    roc_auc = roc_auc_score(y_test, proba)
                                else:
                                    proba = model.predict_proba(X)[:, 1]
                                    roc_auc = roc_auc_score(y, proba)
                            else:
                                roc_auc = np.nan
                        except:
                            roc_auc = np.nan
                        
                        group_results[clf_name] = {
                            "accuracy": float(accuracy),
                            "precision": float(precision),
                            "recall": float(recall),
                            "f1": float(f1),
                            "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else 0.0,
                        }
                        
                    except Exception as e:
                        print(f"Error with {clf_name} on {grp}: {e}")
                        group_results[clf_name] = {
                            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, 
                            "f1": 0.0, "roc_auc": 0.0
                        }
                
                results[grp] = group_results
            
            if not results:
                self._show_error("No results generated. Check that groups have variation.")
                return
            
            self._results = results
            self._display_results(results)
            
        except Exception as e:
            self._show_error(str(e))
            import traceback
            traceback.print_exc()
    
    def _display_results(self, results: Dict):
        """Display classification results."""
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
        
        # Convert results to DataFrame
        rows = []
        for group_or_label, models in results.items():
            for model_name, metrics in models.items():
                row = {"Group": group_or_label, "Model": model_name}
                row.update(metrics)
                rows.append(row)
        
        if not rows:
            self._show_error("No results generated.")
            return
        
        df = pd.DataFrame(rows)
        
        # Summary stats
        metric_cols = [c for c in df.columns if c not in ["Group", "Model"]]
        
        # Stats cards
        grid = CardGrid(self.results_tab, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 8))
        
        n_groups = df["Group"].nunique()
        n_models = df["Model"].nunique()
        
        # Best AUC
        if "roc_auc" in df.columns:
            best_auc = df["roc_auc"].max()
            best_auc_row = df.loc[df["roc_auc"].idxmax()]
            grid.add_card(StatsCard(grid, "Best AUC", f"{best_auc:.3f}", "🏆", self.theme_name))
        
        # Best Accuracy
        if "accuracy" in df.columns:
            best_acc = df["accuracy"].max()
            grid.add_card(StatsCard(grid, "Best Accuracy", f"{best_acc:.3f}", "✓", self.theme_name))
        
        grid.add_card(StatsCard(grid, "Groups", str(n_groups), "📊", self.theme_name))
        grid.add_card(StatsCard(grid, "Models", str(n_models), "🤖", self.theme_name))
        
        # Create notebook for different views
        notebook = ttk.Notebook(self.results_tab)
        notebook.pack(fill=tk.BOTH, expand=True, pady=8)
        
        # Tab 1: Full Results Table
        table_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(table_frame, text="Full Results")
        
        # Round numeric columns for display
        df_display = df.copy()
        for col in metric_cols:
            if col in df_display.columns:
                df_display[col] = df_display[col].round(4)
        
        table = DataTable(table_frame, theme=self.theme_name)
        table.pack(fill=tk.BOTH, expand=True)
        table.set_data(df_display)
        
        # Tab 2: Summary by Model
        model_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(model_frame, text="By Model")
        
        summary_model = df.groupby("Model")[metric_cols].agg(["mean", "std"]).round(4)
        summary_model.columns = [f"{col}_{stat}" for col, stat in summary_model.columns]
        summary_model = summary_model.reset_index()
        
        table_model = DataTable(model_frame, theme=self.theme_name)
        table_model.pack(fill=tk.BOTH, expand=True)
        table_model.set_data(summary_model)
        
        # Tab 3: Summary by Group
        group_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(group_frame, text="By Group")

        

        # Info tab

        info_frame = tk.Frame(notebook, bg=self.theme["bg_card"])

        notebook.add(info_frame, text="ℹ️ Info")

        self._create_info_content(info_frame)
        
        summary_group = df.groupby("Group")[metric_cols].agg(["mean", "std"]).round(4)
        summary_group.columns = [f"{col}_{stat}" for col, stat in summary_group.columns]
        summary_group = summary_group.reset_index()
        
        table_group = DataTable(group_frame, theme=self.theme_name)
        table_group.pack(fill=tk.BOTH, expand=True)
        table_group.set_data(summary_group)
        
        # Store DataFrame for export
        self._results_df = df
    
    def _export_results(self):
        """Export results to Excel."""
        if self._results is None or not hasattr(self, '_results_df'):
            messagebox.showwarning("No Results", "Please run classification first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Save Classification Results"
        )
        
        if not filename:
            return
        
        try:
            df = self._results_df
            metric_cols = [c for c in df.columns if c not in ["Group", "Model"]]
            
            with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
                # Full results
                df.to_excel(writer, sheet_name="Performance", index=False)
                
                # Summary by model
                summary_model = df.groupby("Model")[metric_cols].agg(["mean", "std", "min", "max"])
                summary_model.to_excel(writer, sheet_name="Summary_by_Model")
                
                # Summary by group
                summary_group = df.groupby("Group")[metric_cols].agg(["mean", "std", "min", "max"])
                summary_group.to_excel(writer, sheet_name="Summary_by_Group")
            
            messagebox.showinfo("Success", f"Results saved to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {e}")
    
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
GROUP CLASSIFICATION (ML)
═════════════════════════

Predict group membership using machine learning.

CLASSIFIERS
───────────
• Logistic Regression
  Simple, interpretable baseline
  
• Random Forest
  Ensemble of decision trees
  Feature importance
  
• Gradient Boosting (GBM)
  Sequential ensemble
  Often highest accuracy
  
• Support Vector Machine (SVM)
  Effective for text
  High-dimensional data
  
• Naive Bayes
  Fast, simple
  Text classification standard

EVALUATION METRICS
──────────────────
• Accuracy: Overall correct %
• Precision: TP / (TP + FP)
• Recall: TP / (TP + FN)
• F1 Score: Harmonic mean P & R
• AUC: Area under ROC curve

CROSS-VALIDATION
────────────────
• K-fold (default: 5)
• Stratified sampling
• Prevents overfitting

FEATURE IMPORTANCE
──────────────────
• Which terms predict groups
• Coefficient magnitudes
• Model interpretability
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
