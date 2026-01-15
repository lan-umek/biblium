# -*- coding: utf-8 -*-
"""
Group Associations Panel
========================
Panel for analyzing entity-group associations.

Features:
- Associate entities (keywords, authors, etc.) with groups
- Compute diversity and correspondence statistics
- Chi-square tests
- Correspondence analysis visualization
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
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox, LabeledSpinbox, LabeledEntry
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.plots import PlotFrame
from biblium.gui.widgets.tooltips import ToolTip

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


class GroupAssociationsPanel(BasePanel):
    """
    Panel for analyzing entity-group associations.
    
    Provides access to all associate_* methods from BiblioGroup.
    """
    
    title = "Group Associations"
    icon = "🔗"
    description = "Analyze entity-group relationships"
    requires_data = True
    
    # Entity configurations
    ENTITY_CONFIGS = {
        "sources": {
            "label": "Sources (Journals)",
            "method": "associate_sources",
            "description": "Associate journals with groups",
        },
        "author_keywords": {
            "label": "Author Keywords",
            "method": "associate_author_keywords",
            "description": "Associate author keywords with groups",
        },
        "index_keywords": {
            "label": "Index Keywords",
            "method": "associate_index_keywords",
            "description": "Associate index keywords with groups",
        },
        "abstract_words": {
            "label": "Abstract Words",
            "method": "associate_abstract_words",
            "description": "Associate abstract n-grams with groups",
        },
        "title_words": {
            "label": "Title Words",
            "method": "associate_title_words",
            "description": "Associate title n-grams with groups",
        },
        "authors": {
            "label": "Authors",
            "method": "associate_authors",
            "description": "Associate authors with groups",
        },
        "countries": {
            "label": "Countries",
            "method": "associate_countries",
            "description": "Associate countries with groups",
        },
        "affiliations": {
            "label": "Affiliations",
            "method": "associate_affiliations",
            "description": "Associate affiliations with groups",
        },
        "references": {
            "label": "References",
            "method": "associate_references",
            "description": "Associate cited references with groups",
        },
    }
    
    # Statistics options
    STATISTICS_OPTIONS = [
        ("diversity", "Diversity Index", "Shannon entropy of item distribution"),
        ("correspondence", "Correspondence Analysis", "Row/column coordinates"),
        ("chi_square", "Chi-square Test", "Statistical independence test"),
        ("svd", "SVD Analysis", "Singular value decomposition"),
        ("log_ratio", "Log-ratio", "Relative association strength"),
    ]
    
    @property
    def bib_group(self):
        """Get BiblioGroup instance."""
        # First check stored attribute (set by BasePanel or directly)
        if hasattr(self, '_bib_group') and self._bib_group is not None:
            return self._bib_group
        # Try to get from workspace (master.master since self.master is Notebook)
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
        
        # Entity Selection Card
        entity_card = Card(
            self.options_content,
            title="📋 Entity Selection",
            theme=self.theme_name
        )
        entity_card.pack(fill=tk.X, padx=8, pady=8)
        
        entity_values = [config["label"] for config in self.ENTITY_CONFIGS.values()]
        self.entity_var = tk.StringVar(value=entity_values[0])
        
        self.entity_combo = LabeledCombobox(
            entity_card.content,
            label="Entity Type:",
            values=entity_values,
            variable=self.entity_var,
            theme=self.theme_name,
            label_width=12
        )
        self.entity_combo.pack(fill=tk.X, pady=4)
        self.entity_combo.combobox.bind("<<ComboboxSelected>>", self._update_description)
        
        # Description
        self.entity_desc = tk.Label(
            entity_card.content,
            text="",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            wraplength=280
        )
        self.entity_desc.pack(fill=tk.X, pady=4)
        self._update_description()
        
        # Statistics Card
        stats_card = Card(
            self.options_content,
            title="📊 Statistics to Include",
            theme=self.theme_name
        )
        stats_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Checkboxes for each statistic
        self.stat_vars = {}
        for stat_id, stat_name, stat_desc in self.STATISTICS_OPTIONS:
            var = tk.BooleanVar(value=True)
            self.stat_vars[stat_id] = var
            
            cb = LabeledCheckbox(
                stats_card.content,
                label=stat_name,
                variable=var,
                theme=self.theme_name,
            )
            cb.pack(fill=tk.X, pady=1)
            ToolTip(cb, stat_desc)
        
        # Selection Criteria Card
        criteria_card = Card(
            self.options_content,
            title="🔍 Selection Criteria",
            theme=self.theme_name
        )
        criteria_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Top N items
        self.top_n = LabeledSpinbox(
            criteria_card.content,
            label="Top N Items:",
            from_=10, to=500, default=50,
            theme=self.theme_name,
            label_width=14
        )
        self.top_n.pack(fill=tk.X, pady=4)
        
        # Min frequency
        self.min_freq = LabeledSpinbox(
            criteria_card.content,
            label="Min Frequency:",
            from_=1, to=100, default=5,
            theme=self.theme_name,
            label_width=14
        )
        self.min_freq.pack(fill=tk.X, pady=4)
        
        # Filter Card
        filter_card = CollapsibleCard(
            self.options_content,
            title="🔍 Filtering",
            theme=self.theme_name,
            collapsed=True
        )
        filter_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Include pattern
        self.include_pattern = LabeledEntry(
            filter_card.content,
            label="Include (regex):",
            default="",
            theme=self.theme_name,
            label_width=14
        )
        self.include_pattern.pack(fill=tk.X, pady=4)
        
        # Exclude pattern
        self.exclude_pattern = LabeledEntry(
            filter_card.content,
            label="Exclude (regex):",
            default="",
            theme=self.theme_name,
            label_width=14
        )
        self.exclude_pattern.pack(fill=tk.X, pady=4)
        
        # Display Options Card
        display_card = CollapsibleCard(
            self.options_content,
            title="📊 Display Options",
            theme=self.theme_name,
            collapsed=True
        )
        display_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Show correspondence plot
        self.show_ca_plot_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            display_card.content,
            label="Show correspondence analysis plot",
            variable=self.show_ca_plot_var,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        # Show heatmap
        self.show_heatmap_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            display_card.content,
            label="Show association heatmap",
            variable=self.show_heatmap_var,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        # Run Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame,
            text="Compute Associations",
            icon="🔗",
            command=self._run_associations,
            theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
        
        # Quick actions
        quick_frame = tk.Frame(btn_frame, bg=self.theme["bg_secondary"])
        quick_frame.pack(fill=tk.X, pady=(8, 0))
        
        ThemedButton(
            quick_frame,
            text="All Entities",
            command=self._compute_all_associations,
            theme=self.theme_name,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        ThemedButton(
            quick_frame,
            text="Export",
            command=self._export_results,
            theme=self.theme_name,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
    
    def _check_groups(self) -> bool:
        """Check if groups are available."""
        if not self.bib:
            self._show_message("📂 Please load a dataset first.")
            return False
        
        if not self.bib_group:
            self._show_no_groups_message()
            return False
        
        return True
    
    def _show_message(self, message: str):
        """Show message in options."""
        tk.Label(
            self.options_content,
            text=message,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)
    
    def _show_no_groups_message(self):
        """Show no groups message."""
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
                 "Go to GROUPS → Setup Groups.",
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
    
    def _update_description(self, event=None):
        """Update entity description."""
        selected = self.entity_var.get()
        for key, config in self.ENTITY_CONFIGS.items():
            if config["label"] == selected:
                self.entity_desc.config(text=config["description"])
                break
    
    def _get_selected_entity_key(self) -> str:
        """Get key for selected entity."""
        selected = self.entity_var.get()
        for key, config in self.ENTITY_CONFIGS.items():
            if config["label"] == selected:
                return key
        return "author_keywords"
    
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

    def _run_associations(self):
        """Run association analysis."""
        if not self._check_groups():
            return
        
        entity_key = self._get_selected_entity_key()
        config = self.ENTITY_CONFIGS[entity_key]
        
        self._show_loading(f"Computing {config['label']} associations...")
        
        def do_associations():
            try:
                method_name = config["method"]
                print(f"[DEBUG] Running associations for entity_key='{entity_key}', method='{method_name}'")
                
                method = getattr(self.bib_group, method_name, None)
                print(f"[DEBUG] Method found: {method is not None}")
                
                # Build selected statistics
                selected_stats = []
                stat_map = {
                    "diversity": "diversity",
                    "correspondence": "correspondence", 
                    "chi_square": "chi2",
                    "svd": "svd",
                    "log_ratio": "log-ratio",
                }
                for stat, var in self.stat_vars.items():
                    if var.get() and stat in stat_map:
                        selected_stats.append(stat_map[stat])
                
                print(f"[DEBUG] Selected stats: {selected_stats}")
                
                kwargs = {
                    "top_n": self.top_n.get(),
                    "min_freq": self.min_freq.get(),
                }
                
                # Add include_stats if we have selections
                if selected_stats:
                    kwargs["include_stats"] = tuple(selected_stats)
                
                # Add filters if specified
                include = self.include_pattern.get().strip()
                if include:
                    kwargs["items_included"] = include
                
                exclude = self.exclude_pattern.get().strip()
                if exclude:
                    kwargs["items_excluded"] = exclude
                
                print(f"[DEBUG] Calling method with kwargs: {kwargs}")
                
                if method:
                    # Method returns self for chaining, result is stored as attribute
                    returned = method(**kwargs)
                    print(f"[DEBUG] Method returned: {type(returned)}")
                    
                    # Get the result from the stored attribute
                    # The attribute name is {domain_key}_associations
                    result_attr = f"{entity_key}_associations"
                    print(f"[DEBUG] Looking for attribute: '{result_attr}'")
                    
                    # List all attributes that contain 'associations'
                    all_attrs = [a for a in dir(self.bib_group) if 'association' in a.lower()]
                    print(f"[DEBUG] Available association attributes: {all_attrs}")
                    
                    result = getattr(self.bib_group, result_attr, None)
                    print(f"[DEBUG] Result type: {type(result)}, is None: {result is None}")
                    
                    if result is not None and hasattr(result, 'shape'):
                        print(f"[DEBUG] Result shape: {result.shape}")
                    elif result is not None and isinstance(result, dict):
                        print(f"[DEBUG] Result dict keys: {list(result.keys())}")
                else:
                    # Fallback: use generic associate_items
                    print(f"[DEBUG] Method not found, trying generic associate_items")
                    generic = getattr(self.bib_group, "associate_items", None)
                    if generic:
                        generic(domain_key=entity_key, **kwargs)
                        result_attr = f"{entity_key}_associations"
                        result = getattr(self.bib_group, result_attr, None)
                    else:
                        raise AttributeError(f"Method {method_name} not found")
                
                self.current_result = result
                self.current_entity = entity_key
                
                self.after(0, lambda: self._display_results(result, config))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: self._show_error(msg))
        
        threading.Thread(target=do_associations, daemon=True).start()
    
    def _display_results(self, result, config: Dict):
        """Display association results."""
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
        
        if result is None:
            self._show_initial_message("No results available.")
            return
        
        # Handle different result types
        if isinstance(result, dict):
            self._display_dict_results(result, config)
        elif isinstance(result, pd.DataFrame):
            self._display_df_results(result, config)
        elif hasattr(result, 'rm'):
            # This is a Relation object from utilsbib
            self._display_relation_results(result, config)
        else:
            self._show_initial_message(f"Unexpected result type: {type(result)}")
    
    def _display_relation_results(self, relation, config: Dict):
        """Display results from a Relation object - compute visualizations from matrix."""
        # Get the contingency matrix
        if relation.rm is None or relation.rm.empty:
            self._show_initial_message("No association data found.")
            return
        
        contingency = relation.rm  # Groups (rows) x Items (columns)
        
        # Store for export
        self._contingency_matrix = contingency
        
        # Get stats
        n_groups = contingency.shape[0]
        n_items = contingency.shape[1]
        total = contingency.values.sum()
        
        # Compute chi-square
        chi2_stat, chi2_pval, chi2_dof = self._compute_chi2(contingency)
        
        # Create tabbed interface
        notebook = ttk.Notebook(self.results_tab)
        notebook.pack(fill=tk.BOTH, expand=True, pady=8)
        
        # === Tab 1: Contingency Table ===
        table_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(table_frame, text="📋 Contingency")
        
        # Summary cards
        grid = CardGrid(table_frame, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(8, 16), padx=8)
        
        grid.add_card(StatsCard(grid, "Groups", str(n_groups), "📊", self.theme_name, accent=True))
        grid.add_card(StatsCard(grid, "Items", f"{n_items:,}", "📋", self.theme_name))
        grid.add_card(StatsCard(grid, "Total", f"{int(total):,}", "📈", self.theme_name))
        if chi2_stat is not None:
            grid.add_card(StatsCard(grid, "χ²", f"{chi2_stat:.1f}", "🎯", self.theme_name))
        
        # Table - transpose for display (items as rows)
        table = DataTable(table_frame, theme=self.theme_name)
        table.pack(fill=tk.BOTH, expand=True, pady=8, padx=8)
        
        display_df = contingency.T.reset_index()
        display_df.columns = ['Item'] + list(contingency.index)
        table.set_data(display_df)
        
        # === Tab 2: Heatmap ===
        if HAS_MATPLOTLIB:
            heatmap_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(heatmap_frame, text="🗺️ Heatmap")
            self._create_heatmap(heatmap_frame, contingency, config)
        
        # === Tab 3: Chi-square Residuals ===
        if chi2_stat is not None:
            chi2_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(chi2_frame, text="📊 Chi² Residuals")
            self._create_chi2_tab(chi2_frame, contingency, chi2_stat, chi2_pval, chi2_dof)
        
        # === Tab 4: Correspondence Analysis ===
        if HAS_MATPLOTLIB and n_groups >= 2 and n_items >= 2:
            ca_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(ca_frame, text="🗺️ CA Biplot")
            self._create_ca_plot(ca_frame, contingency, config)
        
        # === Tab 5: Top Associations ===
        top_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(top_frame, text="🏆 Top Pairs")
        self._create_top_pairs_tab(top_frame, contingency)
        
        # === Tab 6: Log-Ratio ===
        lr_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(lr_frame, text="📈 Log-Ratio")

        

        # Info tab

        info_frame = tk.Frame(notebook, bg=self.theme["bg_card"])

        notebook.add(info_frame, text="ℹ️ Info")

        self._create_info_content(info_frame)
        self._create_log_ratio_tab(lr_frame, contingency)
    
    def _compute_chi2(self, contingency):
        """Compute chi-square statistics from contingency table."""
        try:
            from scipy.stats import chi2_contingency
            chi2, p, dof, expected = chi2_contingency(contingency.values)
            return chi2, p, dof
        except:
            return None, None, None
    
    def _create_heatmap(self, parent, contingency, config):
        """Create heatmap visualization."""
        try:
            import seaborn as sns
            
            plot_frame = PlotFrame(parent, theme=self.theme_name, figsize=(10, 8), show_ai_button=True)
            plot_frame.pack(fill=tk.BOTH, expand=True, pady=8, padx=8)
            
            fig, ax = plot_frame.get_figure()
            ax.clear()
            
            # Transpose for display (items as rows, groups as columns)
            data = contingency.T
            
            # Create heatmap
            sns.heatmap(data, annot=True, fmt='.0f', cmap='YlOrRd',
                       ax=ax, cbar_kws={'label': 'Count'})
            
            ax.set_xlabel('Groups')
            ax.set_ylabel(config.get('label', 'Items'))
            ax.set_title(f"{config.get('label', 'Entity')}-Group Associations")
            
            # Rotate labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            plt.setp(ax.get_yticklabels(), rotation=0)
            
            fig.tight_layout()
            plot_frame.canvas.draw()
            
        except Exception as e:
            tk.Label(parent, text=f"Error creating heatmap: {e}",
                    bg=self.theme["bg_card"], fg="red").pack(pady=20)
    
    def _create_chi2_tab(self, parent, contingency, chi2_stat, chi2_pval, chi2_dof):
        """Create chi-square residuals tab."""
        # Info header
        info_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        info_frame.pack(fill=tk.X, pady=8, padx=8)
        
        sig_text = "✓ Significant" if chi2_pval and chi2_pval < 0.05 else "✗ Not significant"
        tk.Label(
            info_frame,
            text=f"χ² = {chi2_stat:.2f}, df = {chi2_dof}, p = {chi2_pval:.4f} ({sig_text})",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(anchor=tk.W)
        
        # Compute standardized residuals
        try:
            obs = contingency.values.astype(float)
            row_sums = obs.sum(axis=1, keepdims=True)
            col_sums = obs.sum(axis=0, keepdims=True)
            total = obs.sum()
            
            expected = (row_sums @ col_sums) / total
            with np.errstate(divide='ignore', invalid='ignore'):
                residuals = (obs - expected) / np.sqrt(expected)
                residuals = np.nan_to_num(residuals, nan=0.0, posinf=0.0, neginf=0.0)
            
            residuals_df = pd.DataFrame(residuals, index=contingency.index, columns=contingency.columns)
            
            # Create sorted pairs table
            pairs = []
            for group in residuals_df.index:
                for item in residuals_df.columns:
                    pairs.append({
                        'Group': group,
                        'Item': item,
                        'Observed': contingency.loc[group, item],
                        'Expected': expected[list(contingency.index).index(group), list(contingency.columns).index(item)],
                        'Residual': residuals_df.loc[group, item]
                    })
            
            pairs_df = pd.DataFrame(pairs)
            pairs_df = pairs_df.sort_values('Residual', ascending=False, key=abs)
            
            # Show top associations
            table = DataTable(parent, theme=self.theme_name)
            table.pack(fill=tk.BOTH, expand=True, pady=8, padx=8)
            
            # Format numbers
            pairs_df['Expected'] = pairs_df['Expected'].round(1)
            pairs_df['Residual'] = pairs_df['Residual'].round(2)
            table.set_data(pairs_df.head(100))
            
        except Exception as e:
            tk.Label(parent, text=f"Error computing residuals: {e}",
                    bg=self.theme["bg_card"], fg="red").pack(pady=20)
    
    def _create_ca_plot(self, parent, contingency, config):
        """Create correspondence analysis biplot using biblium's implementation."""
        try:
            from biblium import utilsbib
            import io
            from PIL import Image, ImageTk
            
            # Compute CA coordinates using biblium's robust implementation
            row_coords, col_coords, explained_inertia = utilsbib.compute_correspondence_analysis(
                contingency, n_components=2, clean_zeros=True
            )
            
            # Check if we have valid CA results
            if row_coords is None or col_coords is None:
                tk.Label(parent, text="Could not compute CA (insufficient data)",
                        bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(pady=20)
                return
            
            # Info header with explained inertia
            if explained_inertia is not None and len(explained_inertia) >= 2:
                info_frame = tk.Frame(parent, bg=self.theme["bg_card"])
                info_frame.pack(fill=tk.X, pady=8, padx=8)
                tk.Label(
                    info_frame,
                    text=f"Explained inertia: Dim1={explained_inertia[0]*100:.1f}%, Dim2={explained_inertia[1]*100:.1f}%",
                    font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"],
                    fg=self.theme["text_primary"],
                ).pack(anchor=tk.W)
            
            # Create plot using biblium's style (without gridlines)
            plot_frame = PlotFrame(parent, theme=self.theme_name, figsize=(10, 8), show_ai_button=True)
            plot_frame.pack(fill=tk.BOTH, expand=True, pady=8, padx=8)
            
            fig, ax = plot_frame.get_figure()
            ax.clear()
            ax.set_facecolor("white")
            ax.grid(False)
            
            # Get frequencies for size scaling
            row_freq = contingency.sum(axis=1)
            col_freq = contingency.sum(axis=0)
            
            size_scale = 300
            row_sizes = (row_freq.loc[row_coords.index] / row_freq.max() * size_scale).fillna(size_scale)
            col_sizes = (col_freq.loc[col_coords.index] / col_freq.max() * size_scale).fillna(size_scale / 2)
            
            # Get group colors if available
            group_colors = getattr(self.bib_group, 'group_colors', None) or {}
            
            # Plot groups (rows) - as circles with group colors
            for i, (idx, row) in enumerate(row_coords.iterrows()):
                if len(row_coords.columns) >= 2:
                    color = group_colors.get(idx, 'tab:blue')
                    size = row_sizes.loc[idx] if idx in row_sizes.index else size_scale
                    ax.scatter(row.iloc[0], row.iloc[1], c=[color], s=size, 
                              alpha=0.8, edgecolors='black', linewidths=1, zorder=3)
            
            # Plot items (columns) - as triangles
            if len(col_coords.columns) >= 2:
                ax.scatter(col_coords.iloc[:, 0], col_coords.iloc[:, 1],
                          c='tab:red', s=col_sizes, alpha=0.6, marker='^', 
                          label=config.get('label', 'Items'), zorder=2)
            
            # Annotate groups (always show all)
            for idx, row in row_coords.iterrows():
                if len(row_coords.columns) >= 2:
                    ax.annotate(str(idx), (row.iloc[0], row.iloc[1]),
                               fontsize=10, fontweight='bold', color='tab:blue',
                               xytext=(5, 5), textcoords='offset points')
            
            # Annotate items (show top items by distance from origin to reduce clutter)
            if len(col_coords.columns) >= 2:
                col_coords_copy = col_coords.copy()
                col_coords_copy['_dist'] = np.sqrt(col_coords.iloc[:, 0]**2 + col_coords.iloc[:, 1]**2)
                top_items = col_coords_copy.nlargest(25, '_dist')
                
                for idx, row in top_items.iterrows():
                    label = str(idx)[:30] + '...' if len(str(idx)) > 30 else str(idx)
                    ax.annotate(label, (row.iloc[0], row.iloc[1]),
                               fontsize=8, color='tab:red', alpha=0.8,
                               xytext=(3, 3), textcoords='offset points')
            
            # Reference lines (no grid)
            ax.axhline(0, color='gray', linewidth=0.5, zorder=1)
            ax.axvline(0, color='gray', linewidth=0.5, zorder=1)
            
            # Labels with explained inertia
            dim1_label = f"Dimension 1 ({explained_inertia[0]*100:.1f}%)" if explained_inertia else "Dimension 1"
            dim2_label = f"Dimension 2 ({explained_inertia[1]*100:.1f}%)" if explained_inertia and len(explained_inertia) > 1 else "Dimension 2"
            ax.set_xlabel(dim1_label)
            ax.set_ylabel(dim2_label)
            ax.set_title(f"Correspondence Analysis: Groups vs {config.get('label', 'Items')}")
            
            # Legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', 
                      markersize=12, markeredgecolor='black', label='Groups'),
                Line2D([0], [0], marker='^', color='w', markerfacecolor='tab:red', 
                      markersize=10, label=config.get('label', 'Items'))
            ]
            ax.legend(handles=legend_elements, loc='best')
            
            fig.tight_layout()
            plot_frame.canvas.draw()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            tk.Label(parent, text=f"Error creating CA plot: {e}",
                    bg=self.theme["bg_card"], fg="red").pack(pady=20)
    
    def _create_top_pairs_tab(self, parent, contingency):
        """Create top associations pairs table."""
        # Info
        tk.Label(
            parent,
            text="Top group-item associations by count",
            font=FONTS.get_font("subheading"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(fill=tk.X, pady=8, padx=8)
        
        # Create pairs dataframe
        pairs = []
        for group in contingency.index:
            for item in contingency.columns:
                count = contingency.loc[group, item]
                if count > 0:
                    pairs.append({
                        'Group': group,
                        'Item': item,
                        'Count': int(count),
                    })
        
        pairs_df = pd.DataFrame(pairs)
        pairs_df = pairs_df.sort_values('Count', ascending=False)
        
        # Add percentage
        total_by_group = contingency.sum(axis=1)
        pairs_df['% of Group'] = pairs_df.apply(
            lambda r: (r['Count'] / total_by_group[r['Group']] * 100) if total_by_group[r['Group']] > 0 else 0, 
            axis=1
        ).round(1)
        
        table = DataTable(parent, theme=self.theme_name)
        table.pack(fill=tk.BOTH, expand=True, pady=8, padx=8)
        table.set_data(pairs_df.head(100))
    
    def _create_log_ratio_tab(self, parent, contingency):
        """Create log-ratio analysis tab."""
        try:
            # Compute log-ratios
            obs = contingency.values.astype(float) + 0.5  # Add smoothing
            row_sums = obs.sum(axis=1, keepdims=True)
            col_sums = obs.sum(axis=0, keepdims=True)
            total = obs.sum()
            
            expected = (row_sums @ col_sums) / total
            
            with np.errstate(divide='ignore', invalid='ignore'):
                log_ratio = np.log2(obs / expected)
                log_ratio = np.nan_to_num(log_ratio, nan=0.0, posinf=3.0, neginf=-3.0)
            
            # Create pairs table
            pairs = []
            for i, group in enumerate(contingency.index):
                for j, item in enumerate(contingency.columns):
                    pairs.append({
                        'Group': group,
                        'Item': item,
                        'Count': int(contingency.iloc[i, j]),
                        'Log-Ratio': log_ratio[i, j]
                    })
            
            pairs_df = pd.DataFrame(pairs)
            pairs_df = pairs_df.sort_values('Log-Ratio', ascending=False, key=abs)
            pairs_df['Log-Ratio'] = pairs_df['Log-Ratio'].round(2)
            
            # Info
            tk.Label(
                parent,
                text="Log-ratio (log₂) measures over/under-representation. Positive = over-represented in group.",
                font=FONTS.get_font("small"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
                wraplength=600,
            ).pack(fill=tk.X, pady=8, padx=8)
            
            table = DataTable(parent, theme=self.theme_name)
            table.pack(fill=tk.BOTH, expand=True, pady=8, padx=8)
            table.set_data(pairs_df.head(100))
            
        except Exception as e:
            tk.Label(parent, text=f"Error computing log-ratios: {e}",
                    bg=self.theme["bg_card"], fg="red").pack(pady=20)
    
    def _display_dict_results(self, result: Dict, config: Dict):
        """Display results returned as dictionary."""
        # Summary cards
        grid = CardGrid(self.results_tab, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 16))
        
        n_groups = len(self.bib_group.groups)
        grid.add_card(StatsCard(grid, "Groups", str(n_groups), "📊", self.theme_name, accent=True))
        
        # Check for common result keys
        if 'contingency_table' in result:
            ct = result['contingency_table']
            n_items = len(ct)
            grid.add_card(StatsCard(grid, "Items", f"{n_items:,}", "📋", self.theme_name))
        
        if 'chi_square' in result:
            chi2 = result['chi_square']
            if isinstance(chi2, dict) and 'statistic' in chi2:
                grid.add_card(StatsCard(grid, "χ²", f"{chi2['statistic']:.1f}", "📈", self.theme_name))
                if 'p_value' in chi2:
                    p_val = chi2['p_value']
                    sig = "✓" if p_val < 0.05 else "✗"
                    grid.add_card(StatsCard(grid, "Significant", sig, "🎯", self.theme_name))
        
        # Visualizations
        if self.show_heatmap_var.get() and 'contingency_table' in result and HAS_MATPLOTLIB:
            heatmap_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
            heatmap_frame.pack(fill=tk.BOTH, expand=True, pady=8)
            self._create_heatmap(heatmap_frame, result['contingency_table'], config)
        
        if self.show_ca_plot_var.get() and 'correspondence' in result and HAS_MATPLOTLIB:
            ca_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
            ca_frame.pack(fill=tk.BOTH, expand=True, pady=8)
            self._create_ca_plot(ca_frame, result['contingency_table'], config)
        
        # Show contingency table
        if 'contingency_table' in result:
            tk.Label(
                self.results_tab,
                text="📋 Contingency Table",
                font=FONTS.get_font("subheading"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
            ).pack(fill=tk.X, pady=(8, 4))
            
            table = DataTable(self.results_tab, theme=self.theme_name)
            table.pack(fill=tk.BOTH, expand=True, pady=8)
            
            ct = result['contingency_table']
            display_df = ct.reset_index() if ct.index.name else ct
            table.set_data(display_df)
    
    def _display_df_results(self, result: pd.DataFrame, config: Dict):
        """Display DataFrame results."""
        # Summary cards
        grid = CardGrid(self.results_tab, columns=3, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 16))
        
        n_items = len(result)
        n_groups = len(self.bib_group.groups)
        
        grid.add_card(StatsCard(grid, "Items", f"{n_items:,}", "📋", self.theme_name, accent=True))
        grid.add_card(StatsCard(grid, "Groups", str(n_groups), "📊", self.theme_name))
        grid.add_card(StatsCard(grid, "Columns", str(len(result.columns)), "📐", self.theme_name))
        
        # Heatmap visualization
        if self.show_heatmap_var.get() and HAS_MATPLOTLIB:
            heatmap_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
            heatmap_frame.pack(fill=tk.BOTH, expand=True, pady=8)
            self._create_heatmap(heatmap_frame, result, config)
        
        # Results table
        tk.Label(
            self.results_tab,
            text=f"📋 {config['label']} Associations",
            font=FONTS.get_font("subheading"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(fill=tk.X, pady=(8, 4))
        
        table = DataTable(self.results_tab, theme=self.theme_name)
        table.pack(fill=tk.BOTH, expand=True, pady=8)
        
        display_df = result.reset_index() if result.index.name else result
        table.set_data(display_df)
    
    def _compute_all_associations(self):
        """Compute associations for all entity types."""
        if not self._check_groups():
            return
        
        self._show_loading("Computing all associations...")
        
        def do_all():
            results = {}
            errors = []
            
            for key, config in self.ENTITY_CONFIGS.items():
                try:
                    method = getattr(self.bib_group, config["method"], None)
                    if method:
                        result = method(
                            top_n=self.top_n.get(),
                            min_freq=self.min_freq.get(),
                        )
                        results[key] = result
                except Exception as e:
                    errors.append(f"{config['label']}: {str(e)}")
            
            self.all_results = results
            self.after(0, lambda: self._display_all_summary(results, errors))
        
        threading.Thread(target=do_all, daemon=True).start()
    
    def _display_all_summary(self, results: Dict, errors: List[str]):
        """Display summary of all associations."""
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
        
        n_success = len(results)
        n_errors = len(errors)
        
        grid = CardGrid(self.results_tab, columns=3, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 16))
        
        grid.add_card(StatsCard(grid, "Completed", str(n_success), "✅", self.theme_name, accent=True))
        grid.add_card(StatsCard(grid, "Errors", str(n_errors), "❌", self.theme_name))
        grid.add_card(StatsCard(grid, "Groups", str(len(self.bib_group.groups)), "📊", self.theme_name))
        
        # Summary table
        tk.Label(
            self.results_tab,
            text="📋 Association Summary",
            font=FONTS.get_font("subheading"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(fill=tk.X, pady=(8, 4))
        
        summary_data = []
        for key, result in results.items():
            config = self.ENTITY_CONFIGS[key]
            if isinstance(result, dict) and 'contingency_table' in result:
                n_items = len(result['contingency_table'])
            elif isinstance(result, pd.DataFrame):
                n_items = len(result)
            else:
                n_items = 0
            
            summary_data.append({
                "Entity Type": config["label"],
                "Items": n_items,
                "Status": "✅",
            })
        
        for error in errors:
            entity_name = error.split(":")[0]
            summary_data.append({
                "Entity Type": entity_name,
                "Items": 0,
                "Status": "❌",
            })
        
        table = DataTable(self.results_tab, theme=self.theme_name, height=12)
        table.pack(fill=tk.BOTH, expand=True, pady=8)
        table.set_data(pd.DataFrame(summary_data))
    
    def _export_results(self):
        """Export results."""
        if not hasattr(self, 'current_result') or self.current_result is None:
            if hasattr(self, 'all_results') and self.all_results:
                self._export_all_results()
                return
            messagebox.showwarning("No Results", "Please compute associations first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx"), ("CSV Files", "*.csv")],
            title="Export Associations",
        )
        
        if filename:
            try:
                if isinstance(self.current_result, dict):
                    # Export dict with multiple sheets
                    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                        for key, val in self.current_result.items():
                            if isinstance(val, pd.DataFrame):
                                val.to_excel(writer, sheet_name=key[:31])
                elif isinstance(self.current_result, pd.DataFrame):
                    if filename.endswith('.csv'):
                        self.current_result.to_csv(filename)
                    else:
                        self.current_result.to_excel(filename)
                
                messagebox.showinfo("Success", f"Exported to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
    
    def _export_all_results(self):
        """Export all results to Excel."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx")],
            title="Export All Associations",
        )
        
        if filename:
            try:
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    for key, result in self.all_results.items():
                        sheet_name = self.ENTITY_CONFIGS[key]["label"][:31]
                        if isinstance(result, dict) and 'contingency_table' in result:
                            result['contingency_table'].to_excel(writer, sheet_name=sheet_name)
                        elif isinstance(result, pd.DataFrame):
                            result.to_excel(writer, sheet_name=sheet_name)
                
                messagebox.showinfo("Success", f"Exported to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
TERM ASSOCIATIONS
═════════════════

Find distinctive terms for each group.

METHODS
───────
• Chi-square test
  Tests term-group independence
  
• TF-IDF weighting
  Term importance in group
  
• Odds Ratio
  How much more likely in group
  
• Log-likelihood
  Statistical significance

OUTPUT
──────
• Ranked terms per group
• Association strength
• P-values (significance)
• Direction (+associated, -disassociated)

INTERPRETATION
──────────────
• High OR: Characteristic of group
• Low OR: Absent from group
• Check both directions

USE CASES
─────────
• Characterize research themes
• Compare vocabularies
• Validate group definitions
• Label clusters
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
        """Refresh panel."""
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
