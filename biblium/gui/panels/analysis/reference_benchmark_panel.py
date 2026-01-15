# -*- coding: utf-8 -*-
"""
Reference Benchmark Panel
=========================
Compare analyzed dataset distributions against global reference data (OpenAlex or user-provided).

This panel allows users to:
1. Compare Scientific Production (year distribution) against global trends
2. Compare SDG distribution against global research patterns
3. Compare Country, Document Type, Open Access distributions
4. Use OpenAlex as default reference or provide custom reference data
5. View statistical significance of differences

@author: Lan.Umek
@version: 2.9.0
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
from typing import Dict, List, Optional, Tuple

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledSpinbox, LabeledCheckbox, LabeledEntry
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.plots import add_plot_context_menu, make_canvas_resizable

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Available comparison types with descriptions
COMPARISON_TYPES = [
    ("Scientific Production (Year)", "Year", "Compare publication year distribution against global trends"),
    ("SDG Distribution", "SDG", "Compare Sustainable Development Goals distribution"),
    ("Country Distribution", "Country", "Compare author country distribution"),
    ("Document Type", "Document Type", "Compare document type distribution (article, review, etc.)"),
    ("Open Access Status", "Open Access", "Compare open access rates"),
]


class ReferenceBenchmarkPanel(BasePanel):
    """
    Panel for comparing dataset distributions against reference benchmarks.
    
    Computes percentage point differences between observed and reference
    distributions to identify over- and under-representation.
    """
    
    title = "Reference Benchmark"
    icon = "📊"
    description = "Compare your dataset against global research patterns (OpenAlex or custom reference)"
    requires_data = True
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._result_df = None
        self._current_fig = None
        self._reference_df = None
        self._custom_reference_path = None
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._run_analysis
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Comparison Type Card
        type_card = Card(self.options_content, title="📊 Comparison Type", theme=self.theme_name)
        type_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Dropdown for comparison type
        type_values = [t[0] for t in COMPARISON_TYPES]
        self.comparison_type = LabeledCombobox(
            type_card.content, label="Compare:",
            values=type_values,
            default=type_values[0],
            theme=self.theme_name, label_width=10,
        )
        self.comparison_type.pack(fill=tk.X, pady=4)
        
        # Bind to update description
        self.comparison_type.combobox.bind("<<ComboboxSelected>>", self._on_type_change)
        
        # Description label
        self.type_desc_label = tk.Label(
            type_card.content,
            text=COMPARISON_TYPES[0][2],
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
            wraplength=250,
            justify=tk.LEFT,
        )
        self.type_desc_label.pack(fill=tk.X, pady=(4, 8))
        
        # Reference Data Card
        ref_card = Card(self.options_content, title="🌐 Reference Data", theme=self.theme_name)
        ref_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Reference source selection
        self.ref_source_var = tk.StringVar(value="openalex")
        
        openalex_frame = tk.Frame(ref_card.content, bg=self.theme["bg_card"])
        openalex_frame.pack(fill=tk.X, pady=4)
        
        self.openalex_radio = tk.Radiobutton(
            openalex_frame, text="OpenAlex (Global)",
            variable=self.ref_source_var, value="openalex",
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            activebackground=self.theme["bg_card"],
            selectcolor=self.theme["bg_card"],
            command=self._on_ref_source_change,
        )
        self.openalex_radio.pack(side=tk.LEFT)
        
        custom_frame = tk.Frame(ref_card.content, bg=self.theme["bg_card"])
        custom_frame.pack(fill=tk.X, pady=4)
        
        self.custom_radio = tk.Radiobutton(
            custom_frame, text="Custom Reference File",
            variable=self.ref_source_var, value="custom",
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            activebackground=self.theme["bg_card"],
            selectcolor=self.theme["bg_card"],
            command=self._on_ref_source_change,
        )
        self.custom_radio.pack(side=tk.LEFT)
        
        # Custom file selection
        self.custom_file_frame = tk.Frame(ref_card.content, bg=self.theme["bg_card"])
        self.custom_file_frame.pack(fill=tk.X, pady=4)
        
        self.custom_file_label = tk.Label(
            self.custom_file_frame,
            text="No file selected",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
            anchor=tk.W,
        )
        self.custom_file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.browse_btn = ThemedButton(
            self.custom_file_frame, text="Browse...",
            command=self._browse_reference_file,
            style="secondary",
            theme=self.theme_name,
        )
        self.browse_btn.pack(side=tk.RIGHT, padx=(8, 0))
        
        # Initially disable custom file selection
        self._toggle_custom_file(False)
        
        # Warning about OpenAlex
        self.warning_frame = tk.Frame(ref_card.content, bg="#fff3cd")
        self.warning_frame.pack(fill=tk.X, pady=(8, 4), padx=2)
        
        warning_text = (
            "⚠️ Using OpenAlex as reference. This represents global research patterns "
            "which may differ from your database (Scopus, WoS). For database-specific "
            "comparisons, provide custom reference data."
        )
        self.warning_label = tk.Label(
            self.warning_frame,
            text=warning_text,
            font=FONTS.get_font("small"),
            bg="#fff3cd",
            fg="#856404",
            wraplength=240,
            justify=tk.LEFT,
            padx=8, pady=6,
        )
        self.warning_label.pack(fill=tk.X)
        
        # Year Range Card (for Scientific Production)
        self.year_range_card = Card(self.options_content, title="📅 Year Range", theme=self.theme_name)
        self.year_range_card.pack(fill=tk.X, padx=8, pady=8)
        
        year_help = tk.Label(
            self.year_range_card.content,
            text="Filter years for comparison (leave empty for all years in dataset)",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
            wraplength=240,
            justify=tk.LEFT,
        )
        year_help.pack(fill=tk.X, pady=(0, 8))
        
        year_row = tk.Frame(self.year_range_card.content, bg=self.theme["bg_card"])
        year_row.pack(fill=tk.X, pady=4)
        
        tk.Label(
            year_row, text="From:", font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT)
        
        self.year_from_var = tk.StringVar(value="")
        self.year_from_entry = tk.Entry(
            year_row, textvariable=self.year_from_var, width=8,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_input"], fg=self.theme["text_primary"],
            relief=tk.FLAT, highlightthickness=1,
            highlightbackground=self.theme["border"],
        )
        self.year_from_entry.pack(side=tk.LEFT, padx=(4, 12))
        
        tk.Label(
            year_row, text="To:", font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT)
        
        self.year_to_var = tk.StringVar(value="")
        self.year_to_entry = tk.Entry(
            year_row, textvariable=self.year_to_var, width=8,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_input"], fg=self.theme["text_primary"],
            relief=tk.FLAT, highlightthickness=1,
            highlightbackground=self.theme["border"],
        )
        self.year_to_entry.pack(side=tk.LEFT, padx=(4, 0))
        
        # Settings Card
        settings_card = Card(self.options_content, title="⚙️ Settings", theme=self.theme_name)
        settings_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.threshold_spin = LabeledSpinbox(
            settings_card.content, label="Threshold (pp):",
            from_=1, to=10, default=1,
            theme=self.theme_name, label_width=14,
        )
        self.threshold_spin.pack(fill=tk.X, pady=4)
        
        threshold_help = tk.Label(
            settings_card.content,
            text="Categories with difference > threshold are classified as over/under-represented",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
            wraplength=240,
            justify=tk.LEFT,
        )
        threshold_help.pack(fill=tk.X, pady=(0, 8))
        
        # Chi-square test option
        self.chi_square_var = tk.BooleanVar(value=True)
        self.chi_square_check = LabeledCheckbox(
            settings_card.content, label="Compute chi-square test",
            variable=self.chi_square_var,
            theme=self.theme_name,
        )
        self.chi_square_check.pack(fill=tk.X, pady=4)
        
        # Plot Options Card
        plot_card = Card(self.options_content, title="🎨 Plot Options", theme=self.theme_name)
        plot_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.cmap = LabeledCombobox(
            plot_card.content, label="Color Map:",
            values=["RdBu", "RdYlBu", "coolwarm", "bwr", "seismic", "PiYG", "PRGn"],
            default="RdBu",
            theme=self.theme_name, label_width=12,
        )
        self.cmap.pack(fill=tk.X, pady=4)
        
        cmap_help = tk.Label(
            plot_card.content,
            text="Blue = over-represented, Red = under-represented",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
        )
        cmap_help.pack(fill=tk.X, pady=(0, 4))
        
        self.show_comparison_var = tk.BooleanVar(value=True)
        self.show_comparison_check = LabeledCheckbox(
            plot_card.content, label="Show distribution comparison",
            variable=self.show_comparison_var,
            theme=self.theme_name,
        )
        self.show_comparison_check.pack(fill=tk.X, pady=4)
        
        # Run button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_primary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        self.run_btn = ActionButton(
            btn_frame, text="Run Analysis",
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
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Difference Plot tab
        self.diff_plot_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.diff_plot_frame, text="📊 Difference")
        
        # Comparison Plot tab
        self.comp_plot_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.comp_plot_frame, text="📈 Comparison")
        
        # Data tab
        self.data_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.data_frame, text="📋 Data")
        
        # Summary tab
        self.summary_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.summary_frame, text="📝 Summary")
        
        # Info tab
        self.info_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.info_frame, text="ℹ️ Info")
        
        # Initialize with placeholder
        self._show_placeholder()
        self._create_info_tab()
    
    def _show_placeholder(self):
        """Show placeholder message in plot frames."""
        for frame in [self.diff_plot_frame, self.comp_plot_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
            
            tk.Label(
                frame,
                text="Select comparison type and click 'Run Analysis'\nto compare against reference data",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_secondary"],
                justify=tk.CENTER,
            ).pack(expand=True)
    
    def _create_info_tab(self):
        """Create the info tab with documentation."""
        for widget in self.info_frame.winfo_children():
            widget.destroy()
        
        info_text = """
REFERENCE BENCHMARK ANALYSIS
════════════════════════════

This analysis compares your dataset's distribution against a global 
reference baseline to identify over- and under-representation.

HOW IT WORKS
────────────
1. Your dataset's distribution is computed (e.g., % of papers per year)
2. Reference distribution is fetched from OpenAlex (or custom file)
3. Percentage point difference is calculated for each category
4. Categories are classified as over/under-represented

SUPPORTED COMPARISONS
─────────────────────
• Scientific Production (Year)
  Compare year distribution against global publication trends.
  Shows if your dataset is biased toward recent/older literature.

• SDG Distribution  
  Compare Sustainable Development Goals against global research.
  Requires SDG columns (run identify_sdgs() first).
  Shows which SDGs are over/under-studied in your corpus.

• Country Distribution
  Compare author country distribution against global patterns.
  Shows geographic bias in your dataset.

• Document Type
  Compare article/review/conference paper ratios.

• Open Access Status
  Compare OA rates against global average.

INTERPRETING RESULTS
────────────────────
• Difference (pp): Percentage point difference
  +5 pp means 5% more in your dataset than globally
  -3 pp means 3% less in your dataset than globally

• Representation:
  "Over-represented": Difference > threshold
  "Under-represented": Difference < -threshold
  "As expected": Within threshold

• Chi-square test:
  Tests if distribution differs significantly from reference.
  p < 0.05 suggests significant difference.

REFERENCE DATA
──────────────
• OpenAlex (default): Uses global publication statistics
  ⚠️ May not match Scopus/WoS coverage exactly
  
• Custom reference: Provide your own baseline
  Excel/CSV with columns: [Category, Count]
  Use database-specific export for accurate comparison

VISUALIZATION
─────────────
• Difference plot: Diverging bars showing pp difference
  Blue = over-represented, Red = under-represented
  
• Comparison plot: Side-by-side bars showing both distributions

EXAMPLE USE CASES
─────────────────
1. "Is my SDG 3 (Health) research representative of global trends?"
2. "Are 2020-2024 papers over-represented in my climate dataset?"
3. "Is my corpus biased toward US/European research?"
"""
        
        text_widget = tk.Text(
            self.info_frame,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            wrap=tk.WORD,
            padx=16, pady=16,
            relief=tk.FLAT,
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
    
    def _on_type_change(self, event=None):
        """Handle comparison type change."""
        selected = self.comparison_type.get()
        for display, key, desc in COMPARISON_TYPES:
            if display == selected:
                self.type_desc_label.config(text=desc)
                # Show year range card only for Scientific Production (Year)
                if key == "Year":
                    self.year_range_card.pack(fill=tk.X, padx=8, pady=8)
                    # Reorder to keep it in the right place
                    self.year_range_card.pack_configure(after=self.warning_frame)
                else:
                    self.year_range_card.pack_forget()
                break
    
    def _on_ref_source_change(self):
        """Handle reference source radio button change."""
        is_custom = self.ref_source_var.get() == "custom"
        self._toggle_custom_file(is_custom)
        
        # Show/hide OpenAlex warning
        if is_custom:
            self.warning_frame.pack_forget()
        else:
            self.warning_frame.pack(fill=tk.X, pady=(8, 4), padx=2)
    
    def _toggle_custom_file(self, enabled: bool):
        """Enable/disable custom file selection."""
        state = tk.NORMAL if enabled else tk.DISABLED
        self.browse_btn.config(state=state)
        fg = self.theme["text_primary"] if enabled else self.theme["text_secondary"]
        self.custom_file_label.config(fg=fg)
    
    def _browse_reference_file(self):
        """Open file dialog to select custom reference file."""
        filetypes = [
            ("Excel files", "*.xlsx *.xls"),
            ("CSV files", "*.csv"),
            ("All files", "*.*"),
        ]
        filepath = filedialog.askopenfilename(
            title="Select Reference Data File",
            filetypes=filetypes,
        )
        if filepath:
            self._custom_reference_path = filepath
            filename = os.path.basename(filepath)
            self.custom_file_label.config(text=filename)
    
    def _get_comparison_key(self) -> str:
        """Get the internal key for selected comparison type."""
        selected = self.comparison_type.get()
        for display, key, desc in COMPARISON_TYPES:
            if display == selected:
                return key
        return "Year"
    
    def _run_analysis(self):
        """Run the reference benchmark analysis."""
        if self.bib is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        # Check for SDG columns if SDG comparison selected
        comparison_key = self._get_comparison_key()
        if comparison_key == "SDG":
            sdg_cols = [c for c in self.bib.df.columns if c.startswith("SDG") and len(c) <= 6]
            if not sdg_cols:
                messagebox.showwarning(
                    "SDG Columns Not Found",
                    "No SDG columns found in dataset.\n\n"
                    "Please run 'Identify SDGs' first:\n"
                    "  ba.identify_sdgs()\n\n"
                    "This will create SDG01-SDG17 columns."
                )
                return
        
        # Check custom reference file if selected
        if self.ref_source_var.get() == "custom":
            if not self._custom_reference_path:
                messagebox.showwarning(
                    "No Reference File",
                    "Please select a custom reference file."
                )
                return
        
        self.run_btn.config(state=tk.DISABLED, text="⏳ Analyzing...")
        
        # Capture values before thread
        threshold = float(self.threshold_spin.get())
        cmap = self.cmap.get()
        show_comparison = self.show_comparison_var.get()
        compute_chi_square = self.chi_square_var.get()
        use_custom = self.ref_source_var.get() == "custom"
        custom_path = self._custom_reference_path if use_custom else None
        
        # Capture year range (for Year comparison)
        year_from_str = self.year_from_var.get().strip()
        year_to_str = self.year_to_var.get().strip()
        user_year_range = None
        if year_from_str or year_to_str:
            try:
                year_from = int(year_from_str) if year_from_str else None
                year_to = int(year_to_str) if year_to_str else None
                user_year_range = (year_from, year_to)
            except ValueError:
                messagebox.showwarning("Invalid Year", "Please enter valid year numbers.")
                self.run_btn.config(state=tk.NORMAL, text="▶ Run Analysis")
                return
        
        def do_analysis():
            error_info = None
            try:
                # Import representation module
                from biblium.representation import (
                    compute_relative_representation,
                    plot_relative_representation,
                    plot_distribution_comparison,
                    fetch_openalex_yearly_counts,
                    fetch_openalex_sdg_counts,
                    fetch_openalex_country_counts,
                    fetch_openalex_doctype_counts,
                    fetch_openalex_oa_counts,
                    SUPPORTED_REFERENCES,
                )
                from scipy import stats as scipy_stats
                
                # Determine year range for filtering
                effective_year_range = None
                if comparison_key == "Year" and "Year" in self.bib.df.columns:
                    # Get data year range
                    data_min_year = int(self.bib.df["Year"].min())
                    data_max_year = int(self.bib.df["Year"].max())
                    
                    # Apply user-specified range if provided
                    if user_year_range:
                        from_year = user_year_range[0] if user_year_range[0] else data_min_year
                        to_year = user_year_range[1] if user_year_range[1] else data_max_year
                        effective_year_range = (max(from_year, data_min_year), min(to_year, data_max_year))
                    else:
                        effective_year_range = (data_min_year, data_max_year)
                elif "Year" in self.bib.df.columns:
                    # For other comparisons, still use year range for OpenAlex filtering
                    effective_year_range = (
                        int(self.bib.df["Year"].min()),
                        int(self.bib.df["Year"].max())
                    )
                
                # Get observed data
                observed_df = self._get_observed_distribution(comparison_key, effective_year_range)
                
                # Get reference data
                if custom_path:
                    # Load custom reference
                    if custom_path.endswith('.csv'):
                        reference_df = pd.read_csv(custom_path)
                    else:
                        reference_df = pd.read_excel(custom_path)
                    
                    # Validate columns
                    if 'Count' not in reference_df.columns:
                        raise ValueError("Reference file must have a 'Count' column")
                    
                    # Try to find category column
                    cat_col = None
                    for col in reference_df.columns:
                        if col != 'Count' and col != 'Percentage':
                            cat_col = col
                            break
                    if cat_col is None:
                        raise ValueError("Reference file must have a category column")
                    
                    reference_df = reference_df.rename(columns={cat_col: comparison_key})
                    
                    # Filter reference by year range if Year comparison
                    if comparison_key == "Year" and effective_year_range:
                        reference_df = reference_df[
                            (reference_df[comparison_key] >= effective_year_range[0]) &
                            (reference_df[comparison_key] <= effective_year_range[1])
                        ]
                else:
                    # Fetch from OpenAlex
                    if comparison_key == "Year" and effective_year_range:
                        reference_df = fetch_openalex_yearly_counts(
                            effective_year_range[0], effective_year_range[1]
                        )
                    elif comparison_key == "SDG":
                        reference_df = fetch_openalex_sdg_counts(year_range=effective_year_range)
                    elif comparison_key == "Country":
                        reference_df = fetch_openalex_country_counts(year_range=effective_year_range)
                    elif comparison_key == "Document Type":
                        reference_df = fetch_openalex_doctype_counts(year_range=effective_year_range)
                    elif comparison_key == "Open Access":
                        reference_df = fetch_openalex_oa_counts(year_range=effective_year_range)
                    else:
                        raise ValueError(f"Unknown comparison type: {comparison_key}")
                
                # Compute relative representation
                result_df = compute_relative_representation(
                    observed_df,
                    reference_df,
                    category_col=comparison_key,
                    count_col="Count",
                    threshold=threshold,
                )
                
                # Chi-square test
                chi_square_result = None
                if compute_chi_square and len(result_df) > 1:
                    observed_counts = result_df["Observed Count"].values
                    reference_counts = result_df["Reference Count"].values
                    
                    # Scale reference to match observed total
                    observed_total = observed_counts.sum()
                    reference_total = reference_counts.sum()
                    expected_counts = reference_counts * (observed_total / reference_total)
                    
                    # Only compute if we have enough data
                    if (expected_counts >= 5).sum() >= len(expected_counts) * 0.8:
                        chi2, p_value = scipy_stats.chisquare(observed_counts, expected_counts)
                        chi_square_result = {
                            "chi2": chi2,
                            "p_value": p_value,
                            "df": len(observed_counts) - 1,
                            "significant": p_value < 0.05,
                        }
                
                # Store results
                self._result_df = result_df
                self._chi_square_result = chi_square_result
                self._comparison_key = comparison_key
                self._used_openalex = not use_custom
                
                # Create plots (in main thread to avoid threading issues)
                self.after(0, lambda: self._create_plots(result_df, comparison_key, cmap, show_comparison))
                self.after(0, self._on_analysis_success)
                
            except Exception as exc:
                import traceback
                error_info = (str(exc), traceback.format_exc())
                self.after(0, lambda ei=error_info: self._on_analysis_error(ei[0], ei[1]))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _get_observed_distribution(self, comparison_key: str, year_range: tuple = None) -> pd.DataFrame:
        """Get observed distribution from dataset.
        
        Parameters
        ----------
        comparison_key : str
            Type of comparison (Year, SDG, Country, etc.)
        year_range : tuple, optional
            (from_year, to_year) to filter data. Only used for Year comparison.
        """
        df = self.bib.df
        
        if comparison_key == "Year":
            # Filter by year range if specified
            if year_range:
                df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]
            
            counts = df["Year"].value_counts().reset_index()
            counts.columns = ["Year", "Count"]
            counts["Year"] = counts["Year"].astype(int)
            
            # Filter again to ensure we only have years in range (handles edge cases)
            if year_range:
                counts = counts[(counts["Year"] >= year_range[0]) & (counts["Year"] <= year_range[1])]
            
            return counts.sort_values("Year")
        
        elif comparison_key == "SDG":
            # Sum each SDG column
            # Look for columns like SDG01, SDG1, SDG 1, etc.
            sdg_cols = [c for c in df.columns if c.upper().startswith("SDG") and any(char.isdigit() for char in c)]
            sdg_counts = {}
            for col in sorted(sdg_cols):
                # Extract SDG number from column name (handles SDG01, SDG1, SDG 1, SDG_1, etc.)
                import re
                match = re.search(r'(\d+)', col)
                if match:
                    sdg_num = int(match.group(1))
                    # Use "SDG X" format (with space, no leading zero) to match OpenAlex
                    sdg_label = f"SDG {sdg_num}"
                    # Sum the column (handle both binary 0/1 and boolean True/False)
                    col_sum = df[col].astype(int).sum() if df[col].dtype == bool else pd.to_numeric(df[col], errors='coerce').fillna(0).sum()
                    # Add to existing count if same SDG from different column names
                    sdg_counts[sdg_label] = sdg_counts.get(sdg_label, 0) + int(col_sum)
            
            if not sdg_counts:
                raise ValueError("No SDG columns found. Run identify_sdgs() first.")
            
            result = pd.DataFrame(list(sdg_counts.items()), columns=["SDG", "Count"])
            # Sort by SDG number
            result["_sort"] = result["SDG"].str.extract(r'(\d+)').astype(int)
            result = result.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
            return result
        
        elif comparison_key == "Country":
            # Handle multi-valued country column
            for col in ["Countries of Authors", "Countries", "Country"]:
                if col in df.columns:
                    sep = getattr(self.bib, 'default_separator', '; ')
                    exploded = df[col].str.split(sep).explode().str.strip()
                    exploded = exploded[exploded != ""]
                    counts = exploded.value_counts().head(50).reset_index()
                    counts.columns = ["Country", "Count"]
                    return counts
            raise ValueError("No country column found")
        
        elif comparison_key == "Document Type":
            for col in ["Document Type", "Type", "DT"]:
                if col in df.columns:
                    counts = df[col].value_counts().reset_index()
                    counts.columns = ["Document Type", "Count"]
                    return counts
            raise ValueError("No document type column found")
        
        elif comparison_key == "Open Access":
            for col in ["Open Access", "OA"]:
                if col in df.columns:
                    counts = df[col].value_counts().reset_index()
                    counts.columns = ["Open Access", "Count"]
                    # Map True/False to Open Access/Closed to match OpenAlex labels
                    oa_mapping = {
                        True: "Open Access",
                        False: "Closed",
                        "True": "Open Access",
                        "False": "Closed",
                        "true": "Open Access",
                        "false": "Closed",
                        1: "Open Access",
                        0: "Closed",
                        "1": "Open Access",
                        "0": "Closed",
                        "Yes": "Open Access",
                        "No": "Closed",
                        "yes": "Open Access",
                        "no": "Closed",
                        "OA": "Open Access",
                        "Closed": "Closed",
                        "Open Access": "Open Access",
                    }
                    counts["Open Access"] = counts["Open Access"].map(
                        lambda x: oa_mapping.get(x, str(x))
                    )
                    # Aggregate after mapping (in case True and "True" both existed)
                    counts = counts.groupby("Open Access", as_index=False)["Count"].sum()
                    return counts
            raise ValueError("No open access column found")
        
        else:
            raise ValueError(f"Unknown comparison key: {comparison_key}")
    
    def _create_plots(self, result_df: pd.DataFrame, category_col: str, cmap: str, show_comparison: bool):
        """Create the visualization plots."""
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import matplotlib.colors as mcolors
        import re
        
        # Clear previous plots
        for widget in self.diff_plot_frame.winfo_children():
            widget.destroy()
        for widget in self.comp_plot_frame.winfo_children():
            widget.destroy()
        
        # Determine if we use vertical bars (Year) or horizontal bars (everything else)
        use_horizontal = category_col != "Year"
        
        # Sort the data appropriately
        plot_df = result_df.copy()
        if category_col == "SDG":
            # Sort SDGs by number (1-17)
            plot_df["_sort"] = plot_df[category_col].apply(
                lambda x: int(re.search(r'(\d+)', str(x)).group(1)) if re.search(r'(\d+)', str(x)) else 999
            )
            plot_df = plot_df.sort_values("_sort", ascending=True).drop(columns=["_sort"])
        elif category_col == "Year":
            # Sort by year ascending
            plot_df = plot_df.sort_values(category_col, ascending=True)
        else:
            # Sort by difference (pp) descending for other categories
            plot_df = plot_df.sort_values("Difference (pp)", ascending=False)
        
        plot_df = plot_df.reset_index(drop=True)
        
        categories = plot_df[category_col].astype(str).values
        values = plot_df["Difference (pp)"].values
        
        # Determine color scale
        max_abs = max(abs(values.min()), abs(values.max())) if len(values) > 0 else 1
        if max_abs == 0:
            max_abs = 1
        
        # Create diverging norm centered at 0
        norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
        cmap_obj = plt.cm.get_cmap(cmap)
        colors = [cmap_obj(norm(v)) for v in values]
        
        # ===== Difference Plot =====
        if use_horizontal:
            # Horizontal bar chart - use standard size, will scale to window
            fig1 = Figure(figsize=(10, 8), facecolor='white')
            ax1 = fig1.add_subplot(111)
            
            y = np.arange(len(categories))
            bars = ax1.barh(y, values, color=colors, edgecolor="white", linewidth=0.5)
            
            # Vertical line at 0
            ax1.axvline(x=0, color="black", linewidth=1, linestyle="-")
            
            # Annotate bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                width = bar.get_width()
                ha = "left" if width >= 0 else "right"
                offset = 3 if width >= 0 else -3
                ax1.annotate(
                    f"{val:+.1f}",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(offset, 0),
                    textcoords="offset points",
                    ha=ha, va="center",
                    fontsize=8,
                    fontweight="bold",
                )
            
            # Labels
            ax1.set_yticks(y)
            ax1.set_yticklabels(categories)
            ax1.set_xlabel("Percentage Point Difference", fontsize=10)
            ax1.set_title(f"Relative Representation: {category_col}", fontsize=12, fontweight="bold")
            
            # Invert y-axis so first item is at top
            if category_col != "SDG":
                # For non-SDG, highest difference at top (already sorted descending)
                ax1.invert_yaxis()
            # For SDG, SDG 1 at top, SDG 17 at bottom - don't invert
            
        else:
            # Vertical bar chart (for Year)
            fig1 = Figure(figsize=(10, 6), facecolor='white')
            ax1 = fig1.add_subplot(111)
            
            x = np.arange(len(categories))
            bars = ax1.bar(x, values, color=colors, edgecolor="white", linewidth=0.5)
            
            # Horizontal line at 0
            ax1.axhline(y=0, color="black", linewidth=1, linestyle="-")
            
            # Annotate bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                va = "bottom" if height >= 0 else "top"
                offset = 3 if height >= 0 else -3
                ax1.annotate(
                    f"{val:+.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center", va=va,
                    fontsize=8,
                    fontweight="bold",
                )
            
            # Labels
            rotation = 45 if len(categories) > 10 else 0
            ax1.set_xticks(x)
            ax1.set_xticklabels(categories, rotation=rotation, ha="right" if rotation > 0 else "center")
            ax1.set_ylabel("Percentage Point Difference", fontsize=10)
            ax1.set_title(f"Relative Representation: {category_col}", fontsize=12, fontweight="bold")
        
        ax1.grid(False)
        ax1.set_facecolor("white")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = fig1.colorbar(sm, ax=ax1, shrink=0.8, pad=0.02)
        cbar.set_label("pp Difference", fontsize=9)
        
        fig1.tight_layout()
        
        # Embed in tkinter
        canvas1 = FigureCanvasTkAgg(fig1, master=self.diff_plot_frame)
        canvas1.draw()
        widget1 = canvas1.get_tk_widget()
        widget1.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Add right-click save menu and resize handling
        add_plot_context_menu(widget1, fig1)
        make_canvas_resizable(canvas1, fig1, self.diff_plot_frame)
        
        self._current_fig = fig1
        
        # ===== Comparison Plot =====
        if show_comparison:
            if use_horizontal:
                # Horizontal grouped bar chart - use standard size, will scale to window
                fig2 = Figure(figsize=(10, 8), facecolor='white')
                ax2 = fig2.add_subplot(111)
                
                y = np.arange(len(categories))
                height = 0.35
                
                bars1 = ax2.barh(y - height/2, plot_df["Observed %"], height, 
                               label="Your Dataset", color="#3498db", alpha=0.9)
                bars2 = ax2.barh(y + height/2, plot_df["Reference %"], height, 
                               label="Reference (Global)", color="#95a5a6", alpha=0.9)
                
                ax2.set_yticks(y)
                ax2.set_yticklabels(categories)
                ax2.set_xlabel("Percentage (%)", fontsize=10)
                ax2.set_title(f"Distribution Comparison: {category_col}", fontsize=12, fontweight="bold")
                ax2.legend(loc="lower right")
                
                # Invert y-axis to match difference plot
                if category_col != "SDG":
                    ax2.invert_yaxis()
                    
            else:
                # Vertical grouped bar chart (for Year)
                fig2 = Figure(figsize=(10, 6), facecolor='white')
                ax2 = fig2.add_subplot(111)
                
                x = np.arange(len(categories))
                width = 0.35
                
                bars1 = ax2.bar(x - width/2, plot_df["Observed %"], width, 
                               label="Your Dataset", color="#3498db", alpha=0.9)
                bars2 = ax2.bar(x + width/2, plot_df["Reference %"], width, 
                               label="Reference (Global)", color="#95a5a6", alpha=0.9)
                
                rotation = 45 if len(categories) > 10 else 0
                ax2.set_xticks(x)
                ax2.set_xticklabels(categories, rotation=rotation, ha="right" if rotation > 0 else "center")
                ax2.set_ylabel("Percentage (%)", fontsize=10)
                ax2.set_title(f"Distribution Comparison: {category_col}", fontsize=12, fontweight="bold")
                ax2.legend(loc="upper right")
            
            ax2.grid(False)
            ax2.set_facecolor("white")
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            
            fig2.tight_layout()
            
            canvas2 = FigureCanvasTkAgg(fig2, master=self.comp_plot_frame)
            canvas2.draw()
            widget2 = canvas2.get_tk_widget()
            widget2.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            
            # Add right-click save menu and resize handling
            add_plot_context_menu(widget2, fig2)
            make_canvas_resizable(canvas2, fig2, self.comp_plot_frame)
    
    def _on_analysis_success(self):
        """Handle successful analysis completion."""
        self.run_btn.config(state=tk.NORMAL, text="▶ Run Analysis")
        
        # Update data table
        self._update_data_table()
        
        # Update summary
        self._update_summary()
        
        # Switch to difference tab
        self.notebook.select(0)
        
        # Status update
        event_bus.emit("status_update", {
            "message": f"Reference benchmark analysis complete. {len(self._result_df)} categories compared.",
            "level": "success"
        })
    
    def _on_analysis_error(self, error_msg: str, traceback_str: str):
        """Handle analysis error."""
        self.run_btn.config(state=tk.NORMAL, text="▶ Run Analysis")
        
        messagebox.showerror(
            "Analysis Error",
            f"An error occurred:\n\n{error_msg}\n\nSee console for details."
        )
        print(f"Error details:\n{traceback_str}")
        
        event_bus.emit("status_update", {
            "message": f"Analysis failed: {error_msg}",
            "level": "error"
        })
    
    def _update_data_table(self):
        """Update the data table with results."""
        for widget in self.data_frame.winfo_children():
            widget.destroy()
        
        if self._result_df is None or self._result_df.empty:
            tk.Label(
                self.data_frame,
                text="No data available",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_secondary"],
            ).pack(expand=True)
            return
        
        # Create treeview
        columns = list(self._result_df.columns)
        tree = ttk.Treeview(self.data_frame, columns=columns, show="headings", height=20)
        
        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
            width = max(80, len(col) * 10)
            tree.column(col, width=width, minwidth=60)
        
        # Add data
        for _, row in self._result_df.iterrows():
            values = []
            for col in columns:
                val = row[col]
                if isinstance(val, float):
                    values.append(f"{val:.2f}")
                else:
                    values.append(str(val))
            tree.insert("", tk.END, values=values)
        
        # Add scrollbars
        vsb = ttk.Scrollbar(self.data_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(self.data_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        self.data_frame.grid_rowconfigure(0, weight=1)
        self.data_frame.grid_columnconfigure(0, weight=1)
    
    def _update_summary(self):
        """Update the summary tab."""
        for widget in self.summary_frame.winfo_children():
            widget.destroy()
        
        if self._result_df is None:
            return
        
        # Build summary text
        threshold = float(self.threshold_spin.get())
        over_repr = (self._result_df["Difference (pp)"] > threshold).sum()
        under_repr = (self._result_df["Difference (pp)"] < -threshold).sum()
        as_expected = len(self._result_df) - over_repr - under_repr
        
        summary_lines = [
            f"REFERENCE BENCHMARK SUMMARY",
            f"═" * 40,
            f"",
            f"Comparison: {self._comparison_key}",
            f"Reference: {'OpenAlex (Global)' if self._used_openalex else 'Custom'}",
            f"Categories compared: {len(self._result_df)}",
            f"Threshold: ±{threshold} pp",
            f"",
            f"CLASSIFICATION",
            f"─" * 40,
            f"  Over-represented (>{threshold} pp):  {over_repr}",
            f"  Under-represented (<-{threshold} pp): {under_repr}",
            f"  As expected (within ±{threshold} pp): {as_expected}",
            f"",
        ]
        
        # Top over-represented
        over_df = self._result_df[self._result_df["Difference (pp)"] > threshold].sort_values(
            "Difference (pp)", ascending=False
        ).head(5)
        if len(over_df) > 0:
            summary_lines.append("TOP OVER-REPRESENTED")
            summary_lines.append("─" * 40)
            for _, row in over_df.iterrows():
                summary_lines.append(f"  {row[self._comparison_key]}: +{row['Difference (pp)']:.1f} pp")
            summary_lines.append("")
        
        # Top under-represented
        under_df = self._result_df[self._result_df["Difference (pp)"] < -threshold].sort_values(
            "Difference (pp)", ascending=True
        ).head(5)
        if len(under_df) > 0:
            summary_lines.append("TOP UNDER-REPRESENTED")
            summary_lines.append("─" * 40)
            for _, row in under_df.iterrows():
                summary_lines.append(f"  {row[self._comparison_key]}: {row['Difference (pp)']:.1f} pp")
            summary_lines.append("")
        
        # Chi-square result
        if hasattr(self, '_chi_square_result') and self._chi_square_result:
            chi = self._chi_square_result
            summary_lines.append("STATISTICAL TEST (Chi-Square)")
            summary_lines.append("─" * 40)
            summary_lines.append(f"  χ² = {chi['chi2']:.2f}")
            summary_lines.append(f"  df = {chi['df']}")
            summary_lines.append(f"  p-value = {chi['p_value']:.4f}")
            if chi['significant']:
                summary_lines.append(f"  → Distribution differs SIGNIFICANTLY from reference (p < 0.05)")
            else:
                summary_lines.append(f"  → Distribution does NOT differ significantly from reference")
            summary_lines.append("")
        
        # Warning about OpenAlex
        if self._used_openalex:
            summary_lines.append("⚠️ NOTE")
            summary_lines.append("─" * 40)
            summary_lines.append("  Reference data from OpenAlex represents global research")
            summary_lines.append("  patterns. Coverage may differ from Scopus/Web of Science.")
            summary_lines.append("  For database-specific comparison, provide custom reference.")
        
        # Create text widget
        text_widget = tk.Text(
            self.summary_frame,
            font=("Consolas", 10),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            wrap=tk.WORD,
            padx=16, pady=16,
            relief=tk.FLAT,
        )
        text_widget.insert("1.0", "\n".join(summary_lines))
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True)
    
    def _export_plot(self):
        """Export the current plot."""
        if self._current_fig is None:
            messagebox.showinfo("No Plot", "Please run analysis first.")
            return
        
        filetypes = [
            ("PNG Image", "*.png"),
            ("SVG Vector", "*.svg"),
            ("PDF Document", "*.pdf"),
        ]
        filepath = filedialog.asksaveasfilename(
            title="Save Plot",
            filetypes=filetypes,
            defaultextension=".png",
        )
        if filepath:
            self._current_fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
            messagebox.showinfo("Saved", f"Plot saved to:\n{filepath}")
    
    def _export_data(self):
        """Export the results data."""
        if self._result_df is None:
            messagebox.showinfo("No Data", "Please run analysis first.")
            return
        
        filetypes = [
            ("Excel File", "*.xlsx"),
            ("CSV File", "*.csv"),
        ]
        filepath = filedialog.asksaveasfilename(
            title="Save Data",
            filetypes=filetypes,
            defaultextension=".xlsx",
        )
        if filepath:
            if filepath.endswith('.csv'):
                self._result_df.to_csv(filepath, index=False)
            else:
                self._result_df.to_excel(filepath, index=False)
            messagebox.showinfo("Saved", f"Data saved to:\n{filepath}")
    
    def update_bib(self, bib):
        """Update the BiblioAnalysis instance."""
        self.bib = bib
        # Reset results when data changes
        self._result_df = None
        self._current_fig = None
        self._show_placeholder()
