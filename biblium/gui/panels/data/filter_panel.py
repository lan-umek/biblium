# -*- coding: utf-8 -*-
"""
Filter Panel
============
Advanced filtering with multiple rules and conditions.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Optional, Tuple
import threading

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, CollapsibleCard, StatsCard, CardGrid
from biblium.gui.widgets.buttons import ThemedButton, ActionButton, IconButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledEntry, LabeledCheckbox, LabeledSpinbox
from biblium.gui.widgets.tables import DataTable

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class FilterRule(tk.Frame):
    """A single filter rule widget."""
    
    OPERATORS = {
        "text": ["contains", "not contains", "equals", "not equals", "starts with", "ends with", "is empty", "is not empty"],
        "numeric": ["=", "≠", ">", "≥", "<", "≤", "between", "is empty", "is not empty"],
        "date": ["=", "≠", ">", "≥", "<", "≤", "between", "is empty", "is not empty"],
    }
    
    def __init__(self, parent, columns: List[str], on_remove=None, theme: str = "light", **kwargs):
        self.theme = get_theme(theme)
        self.columns = columns
        self.on_remove = on_remove
        
        super().__init__(parent, bg=self.theme["bg_card"], **kwargs)
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create rule widgets."""
        # Column selector
        self.column_var = tk.StringVar()
        self.column_combo = ttk.Combobox(
            self, textvariable=self.column_var,
            values=self.columns, state="readonly", width=20,
        )
        self.column_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.column_combo.bind("<<ComboboxSelected>>", self._on_column_change)
        
        # Operator selector
        self.operator_var = tk.StringVar()
        self.operator_combo = ttk.Combobox(
            self, textvariable=self.operator_var,
            values=self.OPERATORS["text"], state="readonly", width=12,
        )
        self.operator_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.operator_combo.current(0)
        
        # Value entry
        self.value_var = tk.StringVar()
        self.value_entry = tk.Entry(
            self, textvariable=self.value_var,
            font=FONTS.get_font("body"), width=20,
            relief=tk.FLAT, bg=self.theme["bg_input"],
            highlightbackground=self.theme["border"], highlightthickness=1,
        )
        self.value_entry.pack(side=tk.LEFT, padx=(0, 8))
        
        # Second value (for "between")
        self.value2_var = tk.StringVar()
        self.value2_entry = tk.Entry(
            self, textvariable=self.value2_var,
            font=FONTS.get_font("body"), width=10,
            relief=tk.FLAT, bg=self.theme["bg_input"],
            highlightbackground=self.theme["border"], highlightthickness=1,
        )
        # Hidden by default
        
        # Remove button
        remove_btn = tk.Button(
            self, text="✕", font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["danger"],
            relief=tk.FLAT, cursor="hand2", width=2,
            command=self._remove,
        )
        remove_btn.pack(side=tk.LEFT)
    
    def _on_column_change(self, event=None):
        """Update operators based on column type."""
        # For now, use text operators
        # Could be enhanced to detect column type
        pass
    
    def _remove(self):
        """Remove this rule."""
        if self.on_remove:
            self.on_remove(self)
        self.destroy()
    
    def get_rule(self) -> Optional[Tuple[str, str, str, str]]:
        """Get the rule as (column, operator, value1, value2)."""
        column = self.column_var.get()
        operator = self.operator_var.get()
        value1 = self.value_var.get()
        value2 = self.value2_var.get()
        
        if not column:
            return None
        
        return (column, operator, value1, value2)


class FilterPanel(BasePanel):
    """Panel for advanced data filtering."""
    
    title = "Filter Data"
    icon = "🔍"
    description = "Create filter rules to subset your dataset"
    requires_data = True
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._filter_rules: List[FilterRule] = []
        self._filtered_df = None
        
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Quick Filters Card
        quick_card = Card(self.options_content, title="⚡ Quick Filters", theme=self.theme_name)
        quick_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Year range
        year_frame = tk.Frame(quick_card.content, bg=self.theme["bg_card"])
        year_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            year_frame, text="Year Range:", font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"], width=12, anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        self.year_from = tk.Entry(year_frame, width=8, font=FONTS.get_font("body"))
        self.year_from.pack(side=tk.LEFT, padx=(0, 4))
        
        tk.Label(year_frame, text="to", bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(side=tk.LEFT, padx=4)
        
        self.year_to = tk.Entry(year_frame, width=8, font=FONTS.get_font("body"))
        self.year_to.pack(side=tk.LEFT, padx=(4, 0))
        
        # Document type
        self.doctype_combo = LabeledCombobox(
            quick_card.content, label="Document Type:",
            values=["All", "Article", "Review", "Conference Paper", "Book Chapter"],
            default="All", theme=self.theme_name, label_width=12,
        )
        self.doctype_combo.pack(fill=tk.X, pady=4)
        
        # Language
        self.lang_combo = LabeledCombobox(
            quick_card.content, label="Language:",
            values=["All", "English", "German", "Spanish", "French", "Chinese"],
            default="All", theme=self.theme_name, label_width=12,
        )
        self.lang_combo.pack(fill=tk.X, pady=4)
        
        # Min citations
        self.min_citations = LabeledSpinbox(
            quick_card.content, label="Min Citations:",
            from_=0, to=10000, default=0,
            theme=self.theme_name, label_width=12,
        )
        self.min_citations.pack(fill=tk.X, pady=4)
        
        # Stopwords Card - for keyword filtering
        stopwords_card = Card(self.options_content, title="🚫 Stopwords (Keyword Filtering)", theme=self.theme_name)
        stopwords_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Stopwords file selection
        stopwords_file_frame = tk.Frame(stopwords_card.content, bg=self.theme["bg_card"])
        stopwords_file_frame.pack(fill=tk.X, pady=(0, 8))
        
        tk.Label(
            stopwords_file_frame, text="Stopwords File:",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"], width=12, anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        self.stopwords_file_var = tk.StringVar(value="")
        self.stopwords_file_entry = tk.Entry(
            stopwords_file_frame, textvariable=self.stopwords_file_var,
            font=FONTS.get_font("body"), width=18,
            relief=tk.FLAT, bg=self.theme["bg_input"],
            highlightbackground=self.theme["border"], highlightthickness=1,
        )
        self.stopwords_file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        
        ThemedButton(
            stopwords_file_frame, text="Browse", style="ghost", size="small",
            command=self._browse_stopwords_file, theme=self.theme_name,
        ).pack(side=tk.LEFT)
        
        ThemedButton(
            stopwords_file_frame, text="Load", style="primary", size="small",
            command=self._load_stopwords_categories, theme=self.theme_name,
        ).pack(side=tk.LEFT, padx=(4, 0))
        
        tk.Label(
            stopwords_card.content,
            text="Excel file with 'specific' sheet containing category columns",
            font=FONTS.get_font("small"), bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(anchor=tk.W)
        
        # Categories frame (will be populated when file is loaded)
        self.stopwords_categories_frame = tk.Frame(stopwords_card.content, bg=self.theme["bg_card"])
        self.stopwords_categories_frame.pack(fill=tk.X, pady=(8, 0))
        
        self.stopwords_category_vars = {}  # Store checkbox variables
        
        # Placeholder message
        self.stopwords_placeholder = tk.Label(
            self.stopwords_categories_frame,
            text="Load a stopwords file to see available categories",
            font=FONTS.get_font("small"), bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        )
        self.stopwords_placeholder.pack(anchor=tk.W)
        
        # Manual keywords to exclude
        ttk.Separator(stopwords_card.content, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        
        tk.Label(
            stopwords_card.content, text="Manual Keywords to Exclude:",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(anchor=tk.W)
        
        self.manual_stopwords_text = tk.Text(
            stopwords_card.content, height=3, width=30,
            font=FONTS.get_font("body"),
            relief=tk.FLAT, bg=self.theme["bg_input"],
            highlightbackground=self.theme["border"], highlightthickness=1,
        )
        self.manual_stopwords_text.pack(fill=tk.X, pady=4)
        
        tk.Label(
            stopwords_card.content,
            text="One keyword per line, or comma-separated",
            font=FONTS.get_font("small"), bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(anchor=tk.W)
        
        # Advanced Rules Card
        rules_card = Card(self.options_content, title="🔧 Advanced Rules", theme=self.theme_name)
        rules_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Logic selector
        logic_frame = tk.Frame(rules_card.content, bg=self.theme["bg_card"])
        logic_frame.pack(fill=tk.X, pady=(0, 8))
        
        tk.Label(
            logic_frame, text="Match:", font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT)
        
        self.logic_var = tk.StringVar(value="all")
        tk.Radiobutton(
            logic_frame, text="All rules (AND)", variable=self.logic_var, value="all",
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT, padx=(8, 4))
        tk.Radiobutton(
            logic_frame, text="Any rule (OR)", variable=self.logic_var, value="any",
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT)
        
        # Rules container
        self.rules_frame = tk.Frame(rules_card.content, bg=self.theme["bg_card"])
        self.rules_frame.pack(fill=tk.X)
        
        # Add rule button
        ThemedButton(
            rules_card.content, text="+ Add Rule", style="ghost",
            command=self._add_rule, theme=self.theme_name,
        ).pack(anchor=tk.W, pady=(8, 0))
        
        # Options Card
        options_card = Card(self.options_content, title="⚙️ Options", theme=self.theme_name)
        options_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.case_sensitive = LabeledCheckbox(
            options_card.content, label="Case sensitive matching",
            default=False, theme=self.theme_name,
        )
        self.case_sensitive.pack(fill=tk.X, pady=2)
        
        self.keep_original = LabeledCheckbox(
            options_card.content, label="Keep original dataset (create filtered copy)",
            default=True, theme=self.theme_name,
        )
        self.keep_original.pack(fill=tk.X, pady=2)
        
        # Action buttons
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Apply Filters", icon="🔍",
            command=self._apply_filters, theme=self.theme_name,
        ).pack(fill=tk.X)
        
        ThemedButton(
            btn_frame, text="Reset All Filters", style="secondary",
            command=self._reset_filters, theme=self.theme_name,
        ).pack(fill=tk.X, pady=(8, 0))
        
        ThemedButton(
            btn_frame, text="Apply to Dataset", style="success",
            command=self._apply_to_dataset, theme=self.theme_name,
        ).pack(fill=tk.X, pady=(8, 0))
    
    def _create_results(self):
        """Create the results panel."""
        super()._create_results()
        
        # Add comparison stats at top
        self.comparison_frame = tk.Frame(self.results_content, bg=self.theme["bg_card"])
        self.comparison_frame.pack(fill=tk.X, pady=(0, 12))
        
        self._show_initial_message("Configure filters and click 'Apply Filters' to preview results")
    
    def _add_rule(self):
        """Add a new filter rule."""
        columns = []
        if self.bib and hasattr(self.bib, 'df'):
            columns = list(self.bib.df.columns)
        
        rule = FilterRule(
            self.rules_frame, columns=columns,
            on_remove=self._remove_rule, theme=self.theme_name,
        )
        rule.pack(fill=tk.X, pady=4)
        self._filter_rules.append(rule)
    
    def _remove_rule(self, rule: FilterRule):
        """Remove a filter rule."""
        if rule in self._filter_rules:
            self._filter_rules.remove(rule)
    
    def _browse_stopwords_file(self):
        """Browse for stopwords Excel file."""
        from tkinter import filedialog
        
        filetypes = [
            ("Excel Files", "*.xlsx *.xls"),
            ("All Files", "*.*"),
        ]
        
        path = filedialog.askopenfilename(
            title="Select Stopwords File",
            filetypes=filetypes,
        )
        
        if path:
            self.stopwords_file_var.set(path)
            # Auto-load categories
            self._load_stopwords_categories()
    
    def _load_stopwords_categories(self):
        """Load stopwords categories from Excel file."""
        filepath = self.stopwords_file_var.get()
        
        if not filepath:
            messagebox.showwarning("No File", "Please select a stopwords file first.")
            return
        
        try:
            # Read the 'specific' sheet to get category columns
            import pandas as pd
            
            # Try 'specific' sheet first, then first sheet
            try:
                df = pd.read_excel(filepath, sheet_name='specific')
            except:
                try:
                    df = pd.read_excel(filepath, sheet_name=0)
                except Exception as e:
                    messagebox.showerror("Error", f"Could not read Excel file: {e}")
                    return
            
            # Get column names as categories
            categories = [col for col in df.columns if not col.startswith('Unnamed')]
            
            if not categories:
                messagebox.showwarning("No Categories", "No category columns found in the stopwords file.")
                return
            
            # Store the stopwords data for later use
            self._stopwords_data = {}
            for cat in categories:
                # Get non-null values from each column
                values = df[cat].dropna().astype(str).str.strip().tolist()
                values = [v for v in values if v and v.lower() != 'nan']
                self._stopwords_data[cat] = values
            
            # Clear existing checkboxes
            for widget in self.stopwords_categories_frame.winfo_children():
                widget.destroy()
            
            self.stopwords_category_vars = {}
            
            # Create header
            tk.Label(
                self.stopwords_categories_frame,
                text=f"Select categories to exclude ({len(categories)} found):",
                font=FONTS.get_font("body", bold=True),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            ).pack(anchor=tk.W, pady=(0, 4))
            
            # Create checkboxes for each category
            for cat in categories:
                count = len(self._stopwords_data.get(cat, []))
                var = tk.BooleanVar(value=False)
                self.stopwords_category_vars[cat] = var
                
                cb = tk.Checkbutton(
                    self.stopwords_categories_frame,
                    text=f"{cat} ({count} terms)",
                    variable=var,
                    font=FONTS.get_font("body"),
                    bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                    activebackground=self.theme["bg_card"],
                    selectcolor=self.theme["bg_input"],
                )
                cb.pack(anchor=tk.W)
            
            # Add "Select All" / "Deselect All" buttons
            btn_frame = tk.Frame(self.stopwords_categories_frame, bg=self.theme["bg_card"])
            btn_frame.pack(anchor=tk.W, pady=(4, 0))
            
            ThemedButton(
                btn_frame, text="Select All", style="ghost", size="small",
                command=self._select_all_stopwords, theme=self.theme_name,
            ).pack(side=tk.LEFT, padx=(0, 4))
            
            ThemedButton(
                btn_frame, text="Deselect All", style="ghost", size="small",
                command=self._deselect_all_stopwords, theme=self.theme_name,
            ).pack(side=tk.LEFT)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load stopwords file: {e}")
    
    def _select_all_stopwords(self):
        """Select all stopwords categories."""
        for var in self.stopwords_category_vars.values():
            var.set(True)
    
    def _deselect_all_stopwords(self):
        """Deselect all stopwords categories."""
        for var in self.stopwords_category_vars.values():
            var.set(False)
    
    def _get_selected_stopwords(self) -> List[str]:
        """Get list of all stopwords from selected categories and manual input."""
        stopwords = []
        
        # From selected categories
        if hasattr(self, '_stopwords_data'):
            for cat, var in self.stopwords_category_vars.items():
                if var.get():
                    stopwords.extend(self._stopwords_data.get(cat, []))
        
        # From manual input
        manual_text = self.manual_stopwords_text.get("1.0", tk.END).strip()
        if manual_text:
            # Split by newlines and commas
            for line in manual_text.split('\n'):
                for term in line.split(','):
                    term = term.strip()
                    if term:
                        stopwords.append(term)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_stopwords = []
        for sw in stopwords:
            sw_lower = sw.lower()
            if sw_lower not in seen:
                seen.add(sw_lower)
                unique_stopwords.append(sw)
        
        return unique_stopwords
    
    def _remove_stopwords_from_keywords(self, keywords_str, stopwords: List[str]) -> str:
        """Remove stopwords from a semicolon-separated keyword string."""
        if pd.isna(keywords_str) or not keywords_str:
            return keywords_str
        
        # Convert stopwords to lowercase set for faster lookup
        stopwords_lower = set(sw.lower() for sw in stopwords)
        
        # Split keywords (try common separators)
        if '; ' in str(keywords_str):
            sep = '; '
        elif ';' in str(keywords_str):
            sep = ';'
        elif '|' in str(keywords_str):
            sep = '|'
        else:
            sep = ';'
        
        keywords = str(keywords_str).split(sep)
        
        # Filter out stopwords
        filtered = []
        for kw in keywords:
            kw_stripped = kw.strip()
            if kw_stripped.lower() not in stopwords_lower:
                filtered.append(kw_stripped)
        
        return sep.join(filtered) if filtered else ""
    
    def _apply_filters(self):
        """Apply filters and show preview."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self._show_loading("Applying filters...")
        
        def do_filter():
            try:
                df = self.bib.df.copy()
                original_count = len(df)
                
                # Apply quick filters
                # Year range
                year_col = self.bib.mapping.get("Year", "Year")
                if year_col in df.columns:
                    year_from = self.year_from.get()
                    year_to = self.year_to.get()
                    if year_from:
                        df = df[df[year_col] >= int(year_from)]
                    if year_to:
                        df = df[df[year_col] <= int(year_to)]
                
                # Document type
                doctype = self.doctype_combo.get()
                if doctype != "All":
                    doctype_col = self.bib.mapping.get("Document_Type", "Document Type")
                    if doctype_col in df.columns:
                        df = df[df[doctype_col].str.contains(doctype, case=False, na=False)]
                
                # Language
                lang = self.lang_combo.get()
                if lang != "All":
                    lang_col = "Language"
                    if lang_col in df.columns:
                        df = df[df[lang_col].str.contains(lang, case=False, na=False)]
                
                # Min citations
                min_cit = self.min_citations.get()
                if min_cit > 0:
                    cit_col = self.bib.mapping.get("Cited_by", "Cited by")
                    if cit_col in df.columns:
                        df = df[pd.to_numeric(df[cit_col], errors='coerce').fillna(0) >= min_cit]
                
                # Apply stopwords filtering to keyword columns
                stopwords = self._get_selected_stopwords()
                if stopwords:
                    # Get keyword columns
                    kw_cols = [
                        self.bib.mapping.get("Author_Keywords", "Author Keywords"),
                        self.bib.mapping.get("Index_Keywords", "Index Keywords"),
                    ]
                    
                    for kw_col in kw_cols:
                        if kw_col in df.columns:
                            df[kw_col] = df[kw_col].apply(
                                lambda x: self._remove_stopwords_from_keywords(x, stopwords)
                            )
                
                # Apply advanced rules
                case_sensitive = self.case_sensitive.get()
                rules = [r.get_rule() for r in self._filter_rules if r.get_rule()]
                
                if rules:
                    masks = []
                    for col, op, val1, val2 in rules:
                        if col not in df.columns:
                            continue
                        
                        series = df[col].astype(str)
                        if not case_sensitive:
                            series = series.str.lower()
                            val1 = val1.lower() if val1 else val1
                        
                        if op == "contains":
                            mask = series.str.contains(val1, na=False)
                        elif op == "not contains":
                            mask = ~series.str.contains(val1, na=False)
                        elif op == "equals" or op == "=":
                            mask = series == val1
                        elif op == "not equals" or op == "≠":
                            mask = series != val1
                        elif op == "starts with":
                            mask = series.str.startswith(val1, na=False)
                        elif op == "ends with":
                            mask = series.str.endswith(val1, na=False)
                        elif op == "is empty":
                            mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
                        elif op == "is not empty":
                            mask = ~(df[col].isna() | (df[col].astype(str).str.strip() == ""))
                        elif op in [">", "≥", "<", "≤"]:
                            numeric = pd.to_numeric(df[col], errors='coerce')
                            val_num = float(val1) if val1 else 0
                            if op == ">":
                                mask = numeric > val_num
                            elif op == "≥":
                                mask = numeric >= val_num
                            elif op == "<":
                                mask = numeric < val_num
                            else:
                                mask = numeric <= val_num
                        else:
                            mask = pd.Series([True] * len(df))
                        
                        masks.append(mask)
                    
                    if masks:
                        if self.logic_var.get() == "all":
                            combined = masks[0]
                            for m in masks[1:]:
                                combined = combined & m
                        else:
                            combined = masks[0]
                            for m in masks[1:]:
                                combined = combined | m
                        df = df[combined]
                
                filtered_count = len(df)
                
                self.after(0, lambda: self._on_filter_success(df, original_count, filtered_count))
            except Exception as e:
                import traceback
                self.after(0, lambda: self._on_filter_error(str(e)))
        
        threading.Thread(target=do_filter, daemon=True).start()
    
    def _on_filter_success(self, df, original_count: int, filtered_count: int):
        """Handle successful filtering."""
        self._filtered_df = df
        
        # Clear results
        for widget in self.results_content.winfo_children():
            widget.destroy()
        
        # Comparison stats
        stats_frame = tk.Frame(self.results_content, bg=self.theme["bg_card"])
        stats_frame.pack(fill=tk.X, pady=(0, 12))
        
        grid = CardGrid(stats_frame, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X)
        
        removed = original_count - filtered_count
        pct = (filtered_count / original_count * 100) if original_count > 0 else 0
        
        grid.add_card(StatsCard(grid, "Original", f"{original_count:,}", "📊", self.theme_name))
        grid.add_card(StatsCard(grid, "Filtered", f"{filtered_count:,}", "✓", self.theme_name, accent=True))
        grid.add_card(StatsCard(grid, "Removed", f"{removed:,}", "✕", self.theme_name))
        grid.add_card(StatsCard(grid, "Retained", f"{pct:.1f}%", "📈", self.theme_name))
        
        # Table preview
        tk.Label(
            self.results_content, text="Filtered Data Preview",
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(anchor=tk.W, pady=(8, 4))
        
        table = DataTable(self.results_content, theme=self.theme_name, max_rows=200)
        table.pack(fill=tk.BOTH, expand=True)
        table.set_data(df)
    
    def _on_filter_error(self, error: str):
        """Handle filter error."""
        self._show_error(f"Filter error: {error}")
    
    def _reset_filters(self):
        """Reset all filters."""
        self.year_from.delete(0, tk.END)
        self.year_to.delete(0, tk.END)
        self.doctype_combo.set("All")
        self.lang_combo.set("All")
        self.min_citations.set(0)
        
        # Reset stopwords
        self._deselect_all_stopwords()
        self.manual_stopwords_text.delete("1.0", tk.END)
        
        for rule in self._filter_rules[:]:
            rule.destroy()
        self._filter_rules.clear()
        
        self._filtered_df = None
        self._show_initial_message("Filters reset. Configure new filters.")
    
    def _apply_to_dataset(self):
        """Apply the filter permanently to the dataset."""
        if self._filtered_df is None:
            messagebox.showwarning("No Filter", "Please apply filters first to preview results.")
            return
        
        if not messagebox.askyesno(
            "Apply Filter",
            f"This will reduce the dataset from {len(self.bib.df):,} to {len(self._filtered_df):,} documents.\n\nContinue?"
        ):
            return
        
        # Apply filter
        if self.keep_original.get():
            # Store original
            if not hasattr(self.bib, '_original_df'):
                self.bib._original_df = self.bib.df.copy()
        
        self.bib.df = self._filtered_df.copy()
        self.bib.n = len(self.bib.df)
        
        # Emit event
        event_bus.emit(EventBus.DATASET_FILTERED, {
            "n_documents": len(self._filtered_df),
            "original_count": len(self.bib._original_df) if hasattr(self.bib, '_original_df') else None,
        })
        
        messagebox.showinfo("Success", f"Dataset filtered to {len(self._filtered_df):,} documents.")
