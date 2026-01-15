# -*- coding: utf-8 -*-
"""
Laws Panel
==========
Bibliometric laws analysis: Lotka, Bradford, Zipf, Price, Pareto.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from typing import Dict, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, StatsCard, CardGrid
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox, LabeledSpinbox
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.plots import PlotFrame

try:
    import pandas as pd
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from scipy import stats as scipy_stats
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class LawsPanel(BasePanel):
    """Panel for bibliometric laws analysis."""
    
    title = "Bibliometric Laws"
    icon = "📚"
    description = "Analyze Lotka, Bradford, Zipf, and other bibliometric laws"
    requires_data = True
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._analyze_law  # Set primary action for toolbar Run button
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Law Selection
        law_card = Card(self.options_content, title="📚 Select Law", theme=self.theme_name)
        law_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.law_combo = LabeledCombobox(
            law_card.content, label="Law:",
            values=[
                "Lotka's Law (Author Productivity)",
                "Bradford's Law (Journal Scatter)",
                "Zipf's Law (Word Frequency)",
                "Price's Law (Elite Productivity)",
                "Pareto Principle (80/20 Rule)",
            ],
            default="Lotka's Law (Author Productivity)",
            theme=self.theme_name, label_width=12,
        )
        self.law_combo.pack(fill=tk.X, pady=4)
        
        # Options
        options_card = Card(self.options_content, title="⚙️ Options", theme=self.theme_name)
        options_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.entity_combo = LabeledCombobox(
            options_card.content, label="Entity:",
            values=["Authors", "Sources", "Keywords"],
            default="Authors", theme=self.theme_name, label_width=12,
        )
        self.entity_combo.pack(fill=tk.X, pady=4)
        
        self.show_fit_cb = LabeledCheckbox(
            options_card.content, label="Show theoretical fit",
            default=True, theme=self.theme_name,
        )
        self.show_fit_cb.pack(fill=tk.X, pady=2)
        
        self.log_scale_cb = LabeledCheckbox(
            options_card.content, label="Use log-log scale",
            default=True, theme=self.theme_name,
        )
        self.log_scale_cb.pack(fill=tk.X, pady=2)
        
        # Run Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Analyze Law", icon="📊",
            command=self._analyze_law, theme=self.theme_name,
        ).pack(fill=tk.X)
    
    def _create_results(self):
        """Create results panel."""
        super()._create_results()
        
        # Create notebook with Info tab
        self.results_notebook = ttk.Notebook(self.results_content)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results tab (will be populated by analysis)
        self.results_tab = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.results_tab, text="📊 Results")
        
        # Info tab
        self.info_tab = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.info_tab, text="ℹ️ Info")
        self._create_info_content(self.info_tab)
        
        # Show placeholder in results tab
        self._show_placeholder_in_tab()
    
    def _show_placeholder_in_tab(self):
        """Show placeholder in results tab."""
        # Check if results_tab still exists
        if not hasattr(self, 'results_tab'):
            return
        try:
            if not self.results_tab.winfo_exists():
                return
        except tk.TclError:
            return
            
        for widget in self.results_tab.winfo_children():
            widget.destroy()
        tk.Label(
            self.results_tab,
            text="📚 Bibliometric Laws\n\nSelect a law and click 'Analyze Law'\nto test bibliometric regularities.",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)
    
    def _show_loading(self, message: str = "Analyzing..."):
        """Show loading state inside the results tab, preserving notebook structure."""
        self._stop_active_spinners()
        
        # Only clear the results_tab content, not the whole notebook
        if self._widget_exists(self.results_tab):
            try:
                for widget in self.results_tab.winfo_children():
                    try:
                        widget.destroy()
                    except:
                        pass
            except tk.TclError:
                pass
            
            # Create loading indicator inside results_tab
            try:
                from biblium.gui.widgets.progress import LoadingSpinner
                
                loading_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
                loading_frame.pack(expand=True)
                
                spinner = LoadingSpinner(loading_frame, size=32, theme=self.theme_name)
                spinner.pack()
                spinner.start()
                
                # Track active spinner for cleanup
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
                
                # Switch to results tab to show loading
                self.results_notebook.select(0)
            except tk.TclError:
                pass
        else:
            # Fallback to base behavior if notebook structure is missing
            super()._show_loading(message)
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
BIBLIOMETRIC LAWS
═════════════════

Test classical bibliometric regularities on your data.

LOTKA'S LAW (1926)
──────────────────
Author productivity distribution:
"The number of authors making n contributions 
is about 1/n² of those making one."

Formula: y = C / x^α
• x = number of publications
• y = number of authors
• α ≈ 2 (Lotka's exponent)
• C = constant

Output:
• Observed vs expected frequencies
• Alpha exponent estimation
• Kolmogorov-Smirnov goodness-of-fit
• Chi-square test

BRADFORD'S LAW (1934)
─────────────────────
Journal scatter:
"Journals can be divided into zones of equal 
yield with counts 1 : n : n²"

• Zone 1 (Core): Few journals, many papers
• Zone 2: More journals, same papers
• Zone 3 (Periphery): Many journals, few papers

Output:
• Zone boundaries
• Core journal list
• Bradford multiplier (n)
• Cumulative distribution plot

ZIPF'S LAW (1935)
─────────────────
Word frequency:
"Frequency × rank ≈ constant"

Formula: f(r) = C / r^α
• r = word rank
• f = frequency
• α ≈ 1 (Zipf's exponent)

Output:
• Rank-frequency distribution
• Alpha estimation
• Log-log plot
• Goodness-of-fit

PRICE'S LAW
───────────
"Half of publications come from √N authors"
• N = total number of authors
• Top √N authors = 50% of output

PARETO PRINCIPLE
────────────────
"80% of effects from 20% of causes"
• 20% of authors → 80% of papers
• 20% of journals → 80% of articles
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
    
    def _analyze_law(self):
        """Run law analysis using biblium methods."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self._show_loading("Analyzing...")
        
        law = self.law_combo.get()
        entity = self.entity_combo.get()
        
        def do_analysis():
            try:
                result = {"law": law, "entity": entity}
                
                # Helper to find column in dataframe
                def find_column(candidates):
                    """Find first existing column from candidates list."""
                    for col in candidates:
                        if col in self.bib.df.columns:
                            return col
                    return None
                
                # Import utilsbib for direct computation
                from biblium import utilsbib
                
                if "Lotka" in law:
                    # Find author column - WoS uses "Authors or Inventors"
                    author_col = find_column([
                        "Authors", "Authors or Inventors", "Author full names",
                        "AU", "author", "authors"
                    ])
                    
                    if author_col is None:
                        raise ValueError("No author column found in dataset")
                    
                    # Compute Lotka distribution directly (no plotting)
                    lotka_df = utilsbib.compute_lotka_distribution(
                        self.bib.df,
                        author_col=author_col,
                        separator=self.bib.default_separator,
                    )
                    lotka_stats = utilsbib.evaluate_lotka_fit(lotka_df)
                    
                    # Store in bib object for consistency
                    self.bib.lotka_df = lotka_df
                    self.bib.lotka_stats_df = lotka_stats
                    
                    if lotka_df is not None and len(lotka_df) > 0:
                        # Rename columns for display
                        result["data"] = lotka_df.rename(columns={
                            'n_pubs': 'Publications',
                            'n_authors': 'Authors',
                            'expected_n_authors': 'Expected'
                        })
                        
                        # Build interpretation from stats
                        interpretation = "Lotka's Law: Authors with n publications ≈ C/n²"
                        if lotka_stats is not None and len(lotka_stats) > 0:
                            stats_dict = dict(zip(lotka_stats['Measure'], lotka_stats['Value']))
                            r2 = stats_dict.get('R2', None)
                            if r2 is not None:
                                interpretation += f"\nR² = {r2:.3f}"
                            ks_p = stats_dict.get('KS_pvalue', None)
                            if ks_p is not None:
                                interpretation += f", KS p-value = {ks_p:.4f}"
                        
                        result["interpretation"] = interpretation
                        result["stats_df"] = lotka_stats
                        result["from_biblium"] = True
                    else:
                        result = self._fallback_lotka(entity)
                
                elif "Bradford" in law:
                    # Find source column
                    source_col = find_column([
                        "Source title", "Source Title", "Journal", "Publication Name",
                        "SO", "source", "journal"
                    ])
                    
                    if source_col is None:
                        raise ValueError("No source/journal column found in dataset")
                    
                    # Compute Bradford distribution directly (no plotting)
                    bradford_df = utilsbib.compute_bradford_distribution(
                        self.bib.df,
                        source_col=source_col,
                    )
                    bradford_stats = utilsbib.evaluate_bradford_fit(bradford_df)
                    
                    # Store in bib object for consistency
                    self.bib.bradford_df = bradford_df
                    self.bib.bradford_stats_df = bradford_stats
                    
                    if bradford_df is not None and len(bradford_df) > 0:
                        # Add rank column and rename for display
                        display_df = bradford_df.copy()
                        display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
                        display_df = display_df.rename(columns={
                            'Document_Count': 'Articles',
                            'Cumulative_Documents': 'Cumulative',
                            'Cumulative_Percentage': 'Cum %'
                        })
                        result["data"] = display_df.head(100)
                        
                        interpretation = "Bradford's Law divides sources into zones with equal articles but increasing journals."
                        if bradford_stats is not None and len(bradford_stats) > 0:
                            mean_dev = bradford_stats['Mean Deviation'].iloc[0] if 'Mean Deviation' in bradford_stats.columns else None
                            if mean_dev is not None:
                                interpretation += f"\nMean deviation from expected: {mean_dev:.3f}"
                        
                        result["interpretation"] = interpretation
                        result["stats_df"] = bradford_stats
                        result["from_biblium"] = True
                    else:
                        result = self._fallback_bradford(entity)
                    
                elif "Zipf" in law:
                    # Use biblium's zipf_law method or fallback
                    # First check if abstract column exists
                    abstract_col = find_column([
                        "Processed Abstract", "Abstract", "AB", "abstract",
                        "Description", "Summary"
                    ])
                    
                    # For keywords, use keywords instead
                    if entity == "Keywords":
                        kw_col = find_column([
                            "Author Keywords", "Keywords", "DE", "ID",
                            "Index Keywords", "Processed Author Keywords"
                        ])
                        if kw_col:
                            # Use fallback for keywords - simpler approach
                            result = self._fallback_zipf_keywords()
                        else:
                            raise ValueError("No keyword column found in dataset")
                    elif abstract_col:
                        # Try using biblium's zipf_law
                        try:
                            self.bib.zipf_law(items='words from abstract', show=False)
                            zipf_df = getattr(self.bib, 'zipf_df', None)
                            zipf_stats = getattr(self.bib, 'zipf_stats', None)
                            
                            if zipf_df is not None and len(zipf_df) > 0:
                                display_df = zipf_df.head(100).copy()
                                display_df = display_df.rename(columns={
                                    'Word': 'Item',
                                    'Frequency': 'Frequency',
                                    'Rank': 'Rank'
                                })
                                if 'Rank' in display_df.columns:
                                    cols = ['Rank'] + [c for c in display_df.columns if c != 'Rank']
                                    display_df = display_df[cols]
                                
                                result["data"] = display_df
                                interpretation = "Zipf's Law: Word frequency is inversely proportional to rank."
                                if zipf_stats is not None:
                                    r2 = zipf_stats.get('R2', None)
                                    if r2 is not None:
                                        interpretation += f"\nR² = {r2:.3f}"
                                result["interpretation"] = interpretation
                                result["from_biblium"] = True
                            else:
                                result = self._fallback_zipf_abstract(abstract_col)
                        except Exception:
                            result = self._fallback_zipf_abstract(abstract_col)
                    else:
                        # No abstract - try title
                        title_col = find_column(["Title", "TI", "title", "Article Title"])
                        if title_col:
                            result = self._fallback_zipf_title(title_col)
                        else:
                            raise ValueError("No text column (Abstract/Title) found for Zipf analysis")
                    
                elif "Price" in law:
                    # Price's Law - use biblium counts if available
                    result = self._analyze_price_law(entity)
                    
                elif "Pareto" in law:
                    # Pareto Principle - use biblium counts if available
                    result = self._analyze_pareto(entity)
                
                self.after(0, lambda r=result: self._on_analysis_success(r))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda msg=str(e): self._on_analysis_error(msg))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _fallback_lotka(self, entity):
        """Fallback for Lotka's law when biblium method fails."""
        df = self.bib.df
        
        # Find appropriate column
        if entity == "Authors":
            col = None
            for c in ["Authors", "Authors or Inventors", "Author full names", "AU"]:
                if c in df.columns:
                    col = c
                    break
        elif entity == "Sources":
            col = None
            for c in ["Source title", "Source Title", "Journal", "Publication Name", "SO"]:
                if c in df.columns:
                    col = c
                    break
        else:
            col = None
            for c in ["Author Keywords", "Keywords", "DE", "ID"]:
                if c in df.columns:
                    col = c
                    break
        
        if col is None:
            raise ValueError(f"No {entity} column found")
        
        if entity in ["Authors", "Keywords"]:
            all_items = df[col].dropna().str.split(";").explode().str.strip()
        else:
            all_items = df[col].dropna()
        
        item_counts = all_items.value_counts()
        productivity = item_counts.value_counts().sort_index()
        x = productivity.index.values.astype(float)
        y = productivity.values.astype(float)
        
        c = y[0] if len(y) > 0 else 1
        n = 2.0
        expected = c / (x ** n)
        
        return {
            "law": "Lotka's Law (Author Productivity)",
            "entity": entity,
            "data": pd.DataFrame({
                "Publications": x.astype(int),
                "Authors": y.astype(int),
                "Expected": expected.round(1),
            }),
            "interpretation": f"Lotka's Law: Authors with n publications ≈ C/n². Exponent n ≈ {n:.2f}"
        }
    
    def _fallback_bradford(self, entity):
        """Fallback for Bradford's law when biblium method fails."""
        df = self.bib.df
        
        # Find source column
        col = None
        for c in ["Source title", "Source Title", "Journal", "Publication Name", "SO"]:
            if c in df.columns:
                col = c
                break
        
        if col is None:
            raise ValueError("No source/journal column found")
        
        item_counts = df[col].dropna().value_counts()
        sorted_counts = item_counts.sort_values(ascending=False)
        cumsum = sorted_counts.cumsum()
        
        return {
            "law": "Bradford's Law (Journal Scatter)",
            "entity": entity,
            "data": pd.DataFrame({
                "Rank": range(1, min(101, len(sorted_counts) + 1)),
                "Source": sorted_counts.index[:100],
                "Articles": sorted_counts.values[:100],
                "Cumulative": cumsum.values[:100],
            }),
            "interpretation": "Bradford's Law divides sources into zones with equal articles but increasing journals."
        }
    
    def _fallback_zipf(self, entity):
        """Fallback for Zipf's law when biblium method fails."""
        df = self.bib.df
        
        if entity == "Keywords":
            col = self.bib.mapping.get("Author_Keywords", "Author Keywords")
        else:
            col = self.bib.mapping.get("Authors", "Authors")
        
        if col not in df.columns:
            raise ValueError(f"Column {col} not found")
        
        if entity in ["Authors", "Keywords"]:
            all_items = df[col].dropna().str.split(";").explode().str.strip()
        else:
            all_items = df[col].dropna()
        
        item_counts = all_items.value_counts()
        sorted_counts = item_counts.sort_values(ascending=False)
        ranks = np.arange(1, len(sorted_counts) + 1)
        freqs = sorted_counts.values
        
        c = freqs[0] if len(freqs) > 0 else 1
        expected = c / ranks
        
        return {
            "law": "Zipf's Law (Word Frequency)",
            "entity": entity,
            "data": pd.DataFrame({
                "Rank": ranks[:100],
                "Item": sorted_counts.index[:100],
                "Frequency": freqs[:100],
                "Expected": expected[:100].round(1),
            }),
            "interpretation": "Zipf's Law: Word frequency is inversely proportional to rank."
        }
    
    def _fallback_zipf_keywords(self):
        """Zipf analysis using keywords."""
        df = self.bib.df
        
        # Find keyword column
        kw_col = None
        for col in ["Author Keywords", "Keywords", "DE", "ID", "Index Keywords"]:
            if col in df.columns:
                kw_col = col
                break
        
        if kw_col is None:
            raise ValueError("No keyword column found")
        
        # Split and count keywords
        all_kws = df[kw_col].dropna().str.split(";").explode().str.strip().str.lower()
        all_kws = all_kws[all_kws != '']
        kw_counts = all_kws.value_counts()
        
        ranks = np.arange(1, len(kw_counts) + 1)
        freqs = kw_counts.values
        c = freqs[0] if len(freqs) > 0 else 1
        expected = c / ranks
        
        return {
            "law": "Zipf's Law (Word Frequency)",
            "entity": "Keywords",
            "data": pd.DataFrame({
                "Rank": ranks[:100],
                "Item": kw_counts.index[:100],
                "Frequency": freqs[:100],
                "Expected": expected[:100].round(1),
            }),
            "interpretation": "Zipf's Law: Keyword frequency is inversely proportional to rank."
        }
    
    def _fallback_zipf_abstract(self, abstract_col):
        """Zipf analysis using abstract text."""
        import re
        df = self.bib.df
        
        # Extract words from abstracts
        all_text = df[abstract_col].dropna().str.cat(sep=' ')
        # Simple tokenization
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
        
        # Remove common stopwords
        stopwords = {'the', 'and', 'for', 'with', 'that', 'this', 'from', 'are', 'was', 
                     'were', 'been', 'have', 'has', 'had', 'not', 'but', 'can', 'which',
                     'their', 'they', 'these', 'there', 'than', 'also', 'into', 'such',
                     'more', 'other', 'some', 'what', 'when', 'where', 'who', 'how'}
        words = [w for w in words if w not in stopwords]
        
        word_counts = pd.Series(words).value_counts()
        
        ranks = np.arange(1, len(word_counts) + 1)
        freqs = word_counts.values
        c = freqs[0] if len(freqs) > 0 else 1
        expected = c / ranks
        
        return {
            "law": "Zipf's Law (Word Frequency)",
            "entity": "Words from Abstract",
            "data": pd.DataFrame({
                "Rank": ranks[:100],
                "Item": word_counts.index[:100],
                "Frequency": freqs[:100],
                "Expected": expected[:100].round(1),
            }),
            "interpretation": "Zipf's Law: Word frequency from abstracts is inversely proportional to rank."
        }
    
    def _fallback_zipf_title(self, title_col):
        """Zipf analysis using title text."""
        import re
        df = self.bib.df
        
        # Extract words from titles
        all_text = df[title_col].dropna().str.cat(sep=' ')
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
        
        # Remove common stopwords
        stopwords = {'the', 'and', 'for', 'with', 'that', 'this', 'from', 'are', 'was', 
                     'were', 'been', 'have', 'has', 'had', 'not', 'but', 'can', 'which',
                     'their', 'they', 'these', 'there', 'than', 'also', 'into', 'such',
                     'more', 'other', 'some', 'what', 'when', 'where', 'who', 'how'}
        words = [w for w in words if w not in stopwords]
        
        word_counts = pd.Series(words).value_counts()
        
        ranks = np.arange(1, len(word_counts) + 1)
        freqs = word_counts.values
        c = freqs[0] if len(freqs) > 0 else 1
        expected = c / ranks
        
        return {
            "law": "Zipf's Law (Word Frequency)",
            "entity": "Words from Title",
            "data": pd.DataFrame({
                "Rank": ranks[:100],
                "Item": word_counts.index[:100],
                "Frequency": freqs[:100],
                "Expected": expected[:100].round(1),
            }),
            "interpretation": "Zipf's Law: Word frequency from titles is inversely proportional to rank."
        }
    
    def _analyze_price_law(self, entity):
        """Analyze Price's Law using biblium counts."""
        # Try to get counts from biblium
        if entity == "Authors":
            count_attr = 'authors_counts_df'
            if not hasattr(self.bib, count_attr) or self.bib.authors_counts_df is None:
                self.bib.count_authors()
            counts_df = getattr(self.bib, count_attr, None)
            name_col = 'Author'
        elif entity == "Sources":
            count_attr = 'sources_counts_df'
            if not hasattr(self.bib, count_attr) or self.bib.sources_counts_df is None:
                self.bib.count_sources()
            counts_df = getattr(self.bib, count_attr, None)
            name_col = 'Source'
        else:
            count_attr = 'author_keywords_counts_df'
            if not hasattr(self.bib, count_attr) or self.bib.author_keywords_counts_df is None:
                self.bib.count_author_keywords()
            counts_df = getattr(self.bib, count_attr, None)
            name_col = 'Keyword'
        
        if counts_df is not None and len(counts_df) > 0:
            # Find the document count column
            doc_col = None
            for col in counts_df.columns:
                if 'number' in col.lower() and 'document' in col.lower():
                    doc_col = col
                    break
            
            if doc_col:
                total_items = len(counts_df)
                total_pubs = counts_df[doc_col].sum()
                elite_n = int(np.sqrt(total_items))
                
                elite_pubs = counts_df.head(elite_n)[doc_col].sum()
                elite_pct = elite_pubs / total_pubs * 100
                
                return {
                    "law": "Price's Law (Elite Productivity)",
                    "entity": entity,
                    "data": pd.DataFrame({
                        "Metric": [f"Total {entity}", f"Elite (√n)", "Elite Publications", "Elite %", "Expected (50%)"],
                        "Value": [total_items, elite_n, int(elite_pubs), f"{elite_pct:.1f}%", "50%"],
                    }),
                    "interpretation": f"Price's Law: √n elite {entity.lower()} ({elite_n}) produce {elite_pct:.1f}% of output (expected ~50%).",
                    "from_biblium": True
                }
        
        # Fallback to manual calculation
        return self._fallback_price(entity)
    
    def _fallback_price(self, entity):
        """Fallback for Price's Law."""
        df = self.bib.df
        
        # Find appropriate column
        if entity == "Authors":
            col = None
            for c in ["Authors", "Authors or Inventors", "Author full names", "AU"]:
                if c in df.columns:
                    col = c
                    break
        elif entity == "Sources":
            col = None
            for c in ["Source title", "Source Title", "Journal", "Publication Name", "SO"]:
                if c in df.columns:
                    col = c
                    break
        else:
            col = None
            for c in ["Author Keywords", "Keywords", "DE", "ID"]:
                if c in df.columns:
                    col = c
                    break
        
        if col is None:
            raise ValueError(f"No {entity} column found")
        
        if entity in ["Authors", "Keywords"]:
            all_items = df[col].dropna().str.split(";").explode().str.strip()
        else:
            all_items = df[col].dropna()
        
        item_counts = all_items.value_counts()
        total_items = len(item_counts)
        total_pubs = item_counts.sum()
        elite_n = int(np.sqrt(total_items))
        
        sorted_counts = item_counts.sort_values(ascending=False)
        elite_pubs = sorted_counts.head(elite_n).sum()
        elite_pct = elite_pubs / total_pubs * 100
        
        return {
            "law": "Price's Law (Elite Productivity)",
            "entity": entity,
            "data": pd.DataFrame({
                "Metric": [f"Total {entity}", f"Elite (√n)", "Elite Publications", "Elite %", "Expected (50%)"],
                "Value": [total_items, elite_n, int(elite_pubs), f"{elite_pct:.1f}%", "50%"],
            }),
            "interpretation": f"Price's Law: √n elite {entity.lower()} ({elite_n}) produce {elite_pct:.1f}% of output (expected ~50%)."
        }
    
    def _analyze_pareto(self, entity):
        """Analyze Pareto Principle (80/20 rule) using biblium counts."""
        # Try to get counts from biblium
        if entity == "Authors":
            count_attr = 'authors_counts_df'
            if not hasattr(self.bib, count_attr) or self.bib.authors_counts_df is None:
                self.bib.count_authors()
            counts_df = getattr(self.bib, count_attr, None)
        elif entity == "Sources":
            count_attr = 'sources_counts_df'
            if not hasattr(self.bib, count_attr) or self.bib.sources_counts_df is None:
                self.bib.count_sources()
            counts_df = getattr(self.bib, count_attr, None)
        else:
            count_attr = 'author_keywords_counts_df'
            if not hasattr(self.bib, count_attr) or self.bib.author_keywords_counts_df is None:
                self.bib.count_author_keywords()
            counts_df = getattr(self.bib, count_attr, None)
        
        if counts_df is not None and len(counts_df) > 0:
            # Find the document count column
            doc_col = None
            for col in counts_df.columns:
                if 'number' in col.lower() and 'document' in col.lower():
                    doc_col = col
                    break
            
            if doc_col:
                total_items = len(counts_df)
                total_pubs = counts_df[doc_col].sum()
                
                top_20_n = max(1, int(total_items * 0.2))
                top_20_pubs = counts_df.head(top_20_n)[doc_col].sum()
                top_20_pct = top_20_pubs / total_pubs * 100
                
                return {
                    "law": "Pareto Principle (80/20 Rule)",
                    "entity": entity,
                    "data": pd.DataFrame({
                        "Metric": [f"Total {entity}", "Top 20%", "Their Publications", "Their %", "Expected (80%)"],
                        "Value": [total_items, top_20_n, int(top_20_pubs), f"{top_20_pct:.1f}%", "80%"],
                    }),
                    "interpretation": f"Pareto: Top 20% of {entity.lower()} ({top_20_n}) produce {top_20_pct:.1f}% of output (expected ~80%).",
                    "from_biblium": True
                }
        
        # Fallback
        return self._fallback_pareto(entity)
    
    def _fallback_pareto(self, entity):
        """Fallback for Pareto analysis."""
        df = self.bib.df
        
        # Find appropriate column
        if entity == "Authors":
            col = None
            for c in ["Authors", "Authors or Inventors", "Author full names", "AU"]:
                if c in df.columns:
                    col = c
                    break
        elif entity == "Sources":
            col = None
            for c in ["Source title", "Source Title", "Journal", "Publication Name", "SO"]:
                if c in df.columns:
                    col = c
                    break
        else:
            col = None
            for c in ["Author Keywords", "Keywords", "DE", "ID"]:
                if c in df.columns:
                    col = c
                    break
        
        if col is None:
            raise ValueError(f"No {entity} column found")
        
        if entity in ["Authors", "Keywords"]:
            all_items = df[col].dropna().str.split(";").explode().str.strip()
        else:
            all_items = df[col].dropna()
        
        item_counts = all_items.value_counts()
        total_items = len(item_counts)
        total_pubs = item_counts.sum()
        
        top_20_n = max(1, int(total_items * 0.2))
        sorted_counts = item_counts.sort_values(ascending=False)
        top_20_pubs = sorted_counts.head(top_20_n).sum()
        top_20_pct = top_20_pubs / total_pubs * 100
        
        return {
            "law": "Pareto Principle (80/20 Rule)",
            "entity": entity,
            "data": pd.DataFrame({
                "Metric": [f"Total {entity}", "Top 20%", "Their Publications", "Their %", "Expected (80%)"],
                "Value": [total_items, top_20_n, int(top_20_pubs), f"{top_20_pct:.1f}%", "80%"],
            }),
            "interpretation": f"Pareto: Top 20% of {entity.lower()} ({top_20_n}) produce {top_20_pct:.1f}% of output (expected ~80%)."
        }
    
    def _on_analysis_success(self, result: Dict):
        """Display analysis results."""
        self._show_results(result)
    
    def _on_analysis_error(self, error: str):
        """Handle analysis error."""
        self._show_error(f"Analysis error: {error}")
    
    def _widget_exists(self, widget) -> bool:
        """Check if a tkinter widget still exists."""
        if widget is None:
            return False
        try:
            return widget.winfo_exists()
        except (tk.TclError, AttributeError):
            return False
    
    def _show_results(self, result: Dict):
        """Display results."""
        # Check if results_tab still exists
        if not self._widget_exists(self.results_tab):
            # Recreate the results tab if it was destroyed
            try:
                if not self._widget_exists(self.results_notebook):
                    return
                self.results_tab = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
                self.results_notebook.insert(0, self.results_tab, text="📊 Results")
            except tk.TclError:
                # If even the notebook is gone, we can't show results
                return
        
        try:
            for widget in self.results_tab.winfo_children():
                widget.destroy()
        except tk.TclError:
            return
        
        # Interpretation
        if "interpretation" in result:
            tk.Label(
                self.results_tab, text=result["interpretation"],
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                wraplength=500, justify=tk.LEFT,
            ).pack(fill=tk.X, pady=(0, 16))
        
        # Show stats if available (from biblium)
        if "stats_df" in result and result["stats_df"] is not None:
            stats_df = result["stats_df"]
            if len(stats_df) > 0:
                stats_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
                stats_frame.pack(fill=tk.X, pady=(0, 8))
                
                tk.Label(
                    stats_frame, text="Goodness of Fit:",
                    font=FONTS.get_font("body", bold=True),
                    bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                ).pack(anchor=tk.W)
                
                for _, row in stats_df.iterrows():
                    measure = row.iloc[0] if len(row) > 0 else ""
                    value = row.iloc[1] if len(row) > 1 else ""
                    tk.Label(
                        stats_frame, text=f"  {measure}: {value:.4f}" if isinstance(value, (int, float)) else f"  {measure}: {value}",
                        font=FONTS.get_font("small"),
                        bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
                    ).pack(anchor=tk.W)
        
        # Plot
        if HAS_MATPLOTLIB and "data" in result:
            plot = PlotFrame(self.results_tab, theme=self.theme_name, figsize=(8, 5), show_ai_button=True)
            plot.pack(fill=tk.X, pady=(0, 16))
            
            fig, ax = plot.get_figure()
            data = result["data"]
            law = result["law"]
            
            if "Lotka" in law:
                x_col = "Publications" if "Publications" in data.columns else data.columns[0]
                y_col = "Authors" if "Authors" in data.columns else data.columns[1]
                
                ax.scatter(data[x_col], data[y_col], color=self.theme["accent_primary"], label="Observed", s=50)
                
                if "Expected" in data.columns and self.show_fit_cb.get():
                    ax.plot(data[x_col], data["Expected"], 'r--', label="Theoretical", linewidth=2)
                
                if self.log_scale_cb.get():
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                
                ax.set_xlabel("Publications")
                ax.set_ylabel("Number of Authors")
                
            elif "Zipf" in law:
                x_col = "Rank" if "Rank" in data.columns else data.columns[0]
                y_col = "Frequency" if "Frequency" in data.columns else data.columns[1]
                
                ax.scatter(data[x_col], data[y_col], color=self.theme["accent_primary"], label="Observed", s=20, alpha=0.7)
                
                if "Expected" in data.columns and self.show_fit_cb.get():
                    ax.plot(data[x_col], data["Expected"], 'r--', label="Theoretical", linewidth=2)
                
                if self.log_scale_cb.get():
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                
                ax.set_xlabel("Rank")
                ax.set_ylabel("Frequency")
                
            elif "Bradford" in law:
                rank_col = "Rank" if "Rank" in data.columns else data.columns[0]
                cum_col = "Cumulative" if "Cumulative" in data.columns else "Cum %" if "Cum %" in data.columns else data.columns[3]
                
                ax.plot(data[rank_col], data[cum_col], color=self.theme["accent_primary"], linewidth=2)
                
                # Add zone markers if Zone column exists
                if "Zone" in data.columns:
                    for zone in data["Zone"].unique():
                        zone_data = data[data["Zone"] == zone]
                        if len(zone_data) > 0:
                            ax.axvline(x=zone_data[rank_col].iloc[-1], color='gray', linestyle='--', alpha=0.5)
                
                ax.set_xlabel("Source Rank")
                ax.set_ylabel("Cumulative Articles")
                
            elif "Price" in law or "Pareto" in law:
                # For summary tables, show a simple bar chart
                if "Metric" in data.columns:
                    metrics = data["Metric"].tolist()
                    values = []
                    for v in data["Value"]:
                        try:
                            if isinstance(v, str) and '%' in v:
                                values.append(float(v.replace('%', '')))
                            else:
                                values.append(float(v))
                        except:
                            values.append(0)
                    
                    # Only plot numeric values
                    plot_data = [(m, v) for m, v in zip(metrics, values) if v > 0 and '%' not in str(data[data["Metric"]==m]["Value"].values[0]) or 'Elite %' in m or 'Their %' in m]
                    if plot_data:
                        ax.barh([p[0] for p in plot_data], [p[1] for p in plot_data], color=self.theme["accent_primary"])
                else:
                    ax.bar(range(len(data)), data.iloc[:, 1], color=self.theme["accent_primary"])
            
            ax.legend()
            ax.set_title(law.split("(")[0].strip())
            fig.tight_layout()
            plot.refresh()
        
        # Table
        if "data" in result:
            table = DataTable(self.results_tab, theme=self.theme_name, max_rows=50)
            table.pack(fill=tk.BOTH, expand=True)
            table.set_data(result["data"])
