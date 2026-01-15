# -*- coding: utf-8 -*-
"""
Top Cited Documents Panel
=========================
Display globally and locally top-cited documents from the dataset.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from typing import Dict, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, CollapsibleCard
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledSpinbox
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.progress import LoadingSpinner

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class TopCitedPanel(BasePanel):
    """Panel for displaying top-cited documents (global and local)."""
    
    title = "Top Cited"
    icon = "🏆"
    description = "Global and local top-cited documents"
    requires_data = True
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._global_df = None
        self._local_df = None
        self._per_year_df = None
        
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._run_analysis  # Set primary action for toolbar Run button
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Analysis Type Card
        type_card = Card(self.options_content, title="Analysis Type", theme=self.theme_name)
        type_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.analysis_type = LabeledCombobox(
            type_card.content,
            label="Type:",
            values=["Global Citations", "Local Citations", "Citations per Year", "All"],
            default="Global Citations",
            theme=self.theme_name,
            label_width=12,
        )
        self.analysis_type.pack(fill=tk.X, pady=4)
        
        # Options Card
        options_card = Card(self.options_content, title="Options", theme=self.theme_name)
        options_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.top_n_spin = LabeledSpinbox(
            options_card.content,
            label="Top N:",
            from_=5,
            to=100,
            default=10,
            theme=self.theme_name,
            label_width=12,
        )
        self.top_n_spin.pack(fill=tk.X, pady=4)
        
        # Include ties option
        self.include_ties_var = tk.BooleanVar(value=False)
        ties_frame = tk.Frame(options_card.content, bg=self.theme["bg_card"])
        ties_frame.pack(fill=tk.X, pady=4)
        
        tk.Checkbutton(
            ties_frame,
            text="Include ties",
            variable=self.include_ties_var,
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            activebackground=self.theme["bg_card"],
            activeforeground=self.theme["text_primary"],
            selectcolor=self.theme["bg_secondary"],
            font=FONTS.get_font("body"),
        ).pack(anchor=tk.W)
        
        # Action Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        self.analyze_btn = ThemedButton(
            btn_frame,
            text="Analyze Top Cited",
            style="primary",
            command=self._run_analysis,
            theme=self.theme_name,
        )
        self.analyze_btn.pack(fill=tk.X)
        
        # Status
        self.status_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        self.status_frame.pack(fill=tk.X, padx=8)
        
        self.spinner = LoadingSpinner(self.status_frame, size=16, theme=self.theme_name)
        self.status_var = tk.StringVar(value="")
        self.status_label = tk.Label(
            self.status_frame,
            textvariable=self.status_var,
            font=FONTS.get_font("small"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_muted"],
        )
    
    def _create_results(self):
        """Create the results panel with tabs."""
        # Results container
        self.results_card = tk.Frame(
            self.results_frame, bg=self.theme["bg_card"],
            highlightbackground=self.theme["border"], highlightthickness=1,
        )
        self.results_card.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        # Header
        header = tk.Frame(self.results_card, bg=self.theme["bg_card"])
        header.pack(fill=tk.X)
        
        tk.Label(
            header, text="Top Cited Documents",
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            padx=12, pady=8,
        ).pack(side=tk.LEFT)
        
        # Notebook for tabs
        style = ttk.Style()
        style.configure("TopCited.TNotebook", background=self.theme["bg_card"])
        style.configure("TopCited.TNotebook.Tab", padding=[12, 4])
        
        self.notebook = ttk.Notebook(self.results_card, style="TopCited.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Create tabs
        self._create_global_tab()
        self._create_local_tab()
        self._create_per_year_tab()
        
        # Initial message
        self._show_initial_message()
    
    def _create_global_tab(self):
        """Create the global citations tab."""
        self.global_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.global_frame, text="Global Citations")
        
        # Table placeholder
        self.global_table_frame = tk.Frame(self.global_frame, bg=self.theme["bg_card"])
        self.global_table_frame.pack(fill=tk.BOTH, expand=True)
    
    def _create_local_tab(self):
        """Create the local citations tab."""
        self.local_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.local_frame, text="Local Citations")
        
        # Table placeholder
        self.local_table_frame = tk.Frame(self.local_frame, bg=self.theme["bg_card"])
        self.local_table_frame.pack(fill=tk.BOTH, expand=True)
    
    def _create_per_year_tab(self):
        """Create the citations per year tab."""
        self.per_year_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.per_year_frame, text="Per Year")

        

        # Info tab

        info_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])

        self.notebook.add(info_frame, text="ℹ️ Info")

        self._create_info_content(info_frame)
        
        # Table placeholder
        self.per_year_table_frame = tk.Frame(self.per_year_frame, bg=self.theme["bg_card"])
        self.per_year_table_frame.pack(fill=tk.BOTH, expand=True)
    
    def _show_initial_message(self):
        """Show initial message in all tabs."""
        for frame in [self.global_table_frame, self.local_table_frame, self.per_year_table_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
            
            tk.Label(
                frame,
                text="Click 'Analyze Top Cited' to display results",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
            ).pack(expand=True)
    
    def _show_placeholder(self):
        """Show detailed placeholder with instructions."""
        for widget in self.results_content.winfo_children():
            widget.destroy()
        
        msg = (
            "🏆 Most Cited Items\n\n"
            "Identify the most highly cited publications.\n\n"
            "Features:\n"
            "• Top cited papers ranking\n"
            "• Citation percentiles\n"
            "• Highly cited threshold analysis\n"
            "• Export top cited lists\n"
            "\n"
            "Understand influential research in the field.\n\n"
            "Steps:\n"
            "1. Load dataset with citations\n"
            "2. Set number of top items\n"
            "3. Choose ranking metric\n"
            "4. View and export results\n"
        )
        
        tk.Label(
            self.results_content,
            text=msg,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)

    def _run_analysis(self):
        """Run the top cited analysis."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        # Show loading state
        self.analyze_btn.set_enabled(False)
        self.spinner.pack(side=tk.LEFT, padx=(0, 8))
        self.spinner.start()
        self.status_label.pack(side=tk.LEFT)
        self.status_var.set("Analyzing...")
        
        def do_analysis():
            try:
                top_n = self.top_n_spin.get()
                analysis_type = self.analysis_type.get()
                include_ties = self.include_ties_var.get()
                
                # Create a working copy of the dataframe
                df = self.bib.df.copy()
                
                # Find citation column
                cite_col = None
                for col in ["Cited by", "Times Cited, All Databases", "TC", "Citations"]:
                    if col in df.columns:
                        cite_col = col
                        break
                
                if cite_col is None:
                    raise ValueError("No citation column found in dataset")
                
                # Convert citation column to numeric
                df[cite_col] = pd.to_numeric(df[cite_col], errors='coerce').fillna(0).astype(int)
                
                # Find and convert year column
                year_col = None
                for col in ["Year", "PY", "Publication Year"]:
                    if col in df.columns:
                        year_col = col
                        break
                
                if year_col:
                    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
                
                # Find title column
                title_col = None
                for col in ["Title", "TI", "Document Title"]:
                    if col in df.columns:
                        title_col = col
                        break
                
                # Find reference column
                ref_col = None
                for col in ["References", "Cited References", "CR"]:
                    if col in df.columns:
                        ref_col = col
                        break
                
                # Find author column - WoS uses "Authors or Inventors"
                author_col = None
                for col in ["Authors", "Authors or Inventors", "AU"]:
                    if col in df.columns:
                        author_col = col
                        break
                
                results = {}
                
                # Global citations
                if analysis_type in ["Global Citations", "All"]:
                    from biblium import utilsbib
                    results["global"] = utilsbib.select_global_top_cited_documents(
                        df,
                        top_n=top_n,
                        cite_col=cite_col,
                        include_ties=include_ties,
                    )
                
                # Local citations
                if analysis_type in ["Local Citations", "All"]:
                    from biblium import utilsbib
                    
                    # Check if this is OpenAlex data (has OpenAlex ID column)
                    is_openalex = "OpenAlex ID" in df.columns or "unique-id" in df.columns
                    
                    if is_openalex and ref_col:
                        # Use OpenAlex-specific function that works with OpenAlex URLs in references
                        try:
                            # Find the ID column for OpenAlex
                            id_col_oa = None
                            for col in ["OpenAlex ID", "unique-id"]:
                                if col in df.columns:
                                    id_col_oa = col
                                    break
                            
                            results["local"] = utilsbib.top_openalex_local_cited_documents(
                                df,
                                refs_col=ref_col,
                                id_col=id_col_oa,
                                sep="|",
                                top_n=top_n,
                            )
                        except Exception as e:
                            results["local_error"] = str(e)
                    elif ref_col and title_col:
                        # Use standard title-matching function for Scopus/WoS
                        try:
                            # Build custom cols list based on available columns
                            cols = []
                            if author_col:
                                cols.append(author_col)
                            if title_col:
                                cols.append(title_col)
                            for col in ["Source title", "SO", "Year", "PY", "Document Type", "DT"]:
                                if col in df.columns and col not in cols:
                                    cols.append(col)
                            
                            results["local"] = utilsbib.select_local_top_cited_documents(
                                df,
                                top_n=top_n,
                                cols=cols if cols else None,
                                title_col=title_col,
                                ref_col=ref_col,
                                cite_col=cite_col,
                            )
                        except Exception as e:
                            results["local_error"] = str(e)
                    else:
                        results["local_error"] = "References column not available"
                
                # Citations per year
                if analysis_type in ["Citations per Year", "All"]:
                    if year_col:
                        from biblium import utilsbib
                        try:
                            # Build custom cols list based on available columns
                            per_year_cols = []
                            if author_col:
                                per_year_cols.append(author_col)
                            if title_col:
                                per_year_cols.append(title_col)
                            for col in ["Source title", "SO", "Document Type", "DT"]:
                                if col in df.columns and col not in per_year_cols:
                                    per_year_cols.append(col)
                            if year_col and year_col not in per_year_cols:
                                per_year_cols.append(year_col)
                            
                            results["per_year"] = utilsbib.select_top_cited_normalized_per_year(
                                df,
                                top_n=top_n,
                                cols=per_year_cols if per_year_cols else None,
                                year_col=year_col,
                                cite_col=cite_col,
                            )
                        except Exception as e:
                            results["per_year_error"] = str(e)
                    else:
                        results["per_year_error"] = "Year column not found"
                
                self.after(0, lambda: self._on_analysis_success(results, analysis_type))
                
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.after(0, lambda: self._on_analysis_error(str(e), tb))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _on_analysis_success(self, results, analysis_type):
        """Handle successful analysis."""
        self.analyze_btn.set_enabled(True)
        self.spinner.stop()
        self.spinner.pack_forget()
        self.status_label.pack_forget()
        
        # Update global tab
        if "global" in results:
            self._global_df = results["global"]
            self._update_table(self.global_table_frame, self._global_df, "Global")
            if analysis_type == "Global Citations":
                self.notebook.select(0)
        
        # Update local tab
        if "local" in results:
            self._local_df = results["local"]
            self._update_table(self.local_table_frame, self._local_df, "Local")
            if analysis_type == "Local Citations":
                self.notebook.select(1)
        elif "local_error" in results:
            self._show_error_in_tab(self.local_table_frame, results["local_error"])
        
        # Update per year tab
        if "per_year" in results:
            self._per_year_df = results["per_year"]
            self._update_table(self.per_year_table_frame, self._per_year_df, "Per Year")
            if analysis_type == "Citations per Year":
                self.notebook.select(2)
        elif "per_year_error" in results:
            self._show_error_in_tab(self.per_year_table_frame, results["per_year_error"])
        
        # Show count
        total = sum(len(df) for df in [self._global_df, self._local_df, self._per_year_df] if df is not None)
        self.status_var.set(f"Found {total} documents")
        self.status_label.pack(side=tk.LEFT)
    
    def _on_analysis_error(self, error, traceback):
        """Handle analysis error."""
        self.analyze_btn.set_enabled(True)
        self.spinner.stop()
        self.spinner.pack_forget()
        self.status_var.set(f"Error: {error}")
        self.status_label.pack(side=tk.LEFT)
        
        print(f"Top Cited Analysis Error: {error}")
        print(traceback)
    
    def _update_table(self, frame, df, label):
        """Update a table frame with data."""
        # Clear existing
        for widget in frame.winfo_children():
            widget.destroy()
        
        if df is None or df.empty:
            tk.Label(
                frame,
                text=f"No {label.lower()} citation data available",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
            ).pack(expand=True)
            return
        
        # Create table
        table = DataTable(
            frame,
            theme=self.theme_name,
            show_index=True,
        )
        table.pack(fill=tk.BOTH, expand=True)
        table.set_data(df)
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
MOST CITED ITEMS
════════════════

Identify highly cited publications and entities.

ANALYSIS MODES
──────────────
• Top Cited Papers
  - Highest citation counts
  - Percentile rankings
  
• Top Cited Authors
  - Most cited researchers
  - H-index leaders
  
• Top Cited Sources
  - High-impact journals

RANKING OPTIONS
───────────────
• Total citations: Raw count
• Citations per year: Age-normalized
• Percentile: Position in distribution

TOP CITED THRESHOLDS
────────────────────
Common definitions:
• Top 1%: Highly cited (WoS ESI)
• Top 10%: High impact
• Top 25%: Above average
• Top 50%: Upper half

OUTPUT
──────
• Ranked list with citations
• Bibliographic details
• Percentile position
• Publication year

INTERPRETATION
──────────────
High citations may indicate:
• Foundational/seminal work
• Methodology papers
• Review articles
• Controversial findings

CONSIDERATIONS
──────────────
• Age affects total citations
• Field norms vary greatly
• Document types differ
• Self-citations possible
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

    def _show_error_in_tab(self, frame, error_msg):
        """Show error message in a tab."""
        for widget in frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            frame,
            text=f"Not available: {error_msg}",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["danger"],
        ).pack(expand=True)
