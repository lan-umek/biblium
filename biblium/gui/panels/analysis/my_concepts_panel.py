# -*- coding: utf-8 -*-
"""
My Concepts Panel
=================
Panel for creating binary concept indicators from user-provided concept files.
Similar to PA Concepts but requires user to load their own file.

@author: Claude (Anthropic) for Lan.Umek
@version: 2.7.0
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
from typing import Dict, Optional, List

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, CollapsibleCard, StatsCard, CardGrid
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox, LabeledSpinbox
from biblium.gui.widgets.tables import DataTable

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class MyConceptsPanel(BasePanel):
    """Panel for user-defined concept indicators from file."""
    
    title = "My Concepts"
    icon = "📁"
    description = "Create concept indicators from your own file"
    requires_data = True
    
    TEXT_COLUMNS = [
        ("auto", "Auto-detect", "Automatically find best text column"),
        ("Combined Text", "Combined Text", "All text fields combined"),
        ("Abstract", "Abstract", "Document abstract"),
        ("Title", "Title", "Document title"),
        ("Author Keywords", "Author Keywords", "Keywords provided by authors"),
        ("Index Keywords", "Index Keywords", "Keywords from indexing"),
        ("Processed Combined Text", "Processed Text", "Pre-processed combined text"),
    ]
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        if not self.bib:
            self._show_no_data_message()
            return
        
        # Initialize
        self._concept_df = None
        self._concept_vars = {}
        self._concept_file_path = None
        
        # Info Card
        info_frame = tk.Frame(self.options_content, bg="#fff3e0", relief=tk.FLAT, bd=1)
        info_frame.pack(fill=tk.X, padx=8, pady=(8, 4))
        
        info_inner = tk.Frame(info_frame, bg="#fff3e0", padx=8, pady=6)
        info_inner.pack(fill=tk.X)
        
        tk.Label(
            info_inner, text="📁 My Concepts",
            font=FONTS.get_font("body_bold"), bg="#fff3e0", fg="#e65100",
        ).pack(anchor=tk.W)
        
        tk.Label(
            info_inner, 
            text="Load your own concept file (Excel/CSV).\nFormat: column names = concept names,\nrows = keywords for each concept.",
            font=FONTS.get_font("small"), bg="#fff3e0", fg="#e65100",
            justify=tk.LEFT,
        ).pack(anchor=tk.W)
        
        # File Loading Card
        file_card = Card(self.options_content, title="📂 Load Concepts File", theme=self.theme_name)
        file_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Load button and status
        load_frame = tk.Frame(file_card.content, bg=self.theme["bg_card"])
        load_frame.pack(fill=tk.X, pady=(0, 8))
        
        ThemedButton(
            load_frame, text="📂 Browse...",
            command=self._load_concepts_file,
            theme=self.theme_name
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        self._file_label = tk.Label(
            load_frame, text="No file loaded",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_muted"]
        )
        self._file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # File format hint
        tk.Label(
            file_card.content, 
            text="Supported: .xlsx, .xls, .csv",
            font=FONTS.get_font("small"), bg=self.theme["bg_card"],
            fg=self.theme["text_muted"]
        ).pack(anchor=tk.W)
        
        # Text Column Selection
        text_card = Card(self.options_content, title="📄 Text Source", theme=self.theme_name)
        text_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Get available text columns
        text_options = ["Auto-detect"]
        for col_id, col_name, _ in self.TEXT_COLUMNS[1:]:
            if col_id in self.bib.df.columns:
                text_options.append(col_name)
        
        self.text_column = LabeledCombobox(
            text_card.content, label="Search in:",
            values=text_options, default="Auto-detect",
            theme=self.theme_name, label_width=12,
            tooltip="Column to search for keywords"
        )
        self.text_column.pack(fill=tk.X, pady=4)
        
        self.use_regex_var = tk.BooleanVar(value=False)
        LabeledCheckbox(
            text_card.content, label="Use regular expressions",
            variable=self.use_regex_var, theme=self.theme_name,
            tooltip="Treat keywords as regex patterns"
        ).pack(fill=tk.X, pady=4)
        
        # Concept Selection Card (will be populated after file load)
        self._concept_card = Card(self.options_content, title="🏷️ Select Concepts", theme=self.theme_name)
        self._concept_card.pack(fill=tk.X, padx=8, pady=8)
        
        self._concept_content_frame = tk.Frame(self._concept_card.content, bg=self.theme["bg_card"])
        self._concept_content_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(
            self._concept_content_frame, 
            text="Load a concepts file to see available concepts.",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_muted"]
        ).pack(anchor=tk.W, pady=8)
        
        # Action Buttons
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Create Concept Variables", icon="🏷️",
            command=self._run_analysis, theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
    
    def _load_concepts_file(self):
        """Load concepts from user-selected file."""
        filetypes = [
            ("Excel files", "*.xlsx *.xls"),
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select Concepts File",
            filetypes=filetypes
        )
        
        if not filepath:
            return
        
        try:
            if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath)
            
            if len(df.columns) == 0:
                messagebox.showerror("Error", "File has no columns.")
                return
            
            self._concept_df = df
            self._concept_file_path = filepath
            
            # Update file label
            filename = os.path.basename(filepath)
            self._file_label.config(text=f"✓ {filename}", fg="#27ae60")
            
            # Update concept selection UI
            self._update_concept_selection()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
    
    def _update_concept_selection(self):
        """Update the concept selection UI after loading a file."""
        # Clear existing content
        for widget in self._concept_content_frame.winfo_children():
            widget.destroy()
        
        if self._concept_df is None:
            tk.Label(
                self._concept_content_frame, 
                text="Load a concepts file to see available concepts.",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                fg=self.theme["text_muted"]
            ).pack(anchor=tk.W, pady=8)
            return
        
        # Show available concepts with checkboxes
        tk.Label(
            self._concept_content_frame, 
            text=f"Available concepts ({len(self._concept_df.columns)}):",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"]
        ).pack(anchor=tk.W, pady=(0, 8))
        
        self._concept_vars = {}
        
        # Create scrollable frame if many concepts
        if len(self._concept_df.columns) > 8:
            canvas = tk.Canvas(self._concept_content_frame, bg=self.theme["bg_card"], 
                              highlightthickness=0, height=200)
            scrollbar = ttk.Scrollbar(self._concept_content_frame, orient=tk.VERTICAL, 
                                     command=canvas.yview)
            scrollable_frame = tk.Frame(canvas, bg=self.theme["bg_card"])
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            checkbox_parent = scrollable_frame
        else:
            checkbox_parent = self._concept_content_frame
        
        for concept in self._concept_df.columns:
            # Get keywords for display
            keywords = self._concept_df[concept].dropna().astype(str).str.strip().tolist()
            keywords = [k for k in keywords if k and k.lower() != 'nan']
            
            var = tk.BooleanVar(value=True)  # Selected by default
            self._concept_vars[concept] = var
            
            cb_frame = tk.Frame(checkbox_parent, bg=self.theme["bg_card"])
            cb_frame.pack(fill=tk.X, pady=2)
            
            cb = tk.Checkbutton(
                cb_frame, text=concept,
                variable=var,
                bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                selectcolor=self.theme["bg_secondary"],
                font=FONTS.get_font("body"),
                anchor=tk.W,
            )
            cb.pack(side=tk.LEFT)
            
            tk.Label(
                cb_frame, text=f"({len(keywords)} keywords)",
                font=FONTS.get_font("small"), bg=self.theme["bg_card"],
                fg=self.theme["text_muted"]
            ).pack(side=tk.LEFT, padx=(4, 0))
        
        # Select all / none buttons
        sel_frame = tk.Frame(self._concept_content_frame, bg=self.theme["bg_card"])
        sel_frame.pack(fill=tk.X, pady=(8, 0))
        
        ThemedButton(
            sel_frame, text="Select All",
            command=self._select_all_concepts,
            theme=self.theme_name
        ).pack(side=tk.LEFT, padx=(0, 4))
        
        ThemedButton(
            sel_frame, text="Select None",
            command=self._select_no_concepts,
            theme=self.theme_name
        ).pack(side=tk.LEFT)
        
        # Keywords preview (collapsible)
        preview_card = CollapsibleCard(
            self._concept_content_frame, title="🔤 Keywords Preview",
            theme=self.theme_name, collapsed=True
        )
        preview_card.pack(fill=tk.X, pady=(8, 0))
        
        for concept in self._concept_df.columns:
            keywords = self._concept_df[concept].dropna().astype(str).str.strip().tolist()
            keywords = [k for k in keywords if k and k.lower() != 'nan']
            
            tk.Label(
                preview_card.content, text=f"{concept}:",
                font=FONTS.get_font("body_bold"), bg=self.theme["bg_card"],
                fg=self.theme["text_primary"], anchor=tk.W
            ).pack(fill=tk.X, pady=(4, 0))
            
            tk.Label(
                preview_card.content, text=", ".join(keywords) if keywords else "(no keywords)",
                font=FONTS.get_font("small"), bg=self.theme["bg_card"],
                fg=self.theme["text_muted"], anchor=tk.W, wraplength=280,
                justify=tk.LEFT
            ).pack(fill=tk.X)
    
    def _select_all_concepts(self):
        """Select all concepts."""
        for var in self._concept_vars.values():
            var.set(True)
    
    def _select_no_concepts(self):
        """Deselect all concepts."""
        for var in self._concept_vars.values():
            var.set(False)
    
    def _create_results(self):
        """Create results panel."""
        super()._create_results()
        
        # Create notebook with Info tab
        self.results_notebook = ttk.Notebook(self.results_content)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        self.results_tab = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.results_tab, text="📊 Results")
        
        # Info tab
        self.info_tab = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.info_tab, text="ℹ️ Info")
        self._create_info_content(self.info_tab)
        
        self._show_placeholder()
    
    def _show_placeholder(self):
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
            text="📁 My Concepts\n\n"
                 "Load your concept file and click 'Create Variables'.",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)

    def _run_analysis(self):
        """Run concept creation."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        if self._concept_df is None:
            messagebox.showwarning(
                "No Concepts File",
                "Please load a concepts file first.\n\n"
                "Click 'Browse...' to select your file."
            )
            return
        
        # Get selected concepts
        selected = [c for c, var in self._concept_vars.items() if var.get()]
        
        if not selected:
            messagebox.showwarning(
                "No Concepts Selected",
                "Please select at least one concept."
            )
            return
        
        self._show_loading("Creating concept variables...")
        
        def do_analysis():
            try:
                # Build concepts dict from selected
                concepts = {}
                for concept in selected:
                    keywords = self._concept_df[concept].dropna().astype(str).str.strip().tolist()
                    keywords = [k for k in keywords if k and k.lower() != 'nan']
                    if keywords:
                        concepts[concept] = keywords
                
                if not concepts:
                    self.after(0, lambda: self._show_error("No valid keywords found in selected concepts."))
                    return
                
                # Determine text column
                text_col_selection = self.text_column.get()
                if text_col_selection == "Auto-detect":
                    text_col = None
                else:
                    text_col = None
                    for col_id, col_name, _ in self.TEXT_COLUMNS:
                        if col_name == text_col_selection:
                            text_col = col_id
                            break
                
                # Call add_concepts
                self.bib.add_concepts(
                    concepts=concepts,
                    text_column=text_col,
                    use_regex=self.use_regex_var.get(),
                    verbose=True
                )
                
                # Determine which text column was actually used
                actual_text_col = text_col
                if actual_text_col is None:
                    for col in ["Processed Combined Text", "Combined Text", "Processed Abstract",
                                "Abstract", "Processed Title", "Title", "Processed Author Keywords",
                                "Author Keywords", "Index Keywords"]:
                        if col in self.bib.df.columns:
                            actual_text_col = col
                            break
                
                # Prepare summary results
                results = []
                concept_cols = []
                for concept in concepts.keys():
                    if concept in self.bib.df.columns:
                        count = self.bib.df[concept].sum()
                        pct = 100 * count / len(self.bib.df)
                        results.append({
                            'Concept': concept,
                            'Keywords': len(concepts[concept]),
                            'Documents': int(count),
                            'Percentage': round(pct, 1)
                        })
                        concept_cols.append(concept)
                
                result_df = pd.DataFrame(results)
                
                # Prepare data view
                display_cols = concept_cols.copy()
                if actual_text_col and actual_text_col in self.bib.df.columns:
                    display_cols.append(actual_text_col)
                
                title_col = self.bib.mapping.get("Title", "Title")
                if title_col in self.bib.df.columns and title_col not in display_cols:
                    display_cols.insert(0, title_col)
                
                data_df = self.bib.df[display_cols].copy()
                
                self.after(0, lambda r=result_df, d=data_df, t=actual_text_col, c=concept_cols: 
                          self._on_success(r, d, t, c))
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: self._show_error(msg))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _on_success(self, summary_df: pd.DataFrame, data_df: pd.DataFrame, 
                    text_col: str, concept_cols: List[str]):
        """Display results."""
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
        
        if summary_df is None or len(summary_df) == 0:
            self._show_error("No concepts were created.")
            return
        
        # Store data for filtering
        self._current_data_full = data_df.copy()
        self._concept_columns = concept_cols
        self._text_col = text_col
        
        # Summary Cards
        grid = CardGrid(self.results_tab, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 16))
        
        grid.add_card(StatsCard(grid, "Concepts", f"{len(summary_df):,}", "🏷️", self.theme_name))
        
        total_docs = summary_df['Documents'].sum()
        grid.add_card(StatsCard(grid, "Tagged Docs", f"{total_docs:,}", "📄", self.theme_name))
        
        avg_pct = summary_df['Percentage'].mean()
        grid.add_card(StatsCard(grid, "Avg Coverage", f"{avg_pct:.1f}%", "📊", self.theme_name))
        
        total_kw = summary_df['Keywords'].sum()
        grid.add_card(StatsCard(grid, "Total Keywords", f"{total_kw:,}", "🔤", self.theme_name))
        
        # File info
        if self._concept_file_path:
            filename = os.path.basename(self._concept_file_path)
            tk.Label(
                self.results_tab, 
                text=f"Concepts file: {filename}",
                font=FONTS.get_font("small"), bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
            ).pack(anchor=tk.W, pady=(0, 4))
        
        # Text column info
        if text_col:
            tk.Label(
                self.results_tab, 
                text=f"Text column used: {text_col}",
                font=FONTS.get_font("small"), bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
            ).pack(anchor=tk.W, pady=(0, 8))
        
        # Results Table - Summary
        tk.Label(
            self.results_tab, text="📋 Concept Summary",
            font=FONTS.get_font("subheading"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(fill=tk.X, pady=(8, 4))
        
        summary_table = DataTable(self.results_tab, theme=self.theme_name, height=6)
        summary_table.pack(fill=tk.X, pady=(0, 16))
        summary_table.set_data(summary_df)
        
        # Filter controls
        filter_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
        filter_frame.pack(fill=tk.X, pady=(8, 4))
        
        tk.Label(
            filter_frame, text="📄 Document Data with Concept Indicators",
            font=FONTS.get_font("subheading"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT)
        
        # Filter checkbox
        self._filter_var = tk.BooleanVar(value=False)
        filter_cb = tk.Checkbutton(
            filter_frame, text="Show only tagged documents",
            variable=self._filter_var,
            command=self._apply_filter,
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            selectcolor=self.theme["bg_secondary"],
            font=FONTS.get_font("body"),
        )
        filter_cb.pack(side=tk.RIGHT, padx=8)
        
        # Document count label
        self._doc_count_label = tk.Label(
            filter_frame, text=f"({len(data_df):,} documents)",
            font=FONTS.get_font("small"), bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        )
        self._doc_count_label.pack(side=tk.RIGHT)
        
        # Data table
        self._data_table = DataTable(self.results_tab, theme=self.theme_name, height=12)
        self._data_table.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self._data_table.set_data(data_df)
        
        # Action buttons
        btn_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
        btn_frame.pack(fill=tk.X, pady=8)
        
        ThemedButton(
            btn_frame, text="📥 Export Summary",
            command=self._export_results, theme=self.theme_name
        ).pack(side=tk.LEFT, padx=4)
        
        ThemedButton(
            btn_frame, text="📥 Export Full Data",
            command=lambda: self._export_full_data(self._current_data_full), theme=self.theme_name
        ).pack(side=tk.LEFT, padx=4)
        
        ThemedButton(
            btn_frame, text="📥 Export Filtered Data",
            command=self._export_filtered_data, theme=self.theme_name
        ).pack(side=tk.LEFT, padx=4)
        
        # Store results
        self._current_result = summary_df
        self._current_data = data_df
        
        # Emit event
        event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": "My Concepts"})
    
    def _apply_filter(self):
        """Apply or remove filter to show only tagged documents."""
        if not hasattr(self, '_current_data_full') or self._current_data_full is None:
            return
        
        if self._filter_var.get():
            mask = self._current_data_full[self._concept_columns].sum(axis=1) > 0
            filtered_df = self._current_data_full[mask].copy()
            self._data_table.set_data(filtered_df)
            self._doc_count_label.config(text=f"({len(filtered_df):,} of {len(self._current_data_full):,} documents)")
        else:
            self._data_table.set_data(self._current_data_full)
            self._doc_count_label.config(text=f"({len(self._current_data_full):,} documents)")
    
    def _export_results(self):
        """Export summary results to Excel."""
        if not hasattr(self, '_current_result') or self._current_result is None:
            messagebox.showwarning("No Data", "No results to export.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")],
            title="Save Concept Summary"
        )
        
        if not filepath:
            return
        
        try:
            if filepath.endswith('.csv'):
                self._current_result.to_csv(filepath, index=False)
            else:
                self._current_result.to_excel(filepath, index=False)
            
            messagebox.showinfo("Success", f"Summary exported to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def _export_full_data(self, data_df: pd.DataFrame):
        """Export full data with concept indicators."""
        if data_df is None or len(data_df) == 0:
            messagebox.showwarning("No Data", "No data to export.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")],
            title="Save Full Data with Concept Indicators"
        )
        
        if not filepath:
            return
        
        try:
            if filepath.endswith('.csv'):
                data_df.to_csv(filepath, index=False)
            else:
                data_df.to_excel(filepath, index=False)
            
            messagebox.showinfo("Success", f"Data exported to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def _export_filtered_data(self):
        """Export filtered data (only tagged documents)."""
        if not hasattr(self, '_current_data_full') or self._current_data_full is None:
            messagebox.showwarning("No Data", "No data to export.")
            return
        
        mask = self._current_data_full[self._concept_columns].sum(axis=1) > 0
        filtered_df = self._current_data_full[mask].copy()
        
        if len(filtered_df) == 0:
            messagebox.showwarning("No Data", "No documents match any concept.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")],
            title="Save Filtered Data"
        )
        
        if not filepath:
            return
        
        try:
            if filepath.endswith('.csv'):
                filtered_df.to_csv(filepath, index=False)
            else:
                filtered_df.to_excel(filepath, index=False)
            
            messagebox.showinfo("Success", f"Filtered data ({len(filtered_df):,} documents) exported to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
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
        
        tk.Label(
            frame, text="⏳", font=("Segoe UI", 32),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"]
        ).pack(pady=(20, 10))
        
        tk.Label(
            frame, text=message, font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_muted"]
        ).pack()
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
MY CONCEPTS
═══════════

Manage and apply your saved concept definitions.

FEATURES
────────
• View saved concepts
• Edit definitions
• Apply to dataset
• Create groups from concepts

CONCEPT MANAGEMENT
──────────────────
• Create new concepts
• Duplicate existing
• Delete unused
• Rename concepts

APPLYING CONCEPTS
─────────────────
• Tag documents by concept
• Create concept columns
• Count matches
• Overlap analysis

ANALYSIS OPTIONS
────────────────
• Concept distribution
• Concept trends over time
• Concept co-occurrence
• Concept comparison

EXPORT
──────
• Export concept definitions
• Export tagged data
• Share concept library
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
        
        tk.Label(
            self.results_tab,
            text=f"❌ Error\n\n{message}",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["error"], justify=tk.CENTER, wraplength=400,
        ).pack(expand=True)
