# -*- coding: utf-8 -*-
"""
Concept Builder Panel
=====================
Panel for creating binary concept indicator variables from keywords.

@author: Claude (Anthropic) for Lan.Umek
@version: 2.7.0
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from typing import Dict, Optional, List

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, CollapsibleCard, StatsCard, CardGrid
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox, LabeledSpinbox, LabeledEntry
from biblium.gui.widgets.tables import DataTable

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class ConceptBuilderPanel(BasePanel):
    """Panel for building concept indicator variables."""
    
    title = "Concept Builder"
    icon = "🏷️"
    description = "Create binary variables from keyword-based concepts"
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
        
        # Info Card
        info_frame = tk.Frame(self.options_content, bg="#e3f2fd", relief=tk.FLAT, bd=1)
        info_frame.pack(fill=tk.X, padx=8, pady=(8, 4))
        
        info_inner = tk.Frame(info_frame, bg="#e3f2fd", padx=8, pady=6)
        info_inner.pack(fill=tk.X)
        
        tk.Label(
            info_inner, text="ℹ️ Create binary concept variables",
            font=FONTS.get_font("body_bold"), bg="#e3f2fd", fg="#1565c0",
        ).pack(anchor=tk.W)
        
        tk.Label(
            info_inner, 
            text="Define concepts using keywords.\nUse * as wildcard (e.g., govern* matches\ngovernment, governance, governing).",
            font=FONTS.get_font("small"), bg="#e3f2fd", fg="#1565c0",
            justify=tk.LEFT,
        ).pack(anchor=tk.W)
        
        # Text Column Selection
        text_card = Card(self.options_content, title="📄 Text Source", theme=self.theme_name)
        text_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Get available text columns
        text_options = ["Auto-detect"]
        available_cols = []
        for col_id, col_name, _ in self.TEXT_COLUMNS[1:]:  # Skip auto
            if col_id in self.bib.df.columns:
                text_options.append(col_name)
                available_cols.append(col_id)
        
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
        
        # Concept Definition Card
        concept_card = Card(self.options_content, title="🏷️ Define Concepts", theme=self.theme_name)
        concept_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Load from file button
        load_frame = tk.Frame(concept_card.content, bg=self.theme["bg_card"])
        load_frame.pack(fill=tk.X, pady=(0, 8))
        
        ThemedButton(
            load_frame, text="📂 Load from File",
            command=self._load_concepts_from_file,
            theme=self.theme_name
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        self.file_label = tk.Label(
            load_frame, text="No file loaded",
            font=FONTS.get_font("small"), bg=self.theme["bg_card"],
            fg=self.theme["text_muted"]
        )
        self.file_label.pack(side=tk.LEFT)
        
        # Manual concept entry
        tk.Label(
            concept_card.content, text="Or define manually:",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"]
        ).pack(anchor=tk.W, pady=(8, 4))
        
        # Concept name entry
        name_frame = tk.Frame(concept_card.content, bg=self.theme["bg_card"])
        name_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            name_frame, text="Concept name:",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"], width=12, anchor=tk.W
        ).pack(side=tk.LEFT)
        
        self.concept_name_var = tk.StringVar()
        self.concept_name_entry = tk.Entry(
            name_frame, textvariable=self.concept_name_var,
            font=FONTS.get_font("body"), width=25
        )
        self.concept_name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Keywords entry
        kw_frame = tk.Frame(concept_card.content, bg=self.theme["bg_card"])
        kw_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            kw_frame, text="Keywords:",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"], width=12, anchor=tk.W
        ).pack(side=tk.LEFT)
        
        self.keywords_var = tk.StringVar()
        self.keywords_entry = tk.Entry(
            kw_frame, textvariable=self.keywords_var,
            font=FONTS.get_font("body"), width=25
        )
        self.keywords_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(
            concept_card.content, text="(separate keywords with semicolon ;)",
            font=FONTS.get_font("small"), bg=self.theme["bg_card"],
            fg=self.theme["text_muted"]
        ).pack(anchor=tk.W)
        
        # Add concept button
        ThemedButton(
            concept_card.content, text="➕ Add Concept",
            command=self._add_concept_manually,
            theme=self.theme_name
        ).pack(anchor=tk.W, pady=(8, 4))
        
        # Current concepts list
        list_card = CollapsibleCard(
            self.options_content, title="📋 Defined Concepts",
            theme=self.theme_name, collapsed=False
        )
        list_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Listbox for concepts
        list_frame = tk.Frame(list_card.content, bg=self.theme["bg_card"])
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.concepts_listbox = tk.Listbox(
            list_frame, height=6, font=FONTS.get_font("body"),
            selectmode=tk.SINGLE, bg=self.theme["bg_primary"],
            fg=self.theme["text_primary"]
        )
        self.concepts_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                  command=self.concepts_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.concepts_listbox.config(yscrollcommand=scrollbar.set)
        
        # Double-click to edit
        self.concepts_listbox.bind("<Double-Button-1>", lambda e: self._edit_selected_concept())
        
        # Concept management buttons
        btn_frame2 = tk.Frame(list_card.content, bg=self.theme["bg_card"])
        btn_frame2.pack(fill=tk.X, pady=(4, 0))
        
        ThemedButton(
            btn_frame2, text="✏️ Edit",
            command=self._edit_selected_concept,
            theme=self.theme_name
        ).pack(side=tk.LEFT, padx=(0, 4))
        
        ThemedButton(
            btn_frame2, text="🗑️ Remove",
            command=self._remove_selected_concept,
            theme=self.theme_name
        ).pack(side=tk.LEFT, padx=(0, 4))
        
        ThemedButton(
            btn_frame2, text="🗑️ Clear All",
            command=self._clear_all_concepts,
            theme=self.theme_name
        ).pack(side=tk.LEFT)
        
        # Hint for editing
        tk.Label(
            list_card.content, text="(Double-click to edit a concept)",
            font=FONTS.get_font("small"), bg=self.theme["bg_card"],
            fg=self.theme["text_muted"]
        ).pack(anchor=tk.W, pady=(4, 0))
        
        # Action Buttons
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Create Concept Variables", icon="🏷️",
            command=self._run_analysis, theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
        
        # Initialize concepts storage
        self._concepts: Dict[str, List[str]] = {}
        self._loaded_from_file = False
        self._concept_file_path = None
    
    def _load_concepts_from_file(self):
        """Load concepts from Excel/CSV file."""
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
            
            # Convert DataFrame to concepts dict
            self._concepts.clear()
            for col in df.columns:
                keywords = df[col].dropna().astype(str).str.strip().tolist()
                keywords = [k for k in keywords if k and k.lower() != 'nan']
                if keywords:
                    self._concepts[col] = keywords
            
            self._loaded_from_file = True
            self._concept_file_path = filepath
            
            # Update file label
            filename = filepath.split('/')[-1].split('\\')[-1]
            self.file_label.config(text=f"✓ {filename}", fg="#27ae60")
            
            # Update listbox
            self._update_concepts_listbox()
            
            messagebox.showinfo(
                "Concepts Loaded",
                f"Loaded {len(self._concepts)} concepts from file:\n" +
                "\n".join(f"• {c}: {len(kw)} keywords" for c, kw in self._concepts.items())
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
    
    def _add_concept_manually(self):
        """Add a concept manually."""
        name = self.concept_name_var.get().strip()
        keywords_str = self.keywords_var.get().strip()
        
        if not name:
            messagebox.showwarning("Missing Name", "Please enter a concept name.")
            return
        
        if not keywords_str:
            messagebox.showwarning("Missing Keywords", "Please enter at least one keyword.")
            return
        
        # Parse keywords
        keywords = [k.strip() for k in keywords_str.split(';') if k.strip()]
        
        if not keywords:
            messagebox.showwarning("Missing Keywords", "Please enter at least one keyword.")
            return
        
        # Add to concepts
        if name in self._concepts:
            # Append keywords
            self._concepts[name].extend(keywords)
            # Remove duplicates
            self._concepts[name] = list(dict.fromkeys(self._concepts[name]))
        else:
            self._concepts[name] = keywords
        
        # Clear entries
        self.concept_name_var.set("")
        self.keywords_var.set("")
        
        # Update listbox
        self._update_concepts_listbox()
    
    def _remove_selected_concept(self):
        """Remove selected concept from list."""
        selection = self.concepts_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a concept to remove.")
            return
        
        idx = selection[0]
        concept_name = list(self._concepts.keys())[idx]
        
        del self._concepts[concept_name]
        self._update_concepts_listbox()
    
    def _edit_selected_concept(self):
        """Edit selected concept - load it into the entry fields for modification."""
        selection = self.concepts_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a concept to edit.")
            return
        
        idx = selection[0]
        concept_name = list(self._concepts.keys())[idx]
        keywords = self._concepts[concept_name]
        
        # Load into entry fields
        self.concept_name_var.set(concept_name)
        self.keywords_var.set("; ".join(keywords))
        
        # Remove from concepts (will be re-added when user clicks Add)
        del self._concepts[concept_name]
        self._update_concepts_listbox()
        
        # Focus on keywords entry for easy editing
        self.keywords_entry.focus_set()
        self.keywords_entry.select_range(0, tk.END)
    
    def _clear_all_concepts(self):
        """Clear all concepts."""
        if not self._concepts:
            return
        
        if messagebox.askyesno("Confirm", "Clear all defined concepts?"):
            self._concepts.clear()
            self._loaded_from_file = False
            self._concept_file_path = None
            self.file_label.config(text="No file loaded", fg=self.theme["text_muted"])
            self._update_concepts_listbox()
    
    def _update_concepts_listbox(self):
        """Update the concepts listbox."""
        self.concepts_listbox.delete(0, tk.END)
        
        for name, keywords in self._concepts.items():
            display = f"{name} ({len(keywords)} keywords)"
            self.concepts_listbox.insert(tk.END, display)
    
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
            text="🏷️ Concept Builder\n\n"
                 "Load or define concepts, then click 'Create Concept Variables'.",
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
        
        if not self._concepts:
            messagebox.showwarning(
                "No Concepts",
                "Please define at least one concept.\n\n"
                "Either load from file or add manually."
            )
            return
        
        self._show_loading("Creating concept variables...")
        
        def do_analysis():
            try:
                # Determine text column
                text_col_selection = self.text_column.get()
                if text_col_selection == "Auto-detect":
                    text_col = None  # Let add_concepts auto-detect
                else:
                    # Find the actual column name
                    text_col = None
                    for col_id, col_name, _ in self.TEXT_COLUMNS:
                        if col_name == text_col_selection:
                            text_col = col_id
                            break
                
                # Call add_concepts
                self.bib.add_concepts(
                    concepts=self._concepts,
                    text_column=text_col,
                    use_regex=self.use_regex_var.get(),
                    verbose=True
                )
                
                # Determine which text column was actually used
                actual_text_col = text_col
                if actual_text_col is None:
                    # Find the auto-detected column
                    for col in ["Processed Combined Text", "Combined Text", "Processed Abstract",
                                "Abstract", "Processed Title", "Title", "Processed Author Keywords",
                                "Author Keywords", "Index Keywords"]:
                        if col in self.bib.df.columns:
                            actual_text_col = col
                            break
                
                # Prepare summary results
                results = []
                concept_cols = []
                for concept in self._concepts.keys():
                    if concept in self.bib.df.columns:
                        count = self.bib.df[concept].sum()
                        pct = 100 * count / len(self.bib.df)
                        results.append({
                            'Concept': concept,
                            'Keywords': len(self._concepts[concept]),
                            'Documents': int(count),
                            'Percentage': round(pct, 1)
                        })
                        concept_cols.append(concept)
                
                result_df = pd.DataFrame(results)
                
                # Prepare data view with concept columns and text column
                display_cols = concept_cols.copy()
                if actual_text_col and actual_text_col in self.bib.df.columns:
                    display_cols.append(actual_text_col)
                
                # Add Title if available and not already included
                title_col = self.bib.mapping.get("Title", "Title")
                if title_col in self.bib.df.columns and title_col not in display_cols:
                    display_cols.insert(0, title_col)
                
                data_df = self.bib.df[display_cols].copy()
                
                self.after(0, lambda r=result_df, d=data_df, t=actual_text_col: self._on_success(r, d, t))
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: self._show_error(msg))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _on_success(self, summary_df: pd.DataFrame, data_df: pd.DataFrame, text_col: str):
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
        self._concept_columns = list(summary_df['Concept'])
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
        event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": "Concept Builder"})
    
    def _apply_filter(self):
        """Apply or remove filter to show only tagged documents."""
        if not hasattr(self, '_current_data_full') or self._current_data_full is None:
            return
        
        if self._filter_var.get():
            # Filter to documents that match at least one concept
            mask = self._current_data_full[self._concept_columns].sum(axis=1) > 0
            filtered_df = self._current_data_full[mask].copy()
            self._data_table.set_data(filtered_df)
            self._doc_count_label.config(text=f"({len(filtered_df):,} of {len(self._current_data_full):,} documents)")
        else:
            # Show all documents
            self._data_table.set_data(self._current_data_full)
            self._doc_count_label.config(text=f"({len(self._current_data_full):,} documents)")
    
    def _export_filtered_data(self):
        """Export filtered data (only tagged documents)."""
        if not hasattr(self, '_current_data_full') or self._current_data_full is None:
            messagebox.showwarning("No Data", "No data to export.")
            return
        
        # Get filtered data
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
        self._current_result = summary_df
        self._current_data = data_df
        
        # Emit event
        event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": "Concept Builder"})
    
    def _export_full_data(self, data_df: pd.DataFrame):
        """Export full data with concept indicators."""
        if data_df is None or len(data_df) == 0:
            messagebox.showwarning("No Data", "No data to export.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")],
            title="Save Full Data"
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
    
    def _export_results(self):
        """Export results to Excel."""
        if not hasattr(self, '_current_result') or self._current_result is None:
            messagebox.showwarning("No Data", "No results to export.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")],
            title="Save Results"
        )
        
        if not filepath:
            return
        
        try:
            if filepath.endswith('.csv'):
                self._current_result.to_csv(filepath, index=False)
            else:
                self._current_result.to_excel(filepath, index=False)
            
            messagebox.showinfo("Success", f"Results exported to:\n{filepath}")
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
CONCEPT BUILDER
═══════════════

Create custom concept definitions for analysis.

WHAT IT DOES
────────────
• Define concepts using keywords
• Boolean logic (AND, OR, NOT)
• Combine multiple terms
• Test against your data

CONCEPT SYNTAX
──────────────
Simple: climate
Phrase: "machine learning"
OR: climate OR environment
AND: climate AND policy
NOT: energy NOT nuclear
Combined: (solar OR wind) AND policy

CONCEPT TYPES
─────────────
• Topic concepts
• Method concepts
• Geographic concepts
• Temporal concepts

TESTING
───────
• Preview matching documents
• Count matches
• Refine definition
• Save for reuse

APPLICATIONS
────────────
• Define research themes
• Create group filters
• Build SDG categories
• Custom classifications

EXPORT
──────
• Save concept definitions
• Load saved concepts
• Share with colleagues
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
