# -*- coding: utf-8 -*-
"""
Load Data Panel
===============
Panel for loading bibliometric datasets.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
from typing import Optional

from biblium.gui.config import (
    FONTS, LAYOUT, get_theme,
    DATABASE_OPTIONS, PREPROCESS_LEVELS, KEYWORD_OPTIONS, LANGUAGE_OPTIONS, COLOR_PALETTES,
)
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, CollapsibleCard
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import (
    LabeledEntry, LabeledCombobox, LabeledCheckbox, 
    LabeledSpinbox, LabeledTextArea, RadioGroup,
)
from biblium.gui.widgets.progress import LoadingSpinner
from biblium.gui.widgets.tables import DataTable

try:
    from biblium import BiblioAnalysis, BiblioPlot
    HAS_BIBLIUM = True
except ImportError:
    HAS_BIBLIUM = False


class LoadDataPanel(BasePanel):
    """Panel for loading bibliometric datasets."""
    
    title = "Load Dataset"
    icon = "📂"
    description = "Import data from OpenAlex, Scopus, Web of Science, or other sources"
    requires_data = False
    
    def __init__(self, parent, theme: str = "light", **kwargs):
        self.dataset_path: Optional[str] = None
        self._loading = False
        self._load_result = None
        self._load_error = None
        self._load_complete = False
        self._general_stopwords = []
        self._stopwords_data = {}
        self.stopwords_category_vars = {}
        
        super().__init__(parent, theme=theme, **kwargs)
        self._primary_action = self._load_dataset  # Set primary action for toolbar Run button
        
        # Try to auto-load default stopwords file
        self._try_load_default_stopwords()
    
    def _try_load_default_stopwords(self):
        """Try to automatically load the default stopwords file."""
        import os
        
        # Common locations for the stopwords file
        possible_paths = []
        
        # Also try relative to the biblium package
        try:
            import biblium
            biblium_pkg_dir = os.path.dirname(biblium.__file__)
            possible_paths.extend([
                os.path.join(biblium_pkg_dir, "additional files", "stopwords.xlsx"),
                os.path.join(os.path.dirname(biblium_pkg_dir), "additional files", "stopwords.xlsx"),
            ])
        except:
            pass
        
        # Try current working directory variations
        cwd = os.getcwd()
        possible_paths.extend([
            os.path.join(cwd, "additional files", "stopwords.xlsx"),
            os.path.join(cwd, "biblium", "additional files", "stopwords.xlsx"),
            "additional files/stopwords.xlsx",
        ])
        
        # Try relative to this file
        try:
            this_dir = os.path.dirname(os.path.abspath(__file__))
            biblium_pkg = os.path.dirname(os.path.dirname(os.path.dirname(this_dir)))
            possible_paths.extend([
                os.path.join(biblium_pkg, "additional files", "stopwords.xlsx"),
            ])
        except:
            pass
        
        # Search parent directories
        try:
            search_dir = cwd
            for _ in range(5):
                path = os.path.join(search_dir, "additional files", "stopwords.xlsx")
                if os.path.exists(path):
                    possible_paths.insert(0, path)
                    break
                # Also check biblium subfolder
                path2 = os.path.join(search_dir, "biblium", "additional files", "stopwords.xlsx")
                if os.path.exists(path2):
                    possible_paths.insert(0, path2)
                    break
                parent = os.path.dirname(search_dir)
                if parent == search_dir:
                    break
                search_dir = parent
        except:
            pass
        
        for path in possible_paths:
            if os.path.exists(path):
                self.stopwords_file_var.set(path)
                try:
                    self._load_stopwords_categories()
                    print(f"Loaded stopwords from: {path}")
                except Exception as e:
                    print(f"Failed to load stopwords from {path}: {e}")
                break
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # File Selection Card
        file_card = Card(self.options_content, title="📂 Select File", theme=self.theme_name)
        file_card.pack(fill=tk.X, padx=8, pady=8)
        
        # File path display
        file_frame = tk.Frame(file_card.content, bg=self.theme["bg_card"])
        file_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.file_var = tk.StringVar(value="No file selected")
        self.file_label = tk.Label(
            file_frame,
            textvariable=self.file_var,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_primary"],
            padx=12,
            pady=8,
            anchor=tk.W,
            width=30,
        )
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ThemedButton(
            file_frame,
            text="Browse...",
            style="secondary",
            command=self._browse_file,
            theme=self.theme_name,
        ).pack(side=tk.LEFT, padx=(8, 0))
        
        # Drop zone hint
        tk.Label(
            file_card.content,
            text="Supported: CSV, XLSX, XLS, TXT",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(anchor=tk.W)
        
        # Sample Datasets Card
        sample_card = Card(self.options_content, title="Sample Datasets", theme=self.theme_name)
        sample_card.pack(fill=tk.X, padx=8, pady=8)
        
        tk.Label(
            sample_card.content,
            text="Or select a sample dataset:",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(anchor=tk.W, pady=(0, 4))
        
        # Sample dataset radio buttons - use "none" as default to ensure none selected
        self.sample_var = tk.StringVar(value="none")
        
        samples = [
            ("Scopus (CSV)", "scopus"),
            ("Web of Science (TXT)", "wos"),
            ("OpenAlex (CSV)", "openalex"),
        ]
        
        for label, value in samples:
            rb = tk.Radiobutton(
                sample_card.content,
                text=label,
                variable=self.sample_var,
                value=value,
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
                activebackground=self.theme["bg_card"],
                activeforeground=self.theme["text_primary"],
                selectcolor=self.theme["bg_secondary"],
                highlightthickness=0,
                command=self._on_sample_selected,
            )
            rb.pack(anchor=tk.W, pady=1)
        
        # Data Source Card
        source_card = Card(self.options_content, title="🗄️ Data Source", theme=self.theme_name)
        source_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.db_combo = LabeledCombobox(
            source_card.content,
            label="Database:",
            values=list(DATABASE_OPTIONS.keys()),
            default="Scopus",
            theme=self.theme_name,
            label_width=15,
        )
        self.db_combo.pack(fill=tk.X, pady=4)
        
        # Preprocessing Card
        preprocess_card = Card(self.options_content, title="⚙️ Preprocessing", theme=self.theme_name)
        preprocess_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.preprocess_combo = LabeledCombobox(
            preprocess_card.content,
            label="Level:",
            values=list(PREPROCESS_LEVELS.keys()),
            default="2 - Standard (keywords, text processing)",
            theme=self.theme_name,
            label_width=15,
        )
        self.preprocess_combo.pack(fill=tk.X, pady=4)
        
        # Common options
        self.add_country_cb = LabeledCheckbox(
            preprocess_card.content,
            label="Add country information",
            default=True,
            theme=self.theme_name,
        )
        self.add_country_cb.pack(fill=tk.X, pady=2)
        
        self.process_kw_cb = LabeledCheckbox(
            preprocess_card.content,
            label="Process keywords",
            default=True,
            theme=self.theme_name,
        )
        self.process_kw_cb.pack(fill=tk.X, pady=2)
        
        self.label_docs_cb = LabeledCheckbox(
            preprocess_card.content,
            label="Generate document labels",
            default=True,
            theme=self.theme_name,
        )
        self.label_docs_cb.pack(fill=tk.X, pady=2)
        
        self.fresh_start_cb = LabeledCheckbox(
            preprocess_card.content,
            label="Fresh start (close all tabs)",
            default=True,
            theme=self.theme_name,
        )
        self.fresh_start_cb.pack(fill=tk.X, pady=2)
        
        # Load Button - placed prominently after main options
        self._add_load_button()
        
        # Advanced Options (Collapsible)
        advanced_card = CollapsibleCard(
            self.options_content,
            title="Advanced Options",
            collapsed=True,
            theme=self.theme_name,
        )
        advanced_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.results_folder = LabeledEntry(
            advanced_card.content,
            label="Results Folder:",
            default="results",
            theme=self.theme_name,
            label_width=15,
        )
        self.results_folder.pack(fill=tk.X, pady=4)
        
        self.keywords_combo = LabeledCombobox(
            advanced_card.content,
            label="Default Keywords:",
            values=list(KEYWORD_OPTIONS.keys()),
            default="Author Keywords",
            theme=self.theme_name,
            label_width=15,
        )
        self.keywords_combo.pack(fill=tk.X, pady=4)
        
        self.doc_lang_combo = LabeledCombobox(
            advanced_card.content,
            label="Document Language:",
            values=list(LANGUAGE_OPTIONS.keys()),
            default="English",
            theme=self.theme_name,
            label_width=15,
        )
        self.doc_lang_combo.pack(fill=tk.X, pady=4)
        
        self.lemmatize_cb = LabeledCheckbox(
            advanced_card.content,
            label="Lemmatize keywords",
            default=False,
            theme=self.theme_name,
        )
        self.lemmatize_cb.pack(fill=tk.X, pady=2)
        
        self.combine_kw_cb = LabeledCheckbox(
            advanced_card.content,
            label="Combine author & index keywords",
            default=False,
            theme=self.theme_name,
        )
        self.combine_kw_cb.pack(fill=tk.X, pady=2)
        
        # Visualization Options (Collapsible)
        vis_card = CollapsibleCard(
            self.options_content,
            title="Visualization",
            collapsed=True,
            theme=self.theme_name,
        )
        vis_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.colormap_combo = LabeledCombobox(
            vis_card.content,
            label="Color Palette:",
            values=COLOR_PALETTES,
            default="viridis",
            theme=self.theme_name,
            label_width=15,
        )
        self.colormap_combo.pack(fill=tk.X, pady=4)
        
        self.dpi_spin = LabeledSpinbox(
            vis_card.content,
            label="Plot DPI:",
            from_=72,
            to=600,
            default=600,
            theme=self.theme_name,
            label_width=15,
        )
        self.dpi_spin.pack(fill=tk.X, pady=4)
        
        # Keyword Processing (Collapsible)
        kw_card = CollapsibleCard(
            self.options_content,
            title="Keyword Processing",
            collapsed=True,
            theme=self.theme_name,
        )
        kw_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Stopwords file selection
        stopwords_frame = tk.Frame(kw_card.content, bg=self.theme["bg_card"])
        stopwords_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            stopwords_frame, text="Stopwords File:",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"], width=15, anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        self.stopwords_file_var = tk.StringVar(value="")
        tk.Entry(
            stopwords_frame, textvariable=self.stopwords_file_var,
            font=FONTS.get_font("body"), width=15,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        
        ThemedButton(
            stopwords_frame, text="Browse", style="ghost", size="small",
            command=self._browse_stopwords, theme=self.theme_name,
        ).pack(side=tk.LEFT)
        
        ThemedButton(
            stopwords_frame, text="Load", style="primary", size="small",
            command=self._load_stopwords_categories, theme=self.theme_name,
        ).pack(side=tk.LEFT, padx=(4, 0))
        
        tk.Label(
            kw_card.content,
            text="Excel file with 'specific' sheet containing category columns",
            font=FONTS.get_font("small"), bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(anchor=tk.W)
        
        # Stopwords categories frame (populated when file is loaded)
        self.stopwords_categories_frame = tk.Frame(kw_card.content, bg=self.theme["bg_card"])
        self.stopwords_categories_frame.pack(fill=tk.X, pady=(8, 0))
        
        self.stopwords_category_vars = {}  # Store checkbox variables
        self._stopwords_data = {}  # Store actual stopwords per category
        
        # Placeholder message
        self.stopwords_placeholder = tk.Label(
            self.stopwords_categories_frame,
            text="Load a stopwords file to see available categories",
            font=FONTS.get_font("small"), bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        )
        self.stopwords_placeholder.pack(anchor=tk.W)
        
        ttk.Separator(kw_card.content, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        
        # Synonyms file selection
        synonyms_frame = tk.Frame(kw_card.content, bg=self.theme["bg_card"])
        synonyms_frame.pack(fill=tk.X, pady=(8, 4))
        
        tk.Label(
            synonyms_frame, text="Synonyms File:",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"], width=15, anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        self.synonyms_file_var = tk.StringVar(value="")
        tk.Entry(
            synonyms_frame, textvariable=self.synonyms_file_var,
            font=FONTS.get_font("body"), width=20,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        
        ThemedButton(
            synonyms_frame, text="Browse", style="ghost", size="small",
            command=self._browse_synonyms, theme=self.theme_name,
        ).pack(side=tk.LEFT)
        
        tk.Label(
            kw_card.content,
            text="Excel/CSV file with synonym mappings (term1, term2)",
            font=FONTS.get_font("small"), bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(anchor=tk.W)
        
        # Manual text input (collapsible subsection)
        ttk.Separator(kw_card.content, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        
        tk.Label(
            kw_card.content, text="Or enter manually:",
            font=FONTS.get_font("body", bold=True), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(anchor=tk.W, pady=(0, 4))
        
        self.exclude_keywords = LabeledTextArea(
            kw_card.content,
            label="Keywords to Exclude (one per line):",
            height=3,
            theme=self.theme_name,
        )
        self.exclude_keywords.pack(fill=tk.X, pady=4)
        
        self.synonyms_text = LabeledTextArea(
            kw_card.content,
            label="Synonyms (term1=term2):",
            height=3,
            theme=self.theme_name,
        )
        self.synonyms_text.pack(fill=tk.X, pady=4)
    
    def _add_load_button(self):
        """Add the load button."""
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        self.load_btn = ActionButton(
            btn_frame,
            text="Load Dataset",
            icon="📂",
            command=self._load_dataset,
            theme=self.theme_name,
        )
        self.load_btn.pack(fill=tk.X)
        
        # Status frame
        self.status_frame = tk.Frame(btn_frame, bg=self.theme["bg_secondary"])
        self.status_frame.pack(fill=tk.X, pady=(8, 0))
        
        self.spinner = LoadingSpinner(self.status_frame, theme=self.theme_name)
        self.status_var = tk.StringVar(value="")
        self.status_label = tk.Label(
            self.status_frame,
            textvariable=self.status_var,
            font=FONTS.get_font("small"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_muted"],
        )
    
    def _create_results(self):
        """Create the results panel with data preview."""
        super()._create_results()
        self._show_initial_message("Select a file and click 'Load Dataset' to preview data")
    
    def _browse_file(self):
        """Open file browser."""
        filetypes = [
            ("All Supported", "*.csv *.xlsx *.xls *.txt"),
            ("CSV Files", "*.csv"),
            ("Excel Files", "*.xlsx *.xls"),
            ("Text Files", "*.txt"),
            ("All Files", "*.*"),
        ]
        
        path = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=filetypes,
        )
        
        if path:
            self.dataset_path = path
            filename = os.path.basename(path)
            self.file_var.set(filename)
            
            # Show file info
            size = os.path.getsize(path)
            size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"
            self.status_var.set(f"Selected: {size_str}")
    
    def _load_sample(self, sample_type: str):
        """Load a sample dataset from the data folder.
        
        Parameters
        ----------
        sample_type : str
            One of 'scopus', 'wos', or 'openalex'
        """
        # Define sample file mappings
        sample_files = {
            "scopus": ("scopus dataset.csv", "Scopus"),
            "wos": ("wos dataset.txt", "Web of Science"),
            "openalex": ("open alex dataset.csv", "OpenAlex"),
        }
        
        if sample_type not in sample_files:
            messagebox.showerror("Error", f"Unknown sample type: {sample_type}")
            return
        
        filename, db_name = sample_files[sample_type]
        
        # Find the data folder - try multiple locations
        possible_paths = []
        
        # Try relative to biblium package
        try:
            import biblium
            biblium_pkg_dir = os.path.dirname(biblium.__file__)
            biblium_parent = os.path.dirname(biblium_pkg_dir)
            biblium_grandparent = os.path.dirname(biblium_parent)
            
            possible_paths.extend([
                # Direct data folder in biblium package directory
                os.path.join(biblium_pkg_dir, "data", filename),
                # Data folder next to biblium package
                os.path.join(biblium_parent, "data", filename),
                # Data folder in grandparent (project root)
                os.path.join(biblium_grandparent, "data", filename),
            ])
        except:
            pass
        
        # Try current working directory variations
        cwd = os.getcwd()
        possible_paths.extend([
            os.path.join(cwd, "data", filename),
            os.path.join(cwd, "..", "data", filename),
            os.path.join(cwd, "biblium", "data", filename),
        ])
        
        # Try relative to this file
        try:
            this_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up from gui/panels/data to biblium package
            biblium_pkg = os.path.dirname(os.path.dirname(os.path.dirname(this_dir)))
            project_root = os.path.dirname(biblium_pkg)
            
            possible_paths.extend([
                os.path.join(project_root, "data", filename),
                os.path.join(biblium_pkg, "data", filename),
            ])
        except:
            pass
        
        # Also search parent directories for a data folder
        try:
            search_dir = cwd
            for _ in range(5):  # Go up to 5 levels
                data_path = os.path.join(search_dir, "data", filename)
                if os.path.exists(data_path):
                    possible_paths.insert(0, data_path)
                    break
                parent = os.path.dirname(search_dir)
                if parent == search_dir:  # Reached root
                    break
                search_dir = parent
        except:
            pass
        
        # Find the first existing path
        sample_path = None
        for path in possible_paths:
            normalized = os.path.normpath(path)
            if os.path.exists(normalized):
                sample_path = normalized
                break
        
        if not sample_path:
            # Show helpful error with searched paths
            unique_paths = list(dict.fromkeys(os.path.normpath(p) for p in possible_paths[:8]))
            searched = "\n".join(unique_paths)
            messagebox.showerror(
                "Sample Not Found",
                f"Could not find sample file: {filename}\n\n"
                f"Searched in:\n{searched}\n\n"
                "Please ensure the 'data' folder with sample datasets is present."
            )
            return
        
        # Set the file path and database type
        self.dataset_path = sample_path
        self.file_var.set(f"{filename} (sample)")
        
        # Auto-select the correct database
        self.db_combo.set(db_name)
        
        # Show file info
        size = os.path.getsize(sample_path)
        size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"
        self.status_var.set(f"Sample selected: {size_str}")
    
    def _on_sample_selected(self):
        """Handle sample dataset radio button selection."""
        sample_type = self.sample_var.get()
        if sample_type:
            self._load_sample(sample_type)
    
    def _browse_stopwords(self):
        """Browse for stopwords file."""
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
        """Load stopwords categories from Excel file.
        
        Structure expected:
        - Sheet 'general': Words to always exclude (applied automatically)
        - Sheet 'specific': Has 'Category' column and word columns; 
          unique Category values become checkboxes
        """
        filepath = self.stopwords_file_var.get()
        
        if not filepath:
            messagebox.showwarning("No File", "Please select a stopwords file first.")
            return
        
        try:
            import pandas as pd
            
            # Load general stopwords (always applied)
            self._general_stopwords = []
            try:
                df_general = pd.read_excel(filepath, sheet_name='general')
                # Get all non-null values from the first column (or all columns)
                for col in df_general.columns:
                    values = df_general[col].dropna().astype(str).str.strip().tolist()
                    self._general_stopwords.extend([v for v in values if v and v.lower() != 'nan'])
            except Exception as e:
                print(f"Note: Could not load 'general' sheet: {e}")
            
            # Load specific stopwords with Category column
            self._stopwords_data = {}
            self._specific_df = None
            try:
                df_specific = pd.read_excel(filepath, sheet_name='specific')
                self._specific_df = df_specific
                
                # Find the Category column (case-insensitive)
                category_col = None
                for col in df_specific.columns:
                    if col.lower() == 'category':
                        category_col = col
                        break
                
                if category_col is None:
                    messagebox.showwarning("No Category Column", 
                        "The 'specific' sheet must have a 'Category' column.")
                    return
                
                # Get unique categories
                categories = df_specific[category_col].dropna().unique().tolist()
                categories = [str(c).strip() for c in categories if str(c).strip() and str(c).lower() != 'nan']
                
                # For each category, collect all words from other columns
                word_columns = [col for col in df_specific.columns if col.lower() != 'category']
                
                for cat in categories:
                    # Get rows for this category
                    cat_rows = df_specific[df_specific[category_col].astype(str).str.strip() == cat]
                    
                    # Collect words from all word columns
                    words = []
                    for col in word_columns:
                        col_values = cat_rows[col].dropna().astype(str).str.strip().tolist()
                        words.extend([v for v in col_values if v and v.lower() != 'nan'])
                    
                    self._stopwords_data[cat] = words
                
            except Exception as e:
                print(f"Note: Could not load 'specific' sheet: {e}")
                import traceback
                traceback.print_exc()
                messagebox.showerror("Error", f"Could not load 'specific' sheet: {e}")
                return
            
            # Clear existing checkboxes
            for widget in self.stopwords_categories_frame.winfo_children():
                widget.destroy()
            
            self.stopwords_category_vars = {}
            
            # Show general stopwords info
            if self._general_stopwords:
                tk.Label(
                    self.stopwords_categories_frame,
                    text=f"✓ General stopwords: {len(self._general_stopwords)} terms (always applied)",
                    font=FONTS.get_font("small"),
                    bg=self.theme["bg_card"], fg=self.theme["success"],
                ).pack(anchor=tk.W, pady=(0, 4))
            
            # Create header for specific categories
            if self._stopwords_data:
                tk.Label(
                    self.stopwords_categories_frame,
                    text=f"Select specific categories ({len(self._stopwords_data)} available):",
                    font=FONTS.get_font("body", bold=True),
                    bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                ).pack(anchor=tk.W, pady=(4, 4))
                
                # Create checkboxes for each category
                for cat in sorted(self._stopwords_data.keys()):
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
            else:
                tk.Label(
                    self.stopwords_categories_frame,
                    text="No categories found in 'specific' sheet",
                    font=FONTS.get_font("small"),
                    bg=self.theme["bg_card"], fg=self.theme["text_muted"],
                ).pack(anchor=tk.W)
            
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
    
    def _get_selected_stopwords(self):
        """Get list of all stopwords: general (always) + selected specific categories."""
        stopwords = []
        
        # Always include general stopwords
        if hasattr(self, '_general_stopwords') and self._general_stopwords:
            stopwords.extend(self._general_stopwords)
        
        # Add stopwords from selected specific categories
        if hasattr(self, '_stopwords_data'):
            for cat, var in self.stopwords_category_vars.items():
                if var.get():
                    stopwords.extend(self._stopwords_data.get(cat, []))
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for sw in stopwords:
            sw_lower = sw.lower()
            if sw_lower not in seen:
                seen.add(sw_lower)
                unique.append(sw)
        
        return unique
    
    def _browse_synonyms(self):
        """Browse for synonyms file."""
        filetypes = [
            ("Excel Files", "*.xlsx *.xls"),
            ("CSV Files", "*.csv"),
            ("Text Files", "*.txt"),
            ("All Files", "*.*"),
        ]
        
        path = filedialog.askopenfilename(
            title="Select Synonyms File",
            filetypes=filetypes,
        )
        
        if path:
            self.synonyms_file_var.set(path)
    
    def _load_dataset(self):
        """Load the dataset."""
        if not self.dataset_path:
            messagebox.showwarning("No File", "Please select a file first.")
            return
        
        if self._loading:
            return
        
        self._loading = True
        self.load_btn.set_enabled(False)
        
        self.spinner.pack(side=tk.LEFT, padx=(0, 8))
        self.spinner.start()
        self.status_label.pack(side=tk.LEFT)
        self.status_var.set("Loading dataset...")
        
        # Show loading in results
        self._show_loading("Loading and processing dataset...")
        
        # Get options
        db = DATABASE_OPTIONS.get(self.db_combo.get(), "")  # Default to auto-detect
        preprocess = PREPROCESS_LEVELS.get(self.preprocess_combo.get(), 2)
        
        # Build kwargs
        kwargs = {
            "f_name": self.dataset_path,
            "db": db,
            "preprocess_level": preprocess,
            "res_folder": self.results_folder.get() or "results",
            "default_keywords": KEYWORD_OPTIONS.get(self.keywords_combo.get(), "author"),
            "lang_of_docs": LANGUAGE_OPTIONS.get(self.doc_lang_combo.get(), "en"),
            "lemmatize_kw": self.lemmatize_cb.get(),
            "combine_with_index_keywords": self.combine_kw_cb.get(),
            "label_docs": self.label_docs_cb.get(),
            "cmap": self.colormap_combo.get(),
            "dpi": self.dpi_spin.get(),
        }
        
        # Exclusion list - combine manual entries with selected stopwords categories
        exclude_list = []
        
        # From manual text input
        exclude = self.exclude_keywords.get()
        if exclude:
            exclude_list.extend([x.strip() for x in exclude.split("\n") if x.strip()])
        
        # From selected stopwords categories
        selected_stopwords = self._get_selected_stopwords()
        if selected_stopwords:
            exclude_list.extend(selected_stopwords)
        
        # Remove duplicates while preserving order
        if exclude_list:
            seen = set()
            unique_exclude = []
            for item in exclude_list:
                item_lower = item.lower()
                if item_lower not in seen:
                    seen.add(item_lower)
                    unique_exclude.append(item)
            kwargs["exclude_list_kw"] = unique_exclude
        
        # Synonyms
        synonyms = self.synonyms_text.get()
        if synonyms:
            syn_dict = {}
            for line in synonyms.split("\n"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    syn_dict[k.strip()] = v.strip()
            if syn_dict:
                kwargs["synonyms_kw"] = syn_dict
        
        def do_load():
            try:
                # Check if stop was requested before starting
                print(f"[DEBUG] Starting load...", flush=True)
                print(f"[DEBUG] File: {kwargs.get('f_name')}", flush=True)
                print(f"[DEBUG] DB: {kwargs.get('db')}", flush=True)
                print(f"[DEBUG] Preprocess level: {kwargs.get('preprocess_level')}", flush=True)
                exclude_list = kwargs.get('exclude_list_kw', [])
                print(f"[DEBUG] Exclude list size: {len(exclude_list)}", flush=True)
                
                # If exclude list is very large, warn user
                if len(exclude_list) > 500:
                    print(f"[DEBUG] WARNING: Large exclude list ({len(exclude_list)} items) may slow loading", flush=True)
                
                import time
                start = time.time()
                
                # Always use BiblioAnalysis (includes all features from BiblioPlot)
                bib = None
                try:
                    print("[DEBUG] Creating BiblioAnalysis object...", flush=True)
                    bib = BiblioAnalysis(**kwargs)
                    elapsed = time.time() - start
                    print(f"[DEBUG] BiblioAnalysis succeeded in {elapsed:.2f}s, n={bib.n}", flush=True)
                except Exception as e:
                    elapsed = time.time() - start
                    print(f"[DEBUG] BiblioAnalysis failed after {elapsed:.2f}s: {type(e).__name__}: {e}", flush=True)
                    raise e
                
                if bib is None:
                    raise ValueError("Failed to create bibliometric object")
                
                print("[DEBUG] Copying dataframe...", flush=True)
                original_df = bib.df.copy()
                print(f"[DEBUG] DataFrame copied, shape: {original_df.shape}", flush=True)
                
                print("[DEBUG] Scheduling success callback...", flush=True)
                # Store results for pickup by main thread
                self._load_result = (bib, original_df)
                self._load_error = None
                
                # Try to schedule callback
                try:
                    if self.winfo_exists():
                        self.after(0, self._process_load_result)
                        print("[DEBUG] Success callback scheduled", flush=True)
                    else:
                        print("[DEBUG] Widget no longer exists", flush=True)
                except RuntimeError as e:
                    print(f"[DEBUG] RuntimeError scheduling callback: {e}", flush=True)
                    # Set a flag so polling can pick it up
                    self._load_complete = True
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                error_str = str(e)
                print(f"[DEBUG] Load failed with error: {error_str}", flush=True)
                print(f"[DEBUG] Traceback: {tb}", flush=True)
                # Store error for pickup
                self._load_error = (error_str, tb)
                self._load_result = None
                try:
                    self.after(0, self._process_load_result)
                except RuntimeError:
                    print("[DEBUG] RuntimeError scheduling error callback", flush=True)
                    self._load_complete = True
        
        print("[DEBUG] Starting load thread...", flush=True)
        t = threading.Thread(target=do_load, daemon=True)
        t.start()
        print("[DEBUG] Load thread started", flush=True)
        
        # Start polling for result in case callback fails
        self._poll_load_result()
    
    def _poll_load_result(self):
        """Poll for load result if callback scheduling failed."""
        if hasattr(self, '_load_complete') and self._load_complete:
            self._load_complete = False
            self._process_load_result()
        elif self._loading:
            # Keep polling while loading
            self.after(100, self._poll_load_result)
    
    def _process_load_result(self):
        """Process load result from background thread."""
        if hasattr(self, '_load_result') and self._load_result:
            bib, original_df = self._load_result
            self._load_result = None
            self._on_load_success(bib, original_df)
        elif hasattr(self, '_load_error') and self._load_error:
            error_str, tb = self._load_error
            self._load_error = None
            self._on_load_error(error_str, tb)
    
    def _on_load_success(self, bib, original_df):
        """Handle successful load."""
        print(f"[DEBUG] _on_load_success called with bib.n={bib.n}", flush=True)
        
        self._loading = False
        
        try:
            self.load_btn.set_enabled(True)
        except Exception as e:
            print(f"[DEBUG] Error enabling load button: {e}")
            
        try:
            self.spinner.stop()
            self.spinner.pack_forget()
        except Exception as e:
            print(f"[DEBUG] Error stopping spinner: {e}")
        
        self.bib = bib
        
        # Get year range
        year_col = bib.mapping.get("Year", "Year")
        year_range = (None, None)
        if year_col in bib.df.columns:
            years = bib.df[year_col].dropna()
            if len(years) > 0:
                year_range = (int(years.min()), int(years.max()))
        
        try:
            self.status_var.set(f"✓ Loaded {bib.n:,} documents")
        except Exception as e:
            print(f"[DEBUG] Error setting status: {e}")
        
        # Fresh start - close all other tabs
        print(f"[DEBUG] Fresh start checkbox: {self.fresh_start_cb.get()}")
        if self.fresh_start_cb.get():
            print("[DEBUG] Emitting RESET_PANELS event...")
            try:
                event_bus.emit(EventBus.RESET_PANELS, {})
                print("[DEBUG] RESET_PANELS emitted successfully")
            except Exception as e:
                print(f"[DEBUG] Error emitting RESET_PANELS: {e}")
                import traceback
                traceback.print_exc()
        
        # Emit dataset loaded event
        print("[DEBUG] Emitting DATASET_LOADED event...")
        try:
            event_bus.emit(EventBus.DATASET_LOADED, {
                "bib": bib,
                "original_df": original_df,
                "path": self.dataset_path,
                "n_documents": bib.n,
                "database": bib.db,
                "year_range": year_range,
            })
            print("[DEBUG] DATASET_LOADED emitted successfully")
        except Exception as e:
            print(f"[DEBUG] Error emitting DATASET_LOADED: {e}")
            import traceback
            traceback.print_exc()
        
        # Show preview - schedule to allow event processing to complete
        print("[DEBUG] Scheduling preview...")
        self.after(100, lambda b=bib: self._show_preview(b))
        
        print("[DEBUG] Showing success messagebox...")
        messagebox.showinfo("Success", f"Successfully loaded {bib.n:,} documents from {bib.db.upper()}")
        print("[DEBUG] _on_load_success completed")
    
    def _on_load_error(self, error: str, traceback_str: str = ""):
        """Handle load error."""
        self._loading = False
        
        try:
            self.load_btn.set_enabled(True)
        except:
            pass
        
        try:
            self.spinner.stop()
            self.spinner.pack_forget()
        except:
            pass
        
        try:
            self.status_var.set(f"✗ Failed to load")
        except:
            pass
        
        self._show_error(f"Failed to load dataset:\n{error}")
        
        event_bus.emit(EventBus.ERROR_OCCURRED, {"message": error})
        
        print(f"Load error: {error}")
        if traceback_str:
            print(traceback_str)
    
    def _show_preview(self, bib):
        """Show data preview in results area."""
        # Close any open matplotlib figures first
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass
        
        # Safe cleanup of existing widgets - do this in stages with idle tasks
        try:
            children = list(self.results_content.winfo_children())
            for widget in children:
                try:
                    widget.destroy()
                except Exception:
                    pass
            # Allow destruction to complete
            self.update_idletasks()
        except Exception:
            pass
        
        try:
            # Summary stats
            stats_frame = tk.Frame(self.results_content, bg=self.theme["bg_card"])
            stats_frame.pack(fill=tk.X, pady=(0, 12))
            
            from biblium.gui.widgets.cards import StatsCard, CardGrid
            
            grid = CardGrid(stats_frame, columns=4, theme=self.theme_name)
            grid.pack(fill=tk.X)
            
            grid.add_card(StatsCard(grid, "Documents", f"{bib.n:,}", "📄", self.theme_name))
            grid.add_card(StatsCard(grid, "Columns", f"{len(bib.df.columns)}", "📋", self.theme_name))
            grid.add_card(StatsCard(grid, "Database", bib.db.upper(), "🗄️", self.theme_name))
            
            year_col = bib.mapping.get("Year", "Year")
            if year_col in bib.df.columns:
                years = bib.df[year_col].dropna()
                if len(years) > 0:
                    year_range = f"{int(years.min())}-{int(years.max())}"
                    grid.add_card(StatsCard(grid, "Years", year_range, "📅", self.theme_name))
            
            # Data table preview
            tk.Label(
                self.results_content,
                text="Data Preview",
                font=FONTS.get_font("heading", bold=True),
                bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
                anchor=tk.W,
            ).pack(fill=tk.X, pady=(8, 4))
            
            table = DataTable(
                self.results_content,
                theme=self.theme_name,
                max_rows=100,
            )
            table.pack(fill=tk.BOTH, expand=True)
            table.set_data(bib.df)
        except Exception as e:
            print(f"Error showing preview: {e}")
