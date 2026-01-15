# -*- coding: utf-8 -*-
"""
API Data Panel
==============
Panel for fetching bibliometric data from OpenAlex and Dimensions APIs.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import json
import os
from typing import Optional, Dict, List
from datetime import datetime

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, CollapsibleCard, StatsCard, CardGrid
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import (
    LabeledEntry, LabeledCombobox, LabeledCheckbox, 
    LabeledSpinbox, LabeledTextArea,
)
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.progress import LoadingSpinner

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class APIDataPanel(BasePanel):
    """Panel for fetching data from OpenAlex and Dimensions APIs."""
    
    title = "API Data"
    icon = "🌐"
    description = "Fetch bibliometric data from OpenAlex or Dimensions APIs"
    requires_data = False
    
    def __init__(self, parent, theme: str = "light", **kwargs):
        self._fetched_data = None
        self._fetching = False
        self._general_stopwords = []
        self._stopwords_data = {}
        self.stopwords_category_vars = {}
        
        super().__init__(parent, theme=theme, **kwargs)
        
        # Try to auto-load default stopwords file
        self._try_load_default_stopwords()
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # API Selection Card
        api_card = Card(self.options_content, title="🔌 API Source", theme=self.theme_name)
        api_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.api_combo = LabeledCombobox(
            api_card.content,
            label="API:",
            values=["OpenAlex (Free)", "PubMed (Free)", "Dimensions (API Key Required)"],
            default="OpenAlex (Free)",
            theme=self.theme_name,
            label_width=12,
        )
        self.api_combo.pack(fill=tk.X, pady=4)
        self.api_combo.combo.bind("<<ComboboxSelected>>", self._on_api_changed)
        
        # API Key Frame (for Dimensions)
        self.api_key_frame = tk.Frame(api_card.content, bg=self.theme["bg_card"])
        self.api_key_frame.pack(fill=tk.X, pady=4)
        
        # Create password entry manually (LabeledEntry doesn't support show parameter)
        api_key_label = tk.Label(
            self.api_key_frame, text="API Key:", font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            width=12, anchor=tk.W,
        )
        api_key_label.pack(side=tk.LEFT)
        
        self.api_key_var = tk.StringVar()
        self.api_key_entry = tk.Entry(
            self.api_key_frame, textvariable=self.api_key_var,
            font=FONTS.get_font("body"), width=25, show="*",
            relief=tk.FLAT, bg=self.theme["bg_input"],
            fg=self.theme["text_primary"],
            highlightbackground=self.theme["border"],
            highlightthickness=1,
        )
        self.api_key_entry.pack(side=tk.LEFT, padx=(4, 0))
        
        self.api_key_frame.pack_forget()  # Hide by default (OpenAlex is free)
        
        # Email for OpenAlex/PubMed (polite pool)
        self.email_frame = tk.Frame(api_card.content, bg=self.theme["bg_card"])
        self.email_frame.pack(fill=tk.X, pady=4)
        
        self.email_entry = LabeledEntry(
            self.email_frame,
            label="Email:",
            placeholder="your@email.com (for polite API access)",
            theme=self.theme_name,
            label_width=12,
        )
        self.email_entry.pack(fill=tk.X)
        
        # Search Query Card
        query_card = Card(self.options_content, title="🔍 Search Query", theme=self.theme_name)
        query_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.search_type = LabeledCombobox(
            query_card.content,
            label="Search By:",
            values=["Keywords", "Author", "Institution", "ISSN/Journal", "DOI List"],
            default="Keywords",
            theme=self.theme_name,
            label_width=12,
        )
        self.search_type.pack(fill=tk.X, pady=4)
        
        self.query_entry = LabeledEntry(
            query_card.content,
            label="Query:",
            placeholder="e.g., machine learning, bibliometrics",
            theme=self.theme_name,
            label_width=12,
        )
        self.query_entry.pack(fill=tk.X, pady=4)
        
        # Filters Card
        filter_card = Card(self.options_content, title="🎯 Filters", theme=self.theme_name)
        filter_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Year range
        year_frame = tk.Frame(filter_card.content, bg=self.theme["bg_card"])
        year_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            year_frame, text="Years:", font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"], width=12, anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        current_year = datetime.now().year
        self.year_from_spin = tk.Spinbox(
            year_frame, from_=1900, to=current_year, width=6,
            font=FONTS.get_font("body"),
        )
        self.year_from_spin.delete(0, tk.END)
        self.year_from_spin.insert(0, str(current_year - 10))
        self.year_from_spin.pack(side=tk.LEFT)
        
        tk.Label(
            year_frame, text=" to ", font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT)
        
        self.year_to_spin = tk.Spinbox(
            year_frame, from_=1900, to=current_year, width=6,
            font=FONTS.get_font("body"),
        )
        self.year_to_spin.delete(0, tk.END)
        self.year_to_spin.insert(0, str(current_year))
        self.year_to_spin.pack(side=tk.LEFT)
        
        # Document type
        self.doc_type_combo = LabeledCombobox(
            filter_card.content,
            label="Doc Type:",
            values=["All", "Article", "Review", "Book Chapter", "Conference Paper", "Preprint"],
            default="All",
            theme=self.theme_name,
            label_width=12,
        )
        self.doc_type_combo.pack(fill=tk.X, pady=4)
        
        # Max results
        self.max_results_spin = LabeledSpinbox(
            filter_card.content,
            label="Max Results:",
            from_=10,
            to=10000,
            default=200,
            theme=self.theme_name,
            label_width=12,
        )
        self.max_results_spin.pack(fill=tk.X, pady=4)
        
        # Open Access filter
        self.open_access_cb = LabeledCheckbox(
            filter_card.content,
            label="Open Access only",
            default=False,
            theme=self.theme_name,
        )
        self.open_access_cb.pack(fill=tk.X, pady=2)
        
        # Processing Options Card
        from biblium.gui.config import PREPROCESS_LEVELS
        
        process_card = Card(self.options_content, title="⚙️ Processing", theme=self.theme_name)
        process_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.preprocess_combo = LabeledCombobox(
            process_card.content,
            label="Level:",
            values=list(PREPROCESS_LEVELS.keys()),
            default="2 - Standard (keywords, text processing)",
            theme=self.theme_name,
            label_width=12,
        )
        self.preprocess_combo.pack(fill=tk.X, pady=4)
        
        # Auto-load checkbox
        self.auto_load_var = tk.BooleanVar(value=True)
        auto_load_cb = tk.Checkbutton(
            process_card.content,
            text="Auto-load into Biblium after fetch",
            variable=self.auto_load_var,
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            activebackground=self.theme["bg_card"],
            selectcolor=self.theme["bg_card"],
            font=FONTS.get_font("body"),
        )
        auto_load_cb.pack(anchor=tk.W, pady=(8, 4))
        
        # Keyword Processing Card (Collapsible)
        from biblium.gui.widgets.cards import CollapsibleCard
        
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
            fg=self.theme["text_primary"], width=12, anchor=tk.W,
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
            stopwords_frame, text="Apply", style="primary", size="small",
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
        
        # Placeholder message
        self.stopwords_placeholder = tk.Label(
            self.stopwords_categories_frame,
            text="Load a stopwords file to see available categories",
            font=FONTS.get_font("small"), bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        )
        self.stopwords_placeholder.pack(anchor=tk.W)
        
        # Action Buttons
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Fetch Data", icon="🔍",
            command=self._fetch_data, theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
        
        # Manual load button (for when auto-load is disabled)
        self.load_btn = ActionButton(
            btn_frame, text="Load into Biblium", icon="📥",
            command=self._load_into_biblium, theme=self.theme_name,
        )
        self.load_btn.pack(fill=tk.X, pady=4)
    
    def _try_load_default_stopwords(self):
        """Try to automatically load the default stopwords file."""
        import os
        
        # Common locations for the stopwords file
        possible_paths = [
            "additional files/stopwords.xlsx",
            "additional files\\stopwords.xlsx",
            os.path.join("additional files", "stopwords.xlsx"),
        ]
        
        # Add paths relative to biblium package
        try:
            import biblium
            biblium_dir = os.path.dirname(biblium.__file__)
            possible_paths.extend([
                os.path.join(biblium_dir, "additional files", "stopwords.xlsx"),
                os.path.join(biblium_dir, "..", "additional files", "stopwords.xlsx"),
                os.path.join(os.path.dirname(biblium_dir), "additional files", "stopwords.xlsx"),
            ])
        except:
            pass
        
        # Also check current working directory
        cwd = os.getcwd()
        possible_paths.append(os.path.join(cwd, "additional files", "stopwords.xlsx"))
        
        for path in possible_paths:
            if os.path.exists(path):
                self.stopwords_file_var.set(path)
                try:
                    self._load_stopwords_categories()
                except:
                    pass
                break
    
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
            self._load_stopwords_categories()
    
    def _load_stopwords_categories(self):
        """Load stopwords categories from Excel file."""
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
                for col in df_general.columns:
                    values = df_general[col].dropna().astype(str).str.strip().tolist()
                    self._general_stopwords.extend([v for v in values if v and v.lower() != 'nan'])
            except Exception as e:
                print(f"Note: Could not load 'general' sheet: {e}")
            
            # Load specific stopwords with Category column
            self._stopwords_data = {}
            try:
                df_specific = pd.read_excel(filepath, sheet_name='specific')
                
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
                    cat_rows = df_specific[df_specific[category_col].astype(str).str.strip() == cat]
                    words = []
                    for col in word_columns:
                        col_values = cat_rows[col].dropna().astype(str).str.strip().tolist()
                        words.extend([v for v in col_values if v and v.lower() != 'nan'])
                    self._stopwords_data[cat] = words
                
            except Exception as e:
                print(f"Note: Could not load 'specific' sheet: {e}")
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

    def _on_api_changed(self, event=None):
        """Handle API selection change."""
        api = self.api_combo.get()
        
        if "Dimensions" in api:
            self.api_key_frame.pack(fill=tk.X, pady=4, after=self.api_combo)
            self.email_frame.pack_forget()
        else:
            self.api_key_frame.pack_forget()
            self.email_frame.pack(fill=tk.X, pady=4, after=self.api_combo)
    
    def _create_results(self):
        """Create the results panel."""
        self.results_card = tk.Frame(
            self.results_frame, bg=self.theme["bg_card"],
            highlightbackground=self.theme["border"], highlightthickness=1,
        )
        self.results_card.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        # Header
        header = tk.Frame(self.results_card, bg=self.theme["bg_card"])
        header.pack(fill=tk.X)
        
        tk.Label(
            header, text="Fetched Data",
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            padx=12, pady=8,
        ).pack(side=tk.LEFT)
        
        # Export button
        ThemedButton(
            header, text="💾 Save Data", style="ghost", size="small",
            command=self._save_csv, theme=self.theme_name,
        ).pack(side=tk.RIGHT, padx=8, pady=4)
        
        ttk.Separator(self.results_card, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12)
        
        # Results content
        self.results_content = tk.Frame(self.results_card, bg=self.theme["bg_card"])
        self.results_content.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        
        self._show_initial_message()
    
    def _show_initial_message(self):
        """Show initial message."""
        for widget in self.results_content.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.results_content,
            text="Enter search query and click 'Fetch Data' to retrieve publications",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_muted"],
        ).pack(expand=True, pady=50)
    
    def _update_loading(self, message: str):
        """Update the loading message without recreating the loading UI."""
        # Find and update the label in results_content
        for widget in self.results_content.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Label):
                        child.config(text=message)
                        return
    
    def _fetch_data(self):
        """Fetch data from selected API."""
        if self._fetching:
            messagebox.showwarning("Busy", "Already fetching data...")
            return
        
        if not HAS_REQUESTS:
            messagebox.showerror("Error", "requests library not installed.\nInstall with: pip install requests")
            return
        
        query = self.query_entry.get().strip()
        if not query:
            messagebox.showwarning("No Query", "Please enter a search query.")
            return
        
        api = self.api_combo.get()
        
        # Check API key for Dimensions
        if "Dimensions" in api:
            api_key = self.api_key_var.get().strip()
            if not api_key:
                messagebox.showwarning("API Key Required", "Please enter your Dimensions API key.")
                return
        
        self._fetching = True
        self._show_loading("Fetching data from API...")
        
        # Capture all values from UI before entering thread
        search_type = self.search_type.get()
        year_from = int(self.year_from_spin.get())
        year_to = int(self.year_to_spin.get())
        max_results = self.max_results_spin.get()
        doc_type = self.doc_type_combo.get()
        open_access = self.open_access_cb.get()
        email = self.email_entry.get().strip()
        api_key = self.api_key_var.get().strip() if "Dimensions" in api else ""
        
        # Store params for use in fetch methods
        self._fetch_params = {
            "query": query,
            "search_type": search_type,
            "year_from": year_from,
            "year_to": year_to,
            "max_results": max_results,
            "doc_type": doc_type,
            "open_access": open_access,
            "email": email,
            "api_key": api_key,
        }
        
        def do_fetch():
            try:
                if "OpenAlex" in api:
                    result = self._fetch_openalex()
                elif "PubMed" in api:
                    result = self._fetch_pubmed()
                else:
                    result = self._fetch_dimensions()
                
                self._fetching = False
                # Use default argument to capture result
                self.after(0, lambda r=result: self._on_fetch_success(r))
            except Exception as e:
                self._fetching = False
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: self._on_fetch_error(msg))
        
        threading.Thread(target=do_fetch, daemon=True).start()
    
    def _fetch_openalex(self) -> pd.DataFrame:
        """Fetch data from OpenAlex API."""
        # Use stored params (captured from UI in main thread)
        params = self._fetch_params
        query = params["query"]
        search_type = params["search_type"]
        year_from = params["year_from"]
        year_to = params["year_to"]
        max_results = params["max_results"]
        doc_type = params["doc_type"]
        open_access = params["open_access"]
        email = params["email"]
        
        # Build OpenAlex API URL
        base_url = "https://api.openalex.org/works"
        
        # Build filter
        filters = []
        
        # Search filter based on type
        if search_type == "Keywords":
            filters.append(f"title_and_abstract.search:{query}")
        elif search_type == "Author":
            # OpenAlex requires 2-step process: first find author ID, then filter works
            self.after(0, lambda: self._update_loading("Searching for author..."))
            author_id = self._find_openalex_author_id(query, email)
            if author_id:
                filters.append(f"authorships.author.id:{author_id}")
            else:
                raise ValueError(f"No author found matching '{query}'")
        elif search_type == "Institution":
            # OpenAlex requires 2-step process: first find institution ID, then filter works
            self.after(0, lambda: self._update_loading("Searching for institution..."))
            institution_id = self._find_openalex_institution_id(query, email)
            if institution_id:
                filters.append(f"authorships.institutions.id:{institution_id}")
            else:
                raise ValueError(f"No institution found matching '{query}'")
        elif search_type == "ISSN/Journal":
            filters.append(f"primary_location.source.issn:{query}")
        elif search_type == "DOI List":
            # Handle multiple DOIs
            dois = [d.strip() for d in query.replace(",", "|").split("|")]
            doi_filter = "|".join(dois)
            filters.append(f"doi:{doi_filter}")
        
        # Year filter
        filters.append(f"publication_year:{year_from}-{year_to}")
        
        # Document type filter
        type_map = {
            "Article": "article",
            "Review": "review",
            "Book Chapter": "book-chapter",
            "Conference Paper": "proceedings-article",
            "Preprint": "preprint",
        }
        if doc_type != "All" and doc_type in type_map:
            filters.append(f"type:{type_map[doc_type]}")
        
        # Open Access filter
        if open_access:
            filters.append("is_oa:true")
        
        filter_str = ",".join(filters)
        
        # Build request params
        request_params = {
            "filter": filter_str,
            "per_page": min(200, max_results),  # OpenAlex max is 200 per page
            "sort": "cited_by_count:desc",
        }
        
        if email:
            request_params["mailto"] = email
        
        # Create session for connection reuse (faster for multiple requests)
        session = requests.Session()
        session.headers.update({
            "Accept": "application/json",
            "User-Agent": "Biblium/2.8",
        })
        
        # Fetch data with pagination
        all_results = []
        cursor = "*"
        page = 0
        
        try:
            while len(all_results) < max_results:
                page += 1
                request_params["cursor"] = cursor
                
                # Update progress in UI thread
                self.after(0, lambda p=page, c=len(all_results), m=max_results: 
                          self._update_loading(f"Fetching page {p}... ({c}/{m} records)"))
                
                response = session.get(base_url, params=request_params, timeout=60)
                response.raise_for_status()
                data = response.json()
                
                results = data.get("results", [])
                if not results:
                    break
                
                all_results.extend(results)
                
                # Get next cursor
                meta = data.get("meta", {})
                cursor = meta.get("next_cursor")
                if not cursor:
                    break
        finally:
            session.close()
        
        # Limit to max_results
        all_results = all_results[:max_results]
        
        # Update progress for conversion
        self.after(0, lambda n=len(all_results): self._update_loading(f"Converting {n} records..."))
        
        # Convert to DataFrame
        df = self._openalex_to_dataframe(all_results)
        
        return df
    
    def _find_openalex_author_id(self, query: str, email: str = None) -> str:
        """Find OpenAlex author ID by searching for author name."""
        url = "https://api.openalex.org/authors"
        params = {"search": query}
        if email:
            params["mailto"] = email
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", [])
        if results:
            # Return the first (best match) author's ID
            return results[0].get("id", "").replace("https://openalex.org/", "")
        return None
    
    def _find_openalex_institution_id(self, query: str, email: str = None) -> str:
        """Find OpenAlex institution ID by searching for institution name."""
        url = "https://api.openalex.org/institutions"
        params = {"search": query}
        if email:
            params["mailto"] = email
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", [])
        if results:
            # Return the first (best match) institution's ID
            return results[0].get("id", "").replace("https://openalex.org/", "")
        return None
    
    def _openalex_to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Convert OpenAlex results to Scopus-compatible DataFrame."""
        records = []
        
        for work in results:
            # Extract authors
            authors = []
            author_ids = []
            affiliations = []
            countries = []
            is_corresponding = []
            
            for authorship in work.get("authorships", []) or []:
                author = authorship.get("author", {}) or {}
                author_name = author.get("display_name", "") or ""
                author_id = (author.get("id") or "").replace("https://openalex.org/", "")
                
                if author_name:
                    authors.append(author_name)
                    author_ids.append(author_id)
                
                # Extract countries from authorship
                author_countries = authorship.get("countries", []) or []
                for country_code in author_countries:
                    if country_code and country_code not in countries:
                        countries.append(country_code)
                
                # Track is_corresponding for each authorship's countries
                is_corr = authorship.get("is_corresponding", False)
                for _ in author_countries:
                    is_corresponding.append(str(is_corr).lower())
                
                for inst in authorship.get("institutions", []) or []:
                    if inst:
                        inst_name = inst.get("display_name") or ""
                        if inst_name and inst_name not in affiliations:
                            affiliations.append(inst_name)
                        # Also get country from institution
                        inst_country = inst.get("country_code") or ""
                        if inst_country and inst_country not in countries:
                            countries.append(inst_country)
                            is_corresponding.append(str(is_corr).lower())
            
            # Extract source info
            primary_loc = work.get("primary_location") or {}
            source = primary_loc.get("source") or {}
            source_title = (source.get("display_name") or "") if source else ""
            issn = (source.get("issn_l") or "") if source else ""
            
            # Extract keywords (only top concepts)
            keywords = []
            for concept in (work.get("concepts") or [])[:15]:  # Limit to first 15
                if concept and concept.get("score", 0) > 0.3:
                    kw = concept.get("display_name")
                    if kw:
                        keywords.append(kw)
            
            # Extract abstract - optimized reconstruction
            abstract = ""
            abstract_inverted = work.get("abstract_inverted_index")
            if abstract_inverted:
                # Pre-allocate list based on maximum position
                try:
                    max_pos = max(max(positions) for positions in abstract_inverted.values() if positions)
                    words = [""] * (max_pos + 1)
                    for word, positions in abstract_inverted.items():
                        if positions:
                            for pos in positions:
                                words[pos] = word
                    abstract = " ".join(w for w in words if w)
                except (ValueError, TypeError):
                    # Fallback if something goes wrong
                    abstract = ""
            
            # Build record with safe null handling
            work_id = (work.get("id") or "").replace("https://openalex.org/", "")
            work_doi = work.get("doi") or ""
            if work_doi:
                work_doi = work_doi.replace("https://doi.org/", "")
            
            # Ensure is_corresponding has same length as countries
            while len(is_corresponding) < len(countries):
                is_corresponding.append("false")
            
            # Convert country codes to full names
            country_names = [self._country_code_to_name(c) for c in countries]
            
            record = {
                "Authors": "; ".join(authors),
                "Author(s) ID": "; ".join(author_ids),
                "Title": work.get("title") or "",
                "Year": work.get("publication_year"),
                "Source title": source_title,
                "Cited by": work.get("cited_by_count") or 0,
                "DOI": work_doi,
                "Abstract": abstract[:5000] if abstract else "",
                "Author Keywords": "; ".join(keywords[:10]),
                "Affiliations": "; ".join(affiliations),
                "Countries of Authors": "; ".join(country_names),  # Scopus-compatible country column
                "Corresponding Author Country": country_names[0] if country_names and any(c == "true" for c in is_corresponding) else (country_names[0] if country_names else ""),
                "Document Type": work.get("type") or "article",
                "ISSN": issn,
                "Open Access": "Yes" if work.get("is_oa") else "No",
                "OpenAlex ID": work_id,
                "Language": work.get("language") or "",
            }
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def _country_code_to_name(self, code: str) -> str:
        """Convert ISO 2-letter country code to full country name."""
        country_map = {
            "US": "United States", "GB": "United Kingdom", "CN": "China",
            "DE": "Germany", "FR": "France", "JP": "Japan", "CA": "Canada",
            "AU": "Australia", "IT": "Italy", "ES": "Spain", "NL": "Netherlands",
            "BR": "Brazil", "IN": "India", "KR": "South Korea", "CH": "Switzerland",
            "SE": "Sweden", "BE": "Belgium", "AT": "Austria", "PL": "Poland",
            "DK": "Denmark", "NO": "Norway", "FI": "Finland", "PT": "Portugal",
            "GR": "Greece", "IL": "Israel", "SG": "Singapore", "TW": "Taiwan",
            "RU": "Russia", "MX": "Mexico", "AR": "Argentina", "TR": "Turkey",
            "IR": "Iran", "SA": "Saudi Arabia", "EG": "Egypt", "ZA": "South Africa",
            "NZ": "New Zealand", "IE": "Ireland", "CZ": "Czech Republic",
            "HU": "Hungary", "SI": "Slovenia", "MY": "Malaysia", "TH": "Thailand",
            "ID": "Indonesia", "PH": "Philippines", "VN": "Vietnam", "PK": "Pakistan",
            "CL": "Chile", "CO": "Colombia", "PE": "Peru", "UA": "Ukraine",
            "RO": "Romania", "SK": "Slovakia", "HR": "Croatia", "RS": "Serbia",
            "BG": "Bulgaria", "LT": "Lithuania", "LV": "Latvia", "EE": "Estonia",
            "CY": "Cyprus", "MT": "Malta", "LU": "Luxembourg", "IS": "Iceland",
            "HK": "Hong Kong", "AE": "United Arab Emirates", "QA": "Qatar",
            "KW": "Kuwait", "NG": "Nigeria", "KE": "Kenya", "GH": "Ghana",
            "ET": "Ethiopia", "TZ": "Tanzania", "UG": "Uganda", "MA": "Morocco",
            "TN": "Tunisia", "DZ": "Algeria", "BD": "Bangladesh", "LK": "Sri Lanka",
            "NP": "Nepal", "MM": "Myanmar", "KH": "Cambodia", "LA": "Laos",
        }
        return country_map.get(code, code)  # Return code if not found
    
    def _fetch_pubmed(self) -> pd.DataFrame:
        """Fetch data from PubMed API (E-utilities)."""
        # Use stored params (captured from UI in main thread)
        params = self._fetch_params
        query = params["query"]
        search_type = params["search_type"]
        year_from = params["year_from"]
        year_to = params["year_to"]
        max_results = params["max_results"]
        email = params["email"]
        
        # PubMed E-utilities base URLs
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        
        # Build search query based on type
        if search_type == "Keywords":
            pubmed_query = query
        elif search_type == "Author":
            pubmed_query = f"{query}[Author]"
        elif search_type == "Institution":
            pubmed_query = f"{query}[Affiliation]"
        elif search_type == "ISSN/Journal":
            pubmed_query = f"{query}[Journal]"
        elif search_type == "DOI List":
            # Handle multiple DOIs
            dois = [d.strip() for d in query.replace(",", " ").split()]
            pubmed_query = " OR ".join([f"{d}[DOI]" for d in dois])
        else:
            pubmed_query = query
        
        # Add year filter
        pubmed_query += f" AND {year_from}:{year_to}[dp]"
        
        # Search for PMIDs
        search_params = {
            "db": "pubmed",
            "term": pubmed_query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        
        if email:
            search_params["email"] = email
        
        response = requests.get(esearch_url, params=search_params, timeout=60)
        response.raise_for_status()
        search_data = response.json()
        
        pmids = search_data.get("esearchresult", {}).get("idlist", [])
        
        if not pmids:
            return pd.DataFrame()
        
        # Fetch full records in batches
        all_records = []
        batch_size = 100
        
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(batch_pmids),
                "retmode": "xml",
                "rettype": "abstract",
            }
            
            if email:
                fetch_params["email"] = email
            
            response = requests.get(efetch_url, params=fetch_params, timeout=120)
            response.raise_for_status()
            
            # Parse XML response
            records = self._parse_pubmed_xml(response.text)
            all_records.extend(records)
        
        return pd.DataFrame(all_records)
    
    def _parse_pubmed_xml(self, xml_text: str) -> List[Dict]:
        """Parse PubMed XML response into records."""
        import xml.etree.ElementTree as ET
        
        records = []
        
        try:
            root = ET.fromstring(xml_text)
        except:
            return records
        
        for article in root.findall(".//PubmedArticle"):
            try:
                medline = article.find("MedlineCitation")
                if medline is None:
                    continue
                
                article_elem = medline.find("Article")
                if article_elem is None:
                    continue
                
                # PMID
                pmid_elem = medline.find("PMID")
                pmid = pmid_elem.text if pmid_elem is not None else ""
                
                # Title
                title_elem = article_elem.find("ArticleTitle")
                title = title_elem.text if title_elem is not None else ""
                
                # Abstract
                abstract = ""
                abstract_elem = article_elem.find("Abstract")
                if abstract_elem is not None:
                    abstract_texts = []
                    for text_elem in abstract_elem.findall("AbstractText"):
                        if text_elem.text:
                            label = text_elem.get("Label", "")
                            if label:
                                abstract_texts.append(f"{label}: {text_elem.text}")
                            else:
                                abstract_texts.append(text_elem.text)
                    abstract = " ".join(abstract_texts)
                
                # Authors
                authors = []
                affiliations = []
                author_list = article_elem.find("AuthorList")
                if author_list is not None:
                    for author in author_list.findall("Author"):
                        lastname = author.find("LastName")
                        forename = author.find("ForeName")
                        if lastname is not None:
                            name = lastname.text or ""
                            if forename is not None and forename.text:
                                name = f"{lastname.text}, {forename.text}"
                            authors.append(name)
                        
                        # Affiliations
                        for aff in author.findall("AffiliationInfo/Affiliation"):
                            if aff.text and aff.text not in affiliations:
                                affiliations.append(aff.text)
                
                # Journal
                journal_elem = article_elem.find("Journal")
                source_title = ""
                issn = ""
                year = None
                
                if journal_elem is not None:
                    title_elem = journal_elem.find("Title")
                    if title_elem is not None:
                        source_title = title_elem.text or ""
                    
                    issn_elem = journal_elem.find("ISSN")
                    if issn_elem is not None:
                        issn = issn_elem.text or ""
                    
                    # Publication date
                    pub_date = journal_elem.find("JournalIssue/PubDate")
                    if pub_date is not None:
                        year_elem = pub_date.find("Year")
                        if year_elem is not None:
                            try:
                                year = int(year_elem.text)
                            except:
                                pass
                
                # Keywords
                keywords = []
                keyword_list = medline.find("KeywordList")
                if keyword_list is not None:
                    for kw in keyword_list.findall("Keyword"):
                        if kw.text:
                            keywords.append(kw.text)
                
                # MeSH terms as additional keywords
                mesh_list = medline.find("MeshHeadingList")
                if mesh_list is not None:
                    for mesh in mesh_list.findall("MeshHeading/DescriptorName"):
                        if mesh.text and mesh.text not in keywords:
                            keywords.append(mesh.text)
                
                # DOI
                doi = ""
                article_ids = article.find("PubmedData/ArticleIdList")
                if article_ids is not None:
                    for aid in article_ids.findall("ArticleId"):
                        if aid.get("IdType") == "doi":
                            doi = aid.text or ""
                            break
                
                # Publication type
                doc_type = "article"
                pub_types = article_elem.find("PublicationTypeList")
                if pub_types is not None:
                    for pt in pub_types.findall("PublicationType"):
                        if pt.text:
                            pt_lower = pt.text.lower()
                            if "review" in pt_lower:
                                doc_type = "review"
                                break
                            elif "case" in pt_lower:
                                doc_type = "case report"
                            elif "letter" in pt_lower:
                                doc_type = "letter"
                
                # Extract countries from affiliations
                countries = self._extract_countries_from_affiliations(affiliations)
                
                record = {
                    "Authors": "; ".join(authors),
                    "Title": title or "",
                    "Year": year,
                    "Source title": source_title,
                    "Cited by": 0,  # PubMed doesn't provide citation counts
                    "DOI": doi,
                    "Abstract": abstract[:5000] if abstract else "",
                    "Author Keywords": "; ".join(keywords[:15]),
                    "Affiliations": "; ".join(affiliations),
                    "Countries of Authors": "; ".join(countries),
                    "Corresponding Author Country": countries[0] if countries else "",
                    "Document Type": doc_type,
                    "ISSN": issn,
                    "Open Access": "",  # PubMed doesn't directly provide OA status
                    "PMID": pmid,
                }
                
                records.append(record)
            except Exception as e:
                # Skip problematic records
                continue
        
        return records
    
    def _extract_countries_from_affiliations(self, affiliations: List[str]) -> List[str]:
        """Extract country names from affiliation strings."""
        # Common country patterns (simplified)
        country_patterns = {
            "USA": ["USA", "United States", "U.S.A.", "U.S."],
            "UK": ["United Kingdom", "UK", "England", "Scotland", "Wales"],
            "China": ["China", "P.R. China", "PRC"],
            "Germany": ["Germany", "Deutschland"],
            "France": ["France"],
            "Japan": ["Japan"],
            "Canada": ["Canada"],
            "Australia": ["Australia"],
            "Italy": ["Italy", "Italia"],
            "Spain": ["Spain", "España"],
            "Netherlands": ["Netherlands", "Holland"],
            "Brazil": ["Brazil", "Brasil"],
            "India": ["India"],
            "South Korea": ["South Korea", "Korea", "Republic of Korea"],
            "Switzerland": ["Switzerland"],
            "Sweden": ["Sweden"],
            "Belgium": ["Belgium"],
            "Austria": ["Austria"],
            "Poland": ["Poland"],
            "Denmark": ["Denmark"],
            "Norway": ["Norway"],
            "Finland": ["Finland"],
            "Portugal": ["Portugal"],
            "Greece": ["Greece"],
            "Israel": ["Israel"],
            "Singapore": ["Singapore"],
            "Taiwan": ["Taiwan"],
            "Russia": ["Russia", "Russian Federation"],
            "Mexico": ["Mexico"],
            "Argentina": ["Argentina"],
            "Turkey": ["Turkey", "Türkiye"],
            "Iran": ["Iran"],
            "Saudi Arabia": ["Saudi Arabia"],
            "Egypt": ["Egypt"],
            "South Africa": ["South Africa"],
            "New Zealand": ["New Zealand"],
            "Ireland": ["Ireland"],
            "Czech Republic": ["Czech Republic", "Czechia"],
            "Hungary": ["Hungary"],
            "Slovenia": ["Slovenia"],
        }
        
        countries = []
        for aff in affiliations:
            aff_lower = aff.lower()
            for country, patterns in country_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in aff_lower:
                        if country not in countries:
                            countries.append(country)
                        break
        
        return countries
    
    def _fetch_dimensions(self) -> pd.DataFrame:
        """Fetch data from Dimensions API."""
        # Use stored params (captured from UI in main thread)
        params = self._fetch_params
        query = params["query"]
        search_type = params["search_type"]
        year_from = params["year_from"]
        year_to = params["year_to"]
        max_results = params["max_results"]
        doc_type = params["doc_type"]
        open_access = params["open_access"]
        api_key = params["api_key"]
        
        # Dimensions API endpoint
        base_url = "https://app.dimensions.ai/api/dsl/v2"
        
        # Build DSL query
        if search_type == "Keywords":
            search_clause = f'search publications for "{query}"'
        elif search_type == "Author":
            search_clause = f'search publications where researchers.full_name = "{query}"'
        elif search_type == "Institution":
            search_clause = f'search publications where research_orgs.name = "{query}"'
        elif search_type == "ISSN/Journal":
            search_clause = f'search publications where journal.issn = "{query}"'
        elif search_type == "DOI List":
            dois = [d.strip() for d in query.replace(",", " ").split()]
            doi_list = '["' + '", "'.join(dois) + '"]'
            search_clause = f'search publications where doi in {doi_list}'
        else:
            search_clause = f'search publications for "{query}"'
        
        # Add filters
        where_clauses = [f"year >= {year_from}", f"year <= {year_to}"]
        
        type_map = {
            "Article": "article",
            "Review": "review", 
            "Book Chapter": "chapter",
            "Conference Paper": "proceeding",
            "Preprint": "preprint",
        }
        if doc_type != "All" and doc_type in type_map:
            where_clauses.append(f'type = "{type_map[doc_type]}"')
        
        if open_access:
            where_clauses.append("open_access = 1")
        
        where_str = " and ".join(where_clauses)
        
        # Build full query
        dsl_query = f"""
        {search_clause}
        where {where_str}
        return publications[all]
        limit {min(max_results, 1000)}
        """
        
        # Make request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        response = requests.post(
            base_url,
            headers=headers,
            json={"query": dsl_query},
            timeout=120,
        )
        
        if response.status_code == 401:
            raise ValueError("Invalid API key. Please check your Dimensions API key.")
        
        response.raise_for_status()
        data = response.json()
        
        publications = data.get("publications", [])
        
        # Convert to DataFrame
        df = self._dimensions_to_dataframe(publications)
        
        return df
    
    def _dimensions_to_dataframe(self, publications: List[Dict]) -> pd.DataFrame:
        """Convert Dimensions results to Scopus-compatible DataFrame."""
        records = []
        
        for pub in publications:
            # Extract authors
            authors = []
            affiliations = []
            countries = []
            
            for author in pub.get("authors", []) or []:
                if not author:
                    continue
                name = author.get("full_name", "") or ""
                if name:
                    authors.append(name)
                
                for aff in author.get("affiliations", []) or []:
                    if not aff:
                        continue
                    aff_name = aff.get("name", "") or ""
                    if aff_name and aff_name not in affiliations:
                        affiliations.append(aff_name)
                    # Extract country from affiliation
                    aff_country = aff.get("country", "") or ""
                    if aff_country and aff_country not in countries:
                        countries.append(aff_country)
            
            # Extract source
            source_title = ""
            issn = ""
            journal = pub.get("journal") or {}
            if journal:
                source_title = journal.get("title", "") or ""
                issn = journal.get("issn", "") or ""
            
            # Build record
            record = {
                "Authors": "; ".join(authors),
                "Title": pub.get("title") or "",
                "Year": pub.get("year"),
                "Source title": source_title,
                "Cited by": pub.get("times_cited") or 0,
                "DOI": pub.get("doi") or "",
                "Abstract": (pub.get("abstract") or "")[:5000],
                "Author Keywords": "",  # Dimensions doesn't provide keywords in basic response
                "Affiliations": "; ".join(affiliations),
                "Countries of Authors": "; ".join(countries),  # Scopus-compatible
                "Corresponding Author Country": countries[0] if countries else "",
                "Document Type": pub.get("type") or "article",
                "ISSN": issn,
                "Open Access": "Yes" if pub.get("open_access") else "No",
                "Dimensions ID": pub.get("id") or "",
            }
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def _on_fetch_success(self, df: pd.DataFrame):
        """Handle successful data fetch."""
        self._fetched_data = df
        
        for widget in self.results_content.winfo_children():
            widget.destroy()
        
        # Summary stats
        summary = tk.Frame(self.results_content, bg=self.theme["bg_card"])
        summary.pack(fill=tk.X, pady=(0, 16))
        
        grid = CardGrid(summary, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X)
        
        grid.add_card(StatsCard(grid, "Documents", f"{len(df):,}", "📄", self.theme_name, accent=True))
        
        if "Year" in df.columns:
            years = df["Year"].dropna()
            if len(years) > 0:
                grid.add_card(StatsCard(grid, "Year Range", f"{int(years.min())}-{int(years.max())}", "📅", self.theme_name))
        
        if "Cited by" in df.columns:
            total_cit = df["Cited by"].sum()
            grid.add_card(StatsCard(grid, "Total Citations", f"{int(total_cit):,}", "📈", self.theme_name))
        
        if "Open Access" in df.columns:
            oa_count = (df["Open Access"] == "Yes").sum()
            grid.add_card(StatsCard(grid, "Open Access", f"{oa_count:,}", "🔓", self.theme_name))
        
        # Data table
        table = DataTable(self.results_content, theme=self.theme_name, max_rows=100)
        table.pack(fill=tk.BOTH, expand=True)
        
        # Select key columns for display
        display_cols = ["Title", "Authors", "Year", "Source title", "Cited by", "DOI"]
        display_cols = [c for c in display_cols if c in df.columns]
        table.set_data(df[display_cols].head(100))
        
        # Check if auto-load is enabled
        auto_load = self.auto_load_var.get() if hasattr(self, 'auto_load_var') else False
        
        if auto_load:
            # Status message - loading
            tk.Label(
                self.results_content,
                text=f"✓ Fetched {len(df)} documents. Auto-loading into Biblium...",
                font=FONTS.get_font("small"),
                bg=self.theme["bg_card"], fg=self.theme["success"],
            ).pack(pady=(8, 0))
            
            # Auto-load after a short delay to let UI update
            self.after(100, self._load_into_biblium)
        else:
            # Status message - manual load required
            tk.Label(
                self.results_content,
                text=f"✓ Fetched {len(df)} documents. Click 'Load into Biblium' to analyze.",
                font=FONTS.get_font("small"),
                bg=self.theme["bg_card"], fg=self.theme["success"],
            ).pack(pady=(8, 0))
    
    def _on_fetch_error(self, error: str):
        """Handle fetch error."""
        for widget in self.results_content.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.results_content,
            text=f"❌ Error fetching data:\n{error}",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["danger"],
            wraplength=400,
        ).pack(expand=True, pady=50)
    
    def _save_csv(self):
        """Save fetched data to file (CSV, Excel, or TXT)."""
        if self._fetched_data is None or len(self._fetched_data) == 0:
            messagebox.showwarning("No Data", "No data to save. Fetch data first.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[
                ("CSV", "*.csv"), 
                ("Excel", "*.xlsx"),
                ("Tab-separated", "*.txt"),
                ("All files", "*.*"),
            ],
            title="Save Fetched Data",
        )
        
        if filepath:
            try:
                if filepath.endswith(".xlsx"):
                    self._fetched_data.to_excel(filepath, index=False)
                elif filepath.endswith(".txt"):
                    self._fetched_data.to_csv(filepath, index=False, sep='\t')
                else:
                    self._fetched_data.to_csv(filepath, index=False)
                messagebox.showinfo("Saved", f"Data saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")
    
    def _load_into_biblium(self):
        """Load fetched data into biblium for analysis."""
        if self._fetched_data is None or len(self._fetched_data) == 0:
            messagebox.showwarning("No Data", "No data to load. Fetch data first.")
            return
        
        try:
            from biblium import BiblioAnalysis
            
            # Try to import BiblioPlot for full functionality
            try:
                from biblium import BiblioPlot
                has_plot = True
            except ImportError:
                has_plot = False
            
            # Determine database based on source
            # Use scopus format since we converted column names to Scopus-compatible format
            api = self.api_combo.get()
            db = "scopus"  # Both OpenAlex and Dimensions data are converted to Scopus format
            
            # First save to temp file, then load (BiblioAnalysis expects file path)
            import tempfile
            import os
            
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            try:
                # Save DataFrame to temp file
                print(f"DEBUG Load: Saving {len(self._fetched_data)} rows with columns: {list(self._fetched_data.columns)}")
                
                # Check country column
                country_col = "Corresponding Author Country"
                if country_col in self._fetched_data.columns:
                    non_empty = self._fetched_data[country_col].notna() & (self._fetched_data[country_col] != "")
                    print(f"DEBUG Load: {country_col} has {non_empty.sum()} non-empty values")
                    if non_empty.sum() > 0:
                        print(f"DEBUG Load: Sample country values: {self._fetched_data[non_empty][country_col].head(3).tolist()}")
                
                self._fetched_data.to_csv(temp_path, index=False)
                
                # Get preprocess level from combo
                from biblium.gui.config import PREPROCESS_LEVELS
                preprocess_level = PREPROCESS_LEVELS.get(self.preprocess_combo.get(), 2)
                
                # Build kwargs for BiblioAnalysis
                kwargs = {
                    "f_name": temp_path,
                    "db": db,
                    "preprocess_level": preprocess_level,
                    "res_folder": "results",
                    "label_docs": True,
                }
                
                # Add stopwords if selected
                stopwords = self._get_selected_stopwords()
                if stopwords:
                    kwargs["extra_stopwords"] = stopwords
                    print(f"DEBUG Load: Applying {len(stopwords)} extra_stopwords")
                
                # Create BiblioAnalysis instance (always use BiblioAnalysis for full features)
                bib = BiblioAnalysis(**kwargs)
                
                # IMPORTANT: Copy country data to the column biblium expects
                # Biblium's count methods look for "Countries of Authors" but we have data in "Corresponding Author Country"
                if "Corresponding Author Country" in bib.df.columns and "Countries of Authors" in bib.df.columns:
                    # Check if Countries of Authors is empty but Corresponding Author Country has data
                    ca_country = bib.df["Corresponding Author Country"]
                    all_countries = bib.df["Countries of Authors"]
                    
                    ca_non_empty = ca_country.notna() & (ca_country != "")
                    all_non_empty = all_countries.notna() & (all_countries != "")
                    
                    if ca_non_empty.sum() > 0 and all_non_empty.sum() == 0:
                        print(f"DEBUG Load: Copying {ca_non_empty.sum()} values from 'Corresponding Author Country' to 'Countries of Authors'")
                        bib.df["Countries of Authors"] = bib.df["Corresponding Author Country"]
                        # Also update the Multiple column if it exists
                        if "Countries of Authors Multiple" in bib.df.columns:
                            bib.df["Countries of Authors Multiple"] = bib.df["Corresponding Author Country"]
                
                # Debug: Show what columns biblium has
                print(f"DEBUG Load: Biblium df columns: {list(bib.df.columns)}")
                country_cols = [c for c in bib.df.columns if 'country' in c.lower() or 'countr' in c.lower()]
                print(f"DEBUG Load: Country-related columns in bib.df: {country_cols}")
                for col in country_cols:
                    non_empty = bib.df[col].notna() & (bib.df[col] != "")
                    print(f"DEBUG Load: {col}: {non_empty.sum()} non-empty values")
                
                # Emit dataset loaded event
                event_bus.emit(EventBus.DATASET_LOADED, {
                    "bib": bib,
                    "path": f"API: {api}",
                    "n_docs": len(self._fetched_data),
                })
                
                messagebox.showinfo(
                    "Loaded",
                    f"Successfully loaded {len(self._fetched_data)} documents into Biblium.\n"
                    "You can now use the Analysis panels."
                )
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
        except ImportError:
            messagebox.showerror("Error", "Biblium library not found.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load data: {e}")
