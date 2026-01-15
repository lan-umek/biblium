"""
Repository Links Panel
======================

Extract and analyze data/code repository links from publications.

Combines:
- Text mining: Extract URLs from abstracts/titles
- DataCite API: Find related datasets
- Papers With Code API: Find code repositories
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import threading
import webbrowser

from biblium.gui.panels.base import BasePanel
from biblium.gui.config import FONTS, get_theme
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.plots import PlotFrame
from biblium.gui.widgets.forms import LabeledEntry, LabeledCombobox, LabeledSpinbox
from biblium.gui.widgets.cards import Card, StatsCard, CardGrid
from biblium.gui.widgets.buttons import ActionButton

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class RepositoryLinksPanel(BasePanel):
    """
    Panel for extracting and analyzing data/code repository links.
    """
    
    title = "Repository Links"
    icon = "🔗"
    description = "Extract data/code repository links from publications"
    requires_data = True
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._results = None
        self._links_df = None
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Methods Card
        methods_card = Card(self.options_content, title="🔍 Extraction Methods", theme=self.theme_name)
        methods_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Text mining (always recommended)
        self.text_mining_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            methods_card.content, text="Text Mining (extract URLs from abstracts)",
            variable=self.text_mining_var, bg=self.theme["bg_card"],
            fg=self.theme["text_primary"], font=FONTS.get_font("body"),
            activebackground=self.theme["bg_card"],
        ).pack(anchor="w", pady=2)
        
        # DataCite API
        self.datacite_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            methods_card.content, text="DataCite API (find related datasets)",
            variable=self.datacite_var, bg=self.theme["bg_card"],
            fg=self.theme["text_primary"], font=FONTS.get_font("body"),
            activebackground=self.theme["bg_card"],
        ).pack(anchor="w", pady=2)
        
        # Papers With Code API
        self.pwc_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            methods_card.content, text="Papers With Code API (find code repos)",
            variable=self.pwc_var, bg=self.theme["bg_card"],
            fg=self.theme["text_primary"], font=FONTS.get_font("body"),
            activebackground=self.theme["bg_card"],
        ).pack(anchor="w", pady=2)
        
        # Note about API methods
        tk.Label(
            methods_card.content, 
            text="Note: API methods are slower but find additional links",
            font=FONTS.get_font("small"), bg=self.theme["bg_card"],
            fg=self.theme["text_muted"], wraplength=280
        ).pack(anchor="w", pady=(8, 2))
        
        # Parameters Card
        params_card = Card(self.options_content, title="⚙️ Parameters", theme=self.theme_name)
        params_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.max_requests_spin = LabeledSpinbox(
            params_card.content, label="Max API Requests:",
            from_=10, to=200, default=50,
            theme=self.theme_name, label_width=16,
            tooltip="Maximum requests per API source"
        )
        self.max_requests_spin.pack(fill=tk.X, pady=4)
        
        # Repository Types Card
        types_card = Card(self.options_content, title="📦 Repository Types", theme=self.theme_name)
        types_card.pack(fill=tk.X, padx=8, pady=8)
        
        tk.Label(
            types_card.content, text="Detected repository types:",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"]
        ).pack(anchor="w")
        
        repos_text = "Code: GitHub, GitLab, Bitbucket, Hugging Face, Code Ocean\n"
        repos_text += "Data: Zenodo, Figshare, Dryad, OSF, Dataverse, Kaggle"
        
        tk.Label(
            types_card.content, text=repos_text,
            font=FONTS.get_font("small"), bg=self.theme["bg_card"],
            fg=self.theme["text_muted"], justify=tk.LEFT
        ).pack(anchor="w", pady=4)
        
        # Run Button
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Extract Links", icon="🔗",
            command=self._run_analysis, theme=self.theme_name,
        ).pack(fill=tk.X)
        
        # Export
        export_card = Card(self.options_content, title="📥 Export", theme=self.theme_name)
        export_card.pack(fill=tk.X, padx=8, pady=8)
        
        tk.Button(
            export_card.content, text="Export Results to Excel",
            command=self._export_results,
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
            font=FONTS.get_font("body"), relief=tk.FLAT, cursor="hand2"
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
            "🔗 Repository Links\n\n"
            "Analyze open access and repository availability.\n\n"
            "Features:\n"
            "• Open access percentage\n"
            "• Repository distribution\n"
            "• DOI coverage\n"
            "• Access type breakdown\n"
            "\n"
            "Assess open science practices in your field.\n\n"
            "Steps:\n"
            "1. Load dataset with DOIs/URLs\n"
            "2. Select analysis type\n"
            "3. Configure options\n"
            "4. Click 'Analyze'\n"
        )
        
        tk.Label(
            self.results_tab,
            text=msg,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)

    def _run_analysis(self):
        """Run repository link extraction."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        use_text = self.text_mining_var.get()
        use_datacite = self.datacite_var.get()
        use_pwc = self.pwc_var.get()
        
        if not any([use_text, use_datacite, use_pwc]):
            messagebox.showwarning("No Methods", "Please select at least one extraction method.")
            return
        
        self._show_loading("Extracting repository links...")
        
        def do_analysis():
            try:
                print("[DEBUG] Starting repository link extraction")
                
                max_requests = self.max_requests_spin.get()
                
                # Call biblium's extract_repository_links
                if hasattr(self.bib, 'extract_repository_links'):
                    result = self.bib.extract_repository_links(
                        use_text_mining=use_text,
                        use_datacite=use_datacite,
                        use_paperswithcode=use_pwc,
                        max_api_requests=max_requests,
                    )
                else:
                    # Fallback to utilsbib directly
                    from biblium import utilsbib
                    self.bib.df, stats = utilsbib.enrich_with_repository_links(
                        self.bib.df,
                        use_text_mining=use_text,
                        use_datacite=use_datacite,
                        use_paperswithcode=use_pwc,
                        max_api_requests=max_requests,
                    )
                    result = {"stats": stats, "links_df": pd.DataFrame()}
                
                print(f"[DEBUG] Extraction complete. Found {result['stats']['total_links_found']} links")
                
                self._results = result
                self._links_df = result.get("links_df", pd.DataFrame())
                
                self.after(0, lambda: self._on_success(result))
                
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[DEBUG] Error: {e}")
                print(f"[DEBUG] Traceback: {tb}")
                self.after(0, lambda: self._show_error(f"{e}\n\n{tb}"))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _on_success(self, result):
        """Display extraction results."""
        self._stop_active_spinners()
        self._safe_clear_results()
        
        stats = result.get("stats", {})
        
        # Create notebook for results
        notebook = ttk.Notebook(self.results_tab)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Summary
        summary_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(summary_frame, text="Summary")
        self._create_summary_tab(summary_frame, stats)
        
        # Tab 2: Distribution
        if HAS_MATPLOTLIB:
            dist_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
            notebook.add(dist_frame, text="Distribution")
            self._create_distribution_plot(dist_frame)
        
        # Tab 3: Links List
        links_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(links_frame, text="All Links")
        self._create_links_tab(links_frame)
        
        # Tab 4: Documents with Links
        docs_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(docs_frame, text="Documents")
        
        # Info tab
        info_frame = tk.Frame(notebook, bg=self.theme["bg_card"])
        notebook.add(info_frame, text="ℹ️ Info")
        self._create_info_content(info_frame)
        self._create_documents_tab(docs_frame)
        
        total = stats.get("total_links_found", 0)
        code = stats.get("docs_with_code_links", 0)
        data = stats.get("docs_with_data_links", 0)
        messagebox.showinfo("Complete", 
            f"Found {total} repository links\n"
            f"Documents with code: {code}\n"
            f"Documents with data: {data}")
    
    def _create_summary_tab(self, parent, stats):
        """Create summary statistics tab."""
        # Title
        tk.Label(
            parent, text="Repository Links Summary",
            font=FONTS.get_font("heading2"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"]
        ).pack(pady=(16, 8))
        
        # Stats cards
        total_docs = stats.get("total_docs", len(self.bib.df) if self.bib else 0)
        docs_code = stats.get("docs_with_code_links", 0)
        docs_data = stats.get("docs_with_data_links", 0)
        total_links = stats.get("total_links_found", 0)
        
        grid = CardGrid(parent, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X, padx=16, pady=8)
        
        grid.add_card(StatsCard(grid, "Total Documents", f"{total_docs:,}", "📄", self.theme_name))
        grid.add_card(StatsCard(grid, "With Code Links", f"{docs_code}", "💻", self.theme_name, accent=True))
        grid.add_card(StatsCard(grid, "With Data Links", f"{docs_data}", "📊", self.theme_name))
        grid.add_card(StatsCard(grid, "Total Links", f"{total_links}", "🔗", self.theme_name))
        
        # Percentages
        pct_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        pct_frame.pack(fill=tk.X, padx=16, pady=16)
        
        if total_docs > 0:
            pct_code = docs_code / total_docs * 100
            pct_data = docs_data / total_docs * 100
            
            tk.Label(
                pct_frame, 
                text=f"Code availability: {pct_code:.1f}% of documents",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                fg=self.theme["text_primary"]
            ).pack(anchor="w")
            
            tk.Label(
                pct_frame,
                text=f"Data availability: {pct_data:.1f}% of documents",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                fg=self.theme["text_primary"]
            ).pack(anchor="w")
        
        # Methods used
        methods = stats.get("sources_used", [])
        if methods:
            tk.Label(
                pct_frame,
                text=f"Methods used: {', '.join(methods)}",
                font=FONTS.get_font("small"), bg=self.theme["bg_card"],
                fg=self.theme["text_muted"]
            ).pack(anchor="w", pady=(8, 0))
    
    def _create_distribution_plot(self, parent):
        """Create repository distribution plot."""
        plot_frame = PlotFrame(parent, theme=self.theme_name, show_toolbar=True, show_ai_button=True)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Count by repository type
        if self._links_df is not None and len(self._links_df) > 0:
            repo_counts = self._links_df["Repository"].value_counts()
            
            fig, ax = plot_frame.get_figure()
            
            colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]
            ax.barh(repo_counts.index, repo_counts.values, 
                   color=colors[:len(repo_counts)], edgecolor="white")
            ax.set_xlabel("Number of Links")
            ax.set_ylabel("Repository")
            ax.set_title("Links by Repository Type")
            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
        else:
            fig, ax = plot_frame.get_figure()
            ax.text(0.5, 0.5, "No links found", ha='center', va='center',
                   fontsize=14, color='gray')
            ax.axis('off')
        
        plot_frame.refresh()
    
    def _create_links_tab(self, parent):
        """Create links list tab."""
        if self._links_df is None or len(self._links_df) == 0:
            tk.Label(
                parent, text="No repository links found.",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                fg=self.theme["text_muted"]
            ).pack(expand=True, pady=50)
            return
        
        # Filter controls
        control_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        control_frame.pack(fill=tk.X, padx=8, pady=8)
        
        tk.Label(
            control_frame, text="Filter by Type:",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"]
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        types = ["All"] + list(self._links_df["Type"].unique())
        self._type_filter_var = tk.StringVar(value="All")
        type_combo = ttk.Combobox(
            control_frame, textvariable=self._type_filter_var,
            values=types, state="readonly", width=12
        )
        type_combo.pack(side=tk.LEFT, padx=4)
        
        # Open link button
        tk.Button(
            control_frame, text="Open Selected Link",
            command=self._open_selected_link,
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
            font=FONTS.get_font("body"), relief=tk.FLAT, cursor="hand2"
        ).pack(side=tk.RIGHT, padx=4)
        
        # Table
        table_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        self._links_table = DataTable(table_frame, theme=self.theme_name)
        self._links_table.pack(fill=tk.BOTH, expand=True)
        self._update_links_table()
        
        # Bind filter change
        type_combo.bind("<<ComboboxSelected>>", lambda e: self._update_links_table())
    
    def _update_links_table(self):
        """Update links table based on filter."""
        if self._links_df is None:
            return
        
        filter_val = self._type_filter_var.get()
        
        if filter_val == "All":
            filtered = self._links_df
        else:
            filtered = self._links_df[self._links_df["Type"] == filter_val]
        
        self._links_table.set_data(filtered)
    
    def _open_selected_link(self):
        """Open selected link in browser."""
        # Get selected row from table
        try:
            selection = self._links_table.tree.selection()
            if selection:
                item = self._links_table.tree.item(selection[0])
                values = item.get("values", [])
                # URL is typically the 3rd column
                if len(values) >= 3:
                    url = values[2]
                    if url and isinstance(url, str):
                        if not url.startswith("http"):
                            url = "https://" + url
                        webbrowser.open(url)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open link: {e}")
    
    def _create_documents_tab(self, parent):
        """Create documents with links tab."""
        if not self.bib:
            return
        
        # Filter to documents with links
        df = self.bib.df
        
        if "Has Code Link" not in df.columns:
            tk.Label(
                parent, text="Run extraction first.",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                fg=self.theme["text_muted"]
            ).pack(expand=True, pady=50)
            return
        
        # Filter controls
        control_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        control_frame.pack(fill=tk.X, padx=8, pady=8)
        
        tk.Label(
            control_frame, text="Show:",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"]
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        self._doc_filter_var = tk.StringVar(value="With Any Link")
        doc_filter = ttk.Combobox(
            control_frame, textvariable=self._doc_filter_var,
            values=["All", "With Any Link", "With Code Link", "With Data Link", "Without Links"],
            state="readonly", width=15
        )
        doc_filter.pack(side=tk.LEFT, padx=4)
        
        # Table
        table_frame = tk.Frame(parent, bg=self.theme["bg_card"])
        table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        self._docs_table = DataTable(table_frame, theme=self.theme_name)
        self._docs_table.pack(fill=tk.BOTH, expand=True)
        self._update_docs_table()
        
        # Bind filter change
        doc_filter.bind("<<ComboboxSelected>>", lambda e: self._update_docs_table())
    
    def _update_docs_table(self):
        """Update documents table based on filter."""
        if not self.bib:
            return
        
        df = self.bib.df
        filter_val = self._doc_filter_var.get()
        
        if filter_val == "With Any Link":
            filtered = df[(df.get("Has Code Link", False)) | (df.get("Has Data Link", False))]
        elif filter_val == "With Code Link":
            filtered = df[df.get("Has Code Link", False) == True]
        elif filter_val == "With Data Link":
            filtered = df[df.get("Has Data Link", False) == True]
        elif filter_val == "Without Links":
            filtered = df[(df.get("Has Code Link", False) == False) & (df.get("Has Data Link", False) == False)]
        else:
            filtered = df
        
        # Select display columns
        title_col = self.bib.mapping.get("Title", "Title")
        display_cols = ["Doc ID"]
        
        if title_col in filtered.columns:
            display_cols.append(title_col)
        
        for col in ["Has Code Link", "Has Data Link", "Repository Links"]:
            if col in filtered.columns:
                display_cols.append(col)
        
        display_df = filtered[display_cols].copy()
        self._docs_table.set_data(display_df)
    
    def _export_results(self):
        """Export results to Excel."""
        if self._results is None:
            messagebox.showwarning("No Results", "Run extraction first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Export Repository Links"
        )
        
        if not filename:
            return
        
        try:
            with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
                # Links
                if self._links_df is not None and len(self._links_df) > 0:
                    self._links_df.to_excel(writer, sheet_name="Links", index=False)
                
                # Documents with links
                if self.bib and "Has Code Link" in self.bib.df.columns:
                    docs_with_links = self.bib.df[
                        (self.bib.df["Has Code Link"]) | (self.bib.df["Has Data Link"])
                    ]
                    title_col = self.bib.mapping.get("Title", "Title")
                    cols = ["Doc ID", title_col, "Has Code Link", "Has Data Link", 
                           "Code Repositories", "Data Repositories"]
                    cols = [c for c in cols if c in docs_with_links.columns]
                    docs_with_links[cols].to_excel(writer, sheet_name="Documents", index=False)
                
                # Summary stats
                stats = self._results.get("stats", {})
                summary_df = pd.DataFrame([stats])
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
            
            messagebox.showinfo("Success", f"Results exported to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {e}")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
REPOSITORY LINKS
════════════════

Analyze open access and repository availability.

ANALYSIS
────────
• Open access status
• Repository deposits
• DOI availability
• Full-text links

OA CATEGORIES
─────────────
• Gold: Journal OA
• Green: Repository version
• Hybrid: OA option in closed
• Bronze: Free but unlicensed
• Closed: Paywalled

METRICS
───────
• OA percentage
• Repository rate
• DOI coverage
• By OA type

DATA SOURCES
────────────
• DOI lookup
• OpenAlex OA status
• Unpaywall data
• Repository APIs
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

    def set_bib(self, bib):
        """Set the bibliometric data object."""
        self.bib = bib
        self._results = None
        self._links_df = None
