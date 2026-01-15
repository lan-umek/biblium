# -*- coding: utf-8 -*-
"""
Geographic Analysis Panel
=========================
Panel for visualizing country data on maps and analyzing international collaboration.

@author: Claude (Anthropic) for Lan.Umek
@version: 2.7.0
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
import tempfile
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

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class GeographicPanel(BasePanel):
    """Panel for geographic analysis and country visualization."""
    
    title = "Geographic Analysis"
    icon = "🌍"
    description = "Visualize country data and international collaboration"
    requires_data = True
    
    SCOPES = [
        ("world", "World"),
        ("europe", "Europe"),
        ("asia", "Asia"),
        ("africa", "Africa"),
        ("north america", "North America"),
        ("south america", "South America"),
    ]
    
    METRICS = [
        ("Number of documents", "Documents"),
        ("Number of citations", "Citations"),
        ("h-index", "H-Index"),
    ]
    
    ANALYSIS_TYPES = [
        ("country_production", "Country Production", "Publications by corresponding author country"),
        ("country_collaboration", "Country Collaboration", "International collaboration analysis"),
    ]
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        if not self.bib:
            self._show_no_data_message()
            return
        
        # Info Card
        info_frame = tk.Frame(self.options_content, bg="#e8f5e9", relief=tk.FLAT, bd=1)
        info_frame.pack(fill=tk.X, padx=8, pady=(8, 4))
        
        info_inner = tk.Frame(info_frame, bg="#e8f5e9", padx=8, pady=6)
        info_inner.pack(fill=tk.X)
        
        tk.Label(
            info_inner, text="🌍 Geographic Analysis",
            font=FONTS.get_font("body_bold"), bg="#e8f5e9", fg="#2e7d32",
        ).pack(anchor=tk.W)
        
        tk.Label(
            info_inner, 
            text="Visualize country data on world maps.\nRequires: plotly, kaleido",
            font=FONTS.get_font("small"), bg="#e8f5e9", fg="#2e7d32",
            justify=tk.LEFT,
        ).pack(anchor=tk.W)
        
        # Analysis Type Card
        type_card = Card(self.options_content, title="📊 Analysis Type", theme=self.theme_name)
        type_card.pack(fill=tk.X, padx=8, pady=8)
        
        self._analysis_type_var = tk.StringVar(value="country_production")
        
        for type_id, type_name, type_desc in self.ANALYSIS_TYPES:
            rb_frame = tk.Frame(type_card.content, bg=self.theme["bg_card"])
            rb_frame.pack(fill=tk.X, pady=2)
            
            rb = tk.Radiobutton(
                rb_frame, text=type_name,
                variable=self._analysis_type_var, value=type_id,
                bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                selectcolor=self.theme["bg_secondary"],
                font=FONTS.get_font("body"),
                command=self._on_analysis_type_change,
            )
            rb.pack(side=tk.LEFT)
        
        # Region/Scope Card
        scope_card = Card(self.options_content, title="🗺️ Geographic Scope", theme=self.theme_name)
        scope_card.pack(fill=tk.X, padx=8, pady=8)
        
        scope_values = [name for _, name in self.SCOPES]
        self.scope_combo = LabeledCombobox(
            scope_card.content, label="Region:",
            values=scope_values, default="World",
            theme=self.theme_name, label_width=12,
            tooltip="Geographic region to focus on"
        )
        self.scope_combo.pack(fill=tk.X, pady=4)
        
        # Metric Card (for country production)
        self._metric_card = Card(self.options_content, title="📈 Metric", theme=self.theme_name)
        self._metric_card.pack(fill=tk.X, padx=8, pady=8)
        
        metric_values = [name for _, name in self.METRICS]
        self.metric_combo = LabeledCombobox(
            self._metric_card.content, label="Color by:",
            values=metric_values, default="Documents",
            theme=self.theme_name, label_width=12,
            tooltip="Metric to visualize on the map"
        )
        self.metric_combo.pack(fill=tk.X, pady=4)
        
        # Collaboration Options Card (for collaboration analysis)
        self._collab_card = Card(self.options_content, title="🤝 Collaboration Options", theme=self.theme_name)
        
        self.top_countries_spin = LabeledSpinbox(
            self._collab_card.content, label="Top countries:",
            from_=5, to=50, default=20,
            theme=self.theme_name, label_width=14,
            tooltip="Number of top countries for heatmap"
        )
        self.top_countries_spin.pack(fill=tk.X, pady=4)
        
        self.show_links_var = tk.BooleanVar(value=True)
        LabeledCheckbox(
            self._collab_card.content, label="Show collaboration links",
            variable=self.show_links_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
        
        # Visualization Options
        viz_card = CollapsibleCard(
            self.options_content, title="🎨 Visualization Options",
            theme=self.theme_name, collapsed=True
        )
        viz_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.dark_mode_var = tk.BooleanVar(value=False)
        LabeledCheckbox(
            viz_card.content, label="Dark mode",
            variable=self.dark_mode_var, theme=self.theme_name,
        ).pack(fill=tk.X, pady=2)
        
        colormaps = ["Viridis", "Blues", "Greens", "Reds", "YlOrRd", "Plasma", "Cividis"]
        self.colormap_combo = LabeledCombobox(
            viz_card.content, label="Colormap:",
            values=colormaps, default="Viridis",
            theme=self.theme_name, label_width=12,
        )
        self.colormap_combo.pack(fill=tk.X, pady=4)
        
        # Action Buttons
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Generate Map", icon="🌍",
            command=self._run_analysis, theme=self.theme_name,
        ).pack(fill=tk.X, pady=4)
    
    def _on_analysis_type_change(self):
        """Handle analysis type change."""
        analysis_type = self._analysis_type_var.get()
        
        if analysis_type == "country_collaboration":
            self._metric_card.pack_forget()
            self._collab_card.pack(fill=tk.X, padx=8, pady=8, after=self._metric_card.master.winfo_children()[3])
        else:
            self._collab_card.pack_forget()
            self._metric_card.pack(fill=tk.X, padx=8, pady=8, after=self._metric_card.master.winfo_children()[3])
    
    def _create_results(self):
        """Create results panel."""
        super()._create_results()
        
        # Create notebook with Info tab
        self.results_notebook = ttk.Notebook(self.results_content)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        self.results_tab = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.results_tab, text="🌍 Map")
        
        # Info tab
        self.info_frame = tk.Frame(self.results_notebook, bg=self.theme["bg_card"])
        self.results_notebook.add(self.info_frame, text="ℹ️ Info")
        self._create_info_content(self.info_frame)
        
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
            text="🌍 Geographic Analysis\n\n"
                 "Select options and click 'Generate Map'\n"
                 "to visualize country data on world maps.",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            justify=tk.CENTER,
        ).pack(expand=True)

    def _run_analysis(self):
        """Run geographic analysis."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        # Check for plotly
        try:
            import plotly.express as px
        except ImportError:
            messagebox.showerror(
                "Missing Dependency",
                "Plotly is required for geographic visualization.\n\nInstall with: pip install plotly"
            )
            return
        
        analysis_type = self._analysis_type_var.get()
        
        self._show_loading("Generating map...")
        
        def do_analysis():
            try:
                # Get scope
                scope_name = self.scope_combo.get()
                scope = "world"
                for scope_id, name in self.SCOPES:
                    if name == scope_name:
                        scope = scope_id
                        break
                
                if analysis_type == "country_production":
                    result = self._analyze_country_production(scope)
                else:
                    result = self._analyze_country_collaboration(scope)
                
                self.after(0, lambda r=result: self._on_success(r))
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: self._show_error(msg))
        
        threading.Thread(target=do_analysis, daemon=True).start()
    
    def _analyze_country_production(self, scope: str) -> Dict:
        """Analyze country production using biblium methods."""
        from biblium import plotbib
        
        # Count CA countries using biblium method
        if not hasattr(self.bib, 'ca_country_counts_df'):
            self.bib.count_ca_countries()
        
        df = self.bib.ca_country_counts_df.copy()
        
        # Get metric column
        metric_display = self.metric_combo.get()
        metric_col = "Number of documents"
        for col, name in self.METRICS:
            if name == metric_display:
                metric_col = col
                break
        
        if metric_col not in df.columns:
            metric_col = "Number of documents"
        
        # Generate map using biblium's plotbib function
        temp_dir = tempfile.mkdtemp()
        temp_prefix = os.path.join(temp_dir, "country_map")
        
        fig = plotbib.save_plotly_choropleth_map(
            df, 
            metric_col,
            filename_prefix=temp_prefix,
            scope=scope,
            dark_mode=self.dark_mode_var.get(),
            colormap=self.colormap_combo.get(),
            title=f"Country Production - {metric_col}",
            hover_name="Country" if "Country" in df.columns else None,
        )
        
        # Get PNG path
        png_path = f"{temp_prefix}.png"
        html_path = f"{temp_prefix}.html"
        
        return {
            "type": "country_production",
            "data": df,
            "metric": metric_col,
            "scope": scope,
            "png_path": png_path if os.path.exists(png_path) else None,
            "html_path": html_path if os.path.exists(html_path) else None,
            "temp_dir": temp_dir,
            "total_countries": len(df),
            "total_docs": df["Number of documents"].sum() if "Number of documents" in df.columns else 0,
        }
    
    def _analyze_country_collaboration(self, scope: str) -> Dict:
        """Analyze country collaboration using biblium methods."""
        from biblium import plotbib
        
        # Get region parameter for biblium
        region = None
        if scope and scope != "world":
            region = scope.title() if scope not in ["north america", "south america"] else " ".join(w.title() for w in scope.split())
        
        # Compute collaboration using biblium method
        self.bib.get_country_collaboration(region=region)
        
        collab_matrix = self.bib.country_collab_matrix
        links_df = getattr(self.bib, 'countries_links_df', None)
        
        if collab_matrix is None or len(collab_matrix) == 0:
            raise ValueError("No collaboration data available. Check if country information exists in your dataset.")
        
        # Count all countries for the map
        if not hasattr(self.bib, 'all_countries_counts_df'):
            try:
                self.bib.count_all_countries()
            except:
                pass
        
        all_countries_df = getattr(self.bib, 'all_countries_counts_df', None)
        
        # Generate map with collaboration links
        temp_dir = tempfile.mkdtemp()
        temp_prefix = os.path.join(temp_dir, "collab_map")
        
        # Use all_countries_df if available, else use CA countries
        map_df = all_countries_df if all_countries_df is not None else self.bib.ca_country_counts_df
        
        fig = plotbib.save_plotly_choropleth_map(
            map_df,
            "Number of documents",
            filename_prefix=temp_prefix,
            scope=scope,
            dark_mode=self.dark_mode_var.get(),
            colormap=self.colormap_combo.get(),
            title=f"Country Collaboration - {scope.title()}",
            hover_name="Country" if "Country" in map_df.columns else None,
            links_df=links_df if self.show_links_var.get() else None,
            link_color="red",
            link_opacity=0.6,
        )
        
        png_path = f"{temp_prefix}.png"
        html_path = f"{temp_prefix}.html"
        
        # Extract top pairs
        top_pairs_df = None
        pairs = []
        for i, row_label in enumerate(collab_matrix.index):
            for j, col_label in enumerate(collab_matrix.columns):
                if i < j:
                    val = collab_matrix.iloc[i, j]
                    if val > 0:
                        pairs.append({
                            'Country 1': row_label,
                            'Country 2': col_label,
                            'Collaborations': int(val)
                        })
        
        if pairs:
            top_pairs_df = pd.DataFrame(pairs).sort_values(
                'Collaborations', ascending=False
            ).head(20).reset_index(drop=True)
        
        return {
            "type": "country_collaboration",
            "data": map_df,
            "collab_matrix": collab_matrix,
            "top_pairs": top_pairs_df,
            "scope": scope,
            "png_path": png_path if os.path.exists(png_path) else None,
            "html_path": html_path if os.path.exists(html_path) else None,
            "temp_dir": temp_dir,
            "total_countries": len(collab_matrix),
            "total_collaborations": int(collab_matrix.values.sum() / 2),
        }
    
    def _on_success(self, result: Dict):
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
        
        df = result["data"]
        
        # Summary Cards
        grid = CardGrid(self.results_tab, columns=3, theme=self.theme_name)
        grid.pack(fill=tk.X, pady=(0, 12))
        
        grid.add_card(StatsCard(grid, "Countries", f"{result['total_countries']:,}", "🌍", self.theme_name))
        
        if result["type"] == "country_production":
            grid.add_card(StatsCard(grid, "Documents", f"{result['total_docs']:,}", "📄", self.theme_name))
            if len(df) > 0 and "Country" in df.columns:
                top_country = df.iloc[0]["Country"]
                grid.add_card(StatsCard(grid, "Top Country", str(top_country), "🏆", self.theme_name))
        else:
            grid.add_card(StatsCard(grid, "Collaborations", f"{result['total_collaborations']:,}", "🤝", self.theme_name))
            if result.get("top_pairs") is not None and len(result["top_pairs"]) > 0:
                top_pair = f"{result['top_pairs'].iloc[0]['Country 1'][:10]}-{result['top_pairs'].iloc[0]['Country 2'][:10]}"
                grid.add_card(StatsCard(grid, "Top Pair", top_pair, "🏆", self.theme_name))
        
        # Display Map Image
        tk.Label(
            self.results_tab, text="🗺️ World Map",
            font=FONTS.get_font("subheading"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(anchor=tk.W, pady=(8, 4))
        
        png_path = result.get("png_path")
        if png_path and os.path.exists(png_path) and HAS_PIL:
            self._display_map_image(png_path)
        else:
            # Show message that map was generated
            html_path = result.get("html_path")
            if html_path and os.path.exists(html_path):
                tk.Label(
                    self.results_tab, 
                    text="✅ Map generated successfully!\nClick 'Open Interactive Map' to view.",
                    font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                    fg=self.theme["success"],
                ).pack(pady=20)
            else:
                tk.Label(
                    self.results_tab, 
                    text="⚠️ Could not generate map image.\nInstall kaleido: pip install kaleido",
                    font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                    fg=self.theme["warning"],
                ).pack(pady=20)
        
        # Country Data Table
        tk.Label(
            self.results_tab, text="📋 Country Data",
            font=FONTS.get_font("subheading"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(anchor=tk.W, pady=(16, 4))
        
        display_cols = ["Country", "Number of documents", "ISO-3", "Continent"]
        display_cols = [c for c in display_cols if c in df.columns]
        
        table = DataTable(self.results_tab, theme=self.theme_name, height=8)
        table.pack(fill=tk.X, pady=(0, 8))
        table.set_data(df[display_cols].head(30))
        
        # Top pairs table for collaboration
        if result["type"] == "country_collaboration" and result.get("top_pairs") is not None:
            tk.Label(
                self.results_tab, text="🤝 Top Collaborating Pairs",
                font=FONTS.get_font("subheading"), bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
            ).pack(anchor=tk.W, pady=(8, 4))
            
            pairs_table = DataTable(self.results_tab, theme=self.theme_name, height=6)
            pairs_table.pack(fill=tk.X, pady=(0, 8))
            pairs_table.set_data(result["top_pairs"])
        
        # Action buttons
        btn_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"])
        btn_frame.pack(fill=tk.X, pady=8)
        
        html_path = result.get("html_path")
        if html_path and os.path.exists(html_path):
            ThemedButton(
                btn_frame, text="🌐 Open Interactive Map",
                command=lambda: self._open_html_map(html_path), theme=self.theme_name
            ).pack(side=tk.LEFT, padx=4)
        
        ThemedButton(
            btn_frame, text="💾 Save Map",
            command=lambda: self._save_map(result), theme=self.theme_name
        ).pack(side=tk.LEFT, padx=4)
        
        ThemedButton(
            btn_frame, text="📥 Export Data",
            command=lambda: self._export_data(df), theme=self.theme_name
        ).pack(side=tk.LEFT, padx=4)
        
        # Store result for later use
        self._current_result = result
        
        event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": "Geographic Analysis"})
    
    def _display_map_image(self, png_path: str):
        """Display map image in the panel."""
        try:
            # Load and resize image
            img = Image.open(png_path)
            
            # Get available width (approximately)
            max_width = 700
            max_height = 400
            
            # Calculate resize ratio
            ratio = min(max_width / img.width, max_height / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img_resized)
            
            # Create frame for image
            img_frame = tk.Frame(self.results_tab, bg=self.theme["bg_card"], bd=1, relief=tk.SOLID)
            img_frame.pack(fill=tk.X, pady=8)
            
            img_label = tk.Label(img_frame, image=photo, bg=self.theme["bg_card"])
            img_label.image = photo  # Keep reference
            img_label.pack(padx=2, pady=2)
            
        except Exception as e:
            tk.Label(
                self.results_tab, 
                text=f"⚠️ Could not display map: {str(e)}",
                font=FONTS.get_font("small"), bg=self.theme["bg_card"],
                fg=self.theme["warning"],
            ).pack(pady=8)
    
    def _open_html_map(self, html_path: str):
        """Open interactive HTML map in browser."""
        import webbrowser
        webbrowser.open(f"file://{html_path}")
    
    def _save_map(self, result: Dict):
        """Save map to user-specified location."""
        filetypes = [
            ("HTML files", "*.html"),
            ("PNG files", "*.png"),
            ("PDF files", "*.pdf"),
            ("SVG files", "*.svg"),
        ]
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=filetypes,
            title="Save Map"
        )
        
        if not filepath:
            return
        
        try:
            import shutil
            
            ext = os.path.splitext(filepath)[1].lower()
            temp_dir = result.get("temp_dir")
            
            if temp_dir:
                # Find matching temp file
                if ext == ".html":
                    src = result.get("html_path")
                elif ext == ".png":
                    src = result.get("png_path")
                elif ext == ".pdf":
                    src = os.path.join(temp_dir, "country_map.pdf") if result["type"] == "country_production" else os.path.join(temp_dir, "collab_map.pdf")
                elif ext == ".svg":
                    src = os.path.join(temp_dir, "country_map.svg") if result["type"] == "country_production" else os.path.join(temp_dir, "collab_map.svg")
                else:
                    src = result.get("html_path")
                
                if src and os.path.exists(src):
                    shutil.copy(src, filepath)
                    messagebox.showinfo("Success", f"Map saved to:\n{filepath}")
                else:
                    messagebox.showwarning("Not Found", f"File format not available. Try HTML or PNG.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Save failed:\n{str(e)}")
    
    def _export_data(self, df: pd.DataFrame):
        """Export data to Excel."""
        if df is None or len(df) == 0:
            messagebox.showwarning("No Data", "No data to export.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")],
            title="Save Data"
        )
        
        if not filepath:
            return
        
        try:
            if filepath.endswith('.csv'):
                df.to_csv(filepath, index=False)
            else:
                df.to_excel(filepath, index=False)
            
            messagebox.showinfo("Success", f"Data exported to:\n{filepath}")
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
GEOGRAPHIC ANALYSIS
═══════════════════

Analyze geographical distribution of research.

MAP TYPES
─────────
• Choropleth: Color by value
• Bubble map: Size by value
• Connection map: Collaboration links

DATA SOURCES
────────────
• Author countries
• Affiliation countries
• Corresponding author country
• Collaboration pairs

METRICS MAPPED
──────────────
• Publication count
• Citation count
• H-index
• Collaboration intensity

VISUALIZATION OPTIONS
─────────────────────
• World map projection
• Regional focus (Europe, Asia, etc.)
• Color schemes
• Legend customization

OUTPUT
──────
• Interactive map
• Country statistics table
• Top countries ranking
• Export to image
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
