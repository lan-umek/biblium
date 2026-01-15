# -*- coding: utf-8 -*-
"""
Citation Network Panel
======================
Document citation network and Historiograph using biblium's implementation.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import io
from typing import Dict, List, Set, Optional

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
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class CitationNetworkPanel(BasePanel):
    """Panel for document citation network and historiograph."""
    
    title = "Citation Network"
    icon = "📄"
    description = "Document citation network and Historiograph"
    requires_data = True
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._graph = None
        self._node_df = None
        self._edge_df = None
        self._current_fig = None
        self._photo_image = None
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._build_network  # Set primary action for toolbar Run button
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Network Type Card
        type_card = Card(self.options_content, title="📊 Network Type", theme=self.theme_name)
        type_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.network_type = LabeledCombobox(
            type_card.content, label="Type:",
            values=["Citation Network"],
            default="Citation Network",
            theme=self.theme_name, label_width=12,
        )
        self.network_type.pack(fill=tk.X, pady=4)
        
        # Parameters Card
        params_card = Card(self.options_content, title="⚙️ Parameters", theme=self.theme_name)
        params_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.min_citations = LabeledSpinbox(
            params_card.content, label="Min Citations:",
            from_=0, to=500, default=1,
            theme=self.theme_name, label_width=15,
        )
        self.min_citations.pack(fill=tk.X, pady=4)
        
        self.top_n = LabeledSpinbox(
            params_card.content, label="Top N Documents:",
            from_=10, to=500, default=50,
            theme=self.theme_name, label_width=15,
        )
        self.top_n.pack(fill=tk.X, pady=4)
        
        self.match_cutoff = LabeledSpinbox(
            params_card.content, label="Match Cutoff (%):",
            from_=50, to=100, default=85,
            theme=self.theme_name, label_width=15,
        )
        self.match_cutoff.pack(fill=tk.X, pady=4)
        
        # Main Path Analysis Card
        mpa_card = Card(self.options_content, title="🛤️ Main Path Analysis", theme=self.theme_name)
        mpa_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.mpa_method = LabeledCombobox(
            mpa_card.content, label="Method:",
            values=["SPC", "SPLC", "SPNP"],
            default="SPC",
            theme=self.theme_name, label_width=12,
        )
        self.mpa_method.pack(fill=tk.X, pady=4)
        
        # Method help text
        help_label = tk.Label(
            mpa_card.content, 
            text="SPC: Search Path Count\nSPLC: Normalized by path length\nSPNP: Normalized by node pairs",
            font=FONTS.get_font("small"), bg=self.theme["bg_secondary"],
            fg=self.theme["text_muted"], justify=tk.LEFT,
        )
        help_label.pack(anchor="w", pady=2)
        
        # Visualization Card
        viz_card = Card(self.options_content, title="🎨 Visualization", theme=self.theme_name)
        viz_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.layout_combo = LabeledCombobox(
            viz_card.content, label="Layout:",
            values=["chronological", "spring", "kamada_kawai", "circular"],
            default="chronological",
            theme=self.theme_name, label_width=12,
        )
        self.layout_combo.pack(fill=tk.X, pady=4)
        
        self.num_labels = LabeledSpinbox(
            viz_card.content, label="Show Labels:",
            from_=0, to=100, default=20,
            theme=self.theme_name, label_width=12,
        )
        self.num_labels.pack(fill=tk.X, pady=4)
        
        self.color_by = LabeledCombobox(
            viz_card.content, label="Color by:",
            values=["Year", "Citations", "In-Degree"],
            default="Year",
            theme=self.theme_name, label_width=12,
        )
        self.color_by.pack(fill=tk.X, pady=4)
        
        # Run buttons
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Build Network", icon="📄",
            command=self._build_network, theme=self.theme_name,
        ).pack(fill=tk.X)
        
        ThemedButton(
            btn_frame, text="Export Network", style="secondary",
            command=self._export_network, theme=self.theme_name,
        ).pack(fill=tk.X, pady=(8, 0))
    
    def _create_results(self):
        """Create the results panel."""
        self.results_card = tk.Frame(
            self.results_frame, bg=self.theme["bg_card"],
            highlightbackground=self.theme["border"], highlightthickness=1,
        )
        self.results_card.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        header = tk.Frame(self.results_card, bg=self.theme["bg_card"])
        header.pack(fill=tk.X)
        
        tk.Label(
            header, text="Citation Network",
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT, padx=12, pady=8)
        
        # Notebook with tabs
        self.notebook = ttk.Notebook(self.results_card)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Network tab
        self.plot_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.plot_frame, text="📊 Network")
        
        # Main Path tab
        self.main_path_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.main_path_frame, text="🛤️ Main Path")
        
        # Nodes tab
        self.nodes_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.nodes_frame, text="📍 Documents")
        
        # Statistics tab
        self.stats_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.stats_frame, text="📈 Statistics")

        

        # Info tab

        info_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])

        self.notebook.add(info_frame, text="ℹ️ Info")

        self._create_info_content(info_frame)
        
        self._show_placeholder()
    
    def _show_placeholder(self):
        """Show placeholder message."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.plot_frame,
            text="Click 'Build Network' to create\ncitation network or historiograph",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
            justify=tk.CENTER,
        ).pack(expand=True)
    
    def _build_network(self):
        """Build the citation network."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        if not HAS_NETWORKX:
            messagebox.showerror("Missing Package", "NetworkX is required.")
            return
        
        self._show_loading("Building network...")
        
        min_citations = self.min_citations.get()
        top_n = self.top_n.get()
        cutoff = self.match_cutoff.get() / 100.0
        
        def do_build():
            try:
                # Build citation network
                G = self._build_citation_network(min_citations, top_n, cutoff)
                
                if G is None or len(G.nodes()) == 0:
                    raise ValueError("Could not build network. Check if Title, Year, and References columns exist.")
                
                # Handle case with no edges (no internal citations found)
                warning_msg = None
                if len(G.edges()) == 0:
                    warning_msg = "No citations between documents in corpus were found. Showing documents without connections."
                
                # Compute statistics
                stats = self._compute_stats(G)
                
                # Create node dataframe
                node_data = []
                for node in G.nodes():
                    data = G.nodes[node]
                    in_deg = G.in_degree(node) if G.is_directed() else G.degree(node)
                    out_deg = G.out_degree(node) if G.is_directed() else 0
                    
                    node_data.append({
                        "Label": str(node)[:40],
                        "Title": data.get("title", data.get("full_title", str(node)))[:60],
                        "Year": data.get("year", ""),
                        "Citations": data.get("citations", 0),
                        "In-Degree": in_deg,
                        "Out-Degree": out_deg,
                    })
                
                node_df = pd.DataFrame(node_data)
                node_df = node_df.sort_values("Year", ascending=True)
                
                result = {
                    "graph": G,
                    "node_df": node_df,
                    "stats": stats,
                    "network_type": "Citation Network",
                    "warning": warning_msg,
                }
                
                self.after(0, lambda r=result: self._on_build_success(r))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda msg=str(e): self._on_build_error(msg))
        
        threading.Thread(target=do_build, daemon=True).start()
    
    def _build_citation_network(self, min_citations: int, top_n: int, cutoff: float) -> nx.DiGraph:
        """Build citation network using biblium's implementation."""
        from biblium import utilsbib
        
        # Detect database
        db = getattr(self.bib, 'db', None)
        
        # Check if OpenAlex - multiple indicators since column names may be normalized
        # BiblioAnalysis renames 'referenced_works' to 'References' and 'id' to 'unique-id'
        has_openalex_cols = (
            'referenced_works' in self.bib.df.columns or
            'OpenAlex ID' in self.bib.df.columns or
            'ids.openalex' in self.bib.df.columns or
            'cited_by_api_url' in self.bib.df.columns  # OpenAlex-specific column
        )
        is_openalex = (db and db.lower() in ['openalex', 'oa', 'open alex']) or has_openalex_cols
        
        print(f"DEBUG: db={db}, is_openalex={is_openalex}")
        
        if is_openalex:
            # Use OpenAlex direct citation network (exact ID matching, no fuzzy needed)
            print("Using OpenAlex citation network (exact ID matching)")
            
            # Find ID column - BiblioAnalysis renames 'id' to 'unique-id'
            id_col = None
            for col in ['unique-id', 'id', 'OpenAlex ID', 'ids.openalex']:
                if col in self.bib.df.columns:
                    id_col = col
                    break
            
            # Find refs column - BiblioAnalysis renames 'referenced_works' to 'References'
            refs_col = None
            for col in ['referenced_works', 'References']:
                if col in self.bib.df.columns:
                    refs_col = col
                    break
            
            if id_col is None or refs_col is None:
                print(f"WARNING: OpenAlex detected but missing columns: id_col={id_col}, refs_col={refs_col}")
                print(f"Available columns: {list(self.bib.df.columns)[:15]}")
            else:
                try:
                    # Check if References column contains OpenAlex URLs (pipe-separated)
                    sample_ref = self.bib.df[refs_col].dropna().iloc[0] if len(self.bib.df[refs_col].dropna()) > 0 else ""
                    is_openalex_refs = 'openalex.org' in str(sample_ref) or sample_ref.startswith('https://openalex')
                    
                    if is_openalex_refs:
                        print(f"Building OpenAlex network: id_col={id_col}, refs_col={refs_col}")
                        
                        G = utilsbib.build_openalex_citation_network(
                            self.bib.df,
                            id_col=id_col,
                            refs_col=refs_col,
                            keep_largest_component=True,
                            verbose=True,
                        )
                        
                        print(f"OpenAlex network result: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                        
                        if G.number_of_nodes() > 0:
                            return G
                        else:
                            print("OpenAlex network returned 0 nodes, falling back to fuzzy matching")
                    else:
                        print(f"References column doesn't contain OpenAlex URLs, using fuzzy matching")
                        
                except Exception as e:
                    print(f"OpenAlex network failed: {e}, falling back to fuzzy matching")
                    import traceback
                    traceback.print_exc()
        
        # For Scopus/WoS: Use biblium's build_citation_network function (fuzzy matching)
        print("Using fuzzy matching citation network")
        
        # Find appropriate columns
        title_col = None
        for col in ['Title', 'TI', 'title', 'display_name', 'Article Title']:
            if col in self.bib.df.columns:
                title_col = col
                break
        
        ref_col = None
        for col in ['References', 'Cited References', 'CR', 'references']:
            if col in self.bib.df.columns:
                ref_col = col
                break
        
        id_col = None
        for col in ['EID', 'DOI', 'UT', 'id', 'unique-id', 'Accession Number']:
            if col in self.bib.df.columns:
                id_col = col
                break
        
        # If no ID column, create one
        if id_col is None:
            self.bib.df['_doc_id'] = [f"DOC_{i}" for i in range(len(self.bib.df))]
            id_col = '_doc_id'
        
        if not title_col:
            raise ValueError(f"No Title column found. Available: {list(self.bib.df.columns)[:10]}")
        if not ref_col:
            raise ValueError(f"No References column found. Available: {list(self.bib.df.columns)[:10]}")
        
        # Convert cutoff from 0-1 to 0-100 for threshold
        threshold = int(cutoff * 100)
        
        print(f"Building citation network with fuzzy matching: title_col={title_col}, ref_col={ref_col}, id_col={id_col}, threshold={threshold}")
        
        result = utilsbib.build_citation_network(
            self.bib.df,
            title_col=title_col,
            ref_col=ref_col,
            id_col=id_col,
            threshold=threshold,
            use_token_set=True,
            largest_only=True,
            verbose=True,
            return_stats=True,
        )
        
        G, unmatched, stats = result
        
        print(f"Citation network stats: {stats}")
        
        # Add node attributes (year, citations) from dataframe
        year_col = None
        for col in ['Year', 'PY', 'year', 'publication_year']:
            if col in self.bib.df.columns:
                year_col = col
                break
        
        cite_col = None
        for col in ['Times Cited', 'TC', 'Cited by', 'cited_by_count', 'citations']:
            if col in self.bib.df.columns:
                cite_col = col
                break
        
        # Build lookup by ID
        id_to_row = {str(row[id_col]): row for _, row in self.bib.df.iterrows() if pd.notna(row.get(id_col))}
        
        for node in G.nodes():
            if node in id_to_row:
                row = id_to_row[node]
                if year_col and pd.notna(row.get(year_col)):
                    try:
                        G.nodes[node]['year'] = int(row[year_col])
                    except:
                        G.nodes[node]['year'] = 2000
                else:
                    G.nodes[node]['year'] = 2000
                
                if cite_col and pd.notna(row.get(cite_col)):
                    try:
                        G.nodes[node]['citations'] = int(row[cite_col])
                    except:
                        G.nodes[node]['citations'] = 0
                else:
                    G.nodes[node]['citations'] = 0
        
        return G
    
    def _compute_stats(self, G) -> Dict:
        """Compute network statistics."""
        stats = {
            "Nodes": len(G.nodes()),
            "Edges": len(G.edges()),
            "Density": round(nx.density(G), 4),
        }
        
        if G.is_directed():
            stats["Avg In-Degree"] = round(sum(d for n, d in G.in_degree()) / max(len(G.nodes()), 1), 2)
            stats["Avg Out-Degree"] = round(sum(d for n, d in G.out_degree()) / max(len(G.nodes()), 1), 2)
            
            # Count sources and sinks
            sources = sum(1 for n in G.nodes() if G.in_degree(n) == 0)
            sinks = sum(1 for n in G.nodes() if G.out_degree(n) == 0)
            stats["Sources"] = sources
            stats["Sinks"] = sinks
        else:
            stats["Avg Degree"] = round(sum(d for n, d in G.degree()) / max(len(G.nodes()), 1), 2)
        
        # Year range
        years = [G.nodes[n].get('year', 0) for n in G.nodes() if G.nodes[n].get('year', 0) > 0]
        if years:
            stats["Year Range"] = f"{min(years)}-{max(years)}"
        
        return stats
    
    def _show_loading(self, message: str):
        """Show loading message."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.plot_frame,
            text=f"⏳ {message}",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
        ).pack(expand=True)
    
    def _on_build_success(self, result: Dict):
        """Handle successful build."""
        self._graph = result["graph"]
        self._node_df = result["node_df"]
        
        self._show_network_plot(result)
        self._show_main_path_results(result)
        self._show_nodes_table(result["node_df"])
        self._show_stats(result["stats"])
        
        # Show warning if present
        if result.get("warning"):
            messagebox.showwarning("Warning", result["warning"])
        
        event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": "Citation Network"})
    
    def _show_main_path_results(self, result: Dict):
        """Display main path analysis results."""
        for widget in self.main_path_frame.winfo_children():
            widget.destroy()
        
        G = result["graph"]
        
        if G.number_of_edges() == 0:
            tk.Label(
                self.main_path_frame,
                text="Main Path Analysis requires a network with edges.\nNo citation links found between documents.",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
                justify=tk.CENTER,
            ).pack(expand=True)
            return
        
        try:
            from biblium.main_path import compute_main_path_analysis
            
            # Build node_data dict for main path analysis
            node_data = {}
            for node in G.nodes():
                data = G.nodes[node]
                node_data[node] = {
                    'title': data.get('title', data.get('full_title', str(node)))[:50],
                    'year': data.get('year', 2000),
                    'citations': data.get('citations', 0),
                }
            
            method = self.mpa_method.get()
            mpa_result = compute_main_path_analysis(G, method=method, node_data=node_data, verbose=True)
            
            # Store for export
            self._mpa_result = mpa_result
            
            # Header
            tk.Label(
                self.main_path_frame,
                text=f"Main Path Analysis ({method})",
                font=FONTS.get_font("heading", bold=True),
                bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
            ).pack(anchor="w", padx=12, pady=(8, 4))
            
            # Info
            info_text = f"Path length: {mpa_result.path_length} documents  |  Key routes: {len(mpa_result.key_routes)}"
            tk.Label(
                self.main_path_frame,
                text=info_text,
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_secondary"],
            ).pack(anchor="w", padx=12)
            
            # Main path table
            if mpa_result.global_main_path:
                tk.Label(
                    self.main_path_frame,
                    text="Global Main Path:",
                    font=FONTS.get_font("body", bold=True),
                    bg=self.theme["bg_card"],
                    fg=self.theme["text_primary"],
                ).pack(anchor="w", padx=12, pady=(12, 4))
                
                tree_frame = tk.Frame(self.main_path_frame, bg=self.theme["bg_card"])
                tree_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
                
                cols = ("Order", "Document", "Year", "Citations")
                tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=12)
                
                tree.heading("Order", text="#")
                tree.heading("Document", text="Document")
                tree.heading("Year", text="Year")
                tree.heading("Citations", text="Citations")
                
                tree.column("Order", width=40, anchor="center")
                tree.column("Document", width=300, anchor="w")
                tree.column("Year", width=60, anchor="center")
                tree.column("Citations", width=80, anchor="center")
                
                for i, node in enumerate(mpa_result.global_main_path, 1):
                    data = node_data.get(node, {})
                    tree.insert("", tk.END, values=(
                        i,
                        data.get('title', str(node)[:50]),
                        data.get('year', '-'),
                        data.get('citations', '-'),
                    ))
                
                vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
                tree.configure(yscrollcommand=vsb.set)
                tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                vsb.pack(side=tk.RIGHT, fill=tk.Y)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            tk.Label(
                self.main_path_frame,
                text=f"Main Path Analysis Error:\n{str(e)[:100]}",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["danger"],
                justify=tk.CENTER,
            ).pack(expand=True)
    
    def _on_build_error(self, error_msg: str):
        """Handle build error."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.plot_frame,
            text=f"❌ Error: {error_msg}",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["danger"],
            wraplength=400,
        ).pack(expand=True, padx=20, pady=20)
    
    def _show_network_plot(self, result: Dict):
        """Display network visualization."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        if not HAS_MATPLOTLIB:
            tk.Label(
                self.plot_frame, text="Matplotlib required",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            ).pack(expand=True)
            return
        
        G = result["graph"]
        network_type = result.get("network_type", "")
        
        try:
            from PIL import Image, ImageTk
            
            fig = Figure(figsize=(14, 10), dpi=100)
            ax = fig.add_subplot(111)
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')
            
            self._current_fig = fig
            
            # Layout
            layout_name = self.layout_combo.get()
            pos = self._compute_layout(G, layout_name)
            
            # Node sizes based on in-degree or citations
            if G.is_directed():
                degrees = dict(G.in_degree())
            else:
                degrees = dict(G.degree())
            max_deg = max(degrees.values()) if degrees else 1
            sizes = [100 + 400 * degrees.get(n, 0) / max(max_deg, 1) for n in G.nodes()]
            
            # Node colors
            color_by = self.color_by.get()
            colors = self._compute_colors(G, color_by)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color='gray',
                                  arrows=True, arrowsize=8, arrowstyle='->')
            
            # Draw nodes
            nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, 
                                          node_color=colors, alpha=0.8, cmap=plt.cm.viridis)
            
            # Labels
            num_labels = self.num_labels.get()
            if num_labels > 0:
                top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:num_labels]
                labels = {n: str(n)[:25] for n, _ in top_nodes}
                nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7)
            
            # Colorbar
            if colors and isinstance(colors[0], (int, float)):
                sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                           norm=plt.Normalize(vmin=min(colors), vmax=max(colors)))
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, shrink=0.5)
                cbar.set_label(color_by)
            
            title = "Historiograph" if "Historiograph" in network_type else "Citation Network"
            ax.set_title(f"{title}: {len(G.nodes())} documents, {len(G.edges())} citations")
            ax.axis('off')
            fig.tight_layout()
            
            # Render to image using ScaledImageFrame
            from biblium.gui.widgets.plots import ScaledImageFrame
            
            scaled_frame = ScaledImageFrame(
                self.plot_frame, 
                theme=self.theme_name,
                maintain_aspect=True,
                max_scale=1.5
            )
            scaled_frame.pack(fill=tk.BOTH, expand=True)
            scaled_frame.set_image_from_figure(fig, dpi=100)
            
            self._scaled_frame = scaled_frame
            
            # Right-click menu
            def show_menu(event):
                menu = tk.Menu(scaled_frame, tearoff=0)
                menu.add_command(label="📄 Add to Report", command=self._add_plot_to_report)
                menu.add_separator()
                menu.add_command(label="💾 Save as PNG...", command=lambda: self._save_plot("png"))
                menu.add_command(label="💾 Save as PDF...", command=lambda: self._save_plot("pdf"))
                menu.add_command(label="💾 Save as SVG...", command=lambda: self._save_plot("svg"))
                menu.tk_popup(event.x_root, event.y_root)
            scaled_frame.bind("<Button-3>", show_menu)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            tk.Label(
                self.plot_frame,
                text=f"Plot error: {e}",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["danger"],
            ).pack(expand=True)
    
    def _compute_layout(self, G, layout_name: str) -> Dict:
        """Compute network layout."""
        if layout_name == "chronological":
            # Time-based layout: x = year, y = based on connections
            pos = {}
            years = {n: G.nodes[n].get('year', 0) for n in G.nodes()}
            
            if not any(years.values()):
                # Fallback if no year data
                return nx.spring_layout(G, k=2, iterations=50)
            
            min_year = min(y for y in years.values() if y > 0) if any(y > 0 for y in years.values()) else 2000
            max_year = max(years.values()) if years.values() else 2020
            year_range = max(max_year - min_year, 1)
            
            # Group by year
            by_year = {}
            for n, y in years.items():
                if y not in by_year:
                    by_year[y] = []
                by_year[y].append(n)
            
            # Assign positions
            for year, nodes in by_year.items():
                x = (year - min_year) / year_range if year > 0 else 0
                n_nodes = len(nodes)
                for i, node in enumerate(nodes):
                    y_pos = (i + 1) / (n_nodes + 1)
                    pos[node] = (x, y_pos)
            
            return pos
            
        elif layout_name == "kamada_kawai":
            try:
                return nx.kamada_kawai_layout(G)
            except:
                return nx.spring_layout(G)
        elif layout_name == "circular":
            return nx.circular_layout(G)
        else:
            return nx.spring_layout(G, k=2, iterations=50)
    
    def _compute_colors(self, G, color_by: str) -> List:
        """Compute node colors based on attribute."""
        if color_by == "Year":
            return [G.nodes[n].get('year', 0) for n in G.nodes()]
        elif color_by == "Citations":
            return [G.nodes[n].get('citations', 0) for n in G.nodes()]
        elif color_by == "In-Degree":
            if G.is_directed():
                return [G.in_degree(n) for n in G.nodes()]
            else:
                return [G.degree(n) for n in G.nodes()]
        else:
            return [0.5] * len(G.nodes())
    
    def _save_plot(self, fmt: str):
        """Save plot to file."""
        if not self._current_fig:
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=f".{fmt}",
            filetypes=[(f"{fmt.upper()} files", f"*.{fmt}")],
        )
        if filepath:
            try:
                self._current_fig.savefig(filepath, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Saved", f"Saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def _add_plot_to_report(self):
        """Add current plot to report queue."""
        if not self._current_fig:
            messagebox.showinfo("No Plot", "No plot to add to report.")
            return
        
        try:
            from biblium.gui.core.state import report_queue
            
            plot_title = "Citation Network"
            if self._current_fig.axes:
                plot_title = self._current_fig.axes[0].get_title() or "Citation Network"
            
            report_queue.add_plot(
                figure_or_bytes=self._current_fig,
                title=plot_title,
                source_panel=self.title,
            )
            
            messagebox.showinfo(
                "Added to Report",
                f"Plot '{plot_title}' has been added to the report queue.\n\n"
                f"Items in queue: {len(report_queue)}\n\n"
                "Go to Reports → Report Builder to generate your report."
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add to report: {e}")
    
    def _show_nodes_table(self, df: pd.DataFrame):
        """Display nodes table."""
        for widget in self.nodes_frame.winfo_children():
            widget.destroy()
        
        if df is None or len(df) == 0:
            tk.Label(
                self.nodes_frame, text="No document data",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            ).pack(expand=True)
            return
        
        table = DataTable(self.nodes_frame, theme=self.theme_name)
        table.pack(fill=tk.BOTH, expand=True)
        table.set_data(df)
    
    def _show_stats(self, stats: Dict):
        """Display statistics."""
        for widget in self.stats_frame.winfo_children():
            widget.destroy()
        
        if not stats:
            return
        
        grid = CardGrid(self.stats_frame, columns=3, theme=self.theme_name)
        grid.pack(fill=tk.X, padx=8, pady=8)
        
        for key, value in stats.items():
            if isinstance(value, float):
                display_val = f"{value:.4f}"
            else:
                display_val = str(value)
            
            # Create StatsCard with grid as parent (not grid.container)
            card = StatsCard(grid, title=key, value=display_val, theme=self.theme_name)
            grid.add_card(card)
    
    def _export_network(self):
        """Export network to file."""
        if not self._graph:
            messagebox.showwarning("No Network", "Build a network first.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".gexf",
            filetypes=[
                ("GEXF (Gephi)", "*.gexf"),
                ("GraphML", "*.graphml"),
                ("Pajek", "*.net"),
                ("Edge List CSV", "*.csv"),
            ],
        )
        
        if filepath:
            try:
                if filepath.endswith(".gexf"):
                    nx.write_gexf(self._graph, filepath)
                elif filepath.endswith(".graphml"):
                    nx.write_graphml(self._graph, filepath)
                elif filepath.endswith(".net"):
                    nx.write_pajek(self._graph, filepath)
                elif filepath.endswith(".csv"):
                    edges = [(u, v) for u, v in self._graph.edges()]
                    pd.DataFrame(edges, columns=["Source", "Target"]).to_csv(filepath, index=False)
                
                messagebox.showinfo("Success", f"Exported to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
CITATION NETWORK
════════════════

Document-level citation network analysis.

NETWORK TYPES
─────────────
• Direct Citation
  A → B means A cites B
  Shows knowledge flow
  
• Co-citation
  A—B both cited by C
  Intellectual similarity
  
• Bibliographic Coupling
  A—B both cite C
  Similar knowledge base

NODE METRICS
────────────
• In-degree: Times cited
• Out-degree: References made
• Betweenness: Bridge position
• PageRank: Network importance

FILTERING
─────────
• Minimum citations
• Time window
• Top N papers

VISUALIZATION
─────────────
• Network graph
• Historiograph (timeline)
• Cluster coloring

APPLICATIONS
────────────
• Identify key papers
• Find research communities
• Track knowledge evolution
• Detect citation patterns
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

    def destroy(self):
        """Clean up resources."""
        if self._current_fig:
            try:
                plt.close(self._current_fig)
            except:
                pass
        self._current_fig = None
        self._photo_image = None
        super().destroy()
