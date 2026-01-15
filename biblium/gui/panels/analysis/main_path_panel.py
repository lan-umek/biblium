# -*- coding: utf-8 -*-
"""
Main Path Analysis Panel
========================
GUI panel for main path analysis in citation networks.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from biblium.gui.config import FONTS, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.buttons import ActionButton
from biblium.gui.widgets.plots import add_plot_context_menu, make_canvas_resizable

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
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class MainPathPanel(BasePanel):
    """Panel for Main Path Analysis."""
    
    title = "Main Path Analysis"
    icon = "🛤️"
    requires_data = True
    
    def __init__(self, parent, theme="light", **kwargs):
        self._result = None
        self._graph = None
        self._current_fig = None
        self._node_data = {}
        super().__init__(parent, theme=theme, **kwargs)
        event_bus.subscribe(EventBus.DATASET_LOADED, self._handle_data_loaded)
    
    def _handle_data_loaded(self, data):
        bib = data.get("bib") if isinstance(data, dict) else data
        self.on_data_loaded(bib)
    
    def _create_options(self):
        self._add_title()
        
        # Warning for OpenAlex
        warn_frame = tk.Frame(self.options_content, bg="#fff3cd")
        warn_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        tk.Label(warn_frame, text="⚠️ Works best with OpenAlex data\n(uses referenced_works for citations)",
                font=FONTS.get_font("small"), bg="#fff3cd", fg="#856404",
                justify=tk.LEFT).pack(padx=8, pady=4)
        
        # Network building section
        build_frame = tk.LabelFrame(self.options_content, text="🔗 Build Citation Network",
                                    font=FONTS.get_font("body"), bg=self.theme["bg_secondary"],
                                    fg=self.theme["text_primary"])
        build_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        
        # Database type
        db_frame = tk.Frame(build_frame, bg=self.theme["bg_secondary"])
        db_frame.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(db_frame, text="Database:", font=FONTS.get_font("body"),
                bg=self.theme["bg_secondary"], fg=self.theme["text_primary"]).pack(side=tk.LEFT)
        self.db_var = tk.StringVar(value="auto")
        ttk.Combobox(db_frame, textvariable=self.db_var, 
                    values=["auto", "openalex", "scopus", "wos"],
                    state="readonly", width=12).pack(side=tk.LEFT, padx=8)
        
        # Min citations
        cite_frame = tk.Frame(build_frame, bg=self.theme["bg_secondary"])
        cite_frame.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(cite_frame, text="Min Citations:", font=FONTS.get_font("body"),
                bg=self.theme["bg_secondary"], fg=self.theme["text_primary"]).pack(side=tk.LEFT)
        self.min_cite_var = tk.StringVar(value="0")
        ttk.Spinbox(cite_frame, from_=0, to=1000, textvariable=self.min_cite_var, width=8).pack(side=tk.LEFT, padx=8)
        
        # Match cutoff (for non-OpenAlex)
        cutoff_frame = tk.Frame(build_frame, bg=self.theme["bg_secondary"])
        cutoff_frame.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(cutoff_frame, text="Match Cutoff %:", font=FONTS.get_font("body"),
                bg=self.theme["bg_secondary"], fg=self.theme["text_primary"]).pack(side=tk.LEFT)
        self.cutoff_var = tk.StringVar(value="85")
        ttk.Spinbox(cutoff_frame, from_=50, to=100, textvariable=self.cutoff_var, width=8).pack(side=tk.LEFT, padx=8)
        
        # Build button
        ActionButton(build_frame, text="Build Network", icon="🔨", command=self._build_network,
                    theme=self.theme_name).pack(fill=tk.X, padx=8, pady=4)
        
        # Network info
        self.network_info_label = tk.Label(build_frame, text="No network built",
                                           font=FONTS.get_font("small"), bg=self.theme["bg_secondary"],
                                           fg=self.theme["text_muted"])
        self.network_info_label.pack(anchor="w", padx=8, pady=(0, 4))
        
        # Analysis section
        analysis_frame = tk.LabelFrame(self.options_content, text="📊 Main Path Analysis",
                                       font=FONTS.get_font("body"), bg=self.theme["bg_secondary"],
                                       fg=self.theme["text_primary"])
        analysis_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        
        # Weight method
        method_frame = tk.Frame(analysis_frame, bg=self.theme["bg_secondary"])
        method_frame.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(method_frame, text="Weight Method:", font=FONTS.get_font("body"),
                bg=self.theme["bg_secondary"], fg=self.theme["text_primary"]).pack(side=tk.LEFT)
        self.method_var = tk.StringVar(value="SPC")
        ttk.Combobox(method_frame, textvariable=self.method_var, 
                    values=["SPC", "SPLC", "SPNP"],
                    state="readonly", width=10).pack(side=tk.LEFT, padx=8)
        
        # Method help
        tk.Label(analysis_frame, text="SPC: Search Path Count\nSPLC: Normalized by path length\nSPNP: Normalized by node pairs",
                font=FONTS.get_font("small"), bg=self.theme["bg_secondary"],
                fg=self.theme["text_muted"], justify=tk.LEFT).pack(anchor="w", padx=8)
        
        # Visualization section
        viz_frame = tk.LabelFrame(self.options_content, text="🎨 Visualization",
                                  font=FONTS.get_font("body"), bg=self.theme["bg_secondary"],
                                  fg=self.theme["text_primary"])
        viz_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        
        # Layout
        layout_frame = tk.Frame(viz_frame, bg=self.theme["bg_secondary"])
        layout_frame.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(layout_frame, text="Layout:", font=FONTS.get_font("body"),
                bg=self.theme["bg_secondary"], fg=self.theme["text_primary"]).pack(side=tk.LEFT)
        self.layout_var = tk.StringVar(value="chronological")
        ttk.Combobox(layout_frame, textvariable=self.layout_var, 
                    values=["chronological", "spring", "kamada_kawai"],
                    state="readonly", width=14).pack(side=tk.LEFT, padx=8)
        
        # Show labels
        labels_frame = tk.Frame(viz_frame, bg=self.theme["bg_secondary"])
        labels_frame.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(labels_frame, text="Show Labels:", font=FONTS.get_font("body"),
                bg=self.theme["bg_secondary"], fg=self.theme["text_primary"]).pack(side=tk.LEFT)
        self.labels_var = tk.StringVar(value="20")
        ttk.Spinbox(labels_frame, from_=0, to=100, textvariable=self.labels_var, width=8).pack(side=tk.LEFT, padx=8)
        
        # Buttons
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=8)
        
        ActionButton(btn_frame, text="Run Analysis", icon="▶", command=self._run_analysis,
                    theme=self.theme_name).pack(fill=tk.X)
        ActionButton(btn_frame, text="Export Results", icon="📊", command=self._export_results,
                    theme=self.theme_name).pack(fill=tk.X, pady=(4, 0))
    
    def _create_results(self):
        self.results_card = tk.Frame(self.results_frame, bg=self.theme["bg_card"],
                                     highlightbackground=self.theme["border"], highlightthickness=1)
        self.results_card.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        self.notebook = ttk.Notebook(self.results_card)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        self.viz_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.viz_frame, text="📈 Visualization")
        
        self.path_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.path_frame, text="🛤️ Main Path")
        
        self.routes_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.routes_frame, text="🔀 Key Routes")
        
        self.stats_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.stats_frame, text="📊 Statistics")
        
        self.summary_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.summary_frame, text="📝 Summary")
        
        # Info tab
        info_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(info_frame, text="ℹ️ Info")
        self._create_info_content(info_frame)
        
        self._show_placeholder()
    
    def _show_placeholder(self):
        for f in [self.viz_frame, self.path_frame, self.routes_frame, self.stats_frame, self.summary_frame]:
            for w in f.winfo_children():
                w.destroy()
        tk.Label(self.viz_frame, 
                text="Main Path Analysis\n\n"
                     "1. Build Citation Network\n"
                     "2. Run Analysis\n\n"
                     "Identifies the most significant paths\n"
                     "of knowledge diffusion in citations.",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"],
                justify=tk.CENTER).pack(expand=True)
    
    def _build_network(self):
        if not self.bib:
            messagebox.showwarning("No Data", "Load a dataset first.")
            return
        
        db = self.db_var.get()
        min_citations = int(self.min_cite_var.get())
        cutoff = int(self.cutoff_var.get()) / 100.0
        
        try:
            # Check for OpenAlex
            is_openalex = ('unique-id' in self.bib.df.columns or 
                          'referenced_works' in self.bib.df.columns or
                          'cited_by_count' in self.bib.df.columns)
            
            if is_openalex or db == "openalex":
                from biblium.utilsbib import build_openalex_citation_network
                
                # Find ID column
                id_col = None
                for col in ['unique-id', 'id', 'work_id', 'openalex_id']:
                    if col in self.bib.df.columns:
                        id_col = col
                        break
                
                if id_col is None:
                    raise ValueError("No ID column found for OpenAlex network")
                
                G = build_openalex_citation_network(
                    self.bib.df,
                    id_col=id_col,
                    refs_col='referenced_works',
                    keep_largest_component=True,
                )
                
                # Add node data
                self._node_data = {}
                for _, row in self.bib.df.iterrows():
                    node_id = str(row.get(id_col, '')).strip()
                    # Normalize ID
                    if 'W' in node_id:
                        import re
                        m = re.search(r'(W\d+)', node_id)
                        if m:
                            node_id = m.group(1)
                    
                    if node_id in G.nodes():
                        self._node_data[node_id] = {
                            'title': str(row.get('title', row.get('Title', '')))[:50],
                            'year': int(row.get('publication_year', row.get('Year', 2000))),
                            'citations': int(row.get('cited_by_count', row.get('Cited by', 0))),
                        }
                
            else:
                # Use historiograph builder for Scopus/WoS
                from biblium.utilsbib import build_historiograph_auto
                
                G = build_historiograph_auto(
                    self.bib.df,
                    db=db if db != "auto" else None,
                    cutoff=cutoff,
                    min_citations=min_citations,
                )
                
                # Node data is already in graph attributes
                self._node_data = {}
                for node in G.nodes():
                    data = G.nodes[node]
                    self._node_data[node] = {
                        'title': data.get('title', node)[:50],
                        'year': data.get('year', 2000),
                        'citations': data.get('citations', 0),
                    }
            
            self._graph = G
            
            # Update info label
            info = f"✓ Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
            self.network_info_label.config(text=info, fg="green")
            
            print(f"Citation network built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.network_info_label.config(text=f"✗ Error: {e}", fg="red")
            messagebox.showerror("Error", f"Failed to build network:\n{e}")
    
    def _run_analysis(self):
        if self._graph is None:
            messagebox.showwarning("No Network", "Build a citation network first.")
            return
        
        if self._graph.number_of_edges() == 0:
            messagebox.showwarning("No Edges", "Citation network has no edges.\nCannot perform main path analysis.")
            return
        
        method = self.method_var.get()
        
        try:
            from biblium.main_path import compute_main_path_analysis
            
            result = compute_main_path_analysis(
                self._graph,
                method=method,
                node_data=self._node_data,
                verbose=True,
            )
            
            self._result = result
            
            self._update_visualization()
            self._update_path()
            self._update_routes()
            self._update_statistics()
            self._update_summary()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Analysis failed:\n{e}")
    
    def _update_visualization(self):
        for w in self.viz_frame.winfo_children():
            w.destroy()
        
        if not self._result or not HAS_MATPLOTLIB:
            return
        
        try:
            from biblium.main_path import plot_main_path
            
            layout = self.layout_var.get()
            
            fig = plot_main_path(
                self._graph,
                self._result.global_main_path,
                node_data=self._node_data,
                layout=layout,
                figsize=(10, 7),
                title=f"Main Path Analysis ({self._result.weight_method})"
            )
            
            self._current_fig = fig
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
            canvas.draw()
            _canvas_widget = canvas.get_tk_widget()

            _canvas_widget.pack(fill=tk.BOTH, expand=True)
            add_plot_context_menu(canvas_widget, fig)

            add_plot_context_menu(_canvas_widget, fig)
            
            # Save button
            btn_frame = tk.Frame(self.viz_frame, bg=self.theme["bg_card"])
            btn_frame.pack(fill=tk.X, padx=8, pady=4)
            tk.Button(btn_frame, text="Save Plot", command=self._save_plot).pack(side=tk.LEFT)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            tk.Label(self.viz_frame, text=f"Visualization error: {e}",
                    font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                    fg="red").pack(expand=True)
    
    def _update_path(self):
        for w in self.path_frame.winfo_children():
            w.destroy()
        
        if not self._result:
            return
        
        r = self._result
        
        tk.Label(self.path_frame, text=f"Global Main Path ({r.weight_method})",
                font=FONTS.get_font("heading", bold=True), bg=self.theme["bg_card"],
                fg=self.theme["text_primary"]).pack(pady=(8, 4), anchor="w", padx=8)
        
        tk.Label(self.path_frame, text=f"Path length: {r.path_length} documents",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                fg=self.theme["text_secondary"]).pack(anchor="w", padx=8)
        
        # Path table
        tree_frame = tk.Frame(self.path_frame, bg=self.theme["bg_card"])
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        cols = ("Order", "Document", "Year", "Citations")
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=15)
        
        for c in cols:
            tree.heading(c, text=c)
            if c == "Order":
                tree.column(c, width=60, anchor="center")
            elif c == "Document":
                tree.column(c, width=300, anchor="w")
            else:
                tree.column(c, width=80, anchor="center")
        
        for i, node in enumerate(r.global_main_path, 1):
            data = self._node_data.get(node, {})
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
    
    def _update_routes(self):
        for w in self.routes_frame.winfo_children():
            w.destroy()
        
        if not self._result:
            return
        
        r = self._result
        
        tk.Label(self.routes_frame, text="Key Routes (Multiple Significant Paths)",
                font=FONTS.get_font("heading", bold=True), bg=self.theme["bg_card"],
                fg=self.theme["text_primary"]).pack(pady=(8, 4), anchor="w", padx=8)
        
        if not r.key_routes:
            tk.Label(self.routes_frame, text="No key routes found.",
                    font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                    fg=self.theme["text_muted"]).pack(expand=True)
            return
        
        # Canvas for scrolling
        canvas = tk.Canvas(self.routes_frame, bg=self.theme["bg_card"], highlightthickness=0)
        sb = ttk.Scrollbar(self.routes_frame, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=self.theme["bg_card"])
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        
        for i, route in enumerate(r.key_routes, 1):
            route_frame = tk.LabelFrame(inner, text=f"Route {i} ({len(route)} documents)",
                                        font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                                        fg=self.theme["text_primary"])
            route_frame.pack(fill=tk.X, padx=8, pady=4)
            
            # Show path as text
            path_labels = []
            for node in route:
                data = self._node_data.get(node, {})
                label = f"{data.get('title', str(node)[:30])} ({data.get('year', '?')})"
                path_labels.append(label[:40])
            
            path_text = " → ".join(path_labels)
            tk.Label(route_frame, text=path_text, font=FONTS.get_font("small"),
                    bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                    wraplength=600, justify=tk.LEFT).pack(padx=8, pady=4, anchor="w")
    
    def _update_statistics(self):
        for w in self.stats_frame.winfo_children():
            w.destroy()
        
        if not self._result:
            return
        
        r = self._result
        
        tk.Label(self.stats_frame, text="Network & Path Statistics",
                font=FONTS.get_font("heading", bold=True), bg=self.theme["bg_card"],
                fg=self.theme["text_primary"]).pack(pady=(8, 4), anchor="w", padx=8)
        
        # Network stats
        net_frame = tk.LabelFrame(self.stats_frame, text="Network",
                                  font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                                  fg=self.theme["text_primary"])
        net_frame.pack(fill=tk.X, padx=8, pady=4)
        
        stats_text = f"""Nodes: {r.n_nodes}
Edges: {r.n_edges}
Sources (start points): {r.n_sources}
Sinks (end points): {r.n_sinks}"""
        tk.Label(net_frame, text=stats_text, font=FONTS.get_font("code"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                justify=tk.LEFT).pack(padx=8, pady=4, anchor="w")
        
        # Path stats
        path_frame = tk.LabelFrame(self.stats_frame, text="Main Path",
                                   font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                                   fg=self.theme["text_primary"])
        path_frame.pack(fill=tk.X, padx=8, pady=4)
        
        path_text = f"""Method: {r.weight_method}
Path length: {r.path_length}
Key routes: {len(r.key_routes)}
Max edge weight: {r.statistics.get('max_edge_weight', 0):.2f}
Avg edge weight: {r.statistics.get('avg_edge_weight', 0):.4f}"""
        tk.Label(path_frame, text=path_text, font=FONTS.get_font("code"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                justify=tk.LEFT).pack(padx=8, pady=4, anchor="w")
        
        # Top weighted nodes
        if r.node_weights:
            top_frame = tk.LabelFrame(self.stats_frame, text="Most Central Documents (by traversal weight)",
                                      font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                                      fg=self.theme["text_primary"])
            top_frame.pack(fill=tk.X, padx=8, pady=4)
            
            sorted_nodes = sorted(r.node_weights.items(), key=lambda x: -x[1])[:10]
            for node, weight in sorted_nodes:
                data = self._node_data.get(node, {})
                tk.Label(top_frame, 
                        text=f"• {data.get('title', str(node)[:40])} ({data.get('year', '?')}): {weight:.2f}",
                        font=FONTS.get_font("small"), bg=self.theme["bg_card"],
                        fg=self.theme["text_primary"]).pack(anchor="w", padx=8)
    
    def _update_summary(self):
        for w in self.summary_frame.winfo_children():
            w.destroy()
        
        if not self._result:
            return
        
        r = self._result
        
        txt = "MAIN PATH ANALYSIS REPORT\n" + "="*50 + "\n\n"
        txt += f"Weight Method: {r.weight_method}\n"
        txt += f"Network: {r.n_nodes} nodes, {r.n_edges} edges\n"
        txt += f"Sources: {r.n_sources}, Sinks: {r.n_sinks}\n\n"
        
        txt += "GLOBAL MAIN PATH\n" + "-"*30 + "\n"
        txt += f"Length: {r.path_length} documents\n\n"
        
        for i, node in enumerate(r.global_main_path, 1):
            data = self._node_data.get(node, {})
            txt += f"{i}. {data.get('title', str(node))} ({data.get('year', '?')})\n"
        
        txt += f"\nKEY ROUTES\n" + "-"*30 + "\n"
        txt += f"Number of routes: {len(r.key_routes)}\n"
        
        for i, route in enumerate(r.key_routes, 1):
            txt += f"\nRoute {i}: {len(route)} documents\n"
            path_str = " → ".join([str(n)[:20] for n in route])
            txt += f"  {path_str}\n"
        
        txt += f"\nINTERPRETATION\n" + "-"*30 + "\n"
        txt += f"The main path represents the primary trajectory of knowledge\n"
        txt += f"diffusion in this citation network. Documents on the main path\n"
        txt += f"are the most influential in connecting foundational works\n"
        txt += f"to recent developments in the field.\n"
        
        text_w = tk.Text(self.summary_frame, font=FONTS.get_font("code"), bg=self.theme["bg_card"],
                        fg=self.theme["text_primary"], wrap=tk.WORD)
        text_w.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        text_w.insert("1.0", txt)
        text_w.config(state=tk.DISABLED)
    
    def _save_plot(self):
        if not self._current_fig:
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")]
        )
        if filepath:
            self._current_fig.savefig(filepath, dpi=150, bbox_inches='tight')
            messagebox.showinfo("Saved", f"Plot saved to:\n{filepath}")
    
    def _export_results(self):
        if not self._result:
            messagebox.showwarning("No Results", "Run analysis first.")
            return
        
        filepath = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                filetypes=[("Excel","*.xlsx"),("Text","*.txt")])
        if not filepath:
            return
        
        try:
            r = self._result
            if filepath.endswith('.xlsx'):
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    # Main path
                    path_data = []
                    for i, node in enumerate(r.global_main_path, 1):
                        data = self._node_data.get(node, {})
                        path_data.append({
                            'Order': i, 'Node': node,
                            'Title': data.get('title', ''),
                            'Year': data.get('year', ''),
                            'Citations': data.get('citations', ''),
                        })
                    pd.DataFrame(path_data).to_excel(writer, sheet_name='Main_Path', index=False)
                    
                    # Key routes
                    routes_data = []
                    for i, route in enumerate(r.key_routes, 1):
                        routes_data.append({
                            'Route': i,
                            'Length': len(route),
                            'Path': ' → '.join([str(n)[:30] for n in route]),
                        })
                    pd.DataFrame(routes_data).to_excel(writer, sheet_name='Key_Routes', index=False)
                    
                    # Statistics
                    stats_data = [{'Metric': k, 'Value': v} for k, v in r.statistics.items()]
                    pd.DataFrame(stats_data).to_excel(writer, sheet_name='Statistics', index=False)
            else:
                with open(filepath, 'w') as f:
                    for w in self.summary_frame.winfo_children():
                        if isinstance(w, tk.Text):
                            f.write(w.get("1.0", tk.END))
            
            messagebox.showinfo("Exported", f"Saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{e}")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
MAIN PATH ANALYSIS
══════════════════

Trace primary knowledge flow in citations.

ALGORITHMS
──────────
• SPC (Search Path Count)
  Count of paths through edge
  
• SPLC (Search Path Link Count)
  Weighted by link position
  
• SPNP (Search Path Node Pair)
  Node pair traversals

COMPONENTS
──────────
• Source nodes: Origin papers
• Sink nodes: End papers
• Main path: Key trajectory
• Branch paths: Alternatives

OUTPUT
──────
• Main path papers
• Path visualization
• Timeline view
• Branch analysis

INTERPRETATION
──────────────
• Path papers: Core contributions
• Path length: Field development
• Branches: Alternative approaches
• Convergence: Paradigm integration
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

    def on_data_loaded(self, bib):
        if isinstance(bib, dict):
            bib = bib.get("bib", bib)
        self.bib = bib
        self._result = None
        self._graph = None
        self._node_data = {}
        self._current_fig = None
        self.network_info_label.config(text="No network built", fg=self.theme["text_muted"])
        self._show_placeholder()
