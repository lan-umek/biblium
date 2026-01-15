# -*- coding: utf-8 -*-
"""
Historiograph Panel
===================
Time-ordered citation network visualization showing knowledge flow.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import io
from typing import Dict, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledSpinbox

try:
    import pandas as pd
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


class HistoriographPanel(BasePanel):
    """Panel for historiograph visualization - time-ordered citation network."""
    
    title = "Historiograph"
    icon = "🕰️"
    description = "Time-ordered citation network showing knowledge flow"
    requires_data = True
    
    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._graph = None
        self._node_df = None
        self._current_fig = None
        self._photo_image = None
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._build_historiograph
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Parameters Card
        params_card = Card(self.options_content, title="⚙️ Parameters", theme=self.theme_name)
        params_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.min_citations = LabeledSpinbox(
            params_card.content, label="Min Citations:",
            from_=0, to=500, default=1,
            theme=self.theme_name, label_width=15,
        )
        self.min_citations.pack(fill=tk.X, pady=4)
        
        self.match_cutoff = LabeledSpinbox(
            params_card.content, label="Match Cutoff (%):",
            from_=50, to=100, default=85,
            theme=self.theme_name, label_width=15,
        )
        self.match_cutoff.pack(fill=tk.X, pady=4)
        
        # Help text
        help_label = tk.Label(
            params_card.content,
            text="Match Cutoff: Similarity threshold for\nfinding citations within the corpus",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_muted"],
            justify=tk.LEFT,
        )
        help_label.pack(anchor="w", pady=2)
        
        # Visualization Card
        viz_card = Card(self.options_content, title="🎨 Visualization", theme=self.theme_name)
        viz_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.top_per_year = LabeledSpinbox(
            viz_card.content, label="Top N per Year:",
            from_=1, to=50, default=1,
            theme=self.theme_name, label_width=14,
        )
        self.top_per_year.pack(fill=tk.X, pady=4)
        
        self.num_labels = LabeledSpinbox(
            viz_card.content, label="Show Labels:",
            from_=0, to=100, default=20,
            theme=self.theme_name, label_width=14,
        )
        self.num_labels.pack(fill=tk.X, pady=4)
        
        self.color_by = LabeledCombobox(
            viz_card.content, label="Color by:",
            values=["Year", "Citations", "In-Degree"],
            default="Year",
            theme=self.theme_name, label_width=14,
        )
        self.color_by.pack(fill=tk.X, pady=4)
        
        self.node_size = LabeledCombobox(
            viz_card.content, label="Node Size:",
            values=["Small", "Medium", "Large", "By Citations"],
            default="Small",
            theme=self.theme_name, label_width=14,
        )
        self.node_size.pack(fill=tk.X, pady=4)
        
        # Run buttons
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Build Historiograph", icon="🕰️",
            command=self._build_historiograph, theme=self.theme_name,
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
            header, text="Historiograph",
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT, padx=12, pady=8)
        
        # Notebook with tabs
        self.notebook = ttk.Notebook(self.results_card)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Network tab
        self.plot_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.plot_frame, text="🕰️ Historiograph")
        
        # Documents tab
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
            text="Click 'Build Historiograph' to create\ntime-ordered citation network",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
            justify=tk.CENTER,
        ).pack(expand=True)
    
    def _build_historiograph(self):
        """Build the historiograph."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        if not HAS_NETWORKX:
            messagebox.showerror("Missing Package", "NetworkX is required.")
            return
        
        self._show_loading("Building historiograph...")
        
        min_citations = self.min_citations.get()
        cutoff = self.match_cutoff.get() / 100.0
        
        def do_build():
            try:
                from biblium import utilsbib
                
                # Detect database
                db = getattr(self.bib, 'db', None)
                separator = getattr(self.bib, 'default_separator', '; ')
                
                print(f"Building historiograph: db={db}, cutoff={cutoff}, min_citations={min_citations}")
                
                # Use biblium's build_historiograph_auto function
                G = utilsbib.build_historiograph_auto(
                    self.bib.df,
                    db=db,
                    cutoff=cutoff,
                    min_citations=min_citations,
                    separator=separator,
                )
                
                if G is None or len(G.nodes()) == 0:
                    raise ValueError("Could not build historiograph. Check if Title, Year, and References columns exist.")
                
                print(f"Historiograph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                
                # Handle case with no edges
                warning_msg = None
                if len(G.edges()) == 0:
                    warning_msg = "No citations between documents in corpus were found."
                
                # Compute statistics
                stats = self._compute_stats(G)
                
                # Create node dataframe
                node_data = []
                for node in G.nodes():
                    data = G.nodes[node]
                    in_deg = G.in_degree(node) if G.is_directed() else G.degree(node)
                    out_deg = G.out_degree(node) if G.is_directed() else 0
                    
                    node_data.append({
                        "Label": str(node)[:50],
                        "Title": data.get("title", data.get("full_title", str(node)))[:60],
                        "Year": data.get("year", ""),
                        "Citations": data.get("Cited by", data.get("citations", 0)),
                        "In-Degree": in_deg,
                        "Out-Degree": out_deg,
                    })
                
                node_df = pd.DataFrame(node_data)
                if len(node_df) > 0:
                    node_df = node_df.sort_values("Year", ascending=True)
                
                result = {
                    "graph": G,
                    "node_df": node_df,
                    "stats": stats,
                    "warning": warning_msg,
                }
                
                self.after(0, lambda r=result: self._on_build_success(r))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.after(0, lambda msg=str(e): self._on_build_error(msg))
        
        threading.Thread(target=do_build, daemon=True).start()
    
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
            stats["Sources (roots)"] = sources
            stats["Sinks (leaves)"] = sinks
        
        # Year range
        years = [G.nodes[n].get('year', 0) for n in G.nodes() if G.nodes[n].get('year', 0) > 0]
        if years:
            stats["Year Range"] = f"{min(years)}-{max(years)}"
            stats["Time Span"] = f"{max(years) - min(years)} years"
        
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
        
        self._show_historiograph_plot(result)
        self._show_nodes_table(result["node_df"])
        self._show_stats(result["stats"])
        
        # Show warning if present
        if result.get("warning"):
            messagebox.showwarning("Warning", result["warning"])
        
        event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": "Historiograph"})
    
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
    
    def _show_historiograph_plot(self, result: Dict):
        """Display historiograph visualization."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        if not HAS_MATPLOTLIB:
            tk.Label(
                self.plot_frame, text="Matplotlib required",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            ).pack(expand=True)
            return
        
        G = result["graph"]
        
        try:
            from PIL import Image, ImageTk
            
            fig = Figure(figsize=(14, 10), dpi=100)
            ax = fig.add_subplot(111)
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')
            
            self._current_fig = fig
            
            # Get top N per year setting
            top_n_per_year = self.top_per_year.get()
            
            # Group nodes by year and select top N per year
            year_groups = {}
            for node in G.nodes():
                year = G.nodes[node].get('year', 2000)
                if pd.isna(year):
                    year = 2000
                year = int(year)
                if year not in year_groups:
                    year_groups[year] = []
                
                # Get score for ranking (in-degree + citations)
                in_deg = G.in_degree(node)
                cites = G.nodes[node].get('Cited by', G.nodes[node].get('citations', 0))
                if pd.isna(cites):
                    cites = 0
                score = in_deg + int(cites)
                year_groups[year].append((node, score, in_deg, cites))
            
            # Select top N per year
            selected_nodes = set()
            for year in sorted(year_groups.keys()):
                nodes_sorted = sorted(year_groups[year], key=lambda x: -x[1])
                for node, score, in_deg, cites in nodes_sorted[:top_n_per_year]:
                    selected_nodes.add(node)
            
            # Create subgraph with selected nodes
            subG = G.subgraph(selected_nodes).copy()
            
            # Compute chronological layout
            pos = self._compute_chronological_layout(subG)
            
            # Node sizes
            size_by = self.node_size.get()
            if size_by == "Small":
                sizes = [30] * len(subG.nodes())
            elif size_by == "Medium":
                sizes = [80] * len(subG.nodes())
            elif size_by == "Large":
                sizes = [150] * len(subG.nodes())
            else:  # By Citations
                sizes = []
                for n in subG.nodes():
                    cites = subG.nodes[n].get('Cited by', subG.nodes[n].get('citations', 0))
                    if pd.isna(cites):
                        cites = 0
                    sizes.append(max(20, min(200, 20 + int(cites) * 3)))
            
            # Node colors
            color_by = self.color_by.get()
            if color_by == "Year":
                colors = [subG.nodes[n].get('year', 2000) for n in subG.nodes()]
                cmap = plt.cm.viridis
            elif color_by == "Citations":
                colors = []
                for n in subG.nodes():
                    c = subG.nodes[n].get('Cited by', subG.nodes[n].get('citations', 0))
                    colors.append(int(c) if pd.notna(c) else 0)
                cmap = plt.cm.YlOrRd
            else:  # In-Degree
                colors = [subG.in_degree(n) for n in subG.nodes()]
                cmap = plt.cm.Blues
            
            # Draw edges from the original graph (connecting selected nodes)
            if subG.number_of_edges() > 0:
                nx.draw_networkx_edges(
                    subG, pos, ax=ax,
                    alpha=0.4,
                    edge_color='#666666',
                    arrows=True,
                    arrowsize=6,
                    width=0.8,
                    connectionstyle="arc3,rad=0.1",
                )
            
            # Also draw "main path" connecting top nodes across years
            years_sorted = sorted(year_groups.keys())
            main_path_edges = []
            prev_top_node = None
            for year in years_sorted:
                # Get top node for this year from selected
                year_selected = [n for n in selected_nodes if subG.nodes[n].get('year', 2000) == year]
                if year_selected:
                    # Sort by score
                    top_node = max(year_selected, key=lambda n: subG.in_degree(n))
                    if prev_top_node is not None and prev_top_node in pos and top_node in pos:
                        main_path_edges.append((prev_top_node, top_node))
                    prev_top_node = top_node
            
            # Draw main path as dashed line
            if main_path_edges:
                for u, v in main_path_edges:
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]
                    ax.plot([x1, x2], [y1, y2], 'r--', alpha=0.3, linewidth=1, zorder=1)
            
            # Draw nodes
            nodes = nx.draw_networkx_nodes(
                subG, pos, ax=ax,
                node_size=sizes,
                node_color=colors,
                cmap=cmap,
                alpha=0.85,
                edgecolors='white',
                linewidths=0.5,
            )
            
            # Add colorbar
            if len(set(colors)) > 1:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(colors), vmax=max(colors)))
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
                cbar.set_label(color_by)
            
            # Labels for top nodes
            num_labels = self.num_labels.get()
            if num_labels > 0:
                # Get top nodes by in-degree
                node_scores = [(n, subG.in_degree(n)) for n in subG.nodes()]
                node_scores.sort(key=lambda x: -x[1])
                top_nodes = [n for n, _ in node_scores[:num_labels]]
                
                labels = {n: str(n)[:30] for n in top_nodes}
                nx.draw_networkx_labels(
                    subG, pos, labels, ax=ax,
                    font_size=6,
                    font_weight='bold',
                )
            
            # Title and info
            n_nodes = subG.number_of_nodes()
            n_edges = subG.number_of_edges()
            total_nodes = G.number_of_nodes()
            ax.set_title(f"Historiograph: {n_nodes} documents (top {top_n_per_year}/year from {total_nodes}), {n_edges} citations", 
                        fontsize=11, fontweight='bold')
            
            # Add year axis
            if year_groups:
                min_year = min(year_groups.keys())
                max_year = max(year_groups.keys())
                ax.set_xlim(-0.05, 1.05)
                ax.set_xlabel(f"← {min_year}  ―――  Year  ―――  {max_year} →", fontsize=10)
            
            ax.axis('off')
            
            fig.tight_layout()
            
            # Render to image using ScaledImageFrame
            from biblium.gui.widgets.plots import ScaledImageFrame
            
            scaled_frame = ScaledImageFrame(
                self.plot_frame, 
                theme=self.theme_name,
            )
            scaled_frame.pack(fill=tk.BOTH, expand=True)
            scaled_frame.set_image_from_figure(fig)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            tk.Label(
                self.plot_frame,
                text=f"Visualization error: {str(e)[:100]}",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["danger"],
            ).pack(expand=True)
    
    def _compute_chronological_layout(self, G) -> Dict:
        """Compute chronological layout: x = year, y = spread within year."""
        pos = {}
        
        # Group nodes by year
        year_groups = {}
        for node in G.nodes():
            year = G.nodes[node].get('year', 2000)
            if pd.isna(year):
                year = 2000
            year = int(year)
            if year not in year_groups:
                year_groups[year] = []
            year_groups[year].append(node)
        
        if not year_groups:
            return nx.spring_layout(G)
        
        # Get year range
        min_year = min(year_groups.keys())
        max_year = max(year_groups.keys())
        year_range = max(1, max_year - min_year)
        
        # Position nodes
        for year, nodes in year_groups.items():
            x = (year - min_year) / year_range
            
            # Sort nodes within year by in-degree (most cited at center)
            nodes_sorted = sorted(nodes, key=lambda n: -G.in_degree(n))
            n_nodes = len(nodes_sorted)
            
            for i, node in enumerate(nodes_sorted):
                # Spread nodes vertically within the year
                if n_nodes == 1:
                    y = 0.5
                else:
                    y = (i + 0.5) / n_nodes
                
                pos[node] = (x, y)
        
        return pos
    
    def _show_nodes_table(self, df: pd.DataFrame):
        """Display nodes table."""
        for widget in self.nodes_frame.winfo_children():
            widget.destroy()
        
        if df is None or df.empty:
            tk.Label(
                self.nodes_frame, text="No data",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            ).pack(expand=True)
            return
        
        # Create treeview
        tree_frame = tk.Frame(self.nodes_frame, bg=self.theme["bg_card"])
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        cols = list(df.columns)
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=20)
        
        for col in cols:
            tree.heading(col, text=col)
            width = 150 if col in ["Label", "Title"] else 80
            tree.column(col, width=width, anchor="center" if col != "Title" else "w")
        
        for _, row in df.iterrows():
            values = [str(v)[:50] if isinstance(v, str) else v for v in row]
            tree.insert("", tk.END, values=values)
        
        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
    
    def _show_stats(self, stats: Dict):
        """Display statistics."""
        for widget in self.stats_frame.winfo_children():
            widget.destroy()
        
        # Create stats display
        stats_text = tk.Text(
            self.stats_frame,
            font=FONTS.get_font("code"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            relief=tk.FLAT,
            padx=16,
            pady=16,
            height=20,
        )
        stats_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        stats_text.insert(tk.END, "Historiograph Statistics\n")
        stats_text.insert(tk.END, "=" * 40 + "\n\n")
        
        for key, value in stats.items():
            stats_text.insert(tk.END, f"{key}: {value}\n")
        
        stats_text.config(state=tk.DISABLED)
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
HISTORIOGRAPH
═════════════

Time-ordered citation network visualization.

WHAT IT SHOWS
─────────────
• Papers arranged by publication year
• Citation links between papers
• Knowledge flow over time
• Field evolution trajectory

COMPONENTS
──────────
• Nodes: Documents (sized by citations)
• Edges: Citation relationships
• X-axis: Time (publication year)
• Y-axis: Arranged for visibility

NODE ATTRIBUTES
───────────────
• Size: Citation count
• Color: Time period or cluster
• Label: First author, year

FILTERING
─────────
• Minimum citations: Quality filter
• Time range: Focus period
• Top N: Main contributors

INTERPRETATION
──────────────
• Early nodes: Foundational works
• Many incoming: Influential papers
• Bridges: Connect eras
• Clusters: Research communities

USE CASES
─────────
• Field history visualization
• Identify seminal works
• Track paradigm development
• Find main knowledge paths
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

    def _export_network(self):
        """Export network to file."""
        if self._graph is None:
            messagebox.showwarning("No Data", "Build a historiograph first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".graphml",
            filetypes=[
                ("GraphML", "*.graphml"),
                ("GEXF", "*.gexf"),
                ("Pajek", "*.net"),
                ("Edge List", "*.csv"),
            ],
            title="Export Historiograph",
        )
        
        if not file_path:
            return
        
        try:
            ext = file_path.lower().split('.')[-1]
            
            if ext == "graphml":
                nx.write_graphml(self._graph, file_path)
            elif ext == "gexf":
                nx.write_gexf(self._graph, file_path)
            elif ext == "net":
                nx.write_pajek(self._graph, file_path)
            elif ext == "csv":
                edges = pd.DataFrame(list(self._graph.edges()), columns=["source", "target"])
                edges.to_csv(file_path, index=False)
            
            messagebox.showinfo("Export Complete", f"Historiograph exported to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
