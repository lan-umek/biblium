# -*- coding: utf-8 -*-
"""
Network Panel
=============
Unified network analysis panel for co-authorship, keywords, countries, citations.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from typing import Dict, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme, NETWORK_TYPES, NETWORK_LAYOUTS, COMMUNITY_ALGORITHMS
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, CollapsibleCard, StatsCard, CardGrid
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox, LabeledSpinbox, LabeledEntry, DualListSelector
from biblium.gui.widgets.tables import DataTable
from biblium.gui.widgets.plots import PlotFrame

try:
    import pandas as pd
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class NetworkPanel(BasePanel):
    """Unified panel for network analysis."""
    
    title = "Co-occurrence Networks"
    icon = "🔗"
    description = "Analyze collaboration, co-occurrence, and citation networks"
    requires_data = True
    
    def __init__(self, parent, network_type: str = "coauthorship", theme: str = "light", bib=None, **kwargs):
        self.network_type = network_type
        self._graph = None
        self._node_df = None
        self._edge_df = None
        
        # Set title based on network type
        if network_type in NETWORK_TYPES:
            self.title = NETWORK_TYPES[network_type]["label"]
            self.icon = NETWORK_TYPES[network_type]["icon"]
            self.description = NETWORK_TYPES[network_type]["description"]
        
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Network Type Card
        type_card = Card(self.options_content, title="🔗 Network Type", theme=self.theme_name)
        type_card.pack(fill=tk.X, padx=8, pady=8)
        
        network_names = [f"{v['icon']} {v['label']}" for k, v in NETWORK_TYPES.items()]
        default_name = f"{NETWORK_TYPES.get(self.network_type, {}).get('icon', '')} {NETWORK_TYPES.get(self.network_type, {}).get('label', 'Co-authorship')}"
        
        self.network_combo = LabeledCombobox(
            type_card.content, label="Type:",
            values=network_names, default=default_name,
            theme=self.theme_name, label_width=12,
        )
        self.network_combo.pack(fill=tk.X, pady=4)
        self.network_combo.combo.bind("<<ComboboxSelected>>", self._on_network_type_changed)
        
        # Variable Selection Card - Two columns with move buttons
        var_card = Card(self.options_content, title="📊 Variable Selection", theme=self.theme_name)
        var_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.variable_selector = DualListSelector(
            var_card.content,
            available_label="Available",
            selected_label="Selected",
            available_color="#c0392b",  # Red for available
            selected_color="#27ae60",   # Green for selected
            height=10,
            theme=self.theme_name,
        )
        self.variable_selector.pack(fill=tk.BOTH, expand=True, pady=4)
        
        # Populate available variables when data is loaded
        self._populate_available_variables()
        
        # Node Options Card
        node_card = Card(self.options_content, title="📍 Node Options", theme=self.theme_name)
        node_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.min_occurrences = LabeledSpinbox(
            node_card.content, label="Min Occurrences:",
            from_=1, to=100, default=2,
            theme=self.theme_name, label_width=15,
        )
        self.min_occurrences.pack(fill=tk.X, pady=4)
        
        self.max_nodes = LabeledSpinbox(
            node_card.content, label="Max Nodes:",
            from_=10, to=500, default=100,
            theme=self.theme_name, label_width=15,
        )
        self.max_nodes.pack(fill=tk.X, pady=4)
        
        self.node_size_by = LabeledCombobox(
            node_card.content, label="Size by:",
            values=["Occurrences", "Degree", "Betweenness", "Closeness"],
            default="Occurrences",
            theme=self.theme_name, label_width=15,
        )
        self.node_size_by.pack(fill=tk.X, pady=4)
        
        self.node_color_by = LabeledCombobox(
            node_card.content, label="Color by:",
            values=["Community", "Average Year", "Occurrences", "Degree", "Betweenness"],
            default="Average Year",
            theme=self.theme_name, label_width=15,
        )
        self.node_color_by.pack(fill=tk.X, pady=4)
        
        # Edge Options Card
        edge_card = Card(self.options_content, title="🔗 Edge Options", theme=self.theme_name)
        edge_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.min_edge_weight = LabeledSpinbox(
            edge_card.content, label="Min Edge Weight:",
            from_=1, to=50, default=1,
            theme=self.theme_name, label_width=15,
        )
        self.min_edge_weight.pack(fill=tk.X, pady=4)
        
        self.normalize_weights = LabeledCheckbox(
            edge_card.content, label="Normalize edge weights",
            default=False, theme=self.theme_name,
        )
        self.normalize_weights.pack(fill=tk.X, pady=2)
        
        # Layout Options Card
        layout_card = Card(self.options_content, title="📐 Layout", theme=self.theme_name)
        layout_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.layout_combo = LabeledCombobox(
            layout_card.content, label="Algorithm:",
            values=NETWORK_LAYOUTS,
            default="spring",
            theme=self.theme_name, label_width=15,
        )
        self.layout_combo.pack(fill=tk.X, pady=4)
        
        # Synonyms Card - for merging similar keywords
        synonyms_card = CollapsibleCard(
            self.options_content, title="Synonyms (Merge Keywords)",
            collapsed=True, theme=self.theme_name,
        )
        synonyms_card.pack(fill=tk.X, padx=8, pady=8)
        
        tk.Label(
            synonyms_card.content, 
            text="Format: keyword1 = keyword2\n(one per line)",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"], fg=self.theme["text_muted"],
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 4))
        
        self.synonyms_text = tk.Text(
            synonyms_card.content, height=5, width=30,
            font=FONTS.get_font("body"),
            wrap=tk.WORD,
        )
        self.synonyms_text.pack(fill=tk.X, pady=4)
        
        # Community Detection
        community_card = CollapsibleCard(
            self.options_content, title="Community Detection",
            collapsed=True, theme=self.theme_name,
        )
        community_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.detect_communities = LabeledCheckbox(
            community_card.content, label="Detect communities",
            default=True, theme=self.theme_name,
        )
        self.detect_communities.pack(fill=tk.X, pady=2)
        
        self.community_algo = LabeledCombobox(
            community_card.content, label="Algorithm:",
            values=COMMUNITY_ALGORITHMS,
            default="louvain",
            theme=self.theme_name, label_width=15,
        )
        self.community_algo.pack(fill=tk.X, pady=4)
        
        # Main Path Analysis (for Document Citation networks)
        mainpath_card = CollapsibleCard(
            self.options_content, title="Main Path Analysis",
            collapsed=True, theme=self.theme_name,
        )
        mainpath_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.show_main_path = LabeledCheckbox(
            mainpath_card.content, label="Highlight main path",
            default=True, theme=self.theme_name,
        )
        self.show_main_path.pack(fill=tk.X, pady=2)
        
        self.main_path_method = LabeledCombobox(
            mainpath_card.content, label="Method:",
            values=["SPC (Search Path Count)", "SPLC (Search Path Link Count)", "SPNP (Search Path Node Pair)"],
            default="SPC (Search Path Count)",
            theme=self.theme_name, label_width=15,
        )
        self.main_path_method.pack(fill=tk.X, pady=4)
        
        self.main_path_type = LabeledCombobox(
            mainpath_card.content, label="Path Type:",
            values=["Global Main Path", "Local Main Path", "Key-Route Main Path"],
            default="Global Main Path",
            theme=self.theme_name, label_width=15,
        )
        self.main_path_type.pack(fill=tk.X, pady=4)
        
        # Run Buttons
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Build Network", icon="🔗",
            command=self._build_network, theme=self.theme_name,
        ).pack(fill=tk.X)
        
        ThemedButton(
            btn_frame, text="Export Network", style="secondary",
            command=self._export_network, theme=self.theme_name,
        ).pack(fill=tk.X, pady=(8, 0))
    
    def _populate_available_variables(self):
        """Populate available variables based on network type and data."""
        # Must be called from main thread - schedule if not
        try:
            if not hasattr(self, 'variable_selector'):
                return
                
            self.variable_selector.clear()
            
            if not self.bib:
                return
            
            network_name = self.network_combo.get().lower()
            
            # For Document Citation, use documents (titles) instead of entity columns
            if "document" in network_name and "citation" in network_name:
                # Use document titles or IDs
                title_col = self.bib.mapping.get("Title", "Title")
                if title_col in self.bib.df.columns:
                    # Get documents with their citation counts
                    cit_col = self.bib.mapping.get("Cited_by", "Cited by")
                    year_col = self.bib.mapping.get("Year", "Year")
                    
                    docs = []
                    for idx, row in self.bib.df.iterrows():
                        title = str(row.get(title_col, ""))[:50]  # Truncate long titles
                        year = row.get(year_col, "")
                        cits = row.get(cit_col, 0) if cit_col in self.bib.df.columns else 0
                        if title:
                            label = f"{title}... ({year}, {cits} cit)" if len(str(row.get(title_col, ""))) > 50 else f"{title} ({year}, {cits} cit)"
                            docs.append((label, cits if cits else 0))
                    
                    # Sort by citations and take top items
                    docs.sort(key=lambda x: x[1], reverse=True)
                    available_items = [d[0] for d in docs[:200]]
                    
                    # Select top 30 by default for document networks
                    top_30 = available_items[:30]
                    rest = available_items[30:]
                    
                    self.variable_selector.set_available_items(rest)
                    self.variable_selector.set_selected_items(top_30)
                return
            
            # Determine which column to use based on network type
            if "keyword" in network_name:
                col = self.bib.mapping.get("Author_Keywords", "Author Keywords")
                separator = ";"
            elif "author" in network_name:
                col = self.bib.mapping.get("Authors", "Authors")
                separator = ";"
            elif "country" in network_name or "countr" in network_name:
                col = "Country" if "Country" in self.bib.df.columns else self.bib.mapping.get("Countries", "Countries")
                separator = ";"
            elif "citation" in network_name:
                col = self.bib.mapping.get("References", "References")
                separator = ";"
            else:
                col = self.bib.mapping.get("Authors", "Authors")
                separator = ";"
            
            if col not in self.bib.df.columns:
                return
            
            # Get unique items and their counts
            items = self.bib.df[col].dropna().str.split(separator).explode().str.strip()
            items = items[items != ""]
            item_counts = items.value_counts()
            
            # Show top items with counts
            top_items = item_counts.head(200)
            available_items = [f"{item} ({count})" for item, count in top_items.items()]
            
            # Select top 20 by default, rest go to available
            top_20 = available_items[:20]
            rest = available_items[20:]
            
            self.variable_selector.set_available_items(rest)
            self.variable_selector.set_selected_items(top_20)
        except Exception as e:
            print(f"Error populating variables: {e}")
    
    def _on_network_type_changed(self, event=None):
        """Handle network type change - repopulate variables."""
        self._populate_available_variables()
    
    def _get_selected_variables(self):
        """Get list of selected variable names (without counts)."""
        items = self.variable_selector.get_selected()
        variables = []
        for item in items:
            # Remove the count part " (123)"
            if " (" in item:
                var_name = item.rsplit(" (", 1)[0]
            else:
                var_name = item
            variables.append(var_name)
        return variables
    
    def _on_dataset_loaded(self, data):
        """Handle dataset loaded event - populate available variables."""
        super()._on_dataset_loaded(data)
        # Schedule for main thread to avoid threading issues
        self.after(100, self._populate_available_variables)
    
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
            header, text="Network Visualization",
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            padx=12, pady=8,
        ).pack(side=tk.LEFT)
        
        ttk.Separator(self.results_card, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12)
        
        # Create results_content for compatibility with base class methods
        self.results_content = tk.Frame(self.results_card, bg=self.theme["bg_card"])
        self.results_content.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Notebook inside results_content
        self.notebook = ttk.Notebook(self.results_content)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Network plot tab
        self.plot_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.plot_frame, text="  🔗 Network  ")
        
        # Nodes table tab
        self.nodes_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.nodes_frame, text="  📍 Nodes  ")
        
        # Edges table tab
        self.edges_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.edges_frame, text="  🔗 Edges  ")
        
        # Statistics tab
        self.stats_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.stats_frame, text="  📊 Statistics  ")
        
        # Show initial placeholder in plot frame
        tk.Label(
            self.plot_frame,
            text="Configure options and click 'Build Network'",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_muted"],
        ).pack(expand=True)
    
    def _show_loading(self, message: str = "Processing..."):
        """Show loading state in plot frame."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.plot_frame,
            text=f"⏳ {message}",
            font=FONTS.get_font("heading"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(expand=True)
    
    def _build_network(self):
        """Build the network."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        if not HAS_NETWORKX:
            messagebox.showerror("Missing Package", "NetworkX is required for network analysis.")
            return
        
        # Get selected variables
        selected_vars = self._get_selected_variables()
        if len(selected_vars) < 2:
            messagebox.showwarning("Selection Required", "Please select at least 2 variables to build a network.")
            return
        
        self._show_loading("Building network...")
        
        network_name = self.network_combo.get()
        
        # Check if this is a document citation network
        if "document" in network_name.lower() and "citation" in network_name.lower():
            self._build_document_citation_network(selected_vars)
            return
        
        def do_build():
            try:
                df = self.bib.df
                
                # Determine column based on network type
                if "authorship" in network_name.lower():
                    col = self.bib.mapping.get("Authors", "Authors")
                    separator = ";"
                elif "keyword" in network_name.lower():
                    col = self.bib.mapping.get("Author_Keywords", "Author Keywords")
                    separator = ";"
                elif "country" in network_name.lower() or "countries" in network_name.lower():
                    col = "Country" if "Country" in df.columns else self.bib.mapping.get("Countries", "Countries")
                    separator = ";"
                elif "citation" in network_name.lower():
                    col = self.bib.mapping.get("References", "References")
                    separator = ";"
                else:
                    col = self.bib.mapping.get("Authors", "Authors")
                    separator = ";"
                
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in dataset")
                
                # Parse synonyms
                synonyms = {}
                synonyms_text = self.synonyms_text.get("1.0", tk.END).strip()
                if synonyms_text:
                    for line in synonyms_text.split("\n"):
                        if "=" in line:
                            parts = line.split("=", 1)
                            if len(parts) == 2:
                                key = parts[0].strip().lower()
                                value = parts[1].strip()
                                synonyms[key] = value
                
                # Function to apply synonyms
                def apply_synonym(item):
                    item_lower = item.lower()
                    return synonyms.get(item_lower, item)
                
                # Use selected variables only
                min_weight = self.min_edge_weight.get()
                
                # Get all items and their counts (with synonyms applied)
                items = df[col].dropna().str.split(separator).explode().str.strip()
                items = items[items != ""]
                
                # Apply synonyms to items
                if synonyms:
                    items = items.apply(apply_synonym)
                
                item_counts = items.value_counts()
                
                # Apply synonyms to selected variables too (create new list to avoid modifying original)
                if synonyms:
                    selected_vars_mapped = [apply_synonym(v) for v in selected_vars]
                    # Remove duplicates while preserving order
                    seen = set()
                    selected_vars_final = []
                    for v in selected_vars_mapped:
                        if v not in seen:
                            seen.add(v)
                            selected_vars_final.append(v)
                else:
                    selected_vars_final = selected_vars
                
                # Filter to selected variables only
                valid_items = [v for v in selected_vars_final if v in item_counts.index]
                
                if len(valid_items) < 2:
                    raise ValueError("Not enough selected items found in the data")
                
                # Build graph
                G = nx.Graph()
                
                # Get year column for average year calculation
                year_col = self.bib.mapping.get("Year", "Year")
                
                # Add nodes with attributes
                for item in valid_items:
                    G.add_node(item, occurrences=int(item_counts[item]))
                
                # Track years for each item for average year calculation
                item_years = {item: [] for item in valid_items}
                
                # Add edges (co-occurrence) and collect years
                for _, row in df.iterrows():
                    if pd.isna(row[col]):
                        continue
                    
                    # Apply synonyms to row items
                    row_items_raw = [x.strip() for x in str(row[col]).split(separator)]
                    if synonyms:
                        row_items_raw = [apply_synonym(x) for x in row_items_raw]
                    row_items = [x for x in row_items_raw if x in valid_items]
                    
                    # Collect year for each item
                    if year_col in df.columns and not pd.isna(row.get(year_col)):
                        year = row[year_col]
                        for item in row_items:
                            item_years[item].append(year)
                    
                    for i, item1 in enumerate(row_items):
                        for item2 in row_items[i+1:]:
                            if G.has_edge(item1, item2):
                                G[item1][item2]["weight"] += 1
                            else:
                                G.add_edge(item1, item2, weight=1)
                
                # Add average year to node attributes
                for item in valid_items:
                    if item in G.nodes() and item_years[item]:
                        avg_year = sum(item_years[item]) / len(item_years[item])
                        G.nodes[item]["avg_year"] = round(avg_year, 1)
                    elif item in G.nodes():
                        G.nodes[item]["avg_year"] = 0
                
                # Filter edges by minimum weight
                edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d.get("weight", 1) < min_weight]
                G.remove_edges_from(edges_to_remove)
                
                # Remove isolated nodes
                G.remove_nodes_from(list(nx.isolates(G)))
                
                if len(G.nodes()) < 2:
                    raise ValueError("Network has too few connected nodes after filtering")
                
                # Compute centrality metrics
                degree_cent = nx.degree_centrality(G)
                betweenness_cent = nx.betweenness_centrality(G)
                closeness_cent = nx.closeness_centrality(G)
                
                for node in G.nodes():
                    G.nodes[node]["degree"] = G.degree(node)
                    G.nodes[node]["degree_centrality"] = degree_cent[node]
                    G.nodes[node]["betweenness"] = betweenness_cent[node]
                    G.nodes[node]["closeness"] = closeness_cent[node]
                
                # Community detection
                communities = {}
                if self.detect_communities.get():
                    algo = self.community_algo.get()
                    try:
                        if algo == "louvain":
                            try:
                                from community import community_louvain
                                partition = community_louvain.best_partition(G)
                                communities = partition
                            except ImportError:
                                communities = {node: 0 for node in G.nodes()}
                        elif algo == "greedy_modularity":
                            comms = nx.algorithms.community.greedy_modularity_communities(G)
                            for i, comm in enumerate(comms):
                                for node in comm:
                                    communities[node] = i
                        elif algo == "label_propagation":
                            comms = nx.algorithms.community.label_propagation_communities(G)
                            for i, comm in enumerate(comms):
                                for node in comm:
                                    communities[node] = i
                        else:
                            communities = {node: 0 for node in G.nodes()}
                    except:
                        communities = {node: 0 for node in G.nodes()}
                    
                    for node in G.nodes():
                        G.nodes[node]["community"] = communities.get(node, 0)
                
                # Compute network statistics
                stats = {
                    "Nodes": len(G.nodes()),
                    "Edges": len(G.edges()),
                    "Density": round(nx.density(G), 4),
                    "Avg Degree": round(sum(dict(G.degree()).values()) / len(G.nodes()), 2),
                    "Avg Clustering": round(nx.average_clustering(G), 4),
                }
                
                if nx.is_connected(G):
                    stats["Diameter"] = nx.diameter(G)
                    stats["Avg Path Length"] = round(nx.average_shortest_path_length(G), 2)
                else:
                    stats["Components"] = nx.number_connected_components(G)
                
                if communities:
                    stats["Communities"] = len(set(communities.values()))
                
                # Create DataFrames
                node_data = []
                for node in G.nodes():
                    node_data.append({
                        "Node": node,
                        "Occurrences": G.nodes[node].get("occurrences", 0),
                        "Avg Year": G.nodes[node].get("avg_year", 0),
                        "Degree": G.nodes[node].get("degree", 0),
                        "Betweenness": round(G.nodes[node].get("betweenness", 0), 4),
                        "Closeness": round(G.nodes[node].get("closeness", 0), 4),
                        "Community": G.nodes[node].get("community", 0),
                    })
                node_df = pd.DataFrame(node_data).sort_values("Degree", ascending=False)
                
                edge_data = []
                for u, v, d in G.edges(data=True):
                    edge_data.append({
                        "Source": u,
                        "Target": v,
                        "Weight": d.get("weight", 1),
                    })
                edge_df = pd.DataFrame(edge_data).sort_values("Weight", ascending=False)
                
                result = {
                    "graph": G,
                    "node_df": node_df,
                    "edge_df": edge_df,
                    "stats": stats,
                    "communities": communities,
                }
                
                self.after(0, lambda r=result: self._on_build_success(r))
            except Exception as e:
                import traceback
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: self._on_build_error(msg))
        
        threading.Thread(target=do_build, daemon=True).start()
    
    def _build_document_citation_network(self, selected_vars):
        """Build document citation network with main path analysis."""
        def do_build():
            try:
                df = self.bib.df
                title_col = self.bib.mapping.get("Title", "Title")
                ref_col = self.bib.mapping.get("References", "References")
                year_col = self.bib.mapping.get("Year", "Year")
                cit_col = self.bib.mapping.get("Cited_by", "Cited by")
                
                # Extract selected document titles (strip the year/citation suffix)
                selected_titles = []
                for var in selected_vars:
                    # Remove " (year, N cit)" suffix
                    if " (" in var:
                        title = var.rsplit(" (", 1)[0]
                    else:
                        title = var
                    # Handle truncated titles
                    if title.endswith("..."):
                        title = title[:-3]
                    selected_titles.append(title)
                
                # Build directed citation graph
                G = nx.DiGraph()
                
                # Map titles to row indices for matching
                title_to_idx = {}
                for idx, row in df.iterrows():
                    title = str(row.get(title_col, ""))
                    if title:
                        # Check if this title matches any selected title (prefix match for truncated)
                        for sel_title in selected_titles:
                            if title.startswith(sel_title) or sel_title.startswith(title[:50]):
                                short_title = title[:40] + "..." if len(title) > 40 else title
                                year = row.get(year_col, "")
                                cits = row.get(cit_col, 0) if cit_col in df.columns else 0
                                G.add_node(idx, title=short_title, year=year, citations=cits, full_title=title)
                                title_to_idx[title.lower()] = idx
                                break
                
                if len(G.nodes()) < 2:
                    raise ValueError("Not enough documents matched")
                
                # Build citation edges by checking references
                for idx in G.nodes():
                    row = df.loc[idx]
                    refs = row.get(ref_col, "")
                    if pd.isna(refs):
                        continue
                    
                    # Parse references and check for matches
                    for ref in str(refs).split(";"):
                        ref_clean = ref.strip().lower()
                        # Check if any node title appears in this reference
                        for other_idx in G.nodes():
                            if other_idx == idx:
                                continue
                            other_title = G.nodes[other_idx].get("full_title", "").lower()
                            # Simple matching - check if title words appear in reference
                            title_words = [w for w in other_title.split()[:5] if len(w) > 3]
                            if title_words and all(w in ref_clean for w in title_words[:3]):
                                # This document cites the other document
                                # Edge direction: citing -> cited
                                G.add_edge(idx, other_idx, weight=1)
                                break
                
                # Remove isolated nodes
                isolated = list(nx.isolates(G))
                G.remove_nodes_from(isolated)
                
                if len(G.nodes()) < 2:
                    raise ValueError("No citation connections found between selected documents")
                
                # Calculate Search Path Count (SPC) for main path analysis
                main_path_edges = set()
                if self.show_main_path.get() and len(G.edges()) > 0:
                    main_path_edges = self._calculate_main_path(G)
                
                # Compute node metrics
                in_degree = dict(G.in_degree())
                out_degree = dict(G.out_degree())
                
                try:
                    betweenness = nx.betweenness_centrality(G)
                except:
                    betweenness = {n: 0 for n in G.nodes()}
                
                for node in G.nodes():
                    G.nodes[node]["in_degree"] = in_degree.get(node, 0)
                    G.nodes[node]["out_degree"] = out_degree.get(node, 0)
                    G.nodes[node]["betweenness"] = betweenness.get(node, 0)
                    G.nodes[node]["on_main_path"] = any(node in e for e in main_path_edges)
                
                # Create DataFrames
                node_data = []
                for node in G.nodes():
                    node_data.append({
                        "Title": G.nodes[node].get("title", ""),
                        "Year": G.nodes[node].get("year", ""),
                        "Citations": G.nodes[node].get("citations", 0),
                        "In-Degree": G.nodes[node].get("in_degree", 0),
                        "Out-Degree": G.nodes[node].get("out_degree", 0),
                        "Betweenness": round(G.nodes[node].get("betweenness", 0), 4),
                        "Main Path": "Yes" if G.nodes[node].get("on_main_path") else "No",
                    })
                node_df = pd.DataFrame(node_data).sort_values("Citations", ascending=False)
                
                edge_data = []
                for u, v, d in G.edges(data=True):
                    edge_data.append({
                        "Source": G.nodes[u].get("title", "")[:30],
                        "Target": G.nodes[v].get("title", "")[:30],
                        "Main Path": "Yes" if (u, v) in main_path_edges else "No",
                    })
                edge_df = pd.DataFrame(edge_data)
                
                # Statistics
                stats = {
                    "Documents": len(G.nodes()),
                    "Citations": len(G.edges()),
                    "Main Path Edges": len(main_path_edges),
                    "Avg In-Degree": round(sum(in_degree.values()) / len(G.nodes()), 2) if G.nodes() else 0,
                    "Avg Out-Degree": round(sum(out_degree.values()) / len(G.nodes()), 2) if G.nodes() else 0,
                }
                
                result = {
                    "graph": G,
                    "node_df": node_df,
                    "edge_df": edge_df,
                    "stats": stats,
                    "communities": {},
                    "main_path_edges": main_path_edges,
                    "is_citation_network": True,
                }
                
                self.after(0, lambda r=result: self._on_build_success(r))
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: self._on_build_error(msg))
        
        threading.Thread(target=do_build, daemon=True).start()
    
    def _calculate_main_path(self, G):
        """Calculate main path using SPC (Search Path Count) method."""
        main_path_edges = set()
        
        # Find source nodes (no incoming edges) and sink nodes (no outgoing edges)
        sources = [n for n in G.nodes() if G.in_degree(n) == 0]
        sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
        
        if not sources or not sinks:
            # If no clear sources/sinks, use nodes with min in-degree and max out-degree
            in_degrees = dict(G.in_degree())
            out_degrees = dict(G.out_degree())
            sources = [min(G.nodes(), key=lambda n: in_degrees.get(n, 0))]
            sinks = [max(G.nodes(), key=lambda n: in_degrees.get(n, 0))]
        
        # Calculate SPC (Search Path Count) for each edge
        # SPC(e) = number of paths from sources through e to sinks
        edge_spc = {}
        
        # Count paths from each source to each node
        source_paths = {n: 0 for n in G.nodes()}
        for source in sources:
            source_paths[source] = 1
        
        # Topological traversal (or approximation for cycles)
        try:
            topo_order = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            # Has cycles - use reverse DFS ordering
            topo_order = list(G.nodes())
        
        # Forward pass - count paths from sources
        for node in topo_order:
            for pred in G.predecessors(node):
                source_paths[node] += source_paths.get(pred, 0)
        
        # Count paths from each node to sinks
        sink_paths = {n: 0 for n in G.nodes()}
        for sink in sinks:
            sink_paths[sink] = 1
        
        # Backward pass - count paths to sinks
        for node in reversed(topo_order):
            for succ in G.successors(node):
                sink_paths[node] += sink_paths.get(succ, 0)
        
        # Calculate SPC for each edge
        for u, v in G.edges():
            spc = source_paths.get(u, 1) * sink_paths.get(v, 1)
            edge_spc[(u, v)] = spc
        
        if not edge_spc:
            return main_path_edges
        
        # Global main path: follow highest SPC edges from sources to sinks
        method = self.main_path_type.get()
        
        if "Global" in method:
            # Start from source with most outgoing paths
            current = max(sources, key=lambda n: sum(edge_spc.get((n, s), 0) for s in G.successors(n)))
            visited = set()
            
            while current not in sinks and current not in visited:
                visited.add(current)
                successors = list(G.successors(current))
                if not successors:
                    break
                
                # Choose successor with highest SPC
                best_succ = max(successors, key=lambda s: edge_spc.get((current, s), 0))
                main_path_edges.add((current, best_succ))
                current = best_succ
        
        elif "Local" in method:
            # Local main path: for each node, include edge with highest SPC
            for node in G.nodes():
                successors = list(G.successors(node))
                if successors:
                    best_succ = max(successors, key=lambda s: edge_spc.get((node, s), 0))
                    main_path_edges.add((node, best_succ))
        
        else:  # Key-Route
            # Include edges with SPC above threshold
            if edge_spc:
                max_spc = max(edge_spc.values())
                threshold = max_spc * 0.5  # Top 50% of SPC values
                for edge, spc in edge_spc.items():
                    if spc >= threshold:
                        main_path_edges.add(edge)
        
        return main_path_edges
    
    def _on_build_success(self, result: Dict):
        """Display network results."""
        self._graph = result["graph"]
        self._node_df = result["node_df"]
        self._edge_df = result["edge_df"]
        
        # Show network plot
        self._show_network_plot(result)
        
        # Show nodes table
        self._show_nodes_table(result["node_df"])
        
        # Show edges table
        self._show_edges_table(result["edge_df"])
        
        # Show statistics
        self._show_network_stats(result["stats"])
    
    def _show_network_plot(self, result: Dict):
        """Display network visualization."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        if not HAS_MATPLOTLIB:
            tk.Label(
                self.plot_frame, text="Matplotlib required for visualization",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"],
            ).pack(expand=True)
            return
        
        G = result["graph"]
        communities = result.get("communities", {})
        
        plot = PlotFrame(self.plot_frame, theme=self.theme_name, figsize=(12, 10))
        plot.pack(fill=tk.BOTH, expand=True)
        
        fig, ax = plot.get_figure()
        
        # Layout
        layout_name = self.layout_combo.get()
        if layout_name == "spring":
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout_name == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout_name == "circular":
            pos = nx.circular_layout(G)
        elif layout_name == "shell":
            pos = nx.shell_layout(G)
        elif layout_name == "spectral":
            pos = nx.spectral_layout(G)
        else:
            pos = nx.random_layout(G)
        
        # Node sizes
        size_by = self.node_size_by.get().lower()
        if size_by == "occurrences":
            sizes = [G.nodes[n].get("occurrences", 1) for n in G.nodes()]
        elif size_by == "degree":
            sizes = [G.nodes[n].get("degree", 1) for n in G.nodes()]
        elif size_by == "betweenness":
            sizes = [G.nodes[n].get("betweenness", 0.01) * 1000 for n in G.nodes()]
        else:
            sizes = [G.nodes[n].get("closeness", 0.1) * 100 for n in G.nodes()]
        
        # Normalize sizes
        max_size = max(sizes) if sizes else 1
        min_size = min(sizes) if sizes else 1
        sizes = [100 + 400 * (s - min_size) / (max_size - min_size + 0.001) for s in sizes]
        
        # Node colors based on color_by option
        color_by = self.node_color_by.get()
        
        if color_by == "Community" and communities:
            n_communities = len(set(communities.values()))
            cmap = plt.cm.get_cmap("tab20", max(n_communities, 1))
            colors = [cmap(communities.get(n, 0)) for n in G.nodes()]
        elif color_by == "Average Year":
            avg_years = [G.nodes[n].get("avg_year", 0) for n in G.nodes()]
            if max(avg_years) > min(avg_years):
                # Normalize to 0-1 range
                min_year, max_year = min(avg_years), max(avg_years)
                norm_years = [(y - min_year) / (max_year - min_year) for y in avg_years]
                cmap = plt.cm.get_cmap("viridis")
                colors = [cmap(y) for y in norm_years]
            else:
                colors = [self.theme["accent_primary"]] * len(G.nodes())
        elif color_by == "Occurrences":
            occurrences = [G.nodes[n].get("occurrences", 1) for n in G.nodes()]
            if max(occurrences) > min(occurrences):
                min_occ, max_occ = min(occurrences), max(occurrences)
                norm_occ = [(o - min_occ) / (max_occ - min_occ) for o in occurrences]
                cmap = plt.cm.get_cmap("YlOrRd")
                colors = [cmap(o) for o in norm_occ]
            else:
                colors = [self.theme["accent_primary"]] * len(G.nodes())
        elif color_by == "Degree":
            degrees = [G.nodes[n].get("degree", 1) for n in G.nodes()]
            if max(degrees) > min(degrees):
                min_deg, max_deg = min(degrees), max(degrees)
                norm_deg = [(d - min_deg) / (max_deg - min_deg) for d in degrees]
                cmap = plt.cm.get_cmap("Blues")
                colors = [cmap(0.3 + 0.7 * d) for d in norm_deg]
            else:
                colors = [self.theme["accent_primary"]] * len(G.nodes())
        elif color_by == "Betweenness":
            betweenness = [G.nodes[n].get("betweenness", 0) for n in G.nodes()]
            if max(betweenness) > min(betweenness):
                min_b, max_b = min(betweenness), max(betweenness)
                norm_b = [(b - min_b) / (max_b - min_b) for b in betweenness]
                cmap = plt.cm.get_cmap("Purples")
                colors = [cmap(0.3 + 0.7 * b) for b in norm_b]
            else:
                colors = [self.theme["accent_primary"]] * len(G.nodes())
        else:
            colors = [self.theme["accent_primary"]] * len(G.nodes())
        
        # Edge weights for width
        edge_weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [0.5 + 2.5 * w / max_weight for w in edge_weights]
        
        # Check if this is a citation network with main path
        main_path_edges = result.get("main_path_edges", set())
        is_citation = result.get("is_citation_network", False)
        
        if is_citation and main_path_edges:
            # Draw regular edges first (gray)
            regular_edges = [(u, v) for u, v in G.edges() if (u, v) not in main_path_edges]
            if regular_edges:
                nx.draw_networkx_edges(G, pos, edgelist=regular_edges, ax=ax, 
                                      alpha=0.2, width=0.5, edge_color='gray',
                                      arrows=True, arrowsize=10, arrowstyle='->')
            
            # Draw main path edges (red, thick)
            main_edges = [(u, v) for u, v in G.edges() if (u, v) in main_path_edges]
            if main_edges:
                nx.draw_networkx_edges(G, pos, edgelist=main_edges, ax=ax,
                                      alpha=0.9, width=3, edge_color='red',
                                      arrows=True, arrowsize=15, arrowstyle='->')
            
            # Color nodes: main path nodes in red, others in blue
            node_colors = []
            for n in G.nodes():
                if G.nodes[n].get("on_main_path", False):
                    node_colors.append('red')
                else:
                    node_colors.append(self.theme["accent_primary"])
            
            # Size by citations for citation network
            sizes = [max(50, G.nodes[n].get("citations", 1) * 5) for n in G.nodes()]
            max_size = max(sizes) if sizes else 1
            min_size = min(sizes) if sizes else 1
            sizes = [100 + 400 * (s - min_size) / (max_size - min_size + 0.001) for s in sizes]
            
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=node_colors, alpha=0.8)
            
            # Labels - show titles for citation network
            labels = {n: G.nodes[n].get("title", "")[:25] for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7)
            
            ax.set_title(f"Document Citation Network: {len(G.nodes())} docs, {len(G.edges())} citations, {len(main_path_edges)} main path edges")
        else:
            # Draw network normally
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=edge_widths, edge_color='gray')
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors, alpha=0.8)
            
            # Labels for top nodes
            if self._node_df is not None and "Node" in self._node_df.columns:
                top_nodes = self._node_df.head(15)["Node"].tolist()
            elif self._node_df is not None and "Title" in self._node_df.columns:
                top_nodes = []  # Don't use titles as node names for regular networks
            else:
                top_nodes = []
            labels = {n: str(n)[:20] for n in top_nodes if n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)
            
            ax.set_title(f"Network: {len(G.nodes())} nodes, {len(G.edges())} edges")
        
        ax.axis('off')
        
        fig.tight_layout()
        plot.refresh()
    
    def _show_nodes_table(self, df: pd.DataFrame):
        """Display nodes table."""
        for widget in self.nodes_frame.winfo_children():
            widget.destroy()
        
        table = DataTable(self.nodes_frame, theme=self.theme_name)
        table.pack(fill=tk.BOTH, expand=True)
        table.set_data(df)
    
    def _show_edges_table(self, df: pd.DataFrame):
        """Display edges table."""
        for widget in self.edges_frame.winfo_children():
            widget.destroy()
        
        table = DataTable(self.edges_frame, theme=self.theme_name)
        table.pack(fill=tk.BOTH, expand=True)
        table.set_data(df)
    
    def _show_network_stats(self, stats: Dict):
        """Display network statistics."""
        for widget in self.stats_frame.winfo_children():
            widget.destroy()
        
        grid = CardGrid(self.stats_frame, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X, padx=8, pady=8)
        
        for key, value in stats.items():
            grid.add_card(StatsCard(grid, key, str(value), "📊", self.theme_name))
    
    def _on_build_error(self, error: str):
        """Handle build error."""
        for frame in [self.plot_frame, self.nodes_frame, self.edges_frame, self.stats_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
            
            tk.Label(
                frame, text=f"Error: {error}",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["danger"],
            ).pack(expand=True)
    
    def _export_network(self):
        """Export network to file."""
        if self._graph is None:
            messagebox.showwarning("No Network", "Build a network first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".gexf",
            filetypes=[
                ("GEXF (Gephi)", "*.gexf"),
                ("GraphML", "*.graphml"),
                ("Pajek", "*.net"),
                ("Edge List", "*.csv"),
            ],
            title="Export Network",
        )
        
        if filename:
            try:
                if filename.endswith(".gexf"):
                    nx.write_gexf(self._graph, filename)
                elif filename.endswith(".graphml"):
                    nx.write_graphml(self._graph, filename)
                elif filename.endswith(".net"):
                    nx.write_pajek(self._graph, filename)
                elif filename.endswith(".csv"):
                    self._edge_df.to_csv(filename, index=False)
                
                messagebox.showinfo("Success", f"Network exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
