# -*- coding: utf-8 -*-
"""
Network Panel
=============
Unified network analysis panel for co-authorship, keywords, countries, citations.

Uses Biblium's built-in network methods for consistency.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from typing import Dict, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme, NETWORK_TYPES, NETWORK_LAYOUTS, COMMUNITY_ALGORITHMS, COLOR_PALETTES
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
    """Unified panel for network analysis using Biblium's built-in methods."""
    
    title = "Co-occurrence Network"
    icon = "🔗"
    description = "Analyze co-occurrence and collaboration networks"
    requires_data = True
    
    def __init__(self, parent, network_type: str = "keyword_cooccurrence", theme: str = "light", bib=None, **kwargs):
        self.network_type = network_type
        self._graph = None
        self._node_df = None
        self._edge_df = None
        self._include_items = set()
        self._exclude_items = set()
        self._current_fig = None
        self._photo_image = None
        
        # Set title based on network type
        if network_type in NETWORK_TYPES:
            self.title = NETWORK_TYPES[network_type]["label"]
            self.icon = NETWORK_TYPES[network_type]["icon"]
            self.description = NETWORK_TYPES[network_type]["description"]
        
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._build_network  # Set primary action for toolbar Run button
    
    def _add_title(self):
        """Add panel title with stored references for updates."""
        title_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        title_frame.pack(fill=tk.X, padx=8, pady=(8, 16))
        
        display_title = f"{self.icon} {self.title}" if self.icon else self.title
        
        self.title_label = tk.Label(
            title_frame,
            text=display_title,
            font=FONTS.get_font("title", bold=True),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_primary"],
        )
        self.title_label.pack(anchor=tk.W)
        
        self.desc_label = tk.Label(
            title_frame,
            text=self.description,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_muted"],
            wraplength=250,
            justify=tk.LEFT,
        )
        self.desc_label.pack(anchor=tk.W, pady=(4, 0))
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Network Type Card
        type_card = Card(self.options_content, title="🔗 Network Type", theme=self.theme_name)
        type_card.pack(fill=tk.X, padx=8, pady=8)
        
        network_names = [f"{v['icon']} {v['label']}" for k, v in NETWORK_TYPES.items()]
        # Default to keyword cooccurrence which is the most common use case
        default_name = f"{NETWORK_TYPES.get('keyword_cooccurrence', {}).get('icon', '🔗')} {NETWORK_TYPES.get('keyword_cooccurrence', {}).get('label', 'Keyword Co-occurrence')}"
        
        self.network_combo = LabeledCombobox(
            type_card.content, label="Type:",
            values=network_names, default=default_name,
            theme=self.theme_name, label_width=12,
        )
        self.network_combo.pack(fill=tk.X, pady=4)
        self.network_combo.combo.bind("<<ComboboxSelected>>", self._on_network_type_changed)
        
        # Node Options Card
        node_card = Card(self.options_content, title="📍 Node Options", theme=self.theme_name)
        node_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.top_n_nodes = LabeledSpinbox(
            node_card.content, label="Top N Nodes:",
            from_=10, to=500, default=50,
            theme=self.theme_name, label_width=15,
        )
        self.top_n_nodes.pack(fill=tk.X, pady=4)
        
        self.min_occurrences = LabeledSpinbox(
            node_card.content, label="Min Occurrences:",
            from_=1, to=100, default=2,
            theme=self.theme_name, label_width=15,
        )
        self.min_occurrences.pack(fill=tk.X, pady=4)
        
        self.num_labels = LabeledSpinbox(
            node_card.content, label="Show Labels:",
            from_=0, to=100, default=20,
            theme=self.theme_name, label_width=15,
        )
        self.num_labels.pack(fill=tk.X, pady=4)
        
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
            default="Community",
            theme=self.theme_name, label_width=15,
        )
        self.node_color_by.pack(fill=tk.X, pady=4)
        
        # Entity Filtering Card
        filter_card = CollapsibleCard(
            self.options_content, title="🔍 Entity Filtering",
            collapsed=True, theme=self.theme_name,
        )
        filter_card.pack(fill=tk.X, padx=8, pady=8)
        
        # Include section
        include_frame = tk.Frame(filter_card.content, bg=self.theme["bg_card"])
        include_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            include_frame, text="Include:",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            width=10, anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        self.include_entry = tk.Entry(
            include_frame, font=FONTS.get_font("body"),
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
        )
        self.include_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        
        self.include_file_btn = tk.Button(
            include_frame, text="📂", font=FONTS.get_font("small"),
            command=self._load_include_file, width=3,
        )
        self.include_file_btn.pack(side=tk.RIGHT)
        
        self.include_regex_var = tk.BooleanVar(value=False)
        include_regex_cb = ttk.Checkbutton(
            filter_card.content,
            text="Use regex for include",
            variable=self.include_regex_var,
        )
        include_regex_cb.pack(anchor=tk.W, pady=2)
        
        # Exclude section
        exclude_frame = tk.Frame(filter_card.content, bg=self.theme["bg_card"])
        exclude_frame.pack(fill=tk.X, pady=4)
        
        tk.Label(
            exclude_frame, text="Exclude:",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            width=10, anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        self.exclude_entry = tk.Entry(
            exclude_frame, font=FONTS.get_font("body"),
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
        )
        self.exclude_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        
        self.exclude_file_btn = tk.Button(
            exclude_frame, text="📂", font=FONTS.get_font("small"),
            command=self._load_exclude_file, width=3,
        )
        self.exclude_file_btn.pack(side=tk.RIGHT)
        
        self.exclude_regex_var = tk.BooleanVar(value=False)
        exclude_regex_cb = ttk.Checkbutton(
            filter_card.content,
            text="Use regex for exclude",
            variable=self.exclude_regex_var,
        )
        exclude_regex_cb.pack(anchor=tk.W, pady=2)
        
        # Filter info
        self.filter_info_label = tk.Label(
            filter_card.content,
            text="Separate items with semicolons (;)",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        )
        self.filter_info_label.pack(anchor=tk.W, pady=2)
        
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
        
        self.curved_edges_var = tk.BooleanVar(value=False)
        curved_edges_cb = tk.Checkbutton(
            edge_card.content,
            text="Curved edges",
            variable=self.curved_edges_var,
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            activebackground=self.theme["bg_card"],
            selectcolor=self.theme["bg_card"],
            font=FONTS.get_font("body"),
        )
        curved_edges_cb.pack(anchor=tk.W, pady=2)
        
        # Layout & Appearance Card
        layout_card = Card(self.options_content, title="🎨 Layout & Appearance", theme=self.theme_name)
        layout_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.layout_combo = LabeledCombobox(
            layout_card.content, label="Layout:",
            values=NETWORK_LAYOUTS,
            default="spring",
            theme=self.theme_name, label_width=15,
        )
        self.layout_combo.pack(fill=tk.X, pady=4)
        
        # Colormap selector - uses COLOR_PALETTES from config
        self.colormap_combo = LabeledCombobox(
            layout_card.content, label="Colormap:",
            values=COLOR_PALETTES,
            default="viridis",
            theme=self.theme_name, label_width=15,
        )
        self.colormap_combo.pack(fill=tk.X, pady=4)
        
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
            mainpath_card.content, label="Compute main path",
            default=True, theme=self.theme_name,
        )
        self.show_main_path.pack(fill=tk.X, pady=2)
        
        # Run Buttons
        # LLM Configuration (v2.11)
        self._create_llm_config_card()
        
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
    
    def _load_include_file(self):
        """Load include entities from file."""
        import re
        filepath = filedialog.askopenfilename(
            title="Load Include List",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    items = [line.strip() for line in f if line.strip()]
                self._include_items = set(items)
                self.include_entry.delete(0, tk.END)
                self.include_entry.insert(0, f"[{len(items)} items from file]")
                self.filter_info_label.config(text=f"Include: {len(items)} items loaded")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
    
    def _load_exclude_file(self):
        """Load exclude entities from file."""
        import re
        filepath = filedialog.askopenfilename(
            title="Load Exclude List",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    items = [line.strip() for line in f if line.strip()]
                self._exclude_items = set(items)
                self.exclude_entry.delete(0, tk.END)
                self.exclude_entry.insert(0, f"[{len(items)} items from file]")
                self.filter_info_label.config(text=f"Exclude: {len(items)} items loaded")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
    
    def _parse_filter_items(self, entry_text: str, is_regex: bool) -> set:
        """Parse filter items from entry text."""
        if not entry_text or entry_text.startswith("["):
            return set()
        
        if is_regex:
            return {entry_text.strip()}
        else:
            items = [item.strip() for item in entry_text.split(";") if item.strip()]
            return set(items)
    
    def _filter_nodes(self, all_nodes: list, include_set: set, exclude_set: set,
                      include_regex: bool, exclude_regex: bool) -> list:
        """Filter nodes based on include/exclude sets and regex patterns."""
        import re
        result = []
        
        for node in all_nodes:
            node_str = str(node)
            
            # Check include
            if include_set:
                if include_regex:
                    pattern = list(include_set)[0] if include_set else ""
                    try:
                        if not re.search(pattern, node_str, re.IGNORECASE):
                            continue
                    except re.error:
                        pass
                else:
                    if node_str not in include_set and node not in include_set:
                        continue
            
            # Check exclude
            if exclude_set:
                if exclude_regex:
                    pattern = list(exclude_set)[0] if exclude_set else ""
                    try:
                        if re.search(pattern, node_str, re.IGNORECASE):
                            continue
                    except re.error:
                        pass
                else:
                    if node_str in exclude_set or node in exclude_set:
                        continue
            
            result.append(node)
        
        return result
    
    def _get_network_type_key(self) -> str:
        """Extract network type key from combo selection."""
        selected = self.network_combo.get().lower()
        
        for key, config in NETWORK_TYPES.items():
            if config["label"].lower() in selected:
                return key
        
        return "coauthorship"
    
    def _on_network_type_changed(self, event=None):
        """Handle network type change - update panel title."""
        network_type = self._get_network_type_key()
        if network_type in NETWORK_TYPES:
            config = NETWORK_TYPES[network_type]
            self.title = config["label"]
            self.icon = config["icon"]
            self.description = config["description"]
            
            # Update the title label if it exists
            if hasattr(self, 'title_label'):
                self.title_label.config(text=f"{self.icon} {self.title}")
            if hasattr(self, 'desc_label'):
                self.desc_label.config(text=self.description)
    
    def _on_dataset_loaded(self, data):
        """Handle dataset loaded event."""
        super()._on_dataset_loaded(data)
    
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
            header, text="Network Visualization",
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            padx=12, pady=8,
        ).pack(side=tk.LEFT)
        
        ttk.Separator(self.results_card, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12)
        
        self.results_content = tk.Frame(self.results_card, bg=self.theme["bg_card"])
        self.results_content.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        self.notebook = ttk.Notebook(self.results_content)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.plot_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.plot_frame, text="  🔗 Network  ")
        
        self.nodes_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.nodes_frame, text="  📍 Nodes  ")
        
        self.edges_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.edges_frame, text="  🔗 Edges  ")
        
        self.stats_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.stats_frame, text="  📊 Statistics  ")

        

        # Info tab

        info_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])

        self.notebook.add(info_frame, text="ℹ️ Info")

        self._create_info_content(info_frame)
        
        tk.Label(
            self.plot_frame,
            text="Configure options and click 'Build Network'",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_muted"],
        ).pack(expand=True)
    
    def _show_loading(self, message: str = "Processing..."):
        """Show loading state."""
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
        """Build the network using Biblium's methods."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        if not HAS_NETWORKX:
            messagebox.showerror("Missing Package", "NetworkX is required for network analysis.")
            return
        
        self._show_loading("Building network...")
        
        network_type = self._get_network_type_key()
        
        # Capture filter settings before thread
        include_text = self.include_entry.get() if hasattr(self, 'include_entry') else ""
        exclude_text = self.exclude_entry.get() if hasattr(self, 'exclude_entry') else ""
        include_regex = self.include_regex_var.get() if hasattr(self, 'include_regex_var') else False
        exclude_regex = self.exclude_regex_var.get() if hasattr(self, 'exclude_regex_var') else False
        
        # Parse filter items
        include_items = self._include_items.copy() if include_text.startswith("[") else self._parse_filter_items(include_text, include_regex)
        exclude_items = self._exclude_items.copy() if exclude_text.startswith("[") else self._parse_filter_items(exclude_text, exclude_regex)
        
        def do_build():
            try:
                G = None
                top_n = self.top_n_nodes.get()
                min_weight = self.min_edge_weight.get()
                
                # Use Biblium's built-in methods
                if network_type == "coauthorship":
                    G = self.bib.build_coauthorship_network(
                        top_n=top_n,
                        min_collabs=min_weight,
                    )
                    node_count_attr = "documents"
                    
                elif network_type in ["keyword_author", "keyword_index", "keyword_all"]:
                    # Use biblium's compute_cooccurrence_matrix for keywords
                    if network_type == "keyword_author":
                        kind = "author"
                    elif network_type == "keyword_index":
                        kind = "index"
                    else:
                        kind = "all"
                    G = self._build_keyword_network(top_n, min_weight, kind)
                    node_count_attr = "frequency"
                    
                elif network_type == "co_citation":
                    try:
                        G = self.bib.build_cocitation_network(
                            top_n=top_n,
                            min_cocitations=min_weight,
                        )
                    except:
                        G = self._build_reference_cocitation_network(top_n, min_weight)
                    node_count_attr = "citations"
                
                elif network_type == "source_cocitation":
                    G = self._build_source_cocitation_network(top_n, min_weight)
                    node_count_attr = "citations"
                    
                elif network_type == "country_collaboration":
                    G = self._build_country_network(top_n, min_weight)
                    node_count_attr = "documents"
                
                elif network_type == "institution_collaboration":
                    G = self._build_institution_network(top_n, min_weight)
                    node_count_attr = "documents"
                
                elif network_type == "title_ngrams":
                    G = self._build_ngrams_network(top_n, min_weight, source="title")
                    node_count_attr = "frequency"
                
                elif network_type == "abstract_ngrams":
                    G = self._build_ngrams_network(top_n, min_weight, source="abstract")
                    node_count_attr = "frequency"
                    
                else:
                    # Try to build generic cooccurrence network
                    G = self._build_generic_cooccurrence_network(network_type, top_n, min_weight)
                    node_count_attr = "frequency"
                
                if G is None or len(G.nodes()) == 0:
                    raise ValueError("Network has no nodes")
                
                # Apply entity filtering
                if include_items or exclude_items:
                    nodes_to_keep = self._filter_nodes(
                        list(G.nodes()), 
                        include_items, 
                        exclude_items,
                        include_regex, 
                        exclude_regex
                    )
                    nodes_to_remove = [n for n in G.nodes() if n not in nodes_to_keep]
                    G.remove_nodes_from(nodes_to_remove)
                    
                    # Remove isolated nodes after filtering
                    isolated = list(nx.isolates(G))
                    G.remove_nodes_from(isolated)
                
                if len(G.nodes()) < 2:
                    raise ValueError("Network has too few nodes after filtering. Try adjusting filters.")
                
                self._add_centrality_metrics(G)
                
                communities = {}
                if self.detect_communities.get():
                    communities = self._detect_communities(G)
                
                stats = self._compute_network_stats(G, communities)
                node_df = self._create_node_dataframe(G, node_count_attr)
                edge_df = self._create_edge_dataframe(G)
                
                result = {
                    "graph": G,
                    "node_df": node_df,
                    "edge_df": edge_df,
                    "stats": stats,
                    "communities": communities,
                    "node_count_attr": node_count_attr,
                }
                
                self.after(0, lambda r=result: self._on_build_success(r))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: self._on_build_error(msg))
        
        threading.Thread(target=do_build, daemon=True).start()
    
    def _build_country_network(self, top_n: int, min_collabs: int) -> nx.Graph:
        """Build country collaboration network."""
        from collections import Counter, defaultdict
        
        country_col = None
        for col_name in ["Country", "Countries", "Affiliations Country"]:
            if col_name in self.bib.df.columns:
                country_col = col_name
                break
        
        if not country_col:
            raise ValueError("No country column found. Run preprocessing first.")
        
        sep = self.bib.default_separator
        
        country_docs = Counter()
        country_collabs = defaultdict(int)
        
        for idx, row in self.bib.df.iterrows():
            if pd.isna(row[country_col]):
                continue
            
            countries = list(set([c.strip() for c in str(row[country_col]).split(sep) if c.strip()]))
            
            for country in countries:
                country_docs[country] += 1
            
            for i, c1 in enumerate(countries):
                for c2 in countries[i+1:]:
                    if c1 != c2:
                        pair = tuple(sorted([c1, c2]))
                        country_collabs[pair] += 1
        
        top_countries = set(c for c, _ in country_docs.most_common(top_n))
        
        G = nx.Graph()
        
        for country in top_countries:
            G.add_node(country, documents=country_docs[country], label=country)
        
        for (c1, c2), count in country_collabs.items():
            if c1 in top_countries and c2 in top_countries and count >= min_collabs:
                G.add_edge(c1, c2, weight=count)
        
        return G
    
    def _build_coupling_network(self, top_n: int, min_coupling: int) -> nx.Graph:
        """Build bibliographic coupling network."""
        from collections import Counter, defaultdict
        
        ref_col = None
        for col_name in ["References", "Cited References", "references"]:
            if col_name in self.bib.df.columns:
                ref_col = col_name
                break
        
        if not ref_col:
            raise ValueError("No reference column found")
        
        title_col = self.bib.mapping.get("Title", "Title")
        cite_col = self.bib.mapping.get("Cited by", "Cited by")
        sep = self.bib.default_separator
        
        doc_refs = {}
        doc_citations = {}
        
        for idx, row in self.bib.df.iterrows():
            title = str(row.get(title_col, f"Doc_{idx}"))[:50]
            if pd.isna(row[ref_col]):
                continue
            
            refs = set(r.strip().lower() for r in str(row[ref_col]).split(sep) if r.strip())
            if refs:
                doc_refs[title] = refs
                doc_citations[title] = row.get(cite_col, 0) if cite_col in self.bib.df.columns else 0
        
        sorted_docs = sorted(doc_citations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_docs = set(d[0] for d in sorted_docs)
        
        G = nx.Graph()
        
        for doc in top_docs:
            G.add_node(doc, citations=doc_citations.get(doc, 0), label=doc)
        
        docs_list = list(top_docs)
        for i, d1 in enumerate(docs_list):
            for d2 in docs_list[i+1:]:
                refs1 = doc_refs.get(d1, set())
                refs2 = doc_refs.get(d2, set())
                shared = len(refs1 & refs2)
                if shared >= min_coupling:
                    G.add_edge(d1, d2, weight=shared)
        
        return G
    
    def _build_keyword_network(self, top_n: int, min_cooccur: int, kind: str = "author") -> nx.Graph:
        """Build keyword co-occurrence network using biblium's compute_cooccurrence_matrix."""
        from biblium import utilsbib
        from collections import Counter, defaultdict
        
        # Find keyword column based on kind
        if kind == "author":
            col_options = ["Author Keywords", "DE", "Keywords"]
        elif kind == "index":
            col_options = ["Index Keywords", "ID", "Keywords Plus"]
        else:  # "all"
            col_options = ["All Keywords", "Keywords", "Author Keywords"]
        
        keyword_col = None
        for col in col_options:
            if col in self.bib.df.columns:
                keyword_col = col
                break
        
        if not keyword_col:
            # For "all", try to combine author and index keywords
            if kind == "all":
                keyword_col = self._create_combined_keywords_column()
            else:
                raise ValueError(f"No {kind} keyword column found")
        
        sep = self.bib.default_separator
        
        # Find year column
        year_col = None
        for col in ["Year", "Publication Year", "PY", "year", "publication_year"]:
            if col in self.bib.df.columns:
                year_col = col
                break
        
        # Calculate average year for each keyword
        keyword_years = defaultdict(list)
        if year_col:
            for idx, row in self.bib.df.iterrows():
                if pd.isna(row[keyword_col]):
                    continue
                year = row.get(year_col)
                if pd.notna(year):
                    try:
                        year = int(year)
                        keywords = [k.strip() for k in str(row[keyword_col]).split(sep) if k.strip()]
                        for kw in keywords:
                            keyword_years[kw].append(year)
                    except (ValueError, TypeError):
                        pass
        
        # Compute co-occurrence matrix
        cooc_matrix = utilsbib.compute_cooccurrence_matrix(
            self.bib.df,
            column=keyword_col,
            top_n=top_n,
            separator=sep,
        )
        
        if cooc_matrix.empty:
            raise ValueError("No keyword co-occurrence data found")
        
        # Build graph
        G = nx.Graph()
        
        for item in cooc_matrix.index:
            freq = cooc_matrix.loc[item, item]
            # Calculate average year for this keyword
            avg_year = 0
            if item in keyword_years and keyword_years[item]:
                avg_year = sum(keyword_years[item]) / len(keyword_years[item])
            G.add_node(item, frequency=freq, label=item, avg_year=avg_year)
        
        items = list(cooc_matrix.index)
        for i, item1 in enumerate(items):
            for j, item2 in enumerate(items):
                if i < j:
                    weight = cooc_matrix.loc[item1, item2]
                    if weight >= min_cooccur:
                        G.add_edge(item1, item2, weight=weight)
        
        return G
    
    def _build_source_cocitation_network(self, top_n: int, min_cocitations: int) -> nx.Graph:
        """Build source (journal) co-citation network."""
        from collections import Counter, defaultdict
        
        ref_col = None
        for col_name in ["Cited Sources", "Cited Journals", "References"]:
            if col_name in self.bib.df.columns:
                ref_col = col_name
                break
        
        if not ref_col:
            raise ValueError("No cited sources column found")
        
        sep = self.bib.default_separator
        
        source_count = Counter()
        cocitations = defaultdict(int)
        
        for idx, row in self.bib.df.iterrows():
            if pd.isna(row[ref_col]):
                continue
            
            sources = list(set([s.strip() for s in str(row[ref_col]).split(sep) if s.strip()]))
            
            for source in sources:
                source_count[source] += 1
            
            for i, s1 in enumerate(sources):
                for s2 in sources[i+1:]:
                    if s1 != s2:
                        pair = tuple(sorted([s1, s2]))
                        cocitations[pair] += 1
        
        top_sources = set(s for s, _ in source_count.most_common(top_n))
        
        G = nx.Graph()
        
        for source in top_sources:
            G.add_node(source, citations=source_count[source], label=source)
        
        for (s1, s2), count in cocitations.items():
            if s1 in top_sources and s2 in top_sources and count >= min_cocitations:
                G.add_edge(s1, s2, weight=count)
        
        return G
    
    def _build_institution_network(self, top_n: int, min_collabs: int) -> nx.Graph:
        """Build institution collaboration network."""
        from collections import Counter, defaultdict
        
        aff_col = None
        for col_name in ["Affiliations", "C1", "Addresses", "Author Affiliations"]:
            if col_name in self.bib.df.columns:
                aff_col = col_name
                break
        
        if not aff_col:
            raise ValueError("No affiliations column found")
        
        sep = self.bib.default_separator
        
        inst_docs = Counter()
        inst_collabs = defaultdict(int)
        
        for idx, row in self.bib.df.iterrows():
            if pd.isna(row[aff_col]):
                continue
            
            institutions = list(set([i.strip() for i in str(row[aff_col]).split(sep) if i.strip()]))
            
            for inst in institutions:
                inst_docs[inst] += 1
            
            for i, i1 in enumerate(institutions):
                for i2 in institutions[i+1:]:
                    if i1 != i2:
                        pair = tuple(sorted([i1, i2]))
                        inst_collabs[pair] += 1
        
        top_insts = set(i for i, _ in inst_docs.most_common(top_n))
        
        G = nx.Graph()
        
        for inst in top_insts:
            G.add_node(inst, documents=inst_docs[inst], label=inst)
        
        for (i1, i2), count in inst_collabs.items():
            if i1 in top_insts and i2 in top_insts and count >= min_collabs:
                G.add_edge(i1, i2, weight=count)
        
        return G
    
    def _build_generic_cooccurrence_network(self, network_type: str, top_n: int, min_cooccur: int) -> nx.Graph:
        """Build generic co-occurrence network for any column type."""
        from biblium import utilsbib
        
        # Try to find a matching column
        col_mapping = {
            "authors": ["Authors", "AU", "Author full names"],
            "sources": ["Source title", "SO", "Journal"],
            "references": ["References", "Cited References", "CR"],
        }
        
        column = None
        for key, options in col_mapping.items():
            if key in network_type.lower():
                for opt in options:
                    if opt in self.bib.df.columns:
                        column = opt
                        break
                if column:
                    break
        
        if not column:
            raise ValueError(f"Could not find column for network type: {network_type}")
        
        sep = self.bib.default_separator
        
        cooc_matrix = utilsbib.compute_cooccurrence_matrix(
            self.bib.df,
            column=column,
            top_n=top_n,
            separator=sep,
        )
        
        if cooc_matrix.empty:
            raise ValueError(f"No co-occurrence data found for {column}")
        
        G = nx.Graph()
        
        for item in cooc_matrix.index:
            freq = cooc_matrix.loc[item, item]
            G.add_node(item, frequency=freq, label=item)
        
        items = list(cooc_matrix.index)
        for i, item1 in enumerate(items):
            for j, item2 in enumerate(items):
                if i < j:
                    weight = cooc_matrix.loc[item1, item2]
                    if weight >= min_cooccur:
                        G.add_edge(item1, item2, weight=weight)
        
        return G
    
    def _create_combined_keywords_column(self) -> str:
        """Create a combined keywords column if it doesn't exist."""
        sep = self.bib.default_separator
        
        # Find available keyword columns
        author_kw_col = None
        index_kw_col = None
        
        for col in ["Author Keywords", "DE"]:
            if col in self.bib.df.columns:
                author_kw_col = col
                break
        
        for col in ["Index Keywords", "ID", "Keywords Plus"]:
            if col in self.bib.df.columns:
                index_kw_col = col
                break
        
        if author_kw_col is None and index_kw_col is None:
            raise ValueError("No keyword columns found")
        
        # Create combined column
        def combine_keywords(row):
            kw = []
            if author_kw_col and pd.notna(row.get(author_kw_col, None)):
                kw.extend([k.strip() for k in str(row[author_kw_col]).split(sep)])
            if index_kw_col and pd.notna(row.get(index_kw_col, None)):
                kw.extend([k.strip() for k in str(row[index_kw_col]).split(sep)])
            return sep.join(list(set(kw)))
        
        self.bib.df["_combined_keywords"] = self.bib.df.apply(combine_keywords, axis=1)
        return "_combined_keywords"
    
    def _build_ngrams_network(self, top_n: int, min_cooccur: int, source: str = "title") -> nx.Graph:
        """Build n-grams co-occurrence network from titles or abstracts."""
        from biblium import utilsbib
        from collections import Counter, defaultdict
        
        # Find source column
        if source == "title":
            col_options = ["Title", "TI", "title"]
        else:
            col_options = ["Abstract", "AB", "abstract"]
        
        text_col = None
        for col in col_options:
            if col in self.bib.df.columns:
                text_col = col
                break
        
        if not text_col:
            raise ValueError(f"No {source} column found")
        
        # Extract n-grams (simple approach: bigrams and trigrams)
        import re
        from collections import Counter
        
        ngram_docs = []
        ngram_count = Counter()
        
        for idx, row in self.bib.df.iterrows():
            if pd.isna(row[text_col]):
                ngram_docs.append([])
                continue
            
            text = str(row[text_col]).lower()
            # Simple tokenization
            words = re.findall(r'\b[a-z]{3,}\b', text)
            
            # Generate bigrams
            ngrams = []
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                ngrams.append(bigram)
                ngram_count[bigram] += 1
            
            ngram_docs.append(ngrams)
        
        # Get top n-grams
        top_ngrams = [ng for ng, _ in ngram_count.most_common(top_n)]
        top_set = set(top_ngrams)
        
        # Build co-occurrence
        cooc = defaultdict(int)
        for doc_ngrams in ngram_docs:
            doc_top = [ng for ng in doc_ngrams if ng in top_set]
            unique_ngrams = list(set(doc_top))
            for i, ng1 in enumerate(unique_ngrams):
                for ng2 in unique_ngrams[i+1:]:
                    pair = tuple(sorted([ng1, ng2]))
                    cooc[pair] += 1
        
        # Build graph
        G = nx.Graph()
        
        for ngram in top_ngrams:
            G.add_node(ngram, frequency=ngram_count[ngram], label=ngram)
        
        for (ng1, ng2), weight in cooc.items():
            if weight >= min_cooccur:
                G.add_edge(ng1, ng2, weight=weight)
        
        return G
    
    def _build_reference_cocitation_network(self, top_n: int, min_cocitations: int) -> nx.Graph:
        """Build reference co-citation network."""
        from collections import Counter, defaultdict
        
        ref_col = None
        for col_name in ["References", "Cited References", "CR", "references"]:
            if col_name in self.bib.df.columns:
                ref_col = col_name
                break
        
        if not ref_col:
            raise ValueError("No references column found")
        
        sep = self.bib.default_separator
        
        ref_count = Counter()
        cocitations = defaultdict(int)
        
        for idx, row in self.bib.df.iterrows():
            if pd.isna(row[ref_col]):
                continue
            
            refs = list(set([r.strip()[:80] for r in str(row[ref_col]).split(sep) if r.strip()]))
            
            for ref in refs:
                ref_count[ref] += 1
            
            for i, r1 in enumerate(refs):
                for r2 in refs[i+1:]:
                    if r1 != r2:
                        pair = tuple(sorted([r1, r2]))
                        cocitations[pair] += 1
        
        top_refs = set(r for r, _ in ref_count.most_common(top_n))
        
        G = nx.Graph()
        
        for ref in top_refs:
            G.add_node(ref, citations=ref_count[ref], label=ref[:40])
        
        for (r1, r2), count in cocitations.items():
            if r1 in top_refs and r2 in top_refs and count >= min_cocitations:
                G.add_edge(r1, r2, weight=count)
        
        return G
    
    def _add_centrality_metrics(self, G: nx.Graph):
        """Add centrality metrics to graph nodes."""
        if len(G.nodes()) == 0:
            return
        
        for node in G.nodes():
            G.nodes[node]["degree"] = G.degree(node)
        
        if len(G.edges()) > 0:
            try:
                degree_cent = nx.degree_centrality(G)
                for node, val in degree_cent.items():
                    G.nodes[node]["degree_centrality"] = val
            except:
                pass
            
            try:
                betweenness = nx.betweenness_centrality(G)
                for node, val in betweenness.items():
                    G.nodes[node]["betweenness"] = val
            except:
                pass
            
            try:
                closeness = nx.closeness_centrality(G)
                for node, val in closeness.items():
                    G.nodes[node]["closeness"] = val
            except:
                pass
    
    def _detect_communities(self, G: nx.Graph) -> Dict:
        """Detect communities in the graph."""
        if len(G.nodes()) == 0 or len(G.edges()) == 0:
            return {node: 0 for node in G.nodes()}
        
        communities = {}
        algo = self.community_algo.get()
        
        try:
            if algo == "louvain":
                try:
                    from community import community_louvain
                    partition = community_louvain.best_partition(G)
                    communities = partition
                except ImportError:
                    comms = nx.algorithms.community.greedy_modularity_communities(G)
                    for i, comm in enumerate(comms):
                        for node in comm:
                            communities[node] = i
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
        except Exception as e:
            print(f"Community detection failed: {e}")
            communities = {node: 0 for node in G.nodes()}
        
        for node in G.nodes():
            G.nodes[node]["community"] = communities.get(node, 0)
        
        return communities
    
    def _compute_network_stats(self, G: nx.Graph, communities: Dict) -> Dict:
        """Compute network statistics."""
        stats = {"Nodes": len(G.nodes()), "Edges": len(G.edges())}
        
        if len(G.nodes()) > 0:
            stats["Density"] = round(nx.density(G), 4)
            stats["Avg Degree"] = round(sum(dict(G.degree()).values()) / len(G.nodes()), 2)
            
            try:
                stats["Avg Clustering"] = round(nx.average_clustering(G), 4)
            except:
                pass
            
            if nx.is_connected(G):
                try:
                    stats["Diameter"] = nx.diameter(G)
                    stats["Avg Path Length"] = round(nx.average_shortest_path_length(G), 2)
                except:
                    pass
            else:
                stats["Components"] = nx.number_connected_components(G)
            
            if communities:
                stats["Communities"] = len(set(communities.values()))
        
        return stats
    
    def _create_node_dataframe(self, G: nx.Graph, count_attr: str) -> pd.DataFrame:
        """Create node DataFrame for display."""
        node_data = []
        
        for node in G.nodes():
            data = {
                "Node": str(node)[:50],
                "Occurrences": G.nodes[node].get(count_attr, G.nodes[node].get("documents", G.nodes[node].get("frequency", G.nodes[node].get("citations", 0)))),
                "Degree": G.nodes[node].get("degree", G.degree(node)),
            }
            
            if "avg_year" in G.nodes[node]:
                data["Avg Year"] = G.nodes[node].get("avg_year", 0)
            
            if "betweenness" in G.nodes[node]:
                data["Betweenness"] = round(G.nodes[node].get("betweenness", 0), 4)
            
            if "closeness" in G.nodes[node]:
                data["Closeness"] = round(G.nodes[node].get("closeness", 0), 4)
            
            if "community" in G.nodes[node]:
                data["Community"] = G.nodes[node].get("community", 0)
            
            node_data.append(data)
        
        df = pd.DataFrame(node_data)
        if "Degree" in df.columns:
            df = df.sort_values("Degree", ascending=False)
        
        return df
    
    def _create_edge_dataframe(self, G: nx.Graph) -> pd.DataFrame:
        """Create edge DataFrame for display."""
        edge_data = []
        
        for u, v, d in G.edges(data=True):
            edge_data.append({
                "Source": str(u)[:40],
                "Target": str(v)[:40],
                "Weight": d.get("weight", 1),
            })
        
        df = pd.DataFrame(edge_data)
        if len(df) > 0 and "Weight" in df.columns:
            df = df.sort_values("Weight", ascending=False)
        
        return df
    
    def _on_build_success(self, result: Dict):
        """Display network results."""
        self._graph = result["graph"]
        self._node_df = result["node_df"]
        self._edge_df = result["edge_df"]
        
        self._show_network_plot(result)
        self._show_nodes_table(result["node_df"])
        self._show_edges_table(result["edge_df"])
        self._show_network_stats(result["stats"])
        
        event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": "Co-occurrence Networks"})
    
    def _show_network_plot(self, result: Dict):
        """Display network visualization using static image rendering."""
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
        node_count_attr = result.get("node_count_attr", "documents")
        
        # Use non-interactive backend for thread safety
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import io
        from PIL import Image, ImageTk
        
        # Create figure with non-interactive backend
        fig = Figure(figsize=(12, 10), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Store figure reference for export
        self._current_fig = fig
        
        # Get selected colormap - this is now consistent across all color modes
        selected_cmap_name = self.colormap_combo.get()
        
        # Layout
        layout_name = self.layout_combo.get()
        pos = self._compute_layout(G, layout_name)
        
        # Node sizes
        sizes = self._compute_node_sizes(G, node_count_attr)
        
        # Node colors - using selected colormap consistently
        colors, color_vmin, color_vmax = self._compute_node_colors(G, communities, selected_cmap_name)
        
        # Edge widths
        edge_weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [0.5 + 2.5 * w / max_weight for w in edge_weights]
        
        # Check for directed graph and curved edges option
        is_directed = G.is_directed() if hasattr(G, 'is_directed') else False
        use_curved = self.curved_edges_var.get() if hasattr(self, 'curved_edges_var') else False
        
        if is_directed:
            if use_curved:
                nx.draw_networkx_edges(
                    G, pos, ax=ax, 
                    alpha=0.3, width=edge_widths, edge_color='gray',
                    arrows=True, arrowsize=10, arrowstyle='->',
                    connectionstyle="arc3,rad=0.1",
                )
            else:
                nx.draw_networkx_edges(
                    G, pos, ax=ax, 
                    alpha=0.3, width=edge_widths, edge_color='gray',
                    arrows=True, arrowsize=10, arrowstyle='->',
                )
        else:
            if use_curved:
                # connectionstyle requires arrows=True to use FancyArrowPatch
                # Use arrowstyle='-' for undirected (no arrowhead)
                nx.draw_networkx_edges(
                    G, pos, ax=ax, 
                    alpha=0.3, width=edge_widths, edge_color='gray',
                    arrows=True, arrowstyle='-',
                    connectionstyle="arc3,rad=0.1",
                )
            else:
                nx.draw_networkx_edges(
                    G, pos, ax=ax, 
                    alpha=0.3, width=edge_widths, edge_color='gray',
                )
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors, alpha=0.8)
        
        # Labels for top nodes (use configurable number)
        num_labels_to_show = self.num_labels.get() if hasattr(self, 'num_labels') else 20
        top_n_labels = min(num_labels_to_show, len(G.nodes()))
        
        if top_n_labels > 0:
            if self._node_df is not None and "Node" in self._node_df.columns:
                top_nodes = self._node_df.head(top_n_labels)["Node"].tolist()
                labels = {n: str(n)[:25] for n in top_nodes if n in G.nodes()}
            else:
                node_degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:top_n_labels]
                labels = {n: str(n)[:25] for n, _ in node_degrees}
            
            nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)
        
        ax.set_title(f"Network: {len(G.nodes())} nodes, {len(G.edges())} edges")
        ax.axis('off')
        
        # Add colorbar for continuous coloring with proper value range
        color_by = self.node_color_by.get()
        if color_by != "Community":
            self._add_colorbar(fig, ax, color_by, selected_cmap_name, color_vmin, color_vmax)
        
        fig.tight_layout()
        
        # Render to static image
        try:
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
            self._add_save_menu(scaled_frame)
            
        except Exception as e:
            tk.Label(
                self.plot_frame,
                text=f"Error rendering plot: {e}",
                font=FONTS.get_font("body"),
                bg=self.theme["bg_card"],
                fg=self.theme["danger"],
            ).pack(expand=True)
    
    def _add_save_menu(self, widget):
        """Add right-click context menu for saving."""
        def show_menu(event):
            menu = tk.Menu(widget, tearoff=0)
            menu.add_command(label="📄 Add to Report", command=self._add_plot_to_report)
            menu.add_separator()
            menu.add_command(label="💾 Save as PNG...", command=lambda: self._save_plot_as("png"))
            menu.add_command(label="💾 Save as PDF...", command=lambda: self._save_plot_as("pdf"))
            menu.add_command(label="💾 Save as SVG...", command=lambda: self._save_plot_as("svg"))
            menu.tk_popup(event.x_root, event.y_root)
        
        widget.bind("<Button-3>", show_menu)
    
    def _add_plot_to_report(self):
        """Add current plot to report queue."""
        if not hasattr(self, '_current_fig') or self._current_fig is None:
            messagebox.showinfo("No Plot", "No plot to add to report.")
            return
        
        try:
            from biblium.gui.core.state import report_queue
            
            # Get title from figure or panel
            plot_title = "Network Plot"
            if self._current_fig.axes:
                plot_title = self._current_fig.axes[0].get_title() or "Network Plot"
            
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
    
    def _save_plot_as(self, fmt):
        """Save current figure in specified format."""
        if not hasattr(self, '_current_fig') or self._current_fig is None:
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=f".{fmt}",
            filetypes=[(f"{fmt.upper()} files", f"*.{fmt}")],
            title=f"Save as {fmt.upper()}",
        )
        if filepath:
            try:
                self._current_fig.savefig(filepath, dpi=300, bbox_inches='tight',
                                          facecolor='white', edgecolor='none')
                messagebox.showinfo("Saved", f"Plot saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")
    
    def _compute_layout(self, G: nx.Graph, layout_name: str) -> Dict:
        """Compute network layout."""
        if layout_name == "spring":
            return nx.spring_layout(G, k=2, iterations=50)
        elif layout_name == "kamada_kawai":
            try:
                return nx.kamada_kawai_layout(G)
            except:
                return nx.spring_layout(G)
        elif layout_name == "circular":
            return nx.circular_layout(G)
        elif layout_name == "shell":
            return nx.shell_layout(G)
        elif layout_name == "spectral":
            try:
                return nx.spectral_layout(G)
            except:
                return nx.spring_layout(G)
        else:
            return nx.random_layout(G)
    
    def _compute_node_sizes(self, G: nx.Graph, default_attr: str) -> list:
        """Compute node sizes based on selected attribute."""
        size_by = self.node_size_by.get().lower()
        
        if size_by == "occurrences":
            sizes = []
            for n in G.nodes():
                val = G.nodes[n].get(default_attr, 
                      G.nodes[n].get("documents",
                      G.nodes[n].get("frequency",
                      G.nodes[n].get("citations",
                      G.nodes[n].get("occurrences", 1)))))
                sizes.append(val)
        elif size_by == "degree":
            sizes = [G.nodes[n].get("degree", G.degree(n)) for n in G.nodes()]
        elif size_by == "betweenness":
            sizes = [G.nodes[n].get("betweenness", 0.01) * 1000 for n in G.nodes()]
        elif size_by == "closeness":
            sizes = [G.nodes[n].get("closeness", 0.1) * 100 for n in G.nodes()]
        else:
            sizes = [1 for _ in G.nodes()]
        
        if sizes:
            max_size = max(sizes) if max(sizes) > 0 else 1
            min_size = min(sizes)
            range_size = max_size - min_size if max_size > min_size else 1
            sizes = [100 + 400 * (s - min_size) / range_size for s in sizes]
        
        return sizes
    
    def _compute_node_colors(self, G: nx.Graph, communities: Dict, cmap_name: str) -> tuple:
        """Compute node colors based on selected attribute and colormap.
        
        Returns:
            tuple: (colors_list, vmin, vmax) where vmin/vmax are the actual value range
        """
        color_by = self.node_color_by.get()
        vmin, vmax = None, None
        
        # Get the selected colormap
        try:
            cmap = plt.cm.get_cmap(cmap_name)
        except:
            cmap = plt.cm.get_cmap("viridis")
        
        if color_by == "Community" and communities:
            n_communities = len(set(communities.values()))
            
            # For communities, always use a qualitative/categorical colormap
            # Check if selected cmap is qualitative (has .colors attribute with discrete colors)
            qualitative_cmaps = ['tab10', 'tab20', 'Set1', 'Set2', 'Set3', 'Pastel1', 'Pastel2', 
                                 'Paired', 'Accent', 'Dark2']
            
            if cmap_name in qualitative_cmaps:
                # Use the selected qualitative colormap
                cat_cmap = plt.cm.get_cmap(cmap_name)
                n_colors = len(cat_cmap.colors) if hasattr(cat_cmap, 'colors') else 10
                colors = [cat_cmap(communities.get(n, 0) % n_colors) for n in G.nodes()]
            else:
                # For continuous colormaps (Blues, viridis, etc.), sample discrete colors
                # This creates distinct colors even for continuous colormaps
                if n_communities <= 10:
                    cat_cmap = plt.cm.get_cmap("tab10")
                else:
                    cat_cmap = plt.cm.get_cmap("tab20")
                n_colors = 10 if n_communities <= 10 else 20
                colors = [cat_cmap(communities.get(n, 0) % n_colors) for n in G.nodes()]
            
        elif color_by == "Average Year":
            avg_years = [G.nodes[n].get("avg_year", 0) for n in G.nodes()]
            if max(avg_years) > min(avg_years):
                vmin, vmax = min(avg_years), max(avg_years)
                norm_years = [(y - vmin) / (vmax - vmin) for y in avg_years]
                colors = [cmap(y) for y in norm_years]
            else:
                colors = [cmap(0.5)] * len(G.nodes())
                
        elif color_by == "Occurrences":
            occurrences = []
            for n in G.nodes():
                val = G.nodes[n].get("documents",
                      G.nodes[n].get("frequency",
                      G.nodes[n].get("citations",
                      G.nodes[n].get("occurrences", 1))))
                occurrences.append(val)
            
            if max(occurrences) > min(occurrences):
                vmin, vmax = min(occurrences), max(occurrences)
                norm_occ = [(o - vmin) / (vmax - vmin) for o in occurrences]
                colors = [cmap(o) for o in norm_occ]
            else:
                colors = [cmap(0.5)] * len(G.nodes())
                
        elif color_by == "Degree":
            degrees = [G.nodes[n].get("degree", G.degree(n)) for n in G.nodes()]
            if max(degrees) > min(degrees):
                vmin, vmax = min(degrees), max(degrees)
                norm_deg = [(d - vmin) / (vmax - vmin) for d in degrees]
                colors = [cmap(0.2 + 0.8 * d) for d in norm_deg]
            else:
                colors = [cmap(0.5)] * len(G.nodes())
                
        elif color_by == "Betweenness":
            betweenness = [G.nodes[n].get("betweenness", 0) for n in G.nodes()]
            if max(betweenness) > min(betweenness):
                vmin, vmax = min(betweenness), max(betweenness)
                norm_b = [(b - vmin) / (vmax - vmin) for b in betweenness]
                colors = [cmap(0.2 + 0.8 * b) for b in norm_b]
            else:
                colors = [cmap(0.5)] * len(G.nodes())
        else:
            colors = [cmap(0.5)] * len(G.nodes())
        
        return colors, vmin, vmax
    
    def _add_colorbar(self, fig, ax, color_by: str, cmap_name: str, vmin=None, vmax=None):
        """Add colorbar to the plot with proper value labels."""
        try:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            import matplotlib.ticker as ticker
            
            cmap = plt.cm.get_cmap(cmap_name)
            
            # Use actual value range if provided
            if vmin is not None and vmax is not None:
                norm = Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = Normalize(vmin=0, vmax=1)
            
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            
            cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.02)
            cbar.set_label(color_by, fontsize=9)
            
            # Format tick labels based on the color_by type
            if color_by == "Average Year" and vmin is not None and vmax is not None:
                # Show integer years
                cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            elif color_by in ["Betweenness", "Closeness"]:
                # Show decimals for centrality measures
                cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            elif vmin is not None and vmax is not None:
                # For count-based values, show integers
                if vmax - vmin > 10:
                    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
                    
        except Exception as e:
            print(f"Could not add colorbar: {e}")
    
    def _show_nodes_table(self, df: pd.DataFrame):
        """Display nodes table."""
        for widget in self.nodes_frame.winfo_children():
            widget.destroy()
        
        if df is None or len(df) == 0:
            tk.Label(
                self.nodes_frame, text="No node data available",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"],
            ).pack(expand=True)
            return
        
        table = DataTable(self.nodes_frame, theme=self.theme_name)
        table.pack(fill=tk.BOTH, expand=True)
        table.set_data(df)
    
    def _show_edges_table(self, df: pd.DataFrame):
        """Display edges table."""
        for widget in self.edges_frame.winfo_children():
            widget.destroy()
        
        if df is None or len(df) == 0:
            tk.Label(
                self.edges_frame, text="No edge data available",
                font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["text_muted"],
            ).pack(expand=True)
            return
        
        table = DataTable(self.edges_frame, theme=self.theme_name)
        table.pack(fill=tk.BOTH, expand=True)
        table.set_data(df)
    
    def _show_network_stats(self, stats: Dict):
        """Display network statistics."""
        for widget in self.stats_frame.winfo_children():
            widget.destroy()
        
        if not stats:
            return
        
        grid = CardGrid(self.stats_frame, columns=4, theme=self.theme_name)
        grid.pack(fill=tk.X, padx=8, pady=8)
        
        icons = {
            "Nodes": "📍", "Edges": "🔗", "Density": "📊",
            "Avg Degree": "📈", "Avg Clustering": "🔄", "Diameter": "📏",
            "Avg Path Length": "🛤️", "Components": "🧩", "Communities": "👥",
        }
        
        for key, value in stats.items():
            icon = icons.get(key, "📊")
            grid.add_card(StatsCard(grid, key, str(value), icon, self.theme_name))
    
    def _on_build_error(self, error: str):
        """Handle build error."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.plot_frame, text=f"Error: {error}",
            font=FONTS.get_font("body"), bg=self.theme["bg_card"], fg=self.theme["danger"],
            wraplength=400,
        ).pack(expand=True)
        
        tk.Label(
            self.plot_frame, 
            text="Try adjusting parameters:\n• Increase Top N Nodes\n• Reduce Min Edge Weight\n• Check if required columns exist",
            font=FONTS.get_font("small"), bg=self.theme["bg_card"], fg=self.theme["text_muted"],
            justify=tk.LEFT,
        ).pack(pady=10)
    
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
                ("Edge List CSV", "*.csv"),
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
                    if self._edge_df is not None:
                        self._edge_df.to_csv(filename, index=False)
                    else:
                        edges = [(u, v, d.get("weight", 1)) for u, v, d in self._graph.edges(data=True)]
                        pd.DataFrame(edges, columns=["Source", "Target", "Weight"]).to_csv(filename, index=False)
                
                messagebox.showinfo("Success", f"Network exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
    
    def _create_info_content(self, parent):
        """Create Info tab with documentation."""
        info_text = """
NETWORK ANALYSIS
════════════════

Build and analyze bibliometric networks.

NETWORK TYPES
─────────────
• Author Keywords Co-occurrence
  Terms assigned by authors appearing together
  
• Index Keywords Co-occurrence
  Database-assigned terms co-occurring
  
• Co-authorship
  Authors collaborating on papers
  
• Reference Co-citation
  References cited together
  
• Source Co-citation
  Journals cited together
  
• Country Collaboration
  International co-authorship links
  
• Institution Collaboration
  Organizational partnerships
  
• Title/Abstract N-grams
  Text-based term co-occurrence

NODE METRICS
────────────
• Degree: Number of connections
• Betweenness: Bridge between groups
• Closeness: Centrality measure
• Occurrences: Frequency count

LAYOUT ALGORITHMS
─────────────────
• Spring (Fruchterman-Reingold)
• Kamada-Kawai
• Circular
• Random

COMMUNITY DETECTION
───────────────────
• Louvain algorithm
• Label propagation
• Girvan-Newman

VISUALIZATION OPTIONS
─────────────────────
• Node size: By metric
• Node color: By community/attribute
• Edge weight: Connection strength
• Interactive zoom/pan

FILTERING
─────────
• Minimum occurrences
• Minimum edge weight
• Top N nodes

EXPORT
──────
• Network files (GraphML, GEXF, Pajek)
• Node/edge tables (CSV)
• Images (PNG, SVG, PDF)
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
        if self._current_fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._current_fig)
            except:
                pass
        self._current_fig = None
        self._photo_image = None
        super().destroy()
