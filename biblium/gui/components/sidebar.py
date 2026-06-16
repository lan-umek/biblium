# -*- coding: utf-8 -*-
"""
Sidebar Component
=================
Navigation sidebar with collapsible sections.
"""

import tkinter as tk
from typing import Callable, Dict, List, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus


class SidebarItem:
    """A single sidebar menu item."""
    
    def __init__(
        self,
        parent,
        text: str,
        icon: str = "",
        command: Optional[Callable] = None,
        theme: str = "light",
        enabled: bool = True,
    ):
        self.theme = get_theme(theme)
        self.text = text
        self.icon = icon
        self.command = command
        self.enabled = enabled
        self.selected = False
        
        self.frame = tk.Frame(parent, bg=self.theme["bg_sidebar"], cursor="hand2")
        self.frame.pack(fill=tk.X, padx=4, pady=1)
        
        display_text = f"{icon}  {text}" if icon else text
        
        self.label = tk.Label(
            self.frame,
            text=display_text,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_sidebar"],
            fg=self.theme["text_sidebar"] if enabled else self.theme["text_muted"],
            anchor=tk.W,
            padx=16,
            pady=8,
        )
        self.label.pack(fill=tk.X)
        
        if enabled:
            self.frame.bind("<Enter>", self._on_enter)
            self.frame.bind("<Leave>", self._on_leave)
            self.frame.bind("<Button-1>", self._on_click)
            self.label.bind("<Button-1>", self._on_click)
    
    def _on_enter(self, event):
        if not self.selected:
            self.frame.configure(bg=self.theme["bg_sidebar_hover"])
            self.label.configure(bg=self.theme["bg_sidebar_hover"])
    
    def _on_leave(self, event):
        if not self.selected:
            self.frame.configure(bg=self.theme["bg_sidebar"])
            self.label.configure(bg=self.theme["bg_sidebar"])
    
    def _on_click(self, event):
        if self.command and self.enabled:
            self.command()
    
    def set_selected(self, selected: bool):
        """Set selected state."""
        self.selected = selected
        if selected:
            self.frame.configure(bg=self.theme["accent_primary"])
            self.label.configure(
                bg=self.theme["accent_primary"],
                fg=self.theme["text_on_accent"],
            )
        else:
            self.frame.configure(bg=self.theme["bg_sidebar"])
            self.label.configure(
                bg=self.theme["bg_sidebar"],
                fg=self.theme["text_sidebar"],
            )
    
    def show(self):
        """Show this item."""
        self.frame.pack(fill=tk.X, padx=4, pady=1)
    
    def hide(self):
        """Hide this item."""
        self.frame.pack_forget()
    
    def matches_search(self, query: str) -> bool:
        """Check if item matches search query."""
        if not query:
            return True
        return query in self.text.lower() or query in self.icon.lower()
    
    def set_enabled(self, enabled: bool):
        """Set enabled state."""
        self.enabled = enabled
        self.label.configure(
            fg=self.theme["text_sidebar"] if enabled else self.theme["text_muted"],
        )


class SidebarSection:
    """A collapsible section in the sidebar."""
    
    def __init__(
        self,
        parent,
        title: str,
        items: List[Dict],
        theme: str = "light",
        expanded: bool = True,
        on_item_click: Optional[Callable] = None,
    ):
        self.theme = get_theme(theme)
        self.title = title
        self.expanded = expanded
        self.on_item_click = on_item_click
        self.items: Dict[str, SidebarItem] = {}
        
        # Section frame
        self.frame = tk.Frame(parent, bg=self.theme["bg_sidebar"])
        self.frame.pack(fill=tk.X, pady=(12, 0))
        
        # Header (clickable)
        self.header = tk.Frame(self.frame, bg=self.theme["bg_sidebar"], cursor="hand2")
        self.header.pack(fill=tk.X, padx=12)
        
        self.toggle_label = tk.Label(
            self.header,
            text="▾" if expanded else "▸",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_sidebar"],
            fg=self.theme["text_sidebar_muted"],
        )
        self.toggle_label.pack(side=tk.LEFT)
        
        self.title_label = tk.Label(
            self.header,
            text=title.upper(),
            font=FONTS.get_font("small", bold=True),
            bg=self.theme["bg_sidebar"],
            fg=self.theme["text_sidebar_muted"],
        )
        self.title_label.pack(side=tk.LEFT, padx=(4, 0))
        
        # Bind click events
        for widget in [self.header, self.toggle_label, self.title_label]:
            widget.bind("<Button-1>", self._toggle)
        
        # Items container
        self.items_frame = tk.Frame(self.frame, bg=self.theme["bg_sidebar"])
        if expanded:
            self.items_frame.pack(fill=tk.X, pady=(4, 0))
        
        # Add items
        for item_config in items:
            self._add_item(item_config)
    
    def _add_item(self, config: Dict):
        """Add an item to the section."""
        item_id = config.get("id", config.get("text", "").lower().replace(" ", "_"))
        
        def on_click():
            if self.on_item_click:
                self.on_item_click(item_id)
        
        item = SidebarItem(
            self.items_frame,
            text=config.get("text", ""),
            icon=config.get("icon", ""),
            command=on_click,
            theme=self.theme.get("name", "light"),
            enabled=config.get("enabled", True),
        )
        
        self.items[item_id] = item
    
    def _toggle(self, event=None):
        """Toggle section expansion."""
        self.expanded = not self.expanded
        
        if self.expanded:
            self.items_frame.pack(fill=tk.X, pady=(4, 0))
            self.toggle_label.configure(text="▾")
        else:
            self.items_frame.pack_forget()
            self.toggle_label.configure(text="▸")
    
    def select_item(self, item_id: str):
        """Select an item in this section."""
        for id_, item in self.items.items():
            item.set_selected(id_ == item_id)
    
    def clear_selection(self):
        """Clear selection in this section."""
        for item in self.items.values():
            item.set_selected(False)
    
    def show(self):
        """Show this section."""
        self.frame.pack(fill=tk.X, pady=(8, 0))
    
    def hide(self):
        """Hide this section."""
        self.frame.pack_forget()
    
    def expand(self):
        """Expand this section."""
        if not self.expanded:
            self.expanded = True
            self.items_frame.pack(fill=tk.X, pady=(4, 0))
            self.toggle_label.configure(text="▾")
    
    def collapse(self):
        """Collapse this section."""
        if self.expanded:
            self.expanded = False
            self.items_frame.pack_forget()
            self.toggle_label.configure(text="▸")
    
    def filter_items(self, query: str) -> int:
        """
        Filter items by search query.
        Returns number of visible items.
        """
        if not query:
            # Show all items
            for item in self.items.values():
                item.show()
            return len(self.items)
        
        visible_count = 0
        for item in self.items.values():
            if item.matches_search(query):
                item.show()
                visible_count += 1
            else:
                item.hide()
        
        return visible_count


class Sidebar(tk.Frame):
    """
    Main navigation sidebar.
    
    Usage:
        sidebar = Sidebar(parent)
        sidebar.set_menu(MENU_CONFIG)
        sidebar.on_navigate = lambda panel_id: switch_to_panel(panel_id)
    """
    
    # Menu configuration
    MENU_CONFIG = [
        {
            "title": "DATA",
            "items": [
                {"id": "load", "text": "Load Dataset", "icon": "📂"},
                {"id": "api_data", "text": "API Data", "icon": "🌐"},
                {"id": "view", "text": "View Data", "icon": "📋"},
                {"id": "filter", "text": "Filter", "icon": "🔍"},
            ]
        },
        {
            "title": "ANALYSIS",
            "items": [
                {"id": "overview", "text": "Overview", "icon": "ℹ️"},
                {"id": "counts", "text": "Counts", "icon": "📊"},
                {"id": "statistics", "text": "Statistics", "icon": "📈"},
                {"id": "laws", "text": "Laws", "icon": "📚"},
                {"id": "top_cited", "text": "Top Cited", "icon": "🏆"},
                {"id": "citation_distribution", "text": "Citation Distribution", "icon": "📉"},
                {"id": "collaboration", "text": "Collaboration Metrics", "icon": "👥"},
                {"id": "reference_benchmark", "text": "Reference Benchmark", "icon": "🎯"},
                {"id": "diversity_indices", "text": "Diversity Indices", "icon": "🌈"},
                {"id": "altmetrics", "text": "Altmetrics", "icon": "📱"},
                {"id": "novelty", "text": "Novelty Analysis", "icon": "🔬"},
                {"id": "sentiment", "text": "Sentiment Analysis", "icon": "💭"},
                {"id": "kfields", "text": "K-Fields Plot", "icon": "🔀"},
                {"id": "relationships", "text": "Relationships", "icon": "🔗"},
            ]
        },
        {
            "title": "TEMPORAL ANALYSIS",
            "items": [
                {"id": "trends", "text": "Scientific Production", "icon": "📊"},
                {"id": "production", "text": "Entity Over Time", "icon": "📅"},
                {"id": "top_items_timeline", "text": "Top Items Timeline", "icon": "⏱️"},
                {"id": "trend_topics", "text": "Trend Topics", "icon": "🔥"},
                {"id": "growth_models", "text": "Growth Models", "icon": "📈"},
                {"id": "life_cycle", "text": "Life Cycle", "icon": "🔄"},
                {"id": "temporal_diversity", "text": "Temporal Diversity", "icon": "🌈"},
            ]
        },
        {
            "title": "VISUALIZATION",
            "items": [
                {"id": "wordcloud", "text": "Word Cloud", "icon": "☁️"},
                {"id": "treemap", "text": "Treemap", "icon": "🗃️"},
                {"id": "distribution", "text": "Distribution", "icon": "📊"},
                {"id": "geographic", "text": "Geographic", "icon": "🌍"},
                {"id": "race_bar", "text": "Race Bar Animation", "icon": "🏁"},
            ]
        },
        {
            "title": "FACTORIAL",
            "items": [
                {"id": "factorial", "text": "Factorial Analysis", "icon": "📐"},
            ]
        },
        {
            "title": "NETWORKS",
            "items": [
                {"id": "network", "text": "Co-occurrence Networks", "icon": "🔗"},
                {"id": "citation_network", "text": "Citation Network", "icon": "📄"},
                {"id": "thematic_map", "text": "Thematic Map", "icon": "🗺️"},
                {"id": "historiograph", "text": "Historiograph", "icon": "🕰️"},
            ]
        },
        {
            "title": "MAPPING",
            "items": [
                {"id": "topic_modeling", "text": "Topic Modeling", "icon": "📚"},
                {"id": "dynamic_topics", "text": "Dynamic Topics", "icon": "🔄"},
            ]
        },
        {
            "title": "CLUSTERING",
            "items": [
                {"id": "document_clustering", "text": "Document Clustering", "icon": "📄"},
                {"id": "entity_clustering", "text": "Entity Clustering", "icon": "🏷️"},
            ]
        },
        {
            "title": "ADVANCED",
            "items": [
                {"id": "concept_builder", "text": "Concept Builder", "icon": "🔧"},
                {"id": "pa_concepts", "text": "PA Concepts", "icon": "🏛️"},
                {"id": "my_concepts", "text": "My Concepts", "icon": "📁"},
                {"id": "sdg", "text": "SDG Identifier", "icon": "🌍"},
                {"id": "sleeping_beauty", "text": "Sleeping Beauty", "icon": "😴"},
                {"id": "citation_patterns", "text": "Citation Patterns", "icon": "📈"},
                {"id": "citation_velocity", "text": "Citation Velocity", "icon": "🚀"},
                {"id": "reference_diversity", "text": "Reference Diversity", "icon": "📚"},
                {"id": "concept_extraction", "text": "Concept Extraction", "icon": "🏷️"},
                {"id": "disruption_index", "text": "Disruption Index", "icon": "💥"},
                {"id": "repository_links", "text": "Repository Links", "icon": "🔗"},
                {"id": "fronts", "text": "Research Fronts", "icon": "🌊"},
            ]
        },
        {
            "title": "STATISTICS",
            "items": [
                {"id": "compare_means", "text": "Compare Means", "icon": "📊"},
                {"id": "crosstabs", "text": "Crosstabs", "icon": "📋"},
                {"id": "correlation", "text": "Correlation", "icon": "📈"},
            ]
        },
        {
            "title": "GROUPS",
            "items": [
                {"id": "group_setup", "text": "Setup Groups", "icon": "⚙️"},
                {"id": "group_counts", "text": "Group Counts", "icon": "📊"},
                {"id": "group_stats", "text": "Group Statistics", "icon": "📈"},
                {"id": "group_compare", "text": "Compare Groups", "icon": "⚖️"},
                {"id": "group_intersections", "text": "Intersections", "icon": "🔀"},
                {"id": "group_associations", "text": "Associations", "icon": "🔗"},
                {"id": "group_diversity", "text": "Group Diversity", "icon": "🌈"},
                {"id": "group_classification", "text": "Classification (ML)", "icon": "🎯"},
                {"id": "group_logistic", "text": "Logistic Regression", "icon": "📉"},
            ]
        },
        {
            "title": "REPORTS",
            "items": [
                {"id": "report_builder", "text": "Report Builder", "icon": "📝"},
                {"id": "custom_report", "text": "Custom Report", "icon": "🎨"},
            ]
        },
    ]
    
    def __init__(
        self,
        parent,
        theme: str = "light",
        on_navigate: Optional[Callable] = None,
        **kwargs
    ):
        self.theme_name = theme
        self.theme = get_theme(theme)
        self.on_navigate = on_navigate
        
        super().__init__(
            parent,
            bg=self.theme["bg_sidebar"],
            width=LAYOUT.sidebar_width,
            **kwargs
        )
        self.pack_propagate(False)
        
        self.sections: Dict[str, SidebarSection] = {}
        self.current_selection = None
        
        self._create_header()
        self._create_menu()
        self._create_footer()
    
    def _create_header(self):
        """Create the sidebar header with logo and search."""
        header = tk.Frame(self, bg=self.theme["bg_sidebar"])
        header.pack(fill=tk.X, pady=(20, 10))
        
        tk.Label(
            header,
            text="📚 BIBLIUM",
            font=("Segoe UI", 18, "bold"),
            bg=self.theme["bg_sidebar"],
            fg=self.theme["text_sidebar"],
        ).pack()
        
        tk.Label(
            header,
            text="Bibliometric Analysis",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_sidebar"],
            fg=self.theme["text_sidebar_muted"],
        ).pack()
        
        # Search field
        search_frame = tk.Frame(self, bg=self.theme["bg_sidebar"])
        search_frame.pack(fill=tk.X, padx=12, pady=(0, 10))
        
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self._on_search_changed)
        
        self.search_entry = tk.Entry(
            search_frame,
            textvariable=self.search_var,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            insertbackground=self.theme["text_primary"],
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.theme["border"],
            highlightcolor=self.theme["accent_primary"],
        )
        self.search_entry.pack(fill=tk.X, ipady=6)
        
        # Placeholder text
        self.search_entry.insert(0, "🔍 Search panels...")
        self.search_entry.config(fg=self.theme["text_muted"])
        self.search_entry.bind("<FocusIn>", self._on_search_focus_in)
        self.search_entry.bind("<FocusOut>", self._on_search_focus_out)
        self.search_entry.bind("<Escape>", self._on_search_escape)
    
    def _on_search_focus_in(self, event):
        """Handle search entry focus in."""
        if self.search_entry.get() == "🔍 Search panels...":
            self.search_entry.delete(0, tk.END)
            self.search_entry.config(fg=self.theme["text_primary"])
    
    def _on_search_focus_out(self, event):
        """Handle search entry focus out."""
        if not self.search_entry.get():
            self.search_entry.insert(0, "🔍 Search panels...")
            self.search_entry.config(fg=self.theme["text_muted"])
    
    def _on_search_escape(self, event):
        """Handle escape key in search - clear and unfocus."""
        self.search_var.set("")
        self.search_entry.delete(0, tk.END)
        self.search_entry.insert(0, "🔍 Search panels...")
        self.search_entry.config(fg=self.theme["text_muted"])
        self.focus_set()  # Remove focus from search entry
    
    def _on_search_changed(self, *args):
        """Handle search text change."""
        query = self.search_var.get().lower().strip()
        
        # Ignore placeholder text
        if query == "🔍 search panels...":
            query = ""
        
        self._filter_menu(query)
    
    def _create_menu(self):
        """Create the scrollable menu."""
        # Canvas for scrolling
        self.canvas = tk.Canvas(
            self,
            bg=self.theme["bg_sidebar"],
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Scrollable frame
        self.menu_frame = tk.Frame(self.canvas, bg=self.theme["bg_sidebar"])
        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.menu_frame,
            anchor=tk.NW,
            width=LAYOUT.sidebar_width,
        )
        
        # Configure scrolling
        self.menu_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Mouse wheel scrolling
        self.canvas.bind("<Enter>", lambda e: self._bind_wheel())
        self.canvas.bind("<Leave>", lambda e: self._unbind_wheel())
        
        # Add sections
        for section_config in self.MENU_CONFIG:
            section = SidebarSection(
                self.menu_frame,
                title=section_config["title"],
                items=section_config["items"],
                theme=self.theme_name,
                expanded=False,  # Start with menus collapsed
                on_item_click=self._on_item_click,
            )
            self.sections[section_config["title"]] = section
    
    def _create_footer(self):
        """Create the sidebar footer."""
        footer = tk.Frame(self, bg=self.theme["bg_sidebar"])
        footer.pack(fill=tk.X, side=tk.BOTTOM, pady=12)
        
        # Separator
        sep = tk.Frame(footer, bg=self.theme["border"], height=1)
        sep.pack(fill=tk.X, padx=16, pady=(0, 12))
        
        # Settings button - emit special event, not panel change
        SidebarItem(
            footer,
            text="Settings",
            icon="⚙️",
            command=lambda: event_bus.emit("open_settings", {}),
            theme=self.theme_name,
        )
        
        # Help button - emit special event
        SidebarItem(
            footer,
            text="Help",
            icon="❓",
            command=lambda: event_bus.emit("open_help", {}),
            theme=self.theme_name,
        )
    
    def _on_frame_configure(self, event):
        """Update scroll region."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _on_canvas_configure(self, event):
        """Adjust frame width to canvas."""
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def _bind_wheel(self):
        """Bind mouse wheel."""
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
    
    def _unbind_wheel(self):
        """Unbind mouse wheel."""
        self.canvas.unbind_all("<MouseWheel>")
    
    def _on_wheel(self, event):
        """Handle mouse wheel scroll."""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _on_item_click(self, item_id: str):
        """Handle item click."""
        # Clear all selections
        for section in self.sections.values():
            section.clear_selection()
        
        # Set new selection
        for section in self.sections.values():
            if item_id in section.items:
                section.select_item(item_id)
                break
        
        self.current_selection = item_id
        
        # Emit event
        event_bus.emit(EventBus.PANEL_CHANGED, {"panel": item_id})
        
        # Call callback
        if self.on_navigate:
            self.on_navigate(item_id)
    
    def select(self, item_id: str):
        """Programmatically select an item."""
        self._on_item_click(item_id)
    
    def get_selection(self) -> Optional[str]:
        """Get current selection."""
        return self.current_selection
    
    def _filter_menu(self, query: str):
        """Filter menu items based on search query."""
        if not query:
            # Show all sections and items, collapse sections
            for section in self.sections.values():
                section.show()
                section.filter_items("")
                section.collapse()
            return
        
        # Filter each section
        for section in self.sections.values():
            visible_count = section.filter_items(query)
            
            if visible_count > 0:
                section.show()
                section.expand()  # Auto-expand sections with matches
            else:
                section.hide()
    
    def clear_search(self):
        """Clear the search field."""
        self.search_var.set("")
        self._on_search_escape(None)
