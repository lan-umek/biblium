# -*- coding: utf-8 -*-
"""
GUI Configuration and Themes
============================
Central configuration for the Biblium GUI application.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


# =============================================================================
# COLOR THEMES
# =============================================================================

LIGHT_THEME = {
    "name": "light",
    
    # Backgrounds
    "bg_primary": "#ffffff",
    "bg_secondary": "#f8fafc",
    "bg_sidebar": "#1e293b",
    "bg_sidebar_hover": "#334155",
    "bg_card": "#ffffff",
    "bg_input": "#ffffff",
    "bg_input_focus": "#f0fdf4",
    "bg_header": "#f1f5f9",
    
    # Text colors
    "text_primary": "#1e293b",
    "text_secondary": "#64748b",
    "text_muted": "#94a3b8",
    "text_sidebar": "#e2e8f0",
    "text_sidebar_muted": "#94a3b8",
    "text_on_accent": "#ffffff",
    "text_on_success": "#ffffff",
    "text_on_warning": "#1e293b",
    "text_on_danger": "#ffffff",
    
    # Accent colors
    "accent_primary": "#3b82f6",
    "accent_hover": "#2563eb",
    "accent_light": "#dbeafe",
    "success": "#10b981",
    "success_hover": "#059669",
    "success_light": "#d1fae5",
    "warning": "#f59e0b",
    "warning_hover": "#d97706",
    "warning_light": "#fef3c7",
    "danger": "#ef4444",
    "danger_hover": "#dc2626",
    "danger_light": "#fee2e2",
    "info": "#06b6d4",
    "info_light": "#cffafe",
    
    # Borders and dividers
    "border": "#e2e8f0",
    "border_focus": "#3b82f6",
    "divider": "#f1f5f9",
    
    # Table colors
    "table_header": "#f8fafc",
    "table_row_alt": "#f8fafc",
    "table_row_hover": "#f1f5f9",
    "table_row_selected": "#dbeafe",
    
    # Special
    "shadow": "rgba(0, 0, 0, 0.1)",
    "overlay": "rgba(0, 0, 0, 0.5)",
}

DARK_THEME = {
    "name": "dark",
    
    # Backgrounds
    "bg_primary": "#0f172a",
    "bg_secondary": "#1e293b",
    "bg_sidebar": "#020617",
    "bg_sidebar_hover": "#1e293b",
    "bg_card": "#1e293b",
    "bg_input": "#334155",
    "bg_input_focus": "#475569",
    "bg_header": "#1e293b",
    
    # Text colors
    "text_primary": "#f8fafc",
    "text_secondary": "#cbd5e1",
    "text_muted": "#64748b",
    "text_sidebar": "#e2e8f0",
    "text_sidebar_muted": "#64748b",
    "text_on_accent": "#ffffff",
    "text_on_success": "#ffffff",
    "text_on_warning": "#1e293b",
    "text_on_danger": "#ffffff",
    
    # Accent colors
    "accent_primary": "#60a5fa",
    "accent_hover": "#3b82f6",
    "accent_light": "#1e3a5f",
    "success": "#34d399",
    "success_hover": "#10b981",
    "success_light": "#064e3b",
    "warning": "#fbbf24",
    "warning_hover": "#f59e0b",
    "warning_light": "#78350f",
    "danger": "#f87171",
    "danger_hover": "#ef4444",
    "danger_light": "#7f1d1d",
    "info": "#22d3ee",
    "info_light": "#164e63",
    
    # Borders and dividers
    "border": "#334155",
    "border_focus": "#60a5fa",
    "divider": "#1e293b",
    
    # Table colors
    "table_header": "#334155",
    "table_row_alt": "#1e293b",
    "table_row_hover": "#334155",
    "table_row_selected": "#1e3a5f",
    
    # Special
    "shadow": "rgba(0, 0, 0, 0.3)",
    "overlay": "rgba(0, 0, 0, 0.7)",
}

THEMES = {
    "light": LIGHT_THEME,
    "dark": DARK_THEME,
}


# =============================================================================
# FONTS
# =============================================================================

@dataclass
class FontConfig:
    """Font configuration"""
    family: str = "Segoe UI"
    family_mono: str = "Consolas"
    
    # Sizes
    size_title: int = 18
    size_heading: int = 12
    size_body: int = 10
    size_small: int = 9
    size_tiny: int = 8
    
    def get_font(self, style: str = "body", bold: bool = False) -> Tuple:
        """Get font tuple for tkinter"""
        sizes = {
            "title": self.size_title,
            "heading": self.size_heading,
            "body": self.size_body,
            "small": self.size_small,
            "tiny": self.size_tiny,
            "mono": self.size_body,
        }
        family = self.family_mono if style == "mono" else self.family
        size = sizes.get(style, self.size_body)
        weight = "bold" if bold else "normal"
        return (family, size, weight)


FONTS = FontConfig()


# =============================================================================
# LAYOUT CONSTANTS
# =============================================================================

@dataclass
class LayoutConfig:
    """Layout configuration"""
    # Sidebar
    sidebar_width: int = 240
    sidebar_collapsed_width: int = 60
    sidebar_item_height: int = 36
    sidebar_section_padding: int = 16
    
    # Toolbar
    toolbar_height: int = 44
    toolbar_button_size: int = 32
    toolbar_padding: int = 8
    
    # Status bar
    statusbar_height: int = 28
    
    # Content
    content_padding: int = 16
    card_padding: int = 16
    card_border_radius: int = 8
    
    # Spacing
    spacing_xs: int = 4
    spacing_sm: int = 8
    spacing_md: int = 12
    spacing_lg: int = 16
    spacing_xl: int = 24
    
    # Minimum sizes
    min_window_width: int = 1200
    min_window_height: int = 700
    default_window_width: int = 1500
    default_window_height: int = 900
    
    # Split pane
    options_panel_width: int = 320
    options_panel_min_width: int = 280
    options_panel_max_width: int = 450


LAYOUT = LayoutConfig()


# =============================================================================
# DATA SOURCE MAPPINGS
# =============================================================================

DATABASE_OPTIONS = {
    "OpenAlex": "oa",
    "Scopus": "scopus",
    "Web of Science": "wos",
    "PubMed": "pubmed",
    "Dimensions": "dimensions",
    "Auto-detect": "",
}

PREPROCESS_LEVELS = {
    "0 - None (raw data)": 0,
    "1 - Basic (labels, country info)": 1,
    "2 - Standard (keywords, text processing)": 2,
    "3 - Extended (science mappings)": 3,
    "4 - Full (interdisciplinarity measures)": 4,
}

KEYWORD_OPTIONS = {
    "Author Keywords": "author",
    "Index Keywords": "index",
    "Both (combined)": "both",
}

LANGUAGE_OPTIONS = {
    "English": "en",
    "Slovenian": "sl",
    "German": "de",
    "Spanish": "es",
    "French": "fr",
    "Italian": "it",
    "Portuguese": "pt",
}

COLOR_PALETTES = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "Blues", "Greens", "Oranges", "Reds", "Purples",
    "YlOrRd", "YlGnBu", "RdYlBu", "RdYlGn", "Spectral",
    "tab10", "tab20", "Set1", "Set2", "Set3",
]


# =============================================================================
# ENTITY TYPES
# =============================================================================

ENTITY_TYPES = {
    # Primary entities (most commonly used)
    "sources": {
        "label": "Sources",
        "icon": "📚",
        "method": "count_sources",
        "stats_method": "get_sources_stats",
        "result_attr": "sources_counts_df",
        "stats_attr": "sources_stats_df",
    },
    "authors": {
        "label": "Authors",
        "icon": "👤",
        "method": "count_authors",
        "stats_method": "get_authors_stats",
        "result_attr": "authors_counts_df",
        "stats_attr": "authors_stats_df",
    },
    "author_keywords": {
        "label": "Author Keywords",
        "icon": "🏷️",
        "method": "count_author_keywords",
        "stats_method": "get_author_keywords_stats",
        "result_attr": "author_keywords_counts_df",
        "stats_attr": "author_keywords_stats_df",
    },
    "index_keywords": {
        "label": "Index Keywords",
        "icon": "🔖",
        "method": "count_index_keywords",
        "stats_method": "get_index_keywords_stats",
        "result_attr": "index_keywords_counts_df",
        "stats_attr": "index_keywords_stats_df",
    },
    "affiliations": {
        "label": "Affiliations",
        "icon": "🏛️",
        "method": "count_affiliations",
        "stats_method": "get_affiliations_stats",
        "result_attr": "affiliations_counts_df",
        "stats_attr": "affiliations_stats_df",
    },
    "countries": {
        "label": "Countries",
        "icon": "🌍",
        "method": "count_all_countries",
        "stats_method": "get_all_countries_stats",
        "result_attr": "all_countries_counts_df",
        "stats_attr": "all_countries_stats_df",
    },
    "references": {
        "label": "References",
        "icon": "📖",
        "method": "count_references",
        "stats_method": "get_references_stats",
        "result_attr": "references_counts_df",
        "stats_attr": "references_stats_df",
    },
    "document_types": {
        "label": "Document Types",
        "icon": "📄",
        "method": "count_document_types",
        "stats_method": "get_document_type_stats",
        "result_attr": "document_types_counts_df",
        "stats_attr": "document_types_stats_df",
    },
    # Secondary entities
    "both_keywords": {
        "label": "All Keywords (Author + Index)",
        "icon": "🔗",
        "method": "count_both_keywords",
        "stats_method": "get_author_and_index_keywords_stats",
        "result_attr": "author_and_index_keywords_counts_df",
        "stats_attr": "author_and_index_keywords_stats_df",
    },
    # Subject classification entities
    "fields": {
        "label": "Subject Fields",
        "icon": "🔬",
        "method": "count_fields",
        "stats_method": "get_fields_stats",
        "result_attr": "fields_counts_df",
        "stats_attr": "fields_stats_df",
    },
    "areas": {
        "label": "Subject Areas",
        "icon": "📊",
        "method": "count_areas",
        "stats_method": "get_areas_stats",
        "result_attr": "areas_counts_df",
        "stats_attr": "areas_stats_df",
    },
    "sciences": {
        "label": "Sciences",
        "icon": "🧪",
        "method": "count_sciences",
        "stats_method": "get_sciences_stats",
        "result_attr": "sciences_counts_df",
        "stats_attr": "sciences_stats_df",
    },
    # N-gram entities
    "ngrams_title": {
        "label": "N-grams (Title)",
        "icon": "📝",
        "method": "count_ngrams_title",
        "stats_method": "get_ngrams_title_stats",
        "result_attr": "words_tit_counts_df",
        "stats_attr": "ngrams_title_stats_df",
        "has_ngram_options": True,
    },
    "ngrams_abstract": {
        "label": "N-grams (Abstract)",
        "icon": "📜",
        "method": "count_ngrams_abstract",
        "stats_method": "get_ngrams_abstract_stats",
        "result_attr": "words_abs_counts_df",
        "stats_attr": "ngrams_abstract_stats_df",
        "has_ngram_options": True,
    },
    "ngrams_combined": {
        "label": "N-grams (Combined Text)",
        "icon": "📃",
        "method": "count_ngrams_combined_text",
        "stats_method": "get_ngrams_combined_text_stats",
        "result_attr": "words_comb_counts_df",
        "stats_attr": "ngrams_combined_text_stats_df",
        "has_ngram_options": True,
    },
}


# =============================================================================
# NETWORK TYPES
# =============================================================================

NETWORK_TYPES = {
    "keyword_author": {
        "label": "Author Keywords Co-occurrence",
        "icon": "🏷️",
        "description": "Author-assigned keywords appearing together",
    },
    "keyword_index": {
        "label": "Index Keywords Co-occurrence",
        "icon": "📑",
        "description": "Database-assigned keywords co-occurrence",
    },
    "keyword_all": {
        "label": "All Keywords Co-occurrence",
        "icon": "🔗",
        "description": "All keywords (author + index) co-occurrence",
    },
    "coauthorship": {
        "label": "Co-authorship",
        "icon": "👥",
        "description": "Collaboration between authors",
    },
    "co_citation": {
        "label": "Reference Co-citation",
        "icon": "🔄",
        "description": "References cited together in documents",
    },
    "source_cocitation": {
        "label": "Source Co-citation",
        "icon": "📰",
        "description": "Journals/sources cited together",
    },
    "country_collaboration": {
        "label": "Country Collaboration",
        "icon": "🌍",
        "description": "International research collaboration",
    },
    "institution_collaboration": {
        "label": "Institution Collaboration",
        "icon": "🏛️",
        "description": "Collaboration between institutions",
    },
    "title_ngrams": {
        "label": "Title N-grams Co-occurrence",
        "icon": "📝",
        "description": "N-grams from titles appearing together",
    },
    "abstract_ngrams": {
        "label": "Abstract N-grams Co-occurrence",
        "icon": "📄",
        "description": "N-grams from abstracts appearing together",
    },
}


# =============================================================================
# LAYOUT ALGORITHMS
# =============================================================================

NETWORK_LAYOUTS = [
    "spring",
    "kamada_kawai",
    "circular",
    "shell",
    "spectral",
    "random",
]

COMMUNITY_ALGORITHMS = [
    "louvain",
    "greedy_modularity",
    "label_propagation",
    "girvan_newman",
]


# =============================================================================
# BIBLIOMETRIC LAWS
# =============================================================================

BIBLIOMETRIC_LAWS = {
    "lotka": {
        "label": "Lotka's Law",
        "description": "Author productivity distribution",
        "entity": "authors",
    },
    "bradford": {
        "label": "Bradford's Law",
        "description": "Journal scatter",
        "entity": "sources",
    },
    "zipf": {
        "label": "Zipf's Law",
        "description": "Word frequency distribution",
        "entity": "keywords",
    },
    "price": {
        "label": "Price's Law",
        "description": "Square root of contributors produce half of output",
        "entity": "authors",
    },
    "pareto": {
        "label": "Pareto Principle",
        "description": "80/20 distribution",
        "entity": "any",
    },
}


# =============================================================================
# KEYBOARD SHORTCUTS
# =============================================================================

SHORTCUTS = {
    "file_open": "<Control-o>",
    "file_save": "<Control-s>",
    "file_export": "<Control-e>",
    "edit_undo": "<Control-z>",
    "edit_redo": "<Control-y>",
    "run_analysis": "<Control-r>",
    "run_all": "<Control-Shift-R>",
    "find": "<Control-f>",
    "generate_report": "<Control-g>",
    "tab_next": "<Control-Tab>",
    "tab_prev": "<Control-Shift-Tab>",
    "tab_close": "<Control-w>",
    "help": "<F1>",
    "refresh": "<F5>",
    "fullscreen": "<F11>",
    "cancel": "<Escape>",
    "tab_1": "<Control-Key-1>",
    "tab_2": "<Control-Key-2>",
    "tab_3": "<Control-Key-3>",
    "tab_4": "<Control-Key-4>",
    "tab_5": "<Control-Key-5>",
}


# =============================================================================
# APPLICATION STATE
# =============================================================================

@dataclass
class AppState:
    """Application state container"""
    
    # Data state
    dataset_loaded: bool = False
    dataset_path: Optional[str] = None
    database_type: str = ""
    n_documents: int = 0
    year_range: Tuple[int, int] = (0, 0)
    
    # Analysis state
    analyses_completed: List[str] = field(default_factory=list)
    current_results: Dict[str, any] = field(default_factory=dict)
    
    # UI state
    current_panel: str = "home"
    sidebar_collapsed: bool = False
    open_tabs: List[str] = field(default_factory=list)
    active_tab: int = 0
    
    # Settings
    theme: str = "light"
    auto_save: bool = True
    recent_files: List[str] = field(default_factory=list)
    
    def add_recent_file(self, path: str, max_recent: int = 10):
        """Add a file to recent files list"""
        if path in self.recent_files:
            self.recent_files.remove(path)
        self.recent_files.insert(0, path)
        self.recent_files = self.recent_files[:max_recent]
    
    def mark_analysis_complete(self, analysis_name: str):
        """Mark an analysis as completed"""
        if analysis_name not in self.analyses_completed:
            self.analyses_completed.append(analysis_name)
    
    def is_analysis_complete(self, analysis_name: str) -> bool:
        """Check if an analysis has been completed"""
        return analysis_name in self.analyses_completed
    
    def reset_data_state(self):
        """Reset data-related state when loading new dataset"""
        self.analyses_completed.clear()
        self.current_results.clear()


# =============================================================================
# CONFIGURATION FILE HANDLING
# =============================================================================

def get_config_path() -> Path:
    """Get path to configuration file"""
    config_dir = Path.home() / ".biblium"
    config_dir.mkdir(exist_ok=True)
    return config_dir / "gui_config.json"


def load_config() -> Dict:
    """Load configuration from file"""
    config_path = get_config_path()
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_config(config: Dict):
    """Save configuration to file"""
    config_path = get_config_path()
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    except Exception:
        pass


def get_theme(theme_name: str = "light") -> Dict:
    """Get theme colors by name"""
    return THEMES.get(theme_name, LIGHT_THEME)
