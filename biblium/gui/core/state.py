# -*- coding: utf-8 -*-
"""
State Manager
=============
Centralized state management for the application.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path


@dataclass
class DataState:
    """State related to loaded data"""
    loaded: bool = False
    path: Optional[str] = None
    database: str = ""
    n_documents: int = 0
    year_range: Tuple[int, int] = (0, 0)
    columns: List[str] = field(default_factory=list)
    filtered: bool = False
    original_n_documents: int = 0


@dataclass
class ReportItem:
    """A single item in the report queue."""
    item_type: str  # "plot" or "table"
    title: str
    source_panel: str  # Panel that generated this item
    data: Any  # For tables: DataFrame, for plots: figure or image bytes
    timestamp: str = ""
    description: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            from datetime import datetime
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ReportQueue:
    """
    Manages the queue of plots and tables to be added to reports.
    
    Usage:
        queue = ReportQueue()
        queue.add_plot(fig, "Diversity Plot", "Diversity Panel")
        queue.add_table(df, "Top Authors", "Production Panel")
        
        # Get all items
        items = queue.get_items()
        
        # Clear queue
        queue.clear()
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._items: List[ReportItem] = []
            cls._instance._listeners: List[Callable] = []
        return cls._instance
    
    @property
    def items(self) -> List[ReportItem]:
        """Get all queued items."""
        return self._items.copy()
    
    def __len__(self) -> int:
        return len(self._items)
    
    def add_plot(self, figure_or_bytes: Any, title: str, source_panel: str, description: str = "") -> ReportItem:
        """
        Add a plot to the report queue.
        
        Parameters
        ----------
        figure_or_bytes : matplotlib.Figure or bytes
            The matplotlib figure or PNG bytes to add
        title : str
            Title for the plot in the report
        source_panel : str
            Name of the panel that generated this plot
        description : str, optional
            Additional description or caption
            
        Returns
        -------
        ReportItem
            The created report item
        """
        import io
        
        # Convert figure to PNG bytes if needed
        if hasattr(figure_or_bytes, 'savefig'):
            buf = io.BytesIO()
            figure_or_bytes.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            data = buf.getvalue()
        else:
            data = figure_or_bytes
        
        item = ReportItem(
            item_type="plot",
            title=title,
            source_panel=source_panel,
            data=data,
            description=description,
        )
        self._items.append(item)
        self._notify("add", item)
        return item
    
    def add_table(self, dataframe: Any, title: str, source_panel: str, description: str = "") -> ReportItem:
        """
        Add a table to the report queue.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            The DataFrame to add
        title : str
            Title for the table in the report
        source_panel : str
            Name of the panel that generated this table
        description : str, optional
            Additional description or caption
            
        Returns
        -------
        ReportItem
            The created report item
        """
        # Make a copy of the DataFrame
        try:
            import pandas as pd
            if isinstance(dataframe, pd.DataFrame):
                data = dataframe.copy()
            else:
                data = dataframe
        except ImportError:
            data = dataframe
        
        item = ReportItem(
            item_type="table",
            title=title,
            source_panel=source_panel,
            data=data,
            description=description,
        )
        self._items.append(item)
        self._notify("add", item)
        return item
    
    def remove(self, item: ReportItem) -> bool:
        """Remove an item from the queue."""
        if item in self._items:
            self._items.remove(item)
            self._notify("remove", item)
            return True
        return False
    
    def remove_by_index(self, index: int) -> bool:
        """Remove an item by its index."""
        if 0 <= index < len(self._items):
            item = self._items.pop(index)
            self._notify("remove", item)
            return True
        return False
    
    def clear(self):
        """Clear all items from the queue."""
        self._items.clear()
        self._notify("clear", None)
    
    def get_items(self, item_type: str = None) -> List[ReportItem]:
        """
        Get items from the queue.
        
        Parameters
        ----------
        item_type : str, optional
            Filter by type: "plot", "table", or None for all
            
        Returns
        -------
        List[ReportItem]
            Matching items
        """
        if item_type is None:
            return self._items.copy()
        return [item for item in self._items if item.item_type == item_type]
    
    def get_plots(self) -> List[ReportItem]:
        """Get all plots in the queue."""
        return self.get_items("plot")
    
    def get_tables(self) -> List[ReportItem]:
        """Get all tables in the queue."""
        return self.get_items("table")
    
    def move_up(self, index: int) -> bool:
        """Move an item up in the queue."""
        if 0 < index < len(self._items):
            self._items[index], self._items[index - 1] = self._items[index - 1], self._items[index]
            self._notify("reorder", None)
            return True
        return False
    
    def move_down(self, index: int) -> bool:
        """Move an item down in the queue."""
        if 0 <= index < len(self._items) - 1:
            self._items[index], self._items[index + 1] = self._items[index + 1], self._items[index]
            self._notify("reorder", None)
            return True
        return False
    
    def on_change(self, callback: Callable):
        """Register a callback for queue changes."""
        self._listeners.append(callback)
    
    def off_change(self, callback: Callable):
        """Unregister a callback."""
        if callback in self._listeners:
            self._listeners.remove(callback)
    
    def _notify(self, action: str, item: Optional[ReportItem]):
        """Notify listeners of changes."""
        for callback in self._listeners:
            try:
                callback(action, item, self._items)
            except Exception as e:
                print(f"Error in report queue callback: {e}")


# Global report queue instance
report_queue = ReportQueue()


@dataclass  
class AnalysisState:
    """State related to analyses"""
    completed: List[str] = field(default_factory=list)
    in_progress: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)
    
    def is_complete(self, name: str) -> bool:
        return name in self.completed
    
    def mark_complete(self, name: str):
        if name not in self.completed:
            self.completed.append(name)
    
    def clear(self):
        self.completed.clear()
        self.results.clear()
        self.in_progress = None


@dataclass
class UIState:
    """State related to UI"""
    current_panel: str = "home"
    open_tabs: List[str] = field(default_factory=list)
    active_tab: int = 0
    sidebar_collapsed: bool = False
    sidebar_sections: Dict[str, bool] = field(default_factory=dict)  # section: expanded
    window_geometry: str = ""
    
    def toggle_sidebar_section(self, section: str):
        self.sidebar_sections[section] = not self.sidebar_sections.get(section, True)
    
    def is_section_expanded(self, section: str) -> bool:
        return self.sidebar_sections.get(section, True)


@dataclass
class SettingsState:
    """Application settings"""
    theme: str = "light"
    auto_save: bool = True
    confirm_exit: bool = True
    recent_files: List[str] = field(default_factory=list)
    default_database: str = "scopus"
    default_preprocess: int = 2
    plot_dpi: int = 150
    plot_colormap: str = "viridis"
    results_folder: str = "results"
    max_recent_files: int = 10
    default_top_n: int = 50
    
    # AI settings
    ai_provider: str = "openai"
    ai_model: str = "gpt-4o-mini"
    ai_api_key: str = ""
    ai_custom_prompt: str = ""
    
    def add_recent_file(self, path: str):
        if path in self.recent_files:
            self.recent_files.remove(path)
        self.recent_files.insert(0, path)
        self.recent_files = self.recent_files[:self.max_recent_files]


class StateManager:
    """
    Centralized state manager with change notification.
    
    Usage:
        state = StateManager()
        
        # Access state
        if state.data.loaded:
            print(f"Documents: {state.data.n_documents}")
        
        # Update state
        state.update_data(loaded=True, n_documents=1234)
        
        # Listen for changes
        state.on_change("data", callback)
    """
    
    def __init__(self):
        self.data = DataState()
        self.analysis = AnalysisState()
        self.ui = UIState()
        self.settings = SettingsState()
        
        self._listeners: Dict[str, List[Callable]] = {
            "data": [],
            "analysis": [],
            "ui": [],
            "settings": [],
        }
        
        # Load persisted settings
        self._load_settings()
    
    def on_change(self, state_type: str, callback: Callable):
        """Register a callback for state changes."""
        if state_type in self._listeners:
            self._listeners[state_type].append(callback)
    
    def off_change(self, state_type: str, callback: Callable):
        """Remove a change callback."""
        if state_type in self._listeners:
            if callback in self._listeners[state_type]:
                self._listeners[state_type].remove(callback)
    
    def _notify(self, state_type: str):
        """Notify listeners of state change."""
        state_obj = getattr(self, state_type)
        for callback in self._listeners.get(state_type, []):
            try:
                callback(state_obj)
            except Exception as e:
                print(f"Error in state change callback: {e}")
    
    def update_data(self, **kwargs):
        """Update data state."""
        for key, value in kwargs.items():
            if hasattr(self.data, key):
                setattr(self.data, key, value)
        self._notify("data")
    
    def update_analysis(self, **kwargs):
        """Update analysis state."""
        for key, value in kwargs.items():
            if hasattr(self.analysis, key):
                setattr(self.analysis, key, value)
        self._notify("analysis")
    
    def update_ui(self, **kwargs):
        """Update UI state."""
        for key, value in kwargs.items():
            if hasattr(self.ui, key):
                setattr(self.ui, key, value)
        self._notify("ui")
    
    def update_settings(self, **kwargs):
        """Update settings and persist."""
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
        self._save_settings()
        self._notify("settings")
    
    def reset_data(self):
        """Reset data state for new dataset."""
        self.data = DataState()
        self.analysis = AnalysisState()
        self._notify("data")
        self._notify("analysis")
    
    def mark_analysis_complete(self, name: str, result: Any = None):
        """Mark an analysis as complete with optional result."""
        self.analysis.mark_complete(name)
        if result is not None:
            self.analysis.results[name] = result
        self.analysis.in_progress = None
        self._notify("analysis")
    
    def start_analysis(self, name: str):
        """Mark an analysis as in progress."""
        self.analysis.in_progress = name
        self._notify("analysis")
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def _get_settings_path(self) -> Path:
        """Get path to settings file."""
        config_dir = Path.home() / ".biblium"
        config_dir.mkdir(exist_ok=True)
        return config_dir / "settings.json"
    
    def _load_settings(self):
        """Load settings from disk."""
        path = self._get_settings_path()
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                for key, value in data.items():
                    if hasattr(self.settings, key):
                        setattr(self.settings, key, value)
            except Exception as e:
                print(f"Error loading settings: {e}")
    
    def _save_settings(self):
        """Save settings to disk."""
        path = self._get_settings_path()
        try:
            data = asdict(self.settings)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def save_ui_state(self):
        """Save UI state for session restore."""
        path = self._get_settings_path().parent / "ui_state.json"
        try:
            data = asdict(self.ui)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
    
    def load_ui_state(self):
        """Load UI state from previous session."""
        path = self._get_settings_path().parent / "ui_state.json"
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                for key, value in data.items():
                    if hasattr(self.ui, key):
                        setattr(self.ui, key, value)
            except Exception:
                pass
