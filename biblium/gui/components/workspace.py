# -*- coding: utf-8 -*-
"""
Workspace Component
===================
Main content workspace with tabbed panels.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Optional, Type, Any

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus


class Workspace(tk.Frame):
    """
    Main workspace area with tabbed interface.
    
    Features:
    - Tabbed panel interface
    - Closable tabs
    - Panel registry
    - Lazy panel loading
    - Shared bib object across all panels
    """
    
    def __init__(
        self,
        parent,
        theme: str = "light",
        **kwargs
    ):
        self.theme_name = theme
        self.theme = get_theme(theme)
        
        super().__init__(parent, bg=self.theme["bg_secondary"], **kwargs)
        
        # Panel registry: panel_id -> panel_class
        self._panel_registry: Dict[str, Type] = {}
        
        # Active panels: tab_id -> panel_instance
        self._active_panels: Dict[str, tk.Frame] = {}
        
        # Tab mapping: tab_id -> panel_id
        self._tab_mapping: Dict[str, str] = {}
        
        # Shared bib object - THE KEY FIX
        self._shared_bib = None
        
        # Shared bib_group object for group analysis
        self._shared_bib_group = None
        
        # Shared state manager
        self._state = None
        
        self._create_widgets()
        self._create_welcome_tab()
        
        # Subscribe to events
        event_bus.subscribe(EventBus.PANEL_CHANGED, self._on_panel_changed)
        event_bus.subscribe(EventBus.DATASET_LOADED, self._on_dataset_loaded)
        event_bus.subscribe(EventBus.RESET_PANELS, self._on_reset_panels)
        event_bus.subscribe("group_created", self._on_group_created)
    
    def _create_widgets(self):
        """Create workspace widgets."""
        # Notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Configure notebook style
        style = ttk.Style()
        style.configure(
            "TNotebook",
            background=self.theme["bg_secondary"],
            borderwidth=0,
        )
        style.configure(
            "TNotebook.Tab",
            padding=[16, 8],
            font=FONTS.get_font("body"),
            background=self.theme["bg_secondary"],
            foreground=self.theme["text_muted"],
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.theme["bg_card"])],
            foreground=[("selected", self.theme["accent_primary"])],
        )
        
        # Bind tab change
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
    
    def _create_welcome_tab(self):
        """Create the welcome/home tab."""
        welcome = tk.Frame(self.notebook, bg=self.theme["bg_secondary"])
        self.notebook.add(welcome, text="  🏠 Home  ")
        
        # Center content
        center = tk.Frame(welcome, bg=self.theme["bg_secondary"])
        center.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
        
        tk.Label(
            center,
            text="📚",
            font=("Segoe UI", 64),
            bg=self.theme["bg_secondary"],
        ).pack()
        
        tk.Label(
            center,
            text="Biblium",
            font=("Segoe UI", 36, "bold"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_primary"],
        ).pack(pady=(12, 4))
        
        tk.Label(
            center,
            text="Bibliometric Analysis Platform",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_muted"],
        ).pack()
        
        tk.Label(
            center,
            text="Load a dataset to get started",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_muted"],
        ).pack(pady=(24, 0))
        
        # Quick start button
        from biblium.gui.widgets.buttons import ThemedButton
        
        btn_frame = tk.Frame(center, bg=self.theme["bg_secondary"])
        btn_frame.pack(pady=24)
        
        ThemedButton(
            btn_frame,
            text="Load Dataset",
            icon="📂",
            style="primary",
            size="large",
            theme=self.theme_name,
            command=lambda: event_bus.emit(EventBus.PANEL_CHANGED, {"panel": "load"}),
        ).pack()
        
        self._active_panels["home"] = welcome
        self._tab_mapping["home"] = "home"
    
    @property
    def bib(self):
        """Get the shared bib object."""
        return self._shared_bib
    
    @bib.setter
    def bib(self, value):
        """Set the shared bib object and update all panels."""
        self._shared_bib = value
        self._update_all_panels_bib()
    
    def _update_all_panels_bib(self):
        """Update bib reference in all active panels."""
        for tab_id, panel in self._active_panels.items():
            if tab_id != "home" and hasattr(panel, 'bib'):
                panel.bib = self._shared_bib
    
    @property
    def bib_group(self):
        """Get the shared bib_group object."""
        return self._shared_bib_group
    
    @bib_group.setter
    def bib_group(self, value):
        """Set the shared bib_group object and update all panels."""
        self._shared_bib_group = value
        self._update_all_panels_bib_group()
    
    def _update_all_panels_bib_group(self):
        """Update bib_group reference in all active panels."""
        for tab_id, panel in self._active_panels.items():
            if tab_id != "home":
                # Set bib_group if panel has it
                if hasattr(panel, 'bib_group'):
                    panel.bib_group = self._shared_bib_group
                # Also try _bib_group for panels that use internal storage
                if hasattr(panel, '_bib_group'):
                    panel._bib_group = self._shared_bib_group
    
    def _on_group_created(self, data):
        """Handle group created event - update shared bib_group."""
        if isinstance(data, dict):
            bib_group = data.get("bib_group")
            if bib_group:
                self._shared_bib_group = bib_group
                self._update_all_panels_bib_group()
    
    def _on_dataset_loaded(self, data):
        """Handle dataset loaded event - update shared bib."""
        if isinstance(data, dict):
            bib = data.get("bib")
            if bib:
                self._shared_bib = bib
                self._update_all_panels_bib()
    
    def _on_reset_panels(self, data=None):
        """Handle reset panels event - close all tabs except Home and Load Dataset."""
        # Close all matplotlib figures first to prevent thread issues
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass
        
        # Get list of tabs to close (avoid modifying dict during iteration)
        tabs_to_close = []
        for tab_id, panel_id in self._tab_mapping.items():
            if panel_id not in ("home", "load"):
                tabs_to_close.append(tab_id)
        
        # Close each tab safely
        for tab_id in tabs_to_close:
            try:
                panel = self._active_panels.get(tab_id)
                if panel:
                    # Stop any active spinners or operations
                    if hasattr(panel, '_stop_active_spinners'):
                        panel._stop_active_spinners()
                    # Remove from notebook first
                    try:
                        self.notebook.forget(panel)
                    except:
                        pass
                    # Allow UI to update
                    self.update_idletasks()
                    # Destroy panel
                    try:
                        panel.destroy()
                    except:
                        pass
                # Clean up tracking
                if tab_id in self._active_panels:
                    del self._active_panels[tab_id]
                if tab_id in self._tab_mapping:
                    del self._tab_mapping[tab_id]
            except Exception as e:
                print(f"Error closing tab {tab_id}: {e}")
        
        # Clear shared bib
        self._shared_bib = None
    
    def register_panel(self, panel_id: str, panel_class: Type):
        """Register a panel class."""
        self._panel_registry[panel_id] = panel_class
    
    def open_panel(self, panel_id: str, **kwargs) -> Optional[tk.Frame]:
        """
        Open a panel (create tab if not exists).
        
        Parameters
        ----------
        panel_id : str
            Panel identifier.
        **kwargs
            Arguments to pass to panel constructor.
        
        Returns
        -------
        tk.Frame or None
            The panel instance.
        """
        # Check if already open
        for tab_id, pid in list(self._tab_mapping.items()):
            if pid == panel_id:
                # Select existing tab
                panel = self._active_panels.get(tab_id)
                if panel:
                    # Update bib reference in case it changed
                    if hasattr(panel, 'bib') and self._shared_bib:
                        panel.bib = self._shared_bib
                    # Update bib_group reference in case it changed
                    if self._shared_bib_group:
                        if hasattr(panel, '_bib_group'):
                            panel._bib_group = self._shared_bib_group
                        if hasattr(panel, 'bib_group'):
                            try:
                                panel.bib_group = self._shared_bib_group
                            except AttributeError:
                                pass  # Property might be read-only
                        # Refresh the panel if it has a refresh method
                        if hasattr(panel, 'refresh'):
                            try:
                                panel.refresh()
                            except Exception as e:
                                print(f"Error refreshing panel: {e}")
                    self.notebook.select(panel)
                    return panel
        
        # Check if panel is currently being created (prevent race condition)
        if not hasattr(self, '_creating_panels'):
            self._creating_panels = set()
        
        if panel_id in self._creating_panels:
            return None
        
        self._creating_panels.add(panel_id)
        
        try:
            # Get panel class
            panel_class = self._panel_registry.get(panel_id)
            if not panel_class:
                print(f"Warning: Panel '{panel_id}' not registered")
                return None
            
            # Always pass the shared bib to new panels
            if 'bib' not in kwargs and self._shared_bib:
                kwargs['bib'] = self._shared_bib
            
            # Always pass the shared bib_group to new panels
            if 'bib_group' not in kwargs and self._shared_bib_group:
                kwargs['bib_group'] = self._shared_bib_group
            
            # Pass state for settings panel
            if panel_id == "settings" and hasattr(self, '_state') and self._state:
                kwargs['state'] = self._state
            
            # Create panel
            try:
                panel = panel_class(self.notebook, theme=self.theme_name, **kwargs)
                
                # Also set bib_group after creation (in case panel creates it internally)
                if self._shared_bib_group and hasattr(panel, '_bib_group'):
                    panel._bib_group = self._shared_bib_group
                    
            except Exception as e:
                print(f"Error creating panel '{panel_id}': {e}")
                import traceback
                traceback.print_exc()
                return None
            
            # Get panel title
            title = getattr(panel, "title", panel_id.replace("_", " ").title())
            icon = getattr(panel, "icon", "")
            tab_text = f"  {icon} {title}  " if icon else f"  {title}  "
            
            # Add tab
            self.notebook.add(panel, text=tab_text)
            self.notebook.select(panel)
            
            # Track
            tab_id = str(id(panel))
            self._active_panels[tab_id] = panel
            self._tab_mapping[tab_id] = panel_id
            
            # Add close button behavior
            self._setup_tab_close(panel)
            
            return panel
        finally:
            # Always remove from creating set
            self._creating_panels.discard(panel_id)
    
    def _setup_tab_close(self, panel):
        """Setup middle-click and right-click to close tab."""
        # Middle click to close
        def close_tab_middle(event):
            try:
                tab_id = self.notebook.index("@%d,%d" % (event.x, event.y))
                if tab_id > 0:  # Don't close home tab
                    self.close_tab_by_index(tab_id)
            except tk.TclError:
                pass
        
        self.notebook.bind("<Button-2>", close_tab_middle)  # Middle click
        
        # Right-click context menu
        def show_tab_menu(event):
            try:
                tab_id = self.notebook.index("@%d,%d" % (event.x, event.y))
                if tab_id > 0:  # Don't show menu for home tab
                    self._show_tab_context_menu(event, tab_id)
            except tk.TclError:
                pass
        
        self.notebook.bind("<Button-3>", show_tab_menu)  # Right click
    
    def _show_tab_context_menu(self, event, tab_index: int):
        """Show context menu for tab."""
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Close Tab", command=lambda: self.close_tab_by_index(tab_index))
        menu.add_command(label="Close Other Tabs", command=lambda: self._close_other_tabs(tab_index))
        menu.add_command(label="Close All Tabs", command=self._close_all_tabs)
        menu.add_separator()
        menu.add_command(label="Refresh", command=lambda: self._refresh_tab(tab_index))
        
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
    
    def _close_other_tabs(self, keep_index: int):
        """Close all tabs except the specified one and home."""
        tabs_to_close = []
        for i in range(self.notebook.index("end") - 1, 0, -1):  # Reverse order, skip home (0)
            if i != keep_index:
                tabs_to_close.append(i)
        
        for idx in tabs_to_close:
            self.close_tab_by_index(idx)
    
    def _close_all_tabs(self):
        """Close all tabs except home."""
        for i in range(self.notebook.index("end") - 1, 0, -1):  # Reverse order, skip home (0)
            self.close_tab_by_index(i)
    
    def _refresh_tab(self, tab_index: int):
        """Refresh the specified tab."""
        try:
            panel = self.notebook.nametowidget(self.notebook.tabs()[tab_index])
            if hasattr(panel, 'refresh'):
                panel.refresh()
            elif hasattr(panel, '_on_dataset_loaded') and self._shared_bib:
                panel._on_dataset_loaded({"bib": self._shared_bib})
        except (tk.TclError, IndexError):
            pass
    
    def close_panel(self, panel_id: str):
        """Close a panel by ID."""
        for tab_id, pid in list(self._tab_mapping.items()):
            if pid == panel_id:
                panel = self._active_panels.get(tab_id)
                if panel:
                    self.notebook.forget(panel)
                    panel.destroy()
                    del self._active_panels[tab_id]
                    del self._tab_mapping[tab_id]
                break
    
    def close_tab_by_index(self, index: int):
        """Close tab by index."""
        if index <= 0:  # Don't close home tab
            return
        
        try:
            panel = self.notebook.nametowidget(self.notebook.tabs()[index])
            tab_id = str(id(panel))
            
            self.notebook.forget(index)
            panel.destroy()
            
            if tab_id in self._active_panels:
                del self._active_panels[tab_id]
            if tab_id in self._tab_mapping:
                del self._tab_mapping[tab_id]
        except (tk.TclError, IndexError):
            pass
    
    def get_active_panel(self) -> Optional[tk.Frame]:
        """Get currently active panel."""
        try:
            selected = self.notebook.select()
            current = self.notebook.nametowidget(selected)
            return current
        except tk.TclError:
            return None
    
    def _on_panel_changed(self, data):
        """Handle panel change event."""
        if isinstance(data, dict):
            panel_id = data.get("panel")
            kwargs = data.get("kwargs", {})
        else:
            panel_id = data
            kwargs = {}
        
        if panel_id:
            self.open_panel(panel_id, **kwargs)
    
    def _on_tab_changed(self, event):
        """Handle tab change event."""
        try:
            current = self.notebook.select()
            panel = self.notebook.nametowidget(current)
            tab_id = str(id(panel))
            panel_id = self._tab_mapping.get(tab_id, "home")
            
            event_bus.emit(EventBus.TAB_CHANGED, {"panel_id": panel_id, "tab_id": tab_id})
        except tk.TclError:
            pass
