# -*- coding: utf-8 -*-
"""
Toolbar Component
=================
Main application toolbar with Run/Stop controls.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus


class ToolbarButton(tk.Button):
    """A toolbar button with icon and tooltip."""
    
    def __init__(
        self,
        parent,
        icon: str,
        tooltip: str = "",
        command: Optional[Callable] = None,
        theme: str = "light",
        **kwargs
    ):
        self.theme = get_theme(theme)
        
        super().__init__(
            parent,
            text=icon,
            font=("Segoe UI", 14),
            width=2,
            bg=self.theme["bg_primary"],
            fg=self.theme["text_secondary"],
            activebackground=self.theme["bg_secondary"],
            activeforeground=self.theme["text_primary"],
            relief=tk.FLAT,
            cursor="hand2",
            command=command,
            bd=0,
            **kwargs
        )
        
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        
        if tooltip:
            from biblium.gui.widgets.tooltips import create_tooltip
            create_tooltip(self, tooltip)
    
    def _on_enter(self, event):
        self.configure(bg=self.theme["bg_secondary"])
    
    def _on_leave(self, event):
        self.configure(bg=self.theme["bg_primary"])


class Toolbar(tk.Frame):
    """
    Main application toolbar with Run controls and Undo/Redo.
    """
    
    def __init__(
        self,
        parent,
        theme: str = "light",
        on_open: Optional[Callable] = None,
        on_save: Optional[Callable] = None,
        on_export: Optional[Callable] = None,
        on_run: Optional[Callable] = None,
        **kwargs
    ):
        self.theme_name = theme
        self.theme = get_theme(theme)
        
        self.on_open = on_open
        self.on_save = on_save
        self.on_export = on_export
        self.on_run = on_run
        
        self._is_running = False
        
        super().__init__(
            parent,
            bg=self.theme["bg_primary"],
            height=LAYOUT.toolbar_height,
            **kwargs
        )
        self.pack_propagate(False)
        
        self._create_widgets()
        
        # Subscribe to events
        event_bus.subscribe(EventBus.DATASET_LOADED, self._on_dataset_loaded)
        event_bus.subscribe(EventBus.ANALYSIS_STARTED, self._on_analysis_started)
        event_bus.subscribe(EventBus.ANALYSIS_COMPLETED, self._on_analysis_finished)
        event_bus.subscribe(EventBus.ANALYSIS_CANCELLED, self._on_analysis_finished)
        event_bus.subscribe(EventBus.ERROR_OCCURRED, self._on_analysis_finished)
    
    def _create_widgets(self):
        """Create toolbar widgets."""
        # Left section - File operations
        left_frame = tk.Frame(self, bg=self.theme["bg_primary"])
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))
        
        ToolbarButton(
            left_frame, icon="📂", tooltip="Open Dataset (Ctrl+O)",
            command=self._handle_open, theme=self.theme_name,
        ).pack(side=tk.LEFT, padx=2, pady=6)
        
        ToolbarButton(
            left_frame, icon="💾", tooltip="Save Results (Ctrl+S)",
            command=self._handle_save, theme=self.theme_name,
        ).pack(side=tk.LEFT, padx=2, pady=6)
        
        ToolbarButton(
            left_frame, icon="📤", tooltip="Export (Ctrl+E)",
            command=self._handle_export, theme=self.theme_name,
        ).pack(side=tk.LEFT, padx=2, pady=6)
        
        # Separator
        self._add_separator()
        
        # Run section
        run_frame = tk.Frame(self, bg=self.theme["bg_primary"])
        run_frame.pack(side=tk.LEFT, fill=tk.Y, padx=4)
        
        # Run button (green play icon)
        self.run_btn = tk.Button(
            run_frame,
            text="▶",
            font=("Segoe UI", 14),
            width=2,
            bg="#4caf50",  # Green
            fg="white",
            activebackground="#388e3c",
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self._handle_run,
            bd=0,
        )
        self.run_btn.pack(side=tk.LEFT, padx=2, pady=6)
        self.run_btn.bind("<Enter>", lambda e: self.run_btn.configure(bg="#388e3c") if not self._is_running else None)
        self.run_btn.bind("<Leave>", lambda e: self.run_btn.configure(bg="#4caf50") if not self._is_running else None)
        
        from biblium.gui.widgets.tooltips import create_tooltip
        create_tooltip(self.run_btn, "Run current analysis (F5)")
        
        # Running indicator
        self.status_label = tk.Label(
            run_frame,
            text="",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_secondary"],
        )
        self.status_label.pack(side=tk.LEFT, padx=(8, 0), pady=6)
    
    def _add_separator(self):
        """Add a vertical separator."""
        sep_frame = tk.Frame(self, bg=self.theme["bg_primary"])
        sep_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Separator(sep_frame, orient=tk.VERTICAL).pack(fill=tk.Y, pady=8)
    
    def _handle_open(self):
        if self.on_open:
            self.on_open()
    
    def _handle_save(self):
        if self.on_save:
            self.on_save()
    
    def _handle_export(self):
        if self.on_export:
            self.on_export()
    
    def _handle_run(self):
        """Handle Run button click."""
        if self._is_running:
            return
        if self.on_run:
            self.on_run()
    
    def _on_dataset_loaded(self, data):
        """Handle dataset loaded event."""
        pass
    
    def _on_analysis_started(self, data):
        """Handle analysis started event."""
        self._set_running(data.get("name", "Analysis"))
    
    def _on_analysis_finished(self, data):
        """Handle analysis finished/cancelled/error event."""
        self._set_stopped()
    
    def _set_running(self, name: str = ""):
        """Set toolbar to running state."""
        self._is_running = True
        
        # Update Run button (disabled, grayed out)
        self.run_btn.configure(
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_muted"],
            state=tk.DISABLED,
        )
        
        # Update status label
        self.status_label.configure(text=f"⏳ Running: {name}..." if name else "⏳ Running...")
    
    def _set_stopped(self):
        """Set toolbar to stopped state."""
        self._is_running = False
        
        # Update Run button (enabled, green)
        self.run_btn.configure(
            bg="#4caf50",  # Green
            fg="white",
            state=tk.NORMAL,
        )
        
        # Clear status label
        self.status_label.configure(text="")
