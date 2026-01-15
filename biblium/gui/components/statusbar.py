# -*- coding: utf-8 -*-
"""
Status Bar Component
====================
Application status bar with progress indicator.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus


class StatusBar(tk.Frame):
    """
    Application status bar.
    
    Features:
    - Status message
    - Document count
    - Database badge
    - Progress indicator
    """
    
    def __init__(self, parent, theme: str = "light", **kwargs):
        self.theme_name = theme
        self.theme = get_theme(theme)
        
        super().__init__(
            parent,
            bg=self.theme["bg_card"],
            height=LAYOUT.statusbar_height,
            **kwargs
        )
        self.pack_propagate(False)
        
        self._create_widgets()
        self._subscribe_events()
    
    def _create_widgets(self):
        """Create status bar widgets."""
        # Left section - Status message
        left_frame = tk.Frame(self, bg=self.theme["bg_card"])
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.status_icon = tk.Label(
            left_frame,
            text="✓",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["success"],
            padx=8,
        )
        self.status_icon.pack(side=tk.LEFT)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(
            left_frame,
            textvariable=self.status_var,
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
        )
        self.status_label.pack(side=tk.LEFT)
        
        # Separator
        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12, pady=4)
        
        # Document count
        self.doc_var = tk.StringVar(value="")
        self.doc_label = tk.Label(
            self,
            textvariable=self.doc_var,
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["accent_primary"],
        )
        self.doc_label.pack(side=tk.LEFT)
        
        # Database badge
        self.db_var = tk.StringVar(value="")
        self.db_label = tk.Label(
            self,
            textvariable=self.db_var,
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
            padx=8,
        )
        self.db_label.pack(side=tk.LEFT)
        
        # Right section - Progress
        right_frame = tk.Frame(self, bg=self.theme["bg_card"])
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=8)
        
        self.progress_var = tk.StringVar(value="")
        self.progress_label = tk.Label(
            right_frame,
            textvariable=self.progress_var,
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
        )
        self.progress_label.pack(side=tk.LEFT, padx=(0, 8))
        
        # Progress bar
        self.progress = ttk.Progressbar(
            right_frame,
            orient=tk.HORIZONTAL,
            length=120,
            mode="determinate",
        )
        # Hidden by default
        self.progress_visible = False
    
    def _subscribe_events(self):
        """Subscribe to events."""
        event_bus.subscribe(EventBus.STATUS_MESSAGE, self._on_status_message)
        event_bus.subscribe(EventBus.STATUS_PROGRESS, self._on_status_progress)
        event_bus.subscribe(EventBus.STATUS_CLEAR, self._on_status_clear)
        event_bus.subscribe(EventBus.DATASET_LOADED, self._on_dataset_loaded)
        event_bus.subscribe(EventBus.ANALYSIS_STARTED, self._on_analysis_started)
        event_bus.subscribe(EventBus.ANALYSIS_COMPLETED, self._on_analysis_completed)
        event_bus.subscribe(EventBus.ERROR_OCCURRED, self._on_error)
    
    def set_status(self, message: str, icon: str = "✓", color: str = None):
        """Set status message."""
        self.status_var.set(message)
        self.status_icon.configure(text=icon)
        
        if color:
            self.status_icon.configure(fg=color)
        else:
            self.status_icon.configure(fg=self.theme["success"])
    
    def set_document_info(self, n_docs: int, database: str = "", year_range: tuple = None):
        """Set document information."""
        self.doc_var.set(f"📄 {n_docs:,} documents")
        
        if database:
            db_display = {
                "scopus": "🔵 Scopus",
                "wos": "🟠 Web of Science",
                "oa": "🟢 OpenAlex",
                "pubmed": "🔴 PubMed",
            }.get(database.lower(), database)
            self.db_var.set(db_display)
        
        if year_range and year_range[0] and year_range[1]:
            current = self.db_var.get()
            self.db_var.set(f"{current}  │  📅 {year_range[0]}-{year_range[1]}")
    
    def set_progress(self, value: float, message: str = ""):
        """Set progress (0.0 to 1.0)."""
        if not self.progress_visible:
            self.progress.pack(side=tk.LEFT)
            self.progress_visible = True
        
        self.progress["value"] = value * 100
        self.progress_var.set(message or f"{int(value * 100)}%")
    
    def set_indeterminate(self, active: bool, message: str = ""):
        """Set indeterminate progress mode."""
        if active:
            if not self.progress_visible:
                self.progress.pack(side=tk.LEFT)
                self.progress_visible = True
            self.progress.configure(mode="indeterminate")
            self.progress.start(10)
            self.progress_var.set(message or "Processing...")
        else:
            self.progress.stop()
            self.progress.configure(mode="determinate")
    
    def clear_progress(self):
        """Hide progress indicator."""
        if self.progress_visible:
            self.progress.stop()
            self.progress.pack_forget()
            self.progress_visible = False
        self.progress_var.set("")
    
    def _on_status_message(self, data):
        """Handle status message event."""
        if isinstance(data, dict):
            self.set_status(
                data.get("message", ""),
                data.get("icon", "✓"),
                data.get("color"),
            )
        else:
            self.set_status(str(data))
    
    def _on_status_progress(self, data):
        """Handle progress event."""
        if isinstance(data, dict):
            self.set_progress(
                data.get("value", 0),
                data.get("message", ""),
            )
        else:
            self.set_progress(float(data))
    
    def _on_status_clear(self, data):
        """Handle clear event."""
        self.clear_progress()
        self.set_status("Ready")
    
    def _on_dataset_loaded(self, data):
        """Handle dataset loaded event."""
        if isinstance(data, dict):
            self.set_document_info(
                data.get("n_documents", 0),
                data.get("database", ""),
                data.get("year_range"),
            )
        self.set_status("Dataset loaded", "✓", self.theme["success"])
        self.clear_progress()
    
    def _on_analysis_started(self, data):
        """Handle analysis started event."""
        name = data.get("name", "Analysis") if isinstance(data, dict) else "Analysis"
        self.set_status(f"Running {name}...", "⏳", self.theme["warning"])
        self.set_indeterminate(True, f"Running {name}...")
    
    def _on_analysis_completed(self, data):
        """Handle analysis completed event."""
        name = data.get("name", "Analysis") if isinstance(data, dict) else "Analysis"
        self.set_status(f"{name} completed", "✓", self.theme["success"])
        self.clear_progress()
    
    def _on_error(self, data):
        """Handle error event."""
        message = data.get("message", "Error") if isinstance(data, dict) else str(data)
        self.set_status(message, "✗", self.theme["danger"])
        self.clear_progress()
