# -*- coding: utf-8 -*-
"""
Tooltip Widget
==============
Tooltips for widgets.
"""

import tkinter as tk
from typing import Optional

from biblium.gui.config import FONTS, get_theme


class ToolTip:
    """
    Tooltip that appears when hovering over a widget.
    
    Usage:
        btn = tk.Button(parent, text="Click")
        ToolTip(btn, "Click this button to perform action")
    """
    
    def __init__(
        self,
        widget: tk.Widget,
        text: str,
        delay: int = 500,
        theme: str = "light",
    ):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.theme = get_theme(theme)
        
        self._tooltip_window: Optional[tk.Toplevel] = None
        self._after_id: Optional[str] = None
        
        # Bind events
        self.widget.bind("<Enter>", self._schedule_show)
        self.widget.bind("<Leave>", self._hide)
        self.widget.bind("<Button>", self._hide)
    
    def _schedule_show(self, event=None):
        """Schedule tooltip to show after delay."""
        self._cancel_scheduled()
        self._after_id = self.widget.after(self.delay, self._show)
    
    def _cancel_scheduled(self):
        """Cancel any scheduled tooltip."""
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
    
    def _show(self):
        """Show the tooltip."""
        if self._tooltip_window:
            return
        
        # Calculate position
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        
        # Create tooltip window
        self._tooltip_window = tk.Toplevel(self.widget)
        self._tooltip_window.wm_overrideredirect(True)
        self._tooltip_window.wm_geometry(f"+{x}+{y}")
        
        # Make sure tooltip appears on top
        self._tooltip_window.wm_attributes("-topmost", True)
        
        # Create tooltip content
        frame = tk.Frame(
            self._tooltip_window,
            bg="#334155",
            highlightbackground="#475569",
            highlightthickness=1,
        )
        frame.pack()
        
        label = tk.Label(
            frame,
            text=self.text,
            font=FONTS.get_font("small"),
            bg="#334155",
            fg="#ffffff",
            padx=8,
            pady=4,
            justify=tk.LEFT,
            wraplength=300,
        )
        label.pack()
    
    def _hide(self, event=None):
        """Hide the tooltip."""
        self._cancel_scheduled()
        if self._tooltip_window:
            self._tooltip_window.destroy()
            self._tooltip_window = None
    
    def update_text(self, text: str):
        """Update tooltip text."""
        self.text = text


def create_tooltip(widget: tk.Widget, text: str, **kwargs) -> ToolTip:
    """
    Convenience function to create a tooltip.
    
    Usage:
        create_tooltip(my_button, "This is a button")
    """
    return ToolTip(widget, text, **kwargs)
