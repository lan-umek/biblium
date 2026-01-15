# -*- coding: utf-8 -*-
"""
Base Widget Classes
===================
Foundation classes for themed, consistent widgets.
"""

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict, Optional, Tuple

from biblium.gui.config import LAYOUT, FONTS, get_theme


class BaseFrame(tk.Frame):
    """
    Base frame with theme support.
    
    All custom frames should inherit from this to get consistent theming.
    """
    
    def __init__(self, parent, theme: str = "light", **kwargs):
        self.theme_name = theme
        self.theme = get_theme(theme)
        
        # Apply default background if not specified
        if "bg" not in kwargs and "background" not in kwargs:
            kwargs["bg"] = self.theme["bg_primary"]
        
        super().__init__(parent, **kwargs)
        
        self._setup_bindings()
    
    def _setup_bindings(self):
        """Setup common bindings."""
        pass
    
    def update_theme(self, theme_name: str):
        """Update the theme."""
        self.theme_name = theme_name
        self.theme = get_theme(theme_name)
        self.configure(bg=self.theme["bg_primary"])
        self._apply_theme_to_children()
    
    def _apply_theme_to_children(self):
        """Recursively apply theme to children."""
        for child in self.winfo_children():
            if hasattr(child, "update_theme"):
                child.update_theme(self.theme_name)
            elif isinstance(child, tk.Frame):
                child.configure(bg=self.theme["bg_primary"])
            elif isinstance(child, tk.Label):
                child.configure(bg=self.theme["bg_primary"], fg=self.theme["text_primary"])


class BaseCanvas(tk.Canvas):
    """Base canvas with theme support and common utilities."""
    
    def __init__(self, parent, theme: str = "light", **kwargs):
        self.theme_name = theme
        self.theme = get_theme(theme)
        
        if "bg" not in kwargs and "background" not in kwargs:
            kwargs["bg"] = self.theme["bg_primary"]
        if "highlightthickness" not in kwargs:
            kwargs["highlightthickness"] = 0
        
        super().__init__(parent, **kwargs)
    
    def update_theme(self, theme_name: str):
        """Update the theme."""
        self.theme_name = theme_name
        self.theme = get_theme(theme_name)
        self.configure(bg=self.theme["bg_primary"])


class ScrollableFrame(BaseFrame):
    """
    A frame with scrolling support.
    
    Usage:
        sf = ScrollableFrame(parent)
        
        # Add widgets to sf.interior
        tk.Label(sf.interior, text="Hello").pack()
        
        # Scroll programmatically
        sf.scroll_to_top()
    """
    
    def __init__(
        self,
        parent,
        theme: str = "light",
        vertical: bool = True,
        horizontal: bool = False,
        **kwargs
    ):
        super().__init__(parent, theme=theme, **kwargs)
        
        self.vertical = vertical
        self.horizontal = horizontal
        
        self._create_widgets()
        self._bind_scroll()
    
    def _create_widgets(self):
        """Create the scrollable structure."""
        # Canvas
        self.canvas = tk.Canvas(
            self,
            bg=self.theme["bg_primary"],
            highlightthickness=0,
        )
        
        # Scrollbars
        if self.vertical:
            self.v_scrollbar = ttk.Scrollbar(
                self,
                orient=tk.VERTICAL,
                command=self.canvas.yview
            )
            self.canvas.configure(yscrollcommand=self.v_scrollbar.set)
        
        if self.horizontal:
            self.h_scrollbar = ttk.Scrollbar(
                self,
                orient=tk.HORIZONTAL,
                command=self.canvas.xview
            )
            self.canvas.configure(xscrollcommand=self.h_scrollbar.set)
        
        # Interior frame
        self.interior = tk.Frame(self.canvas, bg=self.theme["bg_primary"])
        self.interior_id = self.canvas.create_window(
            (0, 0),
            window=self.interior,
            anchor=tk.NW
        )
        
        # Layout
        if self.horizontal:
            self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        if self.vertical:
            self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure canvas scrollregion when interior changes
        self.interior.bind("<Configure>", self._on_interior_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
    
    def _on_interior_configure(self, event):
        """Update scroll region when interior size changes."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _on_canvas_configure(self, event):
        """Adjust interior width to match canvas."""
        if not self.horizontal:
            self.canvas.itemconfig(self.interior_id, width=event.width)
    
    def _bind_scroll(self):
        """Bind mouse wheel scrolling."""
        def _on_mousewheel(event):
            if self.vertical:
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        def _on_shift_mousewheel(event):
            if self.horizontal:
                self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        
        # Bind when mouse enters/leaves
        self.canvas.bind("<Enter>", lambda e: self._bind_wheel())
        self.canvas.bind("<Leave>", lambda e: self._unbind_wheel())
    
    def _bind_wheel(self):
        """Bind mousewheel events."""
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Shift-MouseWheel>", self._on_shift_wheel)
    
    def _unbind_wheel(self):
        """Unbind mousewheel events."""
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Shift-MouseWheel>")
    
    def _on_wheel(self, event):
        """Handle vertical scroll."""
        if self.vertical:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _on_shift_wheel(self, event):
        """Handle horizontal scroll."""
        if self.horizontal:
            self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def scroll_to_top(self):
        """Scroll to top."""
        self.canvas.yview_moveto(0)
    
    def scroll_to_bottom(self):
        """Scroll to bottom."""
        self.canvas.yview_moveto(1)
    
    def update_theme(self, theme_name: str):
        """Update theme."""
        super().update_theme(theme_name)
        self.canvas.configure(bg=self.theme["bg_primary"])
        self.interior.configure(bg=self.theme["bg_primary"])


class ThemedMixin:
    """
    Mixin for adding theme support to any widget.
    
    Usage:
        class MyWidget(ThemedMixin, tk.Frame):
            def __init__(self, parent, **kwargs):
                ThemedMixin.__init__(self, "light")
                tk.Frame.__init__(self, parent, **kwargs)
    """
    
    def __init__(self, theme: str = "light"):
        self.theme_name = theme
        self.theme = get_theme(theme)
    
    def get_color(self, key: str) -> str:
        """Get a theme color."""
        return self.theme.get(key, "#000000")
    
    def get_font(self, style: str = "body", bold: bool = False) -> Tuple:
        """Get a font tuple."""
        return FONTS.get_font(style, bold)


class AutoScrollbar(ttk.Scrollbar):
    """
    Scrollbar that automatically hides when not needed.
    """
    
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.pack_forget()
        else:
            if self.cget("orient") == tk.VERTICAL:
                self.pack(side=tk.RIGHT, fill=tk.Y)
            else:
                self.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Scrollbar.set(self, lo, hi)
