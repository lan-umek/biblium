# -*- coding: utf-8 -*-
"""
Card Widgets
============
Container widgets with card styling.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme


class Card(tk.Frame):
    """
    A card container with optional title and border.
    
    Usage:
        card = Card(parent, title="Settings")
        tk.Label(card.content, text="Option 1").pack()
    """
    
    def __init__(
        self,
        parent,
        title: str = "",
        theme: str = "light",
        padding: int = None,
        **kwargs
    ):
        self.theme_name = theme
        self.theme = get_theme(theme)
        self.title_text = title
        
        padding = padding or LAYOUT.card_padding
        
        super().__init__(
            parent,
            bg=self.theme["bg_card"],
            highlightbackground=self.theme["border"],
            highlightthickness=1,
            **kwargs
        )
        
        # Title
        if title:
            self._create_header(title)
        
        # Content area
        self.content = tk.Frame(
            self,
            bg=self.theme["bg_card"],
            padx=padding,
            pady=padding,
        )
        self.content.pack(fill=tk.BOTH, expand=True)
    
    def _create_header(self, title: str):
        """Create the card header."""
        header = tk.Frame(self, bg=self.theme["bg_card"])
        header.pack(fill=tk.X)
        
        self.title_label = tk.Label(
            header,
            text=title,
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            padx=LAYOUT.card_padding,
            pady=LAYOUT.spacing_sm,
            anchor=tk.W,
        )
        self.title_label.pack(fill=tk.X)
        
        # Separator
        sep = tk.Frame(self, bg=self.theme["border"], height=1)
        sep.pack(fill=tk.X, padx=LAYOUT.card_padding)
    
    def set_title(self, title: str):
        """Update the card title."""
        if hasattr(self, "title_label"):
            self.title_label.configure(text=title)
    
    def update_theme(self, theme_name: str):
        """Update the card theme."""
        self.theme_name = theme_name
        self.theme = get_theme(theme_name)
        
        self.configure(
            bg=self.theme["bg_card"],
            highlightbackground=self.theme["border"],
        )
        self.content.configure(bg=self.theme["bg_card"])
        
        if hasattr(self, "title_label"):
            self.title_label.configure(
                bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
            )


class CollapsibleCard(tk.Frame):
    """
    A card that can be collapsed/expanded.
    
    Usage:
        card = CollapsibleCard(parent, title="Advanced Options", collapsed=True)
        tk.Label(card.content, text="Option 1").pack()
    """
    
    def __init__(
        self,
        parent,
        title: str,
        collapsed: bool = False,
        theme: str = "light",
        on_toggle: Optional[Callable] = None,
        **kwargs
    ):
        self.theme_name = theme
        self.theme = get_theme(theme)
        self.title_text = title
        self.collapsed = collapsed
        self.on_toggle = on_toggle
        
        super().__init__(
            parent,
            bg=self.theme["bg_card"],
            highlightbackground=self.theme["border"],
            highlightthickness=1,
            **kwargs
        )
        
        self._create_header()
        self._create_content()
        
        # Initial state
        if collapsed:
            self.content_frame.pack_forget()
    
    def _create_header(self):
        """Create the collapsible header."""
        self.header = tk.Frame(self, bg=self.theme["bg_card"], cursor="hand2")
        self.header.pack(fill=tk.X)
        
        # Toggle indicator
        self.toggle_label = tk.Label(
            self.header,
            text="▸" if self.collapsed else "▾",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["accent_primary"],
            padx=LAYOUT.spacing_sm,
        )
        self.toggle_label.pack(side=tk.LEFT)
        
        # Title
        self.title_label = tk.Label(
            self.header,
            text=self.title_text,
            font=FONTS.get_font("body", bold=True),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            pady=LAYOUT.spacing_sm,
            anchor=tk.W,
        )
        self.title_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Bind click events
        for widget in [self.header, self.toggle_label, self.title_label]:
            widget.bind("<Button-1>", self._toggle)
            widget.bind("<Enter>", self._on_enter)
            widget.bind("<Leave>", self._on_leave)
    
    def _create_content(self):
        """Create the collapsible content area."""
        # Separator
        self.separator = tk.Frame(self, bg=self.theme["border"], height=1)
        self.separator.pack(fill=tk.X, padx=LAYOUT.card_padding)
        
        # Content frame
        self.content_frame = tk.Frame(self, bg=self.theme["bg_card"])
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Actual content area with padding
        self.content = tk.Frame(
            self.content_frame,
            bg=self.theme["bg_card"],
            padx=LAYOUT.card_padding,
            pady=LAYOUT.card_padding,
        )
        self.content.pack(fill=tk.BOTH, expand=True)
        
        if self.collapsed:
            self.separator.pack_forget()
    
    def _toggle(self, event=None):
        """Toggle collapsed state."""
        self.collapsed = not self.collapsed
        
        if self.collapsed:
            self.content_frame.pack_forget()
            self.separator.pack_forget()
            self.toggle_label.configure(text="▸")
        else:
            self.separator.pack(fill=tk.X, padx=LAYOUT.card_padding)
            self.content_frame.pack(fill=tk.BOTH, expand=True)
            self.toggle_label.configure(text="▾")
        
        if self.on_toggle:
            self.on_toggle(self.collapsed)
    
    def _on_enter(self, event):
        """Mouse enter handler."""
        self.header.configure(bg=self.theme["bg_secondary"])
        self.toggle_label.configure(bg=self.theme["bg_secondary"])
        self.title_label.configure(bg=self.theme["bg_secondary"])
    
    def _on_leave(self, event):
        """Mouse leave handler."""
        self.header.configure(bg=self.theme["bg_card"])
        self.toggle_label.configure(bg=self.theme["bg_card"])
        self.title_label.configure(bg=self.theme["bg_card"])
    
    def expand(self):
        """Expand the card."""
        if self.collapsed:
            self._toggle()
    
    def collapse(self):
        """Collapse the card."""
        if not self.collapsed:
            self._toggle()
    
    def update_theme(self, theme_name: str):
        """Update the card theme."""
        self.theme_name = theme_name
        self.theme = get_theme(theme_name)
        
        self.configure(
            bg=self.theme["bg_card"],
            highlightbackground=self.theme["border"],
        )
        self.header.configure(bg=self.theme["bg_card"])
        self.toggle_label.configure(
            bg=self.theme["bg_card"],
            fg=self.theme["accent_primary"],
        )
        self.title_label.configure(
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        )
        self.separator.configure(bg=self.theme["border"])
        self.content_frame.configure(bg=self.theme["bg_card"])
        self.content.configure(bg=self.theme["bg_card"])


class StatsCard(tk.Frame):
    """
    A compact card displaying a single statistic with label.
    
    Usage:
        card = StatsCard(parent, label="Documents", value="1,234", icon="📄")
    """
    
    def __init__(
        self,
        parent,
        label: str,
        value: str,
        icon: str = "",
        theme: str = "light",
        accent: bool = False,
        **kwargs
    ):
        self.theme_name = theme
        self.theme = get_theme(theme)
        self.accent = accent
        
        bg = self.theme["accent_light"] if accent else self.theme["bg_card"]
        
        super().__init__(
            parent,
            bg=bg,
            highlightbackground=self.theme["border"],
            highlightthickness=1,
            padx=8,
            pady=6,
            **kwargs
        )
        
        # Icon and value row
        top_frame = tk.Frame(self, bg=bg)
        top_frame.pack(fill=tk.X)
        
        if icon:
            tk.Label(
                top_frame,
                text=icon,
                font=("Segoe UI", 14),
                bg=bg,
                fg=self.theme["accent_primary"] if accent else self.theme["text_muted"],
            ).pack(side=tk.LEFT, padx=(0, 4))
        
        self.value_label = tk.Label(
            top_frame,
            text=value,
            font=FONTS.get_font("heading", bold=True),
            bg=bg,
            fg=self.theme["accent_primary"] if accent else self.theme["text_primary"],
        )
        self.value_label.pack(side=tk.LEFT)
        
        # Label
        self.label_label = tk.Label(
            self,
            text=label,
            font=FONTS.get_font("small"),
            bg=bg,
            fg=self.theme["text_secondary"],
            anchor=tk.W,
        )
        self.label_label.pack(fill=tk.X)
    
    def set_value(self, value: str):
        """Update the displayed value."""
        self.value_label.configure(text=value)
    
    def set_label(self, label: str):
        """Update the label."""
        self.label_label.configure(text=label)


class CardGrid(tk.Frame):
    """
    A grid layout for cards that wraps to multiple rows.
    For horizontal scrolling stats, use ScrollableCardRow instead.
    """
    
    def __init__(
        self,
        parent,
        columns: int = 4,
        theme: str = "light",
        spacing: int = None,
        **kwargs
    ):
        self.theme = get_theme(theme)
        self.theme_name = theme
        self.columns = columns
        self.spacing = spacing or LAYOUT.spacing_md
        
        super().__init__(parent, bg=self.theme["bg_primary"], **kwargs)
        
        self._cards = []
        self._current_row = 0
        self._current_col = 0
        
        # Configure grid columns with equal weight
        for i in range(columns):
            self.columnconfigure(i, weight=1, uniform="card")
    
    def add_card(self, card: tk.Widget):
        """Add a card to the grid."""
        card.grid(
            row=self._current_row,
            column=self._current_col,
            padx=self.spacing // 2,
            pady=self.spacing // 2,
            sticky="nsew",
        )
        
        self._cards.append(card)
        
        self._current_col += 1
        if self._current_col >= self.columns:
            self._current_col = 0
            self._current_row += 1
            self.rowconfigure(self._current_row, weight=0)
    
    def clear(self):
        """Remove all cards."""
        for card in self._cards:
            card.destroy()
        self._cards.clear()
        self._current_row = 0
        self._current_col = 0


class ScrollableStatsRow(tk.Frame):
    """
    A horizontally scrollable row of stats cards.
    Shows a scrollbar when content exceeds available width.
    """
    
    def __init__(
        self,
        parent,
        theme: str = "light",
        spacing: int = 8,
        **kwargs
    ):
        self.theme = get_theme(theme)
        self.theme_name = theme
        self.spacing = spacing
        
        super().__init__(parent, bg=self.theme["bg_primary"], **kwargs)
        
        # Canvas for scrolling
        self.canvas = tk.Canvas(
            self,
            bg=self.theme["bg_primary"],
            highlightthickness=0,
            height=70,
        )
        
        # Horizontal scrollbar
        self.scrollbar = ttk.Scrollbar(
            self,
            orient=tk.HORIZONTAL,
            command=self.canvas.xview
        )
        
        # Inner frame for cards
        self.inner_frame = tk.Frame(self.canvas, bg=self.theme["bg_primary"])
        
        # Configure canvas
        self.canvas.configure(xscrollcommand=self.scrollbar.set)
        
        # Pack - canvas on top, scrollbar below
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create window for inner frame
        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.inner_frame,
            anchor="nw",
            height=65
        )
        
        # Bind events
        self.inner_frame.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Mouse wheel scrolling
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_mousewheel)
        self.inner_frame.bind("<MouseWheel>", self._on_mousewheel)
        
        self._cards = []
    
    def _on_configure(self, event=None):
        """Update scroll region when content changes."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        # Check if scrollbar is needed
        self._update_scrollbar_visibility()
    
    def _on_canvas_configure(self, event=None):
        """Handle canvas resize."""
        self._update_scrollbar_visibility()
    
    def _update_scrollbar_visibility(self):
        """Show/hide scrollbar based on content width."""
        self.update_idletasks()
        canvas_width = self.canvas.winfo_width()
        content_width = self.inner_frame.winfo_reqwidth()
        
        if content_width > canvas_width:
            self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        else:
            self.scrollbar.pack_forget()
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel for horizontal scroll."""
        self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def add_card(self, card: tk.Widget):
        """Add a stats card to the row."""
        # Place card in inner frame using pack
        card.pack(in_=self.inner_frame, side=tk.LEFT, padx=self.spacing // 2, pady=2)
        self._cards.append(card)
        
        # Update scroll region
        self.after(10, self._on_configure)
    
    def clear(self):
        """Remove all cards."""
        for card in self._cards:
            card.destroy()
        self._cards.clear()
