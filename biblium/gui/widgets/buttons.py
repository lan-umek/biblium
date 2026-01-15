# -*- coding: utf-8 -*-
"""
Button Widgets
==============
Custom themed buttons.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, Tuple

from biblium.gui.config import FONTS, get_theme


class ThemedButton(tk.Button):
    """
    A themed button with hover effects.
    
    Parameters
    ----------
    parent : widget
        Parent widget.
    text : str
        Button text.
    command : callable, optional
        Callback function.
    style : str
        Button style: "primary", "secondary", "success", "warning", "danger"
    size : str
        Button size: "small", "medium", "large"
    theme : str
        Theme name: "light" or "dark"
    """
    
    def __init__(
        self,
        parent,
        text: str = "",
        command: Optional[Callable] = None,
        style: str = "primary",
        size: str = "medium",
        theme: str = "light",
        icon: str = "",
        **kwargs
    ):
        self.theme_name = theme
        self.theme = get_theme(theme)
        self.style_name = style
        self.size = size
        
        # Get colors based on style
        colors = self._get_style_colors()
        
        # Get font and padding based on size
        font, padx, pady = self._get_size_config()
        
        # Build display text
        display_text = f"{icon}  {text}" if icon else text
        
        super().__init__(
            parent,
            text=display_text,
            command=command,
            font=font,
            bg=colors["bg"],
            fg=colors["fg"],
            activebackground=colors["hover"],
            activeforeground=colors["fg"],
            relief=tk.FLAT,
            cursor="hand2",
            padx=padx,
            pady=pady,
            bd=0,
            **kwargs
        )
        
        self._colors = colors
        self._bind_hover()
    
    def _get_style_colors(self) -> dict:
        """Get colors based on button style."""
        t = self.theme
        styles = {
            "primary": {
                "bg": t["accent_primary"],
                "hover": t["accent_hover"],
                "fg": t["text_on_accent"],
            },
            "secondary": {
                "bg": t["bg_secondary"],
                "hover": t["border"],
                "fg": t["text_primary"],
            },
            "success": {
                "bg": t["success"],
                "hover": t["success_hover"],
                "fg": t["text_on_success"],
            },
            "warning": {
                "bg": t["warning"],
                "hover": t["warning_hover"],
                "fg": t["text_on_warning"],
            },
            "danger": {
                "bg": t["danger"],
                "hover": t["danger_hover"],
                "fg": t["text_on_danger"],
            },
            "ghost": {
                "bg": t["bg_primary"],
                "hover": t["bg_secondary"],
                "fg": t["accent_primary"],
            },
        }
        return styles.get(self.style_name, styles["primary"])
    
    def _get_size_config(self) -> Tuple[Tuple, int, int]:
        """Get font and padding based on size."""
        sizes = {
            "small": (FONTS.get_font("small"), 10, 4),
            "medium": (FONTS.get_font("body"), 16, 8),
            "large": (FONTS.get_font("body", bold=True), 20, 10),
        }
        return sizes.get(self.size, sizes["medium"])
    
    def _bind_hover(self):
        """Bind hover effects."""
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
    
    def _on_enter(self, event):
        """Mouse enter handler."""
        self.configure(bg=self._colors["hover"])
    
    def _on_leave(self, event):
        """Mouse leave handler."""
        self.configure(bg=self._colors["bg"])
    
    def update_theme(self, theme_name: str):
        """Update button theme."""
        self.theme_name = theme_name
        self.theme = get_theme(theme_name)
        self._colors = self._get_style_colors()
        self.configure(
            bg=self._colors["bg"],
            fg=self._colors["fg"],
            activebackground=self._colors["hover"],
        )
    
    def set_enabled(self, enabled: bool):
        """Enable or disable the button."""
        if enabled:
            self.configure(state=tk.NORMAL, cursor="hand2")
        else:
            self.configure(state=tk.DISABLED, cursor="")


class IconButton(tk.Button):
    """
    A compact icon-only button.
    """
    
    def __init__(
        self,
        parent,
        icon: str,
        command: Optional[Callable] = None,
        tooltip: str = "",
        theme: str = "light",
        size: int = 28,
        **kwargs
    ):
        self.theme = get_theme(theme)
        
        super().__init__(
            parent,
            text=icon,
            command=command,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_primary"],
            fg=self.theme["text_secondary"],
            activebackground=self.theme["bg_secondary"],
            activeforeground=self.theme["text_primary"],
            relief=tk.FLAT,
            cursor="hand2",
            width=2,
            height=1,
            bd=0,
            **kwargs
        )
        
        self._bind_hover()
        
        # Add tooltip if provided
        if tooltip:
            from biblium.gui.widgets.tooltips import create_tooltip
            create_tooltip(self, tooltip)
    
    def _bind_hover(self):
        self.bind("<Enter>", lambda e: self.configure(bg=self.theme["bg_secondary"]))
        self.bind("<Leave>", lambda e: self.configure(bg=self.theme["bg_primary"]))


class ActionButton(ThemedButton):
    """
    A prominent action button with icon.
    
    Used for primary actions like "Run Analysis", "Generate Report".
    """
    
    def __init__(
        self,
        parent,
        text: str,
        command: Optional[Callable] = None,
        icon: str = "▶",
        theme: str = "light",
        **kwargs
    ):
        super().__init__(
            parent,
            text=text,
            command=command,
            style="primary",
            size="medium",
            theme=theme,
            icon=icon,
            **kwargs
        )


class ToggleButton(tk.Button):
    """
    A button that toggles between two states.
    """
    
    def __init__(
        self,
        parent,
        text_on: str,
        text_off: str,
        command: Optional[Callable] = None,
        initial_state: bool = False,
        theme: str = "light",
        **kwargs
    ):
        self.theme = get_theme(theme)
        self.text_on = text_on
        self.text_off = text_off
        self.state = initial_state
        self._command = command
        
        super().__init__(
            parent,
            text=text_on if initial_state else text_off,
            command=self._toggle,
            font=FONTS.get_font("body"),
            relief=tk.FLAT,
            cursor="hand2",
            padx=12,
            pady=6,
            bd=0,
            **kwargs
        )
        
        self._update_appearance()
        self._bind_hover()
    
    def _toggle(self):
        """Toggle the state."""
        self.state = not self.state
        self._update_appearance()
        if self._command:
            self._command(self.state)
    
    def _update_appearance(self):
        """Update button appearance based on state."""
        if self.state:
            self.configure(
                text=self.text_on,
                bg=self.theme["accent_primary"],
                fg=self.theme["text_on_accent"],
            )
        else:
            self.configure(
                text=self.text_off,
                bg=self.theme["bg_secondary"],
                fg=self.theme["text_primary"],
            )
    
    def _bind_hover(self):
        def on_enter(e):
            if self.state:
                self.configure(bg=self.theme["accent_hover"])
            else:
                self.configure(bg=self.theme["border"])
        
        def on_leave(e):
            self._update_appearance()
        
        self.bind("<Enter>", on_enter)
        self.bind("<Leave>", on_leave)
    
    def set_state(self, state: bool):
        """Set the toggle state programmatically."""
        self.state = state
        self._update_appearance()
    
    def get_state(self) -> bool:
        """Get the current toggle state."""
        return self.state


class ButtonGroup(tk.Frame):
    """
    A group of buttons arranged horizontally.
    """
    
    def __init__(
        self,
        parent,
        buttons: list,
        theme: str = "light",
        spacing: int = 8,
        **kwargs
    ):
        """
        Parameters
        ----------
        buttons : list
            List of dicts with keys: text, command, style (optional), icon (optional)
        """
        self.theme = get_theme(theme)
        
        super().__init__(parent, bg=self.theme["bg_primary"], **kwargs)
        
        for i, btn_config in enumerate(buttons):
            btn = ThemedButton(
                self,
                text=btn_config.get("text", ""),
                command=btn_config.get("command"),
                style=btn_config.get("style", "secondary"),
                icon=btn_config.get("icon", ""),
                theme=theme,
            )
            btn.pack(side=tk.LEFT, padx=(0 if i == 0 else spacing, 0))
