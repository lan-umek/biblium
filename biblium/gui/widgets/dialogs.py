# -*- coding: utf-8 -*-
"""
Dialog Widgets
==============
Common dialog windows.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, List, Optional, Any

from biblium.gui.config import FONTS, LAYOUT, get_theme


class BaseDialog(tk.Toplevel):
    """
    Base class for modal dialogs.
    """
    
    def __init__(
        self,
        parent,
        title: str = "Dialog",
        width: int = 400,
        height: int = 200,
        theme: str = "light",
        **kwargs
    ):
        super().__init__(parent, **kwargs)
        
        self.theme = get_theme(theme)
        self.result = None
        
        # Configure window
        self.title(title)
        self.configure(bg=self.theme["bg_card"])
        self.resizable(False, False)
        self.transient(parent)
        
        # Center on parent
        self.geometry(f"{width}x{height}")
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - width) // 2
        y = parent.winfo_y() + (parent.winfo_height() - height) // 2
        self.geometry(f"+{x}+{y}")
        
        # Content frame
        self.content = tk.Frame(self, bg=self.theme["bg_card"], padx=24, pady=20)
        self.content.pack(fill=tk.BOTH, expand=True)
        
        # Button frame
        self.button_frame = tk.Frame(self, bg=self.theme["bg_card"], padx=24, pady=(0, 16))
        self.button_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Handle close
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        
        # Grab focus
        self.grab_set()
        self.focus_set()
    
    def _on_ok(self):
        """Handle OK button."""
        self.result = True
        self.destroy()
    
    def _on_cancel(self):
        """Handle cancel/close."""
        self.result = None
        self.destroy()
    
    def show(self) -> Any:
        """Show dialog and wait for result."""
        self.wait_window()
        return self.result


class MessageDialog(BaseDialog):
    """
    Simple message dialog.
    
    Usage:
        MessageDialog(parent, title="Info", message="Operation complete!", icon="info")
    """
    
    def __init__(
        self,
        parent,
        title: str = "Message",
        message: str = "",
        icon: str = "info",  # info, warning, error, question
        theme: str = "light",
    ):
        super().__init__(parent, title=title, width=400, height=160, theme=theme)
        
        # Icon mapping
        icons = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "question": "❓",
            "success": "✅",
        }
        
        # Message with icon
        msg_frame = tk.Frame(self.content, bg=self.theme["bg_card"])
        msg_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(
            msg_frame,
            text=icons.get(icon, ""),
            font=("Segoe UI", 24),
            bg=self.theme["bg_card"],
        ).pack(side=tk.LEFT, padx=(0, 16))
        
        tk.Label(
            msg_frame,
            text=message,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            wraplength=300,
            justify=tk.LEFT,
        ).pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # OK button
        from biblium.gui.widgets.buttons import ThemedButton
        ThemedButton(
            self.button_frame,
            text="OK",
            command=self._on_ok,
            style="primary",
            theme=theme,
        ).pack(side=tk.RIGHT)


class ConfirmDialog(BaseDialog):
    """
    Confirmation dialog with Yes/No buttons.
    
    Usage:
        if ConfirmDialog(parent, message="Delete item?").show():
            delete_item()
    """
    
    def __init__(
        self,
        parent,
        title: str = "Confirm",
        message: str = "Are you sure?",
        yes_text: str = "Yes",
        no_text: str = "No",
        theme: str = "light",
    ):
        super().__init__(parent, title=title, width=400, height=160, theme=theme)
        
        # Message
        tk.Label(
            self.content,
            text="❓",
            font=("Segoe UI", 24),
            bg=self.theme["bg_card"],
        ).pack(side=tk.LEFT, padx=(0, 16))
        
        tk.Label(
            self.content,
            text=message,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            wraplength=300,
            justify=tk.LEFT,
        ).pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Buttons
        from biblium.gui.widgets.buttons import ThemedButton
        
        ThemedButton(
            self.button_frame,
            text=no_text,
            command=self._on_cancel,
            style="secondary",
            theme=theme,
        ).pack(side=tk.RIGHT, padx=(8, 0))
        
        ThemedButton(
            self.button_frame,
            text=yes_text,
            command=self._on_ok,
            style="primary",
            theme=theme,
        ).pack(side=tk.RIGHT)
    
    def _on_ok(self):
        self.result = True
        self.destroy()
    
    def _on_cancel(self):
        self.result = False
        self.destroy()


class InputDialog(BaseDialog):
    """
    Dialog for text input.
    
    Usage:
        value = InputDialog(parent, title="Enter Name", label="Name:").show()
        if value:
            print(f"Entered: {value}")
    """
    
    def __init__(
        self,
        parent,
        title: str = "Input",
        label: str = "Value:",
        default: str = "",
        theme: str = "light",
    ):
        super().__init__(parent, title=title, width=400, height=160, theme=theme)
        
        # Label
        tk.Label(
            self.content,
            text=label,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(anchor=tk.W)
        
        # Entry
        self.var = tk.StringVar(value=default)
        self.entry = tk.Entry(
            self.content,
            textvariable=self.var,
            font=FONTS.get_font("body"),
            width=40,
            relief=tk.FLAT,
            bg=self.theme["bg_input"],
            fg=self.theme["text_primary"],
            highlightbackground=self.theme["border"],
            highlightthickness=1,
        )
        self.entry.pack(fill=tk.X, pady=(8, 0))
        self.entry.focus_set()
        self.entry.select_range(0, tk.END)
        
        # Bind Enter key
        self.entry.bind("<Return>", lambda e: self._on_ok())
        
        # Buttons
        from biblium.gui.widgets.buttons import ThemedButton
        
        ThemedButton(
            self.button_frame,
            text="Cancel",
            command=self._on_cancel,
            style="secondary",
            theme=theme,
        ).pack(side=tk.RIGHT, padx=(8, 0))
        
        ThemedButton(
            self.button_frame,
            text="OK",
            command=self._on_ok,
            style="primary",
            theme=theme,
        ).pack(side=tk.RIGHT)
    
    def _on_ok(self):
        self.result = self.var.get()
        self.destroy()


class SelectDialog(BaseDialog):
    """
    Dialog for selecting from a list.
    
    Usage:
        selected = SelectDialog(parent, title="Select", options=["A", "B", "C"]).show()
    """
    
    def __init__(
        self,
        parent,
        title: str = "Select",
        label: str = "Choose an option:",
        options: List[str] = None,
        default: str = None,
        theme: str = "light",
    ):
        super().__init__(parent, title=title, width=400, height=200, theme=theme)
        
        options = options or []
        
        # Label
        tk.Label(
            self.content,
            text=label,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(anchor=tk.W)
        
        # Listbox
        list_frame = tk.Frame(self.content, bg=self.theme["border"])
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.listbox = tk.Listbox(
            list_frame,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_input"],
            fg=self.theme["text_primary"],
            selectbackground=self.theme["accent_light"],
            selectforeground=self.theme["accent_primary"],
            relief=tk.FLAT,
            yscrollcommand=scrollbar.set,
        )
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        scrollbar.config(command=self.listbox.yview)
        
        for option in options:
            self.listbox.insert(tk.END, option)
        
        if default and default in options:
            idx = options.index(default)
            self.listbox.selection_set(idx)
            self.listbox.see(idx)
        elif options:
            self.listbox.selection_set(0)
        
        # Double-click to select
        self.listbox.bind("<Double-1>", lambda e: self._on_ok())
        
        # Buttons
        from biblium.gui.widgets.buttons import ThemedButton
        
        ThemedButton(
            self.button_frame,
            text="Cancel",
            command=self._on_cancel,
            style="secondary",
            theme=theme,
        ).pack(side=tk.RIGHT, padx=(8, 0))
        
        ThemedButton(
            self.button_frame,
            text="Select",
            command=self._on_ok,
            style="primary",
            theme=theme,
        ).pack(side=tk.RIGHT)
    
    def _on_ok(self):
        selection = self.listbox.curselection()
        if selection:
            self.result = self.listbox.get(selection[0])
        self.destroy()
