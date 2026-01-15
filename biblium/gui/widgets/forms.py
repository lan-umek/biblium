# -*- coding: utf-8 -*-
"""
Form Widgets
============
Input widgets with labels and consistent styling.
"""

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, List, Optional, Tuple, Union

from biblium.gui.config import FONTS, LAYOUT, get_theme


class LabeledEntry(tk.Frame):
    """
    Entry field with label.
    """
    
    def __init__(
        self,
        parent,
        label: str,
        default: str = "",
        placeholder: str = "",
        variable: Optional[tk.StringVar] = None,
        width: int = 25,
        theme: str = "light",
        tooltip: str = "",
        label_width: int = 20,
        **kwargs
    ):
        self.theme = get_theme(theme)
        self.placeholder = placeholder
        self._has_placeholder = False
        
        # Remove variable from kwargs if accidentally passed
        kwargs.pop('variable', None)
        
        super().__init__(parent, bg=self.theme["bg_card"], **kwargs)
        
        self.label = tk.Label(
            self, text=label, font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            width=label_width, anchor=tk.W,
        )
        self.label.pack(side=tk.LEFT)
        
        # Use provided variable or create new one
        if variable is not None:
            self.var = variable
        else:
            self.var = tk.StringVar(value=default)
        
        self.entry = tk.Entry(
            self, textvariable=self.var, font=FONTS.get_font("body"),
            width=width, relief=tk.FLAT, bg=self.theme["bg_input"],
            fg=self.theme["text_primary"], insertbackground=self.theme["text_primary"],
            highlightbackground=self.theme["border"],
            highlightcolor=self.theme["border_focus"], highlightthickness=1,
            disabledforeground=self.theme["text_muted"],
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(LAYOUT.spacing_sm, 0))
        
        if placeholder and not default:
            self._show_placeholder()
        
        self.entry.bind("<FocusIn>", self._on_focus_in)
        self.entry.bind("<FocusOut>", self._on_focus_out)
        self.entry.bind("<Key>", self._on_key)
        
        if tooltip:
            from biblium.gui.widgets.tooltips import create_tooltip
            create_tooltip(self.label, tooltip)
    
    def _show_placeholder(self):
        self.entry.configure(fg=self.theme["text_muted"])
        self.var.set(self.placeholder)
        self._has_placeholder = True
    
    def _hide_placeholder(self):
        self.entry.configure(fg=self.theme["text_primary"])
        self.var.set("")
        self._has_placeholder = False
    
    def _on_focus_in(self, event):
        if self._has_placeholder:
            self._hide_placeholder()
        # Always ensure text color is correct on focus
        self.entry.configure(fg=self.theme["text_primary"])
    
    def _on_focus_out(self, event):
        if not self.var.get() and self.placeholder:
            self._show_placeholder()
    
    def _on_key(self, event):
        """Ensure text is visible when typing."""
        if self._has_placeholder:
            self._hide_placeholder()
        # Ensure text color is always visible when typing
        self.entry.configure(fg=self.theme["text_primary"])
    
    def get(self) -> str:
        if self._has_placeholder:
            return ""
        return self.var.get()
    
    def set(self, value: str):
        if self._has_placeholder:
            self._hide_placeholder()
        self.var.set(value)
    
    def clear(self):
        self.var.set("")
        if self.placeholder:
            self._show_placeholder()


class LabeledCombobox(tk.Frame):
    """
    Combobox with label.
    """
    
    def __init__(
        self,
        parent,
        label: str,
        values: List[str],
        default: str = None,
        variable: Optional[tk.StringVar] = None,
        width: int = 22,
        theme: str = "light",
        tooltip: str = "",
        label_width: int = 20,
        command: Optional[Callable] = None,
        **kwargs
    ):
        self.theme = get_theme(theme)
        self._command = command
        
        # Remove variable from kwargs if accidentally passed
        kwargs.pop('variable', None)
        
        super().__init__(parent, bg=self.theme["bg_card"], **kwargs)
        
        self.label = tk.Label(
            self, text=label, font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            width=label_width, anchor=tk.W,
        )
        self.label.pack(side=tk.LEFT)
        
        # Use provided variable or create new one
        if variable is not None:
            self.var = variable
        else:
            self.var = tk.StringVar()
        
        self.combo = ttk.Combobox(
            self, textvariable=self.var, values=values,
            state="readonly", width=width, font=FONTS.get_font("body"),
        )
        self.combo.pack(side=tk.LEFT, padx=(LAYOUT.spacing_sm, 0))
        
        # Alias for backwards compatibility
        self.combobox = self.combo
        
        if default and default in values:
            self.combo.set(default)
        elif variable is None and values:
            # Only set default if we created the variable
            self.combo.current(0)
        
        if command:
            self.combo.bind("<<ComboboxSelected>>", lambda e: command(self.var.get()))
        
        if tooltip:
            from biblium.gui.widgets.tooltips import create_tooltip
            create_tooltip(self.label, tooltip)
    
    def get(self) -> str:
        return self.var.get()
    
    def set(self, value: str):
        self.var.set(value)
    
    def get_index(self) -> int:
        return self.combo.current()
    
    def set_values(self, values: List[str], keep_selection: bool = True):
        current = self.get() if keep_selection else None
        self.combo["values"] = values
        if current and current in values:
            self.set(current)
        elif values:
            self.combo.current(0)


class LabeledCheckbox(tk.Frame):
    """
    Checkbox with label.
    """
    
    def __init__(
        self,
        parent,
        label: str,
        default: bool = False,
        variable: Optional[tk.BooleanVar] = None,
        theme: str = "light",
        tooltip: str = "",
        command: Optional[Callable] = None,
        **kwargs
    ):
        self.theme = get_theme(theme)
        self._command = command
        
        # Remove variable from kwargs if accidentally passed
        kwargs.pop('variable', None)
        
        super().__init__(parent, bg=self.theme["bg_card"], **kwargs)
        
        # Use provided variable or create new one
        if variable is not None:
            self.var = variable
        else:
            self.var = tk.BooleanVar(value=default)
        
        self.checkbox = tk.Checkbutton(
            self, text=label, variable=self.var,
            font=FONTS.get_font("body"), bg=self.theme["bg_card"],
            fg=self.theme["text_primary"], activebackground=self.theme["bg_card"],
            activeforeground=self.theme["text_primary"],
            selectcolor=self.theme["bg_input"], command=self._on_change,
        )
        self.checkbox.pack(side=tk.LEFT)
        
        if tooltip:
            from biblium.gui.widgets.tooltips import create_tooltip
            create_tooltip(self.checkbox, tooltip)
    
    def _on_change(self):
        if self._command:
            self._command(self.var.get())
    
    def get(self) -> bool:
        return self.var.get()
    
    def set(self, value: bool):
        self.var.set(value)


class LabeledSpinbox(tk.Frame):
    """
    Spinbox with label.
    """
    
    def __init__(
        self,
        parent,
        label: str,
        from_: int = 0,
        to: int = 100,
        default: int = None,
        variable: Optional[tk.IntVar] = None,
        width: int = 10,
        theme: str = "light",
        tooltip: str = "",
        label_width: int = 20,
        command: Optional[Callable] = None,
        **kwargs
    ):
        self.theme = get_theme(theme)
        self._command = command
        
        # Remove variable from kwargs if accidentally passed
        kwargs.pop('variable', None)
        
        super().__init__(parent, bg=self.theme["bg_card"], **kwargs)
        
        self.label = tk.Label(
            self, text=label, font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            width=label_width, anchor=tk.W,
        )
        self.label.pack(side=tk.LEFT)
        
        # Use provided variable or create new one
        if variable is not None:
            self.var = variable
        else:
            self.var = tk.IntVar(value=default if default is not None else from_)
        
        self.spinbox = tk.Spinbox(
            self, from_=from_, to=to, textvariable=self.var,
            font=FONTS.get_font("body"), width=width, relief=tk.FLAT,
            bg=self.theme["bg_input"], fg=self.theme["text_primary"],
            buttonbackground=self.theme["bg_secondary"],
            highlightbackground=self.theme["border"],
            highlightcolor=self.theme["border_focus"], highlightthickness=1,
            command=self._on_change,
        )
        self.spinbox.pack(side=tk.LEFT, padx=(LAYOUT.spacing_sm, 0))
        
        if tooltip:
            from biblium.gui.widgets.tooltips import create_tooltip
            create_tooltip(self.label, tooltip)
    
    def _on_change(self):
        if self._command:
            try:
                self._command(self.var.get())
            except tk.TclError:
                pass
    
    def get(self) -> int:
        try:
            return self.var.get()
        except tk.TclError:
            return 0
    
    def set(self, value: int):
        self.var.set(value)


class FormSection(tk.Frame):
    """
    A section of form fields with optional title.
    """
    
    def __init__(
        self,
        parent,
        title: str = "",
        theme: str = "light",
        **kwargs
    ):
        self.theme = get_theme(theme)
        
        super().__init__(parent, bg=self.theme["bg_card"], **kwargs)
        
        if title:
            title_label = tk.Label(
                self, text=title, font=FONTS.get_font("body", bold=True),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"], anchor=tk.W,
            )
            title_label.pack(fill=tk.X, pady=(0, LAYOUT.spacing_sm))
        
        self.content = tk.Frame(self, bg=self.theme["bg_card"])
        self.content.pack(fill=tk.BOTH, expand=True)
    
    def add_field(self, widget_class, **kwargs) -> Any:
        kwargs.setdefault("theme", self.theme.get("name", "light"))
        widget = widget_class(self.content, **kwargs)
        widget.pack(fill=tk.X, pady=LAYOUT.spacing_xs)
        return widget


class LabeledTextArea(tk.Frame):
    """
    Multi-line text area with label.
    """
    
    def __init__(
        self,
        parent,
        label: str,
        height: int = 4,
        width: int = 40,
        default: str = "",
        theme: str = "light",
        tooltip: str = "",
        **kwargs
    ):
        self.theme = get_theme(theme)
        
        super().__init__(parent, bg=self.theme["bg_card"], **kwargs)
        
        self.label = tk.Label(
            self, text=label, font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"], anchor=tk.W,
        )
        self.label.pack(fill=tk.X)
        
        text_frame = tk.Frame(self, bg=self.theme["border"])
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(LAYOUT.spacing_xs, 0))
        
        self.text = tk.Text(
            text_frame, font=FONTS.get_font("body"), height=height, width=width,
            relief=tk.FLAT, bg=self.theme["bg_input"], fg=self.theme["text_primary"],
            insertbackground=self.theme["text_primary"],
            padx=LAYOUT.spacing_sm, pady=LAYOUT.spacing_sm,
        )
        self.text.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        if default:
            self.text.insert("1.0", default)
        
        if tooltip:
            from biblium.gui.widgets.tooltips import create_tooltip
            create_tooltip(self.label, tooltip)
    
    def get(self) -> str:
        return self.text.get("1.0", tk.END).strip()
    
    def set(self, value: str):
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", value)
    
    def clear(self):
        self.text.delete("1.0", tk.END)


class RadioGroup(tk.Frame):
    """
    Group of radio buttons.
    """
    
    def __init__(
        self,
        parent,
        label: str,
        options: List[str],
        default: str = None,
        orientation: str = "horizontal",
        theme: str = "light",
        command: Optional[Callable] = None,
        **kwargs
    ):
        self.theme = get_theme(theme)
        self._command = command
        
        super().__init__(parent, bg=self.theme["bg_card"], **kwargs)
        
        if label:
            self.label = tk.Label(
                self, text=label, font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"], anchor=tk.W,
            )
            self.label.pack(fill=tk.X)
        
        options_frame = tk.Frame(self, bg=self.theme["bg_card"])
        options_frame.pack(fill=tk.X, pady=(LAYOUT.spacing_xs, 0))
        
        self.var = tk.StringVar(value=default or (options[0] if options else ""))
        
        for option in options:
            rb = tk.Radiobutton(
                options_frame, text=option, variable=self.var, value=option,
                font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                fg=self.theme["text_primary"], activebackground=self.theme["bg_card"],
                activeforeground=self.theme["text_primary"],
                selectcolor=self.theme["bg_input"], command=self._on_change,
            )
            if orientation == "horizontal":
                rb.pack(side=tk.LEFT, padx=(0, LAYOUT.spacing_md))
            else:
                rb.pack(anchor=tk.W)
    
    def _on_change(self):
        if self._command:
            self._command(self.var.get())
    
    def get(self) -> str:
        return self.var.get()
    
    def set(self, value: str):
        self.var.set(value)


class DualListSelector(tk.Frame):
    """
    Two-column list selector with Available and Selected lists.
    Allows moving items between lists using arrow buttons.
    
    By default:
    - Left column (available_label): GREEN - items to include/show
    - Right column (selected_label): RED - items to exclude/hide
    """
    
    def __init__(
        self,
        parent,
        available_label: str = "Available",
        selected_label: str = "Selected",
        available_color: str = "#27ae60",  # Green
        selected_color: str = "#c0392b",   # Red
        height: int = 8,
        theme: str = "light",
        **kwargs
    ):
        self.theme = get_theme(theme)
        self.theme_name = theme
        self.available_color = available_color
        self.selected_color = selected_color
        
        super().__init__(parent, bg=self.theme["bg_card"], **kwargs)
        
        # Main container
        container = tk.Frame(self, bg=self.theme["bg_card"])
        container.pack(fill=tk.BOTH, expand=True)
        
        # Available list (left)
        available_frame = tk.Frame(container, bg=self.theme["bg_card"])
        available_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tk.Label(
            available_frame, text=available_label,
            font=FONTS.get_font("body", bold=True),
            bg=self.theme["bg_card"], fg=available_color,
        ).pack(anchor=tk.W)
        
        # Available listbox with scrollbar
        avail_list_frame = tk.Frame(available_frame, bg=self.theme["bg_card"])
        avail_list_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        
        avail_scrollbar = ttk.Scrollbar(avail_list_frame, orient=tk.VERTICAL)
        avail_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.available_listbox = tk.Listbox(
            avail_list_frame, height=height,
            font=FONTS.get_font("small"),
            selectmode=tk.EXTENDED,
            bg=self.theme["bg_input"],
            fg=self.theme["text_primary"],
            selectbackground=available_color,
            selectforeground="white",
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.theme["border"],
            yscrollcommand=avail_scrollbar.set,
            exportselection=False,
        )
        self.available_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        avail_scrollbar.config(command=self.available_listbox.yview)
        
        # Arrow buttons (center)
        btn_frame = tk.Frame(container, bg=self.theme["bg_card"])
        btn_frame.pack(side=tk.LEFT, padx=4)
        
        btn_style = {
            "font": FONTS.get_font("body", bold=True),
            "width": 3,
            "bg": self.theme["bg_input"],
            "fg": self.theme["text_primary"],
            "relief": tk.FLAT,
            "cursor": "hand2",
        }
        
        tk.Button(
            btn_frame, text="→", command=self._move_to_selected, **btn_style
        ).pack(pady=2)
        
        tk.Button(
            btn_frame, text="←", command=self._move_to_available, **btn_style
        ).pack(pady=2)
        
        tk.Button(
            btn_frame, text="⇒", command=self._move_all_to_selected, **btn_style
        ).pack(pady=2)
        
        tk.Button(
            btn_frame, text="⇐", command=self._move_all_to_available, **btn_style
        ).pack(pady=2)
        
        # Selected list (right)
        selected_frame = tk.Frame(container, bg=self.theme["bg_card"])
        selected_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tk.Label(
            selected_frame, text=selected_label,
            font=FONTS.get_font("body", bold=True),
            bg=self.theme["bg_card"], fg=selected_color,
        ).pack(anchor=tk.W)
        
        # Selected listbox with scrollbar
        sel_list_frame = tk.Frame(selected_frame, bg=self.theme["bg_card"])
        sel_list_frame.pack(fill=tk.BOTH, expand=True, pady=2)
        
        sel_scrollbar = ttk.Scrollbar(sel_list_frame, orient=tk.VERTICAL)
        sel_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.selected_listbox = tk.Listbox(
            sel_list_frame, height=height,
            font=FONTS.get_font("small"),
            selectmode=tk.EXTENDED,
            bg=self.theme["bg_input"],
            fg=self.theme["text_primary"],
            selectbackground=selected_color,
            selectforeground="white",
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.theme["border"],
            yscrollcommand=sel_scrollbar.set,
            exportselection=False,
        )
        self.selected_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sel_scrollbar.config(command=self.selected_listbox.yview)
    
    def _move_to_selected(self):
        """Move selected items from available to selected."""
        selected = list(self.available_listbox.curselection())
        items = [self.available_listbox.get(i) for i in selected]
        for item in items:
            self.selected_listbox.insert(tk.END, item)
        for i in reversed(selected):
            self.available_listbox.delete(i)
    
    def _move_to_available(self):
        """Move selected items from selected to available."""
        selected = list(self.selected_listbox.curselection())
        items = [self.selected_listbox.get(i) for i in selected]
        for item in items:
            self.available_listbox.insert(tk.END, item)
        for i in reversed(selected):
            self.selected_listbox.delete(i)
    
    def _move_all_to_selected(self):
        """Move all items to selected."""
        items = list(self.available_listbox.get(0, tk.END))
        for item in items:
            self.selected_listbox.insert(tk.END, item)
        self.available_listbox.delete(0, tk.END)
    
    def _move_all_to_available(self):
        """Move all items to available."""
        items = list(self.selected_listbox.get(0, tk.END))
        for item in items:
            self.available_listbox.insert(tk.END, item)
        self.selected_listbox.delete(0, tk.END)
    
    def set_available_items(self, items: List[str]):
        """Set the available items list."""
        self.available_listbox.delete(0, tk.END)
        for item in items:
            self.available_listbox.insert(tk.END, item)
    
    def set_selected_items(self, items: List[str]):
        """Set the selected items list."""
        self.selected_listbox.delete(0, tk.END)
        for item in items:
            self.selected_listbox.insert(tk.END, item)
    
    def get_available(self) -> List[str]:
        """Get list of available items."""
        return list(self.available_listbox.get(0, tk.END))
    
    def get_selected(self) -> List[str]:
        """Get list of selected items."""
        return list(self.selected_listbox.get(0, tk.END))
    
    def clear(self):
        """Clear both lists."""
        self.available_listbox.delete(0, tk.END)
        self.selected_listbox.delete(0, tk.END)
