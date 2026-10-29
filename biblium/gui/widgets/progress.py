# -*- coding: utf-8 -*-
"""
Progress Widgets
================
Progress bars, spinners, and progress dialogs.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme


class ProgressBar(tk.Frame):
    """
    A themed progress bar.
    
    Usage:
        pb = ProgressBar(parent)
        pb.set(0.5)  # 50%
        pb.set_indeterminate(True)  # Animated loading
    """
    
    def __init__(
        self,
        parent,
        height: int = 8,
        theme: str = "light",
        show_percentage: bool = False,
        **kwargs
    ):
        self.theme = get_theme(theme)
        self.show_percentage = show_percentage
        
        super().__init__(parent, bg=self.theme["bg_card"], **kwargs)
        
        # Progress bar
        style = ttk.Style()
        style.configure(
            "Custom.Horizontal.TProgressbar",
            background=self.theme["accent_primary"],
            troughcolor=self.theme["bg_secondary"],
        )
        
        self.progress = ttk.Progressbar(
            self,
            style="Custom.Horizontal.TProgressbar",
            orient=tk.HORIZONTAL,
            mode="determinate",
            length=200,
        )
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        if show_percentage:
            self.percent_label = tk.Label(
                self,
                text="0%",
                font=FONTS.get_font("small"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_secondary"],
                width=5,
            )
            self.percent_label.pack(side=tk.LEFT, padx=(LAYOUT.spacing_sm, 0))
    
    def set(self, value: float):
        """Set progress value (0.0 to 1.0)."""
        self.progress["value"] = value * 100
        if self.show_percentage:
            self.percent_label.configure(text=f"{int(value * 100)}%")
    
    def get(self) -> float:
        """Get progress value."""
        return self.progress["value"] / 100
    
    def set_indeterminate(self, indeterminate: bool):
        """Set indeterminate (loading) mode."""
        if indeterminate:
            self.progress.configure(mode="indeterminate")
            self.progress.start(10)
            if self.show_percentage:
                self.percent_label.configure(text="...")
        else:
            self.progress.stop()
            self.progress.configure(mode="determinate")
    
    def reset(self):
        """Reset progress to 0."""
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.set(0)


class LoadingSpinner(tk.Canvas):
    """
    Animated loading spinner.
    
    Usage:
        spinner = LoadingSpinner(parent)
        spinner.start()
        # ... do work ...
        spinner.stop()
    """
    
    def __init__(
        self,
        parent,
        size: int = 24,
        theme: str = "light",
        **kwargs
    ):
        self.theme = get_theme(theme)
        self.size = size
        
        super().__init__(
            parent,
            width=size,
            height=size,
            bg=self.theme["bg_card"],
            highlightthickness=0,
            **kwargs
        )
        
        self.angle = 0
        self.running = False
        self._after_id = None
        
        # Draw the arc
        padding = 3
        self.arc = self.create_arc(
            padding, padding,
            size - padding, size - padding,
            start=0,
            extent=90,
            outline=self.theme["accent_primary"],
            width=2,
            style="arc",
        )
        
        # Bind destroy event to stop animation
        self.bind("<Destroy>", self._on_destroy)
    
    def start(self):
        """Start the spinner animation."""
        self.running = True
        self._animate()
    
    def stop(self):
        """Stop the spinner animation."""
        self.running = False
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except:
                pass
            self._after_id = None
    
    def _on_destroy(self, event):
        """Handle widget destruction."""
        self.running = False
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except:
                pass
            self._after_id = None
    
    def _animate(self):
        """Animation loop."""
        if self.running:
            try:
                # Check if widget still exists
                if not self.winfo_exists():
                    self.running = False
                    return
                    
                self.angle = (self.angle - 15) % 360
                self.itemconfig(self.arc, start=self.angle)
                self._after_id = self.after(40, self._animate)
            except tk.TclError:
                # Widget has been destroyed
                self.running = False
                self._after_id = None


class ProgressDialog(tk.Toplevel):
    """
    Modal dialog showing progress.
    
    Usage:
        dialog = ProgressDialog(parent, title="Processing", message="Please wait...")
        dialog.set_progress(0.5, "Processing item 50 of 100")
        dialog.close()
    """
    
    def __init__(
        self,
        parent,
        title: str = "Processing",
        message: str = "Please wait...",
        theme: str = "light",
        cancellable: bool = True,
        on_cancel: Optional[Callable] = None,
    ):
        super().__init__(parent)
        
        self.theme = get_theme(theme)
        self.on_cancel = on_cancel
        self._cancelled = False
        
        # Configure window
        self.title(title)
        self.configure(bg=self.theme["bg_card"])
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        # Center on parent
        self.geometry("400x180")
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 400) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 180) // 2
        self.geometry(f"+{x}+{y}")
        
        # Content
        content = tk.Frame(self, bg=self.theme["bg_card"], padx=24, pady=24)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Spinner and title row
        header = tk.Frame(content, bg=self.theme["bg_card"])
        header.pack(fill=tk.X)
        
        self.spinner = LoadingSpinner(header, theme=theme)
        self.spinner.pack(side=tk.LEFT, padx=(0, 12))
        self.spinner.start()
        
        self.title_label = tk.Label(
            header,
            text=title,
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        )
        self.title_label.pack(side=tk.LEFT)
        
        # Message
        self.message_label = tk.Label(
            content,
            text=message,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
            anchor=tk.W,
        )
        self.message_label.pack(fill=tk.X, pady=(16, 12))
        
        # Progress bar
        self.progress = ProgressBar(content, theme=theme, show_percentage=True)
        self.progress.pack(fill=tk.X)
        
        # Cancel button
        if cancellable:
            btn_frame = tk.Frame(content, bg=self.theme["bg_card"])
            btn_frame.pack(fill=tk.X, pady=(16, 0))
            
            from biblium.gui.widgets.buttons import ThemedButton
            self.cancel_btn = ThemedButton(
                btn_frame,
                text="Cancel",
                style="secondary",
                command=self._handle_cancel,
                theme=theme,
            )
            self.cancel_btn.pack(side=tk.RIGHT)
        
        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._handle_cancel if cancellable else lambda: None)
    
    def set_progress(self, value: float, message: str = None):
        """Update progress."""
        self.progress.set(value)
        if message:
            self.message_label.configure(text=message)
        self.update_idletasks()
    
    def set_message(self, message: str):
        """Update message."""
        self.message_label.configure(text=message)
        self.update_idletasks()
    
    def set_indeterminate(self, indeterminate: bool):
        """Set indeterminate mode."""
        self.progress.set_indeterminate(indeterminate)
    
    def _handle_cancel(self):
        """Handle cancel button."""
        self._cancelled = True
        if self.on_cancel:
            self.on_cancel()
        self.close()
    
    def is_cancelled(self) -> bool:
        """Check if cancelled."""
        return self._cancelled
    
    def close(self):
        """Close the dialog."""
        self.spinner.stop()
        self.grab_release()
        self.destroy()
