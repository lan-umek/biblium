# -*- coding: utf-8 -*-
"""
Base Panel Class
================
Foundation for all analysis panels.
"""

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict, Optional

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.core.threading import BackgroundTask, ThrottledCallback
from biblium.gui.widgets.cards import Card, CollapsibleCard
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.progress import LoadingSpinner


class BasePanel(tk.Frame):
    """
    Base class for analysis panels.
    
    Provides:
    - Split layout (options | results)
    - Common UI elements
    - Background task execution
    - Event subscriptions
    
    Subclasses should override:
    - _create_options(): Create the options panel
    - _create_results(): Create the results panel
    - _run_analysis(): Perform the analysis
    
    Subclasses should set:
    - _primary_action: Reference to the main action method (for toolbar Run button)
    """
    
    title: str = "Panel"
    icon: str = ""
    requires_data: bool = True
    
    def __init__(self, parent, theme: str = "light", bib=None, bib_group=None, **kwargs):
        self.theme_name = theme
        self.theme = get_theme(theme)
        self.bib = bib
        self._bib_group = bib_group  # Store bib_group for group panels
        self._current_task = None
        self._primary_action = None  # Set by subclass to main action method
        self._is_running = False  # Track if analysis is running
        self._cancel_requested = False  # Track if cancellation was requested
        
        super().__init__(parent, bg=self.theme["bg_secondary"], **kwargs)
        
        self._create_layout()
        self._create_options()
        self._create_results()
        self._subscribe_events()
    
    def _create_layout(self):
        """Create the panel layout."""
        # Main container with split panes
        self.paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Options panel (left)
        self.options_frame = tk.Frame(
            self.paned,
            bg=self.theme["bg_secondary"],
            width=LAYOUT.options_panel_width,
        )
        self.paned.add(self.options_frame, weight=0)
        
        # Create scrollable options area
        self.options_canvas = tk.Canvas(
            self.options_frame,
            bg=self.theme["bg_secondary"],
            highlightthickness=0,
        )
        self.options_scrollbar = ttk.Scrollbar(
            self.options_frame,
            orient=tk.VERTICAL,
            command=self.options_canvas.yview,
        )
        self.options_content = tk.Frame(
            self.options_canvas,
            bg=self.theme["bg_secondary"],
        )
        
        self._options_canvas_window = self.options_canvas.create_window(
            (0, 0), window=self.options_content, anchor=tk.NW
        )
        self.options_canvas.configure(yscrollcommand=self._on_options_scrollbar_set)
        
        # Bind configure events
        self.options_content.bind("<Configure>", self._on_options_content_configure)
        self.options_canvas.bind("<Configure>", self._on_options_canvas_resize)
        
        # Pack canvas only - scrollbar will be shown only when needed
        self.options_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mouse wheel
        self.options_canvas.bind("<Enter>", lambda e: self._bind_options_scroll())
        self.options_canvas.bind("<Leave>", lambda e: self._unbind_options_scroll())
        
        # Results panel (right)
        self.results_frame = tk.Frame(self.paned, bg=self.theme["bg_secondary"])
        self.paned.add(self.results_frame, weight=1)
    
    def _bind_options_scroll(self):
        self.options_canvas.bind_all("<MouseWheel>", self._on_options_scroll)
    
    def _unbind_options_scroll(self):
        self.options_canvas.unbind_all("<MouseWheel>")
    
    def _on_options_scroll(self, event):
        self.options_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _on_options_content_configure(self, event=None):
        """Update scroll region when content changes."""
        self.options_canvas.configure(scrollregion=self.options_canvas.bbox("all"))
        self.after(10, self._update_options_scrollbar)
    
    def _on_options_canvas_resize(self, event=None):
        """Adjust content width to match canvas and update scrollbar."""
        if hasattr(self, '_options_canvas_window'):
            self.options_canvas.itemconfig(self._options_canvas_window, width=event.width)
        self.after(10, self._update_options_scrollbar)
    
    def _on_options_scrollbar_set(self, first, last):
        """Handle scrollbar position and auto-show/hide."""
        self.options_scrollbar.set(first, last)
        self._update_options_scrollbar()
    
    def _update_options_scrollbar(self):
        """Show scrollbar only when content exceeds canvas height."""
        try:
            first, last = self.options_scrollbar.get()
            if float(first) <= 0.0 and float(last) >= 1.0:
                # All content visible - hide scrollbar
                if self.options_scrollbar.winfo_ismapped():
                    self.options_scrollbar.pack_forget()
            else:
                # Need scrollbar
                if not self.options_scrollbar.winfo_ismapped():
                    self.options_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, before=self.options_canvas)
        except Exception:
            pass
    
    def _create_options(self):
        """Create options panel content. Override in subclass."""
        # Title
        self._add_title()
        
        # Placeholder
        Card(self.options_content, title="Options", theme=self.theme_name).pack(
            fill=tk.X, padx=8, pady=8
        )
    
    def _create_results(self):
        """Create results panel content. Override in subclass."""
        # Results card
        self.results_card = tk.Frame(
            self.results_frame,
            bg=self.theme["bg_card"],
            highlightbackground=self.theme["border"],
            highlightthickness=1,
        )
        self.results_card.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        # Header with tabs
        header = tk.Frame(self.results_card, bg=self.theme["bg_card"])
        header.pack(fill=tk.X)
        
        tk.Label(
            header,
            text="Results",
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            padx=12,
            pady=8,
        ).pack(side=tk.LEFT)
        
        # Separator
        ttk.Separator(self.results_card, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12)
        
        # Content area
        self.results_content = tk.Frame(self.results_card, bg=self.theme["bg_card"])
        self.results_content.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        
        # Initial message
        self._show_initial_message()
    
    def _add_title(self):
        """Add panel title."""
        title_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        title_frame.pack(fill=tk.X, padx=8, pady=(8, 16))
        
        display_title = f"{self.icon} {self.title}" if self.icon else self.title
        
        tk.Label(
            title_frame,
            text=display_title,
            font=FONTS.get_font("title", bold=True),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_primary"],
        ).pack(anchor=tk.W)
        
        if hasattr(self, "description"):
            tk.Label(
                title_frame,
                text=self.description,
                font=FONTS.get_font("body"),
                bg=self.theme["bg_secondary"],
                fg=self.theme["text_muted"],
                wraplength=LAYOUT.options_panel_width - 20,
                justify=tk.LEFT,
            ).pack(anchor=tk.W, pady=(4, 0))
    
    def _add_run_button(self, text: str = "Run Analysis", command: Callable = None):
        """Add the run analysis button."""
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        self.run_btn = ActionButton(
            btn_frame,
            text=text,
            icon="▶",
            command=command or self._run_clicked,
            theme=self.theme_name,
        )
        self.run_btn.pack(fill=tk.X)
        
        # Loading spinner (hidden)
        self.spinner_frame = tk.Frame(btn_frame, bg=self.theme["bg_secondary"])
        self.spinner = LoadingSpinner(self.spinner_frame, theme=self.theme_name)
        self.status_label = tk.Label(
            self.spinner_frame,
            text="",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_muted"],
        )
    
    def _show_initial_message(self, message: str = None):
        """Show initial message in results area."""
        self._stop_active_spinners()
        self._safe_clear_results()
        
        if message is None:
            message = "Configure options and click Run to see results"
        
        tk.Label(
            self.results_content,
            text=message,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(expand=True)
    
    def _safe_clear_results(self):
        """Safely clear results content, handling threading issues."""
        try:
            for widget in self.results_content.winfo_children():
                try:
                    widget.destroy()
                except:
                    pass
        except:
            pass
        # Force update to complete widget destruction
        try:
            self.update_idletasks()
        except:
            pass
    
    def _show_loading(self, message: str = "Processing..."):
        """Show loading state."""
        self._stop_active_spinners()
        self._safe_clear_results()
        
        loading_frame = tk.Frame(self.results_content, bg=self.theme["bg_card"])
        loading_frame.pack(expand=True)
        
        spinner = LoadingSpinner(loading_frame, size=32, theme=self.theme_name)
        spinner.pack()
        spinner.start()
        
        # Track active spinner for cleanup
        if not hasattr(self, '_active_spinners'):
            self._active_spinners = []
        self._active_spinners.append(spinner)
        
        tk.Label(
            loading_frame,
            text=message,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(pady=(12, 0))
    
    def _stop_active_spinners(self):
        """Stop all active spinners before clearing content."""
        if hasattr(self, '_active_spinners'):
            for spinner in self._active_spinners:
                try:
                    spinner.stop()
                except:
                    pass
            self._active_spinners.clear()
    
    def _safe_after(self, callback, delay=0):
        """Safely schedule a callback on the main thread.
        
        This wraps self.after() in a try-except to handle cases where
        the widget has been destroyed or the main loop isn't running.
        
        Parameters
        ----------
        callback : callable
            Function to call on main thread
        delay : int
            Milliseconds to wait before calling (default 0)
        """
        try:
            if self.winfo_exists():
                self.after(delay, callback)
        except (tk.TclError, RuntimeError) as e:
            # Widget was destroyed or main loop issue
            print(f"[WARNING] Could not schedule callback: {e}")
    
    def _show_error(self, message: str):
        """Show error message."""
        self._stop_active_spinners()
        self._safe_clear_results()
        
        error_frame = tk.Frame(self.results_content, bg=self.theme["bg_card"])
        error_frame.pack(expand=True)
        
        tk.Label(
            error_frame,
            text="❌",
            font=("Segoe UI", 32),
            bg=self.theme["bg_card"],
        ).pack()
        
        tk.Label(
            error_frame,
            text="Error",
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"],
            fg=self.theme["danger"],
        ).pack(pady=(8, 4))
        
        tk.Label(
            error_frame,
            text=message,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_secondary"],
            wraplength=400,
        ).pack()
    
    def _run_clicked(self):
        """Handle run button click - called by toolbar Run button."""
        if self._is_running:
            return  # Already running
            
        if self.requires_data and self.bib is None:
            self._show_error("Please load a dataset first")
            return
        
        # If panel has a primary action set, call it
        if self._primary_action and callable(self._primary_action):
            self._primary_action()
            return
        
        # Fallback to _run_analysis (common for many panels)
        event_bus.emit(EventBus.ANALYSIS_STARTED, {"name": self.title})
        self._is_running = True
        self._cancel_requested = False
        self._show_loading()
        self._run_analysis()
    
    def _run_analysis(self):
        """Run the analysis. Override in subclass."""
        # Emit completed when done (subclasses should call this or emit their own)
        self._is_running = False
        event_bus.emit(EventBus.ANALYSIS_COMPLETED, {"name": self.title})
    
    def _cancel_analysis(self):
        """Cancel the running analysis."""
        self._cancel_requested = True
        self._is_running = False
        if self._current_task:
            self._current_task.cancel()
        event_bus.emit(EventBus.ANALYSIS_CANCELLED, {"name": self.title})
    
    def _is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancel_requested
    
    def _subscribe_events(self):
        """Subscribe to events. Override to add more."""
        event_bus.subscribe(EventBus.DATASET_LOADED, self._on_dataset_loaded)
    
    def _on_dataset_loaded(self, data):
        """Handle dataset loaded event."""
        if isinstance(data, dict):
            self.bib = data.get("bib")
    
    def set_bib(self, bib):
        """Set the BiblioAnalysis instance."""
        self.bib = bib
    
    def destroy(self):
        """Clean up before destroying panel."""
        # Close any open matplotlib figures to prevent thread issues
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass
        
        # Clear any image references to prevent GC issues
        try:
            if hasattr(self, '_photo_image'):
                self._photo_image = None
            if hasattr(self, '_photo_images'):
                self._photo_images = []
        except:
            pass
        
        # Cancel any running tasks
        if self._current_task:
            try:
                self._current_task.cancel()
            except:
                pass
        
        # Call parent destroy
        super().destroy()
    
    # =========================================================================
    # LLM CONFIGURATION (v2.11)
    # =========================================================================
    
    def _create_llm_config_card(self):
        """Create LLM/AI configuration card in options panel."""
        try:
            self._llm_card = CollapsibleCard(
                self.options_content,
                title="🤖 AI Analysis Settings",
                collapsed=True,
                theme=self.theme_name,
            )
            self._llm_card.pack(fill=tk.X, padx=8, pady=8)
            
            # Enable checkbox
            enable_frame = tk.Frame(self._llm_card.content, bg=self.theme["bg_card"])
            enable_frame.pack(fill=tk.X, pady=2)
            
            self._llm_enabled_var = tk.BooleanVar(value=True)
            tk.Checkbutton(
                enable_frame,
                text="Enable AI descriptions",
                variable=self._llm_enabled_var,
                bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
                selectcolor=self.theme["bg_secondary"],
                command=self._sync_llm_settings,
            ).pack(anchor="w")
            
            # Provider
            prov_frame = tk.Frame(self._llm_card.content, bg=self.theme["bg_card"])
            prov_frame.pack(fill=tk.X, pady=2)
            tk.Label(prov_frame, text="Provider:", bg=self.theme["bg_card"], fg=self.theme["text_primary"], width=12, anchor="w").pack(side=tk.LEFT)
            
            self._llm_provider_var = tk.StringVar(value="openai")
            self._llm_provider_combo = ttk.Combobox(prov_frame, textvariable=self._llm_provider_var,
                                                     values=["huggingface", "openai", "anthropic"], state="readonly", width=18)
            self._llm_provider_combo.pack(side=tk.LEFT, padx=4)
            self._llm_provider_combo.bind("<<ComboboxSelected>>", self._on_llm_provider_change)
            
            # Model
            model_frame = tk.Frame(self._llm_card.content, bg=self.theme["bg_card"])
            model_frame.pack(fill=tk.X, pady=2)
            tk.Label(model_frame, text="Model:", bg=self.theme["bg_card"], fg=self.theme["text_primary"], width=12, anchor="w").pack(side=tk.LEFT)
            
            self._llm_models = {
                "huggingface": ["google/gemma-2-2b-it", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
                "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                "anthropic": ["claude-3-haiku-20240307", "claude-3-sonnet-20240229"],
            }
            self._llm_model_var = tk.StringVar(value="gpt-4o-mini")
            self._llm_model_combo = ttk.Combobox(model_frame, textvariable=self._llm_model_var,
                                                  values=self._llm_models["openai"], state="readonly", width=18)
            self._llm_model_combo.pack(side=tk.LEFT, padx=4)
            self._llm_model_combo.bind("<<ComboboxSelected>>", lambda e: self._sync_llm_settings())
            
            # API Key
            key_frame = tk.Frame(self._llm_card.content, bg=self.theme["bg_card"])
            key_frame.pack(fill=tk.X, pady=2)
            tk.Label(key_frame, text="API Key:", bg=self.theme["bg_card"], fg=self.theme["text_primary"], width=12, anchor="w").pack(side=tk.LEFT)
            
            self._llm_api_key_var = tk.StringVar()
            self._llm_api_key_entry = tk.Entry(key_frame, textvariable=self._llm_api_key_var, width=20, show="*")
            self._llm_api_key_entry.pack(side=tk.LEFT, padx=4)
            self._llm_api_key_var.trace("w", lambda *args: self._sync_llm_settings())
            
            # Note
            tk.Label(
                self._llm_card.content,
                text="After configuring, click '🤖 AI Describe' on any table.",
                font=("Segoe UI", 8),
                bg=self.theme["bg_card"],
                fg=self.theme["text_muted"],
            ).pack(fill=tk.X, pady=(4, 0))
            
        except Exception as e:
            print(f"Error creating LLM config card: {e}")
    
    def _on_llm_provider_change(self, event=None):
        """Update model options when provider changes."""
        try:
            provider = self._llm_provider_var.get()
            models = self._llm_models.get(provider, [])
            self._llm_model_combo["values"] = models
            if models:
                self._llm_model_var.set(models[0])
            self._sync_llm_settings()
        except:
            pass
    
    def _sync_llm_settings(self):
        """Sync LLM settings to DataTable class for all tables to use."""
        try:
            from biblium.gui.widgets.tables import DataTable
            
            enabled = self._llm_enabled_var.get() if hasattr(self, '_llm_enabled_var') else False
            provider = self._llm_provider_var.get() if hasattr(self, '_llm_provider_var') else "openai"
            model = self._llm_model_var.get() if hasattr(self, '_llm_model_var') else "gpt-4o-mini"
            api_key = self._llm_api_key_var.get() if hasattr(self, '_llm_api_key_var') else None
            
            DataTable.set_llm_settings(
                enabled=enabled,
                provider=provider,
                model=model,
                api_key=api_key or None,
            )
        except Exception as e:
            print(f"Error syncing LLM settings: {e}")
