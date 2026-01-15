# -*- coding: utf-8 -*-
"""
Settings Panel
==============
Application settings as a regular panel.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

from biblium.gui.config import FONTS, get_theme
from biblium.gui.widgets.cards import Card
from biblium.gui.widgets.buttons import ThemedButton


class SettingsPanel(tk.Frame):
    """
    Settings panel for configuring application preferences.
    """
    
    title: str = "Settings"
    icon: str = "⚙️"
    requires_data: bool = False
    
    def __init__(self, parent, theme: str = "light", bib=None, state=None, **kwargs):
        self.theme_name = theme
        self.theme = get_theme(theme)
        self.bib = bib
        self.state = state
        
        super().__init__(parent, bg=self.theme["bg_secondary"], **kwargs)
        
        self._create_layout()
    
    def _create_layout(self):
        """Create the settings layout."""
        # Main scrollable container
        self._canvas = tk.Canvas(self, bg=self.theme["bg_secondary"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        
        self.main_frame = tk.Frame(self._canvas, bg=self.theme["bg_secondary"])
        
        self.main_frame.bind("<Configure>", lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.create_window((0, 0), window=self.main_frame, anchor="nw")
        self._canvas.configure(yscrollcommand=scrollbar.set)
        
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Mouse wheel scrolling - only when mouse is over this panel
        def _on_mousewheel(event):
            self._canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_mousewheel(event):
            self._canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_mousewheel(event):
            self._canvas.unbind_all("<MouseWheel>")
        
        self._canvas.bind("<Enter>", _bind_mousewheel)
        self._canvas.bind("<Leave>", _unbind_mousewheel)
        self.main_frame.bind("<Enter>", _bind_mousewheel)
        self.main_frame.bind("<Leave>", _unbind_mousewheel)
        
        # Content area with max width
        content = tk.Frame(self.main_frame, bg=self.theme["bg_secondary"])
        content.pack(fill=tk.BOTH, expand=True, padx=40, pady=20)
        
        # Title
        tk.Label(
            content, text="⚙️ Settings",
            font=("Segoe UI", 20, "bold"),
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
        ).pack(anchor=tk.W, pady=(0, 20))
        
        # ============ AI Settings Card ============
        self._create_ai_settings(content)
        
        # ============ Paths Card ============
        self._create_paths_settings(content)
        
        # ============ Analysis Defaults Card ============
        self._create_analysis_settings(content)
        
        # ============ Save Button ============
        btn_frame = tk.Frame(content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, pady=(20, 40))
        
        save_btn = ThemedButton(
            btn_frame, text="💾 Save All Settings",
            command=self._save_settings,
            style="primary",
            theme=self.theme_name,
        )
        save_btn.pack(side=tk.LEFT)
        
        # Status label
        self._status_label = tk.Label(
            btn_frame, text="",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_secondary"], fg=self.theme["success"],
        )
        self._status_label.pack(side=tk.LEFT, padx=20)
    
    def _create_ai_settings(self, parent):
        """Create AI settings card."""
        card = Card(parent, title="🤖 AI Analysis Settings", theme=self.theme_name)
        card.pack(fill=tk.X, pady=(0, 16))
        
        tk.Label(
            card.content,
            text="Configure AI for automatic table and plot descriptions",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"], fg=self.theme["text_muted"],
        ).pack(anchor=tk.W, pady=(0, 12))
        
        # Provider
        provider_frame = tk.Frame(card.content, bg=self.theme["bg_card"])
        provider_frame.pack(fill=tk.X, pady=6)
        
        tk.Label(
            provider_frame, text="Provider:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
            width=18, anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        self._ai_provider_var = tk.StringVar(value=self._get_state("ai_provider", "openai"))
        provider_combo = ttk.Combobox(
            provider_frame, textvariable=self._ai_provider_var,
            values=["openai", "anthropic", "google", "huggingface"],
            state="readonly", width=30,
        )
        provider_combo.pack(side=tk.LEFT)
        
        # Model
        model_frame = tk.Frame(card.content, bg=self.theme["bg_card"])
        model_frame.pack(fill=tk.X, pady=6)
        
        tk.Label(
            model_frame, text="Model:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
            width=18, anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        self._ai_model_var = tk.StringVar(value=self._get_state("ai_model", "gpt-4o-mini"))
        model_entry = tk.Entry(
            model_frame, textvariable=self._ai_model_var,
            font=FONTS.get_font("body"), width=32,
        )
        model_entry.pack(side=tk.LEFT)
        
        # API Key
        key_frame = tk.Frame(card.content, bg=self.theme["bg_card"])
        key_frame.pack(fill=tk.X, pady=6)
        
        tk.Label(
            key_frame, text="API Key:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
            width=18, anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        self._ai_key_var = tk.StringVar(value=self._get_state("ai_api_key", ""))
        self._ai_key_entry = tk.Entry(
            key_frame, textvariable=self._ai_key_var,
            font=FONTS.get_font("body"), width=28, show="*",
        )
        self._ai_key_entry.pack(side=tk.LEFT)
        
        def toggle_key():
            if self._ai_key_entry.cget('show') == '*':
                self._ai_key_entry.config(show='')
                show_btn.config(text='🙈 Hide')
            else:
                self._ai_key_entry.config(show='*')
                show_btn.config(text='👁 Show')
        
        show_btn = tk.Button(
            key_frame, text="👁 Show", font=FONTS.get_font("small"),
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
            relief=tk.FLAT, command=toggle_key, cursor="hand2",
        )
        show_btn.pack(side=tk.LEFT, padx=(8, 0))
        
        # Test Connection button and status
        test_frame = tk.Frame(card.content, bg=self.theme["bg_card"])
        test_frame.pack(fill=tk.X, pady=(8, 4))
        
        self._test_btn = tk.Button(
            test_frame, text="🔗 Test Connection",
            font=FONTS.get_font("body"),
            bg=self.theme["accent_primary"], fg="white",
            relief=tk.FLAT, cursor="hand2",
            command=self._test_ai_connection,
            padx=12, pady=4,
        )
        self._test_btn.pack(side=tk.LEFT)
        
        self._test_status = tk.Label(
            test_frame, text="",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"], fg=self.theme["text_muted"],
        )
        self._test_status.pack(side=tk.LEFT, padx=(12, 0))
        
        # Custom Prompt
        tk.Label(
            card.content, text="Custom Prompt (optional):",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
        ).pack(anchor=tk.W, pady=(12, 4))
        
        tk.Label(
            card.content,
            text="Use {data} as placeholder for table/chart data. Leave empty for default prompt.",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"], fg=self.theme["text_muted"],
        ).pack(anchor=tk.W, pady=(0, 6))
        
        self._ai_prompt_text = tk.Text(
            card.content, font=FONTS.get_font("body"),
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
            height=4, wrap=tk.WORD, relief=tk.FLAT,
        )
        self._ai_prompt_text.pack(fill=tk.X, pady=(0, 8))
        self._ai_prompt_text.insert("1.0", self._get_state("ai_custom_prompt", ""))
    
    def _create_paths_settings(self, parent):
        """Create paths settings card."""
        card = Card(parent, title="📁 Paths", theme=self.theme_name)
        card.pack(fill=tk.X, pady=(0, 16))
        
        # Results folder
        folder_frame = tk.Frame(card.content, bg=self.theme["bg_card"])
        folder_frame.pack(fill=tk.X, pady=6)
        
        tk.Label(
            folder_frame, text="Results Folder:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
            width=18, anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        self._results_folder_var = tk.StringVar(value=self._get_state("results_folder", "results"))
        folder_entry = tk.Entry(
            folder_frame, textvariable=self._results_folder_var,
            font=FONTS.get_font("body"), width=25,
        )
        folder_entry.pack(side=tk.LEFT)
        
        def browse_folder():
            folder = filedialog.askdirectory(title="Select Results Folder")
            if folder:
                self._results_folder_var.set(folder)
        
        browse_btn = tk.Button(
            folder_frame, text="📂 Browse", font=FONTS.get_font("small"),
            bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
            relief=tk.FLAT, command=browse_folder, cursor="hand2",
        )
        browse_btn.pack(side=tk.LEFT, padx=(8, 0))
    
    def _create_analysis_settings(self, parent):
        """Create analysis defaults settings card."""
        card = Card(parent, title="📊 Analysis Defaults", theme=self.theme_name)
        card.pack(fill=tk.X, pady=(0, 16))
        
        # Default Top N
        topn_frame = tk.Frame(card.content, bg=self.theme["bg_card"])
        topn_frame.pack(fill=tk.X, pady=6)
        
        tk.Label(
            topn_frame, text="Default Top N:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
            width=18, anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        self._topn_var = tk.StringVar(value=str(self._get_state("default_top_n", 50)))
        topn_spin = tk.Spinbox(
            topn_frame, textvariable=self._topn_var,
            from_=5, to=500, width=10, font=FONTS.get_font("body"),
        )
        topn_spin.pack(side=tk.LEFT)
        
        # Default DPI
        dpi_frame = tk.Frame(card.content, bg=self.theme["bg_card"])
        dpi_frame.pack(fill=tk.X, pady=6)
        
        tk.Label(
            dpi_frame, text="Plot Export DPI:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
            width=18, anchor=tk.W,
        ).pack(side=tk.LEFT)
        
        self._dpi_var = tk.StringVar(value=str(self._get_state("default_dpi", 600)))
        dpi_spin = tk.Spinbox(
            dpi_frame, textvariable=self._dpi_var,
            from_=72, to=1200, width=10, font=FONTS.get_font("body"),
        )
        dpi_spin.pack(side=tk.LEFT)
    
    def _get_state(self, key: str, default=None):
        """Get value from state settings."""
        if self.state and hasattr(self.state, 'settings'):
            return getattr(self.state.settings, key, default)
        return default
    
    def _test_ai_connection(self):
        """Test the AI connection with current settings."""
        api_key = self._ai_key_var.get().strip()
        provider = self._ai_provider_var.get()
        model = self._ai_model_var.get().strip()
        
        if not api_key:
            self._test_status.config(text="❌ Please enter an API key", fg=self.theme["danger"])
            return
        
        if not model:
            self._test_status.config(text="❌ Please enter a model name", fg=self.theme["danger"])
            return
        
        # Update button to show testing
        self._test_btn.config(text="⏳ Testing...", state=tk.DISABLED)
        self._test_status.config(text="", fg=self.theme["text_muted"])
        
        # Run test in background thread
        import threading
        def do_test():
            try:
                from biblium.llm_utils import invoke_llm
                
                # Simple test prompt
                result = invoke_llm(
                    "Say 'OK' if you can read this.",
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    max_tokens=10,
                    use_cache=False,
                )
                
                # Success - also save settings
                self.after(0, lambda: self._on_test_success(model))
                
            except Exception as e:
                error_msg = str(e)
                if len(error_msg) > 50:
                    error_msg = error_msg[:50] + "..."
                self.after(0, lambda msg=error_msg: self._show_test_result(False, f"❌ {msg}"))
        
        thread = threading.Thread(target=do_test, daemon=True)
        thread.start()
    
    def _on_test_success(self, model: str):
        """Handle successful test - save and apply settings."""
        self._test_btn.config(text="🔗 Test Connection", state=tk.NORMAL)
        self._test_status.config(text=f"✓ Connected & Saved!", fg=self.theme["success"])
        
        # Auto-save the AI settings
        try:
            # Apply to DataTable immediately
            from biblium.gui.widgets.tables import DataTable
            api_key = self._ai_key_var.get().strip()
            provider = self._ai_provider_var.get()
            custom_prompt = self._ai_prompt_text.get("1.0", tk.END).strip()
            
            DataTable.set_llm_settings(
                enabled=True,
                provider=provider,
                model=model,
                api_key=api_key,
                custom_prompt=custom_prompt,
            )
            
            # Save to state if available
            if self.state:
                self.state.update_settings(
                    ai_provider=provider,
                    ai_model=model,
                    ai_api_key=api_key,
                    ai_custom_prompt=custom_prompt,
                )
        except Exception as e:
            print(f"Warning: Could not save AI settings: {e}")
    
    def _show_test_result(self, success: bool, message: str):
        """Show test connection result."""
        self._test_btn.config(text="🔗 Test Connection", state=tk.NORMAL)
        color = self.theme["success"] if success else self.theme["danger"]
        self._test_status.config(text=message, fg=color)
    
    def _save_settings(self):
        """Save all settings."""
        if not self.state:
            messagebox.showerror("Error", "State manager not available")
            return
        
        try:
            # Update settings via state manager
            self.state.update_settings(
                ai_provider=self._ai_provider_var.get(),
                ai_model=self._ai_model_var.get(),
                ai_api_key=self._ai_key_var.get(),
                ai_custom_prompt=self._ai_prompt_text.get("1.0", tk.END).strip(),
                results_folder=self._results_folder_var.get(),
                default_top_n=int(self._topn_var.get()),
                plot_dpi=int(self._dpi_var.get()),
            )
            
            # Apply AI settings to DataTable
            from biblium.gui.widgets.tables import DataTable
            api_key = self._ai_key_var.get()
            DataTable.set_llm_settings(
                enabled=bool(api_key),
                provider=self._ai_provider_var.get(),
                model=self._ai_model_var.get(),
                api_key=api_key,
                custom_prompt=self._ai_prompt_text.get("1.0", tk.END).strip(),
            )
            
            # Show success
            self._status_label.config(text="✓ Settings saved!", fg=self.theme["success"])
            self.after(3000, lambda: self._status_label.config(text=""))
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
    
    def set_bib(self, bib):
        """Update bib reference."""
        self.bib = bib
    
    def set_state(self, state):
        """Update state reference and refresh values."""
        self.state = state
        self._refresh_values()
    
    def _refresh_values(self):
        """Refresh all values from state."""
        if not self.state:
            return
        
        self._ai_provider_var.set(self._get_state("ai_provider", "openai"))
        self._ai_model_var.set(self._get_state("ai_model", "gpt-4o-mini"))
        self._ai_key_var.set(self._get_state("ai_api_key", ""))
        
        # Clear and re-insert prompt text
        self._ai_prompt_text.delete("1.0", tk.END)
        self._ai_prompt_text.insert("1.0", self._get_state("ai_custom_prompt", ""))
        
        self._results_folder_var.set(self._get_state("results_folder", "results"))
        self._topn_var.set(str(self._get_state("default_top_n", 50)))
        self._dpi_var.set(str(self._get_state("plot_dpi", 600)))
