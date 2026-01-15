# -*- coding: utf-8 -*-
"""
Table Widgets
=============
Data table with sorting, filtering, and export.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Any, Callable, Dict, List, Optional, Tuple

from biblium.gui.config import FONTS, LAYOUT, get_theme

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class DataTable(tk.Frame):
    """
    A data table widget for displaying DataFrames.
    
    Features:
    - Sortable columns
    - Row selection
    - Horizontal and vertical scrolling
    - Copy to clipboard
    - Right-click export to XLSX, CSV, TXT
    - AI-powered description (v2.11)
    
    Usage:
        table = DataTable(parent)
        table.set_data(df)
        selected = table.get_selected_rows()
    """
    
    # Shared LLM settings across all DataTable instances
    _llm_settings = {
        "enabled": False,
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": None,
        "custom_prompt": "",  # Custom prompt template
    }
    
    @classmethod
    def set_llm_settings(cls, enabled: bool = True, provider: str = "openai", 
                         model: str = "gpt-4o-mini", api_key: str = None,
                         custom_prompt: str = ""):
        """Set LLM settings for all DataTable instances."""
        cls._llm_settings = {
            "enabled": enabled,
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "custom_prompt": custom_prompt,
        }
    
    @classmethod
    def get_llm_settings(cls):
        """Get current LLM settings."""
        return cls._llm_settings.copy()
    
    def __init__(
        self,
        parent,
        theme: str = "light",
        show_index: bool = False,
        max_rows: int = 500,
        on_select: Optional[Callable] = None,
        on_double_click: Optional[Callable] = None,
        dataframe: Optional[Any] = None,
        show_ai_button: bool = True,  # v2.11: show AI description button
        **kwargs
    ):
        self.theme = get_theme(theme)
        self.theme_name = theme
        self.show_index = show_index
        self.max_rows = max_rows
        self.on_select = on_select
        self.on_double_click = on_double_click
        self.show_ai_button = show_ai_button
        
        self._data = None
        self._sort_column = None
        self._sort_reverse = False
        self._ai_description = None
        
        super().__init__(parent, bg=self.theme["bg_card"], **kwargs)
        
        self._create_widgets()
        
        # If dataframe provided at init, set it
        if dataframe is not None:
            self.set_data(dataframe)
    
    def _create_widgets(self):
        """Create table widgets."""
        # Toolbar
        self.toolbar = tk.Frame(self, bg=self.theme["bg_card"])
        self.toolbar.pack(fill=tk.X, padx=8, pady=(8, 4))
        
        # Search
        tk.Label(
            self.toolbar,
            text="🔍",
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(side=tk.LEFT)
        
        self.search_var = tk.StringVar()
        self.search_var.trace("w", lambda *args: self._filter_data())
        self.search_entry = tk.Entry(
            self.toolbar,
            textvariable=self.search_var,
            font=FONTS.get_font("body"),
            width=20,
            relief=tk.FLAT,
            bg=self.theme["bg_input"],
            fg=self.theme["text_primary"],
            highlightbackground=self.theme["border"],
            highlightthickness=1,
        )
        self.search_entry.pack(side=tk.LEFT, padx=(4, 8))
        
        # Export button in toolbar
        export_btn = tk.Button(
            self.toolbar,
            text="💾 Export",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
            relief=tk.FLAT,
            command=self._show_export_menu_button,
            cursor="hand2",
        )
        export_btn.pack(side=tk.LEFT, padx=4)
        
        # Add to Report button
        report_btn = tk.Button(
            self.toolbar,
            text="📄 Report",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["accent_primary"],
            relief=tk.FLAT,
            command=self._add_to_report,
            cursor="hand2",
        )
        report_btn.pack(side=tk.LEFT, padx=4)
        
        # AI Description button (v2.11)
        if self.show_ai_button:
            self._ai_btn = tk.Button(
                self.toolbar,
                text="🤖 AI Describe",
                font=FONTS.get_font("small"),
                bg=self.theme["bg_card"],
                fg=self.theme["accent_primary"],
                relief=tk.FLAT,
                command=self._generate_ai_description,
                cursor="hand2",
            )
            self._ai_btn.pack(side=tk.LEFT, padx=4)
        
        # Info label
        self.info_label = tk.Label(
            self.toolbar,
            text="",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        )
        self.info_label.pack(side=tk.RIGHT)
        
        # Table frame with scrollbars
        table_frame = tk.Frame(self, bg=self.theme["bg_card"])
        table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        
        # Scrollbars
        self.v_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)
        self.h_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        
        # Treeview
        self.tree = ttk.Treeview(
            table_frame,
            show="headings",
            yscrollcommand=self.v_scroll.set,
            xscrollcommand=self.h_scroll.set,
        )
        
        self.v_scroll.config(command=self.tree.yview)
        self.h_scroll.config(command=self.tree.xview)
        
        # Layout
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Bindings
        self.tree.bind("<Double-1>", self._on_double_click)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Control-c>", self._copy_selection)
        self.tree.bind("<Button-3>", self._show_context_menu)
        
        # Style
        style = ttk.Style()
        style.configure("Treeview", font=FONTS.get_font("body"), rowheight=26)
        style.configure("Treeview.Heading", font=FONTS.get_font("body"))
    
    def _show_context_menu(self, event):
        """Show right-click context menu for export."""
        if self._data is None:
            return
        
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="📄 Add to Report", command=self._add_to_report)
        menu.add_separator()
        menu.add_command(label="📊 Export to Excel (.xlsx)", command=lambda: self._export_data("xlsx"))
        menu.add_command(label="📄 Export to CSV (.csv)", command=lambda: self._export_data("csv"))
        menu.add_command(label="📝 Export to Text (.txt)", command=lambda: self._export_data("txt"))
        menu.add_separator()
        menu.add_command(label="📋 Copy All to Clipboard", command=self._copy_all)
        menu.add_command(label="📋 Copy Selected to Clipboard", command=lambda: self._copy_selection(None))
        
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
    
    def _add_to_report(self):
        """Add the current table to the report queue."""
        if self._data is None:
            messagebox.showinfo("No Data", "No data to add to report.")
            return
        
        try:
            from biblium.gui.core.state import report_queue
            
            # Get a title for the table
            title = self._get_table_title()
            
            # Get source panel name
            source_panel = self._get_source_panel()
            
            # Add to queue
            report_queue.add_table(
                dataframe=self._data,
                title=title,
                source_panel=source_panel,
            )
            
            # Show confirmation
            messagebox.showinfo(
                "Added to Report",
                f"Table '{title}' has been added to the report queue.\n\n"
                f"Items in queue: {len(report_queue)}\n\n"
                "Go to Reports → Report Builder to generate your report."
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add to report: {e}")
    
    def _get_table_title(self) -> str:
        """Get a title for the table based on context."""
        # Try to get title from various sources
        if hasattr(self, '_table_title') and self._table_title:
            return self._table_title
        
        # Try to get from parent widget hierarchy
        parent = self.master
        while parent:
            if hasattr(parent, 'title') and isinstance(parent.title, str):
                return f"{parent.title} - Table"
            parent = getattr(parent, 'master', None)
        
        # Default title
        if HAS_PANDAS and isinstance(self._data, pd.DataFrame):
            cols = ", ".join(self._data.columns[:3])
            return f"Table ({cols}...)"
        return "Data Table"
    
    def _get_source_panel(self) -> str:
        """Get the name of the panel that contains this table."""
        parent = self.master
        while parent:
            if hasattr(parent, 'title') and isinstance(parent.title, str):
                return parent.title
            if hasattr(parent, '__class__') and 'Panel' in parent.__class__.__name__:
                return parent.__class__.__name__
            parent = getattr(parent, 'master', None)
        return "Unknown Panel"
    
    def set_table_title(self, title: str):
        """Set a custom title for this table (used when adding to report)."""
        self._table_title = title
    
    def _show_export_menu_button(self):
        """Show export menu from button click."""
        if self._data is None:
            messagebox.showinfo("No Data", "No data to export.")
            return
        
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="📊 Export to Excel (.xlsx)", command=lambda: self._export_data("xlsx"))
        menu.add_command(label="📄 Export to CSV (.csv)", command=lambda: self._export_data("csv"))
        menu.add_command(label="📝 Export to Text (.txt)", command=lambda: self._export_data("txt"))
        
        # Get button position
        try:
            menu.tk_popup(self.toolbar.winfo_rootx() + 100, self.toolbar.winfo_rooty() + 30)
        finally:
            menu.grab_release()
    
    def _export_data(self, format: str):
        """Export table data to file."""
        if self._data is None:
            messagebox.showinfo("No Data", "No data to export.")
            return
        
        # Get the data as DataFrame
        if HAS_PANDAS and isinstance(self._data, pd.DataFrame):
            df = self._data
        elif HAS_PANDAS:
            # Convert to DataFrame
            try:
                df = pd.DataFrame(self._data)
            except Exception as e:
                messagebox.showerror("Export Error", f"Could not convert data: {e}")
                return
        else:
            messagebox.showerror("Export Error", "pandas is required for export functionality.")
            return
        
        # File dialog
        filetypes = {
            "xlsx": [("Excel File", "*.xlsx")],
            "csv": [("CSV File", "*.csv")],
            "txt": [("Text File", "*.txt")],
        }
        
        extensions = {
            "xlsx": ".xlsx",
            "csv": ".csv",
            "txt": ".txt",
        }
        
        filename = filedialog.asksaveasfilename(
            defaultextension=extensions.get(format, ".csv"),
            filetypes=filetypes.get(format, [("All Files", "*.*")]),
            title=f"Export Table as {format.upper()}",
        )
        
        if not filename:
            return
        
        try:
            if format == "xlsx":
                df.to_excel(filename, index=self.show_index, engine='openpyxl')
            elif format == "csv":
                df.to_csv(filename, index=self.show_index)
            elif format == "txt":
                df.to_csv(filename, index=self.show_index, sep='\t')
            
            messagebox.showinfo("Export Successful", f"Data exported to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data:\n{e}")
    
    def _copy_all(self):
        """Copy all data to clipboard."""
        if self._data is None:
            return
        
        try:
            columns = list(self.tree["columns"])
            lines = ["\t".join(columns)]
            
            for item in self.tree.get_children(""):
                values = self.tree.item(item)["values"]
                lines.append("\t".join(str(v) for v in values))
            
            text = "\n".join(lines)
            self.clipboard_clear()
            self.clipboard_append(text)
            
            # Show brief notification
            self.info_label.configure(text="✓ Copied to clipboard!")
            self.after(2000, lambda: self._update_info_label())
        except Exception as e:
            messagebox.showerror("Copy Error", f"Failed to copy: {e}")
    
    def _update_info_label(self):
        """Update info label with row count."""
        if self._data is not None:
            total = len(self._data) if hasattr(self._data, '__len__') else 0
            shown = len(self.tree.get_children(""))
            self.info_label.configure(text=f"Showing {shown:,} of {total:,} rows")
        else:
            self.info_label.configure(text="No data")
    
    def export_to_file(self, filename: str, format: str = None):
        """
        Export data to file programmatically.
        
        Parameters
        ----------
        filename : str
            Output file path.
        format : str, optional
            Format ('xlsx', 'csv', 'txt'). Inferred from extension if not provided.
        """
        if self._data is None:
            return False
        
        if format is None:
            ext = filename.lower().split('.')[-1]
            format = ext if ext in ('xlsx', 'csv', 'txt') else 'csv'
        
        if not HAS_PANDAS:
            return False
        
        try:
            if isinstance(self._data, pd.DataFrame):
                df = self._data
            else:
                df = pd.DataFrame(self._data)
            
            if format == "xlsx":
                df.to_excel(filename, index=self.show_index, engine='openpyxl')
            elif format == "csv":
                df.to_csv(filename, index=self.show_index)
            elif format == "txt":
                df.to_csv(filename, index=self.show_index, sep='\t')
            
            return True
        except Exception:
            return False
    
    def set_data(self, data, columns: List[str] = None):
        """
        Set the table data.
        
        Parameters
        ----------
        data : DataFrame or list of dicts
            Data to display.
        columns : list, optional
            Column names (inferred from data if not provided).
        """
        # Clear existing data
        self.tree.delete(*self.tree.get_children())
        
        if data is None:
            self._data = None
            self.info_label.configure(text="No data")
            return
        
        # Convert to list of dicts if DataFrame
        if HAS_PANDAS and hasattr(data, 'to_dict'):
            self._data = data
            columns = columns or list(data.columns)
            rows = data.head(self.max_rows).to_dict('records')
        else:
            self._data = data
            rows = data[:self.max_rows] if isinstance(data, list) else []
            columns = columns or (list(rows[0].keys()) if rows else [])
        
        # Configure columns
        self.tree["columns"] = columns
        for col in columns:
            self.tree.heading(
                col,
                text=col,
                command=lambda c=col: self._sort_by_column(c),
            )
            # Calculate column width
            max_width = max(
                len(str(col)) * 10,
                max((len(str(row.get(col, ""))) * 8 for row in rows[:100]), default=50)
            )
            max_width = min(max(max_width, 80), 300)
            self.tree.column(col, width=max_width, minwidth=60)
        
        # Insert rows
        for row in rows:
            values = [str(row.get(col, ""))[:100] for col in columns]
            self.tree.insert("", tk.END, values=values)
        
        # Update info
        total = len(self._data) if HAS_PANDAS and hasattr(self._data, '__len__') else len(rows)
        shown = min(total, self.max_rows)
        self.info_label.configure(text=f"Showing {shown:,} of {total:,} rows")
    
    def _sort_by_column(self, column: str):
        """Sort table by column."""
        if self._sort_column == column:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = column
            self._sort_reverse = False
        
        # Get all items
        items = [(self.tree.set(item, column), item) for item in self.tree.get_children("")]
        
        # Sort
        try:
            items.sort(key=lambda x: float(x[0]) if x[0].replace(".", "").replace("-", "").isdigit() else x[0].lower(),
                      reverse=self._sort_reverse)
        except:
            items.sort(key=lambda x: x[0].lower(), reverse=self._sort_reverse)
        
        # Rearrange
        for index, (_, item) in enumerate(items):
            self.tree.move(item, "", index)
        
        # Update heading
        arrow = " ▼" if self._sort_reverse else " ▲"
        for col in self.tree["columns"]:
            text = col + (arrow if col == column else "")
            self.tree.heading(col, text=text)
    
    def _filter_data(self):
        """Filter table by search term."""
        search = self.search_var.get().lower()
        
        if not search or self._data is None:
            # Reset to original data
            if self._data is not None:
                self.set_data(self._data)
            return
        
        # Filter
        if HAS_PANDAS and hasattr(self._data, 'apply'):
            mask = self._data.apply(
                lambda row: any(search in str(v).lower() for v in row),
                axis=1
            )
            filtered = self._data[mask]
        else:
            filtered = [
                row for row in self._data
                if any(search in str(v).lower() for v in row.values())
            ]
        
        # Update display without changing self._data
        self.tree.delete(*self.tree.get_children())
        
        if HAS_PANDAS and hasattr(filtered, 'to_dict'):
            rows = filtered.head(self.max_rows).to_dict('records')
            columns = list(filtered.columns)
        else:
            rows = filtered[:self.max_rows]
            columns = list(rows[0].keys()) if rows else []
        
        for row in rows:
            values = [str(row.get(col, ""))[:100] for col in columns]
            self.tree.insert("", tk.END, values=values)
        
        total = len(filtered) if hasattr(filtered, '__len__') else len(rows)
        self.info_label.configure(text=f"Showing {min(total, self.max_rows):,} of {total:,} (filtered)")
    
    def _on_select(self, event):
        """Handle row selection."""
        if self.on_select:
            self.on_select(self.get_selected_rows())
    
    def _on_double_click(self, event):
        """Handle double click."""
        if self.on_double_click:
            selected = self.get_selected_rows()
            if selected:
                self.on_double_click(selected[0])
    
    def _copy_selection(self, event):
        """Copy selected rows to clipboard."""
        selected = self.tree.selection()
        if not selected:
            return
        
        lines = []
        columns = self.tree["columns"]
        lines.append("\t".join(columns))
        
        for item in selected:
            values = self.tree.item(item)["values"]
            lines.append("\t".join(str(v) for v in values))
        
        text = "\n".join(lines)
        self.clipboard_clear()
        self.clipboard_append(text)
    
    def get_selected_rows(self) -> List[Dict]:
        """Get selected rows as list of dicts."""
        selected = []
        columns = self.tree["columns"]
        
        for item in self.tree.selection():
            values = self.tree.item(item)["values"]
            row = dict(zip(columns, values))
            selected.append(row)
        
        return selected
    
    def clear(self):
        """Clear table."""
        self.tree.delete(*self.tree.get_children())
        self._data = None
        self.info_label.configure(text="No data")
    
    def _generate_ai_description(self):
        """Generate AI description of the table data."""
        settings = DataTable._llm_settings
        
        if not settings.get("enabled") or not settings.get("api_key"):
            # Show message to configure in Settings
            messagebox.showinfo("Configure AI", 
                "Please configure your AI API key in Settings first.\n\n"
                "Go to Settings (⚙️) and enter your API key under 'AI Analysis Settings'.")
            return
        
        if self._data is None:
            messagebox.showinfo("No Data", "No data to describe.")
            return
        
        # Update button to show loading
        if hasattr(self, '_ai_btn'):
            self._ai_btn.config(text="⏳ Generating...", state=tk.DISABLED)
        
        # Run in background thread
        import threading
        thread = threading.Thread(target=self._do_ai_generation, daemon=True)
        thread.start()
    
    def _do_ai_generation(self):
        """Perform AI generation in background."""
        try:
            from biblium.llm_utils import llm_describe_table
            
            settings = DataTable._llm_settings
            
            result = llm_describe_table(
                self._data,
                provider=settings["provider"],
                model=settings["model"],
                api_key=settings["api_key"],
                custom_prompt=settings.get("custom_prompt", ""),
            )
            
            self._ai_description = result
            
            # Show result in main thread
            self.after(0, lambda r=result: self._show_ai_result(r))
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.after(0, lambda msg=error_msg: self._show_ai_result(msg))
        finally:
            self.after(0, self._reset_ai_button)
    
    def _reset_ai_button(self):
        """Reset AI button state."""
        if hasattr(self, '_ai_btn'):
            self._ai_btn.config(text="🤖 AI Describe", state=tk.NORMAL)
    
    def _show_ai_result(self, text: str):
        """Show AI description result inline below the table."""
        # Remove any existing result frame
        if hasattr(self, '_ai_result_frame') and self._ai_result_frame:
            try:
                self._ai_result_frame.destroy()
            except:
                pass
        
        # Create result frame below table
        self._ai_result_frame = tk.Frame(self, bg=self.theme["bg_card"])
        self._ai_result_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(4, 0))
        
        # Header with title and buttons
        header = tk.Frame(self._ai_result_frame, bg=self.theme["bg_card"])
        header.pack(fill=tk.X, padx=4)
        
        tk.Label(
            header,
            text="🤖 AI Description",
            font=FONTS.get_font("body", bold=True),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(side=tk.LEFT, padx=4)
        
        # Close button
        tk.Button(
            header,
            text="✕",
            font=("Segoe UI", 8),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_primary"],
            relief=tk.FLAT,
            command=lambda: self._ai_result_frame.destroy(),
            cursor="hand2",
            width=2,
        ).pack(side=tk.RIGHT, padx=2)
        
        # Copy button
        copy_btn = tk.Button(
            header,
            text="📋 Copy",
            font=("Segoe UI", 8),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_primary"],
            relief=tk.FLAT,
            command=lambda: self._copy_ai_text(text, copy_btn),
            cursor="hand2",
        )
        copy_btn.pack(side=tk.RIGHT, padx=2)
        
        # Text widget - selectable and copyable
        text_widget = tk.Text(
            self._ai_result_frame,
            wrap=tk.WORD,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_primary"],
            relief=tk.FLAT,
            height=4,
            padx=8,
            pady=4,
        )
        text_widget.pack(fill=tk.X, padx=4, pady=(2, 4))
        text_widget.insert("1.0", text)
        
        # Allow selection and copy but not editing
        def on_key(e):
            if e.state & 0x4 and e.keysym.lower() in ('c', 'a'):  # Ctrl+C, Ctrl+A
                return
            return "break"
        text_widget.bind("<Key>", on_key)
        text_widget.bind("<Button-1>", lambda e: text_widget.focus_set())
    
    def _copy_ai_text(self, text: str, btn=None):
        """Copy AI description to clipboard."""
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            if btn:
                btn.config(text="✓ Copied")
                self.after(1500, lambda: btn.config(text="📋 Copy"))
        except:
            pass
    
    def _show_ai_config_dialog(self):
        """Show dialog to configure AI settings."""
        dialog = tk.Toplevel(self)
        dialog.title("🤖 AI Configuration")
        dialog.geometry("400x280")
        dialog.configure(bg=self.theme["bg_card"])
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()
        
        # Instructions
        tk.Label(
            dialog,
            text="Configure AI Settings",
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"],
            fg=self.theme["text_primary"],
        ).pack(pady=(16, 8), padx=16, anchor="w")
        
        tk.Label(
            dialog,
            text="Enter your API key to enable AI descriptions.",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"],
            fg=self.theme["text_muted"],
        ).pack(pady=(0, 16), padx=16, anchor="w")
        
        # Provider
        prov_frame = tk.Frame(dialog, bg=self.theme["bg_card"])
        prov_frame.pack(fill=tk.X, padx=16, pady=4)
        tk.Label(prov_frame, text="Provider:", width=12, anchor="w", bg=self.theme["bg_card"]).pack(side=tk.LEFT)
        provider_var = tk.StringVar(value=DataTable._llm_settings.get("provider", "openai"))
        provider_combo = ttk.Combobox(prov_frame, textvariable=provider_var, 
                                       values=["huggingface", "openai", "anthropic"], state="readonly", width=20)
        provider_combo.pack(side=tk.LEFT, padx=4)
        
        # Model
        model_frame = tk.Frame(dialog, bg=self.theme["bg_card"])
        model_frame.pack(fill=tk.X, padx=16, pady=4)
        tk.Label(model_frame, text="Model:", width=12, anchor="w", bg=self.theme["bg_card"]).pack(side=tk.LEFT)
        model_var = tk.StringVar(value=DataTable._llm_settings.get("model", "gpt-4o-mini"))
        model_combo = ttk.Combobox(model_frame, textvariable=model_var, 
                                    values=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], state="readonly", width=20)
        model_combo.pack(side=tk.LEFT, padx=4)
        
        # API Key
        key_frame = tk.Frame(dialog, bg=self.theme["bg_card"])
        key_frame.pack(fill=tk.X, padx=16, pady=4)
        tk.Label(key_frame, text="API Key:", width=12, anchor="w", bg=self.theme["bg_card"]).pack(side=tk.LEFT)
        api_key_var = tk.StringVar(value=DataTable._llm_settings.get("api_key", "") or "")
        api_key_entry = tk.Entry(key_frame, textvariable=api_key_var, width=22, show="*")
        api_key_entry.pack(side=tk.LEFT, padx=4)
        
        def save_settings():
            DataTable.set_llm_settings(
                enabled=True,
                provider=provider_var.get(),
                model=model_var.get(),
                api_key=api_key_var.get() or None,
            )
            dialog.destroy()
            # Trigger generation if API key was provided
            if api_key_var.get():
                self._generate_ai_description()
        
        # Buttons
        btn_frame = tk.Frame(dialog, bg=self.theme["bg_card"])
        btn_frame.pack(fill=tk.X, padx=16, pady=16)
        
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Save & Generate", command=save_settings,
                  bg=self.theme["accent_primary"], fg="white").pack(side=tk.RIGHT)


class SortableTable(DataTable):
    """
    Alias for DataTable with sorting enabled by default.
    """
    pass


def add_table_context_menu(tree, dataframe=None, title=None, source_panel=None):
    """Add right-click context menu with export functionality to any ttk.Treeview.
    
    Parameters
    ----------
    tree : ttk.Treeview
        The treeview widget
    dataframe : pd.DataFrame, optional
        DataFrame to export (if not provided, data is extracted from tree)
    title : str, optional
        Title for the table when adding to report
    source_panel : str, optional
        Source panel name for report
    """
    tree._export_df = dataframe
    tree._table_title = title
    tree._source_panel = source_panel
    
    def _get_data():
        if hasattr(tree, '_export_df') and tree._export_df is not None:
            return tree._export_df
        columns = list(tree["columns"])
        if not columns:
            return None
        rows = []
        for item in tree.get_children(""):
            values = tree.item(item)["values"]
            if values:
                rows.append(dict(zip(columns, values)))
        if HAS_PANDAS and rows:
            return pd.DataFrame(rows)
        return None
    
    def _add_to_report():
        df = _get_data()
        if df is None or (hasattr(df, 'empty') and df.empty):
            messagebox.showinfo("No Data", "No data to add to report.")
            return
        
        try:
            from biblium.gui.core.state import report_queue
            
            # Get title
            tbl_title = getattr(tree, '_table_title', None) or "Data Table"
            panel = getattr(tree, '_source_panel', None) or "Unknown Panel"
            
            # Add to queue
            report_queue.add_table(
                dataframe=df,
                title=tbl_title,
                source_panel=panel,
            )
            
            messagebox.showinfo(
                "Added to Report",
                f"Table '{tbl_title}' has been added to the report queue.\n\n"
                f"Items in queue: {len(report_queue)}\n\n"
                "Go to Reports → Report Builder to generate your report."
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add to report: {e}")
    
    def _show_menu(event):
        menu = tk.Menu(tree, tearoff=0)
        menu.add_command(label="📄 Add to Report", command=_add_to_report)
        menu.add_separator()
        menu.add_command(label="📊 Export to Excel (.xlsx)", command=lambda: _export("xlsx"))
        menu.add_command(label="📄 Export to CSV (.csv)", command=lambda: _export("csv"))
        menu.add_command(label="📝 Export to Text (.txt)", command=lambda: _export("txt"))
        menu.add_separator()
        menu.add_command(label="📋 Copy All to Clipboard", command=_copy_all)
        menu.add_command(label="📋 Copy Selected", command=_copy_selected)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
    
    def _export(fmt):
        df = _get_data()
        if df is None or (hasattr(df, 'empty') and df.empty):
            messagebox.showinfo("No Data", "No data to export.")
            return
        exts = {"xlsx": ".xlsx", "csv": ".csv", "txt": ".txt"}
        ftypes = {"xlsx": [("Excel", "*.xlsx")], "csv": [("CSV", "*.csv")], "txt": [("Text", "*.txt")]}
        filename = filedialog.asksaveasfilename(defaultextension=exts.get(fmt), filetypes=ftypes.get(fmt))
        if not filename:
            return
        try:
            if fmt == "xlsx":
                df.to_excel(filename, index=False, engine='openpyxl')
            elif fmt == "csv":
                df.to_csv(filename, index=False)
            else:
                df.to_csv(filename, index=False, sep='\t')
            messagebox.showinfo("Success", f"Exported to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{e}")
    
    def _copy_all():
        try:
            cols = list(tree["columns"])
            lines = ["\t".join(str(c) for c in cols)]
            for item in tree.get_children(""):
                vals = tree.item(item)["values"]
                lines.append("\t".join(str(v) for v in vals))
            tree.clipboard_clear()
            tree.clipboard_append("\n".join(lines))
        except Exception as e:
            messagebox.showerror("Error", f"Copy failed: {e}")
    
    def _copy_selected():
        sel = tree.selection()
        if not sel:
            return
        try:
            cols = list(tree["columns"])
            lines = ["\t".join(str(c) for c in cols)]
            for item in sel:
                vals = tree.item(item)["values"]
                lines.append("\t".join(str(v) for v in vals))
            tree.clipboard_clear()
            tree.clipboard_append("\n".join(lines))
        except Exception as e:
            messagebox.showerror("Error", f"Copy failed: {e}")
    
    tree.bind("<Button-3>", _show_menu)
    return tree
