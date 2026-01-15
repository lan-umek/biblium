# -*- coding: utf-8 -*-
"""
Plot Widgets
============
Matplotlib plot containers with toolbar.
"""

import tkinter as tk
from tkinter import ttk, filedialog
from typing import Callable, Optional, Tuple

from biblium.gui.config import FONTS, LAYOUT, get_theme

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class PlotToolbar(tk.Frame):
    """
    Custom toolbar for plot controls.
    """
    
    def __init__(
        self,
        parent,
        on_save: Optional[Callable] = None,
        on_copy: Optional[Callable] = None,
        on_reset: Optional[Callable] = None,
        theme: str = "light",
        **kwargs
    ):
        self.theme = get_theme(theme)
        
        super().__init__(parent, bg=self.theme["bg_card"], **kwargs)
        
        from biblium.gui.widgets.buttons import IconButton
        
        # Save button
        IconButton(
            self, icon="💾", tooltip="Save plot",
            command=on_save, theme=theme,
        ).pack(side=tk.LEFT, padx=2)
        
        # Copy button
        IconButton(
            self, icon="📋", tooltip="Copy to clipboard",
            command=on_copy, theme=theme,
        ).pack(side=tk.LEFT, padx=2)
        
        # Separator
        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=4)
        
        # Reset button
        IconButton(
            self, icon="🏠", tooltip="Reset view",
            command=on_reset, theme=theme,
        ).pack(side=tk.LEFT, padx=2)


class PlotFrame(tk.Frame):
    """
    A frame for displaying matplotlib plots.
    
    Features:
    - Embedded matplotlib figure
    - Navigation toolbar
    - Save to file
    - Copy to clipboard
    - AI description (optional)
    
    Usage:
        plot = PlotFrame(parent)
        fig, ax = plot.get_figure()
        ax.plot([1, 2, 3], [1, 4, 9])
        plot.refresh()
    """
    
    def __init__(
        self,
        parent,
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        theme: str = "light",
        show_toolbar: bool = True,
        show_ai_button: bool = False,
        plot_info: Optional[dict] = None,
        **kwargs
    ):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for PlotFrame")
        
        self.theme = get_theme(theme)
        self.theme_name = theme
        self._figsize = figsize
        self._dpi = dpi
        self._destroyed = False
        self._resize_pending = None
        self._plot_info = plot_info or {}
        self._show_ai_button = show_ai_button
        
        super().__init__(parent, bg=self.theme["bg_card"], **kwargs)
        
        self._create_widgets(show_toolbar)
        
        # Bind destroy event for cleanup
        self.bind("<Destroy>", self._on_destroy)
        
        # Bind resize event for responsive plots
        self.bind("<Configure>", self._on_resize)
    
    def _on_resize(self, event):
        """Handle resize events with debouncing."""
        if event.widget == self and not self._destroyed:
            # Cancel any pending resize
            if self._resize_pending is not None:
                self.after_cancel(self._resize_pending)
            # Schedule resize after a short delay (debounce)
            self._resize_pending = self.after(150, self._do_resize)
    
    def _do_resize(self):
        """Actually perform the resize."""
        self._resize_pending = None
        if self._destroyed:
            return
        try:
            # Get current widget size
            width = self.winfo_width()
            height = self.winfo_height()
            
            if width > 50 and height > 50:
                # Convert to inches
                dpi = self._dpi
                fig_width = max(4, (width - 20) / dpi)
                fig_height = max(3, (height - 20) / dpi)
                
                # Update figure size
                self.figure.set_size_inches(fig_width, fig_height, forward=True)
                
                # Only call tight_layout if not using custom margins
                if not getattr(self, '_preserve_margins', False):
                    self.figure.tight_layout()
                
                self.canvas.draw_idle()
        except Exception:
            pass
    
    def set_preserve_margins(self, preserve: bool = True):
        """Set whether to preserve manual subplot margins on resize."""
        self._preserve_margins = preserve
    
    def _on_destroy(self, event):
        """Handle widget destruction - clean up matplotlib resources."""
        if event.widget == self and not self._destroyed:
            self._destroyed = True
            try:
                # Close the figure to free resources
                if hasattr(self, 'figure'):
                    plt.close(self.figure)
            except:
                pass
    
    def _create_widgets(self, show_toolbar: bool):
        """Create plot widgets."""
        # AI button header (if enabled)
        if self._show_ai_button:
            ai_header = tk.Frame(self, bg=self.theme["bg_card"])
            ai_header.pack(fill=tk.X, padx=4, pady=(4, 2))
            
            self._ai_btn = tk.Button(
                ai_header, text="🤖 AI Describe",
                font=("Segoe UI", 9),
                bg=self.theme["accent_primary"], fg="white",
                relief=tk.FLAT, cursor="hand2", padx=8, pady=2,
                command=self._generate_ai_description,
            )
            self._ai_btn.pack(side=tk.RIGHT, padx=4)
        
        # Create figure
        self.figure = Figure(figsize=self._figsize, dpi=self._dpi)
        self.figure.patch.set_facecolor(self.theme["bg_card"])
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Prevent canvas from taking focus (fixes tab switching issue)
        self.canvas_widget.configure(takefocus=False)
        
        # Right-click context menu for saving
        self.canvas_widget.bind("<Button-3>", self._show_context_menu)
        
        # Navigation toolbar
        if show_toolbar:
            toolbar_frame = tk.Frame(self, bg=self.theme["bg_card"])
            toolbar_frame.pack(fill=tk.X)
            
            self.nav_toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
            self.nav_toolbar.update()
            
            # Custom buttons
            ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=4)
            
            save_btn = tk.Button(
                toolbar_frame, text="💾 Save",
                font=FONTS.get_font("small"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
                relief=tk.FLAT,
                command=self._save_figure,
            )
            save_btn.pack(side=tk.LEFT, padx=4)
            
            # Add to Report button
            report_btn = tk.Button(
                toolbar_frame, text="📄 Add to Report",
                font=FONTS.get_font("small"),
                bg=self.theme["bg_card"],
                fg=self.theme["text_primary"],
                relief=tk.FLAT,
                command=self._add_to_report,
            )
            report_btn.pack(side=tk.LEFT, padx=4)
    
    def _show_context_menu(self, event):
        """Show right-click context menu."""
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="📄 Add to Report", command=self._add_to_report)
        menu.add_separator()
        menu.add_command(label="💾 Save as PNG...", command=lambda: self._save_as_format("png"))
        menu.add_command(label="📄 Save as PDF...", command=lambda: self._save_as_format("pdf"))
        menu.add_command(label="🎨 Save as SVG...", command=lambda: self._save_as_format("svg"))
        menu.add_command(label="🖼️ Save as JPEG...", command=lambda: self._save_as_format("jpg"))
        menu.add_separator()
        menu.add_command(label="💾 Save as... (choose format)", command=self._save_figure)
        menu.add_separator()
        menu.add_command(label="📋 Copy to clipboard", command=self._copy_to_clipboard)
        
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
    
    def _add_to_report(self):
        """Add the current plot to the report queue."""
        try:
            from biblium.gui.core.state import report_queue
            from tkinter import messagebox
            
            # Get title from plot_info or figure
            plot_title = self._plot_info.get("title", "")
            if not plot_title and self.figure.axes:
                plot_title = self.figure.axes[0].get_title() or "Plot"
            if not plot_title:
                plot_title = "Plot"
            
            # Get source panel
            source_panel = self._plot_info.get("source_panel", "")
            if not source_panel:
                parent = self.master
                while parent:
                    if hasattr(parent, 'title') and isinstance(parent.title, str):
                        source_panel = parent.title
                        break
                    if hasattr(parent, '__class__') and 'Panel' in parent.__class__.__name__:
                        source_panel = parent.__class__.__name__
                        break
                    parent = getattr(parent, 'master', None)
                if not source_panel:
                    source_panel = "Unknown Panel"
            
            # Add to queue
            report_queue.add_plot(
                figure_or_bytes=self.figure,
                title=plot_title,
                source_panel=source_panel,
            )
            
            messagebox.showinfo(
                "Added to Report",
                f"Plot '{plot_title}' has been added to the report queue.\n\n"
                f"Items in queue: {len(report_queue)}\n\n"
                "Go to Reports → Report Builder to generate your report."
            )
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to add to report: {e}")
    
    def _save_as_format(self, fmt: str):
        """Save figure in specific format."""
        extensions = {"png": ".png", "pdf": ".pdf", "svg": ".svg", "jpg": ".jpg"}
        ext = extensions.get(fmt, ".png")
        
        filename = filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=[(f"{fmt.upper()} File", f"*{ext}")],
            title=f"Save Plot as {fmt.upper()}",
        )
        
        if filename:
            self.figure.savefig(
                filename,
                dpi=600,
                bbox_inches='tight',
                facecolor=self.theme["bg_card"],
                format=fmt,
            )
    
    def _copy_to_clipboard(self):
        """Copy figure to clipboard (Windows only for now)."""
        import io
        try:
            # Save to buffer
            buf = io.BytesIO()
            self.figure.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            
            # Try to copy to clipboard (platform dependent)
            try:
                import win32clipboard
                from PIL import Image
                
                img = Image.open(buf)
                output = io.BytesIO()
                img.convert('RGB').save(output, 'BMP')
                data = output.getvalue()[14:]
                output.close()
                
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
                win32clipboard.CloseClipboard()
            except ImportError:
                # Fallback message
                from tkinter import messagebox
                messagebox.showinfo("Copy", "Clipboard copy requires pywin32 and PIL on Windows.\nUse 'Save as...' instead.")
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Could not copy to clipboard: {e}")
    
    def get_figure(self) -> Tuple[Figure, any]:
        """
        Get the figure and create a new axes.
        
        Returns
        -------
        tuple
            (figure, axes)
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(self.theme["bg_card"])
        # Disable grid by default
        ax.grid(False)
        return self.figure, ax
    
    def set_figure(self, fig: Figure):
        """
        Set a new figure.
        
        Parameters
        ----------
        fig : Figure
            Matplotlib figure to display.
        """
        # Clear old figure
        self.figure.clear()
        
        # Copy axes from new figure
        for ax in fig.axes:
            new_ax = self.figure.add_subplot(111)
            # Copy the content (simplified)
            new_ax.set_title(ax.get_title())
            new_ax.set_xlabel(ax.get_xlabel())
            new_ax.set_ylabel(ax.get_ylabel())
        
        self.refresh()
    
    def refresh(self):
        """Refresh the canvas to show updates."""
        self.canvas.draw()
    
    def set_plot_info(self, plot_info: dict):
        """Set plot info for AI description."""
        self._plot_info = plot_info
    
    def _generate_ai_description(self):
        """Generate AI description of the plot."""
        print("[DEBUG AI] _generate_ai_description called")
        
        from biblium.gui.widgets.tables import DataTable
        settings = DataTable.get_llm_settings()
        
        print(f"[DEBUG AI] Settings: enabled={settings.get('enabled')}, has_key={bool(settings.get('api_key'))}")
        
        if not settings.get("api_key"):
            from tkinter import messagebox
            messagebox.showinfo("Configure AI", 
                "Please configure your AI API key in Settings first.\n\n"
                "Go to Settings (⚙️) and enter your API key.")
            return
        
        # Update button state
        if hasattr(self, '_ai_btn'):
            self._ai_btn.config(text="⏳ Generating...", state=tk.DISABLED)
            print("[DEBUG AI] Button updated to generating state")
        
        # Extract plot info from figure if not provided
        plot_info = self._plot_info.copy() if self._plot_info else {}
        if not plot_info.get("title"):
            try:
                for ax in self.figure.axes:
                    if ax.get_title():
                        plot_info["title"] = ax.get_title()
                    if ax.get_xlabel():
                        plot_info["x_label"] = ax.get_xlabel()
                    if ax.get_ylabel():
                        plot_info["y_label"] = ax.get_ylabel()
                    break
            except:
                pass
        
        print(f"[DEBUG AI] Plot info: {plot_info}")
        
        # Store reference to self for thread callback
        widget = self
        
        import threading
        def do_generate():
            print("[DEBUG AI] Thread started, calling llm_describe_plot...")
            try:
                from biblium.llm_utils import llm_describe_plot
                result = llm_describe_plot(
                    plot_type=plot_info.get("type", "chart"),
                    title=plot_info.get("title", ""),
                    data_summary=plot_info.get("data_summary", ""),
                    x_axis=plot_info.get("x_label", ""),
                    y_axis=plot_info.get("y_label", ""),
                    context=plot_info.get("context", ""),
                    provider=settings["provider"],
                    model=settings["model"],
                    api_key=settings["api_key"],
                )
                print(f"[DEBUG AI] Got result: {result[:100] if result else 'None'}...")
                # Use widget reference to call after
                try:
                    widget.after(0, lambda r=result: widget._show_ai_result(r))
                    print("[DEBUG AI] Scheduled _show_ai_result callback")
                except Exception as e:
                    print(f"[DEBUG AI] Callback scheduling error: {e}")
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(f"[DEBUG AI] Generation error: {e}")
                import traceback
                traceback.print_exc()
                try:
                    widget.after(0, lambda msg=error_msg: widget._show_ai_result(msg))
                except Exception as e2:
                    print(f"[DEBUG AI] Error callback error: {e2}")
        
        thread = threading.Thread(target=do_generate, daemon=True)
        thread.start()
        print("[DEBUG AI] Thread started")
    
    def _show_ai_result(self, text: str):
        """Show AI result inline below the plot, appending to previous if exists."""
        print(f"[DEBUG AI] _show_ai_result called with text length: {len(text) if text else 0}")
        try:
            # Reset button
            if hasattr(self, '_ai_btn'):
                self._ai_btn.config(text="🤖 AI Describe", state=tk.NORMAL)
                print("[DEBUG AI] Button reset")
            
            # Check if we already have an AI result frame - if so, append
            if hasattr(self, '_ai_result_frame') and self._ai_result_frame and self._ai_result_frame.winfo_exists():
                # Append to existing text
                if hasattr(self, '_ai_text_widget') and self._ai_text_widget.winfo_exists():
                    self._ai_text_widget.config(state=tk.NORMAL)
                    existing = self._ai_text_widget.get("1.0", tk.END).strip()
                    if existing:
                        self._ai_text_widget.insert(tk.END, "\n\n" + "─" * 50 + "\n\n")
                    self._ai_text_widget.insert(tk.END, text)
                    self._ai_text_widget.config(state=tk.DISABLED)
                    self._ai_text_widget.see(tk.END)  # Scroll to bottom
                    print("[DEBUG AI] Appended to existing result")
                    return
            
            # Create new result frame - place it BEFORE canvas in pack order
            # First, unpack the canvas temporarily
            canvas_pack_info = None
            try:
                canvas_pack_info = self.canvas_widget.pack_info()
                self.canvas_widget.pack_forget()
            except:
                pass
            
            # Create AI result frame
            self._ai_result_frame = tk.Frame(self, bg=self.theme["bg_secondary"], relief=tk.GROOVE, bd=1)
            self._ai_result_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(4, 0), padx=4)
            print("[DEBUG AI] Result frame created")
            
            # Re-pack canvas
            if canvas_pack_info:
                self.canvas_widget.pack(**canvas_pack_info)
            else:
                self.canvas_widget.pack(fill=tk.BOTH, expand=True)
            
            # Header
            header = tk.Frame(self._ai_result_frame, bg=self.theme["bg_secondary"])
            header.pack(fill=tk.X, padx=4, pady=(4, 2))
            
            tk.Label(
                header, text="🤖 AI Description",
                font=("Segoe UI", 10, "bold"),
                bg=self.theme["bg_secondary"], fg=self.theme["text_primary"],
            ).pack(side=tk.LEFT)
            
            # Close button
            def close_result():
                try:
                    self._ai_result_frame.destroy()
                    self._ai_result_frame = None
                except:
                    pass
            
            tk.Button(
                header, text="✕", font=("Segoe UI", 8),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                relief=tk.FLAT, command=close_result,
                cursor="hand2", width=2,
            ).pack(side=tk.RIGHT, padx=2)
            
            # Copy button
            def copy_text():
                try:
                    full_text = self._ai_text_widget.get("1.0", tk.END).strip()
                    self.clipboard_clear()
                    self.clipboard_append(full_text)
                    copy_btn.config(text="✓ Copied!")
                    self.after(1500, lambda: copy_btn.config(text="📋 Copy"))
                except:
                    pass
            
            copy_btn = tk.Button(
                header, text="📋 Copy", font=("Segoe UI", 8),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                relief=tk.FLAT, command=copy_text, cursor="hand2",
            )
            copy_btn.pack(side=tk.RIGHT, padx=4)
            
            # Text widget with scrollbar
            text_container = tk.Frame(self._ai_result_frame, bg=self.theme["bg_secondary"])
            text_container.pack(fill=tk.X, padx=4, pady=(0, 4))
            
            scrollbar = tk.Scrollbar(text_container, orient=tk.VERTICAL)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            self._ai_text_widget = tk.Text(
                text_container, wrap=tk.WORD,
                font=("Segoe UI", 9),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                relief=tk.FLAT, height=5, padx=8, pady=6,
                yscrollcommand=scrollbar.set,
            )
            self._ai_text_widget.pack(fill=tk.X, expand=True)
            scrollbar.config(command=self._ai_text_widget.yview)
            
            self._ai_text_widget.insert("1.0", text)
            self._ai_text_widget.config(state=tk.DISABLED)  # Make read-only but allow selection
            
            # Enable Ctrl+C and Ctrl+A
            def on_key(e):
                if e.state & 0x4 and e.keysym.lower() in ('c', 'a'):
                    return  # Allow copy/select all
                return "break"
            self._ai_text_widget.bind("<Key>", on_key)
            
            print("[DEBUG AI] Result frame fully created and packed")
            
        except Exception as e:
            print(f"[DEBUG AI] Error showing AI result: {e}")
            import traceback
            traceback.print_exc()
            # Try to at least reset the button
            try:
                if hasattr(self, '_ai_btn'):
                    self._ai_btn.config(text="🤖 AI Describe", state=tk.NORMAL)
            except:
                pass
    
    def clear(self):
        """Clear the figure."""
        self.figure.clear()
        self.refresh()
    
    def _save_figure(self):
        """Save figure to file."""
        filetypes = [
            ("PNG Image", "*.png"),
            ("PDF Document", "*.pdf"),
            ("SVG Vector", "*.svg"),
            ("JPEG Image", "*.jpg"),
            ("All Files", "*.*"),
        ]
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=filetypes,
            title="Save Plot",
        )
        
        if filename:
            self.figure.savefig(
                filename,
                dpi=600,
                bbox_inches='tight',
                facecolor=self.theme["bg_card"],
            )
    
    def save(self, filename: str, dpi: int = 150):
        """
        Save figure to file programmatically.
        
        Parameters
        ----------
        filename : str
            Output file path.
        dpi : int
            Resolution.
        """
        self.figure.savefig(filename, dpi=dpi, bbox_inches='tight')


class PlotNotebook(ttk.Notebook):
    """
    A notebook with multiple plot tabs.
    
    Usage:
        notebook = PlotNotebook(parent)
        fig1, ax1 = notebook.add_plot("Scatter")
        fig2, ax2 = notebook.add_plot("Bar")
    """
    
    def __init__(self, parent, theme: str = "light", **kwargs):
        self.theme = get_theme(theme)
        super().__init__(parent, **kwargs)
        
        self._plots = {}
    
    def add_plot(self, title: str, figsize: Tuple[float, float] = (8, 6)) -> Tuple[Figure, any]:
        """
        Add a new plot tab.
        
        Parameters
        ----------
        title : str
            Tab title.
        figsize : tuple
            Figure size.
        
        Returns
        -------
        tuple
            (figure, axes)
        """
        plot_frame = PlotFrame(self, figsize=figsize, theme=self.theme.get("name", "light"))
        self.add(plot_frame, text=f"  {title}  ")
        
        fig, ax = plot_frame.get_figure()
        self._plots[title] = plot_frame
        
        return fig, ax
    
    def get_plot(self, title: str) -> Optional[PlotFrame]:
        """Get plot frame by title."""
        return self._plots.get(title)
    
    def refresh_all(self):
        """Refresh all plots."""
        for plot in self._plots.values():
            plot.refresh()


class ScaledImageFrame(tk.Frame):
    """
    A frame that displays an image scaled to fit the available space.
    
    Features:
    - Auto-scales image to fit container while maintaining aspect ratio
    - Responds to window resize events
    - Right-click context menu for saving in multiple formats
    
    Usage:
        frame = ScaledImageFrame(parent, theme="light")
        frame.set_image(pil_image)
        frame.set_image_from_file(path)
        frame.set_image_from_figure(matplotlib_fig)
    """
    
    def __init__(
        self,
        parent,
        theme: str = "light",
        maintain_aspect: bool = True,
        max_scale: float = 1.5,
        **kwargs
    ):
        self.theme_dict = get_theme(theme)
        self._maintain_aspect = maintain_aspect
        self._max_scale = max_scale
        self._original_image = None
        self._photo_image = None
        self._resize_job = None
        self._source_figure = None  # Store source figure for high-quality saves
        
        super().__init__(parent, bg=self.theme_dict["bg_card"], **kwargs)
        
        self._image_label = tk.Label(self, bg=self.theme_dict["bg_card"])
        self._image_label.pack(fill=tk.BOTH, expand=True)
        
        self.bind("<Configure>", self._on_resize)
        
        # Right-click context menu for saving
        self._image_label.bind("<Button-3>", self._show_context_menu)
        self.bind("<Button-3>", self._show_context_menu)
    
    def set_image(self, pil_image):
        """Set image from PIL Image object."""
        try:
            self._original_image = pil_image
            self._update_display()
        except Exception as e:
            print(f"Error setting image: {e}")
    
    def set_image_from_file(self, filepath: str):
        """Set image from file path."""
        try:
            from PIL import Image
            img = Image.open(filepath)
            self.set_image(img)
        except Exception as e:
            print(f"Error loading image: {e}")
    
    def set_image_from_figure(self, fig, dpi: int = 100):
        """Set image from matplotlib Figure."""
        try:
            import io
            from PIL import Image
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            
            # Store the figure for high-quality saves
            self._source_figure = fig
            
            canvas_agg = FigureCanvasAgg(fig)
            canvas_agg.draw()
            
            buf = io.BytesIO()
            # Use pad_inches to ensure figtext annotations are not cut off
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none', pad_inches=0.2)
            buf.seek(0)
            
            pil_image = Image.open(buf)
            pil_image = pil_image.copy()
            buf.close()
            
            self.set_image(pil_image)
        except Exception as e:
            print(f"Error setting image from figure: {e}")
    
    def _on_resize(self, event):
        """Handle resize event with debouncing."""
        # Guard against resize during update
        if getattr(self, '_updating', False):
            return
        if self._resize_job:
            try:
                self.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self.after(150, self._update_display)
    
    def _update_display(self):
        """Update the displayed image based on current size."""
        if self._original_image is None:
            return
        
        # Guard against recursive updates
        if getattr(self, '_updating', False):
            return
        self._updating = True
        
        try:
            from PIL import Image, ImageTk
            
            self.update_idletasks()
            available_width = self.winfo_width()
            available_height = self.winfo_height()
            
            if available_width <= 1 or available_height <= 1:
                return
            
            img_width, img_height = self._original_image.size
            
            if self._maintain_aspect:
                scale_w = available_width / img_width
                scale_h = available_height / img_height
                scale = min(scale_w, scale_h, self._max_scale)
                
                new_width = max(1, int(img_width * scale))
                new_height = max(1, int(img_height * scale))
            else:
                new_width = available_width
                new_height = available_height
            
            resized = self._original_image.resize(
                (new_width, new_height), 
                Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
            )
            self._photo_image = ImageTk.PhotoImage(resized)
            self._image_label.configure(image=self._photo_image)
                
        except Exception as e:
            print(f"Error updating display: {e}")
        finally:
            self._updating = False
    
    def _show_context_menu(self, event):
        """Show right-click context menu for saving."""
        if self._original_image is None and self._source_figure is None:
            return
        
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="📄 Add to Report", command=self._add_to_report)
        menu.add_separator()
        menu.add_command(label="💾 Save as PNG...", command=lambda: self._save_as_format("png"))
        menu.add_command(label="📄 Save as PDF...", command=lambda: self._save_as_format("pdf"))
        menu.add_command(label="🎨 Save as SVG...", command=lambda: self._save_as_format("svg"))
        menu.add_command(label="🖼️ Save as JPEG...", command=lambda: self._save_as_format("jpg"))
        menu.add_separator()
        menu.add_command(label="💾 Save as... (choose format)", command=self._save_figure)
        menu.add_separator()
        menu.add_command(label="📋 Copy to clipboard", command=self._copy_to_clipboard)
        
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
    
    def _add_to_report(self):
        """Add the current plot/image to the report queue."""
        try:
            from biblium.gui.core.state import report_queue
            from tkinter import messagebox
            import io
            
            # Get title
            plot_title = getattr(self, '_title', None) or "Image"
            
            # Get source panel
            source_panel = getattr(self, '_source_panel', None)
            if not source_panel:
                parent = self.master
                while parent:
                    if hasattr(parent, 'title') and isinstance(parent.title, str):
                        source_panel = parent.title
                        break
                    if hasattr(parent, '__class__') and 'Panel' in parent.__class__.__name__:
                        source_panel = parent.__class__.__name__
                        break
                    parent = getattr(parent, 'master', None)
                if not source_panel:
                    source_panel = "Unknown Panel"
            
            # Get image data
            if self._source_figure is not None:
                # Save matplotlib figure
                buf = io.BytesIO()
                self._source_figure.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
                buf.seek(0)
                data = buf.getvalue()
            elif self._original_image is not None:
                # Save PIL image
                buf = io.BytesIO()
                self._original_image.save(buf, format='PNG')
                buf.seek(0)
                data = buf.getvalue()
            else:
                messagebox.showinfo("No Image", "No image to add to report.")
                return
            
            # Add to queue
            report_queue.add_plot(
                figure_or_bytes=data,
                title=plot_title,
                source_panel=source_panel,
            )
            
            messagebox.showinfo(
                "Added to Report",
                f"Plot '{plot_title}' has been added to the report queue.\n\n"
                f"Items in queue: {len(report_queue)}\n\n"
                "Go to Reports → Report Builder to generate your report."
            )
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to add to report: {e}")
    
    def _save_as_format(self, fmt: str):
        """Save image in specific format."""
        extensions = {"png": ".png", "pdf": ".pdf", "svg": ".svg", "jpg": ".jpg"}
        ext = extensions.get(fmt, ".png")
        
        filename = filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=[(f"{fmt.upper()} File", f"*{ext}")],
            title=f"Save Image as {fmt.upper()}",
        )
        
        if filename:
            try:
                # If we have a source figure and format supports vector, use it
                if self._source_figure is not None and fmt in ("pdf", "svg"):
                    self._source_figure.savefig(
                        filename,
                        dpi=600,
                        bbox_inches='tight',
                        facecolor='white',
                        format=fmt,
                    )
                elif self._source_figure is not None:
                    # For raster formats, use figure for high quality
                    self._source_figure.savefig(
                        filename,
                        dpi=600,
                        bbox_inches='tight',
                        facecolor='white',
                        format=fmt,
                    )
                elif self._original_image is not None:
                    # Fall back to PIL image
                    if fmt == "jpg":
                        # Convert RGBA to RGB for JPEG
                        if self._original_image.mode == 'RGBA':
                            rgb_image = self._original_image.convert('RGB')
                            rgb_image.save(filename, format="JPEG", quality=95)
                        else:
                            self._original_image.save(filename, format="JPEG", quality=95)
                    elif fmt == "pdf":
                        self._original_image.save(filename, format="PDF")
                    else:
                        self._original_image.save(filename)
            except Exception as e:
                print(f"Error saving image: {e}")
    
    def _save_figure(self):
        """Save image to file with format selection."""
        filetypes = [
            ("PNG Image", "*.png"),
            ("PDF Document", "*.pdf"),
            ("SVG Vector", "*.svg"),
            ("JPEG Image", "*.jpg"),
            ("All Files", "*.*"),
        ]
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=filetypes,
            title="Save Image",
        )
        
        if filename:
            # Determine format from extension
            ext = filename.lower().split('.')[-1]
            fmt_map = {"png": "png", "pdf": "pdf", "svg": "svg", "jpg": "jpg", "jpeg": "jpg"}
            fmt = fmt_map.get(ext, "png")
            self._save_as_format(fmt)
    
    def _copy_to_clipboard(self):
        """Copy image to clipboard."""
        if self._original_image is None:
            return
        
        import io
        try:
            # Save to buffer as PNG
            buf = io.BytesIO()
            self._original_image.save(buf, format='PNG')
            buf.seek(0)
            
            # Try platform-specific clipboard
            import platform
            if platform.system() == "Windows":
                import win32clipboard
                from PIL import Image
                
                # Convert to BMP for Windows clipboard
                output = io.BytesIO()
                self._original_image.convert('RGB').save(output, 'BMP')
                data = output.getvalue()[14:]  # Remove BMP header
                output.close()
                
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
                win32clipboard.CloseClipboard()
            else:
                # For other platforms, try pyperclip or just notify user
                print("Clipboard copy is best supported on Windows. Image saved to temporary buffer.")
        except ImportError:
            print("Clipboard functionality requires pywin32 on Windows")
        except Exception as e:
            print(f"Error copying to clipboard: {e}")


class ResizablePlotCanvas(tk.Frame):
    """
    A resizable matplotlib canvas with right-click save functionality.
    
    This class wraps FigureCanvasTkAgg with:
    - Auto-scaling to fit container
    - Right-click context menu for saving in multiple formats
    - Debounced resize handling
    
    Usage:
        # Create frame and get figure
        plot = ResizablePlotCanvas(parent, figsize=(8, 6))
        fig, ax = plot.get_figure()
        
        # Plot your data
        ax.plot([1, 2, 3], [1, 4, 9])
        
        # Refresh the display
        plot.refresh()
        
        # Or set an existing figure
        plot.set_figure(my_fig)
    """
    
    def __init__(
        self,
        parent,
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        theme: str = "light",
        **kwargs
    ):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for ResizablePlotCanvas")
        
        self.theme = get_theme(theme)
        self._figsize = figsize
        self._dpi = dpi
        self._destroyed = False
        self._resize_pending = None
        
        super().__init__(parent, bg=self.theme["bg_card"], **kwargs)
        
        # Create figure
        self.figure = Figure(figsize=figsize, dpi=dpi)
        self.figure.patch.set_facecolor(self.theme["bg_card"])
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Prevent canvas from taking focus
        self.canvas_widget.configure(takefocus=False)
        
        # Right-click context menu
        self.canvas_widget.bind("<Button-3>", self._show_context_menu)
        
        # Resize handling
        self.bind("<Configure>", self._on_resize)
        self.bind("<Destroy>", self._on_destroy)
    
    def get_figure(self) -> Tuple[Figure, any]:
        """
        Get figure and axes.
        
        Returns
        -------
        tuple
            (figure, axes)
        """
        if not self.figure.axes:
            ax = self.figure.add_subplot(111)
        else:
            ax = self.figure.axes[0]
        return self.figure, ax
    
    def set_figure(self, fig: Figure):
        """
        Replace with a new figure.
        
        Parameters
        ----------
        fig : Figure
            The matplotlib figure to display.
        """
        # Close old figure
        try:
            plt.close(self.figure)
        except:
            pass
        
        # Set new figure
        self.figure = fig
        self.figure.patch.set_facecolor(self.theme["bg_card"])
        
        # Update canvas
        self.canvas.figure = fig
        self.canvas.draw()
    
    def refresh(self):
        """Refresh the canvas to show updates."""
        if not self._destroyed:
            try:
                self.figure.tight_layout()
            except:
                pass
            self.canvas.draw_idle()
    
    def clear(self):
        """Clear the figure."""
        self.figure.clear()
        self.refresh()
    
    def _on_resize(self, event):
        """Handle resize events with debouncing."""
        if event.widget == self and not self._destroyed:
            if self._resize_pending is not None:
                self.after_cancel(self._resize_pending)
            self._resize_pending = self.after(150, self._do_resize)
    
    def _do_resize(self):
        """Actually perform the resize."""
        self._resize_pending = None
        if self._destroyed:
            return
        try:
            width = self.winfo_width()
            height = self.winfo_height()
            
            if width > 50 and height > 50:
                dpi = self._dpi
                fig_width = max(4, (width - 20) / dpi)
                fig_height = max(3, (height - 20) / dpi)
                
                self.figure.set_size_inches(fig_width, fig_height, forward=True)
                
                try:
                    self.figure.tight_layout()
                except:
                    pass
                
                self.canvas.draw_idle()
        except Exception:
            pass
    
    def _on_destroy(self, event):
        """Handle widget destruction."""
        if event.widget == self and not self._destroyed:
            self._destroyed = True
            try:
                plt.close(self.figure)
            except:
                pass
    
    def _show_context_menu(self, event):
        """Show right-click context menu."""
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="📄 Add to Report", command=self._add_to_report)
        menu.add_separator()
        menu.add_command(label="💾 Save as PNG...", command=lambda: self._save_as_format("png"))
        menu.add_command(label="📄 Save as PDF...", command=lambda: self._save_as_format("pdf"))
        menu.add_command(label="🎨 Save as SVG...", command=lambda: self._save_as_format("svg"))
        menu.add_command(label="🖼️ Save as JPEG...", command=lambda: self._save_as_format("jpg"))
        menu.add_separator()
        menu.add_command(label="💾 Save as... (choose format)", command=self._save_figure)
        menu.add_separator()
        menu.add_command(label="📋 Copy to clipboard", command=self._copy_to_clipboard)
        
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
    
    def _add_to_report(self):
        """Add the current plot to the report queue."""
        try:
            from biblium.gui.core.state import report_queue
            from tkinter import messagebox
            
            # Get title from figure
            plot_title = "Plot"
            if self.figure.axes:
                plot_title = self.figure.axes[0].get_title() or "Plot"
            
            # Get source panel
            source_panel = "Unknown Panel"
            parent = self.master
            while parent:
                if hasattr(parent, 'title') and isinstance(parent.title, str):
                    source_panel = parent.title
                    break
                if hasattr(parent, '__class__') and 'Panel' in parent.__class__.__name__:
                    source_panel = parent.__class__.__name__
                    break
                parent = getattr(parent, 'master', None)
            
            # Add to queue
            report_queue.add_plot(
                figure_or_bytes=self.figure,
                title=plot_title,
                source_panel=source_panel,
            )
            
            messagebox.showinfo(
                "Added to Report",
                f"Plot '{plot_title}' has been added to the report queue.\n\n"
                f"Items in queue: {len(report_queue)}\n\n"
                "Go to Reports → Report Builder to generate your report."
            )
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to add to report: {e}")
    
    def _save_as_format(self, fmt: str):
        """Save figure in specific format."""
        extensions = {"png": ".png", "pdf": ".pdf", "svg": ".svg", "jpg": ".jpg"}
        ext = extensions.get(fmt, ".png")
        
        filename = filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=[(f"{fmt.upper()} File", f"*{ext}")],
            title=f"Save Plot as {fmt.upper()}",
        )
        
        if filename:
            self.figure.savefig(
                filename,
                dpi=600,
                bbox_inches='tight',
                facecolor=self.theme["bg_card"],
                format=fmt,
            )
    
    def _save_figure(self):
        """Save figure to file with format selection."""
        filetypes = [
            ("PNG Image", "*.png"),
            ("PDF Document", "*.pdf"),
            ("SVG Vector", "*.svg"),
            ("JPEG Image", "*.jpg"),
            ("All Files", "*.*"),
        ]
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=filetypes,
            title="Save Plot",
        )
        
        if filename:
            self.figure.savefig(
                filename,
                dpi=600,
                bbox_inches='tight',
                facecolor=self.theme["bg_card"],
            )
    
    def _copy_to_clipboard(self):
        """Copy figure to clipboard."""
        import io
        try:
            buf = io.BytesIO()
            self.figure.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            
            import platform
            if platform.system() == "Windows":
                import win32clipboard
                from PIL import Image
                
                img = Image.open(buf)
                output = io.BytesIO()
                img.convert('RGB').save(output, 'BMP')
                data = output.getvalue()[14:]
                output.close()
                
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
                win32clipboard.CloseClipboard()
            else:
                print("Clipboard copy is best supported on Windows.")
        except ImportError:
            print("Clipboard functionality requires pywin32 on Windows")
        except Exception as e:
            print(f"Error copying to clipboard: {e}")
    
    def save(self, filename: str, dpi: int = 150):
        """
        Save figure to file programmatically.
        
        Parameters
        ----------
        filename : str
            Output file path.
        dpi : int
            Resolution.
        """
        self.figure.savefig(filename, dpi=dpi, bbox_inches='tight')


def add_plot_context_menu(canvas_widget, figure, theme_bg="white", parent_frame=None, show_ai_button=True, title=None, source_panel=None):
    """
    Add right-click context menu to a matplotlib canvas widget.
    
    This function can be used to add save functionality to any 
    FigureCanvasTkAgg-based plot without changing the entire structure.
    
    Parameters
    ----------
    canvas_widget : tk.Widget
        The tkinter widget from canvas.get_tk_widget()
    figure : matplotlib.figure.Figure
        The matplotlib figure to save
    theme_bg : str
        Background color for saved images
    parent_frame : tk.Frame, optional
        Parent frame to add AI button to. If None, uses canvas_widget's parent.
    show_ai_button : bool
        Whether to show a visible AI button (default True)
    title : str, optional
        Title for the plot when adding to report
    source_panel : str, optional
        Source panel name for report
    
    Usage:
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
        
        # Add right-click menu
        add_plot_context_menu(widget, fig)
    """
    # Store plot metadata
    canvas_widget._plot_title = title
    canvas_widget._source_panel = source_panel
    
    # Store AI result frame reference
    ai_state = {"result_frame": None, "loading_frame": None, "ai_btn": None}
    
    def ai_describe_plot():
        """Generate AI description of the plot."""
        from biblium.gui.widgets.tables import DataTable
        settings = DataTable.get_llm_settings()
        
        if not settings.get("api_key"):
            from tkinter import messagebox
            messagebox.showinfo("Configure AI", 
                "Please configure your AI API key in Settings first.\n\n"
                "Go to Settings (⚙️) and enter your API key.")
            return
        
        # Update button state
        if ai_state.get("ai_btn"):
            ai_state["ai_btn"].config(text="⏳ Generating...", state=tk.DISABLED)
        
        plot_info = extract_plot_info()
        show_ai_loading()
        
        import threading
        def do_generate():
            try:
                from biblium.llm_utils import llm_describe_plot
                result = llm_describe_plot(
                    plot_type=plot_info.get("type", "chart"),
                    title=plot_info.get("title", ""),
                    data_summary=plot_info.get("data_summary", ""),
                    x_axis=plot_info.get("x_label", ""),
                    y_axis=plot_info.get("y_label", ""),
                    context=plot_info.get("context", ""),
                    provider=settings["provider"],
                    model=settings["model"],
                    api_key=settings["api_key"],
                    custom_prompt=settings.get("custom_prompt", ""),
                )
                canvas_widget.after(0, lambda r=result: show_ai_result(r))
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                canvas_widget.after(0, lambda msg=error_msg: show_ai_result(msg))
        
        thread = threading.Thread(target=do_generate, daemon=True)
        thread.start()
    
    def extract_plot_info():
        """Extract information about the plot."""
        info = {"type": "chart", "title": "", "x_label": "", "y_label": "", "data_summary": "", "context": ""}
        try:
            if figure.axes:
                ax = figure.axes[0]
                info["title"] = ax.get_title() or ""
                info["x_label"] = ax.get_xlabel() or ""
                info["y_label"] = ax.get_ylabel() or ""
                
                lines = ax.get_lines()
                bars = ax.patches
                
                if lines:
                    info["type"] = "line chart"
                    info["data_summary"] = f"{len(lines)} data series"
                elif bars:
                    info["type"] = "bar chart"
                    info["data_summary"] = f"{len(bars)} bars"
                
                # Check for scatter
                collections = ax.collections
                if collections:
                    info["type"] = "scatter plot"
                    total_points = sum(len(c.get_offsets()) for c in collections if hasattr(c, 'get_offsets'))
                    info["data_summary"] = f"{total_points} data points"
        except:
            pass
        return info
    
    def show_ai_loading():
        """Show loading indicator."""
        if ai_state["loading_frame"]:
            try: ai_state["loading_frame"].destroy()
            except: pass
        
        parent = canvas_widget.master
        ai_state["loading_frame"] = tk.Frame(parent, bg="#f0f0f0")
        ai_state["loading_frame"].pack(fill=tk.X, side=tk.BOTTOM, pady=(4, 0))
        tk.Label(ai_state["loading_frame"], text="⏳ Generating AI description...",
                font=("Segoe UI", 9), bg="#f0f0f0", fg="#666").pack(pady=4)
    
    def show_ai_result(text):
        """Show AI description result."""
        # Reset button state
        if ai_state.get("ai_btn"):
            ai_state["ai_btn"].config(text="🤖 AI Describe", state=tk.NORMAL)
        
        if ai_state["loading_frame"]:
            try: ai_state["loading_frame"].destroy()
            except: pass
        if ai_state["result_frame"]:
            try: ai_state["result_frame"].destroy()
            except: pass
        
        parent = canvas_widget.master
        ai_state["result_frame"] = tk.Frame(parent, bg="#f8f8f8")
        ai_state["result_frame"].pack(fill=tk.X, side=tk.BOTTOM, pady=(4, 0))
        
        header = tk.Frame(ai_state["result_frame"], bg="#f8f8f8")
        header.pack(fill=tk.X)
        tk.Label(header, text="🤖 AI Description", font=("Segoe UI", 9, "bold"),
                bg="#f8f8f8", fg="#333").pack(side=tk.LEFT, padx=4)
        
        def copy_text():
            try:
                canvas_widget.clipboard_clear()
                canvas_widget.clipboard_append(text)
                copy_btn.config(text="✓ Copied")
                canvas_widget.after(1500, lambda: copy_btn.config(text="📋 Copy"))
            except: pass
        
        tk.Button(header, text="✕", font=("Segoe UI", 8), bg="#e0e0e0", fg="#333",
                  relief=tk.FLAT, command=lambda: ai_state["result_frame"].destroy(),
                  cursor="hand2", width=2).pack(side=tk.RIGHT, padx=2)
        copy_btn = tk.Button(header, text="📋 Copy", font=("Segoe UI", 8), bg="#e0e0e0", fg="#333",
                             relief=tk.FLAT, command=copy_text, cursor="hand2")
        copy_btn.pack(side=tk.RIGHT, padx=2)
        
        text_widget = tk.Text(ai_state["result_frame"], wrap=tk.WORD, font=("Segoe UI", 9),
                              bg="#f0f0f0", fg="#333", relief=tk.FLAT, height=4, padx=8, pady=4)
        text_widget.pack(fill=tk.X, padx=4, pady=(2, 4))
        text_widget.insert("1.0", text)
        def on_key(e):
            if e.state & 0x4 and e.keysym.lower() in ('c', 'a'): return
            return "break"
        text_widget.bind("<Key>", on_key)
        text_widget.bind("<Button-1>", lambda e: text_widget.focus_set())
    
    # Add visible AI button if requested
    if show_ai_button:
        parent = parent_frame or canvas_widget.master
        ai_header = tk.Frame(parent, bg=theme_bg)
        ai_header.pack(fill=tk.X, side=tk.TOP, before=canvas_widget, padx=4, pady=(4, 2))
        
        ai_btn = tk.Button(
            ai_header, text="🤖 AI Describe",
            font=("Segoe UI", 9),
            bg="#3182CE", fg="white",
            relief=tk.FLAT, cursor="hand2", padx=8, pady=2,
            command=ai_describe_plot,
        )
        ai_btn.pack(side=tk.RIGHT, padx=4)
        ai_state["ai_btn"] = ai_btn
    
    def show_context_menu(event):
        menu = tk.Menu(canvas_widget, tearoff=0)
        menu.add_command(label="📄 Add to Report", command=add_to_report)
        menu.add_separator()
        menu.add_command(label="🤖 AI Describe Plot", command=ai_describe_plot)
        menu.add_separator()
        menu.add_command(label="💾 Save as PNG...", command=lambda: save_as_format("png"))
        menu.add_command(label="📄 Save as PDF...", command=lambda: save_as_format("pdf"))
        menu.add_command(label="🎨 Save as SVG...", command=lambda: save_as_format("svg"))
        menu.add_command(label="🖼️ Save as JPEG...", command=lambda: save_as_format("jpg"))
        menu.add_separator()
        menu.add_command(label="💾 Save as... (choose format)", command=save_figure)
        menu.add_separator()
        menu.add_command(label="📋 Copy to clipboard", command=copy_to_clipboard)
        
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
    
    def add_to_report():
        """Add the current plot to the report queue."""
        try:
            from biblium.gui.core.state import report_queue
            from tkinter import messagebox
            
            # Get title
            plot_title = getattr(canvas_widget, '_plot_title', None)
            if not plot_title:
                # Try to get from figure
                if figure.axes:
                    plot_title = figure.axes[0].get_title() or "Plot"
                else:
                    plot_title = "Plot"
            
            # Get source panel
            panel = getattr(canvas_widget, '_source_panel', None)
            if not panel:
                # Try to find from parent hierarchy
                parent = canvas_widget.master
                while parent:
                    if hasattr(parent, 'title') and isinstance(parent.title, str):
                        panel = parent.title
                        break
                    if hasattr(parent, '__class__') and 'Panel' in parent.__class__.__name__:
                        panel = parent.__class__.__name__
                        break
                    parent = getattr(parent, 'master', None)
                if not panel:
                    panel = "Unknown Panel"
            
            # Add to queue
            report_queue.add_plot(
                figure_or_bytes=figure,
                title=plot_title,
                source_panel=panel,
            )
            
            messagebox.showinfo(
                "Added to Report",
                f"Plot '{plot_title}' has been added to the report queue.\n\n"
                f"Items in queue: {len(report_queue)}\n\n"
                "Go to Reports → Report Builder to generate your report."
            )
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to add to report: {e}")
    
    def save_as_format(fmt):
        extensions = {"png": ".png", "pdf": ".pdf", "svg": ".svg", "jpg": ".jpg"}
        ext = extensions.get(fmt, ".png")
        
        filename = filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=[(f"{fmt.upper()} File", f"*{ext}")],
            title=f"Save Plot as {fmt.upper()}",
        )
        
        if filename:
            figure.savefig(
                filename,
                dpi=600,
                bbox_inches='tight',
                facecolor=theme_bg,
                format=fmt,
            )
    
    def save_figure():
        filetypes = [
            ("PNG Image", "*.png"),
            ("PDF Document", "*.pdf"),
            ("SVG Vector", "*.svg"),
            ("JPEG Image", "*.jpg"),
            ("All Files", "*.*"),
        ]
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=filetypes,
            title="Save Plot",
        )
        
        if filename:
            figure.savefig(
                filename,
                dpi=600,
                bbox_inches='tight',
                facecolor=theme_bg,
            )
    
    def copy_to_clipboard():
        import io
        try:
            buf = io.BytesIO()
            figure.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            
            import platform
            if platform.system() == "Windows":
                import win32clipboard
                from PIL import Image
                
                img = Image.open(buf)
                output = io.BytesIO()
                img.convert('RGB').save(output, 'BMP')
                data = output.getvalue()[14:]
                output.close()
                
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
                win32clipboard.CloseClipboard()
            else:
                print("Clipboard copy supported on Windows only.")
        except ImportError:
            print("Clipboard functionality requires pywin32 on Windows")
        except Exception as e:
            print(f"Error copying to clipboard: {e}")
    
    canvas_widget.bind("<Button-3>", show_context_menu)


def make_canvas_resizable(canvas, figure, parent_frame, dpi=100):
    """
    Add resize handling to a matplotlib canvas.
    
    Parameters
    ----------
    canvas : FigureCanvasTkAgg
        The matplotlib canvas
    figure : matplotlib.figure.Figure
        The matplotlib figure
    parent_frame : tk.Widget
        The parent frame to bind resize events to
    dpi : int
        DPI for calculating figure size
    
    Usage:
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
        
        # Add resize handling
        make_canvas_resizable(canvas, fig, frame)
    """
    resize_job = [None]  # Use list to allow modification in nested function
    
    def on_resize(event):
        if event.widget == parent_frame:
            if resize_job[0] is not None:
                try:
                    parent_frame.after_cancel(resize_job[0])
                except:
                    pass
            resize_job[0] = parent_frame.after(150, do_resize)
    
    def do_resize():
        resize_job[0] = None
        try:
            width = parent_frame.winfo_width()
            height = parent_frame.winfo_height()
            
            if width > 50 and height > 50:
                fig_width = max(4, (width - 20) / dpi)
                fig_height = max(3, (height - 20) / dpi)
                
                figure.set_size_inches(fig_width, fig_height, forward=True)
                
                try:
                    figure.tight_layout()
                except:
                    pass
                
                canvas.draw_idle()
        except Exception:
            pass
    
    parent_frame.bind("<Configure>", on_resize)
