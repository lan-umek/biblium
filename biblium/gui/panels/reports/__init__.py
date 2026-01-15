# -*- coding: utf-8 -*-
"""
Report Builder Panels
=====================
Generate comprehensive bibliometric reports in various formats.

Panels:
- ReportBuilderPanel: Generate template-based bibliometric reports
- CustomReportPanel: Build custom reports from collected plots and tables
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from typing import Dict, List, Optional
from pathlib import Path
import os

from biblium.gui.config import FONTS, LAYOUT, get_theme
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card, CollapsibleCard, StatsCard, CardGrid
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox, LabeledEntry, LabeledSpinbox

# Import custom report panel
from biblium.gui.panels.reports.custom_report_panel import CustomReportPanel

__all__ = ['ReportBuilderPanel', 'CustomReportPanel']

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class ReportBuilderPanel(BasePanel):
    """Panel for generating bibliometric reports."""
    
    title = "Report Builder"
    icon = "📝"
    description = "Generate comprehensive bibliometric analysis reports"
    requires_data = True
    
    # Info tab content
    info_content = """
## Report Builder

Generate professional bibliometric analysis reports in multiple formats.

### Report Levels

| Level | Description | Items |
|-------|-------------|-------|
| **Basic** | Core counts and main information | ~16 items |
| **Standard** | Adds performance stats and visualizations | ~29 items |
| **Extended** | Adds production over time and co-occurrence | ~37 items |
| **Full** | All available analyses | ~76 items |
| **Groups** | Group analysis only (requires group setup) | ~38 items |
| **All** | Full + Groups combined | ~114 items |

### Output Formats

- **DOCX (Word)**: Formatted document with tables and figures
- **XLSX (Excel)**: Multi-sheet workbook with all data tables
- **PPTX (PowerPoint)**: Presentation slides
- **TEX (LaTeX)**: Academic paper format

### Report Sections

**Main Information:**
- Dataset descriptives (documents, years, sources, authors)
- Global performance indicators (citations, h-index)
- Scientific production over time

**Item Counts:**
- Sources, Authors, Keywords
- Countries, Affiliations, References
- Document Types, SDG Classifications

**Performance Statistics:**
- Source, Author, Keyword performance metrics

**Bibliometric Laws:**
- Bradford's Law (source concentration)
- Lotka's Law (author productivity)
- Zipf's Law (word frequency)

**Citation Analysis:**
- Top cited documents (global and local)
- Citation distribution

**Group Analysis (requires BiblioGroupAnalysis):**
- Group matrix and intersections
- Group-level descriptives and performances
- Associations (keywords, sources, authors, countries)
- Group counts and statistics

### Usage Tips

1. **Quick Start**: Select "Basic" level and DOCX format
2. **Check First**: Use "Check Data Availability" to see what's ready
3. **Auto-prepare**: Enable to automatically run required analyses
4. **Custom Reports**: Edit template sheets in additional files
5. **Group Reports**: First set up groups in Group Setup panel

### Methods Used

```python
# Prepare data for report
ba.prepare_for_report(level="basic")

# Generate report
ba.generate_report(
    output="my_report",
    level="basic",
    formats=["docx", "xlsx"]
)
```
"""
    
    def _create_options(self):
        """Create the options panel."""
        self._add_title()
        
        # Report Level Card
        level_card = Card(self.options_content, title="📊 Report Level", theme=self.theme_name)
        level_card.pack(fill=tk.X, padx=8, pady=8)
        
        tk.Label(
            level_card.content,
            text="Select the comprehensiveness of the report:",
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(0, 4))
        
        self.level_combo = LabeledCombobox(
            level_card.content, label="Level:",
            values=[
                "basic - Core counts and main info",
                "standard - Adds performance stats",
                "extended - Adds production over time",
                "full - All available analyses",
                "groups - Group analysis only",
                "all - Full + Groups combined",
            ],
            default="basic - Core counts and main info",
            theme=self.theme_name, label_width=12,
        )
        self.level_combo.pack(fill=tk.X, pady=4)
        
        # Output Format Card
        format_card = Card(self.options_content, title="📄 Output Format", theme=self.theme_name)
        format_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.format_vars = {}
        formats_frame = tk.Frame(format_card.content, bg=self.theme["bg_card"])
        formats_frame.pack(fill=tk.X, pady=4)
        
        self.format_vars["docx"] = tk.BooleanVar(value=True)
        self.format_vars["xlsx"] = tk.BooleanVar(value=True)
        self.format_vars["pptx"] = tk.BooleanVar(value=False)
        self.format_vars["tex"] = tk.BooleanVar(value=False)
        
        for fmt, var in self.format_vars.items():
            cb = tk.Checkbutton(
                formats_frame, text=fmt.upper(),
                variable=var, font=FONTS.get_font("body"),
                bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                selectcolor=self.theme["bg_secondary"],
                activebackground=self.theme["bg_card"],
            )
            cb.pack(side=tk.LEFT, padx=8)
        
        self.output_folder = LabeledEntry(
            format_card.content, label="Output Folder:",
            default="reports",
            theme=self.theme_name, label_width=12,
        )
        self.output_folder.pack(fill=tk.X, pady=4)
        
        self.report_name = LabeledEntry(
            format_card.content, label="Report Name:",
            default="bibliometric_report",
            theme=self.theme_name, label_width=12,
        )
        self.report_name.pack(fill=tk.X, pady=4)
        
        ThemedButton(
            format_card.content, text="Browse...", style="ghost", size="small",
            command=self._browse_folder, theme=self.theme_name,
        ).pack(anchor=tk.E)
        
        # Options Card
        options_card = CollapsibleCard(
            self.options_content, title="⚙️ Options",
            collapsed=True, theme=self.theme_name,
        )
        options_card.pack(fill=tk.X, padx=8, pady=8)
        
        self.auto_prepare = LabeledCheckbox(
            options_card.content, label="Auto-prepare data before generation",
            default=True, theme=self.theme_name,
        )
        self.auto_prepare.pack(fill=tk.X, pady=2)
        
        self.generate_plots = LabeledCheckbox(
            options_card.content, label="Generate plots for report",
            default=True, theme=self.theme_name,
        )
        self.generate_plots.pack(fill=tk.X, pady=2)
        
        self.top_n = LabeledSpinbox(
            options_card.content, label="Top N items:",
            from_=5, to=100, default=20,
            theme=self.theme_name, label_width=15,
        )
        self.top_n.pack(fill=tk.X, pady=4)
        
        # Action Buttons
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        
        ActionButton(
            btn_frame, text="Generate Report", icon="📝",
            command=self._generate_report, theme=self.theme_name,
        ).pack(fill=tk.X)
        
        ThemedButton(
            btn_frame, text="Check Data Availability", style="secondary",
            command=self._check_availability, theme=self.theme_name,
        ).pack(fill=tk.X, pady=(8, 0))
        
        ThemedButton(
            btn_frame, text="Quick Preview", style="ghost",
            command=self._preview_report, theme=self.theme_name,
        ).pack(fill=tk.X, pady=(8, 0))
    
    def _create_results(self):
        """Create the results panel."""
        self.results_card = tk.Frame(
            self.results_frame, bg=self.theme["bg_card"],
            highlightbackground=self.theme["border"], highlightthickness=1,
        )
        self.results_card.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        header = tk.Frame(self.results_card, bg=self.theme["bg_card"])
        header.pack(fill=tk.X)
        
        tk.Label(
            header, text="Report Status",
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
            padx=12, pady=8,
        ).pack(side=tk.LEFT)
        
        ttk.Separator(self.results_card, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12)
        
        canvas = tk.Canvas(self.results_card, bg=self.theme["bg_card"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.results_card, orient=tk.VERTICAL, command=canvas.yview)
        
        self.preview_frame = tk.Frame(canvas, bg=self.theme["bg_card"])
        self.preview_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.preview_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        
        self._show_initial_status()
    
    def _show_initial_status(self):
        """Show initial status message."""
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.preview_frame,
            text="Configure report options and click 'Generate Report'\n\n"
                 "Tips:\n"
                 "• Use 'Check Data Availability' to see what's ready\n"
                 "• Auto-prepare will run required analyses\n"
                 "• Basic level generates fastest",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_muted"],
            justify=tk.LEFT,
        ).pack(pady=30, padx=20, anchor=tk.W)
    
    def _get_selected_level(self) -> str:
        """Get the selected report level."""
        level_str = self.level_combo.get()
        return level_str.split(" - ")[0].strip()
    
    def _get_selected_formats(self) -> List[str]:
        """Get list of selected output formats."""
        return [fmt for fmt, var in self.format_vars.items() if var.get()]
    
    def _browse_folder(self):
        """Browse for output folder."""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder.set(folder)
    
    def _show_loading(self, message: str = "Loading..."):
        """Show loading indicator."""
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.preview_frame,
            text=message,
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_muted"],
        ).pack(pady=50)
    
    def _show_error(self, message: str):
        """Show error message."""
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.preview_frame,
            text=f"❌ Error\n\n{message}",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["danger"],
            wraplength=400, justify=tk.LEFT,
        ).pack(pady=30, padx=20)
    
    def _check_availability(self):
        """Check data availability for selected report level."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self._show_loading("Checking data availability...")
        
        def do_check():
            try:
                from biblium import reportbib
                level = self._get_selected_level()
                status = reportbib.check_report_data_availability(
                    self.bib, template_sheet=level, verbose=False
                )
                self.after(0, lambda: self._show_availability(status))
            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: self._show_error(msg))
        
        threading.Thread(target=do_check, daemon=True).start()
    
    def _show_availability(self, status: Dict):
        """Display data availability status."""
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        
        summary = status.get("summary", {})
        available = status.get("available", [])
        missing_data = status.get("missing_data", [])
        missing_plots = status.get("missing_plots", [])
        
        coverage = summary.get("coverage_pct", 0)
        color = self.theme["success"] if coverage >= 80 else (
            self.theme["warning"] if coverage >= 50 else self.theme["danger"]
        )
        
        tk.Label(
            self.preview_frame,
            text=f"Report Coverage: {coverage}%",
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"], fg=color,
        ).pack(anchor=tk.W, pady=(0, 8))
        
        stats_text = (f"Total: {summary.get('total_items', 0)} | "
                     f"Available: {summary.get('available_items', 0)} | "
                     f"Missing Tables: {summary.get('missing_tables', 0)} | "
                     f"Missing Plots: {summary.get('missing_plots', 0)}")
        
        tk.Label(
            self.preview_frame, text=stats_text,
            font=FONTS.get_font("small"),
            bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
        ).pack(anchor=tk.W, pady=(0, 8))
        
        ttk.Separator(self.preview_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        
        if available:
            tk.Label(
                self.preview_frame,
                text=f"✓ Available Items ({len(available)})",
                font=FONTS.get_font("body", bold=True),
                bg=self.theme["bg_card"], fg=self.theme["success"],
            ).pack(anchor=tk.W, pady=(8, 4))
            
            for item in available[:8]:
                name = item.get("name", "")
                rows = item.get("rows", "")
                text = f"  • {name}" + (f" ({rows} rows)" if rows else "")
                tk.Label(
                    self.preview_frame, text=text,
                    font=FONTS.get_font("small"),
                    bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
                ).pack(anchor=tk.W)
            
            if len(available) > 8:
                tk.Label(
                    self.preview_frame,
                    text=f"  ... and {len(available) - 8} more",
                    font=FONTS.get_font("small"),
                    bg=self.theme["bg_card"], fg=self.theme["text_muted"],
                ).pack(anchor=tk.W)
        
        if missing_data:
            tk.Label(
                self.preview_frame,
                text=f"✗ Missing Tables ({len(missing_data)})",
                font=FONTS.get_font("body", bold=True),
                bg=self.theme["bg_card"], fg=self.theme["danger"],
            ).pack(anchor=tk.W, pady=(12, 4))
            
            for item in missing_data[:5]:
                tk.Label(
                    self.preview_frame,
                    text=f"  • {item.get('name', '')} ({item.get('attr', '')})",
                    font=FONTS.get_font("small"),
                    bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
                ).pack(anchor=tk.W)
        
        if missing_plots:
            tk.Label(
                self.preview_frame,
                text=f"⚠ Missing Plots ({len(missing_plots)})",
                font=FONTS.get_font("body", bold=True),
                bg=self.theme["bg_card"], fg=self.theme["warning"],
            ).pack(anchor=tk.W, pady=(12, 4))
            
            for item in missing_plots[:5]:
                tk.Label(
                    self.preview_frame,
                    text=f"  • {item.get('name', '')}",
                    font=FONTS.get_font("small"),
                    bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
                ).pack(anchor=tk.W)
        
        if missing_data or missing_plots:
            tk.Label(
                self.preview_frame,
                text="\n💡 Enable 'Auto-prepare data' to generate missing items",
                font=FONTS.get_font("small"),
                bg=self.theme["bg_card"], fg=self.theme["accent_primary"],
            ).pack(anchor=tk.W, pady=(12, 0))
    
    def _preview_report(self):
        """Generate a quick preview."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self._show_loading("Generating preview...")
        
        def do_preview():
            try:
                level = self._get_selected_level()
                preview_text = self.bib.preview_report(level=level)
                self.after(0, lambda: self._show_text_preview(preview_text))
            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: self._show_error(msg))
        
        threading.Thread(target=do_preview, daemon=True).start()
    
    def _show_text_preview(self, text: str):
        """Display text preview."""
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.preview_frame, text="Report Preview",
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(anchor=tk.W, pady=(0, 8))
        
        text_widget = tk.Text(
            self.preview_frame,
            font=FONTS.get_font("small"),
            bg=self.theme["bg_secondary"],
            fg=self.theme["text_primary"],
            wrap=tk.WORD, height=20, padx=8, pady=8,
        )
        text_widget.pack(fill=tk.BOTH, expand=True, pady=8)
        text_widget.insert("1.0", text)
        text_widget.config(state=tk.DISABLED)
    
    def _generate_report(self):
        """Generate the report using core library."""
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        level = self._get_selected_level()
        formats = self._get_selected_formats()
        output_folder = self.output_folder.get() or "results/reports"
        report_name = self.report_name.get() or "bibliometric_report"
        auto_prepare = self.auto_prepare.get()
        gen_plots = self.generate_plots.get()
        
        if not formats:
            messagebox.showwarning("No Format", "Please select at least one output format.")
            return
        
        self._show_loading(f"Generating {level.upper()} report...")
        
        def do_generate():
            try:
                if auto_prepare:
                    self.after(0, lambda: self._show_loading("Preparing data..."))
                    self.bib.prepare_for_report(
                        level=level, verbose=False, generate_plots=gen_plots
                    )
                
                self.after(0, lambda: self._show_loading("Generating reports..."))
                
                output_path = os.path.join(output_folder, report_name)
                result = self.bib.generate_report(
                    output=output_path,
                    level=level,
                    formats=formats,
                    prepare=False,
                    verbose=False,
                )
                
                self.after(0, lambda: self._show_success(result, output_folder))
                
            except Exception as e:
                import traceback
                error_msg = f"{e}\n\n{traceback.format_exc()}"
                self.after(0, lambda msg=error_msg: self._show_error(msg))
        
        threading.Thread(target=do_generate, daemon=True).start()
    
    def _show_success(self, result: Dict, output_folder: str):
        """Display success message."""
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.preview_frame,
            text="✓ Report Generated Successfully",
            font=FONTS.get_font("heading", bold=True),
            bg=self.theme["bg_card"], fg=self.theme["success"],
        ).pack(anchor=tk.W, pady=(0, 16))
        
        tk.Label(
            self.preview_frame, text="Generated files:",
            font=FONTS.get_font("body"),
            bg=self.theme["bg_card"], fg=self.theme["text_primary"],
        ).pack(anchor=tk.W)
        
        files_generated = []
        actual_folder = None
        for fmt, path in result.items():
            if not str(path).startswith("Error"):
                files_generated.append((fmt, path))
                # Get actual folder from the first successful file
                if actual_folder is None:
                    actual_folder = str(Path(path).parent)
                tk.Label(
                    self.preview_frame,
                    text=f"  ✓ {fmt.upper()}: {path}",
                    font=FONTS.get_font("small"),
                    bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
                ).pack(anchor=tk.W)
            else:
                tk.Label(
                    self.preview_frame,
                    text=f"  ✗ {fmt.upper()}: {path}",
                    font=FONTS.get_font("small"),
                    bg=self.theme["bg_card"], fg=self.theme["danger"],
                ).pack(anchor=tk.W)
        
        # Use actual folder where files were saved, not the configured output_folder
        folder_to_open = actual_folder or output_folder
        
        btn_frame = tk.Frame(self.preview_frame, bg=self.theme["bg_card"])
        btn_frame.pack(fill=tk.X, pady=16)
        
        ThemedButton(
            btn_frame, text="Open Folder", style="primary",
            command=lambda: self._open_folder(folder_to_open),
            theme=self.theme_name,
        ).pack(side=tk.LEFT)
        
        # Add individual buttons for each generated file
        if files_generated:
            for fmt, path in files_generated:
                btn_text = f"Open {fmt.upper()}"
                ThemedButton(
                    btn_frame, text=btn_text, style="secondary",
                    command=lambda p=path: self._open_file(p),
                    theme=self.theme_name,
                ).pack(side=tk.LEFT, padx=(8, 0))
    
    def _open_folder(self, path: str):
        """Open folder in file explorer."""
        import platform
        import subprocess
        
        try:
            p = Path(path)
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
            
            if platform.system() == "Windows":
                os.startfile(str(p))
            elif platform.system() == "Darwin":
                subprocess.run(["open", str(p)])
            else:
                subprocess.run(["xdg-open", str(p)])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder: {e}")
    
    def _open_file(self, path: str):
        """Open file with default application."""
        import platform
        import subprocess
        
        try:
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                subprocess.run(["open", path])
            else:
                subprocess.run(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file: {e}")
