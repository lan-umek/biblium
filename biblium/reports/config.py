# -*- coding: utf-8 -*-
"""
Report Configuration Module.

Defines configuration classes for report generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ReportLevel(Enum):
    """Report detail levels."""
    BASIC = "basic"
    STANDARD = "standard"
    EXTENDED = "extended"
    FULL = "full"
    CUSTOM = "custom"


class ReportFormat(Enum):
    """Supported report formats."""
    XLSX = "xlsx"
    DOCX = "docx"
    PPTX = "pptx"
    LATEX = "tex"
    HTML = "html"
    PDF = "pdf"  # Via LaTeX or HTML
    ALL = "all"


@dataclass
class ReportStyle:
    """Styling configuration for reports."""
    
    # Colors (hex without #)
    primary_color: str = "1F4E79"
    secondary_color: str = "2E75B6"
    accent_color: str = "70AD47"
    background_color: str = "FFFFFF"
    text_color: str = "333333"
    
    # Section colors
    section_colors: Dict[str, str] = field(default_factory=lambda: {
        "main_info": "E2EFDA",
        "counts": "FFF2CC",
        "stats": "FCE4D6",
        "plots": "DDEBF7",
        "networks": "E2D5F1",
        "advanced": "F4CCCC",
        "time": "D9EAD3",
        "laws": "CFE2F3",
    })
    
    # Typography
    title_font: str = "Arial"
    body_font: str = "Arial"
    title_size: int = 24
    heading1_size: int = 18
    heading2_size: int = 14
    body_size: int = 11
    
    # Spacing
    margin_top: float = 1.0  # inches
    margin_bottom: float = 1.0
    margin_left: float = 1.0
    margin_right: float = 1.0
    
    # Logo/branding
    logo_path: Optional[str] = None
    company_name: Optional[str] = None
    
    @classmethod
    def default(cls) -> "ReportStyle":
        """Default professional style."""
        return cls()
    
    @classmethod
    def academic(cls) -> "ReportStyle":
        """Academic/publication style."""
        return cls(
            primary_color="000000",
            secondary_color="666666",
            title_font="Times New Roman",
            body_font="Times New Roman",
            title_size=16,
            heading1_size=14,
            heading2_size=12,
            body_size=11,
        )
    
    @classmethod
    def modern(cls) -> "ReportStyle":
        """Modern/startup style."""
        return cls(
            primary_color="6366F1",  # Indigo
            secondary_color="8B5CF6",  # Purple
            accent_color="10B981",  # Emerald
            title_font="Helvetica",
            body_font="Helvetica",
        )


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    
    # Basic settings
    title: str = "Bibliometric Analysis Report"
    subtitle: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None  # Auto-generated if None
    
    # Content control
    level: ReportLevel = ReportLevel.STANDARD
    include_toc: bool = True
    include_summary: bool = True
    include_methodology: bool = False
    include_appendix: bool = False
    
    # Item limits
    top_n_tables: int = 20  # Max rows in top-N tables
    top_n_plots: int = 15   # Max items in bar charts
    
    # Plot settings
    plot_dpi: int = 300
    plot_format: str = "png"  # png, svg, pdf
    regenerate_plots: bool = True  # Regenerate or use existing
    
    # Styling
    style: ReportStyle = field(default_factory=ReportStyle.default)
    
    # Output settings
    output_dir: Optional[str] = None
    filename_prefix: str = "report"
    
    # Template
    template_path: Optional[str] = None  # Custom template xlsx
    
    # Format-specific settings
    xlsx_settings: Dict[str, Any] = field(default_factory=lambda: {
        "include_charts": True,
        "autofit_columns": True,
        "freeze_panes": True,
        "conditional_formatting": True,
    })
    
    docx_settings: Dict[str, Any] = field(default_factory=lambda: {
        "page_size": "A4",  # A4, Letter
        "orientation": "portrait",
        "include_page_numbers": True,
        "include_header": True,
        "include_footer": True,
    })
    
    pptx_settings: Dict[str, Any] = field(default_factory=lambda: {
        "aspect_ratio": "16:9",  # 16:9, 4:3
        "include_notes": True,
        "include_slide_numbers": True,
        "max_slides": 50,
    })
    
    latex_settings: Dict[str, Any] = field(default_factory=lambda: {
        "document_class": "article",
        "paper_size": "a4paper",
        "font_size": "11pt",
        "bibliography_style": "apalike",
        "compile_pdf": False,
    })
    
    html_settings: Dict[str, Any] = field(default_factory=lambda: {
        "theme": "light",  # light, dark
        "interactive": True,
        "include_search": True,
        "single_file": True,  # Embed all resources
        "include_download_links": True,
    })
    
    def with_level(self, level: Union[str, ReportLevel]) -> "ReportConfig":
        """Return new config with different level."""
        if isinstance(level, str):
            level = ReportLevel(level)
        return ReportConfig(
            **{**self.__dict__, "level": level}
        )
    
    def with_style(self, style: ReportStyle) -> "ReportConfig":
        """Return new config with different style."""
        return ReportConfig(
            **{**self.__dict__, "style": style}
        )
    
    @classmethod
    def basic(cls, title: str = "Quick Analysis") -> "ReportConfig":
        """Minimal report configuration."""
        return cls(
            title=title,
            level=ReportLevel.BASIC,
            include_toc=False,
            include_summary=True,
            top_n_tables=10,
            top_n_plots=10,
        )
    
    @classmethod
    def standard(cls, title: str = "Bibliometric Analysis") -> "ReportConfig":
        """Standard report configuration."""
        return cls(
            title=title,
            level=ReportLevel.STANDARD,
        )
    
    @classmethod
    def comprehensive(cls, title: str = "Comprehensive Bibliometric Analysis") -> "ReportConfig":
        """Full detailed report configuration."""
        return cls(
            title=title,
            level=ReportLevel.FULL,
            include_methodology=True,
            include_appendix=True,
            top_n_tables=50,
            top_n_plots=25,
        )
    
    @classmethod
    def presentation(cls, title: str = "Research Overview") -> "ReportConfig":
        """Configuration optimized for presentations."""
        return cls(
            title=title,
            level=ReportLevel.BASIC,
            include_toc=False,
            top_n_tables=10,
            top_n_plots=10,
            pptx_settings={
                "aspect_ratio": "16:9",
                "include_notes": True,
                "include_slide_numbers": True,
                "max_slides": 20,
            }
        )
