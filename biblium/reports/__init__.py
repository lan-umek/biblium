# -*- coding: utf-8 -*-
"""
Biblium Report Generation Module.

This module provides comprehensive report generation capabilities for bibliometric analyses.
Supports multiple output formats: XLSX, DOCX, PPTX, LaTeX, and HTML.

Usage
-----
    from biblium import BiblioAnalysis
    from biblium.reports import ReportGenerator, ReportConfig
    
    ba = BiblioAnalysis("data.csv", db="scopus")
    
    # Generate reports
    report = ReportGenerator(ba)
    report.generate("xlsx", level="standard")
    report.generate("pptx", level="basic")
    report.generate("html", level="full")
    
    # Or generate all formats
    report.generate_all(level="standard")
"""

from biblium.reports.config import ReportConfig, ReportLevel
from biblium.reports.generator import ReportGenerator
from biblium.reports.template import ReportTemplate

__all__ = [
    "ReportConfig",
    "ReportLevel", 
    "ReportGenerator",
    "ReportTemplate",
]
