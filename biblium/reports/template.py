# -*- coding: utf-8 -*-
"""
Report Template Module.

Handles loading and parsing of report templates from Excel files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from biblium.reports.config import ReportLevel


@dataclass
class ReportItem:
    """Single item in a report (table or plot)."""
    
    level1: str              # Section name
    level2: str              # Subsection name
    item_type: str           # "Table", "Plot", "Text"
    data_attr: Optional[str] # DataFrame attribute name
    plot_method: Optional[str]  # Method on ba.plot
    plot_filename: Optional[str]
    caption: str
    description: str
    section: str             # Section color key
    page_break: bool
    table_style: Optional[str]
    slide_layout: Optional[str]
    
    @property
    def is_table(self) -> bool:
        return self.item_type == "Table"
    
    @property
    def is_plot(self) -> bool:
        return self.item_type == "Plot"
    
    @property
    def is_text(self) -> bool:
        return self.item_type == "Text"
    
    def get_data(self, biblio: Any) -> Optional[pd.DataFrame]:
        """Get data from BiblioStats instance."""
        if not self.data_attr:
            return None
        
        # Handle nested attributes
        obj = biblio
        for attr in self.data_attr.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                return None
        
        return obj if isinstance(obj, pd.DataFrame) else None
    
    def has_data(self, biblio: Any) -> bool:
        """Check if data is available."""
        if self.is_plot and self.plot_method:
            # Check if plot method exists
            if hasattr(biblio, "plot"):
                return hasattr(biblio.plot, self.plot_method)
        
        if self.data_attr:
            data = self.get_data(biblio)
            return data is not None and len(data) > 0
        
        return False


class ReportTemplate:
    """
    Report template manager.
    
    Loads and manages report templates from Excel files.
    
    Parameters
    ----------
    template_path : str or Path, optional
        Path to custom template file. If None, uses built-in template.
    
    Examples
    --------
    >>> template = ReportTemplate()
    >>> items = template.get_items("standard")
    >>> for item in items:
    ...     print(f"{item.level1} / {item.level2}: {item.item_type}")
    """
    
    # Default template embedded in package
    DEFAULT_TEMPLATE_PATH = None  # Will use embedded data
    
    def __init__(self, template_path: Optional[str] = None):
        self.template_path = template_path
        self._items_cache: Dict[str, List[ReportItem]] = {}
        self._df_all: Optional[pd.DataFrame] = None
        
        self._load_template()
    
    def _get_default_template(self) -> pd.DataFrame:
        """Return default template as DataFrame."""
        # Embedded template data
        template_data = [
            # Main Information
            ("Main Information", "Dataset Overview", "Table", "descriptives_df", None, None,
             "Dataset Overview", "Basic statistics about the bibliometric dataset", "main_info", False, "info", "title_content"),
            ("Main Information", "Performance Indicators", "Table", "performances_df", None, None,
             "Performance Indicators", "Key bibliometric performance metrics", "main_info", False, "metrics", "two_column"),
            ("Main Information", "Scientific Production", "Table", "production_df", None, None,
             "Annual Production", "Publications per year", "main_info", False, "timeline", "content"),
            ("Main Information", "Scientific Production", "Plot", None, "production_line", "scientific_production.png",
             "Production Trend", "Annual scientific production over time", "main_info", False, None, "image_full"),
            ("Main Information", "Missing Values", "Table", "missings_df", None, None,
             "Data Quality", "Missing values analysis", "main_info", True, "quality", "content"),
            
            # Item Counts
            ("Item Counts", "Sources", "Table", "sources_counts_df", None, None,
             "Source Distribution", "Publications by source", "counts", False, "ranked", "content"),
            ("Item Counts", "Sources", "Plot", None, "sources_bar", "sources_bar.png",
             "Top Sources", "Most productive sources", "counts", False, None, "image_full"),
            ("Item Counts", "Authors", "Table", "authors_counts_df", None, None,
             "Author Productivity", "Publications by author", "counts", False, "ranked", "content"),
            ("Item Counts", "Authors", "Plot", None, "authors_bar", "authors_bar.png",
             "Top Authors", "Most productive authors", "counts", False, None, "image_full"),
            ("Item Counts", "Author Keywords", "Table", "author_keywords_counts_df", None, None,
             "Keyword Frequency", "Keyword counts", "counts", False, "ranked", "content"),
            ("Item Counts", "Author Keywords", "Plot", None, "keywords_bar", "keywords_bar.png",
             "Top Keywords", "Most frequent keywords", "counts", False, None, "image_full"),
            ("Item Counts", "Document Types", "Table", "document_types_counts_df", None, None,
             "Document Types", "Distribution by type", "counts", False, "default", "content"),
            ("Item Counts", "Document Types", "Plot", None, "doctypes_bar", "doctypes_bar.png",
             "Document Types", "Bar chart of document type distribution", "counts", False, None, "image_half"),
            ("Item Counts", "Countries", "Table", "ca_country_counts_df", None, None,
             "Countries", "Publications by country", "counts", False, "ranked", "content"),
            ("Item Counts", "Countries", "Plot", None, "countries_bar", "countries_bar.png",
             "Top Countries", "Most productive countries", "counts", True, None, "image_full"),
            
            # Performance Stats
            ("Performance Stats", "Sources", "Table", "sources_stats_df", None, None,
             "Source Metrics", "Performance by source", "stats", False, "metrics", "content"),
            ("Performance Stats", "Authors", "Table", "authors_stats_df", None, None,
             "Author Metrics", "Performance by author", "stats", False, "metrics", "content"),
            ("Performance Stats", "Keywords", "Table", "author_keywords_stats_df", None, None,
             "Keyword Metrics", "Performance by keyword", "stats", True, "metrics", "content"),
            
            # Visualizations
            ("Visualizations", "Citations", "Plot", None, "citations_histogram", "citations_histogram.png",
             "Citation Distribution", "Citation count distribution", "plots", False, None, "image_full"),
            ("Visualizations", "Keywords", "Plot", None, "keywords_wordcloud", "keywords_wordcloud.png",
             "Keyword Cloud", "Keyword visualization", "plots", False, None, "image_full"),
            ("Visualizations", "Cumulative", "Plot", None, "production_area", "production_area.png",
             "Cumulative Growth", "Publication growth", "plots", True, None, "image_full"),
            
            # Co-occurrence
            ("Co-occurrence", "Keywords", "Plot", None, "cooccurrence_heatmap", "keyword_heatmap.png",
             "Keyword Heatmap", "Keyword co-occurrence", "networks", False, None, "image_full"),
            ("Co-occurrence", "Network", "Plot", None, "keyword_network", "keyword_network.png",
             "Keyword Network", "Keyword relationships", "networks", True, None, "image_full"),
            
            # Advanced
            ("Advanced", "Top Cited", "Table", "top_cited_docs_global_df", None, None,
             "Most Cited Papers", "Globally most cited", "advanced", True, "highlight", "content"),
        ]
        
        columns = ["Level 1", "Level 2", "Item Type", "Data Attr", "Plot Method", 
                  "Plot Filename", "Caption", "Description", "Section", "Page Break",
                  "Table Style", "Slide Layout"]
        
        return pd.DataFrame(template_data, columns=columns)
    
    def _load_template(self) -> None:
        """Load template from file or use default."""
        if self.template_path and os.path.exists(self.template_path):
            try:
                xlsx = pd.ExcelFile(self.template_path)
                if "all" in xlsx.sheet_names:
                    self._df_all = pd.read_excel(xlsx, sheet_name="all")
                else:
                    # Use first sheet
                    self._df_all = pd.read_excel(xlsx, sheet_name=0)
            except Exception as e:
                print(f"Warning: Could not load template from {self.template_path}: {e}")
                self._df_all = self._get_default_template()
        else:
            self._df_all = self._get_default_template()
    
    def _parse_items(self, df: pd.DataFrame) -> List[ReportItem]:
        """Parse DataFrame rows into ReportItem objects."""
        items = []
        for _, row in df.iterrows():
            item = ReportItem(
                level1=str(row.get("Level 1", "")),
                level2=str(row.get("Level 2", "")),
                item_type=str(row.get("Item Type", "Table")),
                data_attr=row.get("Data Attr") if pd.notna(row.get("Data Attr")) else None,
                plot_method=row.get("Plot Method") if pd.notna(row.get("Plot Method")) else None,
                plot_filename=row.get("Plot Filename") if pd.notna(row.get("Plot Filename")) else None,
                caption=str(row.get("Caption", row.get("Level 2", ""))),
                description=str(row.get("Description", "")),
                section=str(row.get("Section", "default")),
                page_break=bool(row.get("Page Break", False)),
                table_style=row.get("Table Style") if pd.notna(row.get("Table Style")) else None,
                slide_layout=row.get("Slide Layout") if pd.notna(row.get("Slide Layout")) else None,
            )
            items.append(item)
        return items
    
    def get_items(self, level: str = "standard") -> List[ReportItem]:
        """
        Get report items for a specific level.
        
        Parameters
        ----------
        level : str
            Report level: "basic", "standard", "extended", "full", or "all"
        
        Returns
        -------
        list of ReportItem
            Items to include in report
        """
        if level in self._items_cache:
            return self._items_cache[level]
        
        if self.template_path and os.path.exists(self.template_path):
            try:
                xlsx = pd.ExcelFile(self.template_path)
                if level in xlsx.sheet_names:
                    df = pd.read_excel(xlsx, sheet_name=level)
                    items = self._parse_items(df)
                    self._items_cache[level] = items
                    return items
            except Exception:
                pass
        
        # Filter from all items based on level
        level_sections = {
            "basic": ["Main Information", "Item Counts"],
            "standard": ["Main Information", "Item Counts", "Performance Stats", "Visualizations"],
            "extended": ["Main Information", "Item Counts", "Performance Stats", "Visualizations",
                        "Production Over Time", "Co-occurrence"],
            "full": None,  # All sections
            "all": None,
        }
        
        sections = level_sections.get(level)
        
        if sections:
            df = self._df_all[self._df_all["Level 1"].isin(sections)]
        else:
            df = self._df_all
        
        items = self._parse_items(df)
        self._items_cache[level] = items
        return items
    
    def get_sections(self, level: str = "standard") -> List[str]:
        """Get unique section names for a level."""
        items = self.get_items(level)
        seen = set()
        sections = []
        for item in items:
            if item.level1 not in seen:
                seen.add(item.level1)
                sections.append(item.level1)
        return sections
    
    def get_items_by_section(self, level: str = "standard") -> Dict[str, List[ReportItem]]:
        """Get items grouped by section."""
        items = self.get_items(level)
        by_section: Dict[str, List[ReportItem]] = {}
        for item in items:
            if item.level1 not in by_section:
                by_section[item.level1] = []
            by_section[item.level1].append(item)
        return by_section
    
    def filter_available(self, biblio: Any, level: str = "standard") -> List[ReportItem]:
        """Filter items to only those with available data."""
        items = self.get_items(level)
        return [item for item in items if item.has_data(biblio)]
