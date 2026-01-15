# -*- coding: utf-8 -*-
"""
Dashboard Generator Module.

Creates standalone interactive HTML dashboards using Bokeh.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from biblium.bibstats import BiblioStats


@dataclass
class DashboardConfig:
    """Configuration for dashboard generation."""
    
    title: str = "Bibliometric Analysis Dashboard"
    subtitle: Optional[str] = None
    author: Optional[str] = None
    
    # Theme
    theme: str = "light"  # light, dark
    primary_color: str = "#1F4E79"
    accent_color: str = "#70AD47"
    
    # Layout
    width: int = 1200
    plot_height: int = 400
    table_height: int = 300
    
    # Content
    top_n: int = 15
    include_tables: bool = True
    include_data_download: bool = True
    include_year_slider: bool = True
    
    # Sections to include
    sections: List[str] = field(default_factory=lambda: [
        "overview",
        "production",
        "sources",
        "authors", 
        "keywords",
        "wordcloud",
        "keyword_trends",
        "sankey",
        "keyword_network",
        "collaboration_network",
        "scatter",
        "citations",
        "citation_classics",
        "reference_analysis",
        "countries",
        "documents",
    ])


class Dashboard:
    """
    Interactive HTML dashboard generator.
    
    Creates a standalone HTML file with interactive Bokeh visualizations
    that can be shared and viewed in any web browser.
    
    Parameters
    ----------
    biblio : BiblioStats
        The BiblioStats/BiblioAnalysis instance with analysis results.
    config : DashboardConfig, optional
        Dashboard configuration.
    
    Examples
    --------
    >>> ba = BiblioAnalysis("data.csv", db="scopus")
    >>> dashboard = Dashboard(ba)
    >>> dashboard.create("analysis.html")
    
    >>> # With custom config
    >>> config = DashboardConfig(title="My Research", theme="dark")
    >>> dashboard = Dashboard(ba, config=config)
    >>> dashboard.create("dark_dashboard.html")
    """
    
    def __init__(
        self,
        biblio: "BiblioStats",
        config: Optional[DashboardConfig] = None,
    ):
        self.biblio = biblio
        self.config = config or DashboardConfig()
        
        # Ensure data is available
        self._prepare_data()
    
    def _prepare_data(self) -> None:
        """Ensure required data is computed."""
        b = self.biblio
        
        # Run counts if not already done
        if not hasattr(b, "sources_counts_df") or b.sources_counts_df is None:
            try:
                b.count_sources()
            except:
                pass
        
        if not hasattr(b, "authors_counts_df") or b.authors_counts_df is None:
            try:
                b.count_authors()
            except:
                pass
        
        if not hasattr(b, "author_keywords_counts_df") or b.author_keywords_counts_df is None:
            try:
                b.count_author_keywords()
            except:
                pass
        
        if not hasattr(b, "document_types_counts_df") or b.document_types_counts_df is None:
            try:
                b.count_document_types()
            except:
                pass
        
        if not hasattr(b, "production_df") or b.production_df is None:
            try:
                b.get_production()
            except:
                pass
        
        # Compute stats for scatter plots (h-index, etc.)
        if not hasattr(b, "sources_stats_df") or b.sources_stats_df is None:
            try:
                b.get_sources_stats()
            except:
                pass
        
        if not hasattr(b, "authors_stats_df") or b.authors_stats_df is None:
            try:
                b.get_authors_stats()
            except:
                pass
        
        if not hasattr(b, "author_keywords_stats_df") or b.author_keywords_stats_df is None:
            try:
                b.get_author_keywords_stats()
            except:
                pass
    
    def create(
        self,
        output_path: str = "dashboard.html",
        title: Optional[str] = None,
        theme: Optional[str] = None,
    ) -> Path:
        """
        Create the interactive dashboard.
        
        Parameters
        ----------
        output_path : str
            Output file path for the HTML dashboard.
        title : str, optional
            Override dashboard title.
        theme : str, optional
            Override theme ("light" or "dark").
        
        Returns
        -------
        Path
            Path to the generated dashboard file.
        """
        if title:
            self.config.title = title
        if theme:
            self.config.theme = theme
        
        # Build HTML
        html = self._build_html()
        
        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        print(f"✓ Dashboard created: {output_path}")
        return output_path
    
    def _build_html(self) -> str:
        """Build the complete HTML document."""
        css = self._get_css()
        js = self._get_js()
        
        # Build sections
        sections_html = []
        
        if "overview" in self.config.sections:
            sections_html.append(self._build_overview_section())
        
        if "production" in self.config.sections:
            sections_html.append(self._build_production_section())
        
        if "sources" in self.config.sections:
            sections_html.append(self._build_sources_section())
        
        if "authors" in self.config.sections:
            sections_html.append(self._build_authors_section())
        
        if "keywords" in self.config.sections:
            sections_html.append(self._build_keywords_section())
        
        if "wordcloud" in self.config.sections:
            sections_html.append(self._build_wordcloud_section())
        
        if "keyword_trends" in self.config.sections:
            sections_html.append(self._build_keyword_trends_section())
        
        if "sankey" in self.config.sections:
            sections_html.append(self._build_sankey_section())
        
        if "keyword_network" in self.config.sections:
            sections_html.append(self._build_keyword_network_section())
        
        if "collaboration_network" in self.config.sections:
            sections_html.append(self._build_collaboration_network_section())
        
        if "scatter" in self.config.sections:
            sections_html.append(self._build_scatter_section())
        
        if "citations" in self.config.sections:
            sections_html.append(self._build_citations_section())
        
        if "citation_classics" in self.config.sections:
            sections_html.append(self._build_citation_classics_section())
        
        if "reference_analysis" in self.config.sections:
            sections_html.append(self._build_reference_analysis_section())
        
        if "countries" in self.config.sections:
            sections_html.append(self._build_countries_section())
        
        if "documents" in self.config.sections:
            sections_html.append(self._build_documents_section())
        
        # Navigation
        nav_html = self._build_navigation()
        
        # Combine
        body_content = "\n".join(sections_html)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title}</title>
    <script src="https://cdn.bokeh.org/bokeh/release/bokeh-3.8.1.min.js"></script>
    <script src="https://cdn.bokeh.org/bokeh/release/bokeh-gl-3.8.1.min.js"></script>
    <script src="https://cdn.bokeh.org/bokeh/release/bokeh-widgets-3.8.1.min.js"></script>
    <script src="https://cdn.bokeh.org/bokeh/release/bokeh-tables-3.8.1.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
{css}
    </style>
</head>
<body class="{self.config.theme}">
    <div class="dashboard-container">
        <header class="dashboard-header">
            <h1>{self.config.title}</h1>
            {f'<p class="subtitle">{self.config.subtitle}</p>' if self.config.subtitle else ''}
            <p class="meta">
                {self.biblio.n} documents | 
                Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}
                {f' | {self.config.author}' if self.config.author else ''}
            </p>
        </header>
        {nav_html}
        {self._build_year_slider() if self.config.include_year_slider else ''}
        <main class="dashboard-content">
            {body_content}
        </main>
        <footer class="dashboard-footer">
            <p>Generated by Biblium | Interactive Bibliometric Dashboard</p>
        </footer>
    </div>
    
    <script>
{js}
    </script>
</body>
</html>"""
        
        return html
    
    def _get_css(self) -> str:
        """Get CSS styles for the dashboard."""
        c = self.config
        
        if c.theme == "dark":
            bg_color = "#1a1a2e"
            card_bg = "#16213e"
            text_color = "#eaeaea"
            text_muted = "#a0a0a0"
            border_color = "#0f3460"
            header_bg = f"linear-gradient(135deg, {c.primary_color}, #0f3460)"
        else:
            bg_color = "#f5f7fa"
            card_bg = "#ffffff"
            text_color = "#333333"
            text_muted = "#666666"
            border_color = "#e0e0e0"
            header_bg = f"linear-gradient(135deg, {c.primary_color}, {c.accent_color})"
        
        return f"""
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: {bg_color};
            color: {text_color};
            line-height: 1.5;
        }}
        
        .dashboard-container {{
            max-width: {c.width}px;
            margin: 0 auto;
            padding: 8px 15px;
        }}
        
        .dashboard-header {{
            background: {header_bg};
            color: white;
            padding: 15px 25px;
            border-radius: 8px;
            margin-bottom: 8px;
            text-align: center;
        }}
        
        .dashboard-header h1 {{
            font-size: 1.6rem;
            margin-bottom: 3px;
            font-weight: 700;
        }}
        
        .dashboard-header .subtitle {{
            font-size: 0.95rem;
            opacity: 0.9;
            margin-bottom: 3px;
        }}
        
        .dashboard-header .meta {{
            font-size: 0.8rem;
            opacity: 0.8;
        }}
        
        .dashboard-nav {{
            background: {card_bg};
            border-radius: 6px;
            padding: 6px 10px;
            margin-bottom: 10px;
            border: 1px solid {border_color};
            position: sticky;
            top: 5px;
            z-index: 100;
        }}
        
        .dashboard-nav ul {{
            list-style: none;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: center;
        }}
        
        .dashboard-nav a {{
            color: {c.primary_color};
            text-decoration: none;
            padding: 5px 12px;
            border-radius: 15px;
            transition: all 0.2s;
            font-weight: 500;
            font-size: 0.9rem;
        }}
        
        .dashboard-nav a:hover {{
            background: {c.primary_color};
            color: white;
        }}
        
        .section {{
            background: {card_bg};
            border-radius: 8px;
            padding: 15px 20px;
            margin-bottom: 12px;
            border: 1px solid {border_color};
        }}
        
        .section-title {{
            font-size: 1.2rem;
            color: {c.primary_color};
            margin-bottom: 12px;
            padding-bottom: 6px;
            border-bottom: 2px solid {c.primary_color};
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 12px;
            margin-bottom: 15px;
        }}
        
        .stat-card {{
            background: {bg_color};
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 1.6rem;
            font-weight: 700;
            color: {c.primary_color};
        }}
        
        .stat-label {{
            color: {text_muted};
            font-size: 0.9rem;
            margin-top: 5px;
        }}
        
        .chart-container {{
            margin: 20px 0;
            min-height: {c.plot_height}px;
        }}
        
        .table-container {{
            overflow-x: auto;
            margin: 20px 0;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid {border_color};
        }}
        
        th {{
            background: {c.primary_color};
            color: white;
            font-weight: 600;
            position: sticky;
            top: 0;
        }}
        
        tr:hover {{
            background: {bg_color};
        }}
        
        .two-column {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }}
        
        @media (max-width: 768px) {{
            .two-column {{
                grid-template-columns: 1fr;
            }}
            .dashboard-header h1 {{
                font-size: 1.8rem;
            }}
            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
        
        .dashboard-footer {{
            text-align: center;
            padding: 20px;
            color: {text_muted};
            font-size: 0.85rem;
        }}
        
        .bk-root {{
            margin: 0 auto;
        }}
        
        .download-btn {{
            display: inline-block;
            padding: 8px 16px;
            background: {c.accent_color};
            color: white;
            border-radius: 5px;
            text-decoration: none;
            font-size: 0.85rem;
            margin-top: 10px;
        }}
        
        .download-btn:hover {{
            opacity: 0.9;
        }}
        """
    
    def _get_js(self) -> str:
        """Get JavaScript for interactivity."""
        return """
        // Smooth scroll for navigation
        document.querySelectorAll('.dashboard-nav a').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            });
        });
        
        // Highlight current section in nav
        const sections = document.querySelectorAll('.section');
        const navLinks = document.querySelectorAll('.dashboard-nav a');
        
        window.addEventListener('scroll', () => {
            let current = '';
            sections.forEach(section => {
                const sectionTop = section.offsetTop;
                if (scrollY >= sectionTop - 100) {
                    current = section.getAttribute('id');
                }
            });
            
            navLinks.forEach(link => {
                link.style.background = '';
                link.style.color = '';
                if (link.getAttribute('href') === '#' + current) {
                    link.style.background = link.style.color;
                }
            });
        });
        """
    
    def _build_navigation(self) -> str:
        """Build navigation bar."""
        links = []
        
        section_names = {
            "overview": "Overview",
            "production": "Production",
            "sources": "Sources",
            "authors": "Authors",
            "keywords": "Keywords",
            "wordcloud": "Word Cloud",
            "keyword_trends": "Trends",
            "sankey": "Three-Field",
            "keyword_network": "Keywords Net",
            "collaboration_network": "Collaboration",
            "scatter": "Scatter",
            "citations": "Citations",
            "citation_classics": "Top Papers",
            "reference_analysis": "References",
            "countries": "Countries",
            "documents": "Documents",
        }
        
        for section in self.config.sections:
            name = section_names.get(section, section.title())
            links.append(f'<li><a href="#{section}">{name}</a></li>')
        
        return f"""
        <nav class="dashboard-nav">
            <ul>
                {''.join(links)}
            </ul>
        </nav>
        """
    
    def _build_year_slider(self) -> str:
        """Build interactive year range slider."""
        b = self.biblio
        
        year_col = b._get_column("Year", required=False)
        if not year_col or year_col not in b.df.columns:
            return ""
        
        years = b.df[year_col].dropna()
        if len(years) == 0:
            return ""
        
        min_year = int(years.min())
        max_year = int(years.max())
        
        if min_year == max_year:
            return ""
        
        # Create year distribution data for the mini chart
        year_counts = years.value_counts().sort_index()
        chart_data = [{"year": int(y), "count": int(c)} for y, c in year_counts.items()]
        
        import json
        chart_data_json = json.dumps(chart_data)
        
        return f"""
        <div class="year-slider-container">
            <div class="year-slider-header">
                <span class="year-slider-label">📅 Filter by Year Range:</span>
                <span class="year-slider-value" id="year-range-display">{min_year} - {max_year}</span>
                <button class="year-slider-reset" onclick="resetYearFilter()">Reset</button>
            </div>
            <div class="year-slider-chart" id="year-mini-chart"></div>
            <div class="year-slider-controls">
                <input type="range" id="year-min" min="{min_year}" max="{max_year}" value="{min_year}" 
                       oninput="updateYearFilter()">
                <input type="range" id="year-max" min="{min_year}" max="{max_year}" value="{max_year}"
                       oninput="updateYearFilter()">
            </div>
            <div class="year-slider-labels">
                <span>{min_year}</span>
                <span>{max_year}</span>
            </div>
        </div>
        
        <style>
        .year-slider-container {{
            background: white;
            border-radius: 8px;
            padding: 12px 20px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .year-slider-header {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 10px;
        }}
        .year-slider-label {{
            font-weight: 600;
            color: #333;
        }}
        .year-slider-value {{
            background: {self.config.primary_color};
            color: white;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.9rem;
            font-weight: 500;
        }}
        .year-slider-reset {{
            background: #e0e0e0;
            border: none;
            padding: 4px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85rem;
            margin-left: auto;
        }}
        .year-slider-reset:hover {{
            background: #ccc;
        }}
        .year-slider-chart {{
            height: 40px;
            margin-bottom: 5px;
        }}
        .year-slider-controls {{
            position: relative;
            height: 20px;
        }}
        .year-slider-controls input[type="range"] {{
            position: absolute;
            width: 100%;
            height: 5px;
            background: transparent;
            pointer-events: none;
            -webkit-appearance: none;
        }}
        .year-slider-controls input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            background: {self.config.primary_color};
            border-radius: 50%;
            cursor: pointer;
            pointer-events: auto;
            border: 2px solid white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .year-slider-controls input[type="range"]::-moz-range-thumb {{
            width: 18px;
            height: 18px;
            background: {self.config.primary_color};
            border-radius: 50%;
            cursor: pointer;
            pointer-events: auto;
            border: 2px solid white;
        }}
        .year-slider-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: #666;
            margin-top: 5px;
        }}
        .year-filtered {{
            opacity: 0.3;
            transition: opacity 0.3s;
        }}
        </style>
        
        <script>
        const yearChartData = {chart_data_json};
        const minYearGlobal = {min_year};
        const maxYearGlobal = {max_year};
        
        // Draw mini bar chart
        function drawYearMiniChart() {{
            const container = document.getElementById('year-mini-chart');
            if (!container || yearChartData.length === 0) return;
            
            const maxCount = Math.max(...yearChartData.map(d => d.count));
            const barWidth = 100 / yearChartData.length;
            
            let html = '<div style="display: flex; align-items: flex-end; height: 100%; gap: 1px;">';
            yearChartData.forEach(d => {{
                const height = (d.count / maxCount) * 100;
                html += `<div class="year-bar" data-year="${{d.year}}" style="flex: 1; height: ${{height}}%; background: {self.config.primary_color}; opacity: 0.6; border-radius: 2px 2px 0 0;" title="${{d.year}}: ${{d.count}} docs"></div>`;
            }});
            html += '</div>';
            container.innerHTML = html;
        }}
        
        function updateYearFilter() {{
            const minSlider = document.getElementById('year-min');
            const maxSlider = document.getElementById('year-max');
            
            let minVal = parseInt(minSlider.value);
            let maxVal = parseInt(maxSlider.value);
            
            // Ensure min doesn't exceed max
            if (minVal > maxVal) {{
                if (this === minSlider) {{
                    maxSlider.value = minVal;
                    maxVal = minVal;
                }} else {{
                    minSlider.value = maxVal;
                    minVal = maxVal;
                }}
            }}
            
            // Update display
            document.getElementById('year-range-display').textContent = `${{minVal}} - ${{maxVal}}`;
            
            // Update mini chart highlighting
            document.querySelectorAll('.year-bar').forEach(bar => {{
                const year = parseInt(bar.dataset.year);
                bar.style.opacity = (year >= minVal && year <= maxVal) ? '0.8' : '0.2';
            }});
            
            // Filter tables (add year-filtered class to rows outside range)
            document.querySelectorAll('table tbody tr').forEach(row => {{
                const yearCell = row.querySelector('td:first-child');
                if (yearCell) {{
                    const yearText = yearCell.textContent.trim();
                    const year = parseInt(yearText);
                    if (!isNaN(year) && year >= 1900 && year <= 2100) {{
                        row.classList.toggle('year-filtered', year < minVal || year > maxVal);
                    }}
                }}
            }});
        }}
        
        function resetYearFilter() {{
            document.getElementById('year-min').value = minYearGlobal;
            document.getElementById('year-max').value = maxYearGlobal;
            updateYearFilter();
        }}
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            drawYearMiniChart();
        }});
        </script>
        """
    
    def _build_overview_section(self) -> str:
        """Build overview section with key statistics."""
        b = self.biblio
        
        # Calculate statistics
        n_docs = b.n
        
        # Year range
        year_col = b._get_column("Year", required=False)
        if year_col and year_col in b.df.columns:
            years = b.df[year_col].dropna()
            year_range = f"{int(years.min())} - {int(years.max())}"
            year_span = int(years.max()) - int(years.min()) + 1
        else:
            year_range = "N/A"
            year_span = 1
        
        # Citations
        cite_col = b._get_column("Cited by", required=False)
        if cite_col and cite_col in b.df.columns:
            total_cites = int(b.df[cite_col].sum())
            avg_cites = round(b.df[cite_col].mean(), 2)
        else:
            total_cites = "N/A"
            avg_cites = "N/A"
        
        # Counts
        n_sources = len(b.sources_counts_df) if hasattr(b, "sources_counts_df") and b.sources_counts_df is not None else "N/A"
        n_authors = len(b.authors_counts_df) if hasattr(b, "authors_counts_df") and b.authors_counts_df is not None else "N/A"
        n_keywords = len(b.author_keywords_counts_df) if hasattr(b, "author_keywords_counts_df") and b.author_keywords_counts_df is not None else "N/A"
        
        # Annual growth
        docs_per_year = round(n_docs / year_span, 1) if year_span > 0 else n_docs
        
        return f"""
        <section id="overview" class="section">
            <h2 class="section-title">📊 Overview</h2>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{n_docs:,}</div>
                    <div class="stat-label">Documents</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{year_range}</div>
                    <div class="stat-label">Time Span</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_cites:,}</div>
                    <div class="stat-label">Total Citations</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_cites}</div>
                    <div class="stat-label">Avg. Citations</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{n_sources}</div>
                    <div class="stat-label">Sources</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{n_authors}</div>
                    <div class="stat-label">Authors</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{n_keywords}</div>
                    <div class="stat-label">Keywords</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{docs_per_year}</div>
                    <div class="stat-label">Docs/Year</div>
                </div>
            </div>
        </section>
        """
    
    def _build_production_section(self) -> str:
        """Build production over time section."""
        if not hasattr(self.biblio, "production_df") or self.biblio.production_df is None:
            return ""
        
        df = self.biblio.production_df.copy()
        
        # Create Bokeh chart
        chart_html, chart_js = self._create_line_chart(
            df, 
            x="Year", 
            y="Number of Documents",
            title="Annual Scientific Production",
            chart_id="production_chart"
        )
        
        # Create cumulative chart
        cum_chart_html, cum_chart_js = self._create_area_chart(
            df,
            x="Year",
            y="Cumulative Documents",
            title="Cumulative Production",
            chart_id="cumulative_chart"
        )
        
        return f"""
        <section id="production" class="section">
            <h2 class="section-title">📈 Production Over Time</h2>
            
            <div class="two-column">
                <div>
                    <div class="chart-container" id="production_chart"></div>
                    {chart_js}
                </div>
                <div>
                    <div class="chart-container" id="cumulative_chart"></div>
                    {cum_chart_js}
                </div>
            </div>
            
            {self._create_table(df[["Year", "Number of Documents", "Total Citations", "Cumulative Documents"]].head(10), "production_table") if self.config.include_tables else ""}
        </section>
        """
    
    def _build_sources_section(self) -> str:
        """Build sources analysis section."""
        if not hasattr(self.biblio, "sources_counts_df") or self.biblio.sources_counts_df is None:
            return ""
        
        df = self.biblio.sources_counts_df.head(self.config.top_n).copy()
        
        # Find columns
        name_col = "Source" if "Source" in df.columns else df.columns[0]
        count_col = next((c for c in df.columns if "number" in c.lower() or "count" in c.lower()), df.columns[-1])
        
        chart_html, chart_js = self._create_barh_chart(
            df,
            x=count_col,
            y=name_col,
            title=f"Top {self.config.top_n} Sources",
            chart_id="sources_chart"
        )
        
        return f"""
        <section id="sources" class="section">
            <h2 class="section-title">📚 Sources</h2>
            
            <div class="chart-container" id="sources_chart"></div>
            {chart_js}
            
            {self._create_table(df.head(10), "sources_table") if self.config.include_tables else ""}
        </section>
        """
    
    def _build_authors_section(self) -> str:
        """Build authors analysis section."""
        if not hasattr(self.biblio, "authors_counts_df") or self.biblio.authors_counts_df is None:
            return ""
        
        df = self.biblio.authors_counts_df.head(self.config.top_n).copy()
        
        # Find columns
        name_col = "Author" if "Author" in df.columns else df.columns[0]
        count_col = next((c for c in df.columns if "number" in c.lower() or "count" in c.lower()), df.columns[-1])
        
        chart_html, chart_js = self._create_barh_chart(
            df,
            x=count_col,
            y=name_col,
            title=f"Top {self.config.top_n} Authors",
            chart_id="authors_chart"
        )
        
        return f"""
        <section id="authors" class="section">
            <h2 class="section-title">👥 Authors</h2>
            
            <div class="chart-container" id="authors_chart"></div>
            {chart_js}
            
            {self._create_table(df.head(10), "authors_table") if self.config.include_tables else ""}
        </section>
        """
    
    def _build_keywords_section(self) -> str:
        """Build keywords analysis section."""
        if not hasattr(self.biblio, "author_keywords_counts_df") or self.biblio.author_keywords_counts_df is None:
            return ""
        
        df = self.biblio.author_keywords_counts_df.head(self.config.top_n).copy()
        
        # Find columns
        name_col = "Keyword" if "Keyword" in df.columns else df.columns[0]
        count_col = next((c for c in df.columns if "number" in c.lower() or "count" in c.lower()), df.columns[-1])
        
        chart_html, chart_js = self._create_barh_chart(
            df,
            x=count_col,
            y=name_col,
            title=f"Top {self.config.top_n} Keywords",
            chart_id="keywords_chart"
        )
        
        return f"""
        <section id="keywords" class="section">
            <h2 class="section-title">🏷️ Keywords</h2>
            
            <div class="chart-container" id="keywords_chart"></div>
            {chart_js}
            
            {self._create_table(df.head(10), "keywords_table") if self.config.include_tables else ""}
        </section>
        """
    
    def _build_wordcloud_section(self) -> str:
        """Build word cloud section from keywords, titles, or abstracts."""
        try:
            from collections import Counter
            import json
            import re
            
            b = self.biblio
            sep = b.default_separator
            
            # Collect words from multiple sources
            word_counts = Counter()
            
            # 1. Keywords (primary source)
            kw_col = b._get_column("Author Keywords", required=False)
            if kw_col and kw_col in b.df.columns:
                for keywords in b.df[kw_col].dropna():
                    for kw in str(keywords).split(sep):
                        kw = kw.strip().lower()
                        if kw and len(kw) > 2:
                            word_counts[kw] += 3  # Weight keywords higher
            
            # 2. Index Keywords
            idx_kw_col = b._get_column("Index Keywords", required=False)
            if idx_kw_col and idx_kw_col in b.df.columns:
                for keywords in b.df[idx_kw_col].dropna():
                    for kw in str(keywords).split(sep):
                        kw = kw.strip().lower()
                        if kw and len(kw) > 2:
                            word_counts[kw] += 2
            
            # 3. Title words (optional, lower weight)
            title_col = b._get_column("Title", required=False)
            if title_col and title_col in b.df.columns:
                # Common stopwords
                stopwords = {
                    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
                    "be", "have", "has", "had", "do", "does", "did", "will", "would",
                    "could", "should", "may", "might", "must", "shall", "can", "need",
                    "this", "that", "these", "those", "it", "its", "they", "their",
                    "we", "our", "you", "your", "he", "she", "him", "her", "his",
                    "what", "which", "who", "whom", "when", "where", "why", "how",
                    "all", "each", "every", "both", "few", "more", "most", "other",
                    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
                    "than", "too", "very", "just", "also", "now", "new", "one", "two",
                    "using", "based", "study", "analysis", "research", "paper", "review",
                    "effect", "effects", "approach", "method", "results", "case", "between",
                }
                
                for title in b.df[title_col].dropna():
                    # Extract words
                    words = re.findall(r'\b[a-zA-Z]{3,}\b', str(title).lower())
                    for word in words:
                        if word not in stopwords and len(word) > 3:
                            word_counts[word] += 1
            
            if not word_counts:
                return ""
            
            # Get top words
            top_words = word_counts.most_common(100)
            
            if len(top_words) < 5:
                return ""
            
            # Normalize sizes (font sizes from 12 to 60)
            max_count = top_words[0][1]
            min_count = top_words[-1][1]
            
            def get_size(count):
                if max_count == min_count:
                    return 30
                normalized = (count - min_count) / (max_count - min_count)
                return int(14 + normalized * 46)
            
            # Generate word cloud data for JavaScript
            words_data = [
                {"text": word, "size": get_size(count), "count": count}
                for word, count in top_words
            ]
            
            # Create a simple CSS-based word cloud (more reliable than canvas)
            import random
            random.seed(42)
            
            # Colors from a nice palette
            colors = [
                "#1F4E79", "#2E86AB", "#A23B72", "#F18F01", "#C73E1D",
                "#3A7D44", "#5C4D7D", "#E07A5F", "#3D405B", "#81B29A"
            ]
            
            word_html_items = []
            for i, wd in enumerate(words_data[:80]):  # Limit to 80 words
                color = colors[i % len(colors)]
                size = wd["size"]
                word = wd["text"]
                count = wd["count"]
                # Add slight rotation for visual interest
                rotation = random.choice([0, 0, 0, -5, 5])
                word_html_items.append(
                    f'<span class="cloud-word" style="font-size: {size}px; color: {color}; '
                    f'transform: rotate({rotation}deg);" title="{word}: {count}">{word}</span>'
                )
            
            word_cloud_html = " ".join(word_html_items)
            
            # Add CSS for word cloud
            cloud_css = """
            <style>
            .wordcloud-container {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                align-items: center;
                gap: 8px 15px;
                padding: 20px;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 10px;
                min-height: 300px;
                line-height: 1.2;
            }
            .cloud-word {
                display: inline-block;
                font-weight: 500;
                cursor: default;
                transition: transform 0.2s, opacity 0.2s;
                white-space: nowrap;
            }
            .cloud-word:hover {
                transform: scale(1.15) !important;
                opacity: 0.8;
            }
            </style>
            """
            
            return f"""
            <section id="wordcloud" class="section">
                <h2 class="section-title">☁️ Word Cloud</h2>
                <p style="font-size: 0.85rem; color: #666; margin-bottom: 10px;">
                    Most frequent terms from keywords and titles. Size indicates frequency.
                </p>
                {cloud_css}
                <div class="wordcloud-container">
                    {word_cloud_html}
                </div>
            </section>
            """
        except Exception as e:
            return ""

    def _build_keyword_trends_section(self) -> str:
        """Build keyword trends over time section."""
        try:
            from bokeh.plotting import figure
            from bokeh.embed import components
            from bokeh.models import ColumnDataSource, HoverTool, Legend
            from bokeh.palettes import Category10
            from collections import defaultdict, Counter
            import json
            
            b = self.biblio
            
            # Get columns
            kw_col = b._get_column("Author Keywords", required=False)
            year_col = b._get_column("Year", required=False)
            
            if not kw_col or kw_col not in b.df.columns:
                return ""
            if not year_col or year_col not in b.df.columns:
                return ""
            
            sep = b.default_separator
            
            # Count keywords per year
            kw_by_year = defaultdict(Counter)
            total_by_year = Counter()
            
            for _, row in b.df.iterrows():
                if pd.isna(row[year_col]) or pd.isna(row[kw_col]):
                    continue
                year = int(row[year_col])
                keywords = [k.strip().lower() for k in str(row[kw_col]).split(sep) if k.strip()]
                total_by_year[year] += 1
                for kw in keywords:
                    kw_by_year[year][kw] += 1
            
            if not kw_by_year:
                return ""
            
            # Get overall top keywords
            overall_counts = Counter()
            for year_counts in kw_by_year.values():
                overall_counts.update(year_counts)
            
            top_keywords = [kw for kw, _ in overall_counts.most_common(10)]
            years = sorted(kw_by_year.keys())
            
            if len(years) < 2:
                return ""
            
            # Calculate growth rates
            growth_data = []
            for kw in top_keywords:
                counts = [kw_by_year[y].get(kw, 0) for y in years]
                # Normalize by total docs per year
                normalized = [c / total_by_year[y] * 100 if total_by_year[y] > 0 else 0 
                             for c, y in zip(counts, years)]
                
                # Calculate trend (linear regression slope)
                if len(normalized) >= 2:
                    x = np.arange(len(normalized))
                    slope = np.polyfit(x, normalized, 1)[0]
                else:
                    slope = 0
                
                # Recent vs early period
                mid = len(years) // 2
                early_avg = np.mean(normalized[:mid]) if mid > 0 else 0
                recent_avg = np.mean(normalized[mid:]) if mid < len(normalized) else 0
                
                growth_data.append({
                    "keyword": kw,
                    "counts": counts,
                    "normalized": normalized,
                    "slope": slope,
                    "early_avg": early_avg,
                    "recent_avg": recent_avg,
                    "total": sum(counts),
                    "trend": "📈 Rising" if slope > 0.5 else ("📉 Declining" if slope < -0.5 else "➡️ Stable")
                })
            
            # Sort by slope to show rising/declining
            growth_data.sort(key=lambda x: x["slope"], reverse=True)
            
            # Create Bokeh multi-line chart
            p = figure(
                title="Keyword Trends Over Time (% of papers)",
                width=self.config.width - 80,
                height=self.config.plot_height,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above",
                x_axis_label="Year",
                y_axis_label="% of Papers",
            )
            
            colors = Category10[10]
            legend_items = []
            
            for i, gd in enumerate(growth_data[:8]):  # Top 8 for readability
                source = ColumnDataSource(data={
                    "year": years,
                    "value": gd["normalized"],
                    "keyword": [gd["keyword"]] * len(years),
                })
                
                line = p.line(
                    "year", "value", source=source,
                    line_width=2, color=colors[i % len(colors)],
                    alpha=0.8,
                )
                
                circles = p.scatter(
                    "year", "value", source=source,
                    size=6, color=colors[i % len(colors)],
                    alpha=0.8,
                )
                
                legend_items.append((gd["keyword"][:20], [line, circles]))
            
            # Add legend
            legend = Legend(items=legend_items, location="top_left")
            legend.click_policy = "hide"
            legend.label_text_font_size = "9pt"
            p.add_layout(legend, "right")
            
            # Hover
            hover = HoverTool(
                tooltips=[
                    ("Keyword", "@keyword"),
                    ("Year", "@year"),
                    ("% of Papers", "@value{0.2f}%"),
                ]
            )
            p.add_tools(hover)
            
            p.title.text_font_size = "13pt"
            p.xaxis.ticker = years[::max(1, len(years)//10)]
            
            script, div = components(p)
            
            # Create trends table
            trends_df = pd.DataFrame([
                {
                    "Keyword": gd["keyword"].title(),
                    "Total": gd["total"],
                    "Trend": gd["trend"],
                    "Growth": f"{gd['slope']:+.2f}",
                }
                for gd in growth_data
            ])
            
            return f"""
            <section id="keyword_trends" class="section">
                <h2 class="section-title">📈 Keyword Trends</h2>
                <p style="font-size: 0.85rem; color: #666; margin-bottom: 10px;">
                    Temporal evolution of top keywords. Click legend to show/hide lines.
                </p>
                
                <div class="chart-container">
                    {div}
                    {script}
                </div>
                
                <h3 style="margin-top: 20px; margin-bottom: 10px; font-size: 1.1rem;">Trend Analysis</h3>
                {self._create_table(trends_df, "trends_table")}
            </section>
            """
        except Exception as e:
            return ""
    
    def _build_citation_classics_section(self) -> str:
        """Build citation classics and sleeping beauties section."""
        try:
            b = self.biblio
            
            cite_col = b._get_column("Cited by", required=False)
            title_col = b._get_column("Title", required=False)
            year_col = b._get_column("Year", required=False)
            auth_col = b._get_column("Authors", required=False)
            source_col = b._get_column("Source title", required=False)
            
            if not cite_col or cite_col not in b.df.columns:
                return ""
            if not title_col or title_col not in b.df.columns:
                return ""
            
            # Top cited papers
            top_n = min(20, len(b.df))
            top_cited = b.df.nlargest(top_n, cite_col).copy()
            
            # Build display table
            display_cols = []
            col_names = []
            
            if title_col in top_cited.columns:
                # Truncate long titles
                top_cited["Title_Short"] = top_cited[title_col].apply(
                    lambda x: str(x)[:80] + "..." if len(str(x)) > 80 else str(x)
                )
                display_cols.append("Title_Short")
                col_names.append("Title")
            
            if auth_col and auth_col in top_cited.columns:
                # First author only
                top_cited["First_Author"] = top_cited[auth_col].apply(
                    lambda x: str(x).split(";")[0].strip() if pd.notna(x) else ""
                )
                display_cols.append("First_Author")
                col_names.append("First Author")
            
            if year_col and year_col in top_cited.columns:
                display_cols.append(year_col)
                col_names.append("Year")
            
            if source_col and source_col in top_cited.columns:
                top_cited["Source_Short"] = top_cited[source_col].apply(
                    lambda x: str(x)[:30] + "..." if len(str(x)) > 30 else str(x)
                )
                display_cols.append("Source_Short")
                col_names.append("Source")
            
            display_cols.append(cite_col)
            col_names.append("Citations")
            
            top_cited_display = top_cited[display_cols].copy()
            top_cited_display.columns = col_names
            
            # Citation classics (papers with >100 citations or top 1%)
            threshold = max(100, b.df[cite_col].quantile(0.99))
            classics_count = (b.df[cite_col] >= threshold).sum()
            
            # Sleeping beauties detection
            sleeping_beauties_html = ""
            if year_col and year_col in b.df.columns:
                current_year = b.df[year_col].max()
                
                # Simple sleeping beauty detection:
                # Papers that are old (>5 years) but got most citations recently
                sb_candidates = []
                
                for idx, row in b.df.iterrows():
                    if pd.isna(row[year_col]) or pd.isna(row[cite_col]):
                        continue
                    
                    pub_year = int(row[year_col])
                    citations = int(row[cite_col])
                    age = current_year - pub_year
                    
                    if age >= 5 and citations >= 20:
                        # Citations per year (proxy for late recognition)
                        cpy = citations / age if age > 0 else 0
                        
                        # Beauty coefficient approximation
                        # Real calculation would need yearly citation data
                        # Here we use age-adjusted citation rate as proxy
                        beauty_score = cpy * (age / 5)  # Higher if old but highly cited
                        
                        sb_candidates.append({
                            "title": str(row[title_col])[:60] + "..." if len(str(row[title_col])) > 60 else str(row[title_col]),
                            "year": pub_year,
                            "citations": citations,
                            "age": age,
                            "cpy": round(cpy, 1),
                            "score": round(beauty_score, 1),
                        })
                
                # Sort by beauty score
                sb_candidates.sort(key=lambda x: x["score"], reverse=True)
                top_sb = sb_candidates[:5]
                
                if top_sb:
                    sb_df = pd.DataFrame(top_sb)
                    sb_df.columns = ["Title", "Year", "Citations", "Age", "Cit/Year", "Score"]
                    sleeping_beauties_html = f"""
                    <h3 style="margin-top: 20px; margin-bottom: 10px; font-size: 1.1rem;">
                        💤 Potential Sleeping Beauties
                    </h3>
                    <p style="font-size: 0.85rem; color: #666; margin-bottom: 10px;">
                        Older papers with sustained citation impact (high citations/year ratio)
                    </p>
                    {self._create_table(sb_df, "sleeping_beauties_table")}
                    """
            
            # Citation aging curve
            aging_html = ""
            if year_col and year_col in b.df.columns:
                from bokeh.plotting import figure
                from bokeh.embed import components
                from bokeh.models import ColumnDataSource, HoverTool
                
                current_year = int(b.df[year_col].max())
                
                # Calculate average citations by paper age
                age_cites = defaultdict(list)
                for _, row in b.df.iterrows():
                    if pd.isna(row[year_col]) or pd.isna(row[cite_col]):
                        continue
                    age = current_year - int(row[year_col])
                    if 0 <= age <= 20:
                        age_cites[age].append(row[cite_col])
                
                if age_cites:
                    ages = sorted(age_cites.keys())
                    avg_cites = [np.mean(age_cites[a]) for a in ages]
                    
                    source = ColumnDataSource(data={
                        "age": ages,
                        "citations": avg_cites,
                    })
                    
                    p = figure(
                        title="Citation Aging Curve",
                        width=self.config.width // 2 - 50,
                        height=self.config.plot_height - 50,
                        tools="pan,wheel_zoom,reset,save",
                        toolbar_location="above",
                        x_axis_label="Paper Age (years)",
                        y_axis_label="Average Citations",
                    )
                    
                    p.line("age", "citations", source=source, line_width=3, color=self.config.primary_color)
                    p.scatter("age", "citations", source=source, size=8, color=self.config.primary_color)
                    
                    hover = HoverTool(tooltips=[("Age", "@age years"), ("Avg Citations", "@citations{0.1f}")])
                    p.add_tools(hover)
                    p.title.text_font_size = "12pt"
                    
                    script, div = components(p)
                    aging_html = f"""
                    <div style="margin-top: 20px;">
                        <h3 style="margin-bottom: 10px; font-size: 1.1rem;">📊 Citation Aging Curve</h3>
                        {div}
                        {script}
                    </div>
                    """
            
            return f"""
            <section id="citation_classics" class="section">
                <h2 class="section-title">🏆 Citation Classics</h2>
                
                <div class="stats-grid" style="margin-bottom: 15px;">
                    <div class="stat-card">
                        <div class="stat-value">{int(top_cited[cite_col].iloc[0]):,}</div>
                        <div class="stat-label">Most Cited</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{classics_count}</div>
                        <div class="stat-label">Classics (≥{int(threshold)})</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{int(top_cited[cite_col].sum()):,}</div>
                        <div class="stat-label">Top {top_n} Total</div>
                    </div>
                </div>
                
                <h3 style="margin-bottom: 10px; font-size: 1.1rem;">Most Cited Papers</h3>
                {self._create_table(top_cited_display, "top_cited_table")}
                
                {sleeping_beauties_html}
                {aging_html}
            </section>
            """
        except Exception as e:
            return ""
    
    def _build_reference_analysis_section(self) -> str:
        """Build reference analysis section (co-citation, bibliographic coupling)."""
        try:
            from bokeh.plotting import figure
            from bokeh.embed import components
            from bokeh.models import ColumnDataSource, HoverTool, LabelSet
            import networkx as nx
            from collections import defaultdict, Counter
            
            b = self.biblio
            
            # Check for references column
            ref_col = None
            for col_name in ["References", "Cited References", "Cited references", "Reference"]:
                if col_name in b.df.columns:
                    ref_col = col_name
                    break
            
            cite_col = b._get_column("Cited by", required=False)
            title_col = b._get_column("Title", required=False)
            
            # Bibliographic coupling (papers sharing references)
            coupling_html = ""
            cocitation_html = ""
            ref_stats_html = ""
            
            if ref_col and ref_col in b.df.columns:
                sep = b.default_separator
                
                # Parse references
                doc_refs = {}
                ref_counts = Counter()
                
                for idx, row in b.df.iterrows():
                    if pd.notna(row[ref_col]):
                        refs = [r.strip() for r in str(row[ref_col]).split(sep) if r.strip()]
                        doc_refs[idx] = set(refs)
                        ref_counts.update(refs)
                
                total_refs = sum(len(refs) for refs in doc_refs.values())
                unique_refs = len(ref_counts)
                avg_refs = total_refs / len(doc_refs) if doc_refs else 0
                
                # Most cited references
                top_refs = ref_counts.most_common(10)
                if top_refs:
                    refs_df = pd.DataFrame(top_refs, columns=["Reference", "Citations"])
                    # Truncate long references
                    refs_df["Reference"] = refs_df["Reference"].apply(
                        lambda x: x[:70] + "..." if len(x) > 70 else x
                    )
                    
                    ref_stats_html = f"""
                    <div class="stats-grid" style="margin-bottom: 15px;">
                        <div class="stat-card">
                            <div class="stat-value">{total_refs:,}</div>
                            <div class="stat-label">Total References</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{unique_refs:,}</div>
                            <div class="stat-label">Unique References</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{avg_refs:.1f}</div>
                            <div class="stat-label">Avg per Paper</div>
                        </div>
                    </div>
                    
                    <h3 style="margin-top: 15px; margin-bottom: 10px; font-size: 1.1rem;">Most Cited References</h3>
                    {self._create_table(refs_df, "top_refs_table")}
                    """
                
                # Co-citation analysis (references cited together)
                if len(doc_refs) >= 10:
                    cocitation = defaultdict(int)
                    top_ref_set = set(r for r, _ in ref_counts.most_common(50))
                    
                    for refs in doc_refs.values():
                        common_refs = refs & top_ref_set
                        ref_list = list(common_refs)
                        for i, r1 in enumerate(ref_list):
                            for r2 in ref_list[i+1:]:
                                pair = tuple(sorted([r1, r2]))
                                cocitation[pair] += 1
                    
                    # Build co-citation network
                    if cocitation:
                        G = nx.Graph()
                        
                        # Add top co-cited pairs
                        top_pairs = sorted(cocitation.items(), key=lambda x: x[1], reverse=True)[:30]
                        
                        for (r1, r2), weight in top_pairs:
                            if weight >= 2:
                                # Shorten reference names
                                r1_short = r1[:25] + "..." if len(r1) > 25 else r1
                                r2_short = r2[:25] + "..." if len(r2) > 25 else r2
                                G.add_edge(r1_short, r2_short, weight=weight)
                        
                        if len(G.nodes()) >= 3:
                            pos = nx.spring_layout(G, k=2, iterations=30, seed=42)
                            
                            p = figure(
                                title="Co-citation Network (Top References)",
                                width=self.config.width - 80,
                                height=self.config.plot_height,
                                tools="pan,wheel_zoom,box_zoom,reset,save",
                                toolbar_location="above",
                                x_range=(-2.5, 2.5),
                                y_range=(-2.5, 2.5),
                            )
                            
                            p.xaxis.visible = False
                            p.yaxis.visible = False
                            p.xgrid.visible = False
                            p.ygrid.visible = False
                            
                            # Draw edges
                            for edge in G.edges():
                                x0, y0 = pos[edge[0]]
                                x1, y1 = pos[edge[1]]
                                p.line([x0, x1], [y0, y1], line_color="#CCCCCC", line_alpha=0.5, line_width=1)
                            
                            # Nodes
                            node_x = [pos[n][0] for n in G.nodes()]
                            node_y = [pos[n][1] for n in G.nodes()]
                            node_names = list(G.nodes())
                            node_degree = [G.degree(n) for n in G.nodes()]
                            max_deg = max(node_degree) if node_degree else 1
                            node_sizes = [max(8, (d / max_deg) * 25 + 8) for d in node_degree]
                            
                            source = ColumnDataSource(data={
                                "x": node_x,
                                "y": node_y,
                                "name": node_names,
                                "degree": node_degree,
                                "size": node_sizes,
                            })
                            
                            nodes = p.scatter(
                                "x", "y", size="size", source=source,
                                fill_color=self.config.accent_color, fill_alpha=0.7,
                                line_color="white", line_width=1,
                            )
                            
                            hover = HoverTool(
                                tooltips=[("Reference", "@name"), ("Co-citations", "@degree")],
                                renderers=[nodes]
                            )
                            p.add_tools(hover)
                            p.title.text_font_size = "12pt"
                            
                            script, div = components(p)
                            
                            cocitation_html = f"""
                            <h3 style="margin-top: 20px; margin-bottom: 10px; font-size: 1.1rem;">
                                🔗 Co-citation Network
                            </h3>
                            <p style="font-size: 0.85rem; color: #666; margin-bottom: 10px;">
                                References frequently cited together. Node size = number of co-citations.
                            </p>
                            {div}
                            {script}
                            """
                
                # Bibliographic coupling strength
                if len(doc_refs) >= 5 and title_col and title_col in b.df.columns:
                    coupling_pairs = []
                    doc_indices = list(doc_refs.keys())[:100]  # Limit for performance
                    
                    for i, idx1 in enumerate(doc_indices):
                        for idx2 in doc_indices[i+1:]:
                            shared = len(doc_refs[idx1] & doc_refs[idx2])
                            if shared >= 3:
                                coupling_pairs.append({
                                    "doc1": str(b.df.loc[idx1, title_col])[:40],
                                    "doc2": str(b.df.loc[idx2, title_col])[:40],
                                    "shared": shared,
                                })
                    
                    if coupling_pairs:
                        coupling_pairs.sort(key=lambda x: x["shared"], reverse=True)
                        top_coupling = coupling_pairs[:10]
                        
                        coupling_df = pd.DataFrame(top_coupling)
                        coupling_df.columns = ["Document 1", "Document 2", "Shared Refs"]
                        
                        coupling_html = f"""
                        <h3 style="margin-top: 20px; margin-bottom: 10px; font-size: 1.1rem;">
                            📚 Bibliographic Coupling
                        </h3>
                        <p style="font-size: 0.85rem; color: #666; margin-bottom: 10px;">
                            Document pairs sharing the most references (intellectually similar).
                        </p>
                        {self._create_table(coupling_df, "coupling_table")}
                        """
            
            # If no reference data, show message
            if not ref_col or ref_col not in b.df.columns:
                return f"""
                <section id="reference_analysis" class="section">
                    <h2 class="section-title">📖 Reference Analysis</h2>
                    <p style="color: #666;">
                        Reference data not available in dataset. Reference analysis requires a "References" column.
                    </p>
                </section>
                """
            
            return f"""
            <section id="reference_analysis" class="section">
                <h2 class="section-title">📖 Reference Analysis</h2>
                {ref_stats_html}
                {cocitation_html}
                {coupling_html}
            </section>
            """
        except Exception as e:
            return ""

    def _build_scatter_section(self) -> str:
        """Build interactive scatter plot section for sources/authors/keywords."""
        # Try to get stats dataframes
        scatter_data = []
        
        # Sources stats
        if hasattr(self.biblio, "sources_stats_df") and self.biblio.sources_stats_df is not None:
            scatter_data.append(("Sources", self.biblio.sources_stats_df.copy()))
        elif hasattr(self.biblio, "sources_counts_df") and self.biblio.sources_counts_df is not None:
            # Create basic stats from counts
            df = self.biblio.sources_counts_df.head(50).copy()
            scatter_data.append(("Sources", df))
        
        # Authors stats
        if hasattr(self.biblio, "authors_stats_df") and self.biblio.authors_stats_df is not None:
            scatter_data.append(("Authors", self.biblio.authors_stats_df.copy()))
        elif hasattr(self.biblio, "authors_counts_df") and self.biblio.authors_counts_df is not None:
            df = self.biblio.authors_counts_df.head(50).copy()
            scatter_data.append(("Authors", df))
        
        # Keywords stats
        if hasattr(self.biblio, "author_keywords_stats_df") and self.biblio.author_keywords_stats_df is not None:
            scatter_data.append(("Keywords", self.biblio.author_keywords_stats_df.copy()))
        elif hasattr(self.biblio, "author_keywords_counts_df") and self.biblio.author_keywords_counts_df is not None:
            df = self.biblio.author_keywords_counts_df.head(50).copy()
            scatter_data.append(("Keywords", df))
        
        if not scatter_data:
            return ""
        
        # Build scatter plots for each data type
        scatter_plots = []
        
        for name, df in scatter_data:
            plot_html = self._create_interactive_scatter(df, name)
            if plot_html:
                scatter_plots.append(plot_html)
        
        if not scatter_plots:
            return ""
        
        return f"""
        <section id="scatter" class="section">
            <h2 class="section-title">📊 Scatter Analysis</h2>
            <p style="margin-bottom: 20px; color: #666;">
                Interactive scatter plots with adjustable axes. Use the dropdowns to change X/Y axes. 
                Toggle log scale for better visualization of skewed distributions.
                Bubble size represents h-index (if available), color represents average publication year.
            </p>
            
            {''.join(scatter_plots)}
        </section>
        """
    
    def _create_interactive_scatter(self, df: pd.DataFrame, data_type: str) -> str:
        """Create an interactive scatter plot with axis selection."""
        from bokeh.plotting import figure
        from bokeh.embed import components
        from bokeh.models import (
            ColumnDataSource, HoverTool, Select, CustomJS,
            ColorBar, LinearColorMapper
        )
        from bokeh.layouts import column, row
        from bokeh.palettes import Viridis256
        
        # Identify available numeric columns
        numeric_cols = []
        col_mapping = {}
        
        # Standard column mappings
        possible_mappings = {
            "Number of documents": ["Number of documents", "Number of Documents", "Documents", "Count", "Frequency"],
            "Total citations": ["Total citations", "Total Citations", "Citations", "Cited by"],
            "H-index": ["H-index", "h-index", "h_index", "H index"],
            "Average year": ["Average year", "Average Year", "Avg Year", "Mean Year"],
            "G-index": ["G-index", "g-index"],
            "First year": ["First year", "First Year", "Start Year"],
            "Last year": ["Last year", "Last Year", "End Year"],
        }
        
        for standard_name, variants in possible_mappings.items():
            for variant in variants:
                if variant in df.columns:
                    col_mapping[standard_name] = variant
                    numeric_cols.append(standard_name)
                    break
        
        # Need at least 2 numeric columns
        if len(numeric_cols) < 2:
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    if col not in col_mapping.values():
                        numeric_cols.append(col)
                        col_mapping[col] = col
        
        if len(numeric_cols) < 2:
            return ""
        
        # Find name column
        name_col = None
        for col in ["Source", "Author", "Keyword", "Authors", "Name"]:
            if col in df.columns:
                name_col = col
                break
        if name_col is None:
            name_col = df.columns[0]
        
        # Prepare data - limit to top items
        df = df.head(self.config.top_n * 3).copy()
        
        # Set defaults
        x_col = "Number of documents" if "Number of documents" in numeric_cols else numeric_cols[0]
        y_col = "Total citations" if "Total citations" in numeric_cols else (numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0])
        size_col = "H-index" if "H-index" in numeric_cols else None
        color_col = "Average year" if "Average year" in numeric_cols else None
        
        # Get actual column names
        x_actual = col_mapping.get(x_col, x_col)
        y_actual = col_mapping.get(y_col, y_col)
        
        # Create source data
        x_vals = df[x_actual].fillna(0).tolist()
        y_vals = df[y_actual].fillna(0).tolist()
        names = df[name_col].astype(str).tolist()
        
        # Calculate sizes based on h-index
        if size_col and size_col in col_mapping:
            size_actual = col_mapping[size_col]
            sizes = df[size_actual].fillna(1).tolist()
            max_size = max(sizes) if max(sizes) > 0 else 1
            sizes = [max(8, min(40, (s / max_size) * 35 + 8)) for s in sizes]
        else:
            sizes = [15] * len(df)
        
        # Calculate colors based on average year
        if color_col and color_col in col_mapping:
            color_actual = col_mapping[color_col]
            colors = df[color_actual].fillna(df[color_actual].mean()).tolist()
        else:
            colors = [2020] * len(df)
        
        source = ColumnDataSource(data={
            "x": [max(0.5, v) for v in x_vals],  # Ensure positive for log scale
            "y": [max(0.5, v) for v in y_vals],
            "name": names,
            "size": sizes,
            "color_value": colors,
            "x_display": x_vals,
            "y_display": y_vals,
        })
        
        # Add size and color values if available
        if size_col and size_col in col_mapping:
            source.data["h_index"] = df[col_mapping[size_col]].fillna(0).tolist()
        
        # Color mapper
        color_low = min(colors)
        color_high = max(colors)
        if color_low == color_high:
            color_high = color_low + 1
        
        color_mapper = LinearColorMapper(
            palette=Viridis256,
            low=color_low,
            high=color_high
        )
        
        # Create figure with log axes
        p = figure(
            title=f"{data_type}: {x_col} vs {y_col}",
            width=self.config.width - 80,
            height=self.config.plot_height,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            toolbar_location="above",
            x_axis_type="log",
            y_axis_type="log",
        )
        
        # Scatter plot
        scatter = p.scatter(
            x="x",
            y="y",
            size="size",
            source=source,
            fill_color={"field": "color_value", "transform": color_mapper},
            fill_alpha=0.7,
            line_color="white",
            line_width=1,
        )
        
        # Hover tool
        tooltips = [
            (data_type[:-1] if data_type.endswith("s") else data_type, "@name"),
            (x_col, "@x_display{0,0}"),
            (y_col, "@y_display{0,0}"),
        ]
        if size_col and "h_index" in source.data:
            tooltips.append(("H-index", "@h_index{0.0}"))
        if color_col:
            tooltips.append((color_col, "@color_value{0.0}"))
        
        hover = HoverTool(tooltips=tooltips)
        p.add_tools(hover)
        
        # Color bar
        color_bar = ColorBar(
            color_mapper=color_mapper,
            title=color_col or "Year",
            location=(0, 0),
            title_text_font_size="10pt",
        )
        p.add_layout(color_bar, "right")
        
        # Styling
        p.xaxis.axis_label = x_col
        p.yaxis.axis_label = y_col
        p.title.text_font_size = "13pt"
        p.xaxis.axis_label_text_font_size = "11pt"
        p.yaxis.axis_label_text_font_size = "11pt"
        
        script, div = components(p)
        
        return f"""
        <div class="scatter-plot-container" style="margin-bottom: 25px;">
            <h3 style="margin-bottom: 10px; font-size: 1.1rem;">{data_type}</h3>
            <p style="font-size: 0.85rem; color: #666; margin-bottom: 10px;">
                Size = H-index | Color = Average Year | Axes = Log scale
            </p>
            {div}
            {script}
        </div>
        """

    def _build_sankey_section(self) -> str:
        """Build three-field Sankey diagram section."""
        try:
            from collections import defaultdict, Counter
            import json
            
            b = self.biblio
            sep = b.default_separator
            
            # Get columns
            kw_col = b._get_column("Author Keywords", required=False)
            source_col = b._get_column("Source title", required=False)
            auth_col = b._get_column("Authors", required=False)
            year_col = b._get_column("Year", required=False)
            
            # Need at least 2 fields
            available_fields = []
            if kw_col and kw_col in b.df.columns:
                available_fields.append(("Keywords", kw_col))
            if source_col and source_col in b.df.columns:
                available_fields.append(("Sources", source_col))
            if auth_col and auth_col in b.df.columns:
                available_fields.append(("Authors", auth_col))
            
            if len(available_fields) < 2:
                return ""
            
            # Use first 3 available fields
            fields = available_fields[:3]
            field_names = [f[0] for f in fields]
            field_cols = [f[1] for f in fields]
            
            # Count items per field and get top N
            top_n = min(10, self.config.top_n)
            field_tops = []
            
            for field_name, col in fields:
                counter = Counter()
                for val in b.df[col].dropna():
                    for item in str(val).split(sep):
                        item = item.strip()
                        if item:
                            counter[item] += 1
                top_items = [item for item, _ in counter.most_common(top_n)]
                field_tops.append(set(top_items))
            
            # Build nodes list
            nodes = []
            node_index = {}
            node_years = {}  # For color
            
            for field_idx, (field_name, col) in enumerate(fields):
                for item in field_tops[field_idx]:
                    node_key = (field_idx, item)
                    node_index[node_key] = len(nodes)
                    nodes.append(item)
                    
                    # Calculate average year for this item
                    if year_col and year_col in b.df.columns:
                        years = []
                        for idx, row in b.df.iterrows():
                            if pd.notna(row[col]):
                                items = [i.strip() for i in str(row[col]).split(sep)]
                                if item in items and pd.notna(row[year_col]):
                                    years.append(row[year_col])
                        node_years[node_key] = np.mean(years) if years else 2020
                    else:
                        node_years[node_key] = 2020
            
            # Build links between consecutive fields
            links = defaultdict(int)
            
            for idx, row in b.df.iterrows():
                # Get items from each field for this document
                doc_items = []
                for field_idx, (field_name, col) in enumerate(fields):
                    if pd.notna(row[col]):
                        items = [i.strip() for i in str(row[col]).split(sep)]
                        # Filter to top items only
                        items = [i for i in items if i in field_tops[field_idx]]
                        doc_items.append([(field_idx, i) for i in items])
                    else:
                        doc_items.append([])
                
                # Create links between consecutive fields
                for i in range(len(fields) - 1):
                    for src in doc_items[i]:
                        for tgt in doc_items[i + 1]:
                            if src in node_index and tgt in node_index:
                                links[(node_index[src], node_index[tgt])] += 1
            
            if not links:
                return ""
            
            # Prepare data for Plotly
            sources = [k[0] for k in links.keys()]
            targets = [k[1] for k in links.keys()]
            values = list(links.values())
            
            # Calculate node positions (x based on field)
            num_fields = len(fields)
            node_x = []
            node_y = []
            
            for field_idx, (field_name, col) in enumerate(fields):
                field_nodes = [k for k in node_index.keys() if k[0] == field_idx]
                n_nodes = len(field_nodes)
                for i, node_key in enumerate(sorted(field_nodes, key=lambda k: node_index[k])):
                    node_x.append(field_idx / max(1, num_fields - 1))
                    node_y.append((i + 0.5) / max(1, n_nodes))
            
            # Node colors based on year
            years = [node_years.get((field_idx, nodes[node_index[(field_idx, item)]]), 2020) 
                     for field_idx, (_, _) in enumerate(fields) 
                     for item in field_tops[field_idx] if (field_idx, item) in node_index]
            
            # Normalize years for color
            if len(node_years) > 0:
                all_years = list(node_years.values())
                year_min, year_max = min(all_years), max(all_years)
                if year_max == year_min:
                    year_max = year_min + 1
                
                # Create color array using Viridis
                node_colors = []
                for node_key in node_index.keys():
                    year = node_years.get(node_key, 2020)
                    norm = (year - year_min) / (year_max - year_min)
                    # Viridis-like colors
                    r = int(68 + norm * (253 - 68))
                    g = int(1 + norm * (231 - 1))
                    b = int(84 + norm * (37 - 84))
                    node_colors.append(f"rgb({r},{g},{b})")
            else:
                node_colors = ["#1F4E79"] * len(nodes)
            
            # Create unique div id
            import uuid
            div_id = f"sankey_{uuid.uuid4().hex[:8]}"
            
            # Create Plotly JSON
            sankey_data = {
                "type": "sankey",
                "arrangement": "fixed",
                "node": {
                    "pad": 15,
                    "thickness": 20,
                    "line": {"color": "white", "width": 0.5},
                    "label": nodes,
                    "color": node_colors,
                    "x": node_x,
                    "y": node_y,
                },
                "link": {
                    "source": sources,
                    "target": targets,
                    "value": values,
                }
            }
            
            layout = {
                "title": {"text": ""},
                "font": {"size": 11},
                "height": self.config.plot_height + 100,
                "margin": {"l": 10, "r": 10, "t": 30, "b": 60},
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
                "annotations": [
                    {
                        "x": i / max(1, num_fields - 1),
                        "y": -0.08,
                        "xref": "paper",
                        "yref": "paper",
                        "text": field_names[i],
                        "showarrow": False,
                        "font": {"size": 13, "color": self.config.primary_color},
                    }
                    for i in range(num_fields)
                ]
            }
            
            # Create JavaScript for the plot
            plot_js = f"""
            <script>
                var data = [{json.dumps(sankey_data)}];
                var layout = {json.dumps(layout)};
                Plotly.newPlot('{div_id}', data, layout, {{responsive: true}});
            </script>
            """
            
            return f"""
            <section id="sankey" class="section">
                <h2 class="section-title">🔀 Three-Field Plot</h2>
                <p style="font-size: 0.85rem; color: #666; margin-bottom: 10px;">
                    Flow between {' → '.join(field_names)} | Node color = average publication year
                </p>
                <div id="{div_id}" style="width: 100%; height: {self.config.plot_height + 100}px;"></div>
                {plot_js}
            </section>
            """
        except Exception as e:
            return ""

    def _build_keyword_network_section(self) -> str:
        """Build keyword co-occurrence network section."""
        try:
            # Try to build co-occurrence network
            from bokeh.plotting import figure, from_networkx
            from bokeh.embed import components
            from bokeh.models import (
                ColumnDataSource, HoverTool, Circle, MultiLine,
                NodesAndLinkedEdges, EdgesAndLinkedNodes
            )
            from bokeh.palettes import Spectral8
            import networkx as nx
            
            # Get keyword column
            kw_col = self.biblio._get_column("Author Keywords", required=False)
            if not kw_col or kw_col not in self.biblio.df.columns:
                kw_col = self.biblio._get_column("Index Keywords", required=False)
            
            if not kw_col or kw_col not in self.biblio.df.columns:
                return ""
            
            # Build co-occurrence matrix
            from collections import defaultdict, Counter
            
            cooccurrence = defaultdict(int)
            keyword_freq = Counter()
            
            sep = self.biblio.default_separator
            
            for keywords in self.biblio.df[kw_col].dropna():
                kw_list = [k.strip().lower() for k in str(keywords).split(sep) if k.strip()]
                
                for kw in kw_list:
                    keyword_freq[kw] += 1
                
                # Count co-occurrences
                for i, kw1 in enumerate(kw_list):
                    for kw2 in kw_list[i+1:]:
                        if kw1 != kw2:
                            pair = tuple(sorted([kw1, kw2]))
                            cooccurrence[pair] += 1
            
            # Get top keywords
            top_n = min(30, len(keyword_freq))
            top_keywords = set(k for k, _ in keyword_freq.most_common(top_n))
            
            # Build network
            G = nx.Graph()
            
            # Add nodes
            for kw in top_keywords:
                G.add_node(kw, frequency=keyword_freq[kw])
            
            # Add edges (only between top keywords)
            min_cooccur = 2
            for (kw1, kw2), count in cooccurrence.items():
                if kw1 in top_keywords and kw2 in top_keywords and count >= min_cooccur:
                    G.add_edge(kw1, kw2, weight=count)
            
            if len(G.nodes()) < 3 or len(G.edges()) < 2:
                return ""
            
            # Calculate layout
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            
            # Create Bokeh plot
            p = figure(
                title="Keyword Co-occurrence Network",
                width=self.config.width - 80,
                height=self.config.plot_height + 150,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above",
                x_range=(-2.5, 2.5),
                y_range=(-2.5, 2.5),
            )
            
            p.xaxis.visible = False
            p.yaxis.visible = False
            p.xgrid.visible = False
            p.ygrid.visible = False
            
            # Prepare node data
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            node_names = list(G.nodes())
            node_freq = [G.nodes[node]['frequency'] for node in G.nodes()]
            max_freq = max(node_freq) if node_freq else 1
            node_sizes = [max(10, min(50, (f / max_freq) * 40 + 10)) for f in node_freq]
            
            # Degree for color
            node_degree = [G.degree(node) for node in G.nodes()]
            max_degree = max(node_degree) if node_degree else 1
            
            node_source = ColumnDataSource(data={
                "x": node_x,
                "y": node_y,
                "name": node_names,
                "frequency": node_freq,
                "degree": node_degree,
                "size": node_sizes,
            })
            
            # Draw edges
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.append([x0, x1])
                edge_y.append([y0, y1])
            
            if edge_x:
                p.multi_line(edge_x, edge_y, line_color="#CCCCCC", line_alpha=0.6, line_width=1)
            
            # Draw nodes
            nodes = p.scatter(
                "x", "y", 
                size="size",
                source=node_source,
                fill_color=self.config.primary_color,
                fill_alpha=0.7,
                line_color="white",
                line_width=1,
            )
            
            # Add labels
            from bokeh.models import LabelSet
            labels = LabelSet(
                x="x", y="y", text="name",
                source=node_source,
                x_offset=5, y_offset=5,
                text_font_size="9pt",
                text_color="#333333",
            )
            p.add_layout(labels)
            
            # Hover
            hover = HoverTool(
                tooltips=[
                    ("Keyword", "@name"),
                    ("Frequency", "@frequency"),
                    ("Connections", "@degree"),
                ],
                renderers=[nodes]
            )
            p.add_tools(hover)
            
            p.title.text_font_size = "13pt"
            
            script, div = components(p)
            
            # Network stats
            stats_html = f"""
            <div class="stats-grid" style="margin-bottom: 15px;">
                <div class="stat-card">
                    <div class="stat-value">{len(G.nodes())}</div>
                    <div class="stat-label">Keywords</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(G.edges())}</div>
                    <div class="stat-label">Connections</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{nx.density(G):.3f}</div>
                    <div class="stat-label">Density</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{nx.number_connected_components(G)}</div>
                    <div class="stat-label">Components</div>
                </div>
            </div>
            """
            
            return f"""
            <section id="keyword_network" class="section">
                <h2 class="section-title">🔗 Keyword Co-occurrence Network</h2>
                <p style="font-size: 0.85rem; color: #666; margin-bottom: 10px;">
                    Node size = keyword frequency | Edges = co-occurrence in same document
                </p>
                {stats_html}
                <div class="chart-container">
                    {div}
                    {script}
                </div>
            </section>
            """
        except Exception as e:
            return ""
    
    def _build_collaboration_network_section(self) -> str:
        """Build author collaboration network section."""
        try:
            from bokeh.plotting import figure
            from bokeh.embed import components
            from bokeh.models import ColumnDataSource, HoverTool, LabelSet
            from bokeh.palettes import Spectral8, Category20
            import networkx as nx
            from collections import defaultdict, Counter
            
            b = self.biblio
            
            # Get authors column
            auth_col = b._get_column("Authors", required=False)
            if not auth_col or auth_col not in b.df.columns:
                return ""
            
            # Build co-authorship network
            coauthorship = defaultdict(int)
            author_docs = Counter()
            author_citations = defaultdict(int)
            
            sep = b.default_separator
            cite_col = b._get_column("Cited by", required=False)
            
            for idx, row in b.df.iterrows():
                authors_str = row[auth_col]
                if pd.isna(authors_str):
                    continue
                
                author_list = [a.strip() for a in str(authors_str).split(sep) if a.strip()]
                
                # Count documents per author
                for auth in author_list:
                    author_docs[auth] += 1
                    if cite_col and cite_col in b.df.columns and not pd.isna(row[cite_col]):
                        author_citations[auth] += int(row[cite_col])
                
                # Count co-authorships
                for i, auth1 in enumerate(author_list):
                    for auth2 in author_list[i+1:]:
                        if auth1 != auth2:
                            pair = tuple(sorted([auth1, auth2]))
                            coauthorship[pair] += 1
            
            if not author_docs:
                return ""
            
            # Get top authors by document count
            top_n = min(40, len(author_docs))
            top_authors = set(a for a, _ in author_docs.most_common(top_n))
            
            # Build network
            G = nx.Graph()
            
            # Add nodes (top authors only)
            for auth in top_authors:
                G.add_node(
                    auth, 
                    documents=author_docs[auth],
                    citations=author_citations[auth],
                )
            
            # Add edges (only between top authors, min 1 collaboration)
            for (auth1, auth2), count in coauthorship.items():
                if auth1 in top_authors and auth2 in top_authors and count >= 1:
                    G.add_edge(auth1, auth2, weight=count)
            
            if len(G.nodes()) < 3:
                return ""
            
            # Detect communities for coloring
            try:
                communities = list(nx.community.greedy_modularity_communities(G))
                node_community = {}
                for i, comm in enumerate(communities):
                    for node in comm:
                        node_community[node] = i
            except:
                node_community = {node: 0 for node in G.nodes()}
            
            num_communities = len(set(node_community.values()))
            
            # Calculate layout
            if len(G.edges()) > 0:
                pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
            else:
                # Circular layout if no edges
                pos = nx.circular_layout(G)
            
            # Create Bokeh plot
            p = figure(
                title="Author Collaboration Network",
                width=self.config.width - 80,
                height=self.config.plot_height + 200,
                tools="pan,wheel_zoom,box_zoom,reset,save",
                toolbar_location="above",
                x_range=(-2.5, 2.5),
                y_range=(-2.5, 2.5),
            )
            
            p.xaxis.visible = False
            p.yaxis.visible = False
            p.xgrid.visible = False
            p.ygrid.visible = False
            
            # Color palette for communities
            if num_communities <= 8:
                colors = Spectral8
            else:
                colors = Category20[20]
            
            # Prepare node data
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            node_names = list(G.nodes())
            node_docs = [G.nodes[node]['documents'] for node in G.nodes()]
            node_cites = [G.nodes[node]['citations'] for node in G.nodes()]
            node_degree = [G.degree(node) for node in G.nodes()]
            
            max_docs = max(node_docs) if node_docs else 1
            node_sizes = [max(12, min(50, (d / max_docs) * 40 + 12)) for d in node_docs]
            
            node_colors = [colors[node_community.get(node, 0) % len(colors)] for node in G.nodes()]
            
            # Shorten names for display
            short_names = []
            for name in node_names:
                if len(name) > 20:
                    short_names.append(name[:18] + "...")
                else:
                    short_names.append(name)
            
            node_source = ColumnDataSource(data={
                "x": node_x,
                "y": node_y,
                "name": node_names,
                "short_name": short_names,
                "documents": node_docs,
                "citations": node_cites,
                "collaborators": node_degree,
                "size": node_sizes,
                "color": node_colors,
            })
            
            # Draw edges with width based on collaboration strength
            edge_data = []
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                weight = edge[2].get('weight', 1)
                edge_data.append((x0, y0, x1, y1, weight))
            
            if edge_data:
                max_weight = max(e[4] for e in edge_data)
                for x0, y0, x1, y1, weight in edge_data:
                    line_width = max(0.5, min(4, (weight / max_weight) * 4))
                    p.line([x0, x1], [y0, y1], line_color="#AAAAAA", line_alpha=0.5, line_width=line_width)
            
            # Draw nodes
            nodes = p.scatter(
                "x", "y", 
                size="size",
                source=node_source,
                fill_color="color",
                fill_alpha=0.8,
                line_color="white",
                line_width=1.5,
            )
            
            # Add labels for top authors only (by degree)
            top_by_degree = sorted(range(len(node_degree)), key=lambda i: node_degree[i], reverse=True)[:15]
            label_source = ColumnDataSource(data={
                "x": [node_x[i] for i in top_by_degree],
                "y": [node_y[i] for i in top_by_degree],
                "name": [short_names[i] for i in top_by_degree],
            })
            
            labels = LabelSet(
                x="x", y="y", text="name",
                source=label_source,
                x_offset=5, y_offset=5,
                text_font_size="8pt",
                text_color="#333333",
            )
            p.add_layout(labels)
            
            # Hover
            hover = HoverTool(
                tooltips=[
                    ("Author", "@name"),
                    ("Documents", "@documents"),
                    ("Citations", "@citations"),
                    ("Collaborators", "@collaborators"),
                ],
                renderers=[nodes]
            )
            p.add_tools(hover)
            
            p.title.text_font_size = "13pt"
            
            script, div = components(p)
            
            # Calculate network metrics
            avg_degree = sum(node_degree) / len(node_degree) if node_degree else 0
            
            # Top collaborators table
            top_collabs = sorted(
                [(n, G.nodes[n]['documents'], G.nodes[n]['citations'], G.degree(n)) 
                 for n in G.nodes()],
                key=lambda x: x[3], reverse=True
            )[:10]
            
            collab_df = pd.DataFrame(
                top_collabs, 
                columns=["Author", "Documents", "Citations", "Collaborators"]
            )
            
            # Network stats
            stats_html = f"""
            <div class="stats-grid" style="margin-bottom: 15px;">
                <div class="stat-card">
                    <div class="stat-value">{len(G.nodes())}</div>
                    <div class="stat-label">Authors</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(G.edges())}</div>
                    <div class="stat-label">Collaborations</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{nx.density(G):.3f}</div>
                    <div class="stat-label">Density</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{num_communities}</div>
                    <div class="stat-label">Communities</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_degree:.1f}</div>
                    <div class="stat-label">Avg Collaborators</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{nx.number_connected_components(G)}</div>
                    <div class="stat-label">Components</div>
                </div>
            </div>
            """
            
            return f"""
            <section id="collaboration_network" class="section">
                <h2 class="section-title">🤝 Author Collaboration Network</h2>
                <p style="font-size: 0.85rem; color: #666; margin-bottom: 10px;">
                    Node size = documents | Color = research community | Edge width = collaboration strength
                </p>
                {stats_html}
                <div class="chart-container">
                    {div}
                    {script}
                </div>
                <h3 style="margin-top: 20px; margin-bottom: 10px; font-size: 1.1rem;">Top Collaborators</h3>
                {self._create_table(collab_df, "collab_table")}
            </section>
            """
        except Exception as e:
            return ""
    
    def _build_citations_section(self) -> str:
        """Build citations analysis section."""
        b = self.biblio
        
        cite_col = b._get_column("Cited by", required=False)
        if not cite_col or cite_col not in b.df.columns:
            return ""
        
        citations = b.df[cite_col].fillna(0)
        
        # Citation statistics
        total_cites = int(citations.sum())
        mean_cites = round(citations.mean(), 2)
        median_cites = round(citations.median(), 2)
        max_cites = int(citations.max())
        h_index = self._calculate_h_index(citations)
        
        # Uncited papers
        uncited = int((citations == 0).sum())
        uncited_pct = round(100 * uncited / len(citations), 1)
        
        # Citation distribution
        from bokeh.plotting import figure
        from bokeh.embed import components
        from bokeh.models import ColumnDataSource, HoverTool
        
        # Create histogram data
        bins = [0, 1, 5, 10, 20, 50, 100, 500, 1000, float('inf')]
        bin_labels = ["0", "1-4", "5-9", "10-19", "20-49", "50-99", "100-499", "500-999", "1000+"]
        
        hist_counts = []
        for i in range(len(bins) - 1):
            count = int(((citations >= bins[i]) & (citations < bins[i+1])).sum())
            hist_counts.append(count)
        
        # Only show bins with data
        non_zero = [(l, c) for l, c in zip(bin_labels, hist_counts) if c > 0]
        if non_zero:
            labels, counts = zip(*non_zero)
        else:
            labels, counts = bin_labels[:3], [0, 0, 0]
        
        source = ColumnDataSource(data={
            "labels": list(labels),
            "counts": list(counts),
        })
        
        p = figure(
            title="Citation Distribution",
            x_range=list(labels),
            width=self.config.width - 80,
            height=self.config.plot_height - 50,
            tools="pan,wheel_zoom,reset,save",
            toolbar_location="above",
        )
        
        p.vbar(
            x="labels",
            top="counts",
            width=0.7,
            source=source,
            color=self.config.primary_color,
        )
        
        hover = HoverTool(tooltips=[("Citations", "@labels"), ("Documents", "@counts")])
        p.add_tools(hover)
        
        p.xaxis.axis_label = "Citation Range"
        p.yaxis.axis_label = "Number of Documents"
        p.xaxis.major_label_orientation = 0.7
        p.title.text_font_size = "13pt"
        
        script, div = components(p)
        
        # Most cited papers table
        title_col = b._get_column("Title", required=False)
        year_col = b._get_column("Year", required=False)
        
        top_cited_html = ""
        if title_col and title_col in b.df.columns:
            top_df = b.df.nlargest(10, cite_col)[[title_col, cite_col]]
            if year_col and year_col in b.df.columns:
                top_df = b.df.nlargest(10, cite_col)[[title_col, year_col, cite_col]]
            top_df.columns = ["Title", "Year", "Citations"] if year_col else ["Title", "Citations"]
            top_cited_html = f"""
            <h3 style="margin-top: 20px; margin-bottom: 10px; font-size: 1.1rem;">Most Cited Documents</h3>
            {self._create_table(top_df, "top_cited_table")}
            """
        
        return f"""
        <section id="citations" class="section">
            <h2 class="section-title">📈 Citation Analysis</h2>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{total_cites:,}</div>
                    <div class="stat-label">Total Citations</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{mean_cites}</div>
                    <div class="stat-label">Mean Citations</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{median_cites}</div>
                    <div class="stat-label">Median Citations</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{max_cites:,}</div>
                    <div class="stat-label">Max Citations</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{h_index}</div>
                    <div class="stat-label">H-index</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{uncited_pct}%</div>
                    <div class="stat-label">Uncited</div>
                </div>
            </div>
            
            <div class="chart-container">
                {div}
                {script}
            </div>
            
            {top_cited_html}
        </section>
        """
    
    def _calculate_h_index(self, citations) -> int:
        """Calculate h-index from citation series."""
        sorted_cites = sorted(citations, reverse=True)
        h = 0
        for i, c in enumerate(sorted_cites, 1):
            if c >= i:
                h = i
            else:
                break
        return h
    
    def _build_countries_section(self) -> str:
        """Build countries/affiliations analysis section."""
        b = self.biblio
        
        # Try to find country or affiliation data
        aff_col = b._get_column("Affiliations", required=False)
        country_col = None
        
        # Check for country column
        for col_name in ["Countries", "Country", "Author Countries"]:
            if col_name in b.df.columns:
                country_col = col_name
                break
        
        if not country_col and not aff_col:
            return ""
        
        from collections import Counter
        
        country_counts = Counter()
        sep = b.default_separator
        
        if country_col and country_col in b.df.columns:
            for countries in b.df[country_col].dropna():
                for country in str(countries).split(sep):
                    country = country.strip()
                    if country:
                        country_counts[country] += 1
        elif aff_col and aff_col in b.df.columns:
            # Try to extract countries from affiliations
            # Common country patterns at end of affiliation strings
            import re
            country_pattern = re.compile(r',\s*([A-Za-z\s]+)$')
            
            for affs in b.df[aff_col].dropna():
                for aff in str(affs).split(sep):
                    match = country_pattern.search(aff.strip())
                    if match:
                        country = match.group(1).strip()
                        if len(country) > 2 and len(country) < 30:
                            country_counts[country] += 1
        
        if not country_counts:
            return ""
        
        # Create dataframe
        top_n = min(15, len(country_counts))
        top_countries = country_counts.most_common(top_n)
        
        df = pd.DataFrame(top_countries, columns=["Country", "Documents"])
        
        from bokeh.plotting import figure
        from bokeh.embed import components
        from bokeh.models import ColumnDataSource
        
        # Reverse for horizontal bar chart (top at top)
        df_rev = df.iloc[::-1].reset_index(drop=True)
        
        source = ColumnDataSource(data={
            "country": df_rev["Country"].tolist(),
            "docs": df_rev["Documents"].tolist(),
        })
        
        p = figure(
            title=f"Top {top_n} Countries/Regions",
            y_range=df_rev["Country"].tolist(),
            width=self.config.width - 80,
            height=max(self.config.plot_height, top_n * 25 + 80),
            tools="pan,wheel_zoom,reset,save",
            toolbar_location="above",
        )
        
        p.hbar(
            y="country",
            right="docs",
            height=0.7,
            source=source,
            color=self.config.accent_color,
        )
        
        p.xaxis.axis_label = "Number of Documents"
        p.title.text_font_size = "13pt"
        p.ygrid.visible = False
        
        script, div = components(p)
        
        return f"""
        <section id="countries" class="section">
            <h2 class="section-title">🌍 Geographic Distribution</h2>
            
            <div class="chart-container">
                {div}
                {script}
            </div>
            
            {self._create_table(df, "countries_table") if self.config.include_tables else ""}
        </section>
        """

    def _build_documents_section(self) -> str:
        """Build document types section."""
        if not hasattr(self.biblio, "document_types_counts_df") or self.biblio.document_types_counts_df is None:
            return ""
        
        df = self.biblio.document_types_counts_df.copy()
        
        # Find columns
        name_col = "Document Type" if "Document Type" in df.columns else df.columns[0]
        count_col = next((c for c in df.columns if "number" in c.lower() or "count" in c.lower()), df.columns[-1])
        
        chart_html, chart_js = self._create_bar_chart(
            df,
            x=name_col,
            y=count_col,
            title="Document Types",
            chart_id="doctypes_chart"
        )
        
        return f"""
        <section id="documents" class="section">
            <h2 class="section-title">📄 Document Types</h2>
            
            <div class="chart-container" id="doctypes_chart"></div>
            {chart_js}
            
            {self._create_table(df, "doctypes_table") if self.config.include_tables else ""}
        </section>
        """
    
    def _create_table(self, df: pd.DataFrame, table_id: str) -> str:
        """Create HTML table from DataFrame."""
        # Format values
        def format_val(v):
            if pd.isna(v):
                return ""
            if isinstance(v, (int, np.integer)):
                return f"{v:,}"
            if isinstance(v, (float, np.floating)):
                return f"{v:.2f}"
            return str(v)[:50]
        
        headers = "".join(f"<th>{col}</th>" for col in df.columns)
        
        rows = []
        for _, row in df.iterrows():
            cells = "".join(f"<td>{format_val(v)}</td>" for v in row)
            rows.append(f"<tr>{cells}</tr>")
        
        return f"""
        <div class="table-container">
            <table id="{table_id}">
                <thead><tr>{headers}</tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """
    
    def _create_line_chart(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        title: str,
        chart_id: str,
    ) -> Tuple[str, str]:
        """Create a Bokeh line chart."""
        from bokeh.plotting import figure
        from bokeh.embed import components
        from bokeh.models import HoverTool
        
        p = figure(
            title=title,
            width=self.config.width // 2 - 50,
            height=self.config.plot_height,
            tools="pan,wheel_zoom,reset,save",
            toolbar_location="above",
        )
        
        # Line
        p.line(
            df[x].values,
            df[y].values,
            line_width=3,
            color=self.config.primary_color,
        )
        
        # Points
        p.scatter(
            df[x].values,
            df[y].values,
            size=8,
            color=self.config.primary_color,
            fill_color="white",
            line_width=2,
        )
        
        # Styling
        p.xaxis.axis_label = x
        p.yaxis.axis_label = y
        p.title.text_font_size = "14pt"
        p.xgrid.grid_line_color = None
        
        # Hover
        hover = HoverTool(tooltips=[(x, "@x"), (y, "@y{0,0}")])
        p.add_tools(hover)
        
        script, div = components(p)
        
        return div, script
    
    def _create_area_chart(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        title: str,
        chart_id: str,
    ) -> Tuple[str, str]:
        """Create a Bokeh area chart."""
        from bokeh.plotting import figure
        from bokeh.embed import components
        
        p = figure(
            title=title,
            width=self.config.width // 2 - 50,
            height=self.config.plot_height,
            tools="pan,wheel_zoom,reset,save",
            toolbar_location="above",
        )
        
        x_vals = df[x].values
        y_vals = df[y].values
        
        # Area
        p.varea(
            x=x_vals,
            y1=0,
            y2=y_vals,
            fill_alpha=0.3,
            fill_color=self.config.accent_color,
        )
        
        # Line on top
        p.line(x_vals, y_vals, line_width=2, color=self.config.accent_color)
        
        p.xaxis.axis_label = x
        p.yaxis.axis_label = y
        p.title.text_font_size = "14pt"
        p.xgrid.grid_line_color = None
        
        script, div = components(p)
        
        return div, script
    
    def _create_bar_chart(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        title: str,
        chart_id: str,
    ) -> Tuple[str, str]:
        """Create a Bokeh vertical bar chart."""
        from bokeh.plotting import figure
        from bokeh.embed import components
        from bokeh.models import ColumnDataSource
        
        source = ColumnDataSource(data={
            x: df[x].astype(str).tolist(),
            y: df[y].tolist(),
        })
        
        p = figure(
            title=title,
            x_range=df[x].astype(str).tolist(),
            width=self.config.width - 80,
            height=self.config.plot_height,
            tools="pan,wheel_zoom,reset,save",
            toolbar_location="above",
        )
        
        p.vbar(
            x=x,
            top=y,
            width=0.7,
            source=source,
            color=self.config.primary_color,
        )
        
        p.xaxis.axis_label = x
        p.yaxis.axis_label = y
        p.title.text_font_size = "14pt"
        p.xgrid.grid_line_color = None
        p.xaxis.major_label_orientation = 0.7
        
        script, div = components(p)
        
        return div, script
    
    def _create_barh_chart(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        title: str,
        chart_id: str,
    ) -> Tuple[str, str]:
        """Create a Bokeh horizontal bar chart."""
        from bokeh.plotting import figure
        from bokeh.embed import components
        from bokeh.models import ColumnDataSource
        
        # Reverse for top items at top
        df = df.iloc[::-1].reset_index(drop=True)
        
        # Truncate long labels
        labels = [str(v)[:40] + "..." if len(str(v)) > 40 else str(v) for v in df[y]]
        
        source = ColumnDataSource(data={
            y: labels,
            x: df[x].tolist(),
        })
        
        p = figure(
            title=title,
            y_range=labels,
            width=self.config.width - 80,
            height=max(self.config.plot_height, len(df) * 25 + 100),
            tools="pan,wheel_zoom,reset,save",
            toolbar_location="above",
        )
        
        p.hbar(
            y=y,
            right=x,
            height=0.7,
            source=source,
            color=self.config.primary_color,
        )
        
        p.xaxis.axis_label = x
        p.title.text_font_size = "14pt"
        p.ygrid.grid_line_color = None
        
        script, div = components(p)
        
        return div, script
