# -*- coding: utf-8 -*-
"""
Geographic/Geospatial Analysis Module for Bibliometric Analysis

This module provides tools for analyzing geographic patterns in research
output, international collaborations, and institutional distributions.

Features implemented:
1. Country Extraction - Parse affiliations to extract countries
2. Choropleth Maps - Research output by country visualization
3. International Collaboration Networks - Country-level collaboration mapping
4. Institutional Geocoding - Map institutions to coordinates
5. Regional Research Specialization - Topic focus by region
6. Collaboration Patterns - Bilateral and multilateral analysis
7. Research Migration - Author mobility tracking
8. Continental/Regional Aggregation - Group by world regions

Created for integration with the Biblium bibliometric toolkit.

@author: Claude (Anthropic) for Lan.Umek
"""

from __future__ import annotations

import os
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from itertools import combinations
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
import seaborn as sns

# Import core biblium bridge
try:
    from biblium.addons.core_utils import get_core_function, CORE_AVAILABLE, use_core_or_fallback
except ImportError:
    CORE_AVAILABLE = False
    def get_core_function(name): return None
    def use_core_or_fallback(name, fallback, *args, **kwargs): return fallback(*args, **kwargs)

# Optional imports
try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

try:
    import pycountry
    PYCOUNTRY_AVAILABLE = True
except ImportError:
    PYCOUNTRY_AVAILABLE = False

try:
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# =============================================================================
# DEFAULT COLORMAPS (user-configurable)
# =============================================================================

CMAP_CONTINUOUS = "viridis"  # For continuous/sequential data
CATEGORICAL_COLOR = "lightblue"      # For categorical/discrete data

def set_default_cmaps(continuous: str = None):
    """Set default colormap for continuous data."""
    global CMAP_CONTINUOUS
    if continuous:
        CMAP_CONTINUOUS = continuous

# =============================================================================
# COUNTRY DATA AND MAPPINGS
# =============================================================================

# Country name variations and corrections
COUNTRY_ALIASES = {
    # United States variations
    "usa": "United States",
    "u.s.a.": "United States",
    "u.s.a": "United States",
    "us": "United States",
    "u.s.": "United States",
    "united states of america": "United States",
    "america": "United States",
    
    # United Kingdom variations
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "great britain": "United Kingdom",
    "britain": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "northern ireland": "United Kingdom",
    
    # China variations
    "p.r. china": "China",
    "pr china": "China",
    "p. r. china": "China",
    "peoples republic of china": "China",
    "people's republic of china": "China",
    "mainland china": "China",
    "chinese": "China",
    
    # Other common variations
    "korea": "South Korea",
    "republic of korea": "South Korea",
    "south korea": "South Korea",
    "korea, republic of": "South Korea",
    "north korea": "North Korea",
    "democratic people's republic of korea": "North Korea",
    
    "russia": "Russia",
    "russian federation": "Russia",
    
    "taiwan": "Taiwan",
    "taiwan, china": "Taiwan",
    "republic of china": "Taiwan",
    
    "hong kong": "Hong Kong",
    "hong kong sar": "Hong Kong",
    "hong kong, china": "Hong Kong",
    
    "uae": "United Arab Emirates",
    "u.a.e.": "United Arab Emirates",
    
    "the netherlands": "Netherlands",
    "holland": "Netherlands",
    
    "czech republic": "Czechia",
    "czechia": "Czechia",
    
    "viet nam": "Vietnam",
    "vietnam": "Vietnam",
    
    "iran": "Iran",
    "islamic republic of iran": "Iran",
    
    "türkiye": "Turkey",
    "turkiye": "Turkey",
    
    "brasil": "Brazil",
    "brazil": "Brazil",
    
    "deutschland": "Germany",
    "germany": "Germany",
    
    "españa": "Spain",
    "spain": "Spain",
    
    "italia": "Italy",
    "italy": "Italy",
    
    "france": "France",
    "république française": "France",
    
    "日本": "Japan",
    "japan": "Japan",
    
    "中国": "China",
    
    "singapore": "Singapore",
    "republic of singapore": "Singapore",
    
    "south africa": "South Africa",
    "republic of south africa": "South Africa",
    
    "new zealand": "New Zealand",
    "aotearoa": "New Zealand",
    
    "saudi arabia": "Saudi Arabia",
    "ksa": "Saudi Arabia",
}

# Country to ISO codes mapping
COUNTRY_ISO_CODES = {
    "United States": "USA",
    "United Kingdom": "GBR",
    "China": "CHN",
    "Germany": "DEU",
    "France": "FRA",
    "Japan": "JPN",
    "Canada": "CAN",
    "Australia": "AUS",
    "Italy": "ITA",
    "Spain": "ESP",
    "India": "IND",
    "Brazil": "BRA",
    "South Korea": "KOR",
    "Netherlands": "NLD",
    "Russia": "RUS",
    "Switzerland": "CHE",
    "Sweden": "SWE",
    "Poland": "POL",
    "Belgium": "BEL",
    "Austria": "AUT",
    "Denmark": "DNK",
    "Norway": "NOR",
    "Finland": "FIN",
    "Israel": "ISR",
    "Singapore": "SGP",
    "Taiwan": "TWN",
    "Hong Kong": "HKG",
    "Portugal": "PRT",
    "Greece": "GRC",
    "Ireland": "IRL",
    "Czechia": "CZE",
    "Turkey": "TUR",
    "Mexico": "MEX",
    "South Africa": "ZAF",
    "Argentina": "ARG",
    "Chile": "CHL",
    "Malaysia": "MYS",
    "Thailand": "THA",
    "Indonesia": "IDN",
    "Vietnam": "VNM",
    "Pakistan": "PAK",
    "Egypt": "EGY",
    "Iran": "IRN",
    "Saudi Arabia": "SAU",
    "United Arab Emirates": "ARE",
    "New Zealand": "NZL",
    "Hungary": "HUN",
    "Romania": "ROU",
    "Ukraine": "UKR",
    "Colombia": "COL",
    "Nigeria": "NGA",
    "Philippines": "PHL",
    "Bangladesh": "BGD",
    "Slovenia": "SVN",
    "Croatia": "HRV",
    "Slovakia": "SVK",
    "Serbia": "SRB",
    "Bulgaria": "BGR",
    "Lithuania": "LTU",
    "Latvia": "LVA",
    "Estonia": "EST",
    "Luxembourg": "LUX",
    "Iceland": "ISL",
    "Cyprus": "CYP",
    "Malta": "MLT",
}

# Country coordinates (capital cities for mapping)
COUNTRY_COORDINATES = {
    "United States": (38.9072, -77.0369),
    "United Kingdom": (51.5074, -0.1278),
    "China": (39.9042, 116.4074),
    "Germany": (52.5200, 13.4050),
    "France": (48.8566, 2.3522),
    "Japan": (35.6762, 139.6503),
    "Canada": (45.4215, -75.6972),
    "Australia": (-35.2809, 149.1300),
    "Italy": (41.9028, 12.4964),
    "Spain": (40.4168, -3.7038),
    "India": (28.6139, 77.2090),
    "Brazil": (-15.8267, -47.9218),
    "South Korea": (37.5665, 126.9780),
    "Netherlands": (52.3676, 4.9041),
    "Russia": (55.7558, 37.6173),
    "Switzerland": (46.9480, 7.4474),
    "Sweden": (59.3293, 18.0686),
    "Poland": (52.2297, 21.0122),
    "Belgium": (50.8503, 4.3517),
    "Austria": (48.2082, 16.3738),
    "Denmark": (55.6761, 12.5683),
    "Norway": (59.9139, 10.7522),
    "Finland": (60.1699, 24.9384),
    "Israel": (31.7683, 35.2137),
    "Singapore": (1.3521, 103.8198),
    "Taiwan": (25.0330, 121.5654),
    "Hong Kong": (22.3193, 114.1694),
    "Portugal": (38.7223, -9.1393),
    "Greece": (37.9838, 23.7275),
    "Ireland": (53.3498, -6.2603),
    "Czechia": (50.0755, 14.4378),
    "Turkey": (39.9334, 32.8597),
    "Mexico": (19.4326, -99.1332),
    "South Africa": (-25.7479, 28.2293),
    "Argentina": (-34.6037, -58.3816),
    "Chile": (-33.4489, -70.6693),
    "Malaysia": (3.1390, 101.6869),
    "Thailand": (13.7563, 100.5018),
    "Indonesia": (-6.2088, 106.8456),
    "Vietnam": (21.0285, 105.8542),
    "Pakistan": (33.6844, 73.0479),
    "Egypt": (30.0444, 31.2357),
    "Iran": (35.6892, 51.3890),
    "Saudi Arabia": (24.7136, 46.6753),
    "United Arab Emirates": (24.4539, 54.3773),
    "New Zealand": (-41.2865, 174.7762),
    "Hungary": (47.4979, 19.0402),
    "Romania": (44.4268, 26.1025),
    "Ukraine": (50.4501, 30.5234),
    "Colombia": (4.7110, -74.0721),
    "Nigeria": (9.0765, 7.3986),
    "Philippines": (14.5995, 120.9842),
    "Bangladesh": (23.8103, 90.4125),
    "Slovenia": (46.0569, 14.5058),
    "Croatia": (45.8150, 15.9819),
    "Slovakia": (48.1486, 17.1077),
    "Serbia": (44.7866, 20.4489),
    "Bulgaria": (42.6977, 23.3219),
    "Lithuania": (54.6872, 25.2797),
    "Latvia": (56.9496, 24.1052),
    "Estonia": (59.4370, 24.7536),
    "Luxembourg": (49.6116, 6.1319),
    "Iceland": (64.1466, -21.9426),
    "Cyprus": (35.1856, 33.3823),
    "Malta": (35.8989, 14.5146),
}

# World regions mapping
COUNTRY_REGIONS = {
    "United States": "North America",
    "Canada": "North America",
    "Mexico": "North America",
    
    "United Kingdom": "Europe",
    "Germany": "Europe",
    "France": "Europe",
    "Italy": "Europe",
    "Spain": "Europe",
    "Netherlands": "Europe",
    "Switzerland": "Europe",
    "Sweden": "Europe",
    "Poland": "Europe",
    "Belgium": "Europe",
    "Austria": "Europe",
    "Denmark": "Europe",
    "Norway": "Europe",
    "Finland": "Europe",
    "Portugal": "Europe",
    "Greece": "Europe",
    "Ireland": "Europe",
    "Czechia": "Europe",
    "Hungary": "Europe",
    "Romania": "Europe",
    "Slovenia": "Europe",
    "Croatia": "Europe",
    "Slovakia": "Europe",
    "Serbia": "Europe",
    "Bulgaria": "Europe",
    "Lithuania": "Europe",
    "Latvia": "Europe",
    "Estonia": "Europe",
    "Luxembourg": "Europe",
    "Iceland": "Europe",
    "Cyprus": "Europe",
    "Malta": "Europe",
    "Ukraine": "Europe",
    
    "Russia": "Europe/Asia",
    "Turkey": "Europe/Asia",
    
    "China": "Asia",
    "Japan": "Asia",
    "South Korea": "Asia",
    "India": "Asia",
    "Taiwan": "Asia",
    "Hong Kong": "Asia",
    "Singapore": "Asia",
    "Malaysia": "Asia",
    "Thailand": "Asia",
    "Indonesia": "Asia",
    "Vietnam": "Asia",
    "Pakistan": "Asia",
    "Bangladesh": "Asia",
    "Philippines": "Asia",
    "Iran": "Asia",
    "Israel": "Asia",
    "Saudi Arabia": "Asia",
    "United Arab Emirates": "Asia",
    
    "Australia": "Oceania",
    "New Zealand": "Oceania",
    
    "Brazil": "South America",
    "Argentina": "South America",
    "Chile": "South America",
    "Colombia": "South America",
    
    "South Africa": "Africa",
    "Egypt": "Africa",
    "Nigeria": "Africa",
}

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CountryMetrics:
    """Metrics for a single country."""
    country: str
    iso_code: str
    region: str
    n_papers: int
    n_authors: int
    total_citations: int
    mean_citations: float
    h_index: int
    international_collab_rate: float
    top_collaborators: List[Tuple[str, int]]
    top_keywords: List[Tuple[str, int]]
    year_range: Tuple[int, int]
    coordinates: Optional[Tuple[float, float]]

@dataclass
class CollaborationLink:
    """Collaboration between two countries."""
    country1: str
    country2: str
    n_collaborations: int
    collaboration_strength: float  # Normalized
    top_keywords: List[str]
    year_range: Tuple[int, int]

@dataclass
class InstitutionMetrics:
    """Metrics for a single institution."""
    name: str
    country: str
    city: Optional[str]
    n_papers: int
    total_citations: int
    coordinates: Optional[Tuple[float, float]]
    collaborating_institutions: List[str]

@dataclass
class GeoAnalysisResult:
    """Complete geographic analysis results."""
    country_metrics: Dict[str, CountryMetrics]
    country_df: pd.DataFrame
    collaboration_links: List[CollaborationLink]
    collaboration_matrix: pd.DataFrame
    institution_metrics: Dict[str, InstitutionMetrics]
    regional_summary: pd.DataFrame
    temporal_trends: pd.DataFrame
    parameters: Dict[str, Any]

# =============================================================================
# COUNTRY EXTRACTION
# =============================================================================

def normalize_country_name(country: str) -> Optional[str]:
    """
    Normalize country name to standard form.
    
    Parameters
    ----------
    country : str
        Raw country name.
    
    Returns
    -------
    Normalized country name or None.
    """
    if not country or pd.isna(country):
        return None
    
    country = str(country).strip().lower()
    
    # Remove common suffixes/prefixes
    country = re.sub(r"^\s*the\s+", "", country)
    country = re.sub(r"\s*\([^)]*\)\s*$", "", country)
    
    # Check aliases
    if country in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[country]
    
    # Check if it matches a known country (case-insensitive)
    for known_country in COUNTRY_ISO_CODES.keys():
        if country == known_country.lower():
            return known_country
    
    # Try pycountry if available
    if PYCOUNTRY_AVAILABLE:
        try:
            result = pycountry.countries.search_fuzzy(country)
            if result:
                return result[0].name
        except:
            pass
    
    # Return title case if no match
    return country.title() if len(country) > 2 else None

def extract_countries_from_affiliation(
    affiliation: str,
    return_all: bool = True) -> List[str]:
    """
    Extract country names from affiliation string.
    
    Parameters
    ----------
    affiliation : str
        Affiliation string.
    return_all : bool
        Return all countries found (True) or just first (False).
    
    Returns
    -------
    List of country names.
    """
    if not affiliation or pd.isna(affiliation):
        return []
    
    affiliation = str(affiliation)
    countries = []
    
    # Split by common delimiters
    parts = re.split(r"[;|]", affiliation)
    
    for part in parts:
        # Try to find country at end of affiliation (most common pattern)
        # Pattern: "..., City, Country" or "..., Country"
        segments = [s.strip() for s in part.split(",")]
        
        for segment in reversed(segments[-3:]):  # Check last 3 segments
            normalized = normalize_country_name(segment)
            if normalized and normalized in COUNTRY_ISO_CODES:
                if normalized not in countries:
                    countries.append(normalized)
                break
        
        # Also check for country patterns anywhere in text
        for alias, standard in COUNTRY_ALIASES.items():
            if re.search(r"\b" + re.escape(alias) + r"\b", part.lower()):
                if standard not in countries:
                    countries.append(standard)
    
    # Check for country names directly
    for country in COUNTRY_ISO_CODES.keys():
        if re.search(r"\b" + re.escape(country) + r"\b", affiliation, re.IGNORECASE):
            if country not in countries:
                countries.append(country)
    
    if return_all:
        return countries
    else:
        return countries[:1] if countries else []

def extract_countries_from_dataframe(
    df: pd.DataFrame,
    affiliation_col: str = "Affiliations",
    country_col: str = None,
    authors_col: str = "Authors",
    sep: str = "; ") -> pd.DataFrame:
    """
    Extract countries for all papers in dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    affiliation_col : str
        Affiliations column.
    country_col : str, optional
        Pre-existing country column.
    authors_col : str
        Authors column.
    sep : str
        Separator.
    
    Returns
    -------
    DataFrame with added country columns.
    """
    df = df.copy()
    
    # If country column already exists, use it
    if country_col and country_col in df.columns:
        df["_countries"] = df[country_col].apply(
            lambda x: [normalize_country_name(c.strip()) for c in str(x).split(sep) if c.strip()]
            if pd.notna(x) else []
        )
    elif affiliation_col in df.columns:
        df["_countries"] = df[affiliation_col].apply(extract_countries_from_affiliation)
    else:
        df["_countries"] = [[] for _ in range(len(df))]
    
    # Create derived columns
    df["_n_countries"] = df["_countries"].apply(len)
    df["_first_country"] = df["_countries"].apply(lambda x: x[0] if x else None)
    df["_is_international"] = df["_n_countries"] > 1
    
    return df

# =============================================================================
# COUNTRY-LEVEL ANALYSIS
# =============================================================================

def analyze_countries(
    df: pd.DataFrame,
    affiliation_col: str = "Affiliations",
    country_col: str = None,
    citations_col: str = "Cited by",
    year_col: str = "Year",
    keywords_col: str = "Author Keywords",
    authors_col: str = "Authors",
    sep: str = "; ",
    verbose: bool = True) -> Tuple[Dict[str, CountryMetrics], pd.DataFrame]:
    """
    Analyze research output by country.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    Various column parameters.
    verbose : bool
        Print progress.
    
    Returns
    -------
    Tuple of (country_metrics dict, summary dataframe).
    """
    if verbose:
        print("Analyzing countries...")
    
    # Extract countries
    df = extract_countries_from_dataframe(df, affiliation_col, country_col, authors_col, sep)
    
    # Aggregate by country
    country_data = defaultdict(lambda: {
        "papers": [],
        "citations": [],
        "years": [],
        "keywords": [],
        "collaborators": [],
        "authors": set(),
    })
    
    for idx, row in df.iterrows():
        countries = row.get("_countries", [])
        citations = row.get(citations_col, 0) if citations_col in df.columns else 0
        year = row.get(year_col) if year_col in df.columns else None
        keywords = row.get(keywords_col, "") if keywords_col in df.columns else ""
        
        if pd.isna(citations):
            citations = 0
        
        for country in countries:
            if country:
                country_data[country]["papers"].append(idx)
                country_data[country]["citations"].append(int(citations))
                if year and pd.notna(year):
                    country_data[country]["years"].append(int(year))
                
                # Keywords
                if keywords and pd.notna(keywords):
                    for kw in str(keywords).split(sep):
                        kw = kw.strip().lower()
                        if kw:
                            country_data[country]["keywords"].append(kw)
                
                # Collaborators
                for other_country in countries:
                    if other_country and other_country != country:
                        country_data[country]["collaborators"].append(other_country)
    
    # Build metrics
    country_metrics = {}
    summary_data = []
    
    for country, data in country_data.items():
        n_papers = len(data["papers"])
        citations = data["citations"]
        years = data["years"]
        
        # H-index
        sorted_cit = sorted(citations, reverse=True)
        h_index = sum(1 for i, c in enumerate(sorted_cit) if c >= i + 1)
        
        # International collaboration rate
        n_international = len(data["collaborators"])
        intl_rate = n_international / n_papers if n_papers > 0 else 0
        
        # Top collaborators
        collab_counts = Counter(data["collaborators"])
        top_collabs = collab_counts.most_common(10)
        
        # Top keywords
        kw_counts = Counter(data["keywords"])
        top_kws = kw_counts.most_common(10)
        
        # Year range
        year_range = (min(years), max(years)) if years else (0, 0)
        
        # Coordinates
        coords = COUNTRY_COORDINATES.get(country)
        
        # Region
        region = COUNTRY_REGIONS.get(country, "Other")
        
        # ISO code
        iso_code = COUNTRY_ISO_CODES.get(country, "")
        
        metrics = CountryMetrics(
            country=country,
            iso_code=iso_code,
            region=region,
            n_papers=n_papers,
            n_authors=len(data["authors"]),
            total_citations=sum(citations),
            mean_citations=np.mean(citations) if citations else 0,
            h_index=h_index,
            international_collab_rate=intl_rate,
            top_collaborators=top_collabs,
            top_keywords=top_kws,
            year_range=year_range,
            coordinates=coords)
        
        country_metrics[country] = metrics
        
        summary_data.append({
            "Country": country,
            "ISO_Code": iso_code,
            "Region": region,
            "Papers": n_papers,
            "Citations": sum(citations),
            "Mean Citations": round(np.mean(citations), 2) if citations else 0,
            "H_Index": h_index,
            "Intl Collab Rate": round(intl_rate, 3),
            "Latitude": coords[0] if coords else None,
            "Longitude": coords[1] if coords else None,
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values("Papers", ascending=False)
    
    if verbose:
        print(f"  Found {len(country_metrics)} countries")
        print(f"  Top 5: {', '.join(summary_df.head(5)['Country'].tolist())}")
    
    return country_metrics, summary_df

# =============================================================================
# COLLABORATION ANALYSIS
# =============================================================================

def analyze_collaborations(
    df: pd.DataFrame,
    affiliation_col: str = "Affiliations",
    country_col: str = None,
    keywords_col: str = "Author Keywords",
    year_col: str = "Year",
    sep: str = "; ",
    min_collaborations: int = 1,
    verbose: bool = True) -> Tuple[List[CollaborationLink], pd.DataFrame]:
    """
    Analyze international collaboration patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    Various column parameters.
    min_collaborations : int
        Minimum collaborations to include.
    verbose : bool
        Print progress.
    
    Returns
    -------
    Tuple of (collaboration links, collaboration matrix).
    """
    if verbose:
        print("Analyzing international collaborations...")
    
    # Extract countries
    df = extract_countries_from_dataframe(df, affiliation_col, country_col, sep=sep)
    
    # Count collaborations
    collab_counts = Counter()
    collab_keywords = defaultdict(list)
    collab_years = defaultdict(list)
    
    for idx, row in df.iterrows():
        countries = row.get("_countries", [])
        countries = list(set(countries))  # Unique countries per paper
        
        if len(countries) >= 2:
            year = row.get(year_col) if year_col in df.columns else None
            keywords = row.get(keywords_col, "") if keywords_col in df.columns else ""
            
            for c1, c2 in combinations(sorted(countries), 2):
                key = (c1, c2)
                collab_counts[key] += 1
                
                if year and pd.notna(year):
                    collab_years[key].append(int(year))
                
                if keywords and pd.notna(keywords):
                    for kw in str(keywords).split(sep):
                        kw = kw.strip().lower()
                        if kw:
                            collab_keywords[key].append(kw)
    
    # Build collaboration links
    links = []
    max_collabs = max(collab_counts.values()) if collab_counts else 1
    
    for (c1, c2), count in collab_counts.items():
        if count >= min_collaborations:
            years = collab_years[(c1, c2)]
            year_range = (min(years), max(years)) if years else (0, 0)
            
            kw_counts = Counter(collab_keywords[(c1, c2)])
            top_kws = [kw for kw, _ in kw_counts.most_common(5)]
            
            links.append(CollaborationLink(
                country1=c1,
                country2=c2,
                n_collaborations=count,
                collaboration_strength=count / max_collabs,
                top_keywords=top_kws,
                year_range=year_range))
    
    # Build collaboration matrix
    all_countries = sorted(set(c for link in links for c in [link.country1, link.country2]))
    matrix = pd.DataFrame(0, index=all_countries, columns=all_countries)
    
    for link in links:
        matrix.loc[link.country1, link.country2] = link.n_collaborations
        matrix.loc[link.country2, link.country1] = link.n_collaborations
    
    if verbose:
        print(f"  Found {len(links)} collaboration pairs")
        if links:
            top_link = max(links, key=lambda x: x.n_collaborations)
            print(f"  Strongest: {top_link.country1} - {top_link.country2} ({top_link.n_collaborations})")
    
    return links, matrix

# =============================================================================
# REGIONAL ANALYSIS
# =============================================================================

def analyze_regions(
    country_df: pd.DataFrame,
    verbose: bool = True) -> pd.DataFrame:
    """
    Aggregate country data by world region.
    
    Parameters
    ----------
    country_df : pd.DataFrame
        Country summary dataframe.
    verbose : bool
        Print progress.
    
    Returns
    -------
    Regional summary dataframe.
    """
    if verbose:
        print("Analyzing by region...")
    
    if "Region" not in country_df.columns:
        country_df["Region"] = country_df["Country"].map(
            lambda x: COUNTRY_REGIONS.get(x, "Other")
        )
    
    regional = country_df.groupby("Region").agg({
        "Papers": "sum",
        "Citations": "sum",
        "Country": "count",
    }).reset_index()
    
    regional.columns = ["Region", "Papers", "Citations", "N_Countries"]
    regional["Mean_Citations"] = regional["Citations"] / regional["Papers"]
    regional["Papers_per_Country"] = regional["Papers"] / regional["N_Countries"]
    regional = regional.sort_values("Papers", ascending=False)
    
    if verbose:
        print(f"  Regions: {len(regional)}")
    
    return regional

# =============================================================================
# TEMPORAL ANALYSIS
# =============================================================================

def analyze_geographic_trends(
    df: pd.DataFrame,
    affiliation_col: str = "Affiliations",
    country_col: str = None,
    year_col: str = "Year",
    top_n_countries: int = 10,
    sep: str = "; ",
    verbose: bool = True) -> pd.DataFrame:
    """
    Analyze geographic trends over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    Various column parameters.
    top_n_countries : int
        Number of top countries to track.
    verbose : bool
        Print progress.
    
    Returns
    -------
    Temporal trends dataframe.
    """
    if verbose:
        print("Analyzing geographic trends...")
    
    # Extract countries
    df = extract_countries_from_dataframe(df, affiliation_col, country_col, sep=sep)
    
    # Get top countries
    all_countries = []
    for countries in df["_countries"]:
        all_countries.extend(countries)
    
    top_countries = [c for c, _ in Counter(all_countries).most_common(top_n_countries)]
    
    # Aggregate by year and country
    year_country_data = defaultdict(lambda: defaultdict(int))
    year_intl_data = defaultdict(lambda: {"total": 0, "international": 0})
    
    for idx, row in df.iterrows():
        year = row.get(year_col)
        if pd.isna(year):
            continue
        year = int(year)
        
        countries = row.get("_countries", [])
        is_intl = len(set(countries)) > 1
        
        year_intl_data[year]["total"] += 1
        if is_intl:
            year_intl_data[year]["international"] += 1
        
        for country in set(countries):
            if country in top_countries:
                year_country_data[year][country] += 1
    
    # Build trends dataframe
    trends_data = []
    for year in sorted(year_country_data.keys()):
        row_data = {"Year": year}
        for country in top_countries:
            row_data[country] = year_country_data[year].get(country, 0)
        
        row_data["Total"] = year_intl_data[year]["total"]
        row_data["International"] = year_intl_data[year]["international"]
        row_data["Intl_Rate"] = (
            year_intl_data[year]["international"] / year_intl_data[year]["total"]
            if year_intl_data[year]["total"] > 0 else 0
        )
        
        trends_data.append(row_data)
    
    trends_df = pd.DataFrame(trends_data)
    
    if verbose:
        print(f"  Tracked {len(top_countries)} countries over {len(trends_df)} years")
    
    return trends_df

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_geographic_analysis(
    df: pd.DataFrame,
    affiliation_col: str = "Affiliations",
    country_col: str = None,
    citations_col: str = "Cited by",
    year_col: str = "Year",
    keywords_col: str = "Author Keywords",
    authors_col: str = "Authors",
    sep: str = "; ",
    min_collaborations: int = 2,
    verbose: bool = True) -> GeoAnalysisResult:
    """
    Run comprehensive geographic analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    Various column parameters.
    min_collaborations : int
        Minimum collaborations for links.
    verbose : bool
        Print progress.
    
    Returns
    -------
    GeoAnalysisResult
    """
    if verbose:
        print("=" * 50)
        print("Geographic/Geospatial Analysis")
        print("=" * 50)
    
    # Country analysis
    country_metrics, country_df = analyze_countries(
        df, affiliation_col, country_col, citations_col,
        year_col, keywords_col, authors_col, sep, verbose
    )
    
    # Collaboration analysis
    collab_links, collab_matrix = analyze_collaborations(
        df, affiliation_col, country_col, keywords_col,
        year_col, sep, min_collaborations, verbose
    )
    
    # Regional analysis
    regional_df = analyze_regions(country_df, verbose)
    
    # Temporal trends
    temporal_df = analyze_geographic_trends(
        df, affiliation_col, country_col, year_col,
        top_n_countries=10, sep=sep, verbose=verbose
    )
    
    if verbose:
        print("\n" + "=" * 50)
        print("Analysis complete!")
    
    return GeoAnalysisResult(
        country_metrics=country_metrics,
        country_df=country_df,
        collaboration_links=collab_links,
        collaboration_matrix=collab_matrix,
        institution_metrics={},
        regional_summary=regional_df,
        temporal_trends=temporal_df,
        parameters={
            "affiliation_col": affiliation_col,
            "min_collaborations": min_collaborations,
        })

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_country_output(
    result: GeoAnalysisResult,
    top_n: int = 20,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None) -> plt.Figure:
    """Plot research output by country."""
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    df = result.country_df.head(top_n)
    
    # Papers by country
    bars = ax.barh(range(len(df)), df["Papers"], color="lightblue")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Country"])
    ax.set_xlabel("Number of Papers", fontsize=11)
    ax.set_title(f"Top {top_n} Countries by Research Output", fontsize=12)
    ax.invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df["Papers"])):
        ax.text(val + max(df["Papers"]) * 0.01, i, str(int(val)), va="center", fontsize=9)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_country_citations(
    result: GeoAnalysisResult,
    top_n: int = 20,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """Plot citations by country."""
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    df = result.country_df.nlargest(top_n, "Citations")
    
    bars = ax.barh(range(len(df)), df["Citations"], color="lightblue")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Country"])
    ax.set_xlabel("Total Citations", fontsize=11)
    ax.set_title(f"Top {top_n} Countries by Citations", fontsize=12)
    ax.invert_yaxis()
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_country_impact(
    result: GeoAnalysisResult,
    top_n: int = 20,
    min_papers: int = 10,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """Plot citation impact by country."""
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    impact_df = result.country_df[result.country_df["Papers"] >= min_papers].nlargest(top_n, "Mean_Citations")
    
    bars = ax.barh(range(len(impact_df)), impact_df["Mean_Citations"], color="lightblue")
    ax.set_yticks(range(len(impact_df)))
    ax.set_yticklabels(impact_df["Country"])
    ax.set_xlabel("Mean Citations per Paper", fontsize=11)
    ax.set_title(f"Top {top_n} Countries by Impact (min {min_papers} papers)", fontsize=12)
    ax.invert_yaxis()
    
    # Add paper count labels
    for i, (bar, papers) in enumerate(zip(bars, impact_df["Papers"])):
        ax.text(bar.get_width() + max(impact_df["Mean_Citations"]) * 0.01, i, 
                f"n={int(papers)}", va="center", fontsize=8, color="gray")
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_country_collaboration(
    result: GeoAnalysisResult,
    top_n: int = 20,
    min_papers: int = 10,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """Plot international collaboration rate by country."""
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    collab_df = result.country_df[result.country_df["Papers"] >= min_papers].nlargest(top_n, "Intl_Collab_Rate")
    
    bars = ax.barh(range(len(collab_df)), collab_df["Intl_Collab_Rate"] * 100, color="lightblue")
    ax.set_yticks(range(len(collab_df)))
    ax.set_yticklabels(collab_df["Country"])
    ax.set_xlabel("International Collaboration Rate (%)", fontsize=11)
    ax.set_title(f"Top {top_n} Countries by Intl. Collaboration", fontsize=12)
    ax.invert_yaxis()
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

def plot_collaboration_network(
    result: GeoAnalysisResult,
    top_n_countries: int = 30,
    min_collab_strength: float = 0.05,
    figsize: Tuple[int, int] = (14, 12),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None) -> plt.Figure:
    """Plot international collaboration network."""
    
    if not NETWORKX_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.grid(False)
        ax.text(0.5, 0.5, "NetworkX not available for network visualization",
               ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    # Build network
    G = nx.Graph()
    
    # Get top countries by papers
    top_countries = result.country_df.head(top_n_countries)["Country"].tolist()
    
    # Add nodes
    for country in top_countries:
        if country in result.country_metrics:
            metrics = result.country_metrics[country]
            G.add_node(country, papers=metrics.n_papers, citations=metrics.total_citations)
    
    # Add edges
    for link in result.collaboration_links:
        if (link.country1 in top_countries and link.country2 in top_countries
            and link.collaboration_strength >= min_collab_strength):
            G.add_edge(link.country1, link.country2,
                      weight=link.n_collaborations,
                      strength=link.collaboration_strength)
    
    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, "No collaboration data available",
               ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Node sizes based on papers
    node_sizes = [G.nodes[n].get("papers", 10) * 5 for n in G.nodes()]
    max_size = max(node_sizes) if node_sizes else 1
    node_sizes = [s / max_size * 2000 + 100 for s in node_sizes]
    
    # Node colors based on papers (continuous)
    node_papers = [G.nodes[n].get("papers", 0) for n in G.nodes()]
    node_colors = [p / max(node_papers) for p in node_papers]
    
    # Edge widths based on collaborations
    edge_weights = [G.edges[e].get("weight", 1) for e in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [w / max_weight * 5 + 0.5 for w in edge_weights]
    
    # Draw
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4, edge_color="gray", ax=ax)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                                   alpha=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)
    
    ax.set_title(f"International Collaboration Network (Top {top_n_countries} Countries)", fontsize=14)
    ax.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

def plot_collaboration_heatmap(
    result: GeoAnalysisResult,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None) -> plt.Figure:
    """Plot collaboration heatmap."""
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    # Get top countries
    top_countries = result.country_df.head(top_n)["Country"].tolist()
    
    # Filter matrix
    matrix = result.collaboration_matrix
    available = [c for c in top_countries if c in matrix.index]
    
    if len(available) < 2:
        ax.text(0.5, 0.5, "Insufficient collaboration data",
               ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig
    
    filtered_matrix = matrix.loc[available, available]
    
    # Plot heatmap
    mask = np.triu(np.ones_like(filtered_matrix, dtype=bool), k=1)
    
    sns.heatmap(filtered_matrix, annot=True, fmt="d", cmap=cmap,
               mask=mask, ax=ax, cbar_kws={"label": "Collaborations"})
    
    ax.set_title(f"Collaboration Matrix (Top {len(available)} Countries)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

def plot_regional_summary(
    result: GeoAnalysisResult,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None) -> plt.Figure:
    """Plot papers by region."""
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    df = result.regional_summary
    
    # Papers by region
    bars = ax.barh(range(len(df)), df["Papers"], color="lightblue")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Region"])
    ax.set_xlabel("Number of Papers", fontsize=11)
    ax.set_title("Research Output by Region", fontsize=12)
    ax.invert_yaxis()
    
    for bar, val in zip(bars, df["Papers"]):
        ax.text(bar.get_width() + max(df["Papers"]) * 0.01, bar.get_y() + bar.get_height()/2,
                f"{int(val):,}", ha="left", va="center", fontsize=9)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_regional_impact(
    result: GeoAnalysisResult,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """Plot citation impact by region."""
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    df = result.regional_summary
    
    # Mean citations by region
    bars = ax.barh(range(len(df)), df["Mean_Citations"], color="lightblue")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Region"])
    ax.set_xlabel("Mean Citations per Paper", fontsize=11)
    ax.set_title("Citation Impact by Region", fontsize=12)
    ax.invert_yaxis()
    
    for i, (bar, val, papers) in enumerate(zip(bars, df["Mean_Citations"], df["Papers"])):
        ax.text(bar.get_width() + max(df["Mean_Citations"]) * 0.01, i,
                f"{val:.1f}", ha="left", va="center", fontsize=9)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

def plot_geographic_trends(
    result: GeoAnalysisResult,
    top_n_countries: int = 8,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None) -> plt.Figure:
    """Plot geographic trends over time - stacked area chart."""
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    df = result.temporal_trends
    
    if len(df) == 0:
        ax.text(0.5, 0.5, "Temporal data not available", ha="center", va="center")
        ax.axis("off")
        return fig
    
    # Get country columns (excluding Year, Total, International, Intl_Rate)
    country_cols = [c for c in df.columns if c not in ["Year", "Total", "International", "Intl_Rate"]][:top_n_countries]
    
    # Using single color for all - lightblue with varying alpha
    ax.stackplot(df["Year"], [df[c] for c in country_cols], labels=country_cols, 
                 colors=["lightblue"] * len(country_cols), alpha=0.8)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Number of Papers", fontsize=11)
    ax.set_title("Research Output Over Time by Country", fontsize=12)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_total_output_trends(
    result: GeoAnalysisResult,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """Plot total output over time."""
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    df = result.temporal_trends
    
    if len(df) == 0:
        ax.text(0.5, 0.5, "Temporal data not available", ha="center", va="center")
        ax.axis("off")
        return fig
    
    ax.bar(df["Year"], df["Total"], color="lightblue")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Number of Papers", fontsize=11)
    ax.set_title("Total Output Over Time", fontsize=12)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_intl_collaboration_trends(
    result: GeoAnalysisResult,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """Plot international collaboration rate over time."""
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    
    df = result.temporal_trends
    
    if len(df) == 0:
        ax.text(0.5, 0.5, "Temporal data not available", ha="center", va="center")
        ax.axis("off")
        return fig
    
    ax.bar(df["Year"], df["Intl_Rate"] * 100, color="lightblue")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("International Collaboration Rate (%)", fontsize=11)
    ax.set_title("International Collaboration Rate Over Time", fontsize=12)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

def plot_world_map(
    result: GeoAnalysisResult,
    metric: str = "Papers",
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None) -> plt.Figure:
    """
    Plot choropleth world map.
    
    Requires geopandas and world shapefile.
    """
    
    if not GEOPANDAS_AVAILABLE:
        # Fallback: scatter plot on simple axes
        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(False)
        
        df = result.country_df
        
        # Filter to countries with coordinates
        df_with_coords = df[df["Latitude"].notna() & df["Longitude"].notna()].copy()
        
        if len(df_with_coords) == 0:
            ax.text(0.5, 0.5, "No coordinate data available", ha="center", va="center")
            ax.axis("off")
            return fig
        
        # Normalize sizes
        sizes = df_with_coords[metric].values
        max_size = max(sizes) if len(sizes) > 0 else 1
        normalized_sizes = (sizes / max_size) * 1000 + 50
        
        # Color by metric
        scatter = ax.scatter(
            df_with_coords["Longitude"],
            df_with_coords["Latitude"],
            s=normalized_sizes,
            c=df_with_coords[metric],
            cmap=cmap,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5)
        
        # Add country labels for top 15
        for _, row in df_with_coords.head(15).iterrows():
            ax.annotate(
                row["Country"],
                (row["Longitude"], row["Latitude"]),
                fontsize=8,
                ha="center",
                va="bottom")
        
        plt.colorbar(scatter, label=metric, shrink=0.7)
        
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.set_title(f"Research Output by Country ({metric})", fontsize=14)
        
    else:
        # Use geopandas for proper choropleth
        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(False)
        
        try:
            # Try to load world shapefile
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            
            # Merge with our data
            df = result.country_df.copy()
            
            # Create ISO code mapping
            world_merged = world.merge(df, left_on="iso_a3", right_on="ISO_Code", how="left")
            
            # Plot
            world_merged.plot(
                column=metric,
                ax=ax,
                legend=True,
                legend_kwds={"label": metric, "shrink": 0.7},
                missing_kwds={"color": "lightgray"},
                cmap=cmap)
            
            ax.set_title(f"Research Output by Country ({metric})", fontsize=14)
            ax.axis("off")
            
        except Exception as e:
            # Fallback to scatter
            ax.text(0.5, 0.5, f"Map rendering failed: {e}", ha="center", va="center")
            ax.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

def create_interactive_map(
    result: GeoAnalysisResult,
    metric: str = "Papers",
    save_path: Optional[str] = None) -> Optional[Any]:
    """
    Create interactive Folium map.
    
    Parameters
    ----------
    result : GeoAnalysisResult
        Analysis results.
    metric : str
        Metric to visualize.
    save_path : str, optional
        Path to save HTML file.
    
    Returns
    -------
    Folium Map object or None.
    """
    if not FOLIUM_AVAILABLE:
        print("Folium not available for interactive maps")
        return None
    
    # Create base map
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")
    
    df = result.country_df
    max_val = df[metric].max()
    
    # Add markers for each country
    for _, row in df.iterrows():
        if pd.notna(row.get("Latitude")) and pd.notna(row.get("Longitude")):
            # Scale radius
            radius = (row[metric] / max_val) * 30 + 5
            
            # Popup content
            popup_html = f"""
            <b>{row['Country']}</b><br>
            Papers: {row['Papers']:,}<br>
            Citations: {row['Citations']:,}<br>
            Mean Citations: {row['Mean_Citations']:.1f}<br>
            H-Index: {row['H_Index']}<br>
            Intl. Collab Rate: {row['Intl_Collab_Rate']*100:.1f}%
            """
            
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=radius,
                popup=folium.Popup(popup_html, max_width=300),
                color="steelblue",
                fill=True,
                fill_color="steelblue",
                fill_opacity=0.6).add_to(m)
    
    # Add collaboration lines for top collaborations
    top_links = sorted(result.collaboration_links, key=lambda x: -x.n_collaborations)[:50]
    
    for link in top_links:
        c1_metrics = result.country_metrics.get(link.country1)
        c2_metrics = result.country_metrics.get(link.country2)
        
        if c1_metrics and c2_metrics and c1_metrics.coordinates and c2_metrics.coordinates:
            coords = [
                [c1_metrics.coordinates[0], c1_metrics.coordinates[1]],
                [c2_metrics.coordinates[0], c2_metrics.coordinates[1]],
            ]
            
            weight = (link.collaboration_strength * 3) + 0.5
            
            folium.PolyLine(
                coords,
                weight=weight,
                color="coral",
                opacity=0.5).add_to(m)
    
    # Save if path provided
    if save_path:
        m.save(save_path)
        print(f"Interactive map saved to: {save_path}")
    
    return m

# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_geographic_results(
    result: GeoAnalysisResult,
    output_dir: str) -> Dict[str, str]:
    """Export geographic analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    excel_path = os.path.join(output_dir, "geographic_analysis.xlsx")
    
    with pd.ExcelWriter(excel_path) as writer:
        # Country summary
        result.country_df.to_excel(writer, sheet_name="Countries", index=False)
        
        # Regional summary
        result.regional_summary.to_excel(writer, sheet_name="Regions", index=False)
        
        # Collaboration matrix
        result.collaboration_matrix.to_excel(writer, sheet_name="Collaboration_Matrix")
        
        # Collaboration links
        collab_data = []
        for link in result.collaboration_links:
            collab_data.append({
                "Country1": link.country1,
                "Country2": link.country2,
                "Collaborations": link.n_collaborations,
                "Strength": link.collaboration_strength,
                "Keywords": ", ".join(link.top_keywords),
            })
        pd.DataFrame(collab_data).to_excel(writer, sheet_name="Collaborations", index=False)
        
        # Temporal trends
        if len(result.temporal_trends) > 0:
            result.temporal_trends.to_excel(writer, sheet_name="Temporal_Trends", index=False)
    
    print(f"Exported to: {excel_path}")
    
    # Create interactive map in same folder
    html_path = os.path.join(output_dir, "interactive_map.html")
    create_interactive_map(result, save_path=html_path)
    
    return {"excel": excel_path, "html": html_path}

# =============================================================================
# CLASS INTEGRATION
# =============================================================================

def add_geographic_methods(cls):
    """Add geographic analysis methods to BiblioAnalysis class."""
    
    def run_geographic_analysis_method(
        self,
        affiliation_col: str = "Affiliations",
        country_col: str = None,
        min_collaborations: int = 2,
        **kwargs
    ) -> GeoAnalysisResult:
        """Run geographic analysis."""
        self.geographic_results = run_geographic_analysis(
            self.df,
            affiliation_col=affiliation_col,
            country_col=country_col,
            citations_col="Cited by",
            year_col="Year",
            keywords_col="Author Keywords",
            authors_col="Authors",
            min_collaborations=min_collaborations,
            **kwargs
        )
        
        if hasattr(self, "res_folder") and self.res_folder:
            export_geographic_results(
                self.geographic_results,
                self.res_folder
            )
        
        return self.geographic_results
    
    def plot_geographic_method(
        self,
        plot_type: str = "countries",
        save: bool = True,
        **kwargs
    ) -> plt.Figure:
        """Create geographic visualizations."""
        if not hasattr(self, "geographic_results"):
            raise ValueError("Run run_geographic_analysis first")
        
        save_path = None
        if save and hasattr(self, "res_folder") and self.res_folder:
            save_path = os.path.join(self.res_folder, f"geo_{plot_type}")
        
        result = self.geographic_results
        
        plot_functions = {
            "countries": plot_country_output,
            "network": plot_collaboration_network,
            "heatmap": plot_collaboration_heatmap,
            "regions": plot_regional_summary,
            "trends": plot_geographic_trends,
            "map": plot_world_map,
        }
        
        if plot_type not in plot_functions:
            raise ValueError(f"Unknown plot_type: {plot_type}")
        
        return plot_functions[plot_type](result, save_path=save_path, **kwargs)
    
    cls.run_geographic_analysis = run_geographic_analysis_method
    cls.plot_geographic = plot_geographic_method
    
    return cls
# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import biblium as bb
    ba = bb.BiblioAnalysis(f_name="data\\open alex dataset.csv", db="oa")
    
    print("Geographic Analysis")
    print("=" * 60)
    
    # Run analysis
    result = run_geographic_analysis(
        df=ba.df,
        affiliation_col="Affiliations",
        citations_col="Cited by",
        year_col="Year",
        keywords_col="Author Keywords",
        min_collaborations=2,
        verbose=True)
    
    # Print summary
    print(f"\nFound {len(result.country_metrics)} countries")
    print("\nTop 5 Countries by Output:")
    print(result.country_df[["Country", "Papers", "Citations"]].head().to_string(index=False))
    
    # Visualizations
    print("\nGenerating plots...")
    plot_country_output(result, save_path="results/country_output")
    plot_collaboration_network(result, save_path="results/collaboration_network")
    plot_collaboration_heatmap(result, save_path="results/collaboration_heatmap")
    plot_regional_summary(result, save_path="results/regional_summary")
    plot_geographic_trends(result, save_path="results/geographic_trends")
    
    # Export
    print("\nExporting results...")
    export_geographic_results(result, "results")
    
    print("\nDone!")
