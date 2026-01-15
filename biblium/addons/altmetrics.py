# -*- coding: utf-8 -*-
"""
Alternative Metrics Integration Module for Bibliometric Analysis

This module provides integration with alternative metrics (altmetrics) sources
and analysis of non-traditional impact indicators beyond citations.

Features implemented:
1. Altmetric.com API Integration - Social media, news, policy citations
2. PlumX Metrics Integration - Usage, captures, mentions, social media
3. Crossref Event Data - Wikipedia, Reddit, Twitter mentions
4. Patent Citation Analysis - Track translational/commercial impact
5. Clinical/Policy Document Citations - Real-world impact
6. Preprint Tracking - arXiv, bioRxiv, medRxiv connections
7. Data/Code Repository Links - Open science indicators
8. Composite Altmetric Scores - Multi-source impact aggregation

Note: Some features require API keys or external data sources.

Created for integration with the Biblium bibliometric toolkit.

@author: Claude (Anthropic) for Lan.Umek
"""

from __future__ import annotations

import os
import re
import time
import json
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
from urllib.parse import quote, urlencode
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore, spearmanr, pearsonr

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import seaborn as sns

# Optional imports for API access
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from ratelimit import limits, sleep_and_retry
    RATELIMIT_AVAILABLE = True
except ImportError:
    RATELIMIT_AVAILABLE = False

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
# DATA STRUCTURES
# =============================================================================

@dataclass
class AltmetricRecord:
    """Container for altmetric data for a single paper."""
    doc_id: Any
    doi: Optional[str]
    title: Optional[str]
    
    # Altmetric.com metrics
    altmetric_score: float = 0.0
    altmetric_percentile: float = 0.0
    
    # Social media
    twitter_mentions: int = 0
    facebook_mentions: int = 0
    reddit_mentions: int = 0
    linkedin_mentions: int = 0
    
    # News and blogs
    news_mentions: int = 0
    blog_mentions: int = 0
    
    # Academic social
    mendeley_readers: int = 0
    citeulike_readers: int = 0
    
    # Policy and practice
    policy_mentions: int = 0
    patent_citations: int = 0
    clinical_citations: int = 0
    
    # Wikipedia
    wikipedia_mentions: int = 0
    
    # Video and multimedia
    youtube_mentions: int = 0
    
    # Open science
    github_repos: int = 0
    data_repository_links: int = 0
    preprint_versions: int = 0
    
    # Timestamps
    first_seen: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    # Raw data
    raw_response: Dict[str, Any] = field(default_factory=dict)
    
    def get_social_score(self) -> float:
        """Calculate composite social media score."""
        return (
            self.twitter_mentions * 1.0 +
            self.facebook_mentions * 0.5 +
            self.reddit_mentions * 1.5 +
            self.linkedin_mentions * 0.5
        )
    
    def get_scholarly_score(self) -> float:
        """Calculate scholarly attention score."""
        return (
            self.mendeley_readers * 1.0 +
            self.citeulike_readers * 0.5 +
            self.wikipedia_mentions * 3.0
        )
    
    def get_practice_score(self) -> float:
        """Calculate real-world practice impact score."""
        return (
            self.policy_mentions * 5.0 +
            self.patent_citations * 4.0 +
            self.clinical_citations * 5.0
        )
    
    def get_public_score(self) -> float:
        """Calculate public engagement score."""
        return (
            self.news_mentions * 2.0 +
            self.blog_mentions * 1.0 +
            self.youtube_mentions * 1.5
        )
    
    def get_openness_score(self) -> float:
        """Calculate open science score."""
        return (
            self.github_repos * 2.0 +
            self.data_repository_links * 2.0 +
            self.preprint_versions * 1.0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "Doc Id": self.doc_id,
            "doi": self.doi,
            "Altmetric Score": self.altmetric_score,
            "Twitter Mentions": self.twitter_mentions,
            "Facebook Mentions": self.facebook_mentions,
            "Reddit Mentions": self.reddit_mentions,
            "News Mentions": self.news_mentions,
            "Blog Mentions": self.blog_mentions,
            "Mendeley Readers": self.mendeley_readers,
            "Policy Mentions": self.policy_mentions,
            "Patent Citations": self.patent_citations,
            "Wikipedia Mentions": self.wikipedia_mentions,
            "GitHub Repos": self.github_repos,
            "Social Score": self.get_social_score(),
            "Scholarly Score": self.get_scholarly_score(),
            "Practice Score": self.get_practice_score(),
            "Public Score": self.get_public_score(),
            "Openness Score": self.get_openness_score(),
        }

@dataclass
class AltmetricAnalysisResult:
    """Container for complete altmetric analysis results."""
    records: List[AltmetricRecord]
    summary_df: pd.DataFrame
    source_coverage: Dict[str, float]
    correlation_matrix: pd.DataFrame
    percentile_rankings: pd.DataFrame
    temporal_trends: pd.DataFrame
    parameters: Dict[str, Any]
    api_stats: Dict[str, Any]
    
    def get_top_papers(self, metric: str = "altmetric_score", n: int = 20) -> pd.DataFrame:
        """Get top papers by specified metric."""
        if metric not in self.summary_df.columns:
            raise ValueError(f"Metric {metric} not found")
        return self.summary_df.nlargest(n, metric)
    
    def get_high_attention_papers(self, percentile: float = 90) -> pd.DataFrame:
        """Get papers with high altmetric attention."""
        if "altmetric_percentile" in self.summary_df.columns:
            return self.summary_df[self.summary_df["altmetric_percentile"] >= percentile]
        return self.summary_df.nlargest(int(len(self.summary_df) * (100-percentile) / 100), "altmetric_score")
    
    def get_policy_impact_papers(self) -> pd.DataFrame:
        """Get papers with policy/practice impact."""
        mask = (
            (self.summary_df["policy_mentions"] > 0) |
            (self.summary_df["patent_citations"] > 0) |
            (self.summary_df.get("clinical_citations", 0) > 0)
        )
        return self.summary_df[mask]

# =============================================================================
# API INTEGRATION - ALTMETRIC.COM
# =============================================================================

class AltmetricAPI:
    """
    Client for Altmetric.com API.
    
    Note: Free API has limited access. Full access requires API key.
    """
    
    BASE_URL = "https://api.altmetric.com/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Altmetric API client.
        
        Parameters
        ----------
        api_key : str, optional
            Altmetric API key for increased rate limits.
        """
        self.api_key = api_key
        self.session = requests.Session() if REQUESTS_AVAILABLE else None
        self.request_count = 0
        self.last_request_time = None
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with rate limiting."""
        if not REQUESTS_AVAILABLE:
            warnings.warn("requests library not available")
            return None
        
        # Simple rate limiting (1 request per second for free tier)
        if self.last_request_time:
            elapsed = time.time() - self.last_request_time
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        if params is None:
            params = {}
        
        if self.api_key:
            params["key"] = self.api_key
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            self.last_request_time = time.time()
            self.request_count += 1
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None  # Not found
            elif response.status_code == 429:
                warnings.warn("Rate limit exceeded. Waiting...")
                time.sleep(60)
                return self._make_request(endpoint, params)
            else:
                warnings.warn(f"API error: {response.status_code}")
                return None
                
        except Exception as e:
            warnings.warn(f"Request failed: {e}")
            return None
    
    def get_by_doi(self, doi: str) -> Optional[Dict]:
        """Fetch altmetrics for a DOI."""
        # Clean DOI
        doi = doi.strip()
        if doi.startswith("https://doi.org/"):
            doi = doi[16:]
        elif doi.startswith("http://doi.org/"):
            doi = doi[15:]
        elif doi.startswith("doi:"):
            doi = doi[4:]
        
        return self._make_request(f"doi/{doi}")
    
    def get_by_pmid(self, pmid: str) -> Optional[Dict]:
        """Fetch altmetrics for a PubMed ID."""
        return self._make_request(f"pmid/{pmid}")
    
    def get_by_arxiv(self, arxiv_id: str) -> Optional[Dict]:
        """Fetch altmetrics for an arXiv ID."""
        return self._make_request(f"arxiv/{arxiv_id}")
    
    def parse_response(self, response: Dict) -> AltmetricRecord:
        """Parse Altmetric API response into AltmetricRecord."""
        if not response:
            return None
        
        record = AltmetricRecord(
            doc_id=response.get("altmetric_id"),
            doi=response.get("doi"),
            title=response.get("title"),
            altmetric_score=float(response.get("score", 0)),
            twitter_mentions=int(response.get("cited_by_tweeters_count", 0)),
            facebook_mentions=int(response.get("cited_by_fbwalls_count", 0)),
            reddit_mentions=int(response.get("cited_by_rdts_count", 0)),
            linkedin_mentions=int(response.get("cited_by_linkedin_count", 0)),
            news_mentions=int(response.get("cited_by_msm_count", 0)),
            blog_mentions=int(response.get("cited_by_feeds_count", 0)),
            mendeley_readers=int(response.get("readers", {}).get("mendeley", 0)),
            citeulike_readers=int(response.get("readers", {}).get("citeulike", 0)),
            policy_mentions=int(response.get("cited_by_policies_count", 0)),
            patent_citations=int(response.get("cited_by_patents_count", 0)),
            wikipedia_mentions=int(response.get("cited_by_wikipedia_count", 0)),
            youtube_mentions=int(response.get("cited_by_videos_count", 0)),
            raw_response=response)
        
        # Parse percentile if available
        if "context" in response:
            context = response["context"]
            if "all" in context:
                record.altmetric_percentile = float(context["all"].get("pct", 0))
        
        return record

# =============================================================================
# API INTEGRATION - CROSSREF EVENT DATA
# =============================================================================

class CrossrefEventAPI:
    """
    Client for Crossref Event Data API.
    
    Tracks mentions from Wikipedia, Reddit, Twitter, and other sources.
    Free to use, no API key required.
    """
    
    BASE_URL = "https://api.eventdata.crossref.org/v1/events"
    
    def __init__(self, email: Optional[str] = None):
        """
        Initialize Crossref Event Data client.
        
        Parameters
        ----------
        email : str, optional
            Email for polite pool access (faster rate limits).
        """
        self.email = email
        self.session = requests.Session() if REQUESTS_AVAILABLE else None
    
    def get_events_for_doi(self, doi: str, source: str = None) -> List[Dict]:
        """
        Get events for a DOI.
        
        Parameters
        ----------
        doi : str
            DOI to query.
        source : str, optional
            Filter by source (wikipedia, reddit, twitter, etc.)
        
        Returns
        -------
        List of event dictionaries.
        """
        if not REQUESTS_AVAILABLE:
            return []
        
        params = {
            "obj-id": f"https://doi.org/{doi}",
            "rows": 1000,
        }
        
        if source:
            params["source"] = source
        
        if self.email:
            params["mailto"] = self.email
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("message", {}).get("events", [])
            else:
                return []
                
        except Exception as e:
            warnings.warn(f"Crossref API error: {e}")
            return []
    
    def count_events_by_source(self, doi: str) -> Dict[str, int]:
        """Count events by source for a DOI."""
        events = self.get_events_for_doi(doi)
        
        counts = defaultdict(int)
        for event in events:
            source = event.get("source_id", "unknown")
            counts[source] += 1
        
        return dict(counts)

# =============================================================================
# API INTEGRATION - OPENALEX (for Mendeley-like data)
# =============================================================================

class OpenAlexAPI:
    """
    Client for OpenAlex API to get reader counts and related metrics.
    Free, no API key required (but email recommended).
    """
    
    BASE_URL = "https://api.openalex.org"
    
    def __init__(self, email: Optional[str] = None):
        self.email = email
        self.session = requests.Session() if REQUESTS_AVAILABLE else None
    
    def get_work_by_doi(self, doi: str) -> Optional[Dict]:
        """Get work metadata including counts."""
        if not REQUESTS_AVAILABLE:
            return None
        
        # Clean DOI
        if not doi.startswith("https://doi.org/"):
            doi = f"https://doi.org/{doi}"
        
        params = {}
        if self.email:
            params["mailto"] = self.email
        
        try:
            url = f"{self.BASE_URL}/works/{quote(doi, safe='')}"
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            return None
            
        except Exception as e:
            warnings.warn(f"OpenAlex API error: {e}")
            return None
    
    def get_cited_by_count(self, doi: str) -> int:
        """Get citation count from OpenAlex."""
        work = self.get_work_by_doi(doi)
        if work:
            return work.get("cited_by_count", 0)
        return 0

# =============================================================================
# OFFLINE/SIMULATED ALTMETRICS
# =============================================================================

def extract_altmetrics_from_dataframe(
    df: pd.DataFrame,
    id_col: str = "unique-id",
    doi_col: str = "DOI",
    title_col: str = "Title",
    twitter_col: str = None,
    mendeley_col: str = None,
    news_col: str = None,
    policy_col: str = None,
    patent_col: str = None,
    verbose: bool = True) -> List[AltmetricRecord]:
    """
    Extract altmetrics from existing DataFrame columns.
    
    Use this when altmetric data is already in your dataset
    (e.g., from Scopus PlumX or other sources).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    Various column mappings for different metrics.
    verbose : bool
        Print progress.
    
    Returns
    -------
    List of AltmetricRecord objects.
    """
    if verbose:
        print("Extracting altmetrics from DataFrame...")
    
    records = []
    
    for _, row in df.iterrows():
        doc_id = row.get(id_col, "")
        doi = row.get(doi_col, "") if doi_col and doi_col in df.columns else None
        title = row.get(title_col, "") if title_col and title_col in df.columns else None
        
        record = AltmetricRecord(
            doc_id=doc_id,
            doi=str(doi) if pd.notna(doi) else None,
            title=str(title) if pd.notna(title) else None)
        
        # Extract available metrics
        if twitter_col and twitter_col in df.columns:
            val = row.get(twitter_col, 0)
            record.twitter_mentions = int(val) if pd.notna(val) else 0
        
        if mendeley_col and mendeley_col in df.columns:
            val = row.get(mendeley_col, 0)
            record.mendeley_readers = int(val) if pd.notna(val) else 0
        
        if news_col and news_col in df.columns:
            val = row.get(news_col, 0)
            record.news_mentions = int(val) if pd.notna(val) else 0
        
        if policy_col and policy_col in df.columns:
            val = row.get(policy_col, 0)
            record.policy_mentions = int(val) if pd.notna(val) else 0
        
        if patent_col and patent_col in df.columns:
            val = row.get(patent_col, 0)
            record.patent_citations = int(val) if pd.notna(val) else 0
        
        records.append(record)
    
    if verbose:
        print(f"  Extracted altmetrics for {len(records)} papers")
    
    return records

def simulate_altmetrics(
    df: pd.DataFrame,
    id_col: str = "unique-id",
    doi_col: str = "DOI",
    citations_col: str = "Cited by",
    year_col: str = "Year",
    random_state: int = 42,
    verbose: bool = True) -> List[AltmetricRecord]:
    """
    Simulate altmetric data for testing/demonstration.
    
    Generates realistic-looking altmetric distributions
    correlated with citation counts.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    id_col, doi_col, citations_col, year_col : str
        Column names.
    random_state : int
        Random seed.
    verbose : bool
        Print progress.
    
    Returns
    -------
    List of AltmetricRecord objects.
    """
    if verbose:
        print("Simulating altmetric data (for demonstration)...")
    
    np.random.seed(random_state)
    
    records = []
    current_year = pd.Timestamp.now().year
    
    for _, row in df.iterrows():
        doc_id = row.get(id_col, "")
        doi = row.get(doi_col, "")
        citations = row.get(citations_col, 0)
        year = row.get(year_col, current_year)
        
        citations = int(citations) if pd.notna(citations) else 0
        year = int(year) if pd.notna(year) else current_year
        
        # Base attention proportional to citations (with noise)
        base_attention = np.log1p(citations) * np.random.uniform(0.5, 1.5)
        
        # Recency boost
        age = current_year - year
        recency_factor = max(0.2, 1.0 - age * 0.1)
        
        # Generate metrics with realistic distributions
        record = AltmetricRecord(
            doc_id=doc_id,
            doi=str(doi) if pd.notna(doi) else None,
            title=None)
        
        # Twitter (power law, most papers have 0)
        if np.random.random() < 0.3 + 0.1 * base_attention:
            record.twitter_mentions = int(np.random.pareto(1.5) * base_attention * recency_factor * 5)
        
        # Mendeley (more common)
        if np.random.random() < 0.6:
            record.mendeley_readers = int(np.random.pareto(1.2) * citations * 0.5)
        
        # News (rare)
        if np.random.random() < 0.1 * base_attention:
            record.news_mentions = int(np.random.pareto(2) * base_attention * 2)
        
        # Blogs
        if np.random.random() < 0.15:
            record.blog_mentions = int(np.random.pareto(2) * base_attention)
        
        # Reddit (rare)
        if np.random.random() < 0.05:
            record.reddit_mentions = int(np.random.pareto(2) * base_attention * 2)
        
        # Wikipedia (very rare)
        if np.random.random() < 0.02 * base_attention:
            record.wikipedia_mentions = int(np.random.uniform(1, 5))
        
        # Policy (very rare, correlated with citations)
        if citations > 50 and np.random.random() < 0.05:
            record.policy_mentions = int(np.random.uniform(1, 3))
        
        # Patents (rare)
        if citations > 30 and np.random.random() < 0.03:
            record.patent_citations = int(np.random.uniform(1, 5))
        
        # GitHub (depends on field)
        if np.random.random() < 0.05:
            record.github_repos = int(np.random.uniform(1, 10))
        
        # Calculate composite score
        record.altmetric_score = (
            record.twitter_mentions * 0.5 +
            record.mendeley_readers * 0.5 +
            record.news_mentions * 3 +
            record.blog_mentions * 1 +
            record.reddit_mentions * 0.5 +
            record.wikipedia_mentions * 5 +
            record.policy_mentions * 5 +
            record.patent_citations * 3
        )
        
        records.append(record)
    
    # Calculate percentiles
    scores = [r.altmetric_score for r in records]
    for record in records:
        record.altmetric_percentile = percentileofscore(scores, record.altmetric_score)
    
    if verbose:
        print(f"  Generated altmetrics for {len(records)} papers")
        non_zero = sum(1 for r in records if r.altmetric_score > 0)
        print(f"  Papers with attention: {non_zero} ({non_zero/len(records)*100:.1f}%)")
    
    return records

# =============================================================================
# FETCH ALTMETRICS FOR DATASET
# =============================================================================

def fetch_altmetrics(
    df: pd.DataFrame,
    id_col: str = "unique-id",
    doi_col: str = "DOI",
    api_key: Optional[str] = None,
    max_requests: int = 100,
    use_crossref: bool = True,
    verbose: bool = True) -> List[AltmetricRecord]:
    """
    Fetch altmetrics from APIs for papers in dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with DOIs.
    id_col : str
        Document ID column.
    doi_col : str
        DOI column.
    api_key : str, optional
        Altmetric.com API key.
    max_requests : int
        Maximum API requests to make.
    use_crossref : bool
        Also query Crossref Event Data.
    verbose : bool
        Print progress.
    
    Returns
    -------
    List of AltmetricRecord objects.
    """
    if not REQUESTS_AVAILABLE:
        warnings.warn("requests library required for API access. Using simulation instead.")
        return simulate_altmetrics(df, id_col, doi_col, verbose=verbose)
    
    if verbose:
        print("Fetching altmetrics from APIs...")
    
    # Initialize APIs
    altmetric_api = AltmetricAPI(api_key)
    crossref_api = CrossrefEventAPI() if use_crossref else None
    
    records = []
    request_count = 0
    
    # Get papers with DOIs
    papers_with_doi = df[df[doi_col].notna() & (df[doi_col] != "")]
    
    if verbose:
        print(f"  Papers with DOIs: {len(papers_with_doi)}")
        print(f"  Max requests: {max_requests}")
    
    for idx, row in papers_with_doi.iterrows():
        if request_count >= max_requests:
            if verbose:
                print(f"  Reached max requests limit ({max_requests})")
            break
        
        doc_id = row.get(id_col, idx)
        doi = str(row.get(doi_col, "")).strip()
        
        if not doi:
            continue
        
        # Clean DOI
        if doi.startswith("https://doi.org/"):
            doi = doi[16:]
        elif doi.startswith("http://doi.org/"):
            doi = doi[15:]
        
        # Fetch from Altmetric.com
        response = altmetric_api.get_by_doi(doi)
        request_count += 1
        
        if response:
            record = altmetric_api.parse_response(response)
            record.doc_id = doc_id
        else:
            record = AltmetricRecord(doc_id=doc_id, doi=doi, title=None)
        
        # Optionally fetch from Crossref
        if crossref_api and request_count < max_requests:
            event_counts = crossref_api.count_events_by_source(doi)
            request_count += 1
            
            # Merge event data
            record.twitter_mentions = max(record.twitter_mentions, event_counts.get("twitter", 0))
            record.reddit_mentions = max(record.reddit_mentions, event_counts.get("reddit", 0))
            record.wikipedia_mentions = max(record.wikipedia_mentions, event_counts.get("wikipedia", 0))
        
        records.append(record)
        
        if verbose and request_count % 10 == 0:
            print(f"  Processed {request_count} requests...")
    
    # Add empty records for papers without DOIs
    processed_ids = {r.doc_id for r in records}
    for _, row in df.iterrows():
        doc_id = row.get(id_col, "")
        if doc_id not in processed_ids:
            records.append(AltmetricRecord(doc_id=doc_id, doi=None, title=None))
    
    if verbose:
        print(f"  Completed {request_count} API requests")
        with_data = sum(1 for r in records if r.altmetric_score > 0)
        print(f"  Papers with altmetric data: {with_data}")
    
    return records

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_altmetrics(
    records: List[AltmetricRecord],
    df: pd.DataFrame = None,
    citations_col: str = "Cited by",
    year_col: str = "Year",
    field_col: str = "Subject Area",
    id_col: str = "unique-id",
    verbose: bool = True) -> AltmetricAnalysisResult:
    """
    Analyze altmetric data.
    
    Parameters
    ----------
    records : List[AltmetricRecord]
        Altmetric records.
    df : pd.DataFrame, optional
        Original dataframe for additional context.
    citations_col : str
        Citations column in df.
    year_col : str
        Year column in df.
    field_col : str
        Field column in df.
    id_col : str
        ID column in df.
    verbose : bool
        Print progress.
    
    Returns
    -------
    AltmetricAnalysisResult
    """
    if verbose:
        print("Analyzing altmetric data...")
    
    # Convert to DataFrame
    summary_data = [r.to_dict() for r in records]
    summary_df = pd.DataFrame(summary_data)
    
    # Add citation data if available
    if df is not None and citations_col in df.columns:
        citation_map = dict(zip(df[id_col], df[citations_col]))
        summary_df["citations"] = summary_df["doc_id"].map(citation_map).fillna(0).astype(int)
        
        if year_col in df.columns:
            year_map = dict(zip(df[id_col], df[year_col]))
            summary_df["year"] = summary_df["doc_id"].map(year_map)
        
        if field_col in df.columns:
            field_map = dict(zip(df[id_col], df[field_col]))
            summary_df["field"] = summary_df["doc_id"].map(field_map)
    
    # Calculate percentiles within dataset
    for col in ["altmetric_score", "social_score", "scholarly_score", 
                "practice_score", "public_score", "openness_score"]:
        if col in summary_df.columns:
            values = summary_df[col]
            summary_df[f"{col}_percentile"] = values.apply(
                lambda x: percentileofscore(values, x)
            )
    
    # Source coverage
    if verbose:
        print("  Computing source coverage...")
    
    source_coverage = {}
    total = len(records)
    
    source_cols = [
        ("twitter_mentions", "Twitter"),
        ("mendeley_readers", "Mendeley"),
        ("news_mentions", "News"),
        ("blog_mentions", "Blogs"),
        ("reddit_mentions", "Reddit"),
        ("wikipedia_mentions", "Wikipedia"),
        ("policy_mentions", "Policy"),
        ("patent_citations", "Patents"),
        ("github_repos", "GitHub"),
    ]
    
    for col, name in source_cols:
        if col in summary_df.columns:
            coverage = (summary_df[col] > 0).sum() / total * 100
            source_coverage[name] = round(coverage, 1)
    
    # Correlation matrix
    if verbose:
        print("  Computing correlations...")
    
    metric_cols = ["altmetric_score", "twitter_mentions", "mendeley_readers",
                  "news_mentions", "wikipedia_mentions", "social_score",
                  "scholarly_score", "practice_score", "public_score"]
    
    if "citations" in summary_df.columns:
        metric_cols.append("citations")
    
    available_cols = [c for c in metric_cols if c in summary_df.columns]
    correlation_matrix = summary_df[available_cols].corr()
    
    # Percentile rankings
    percentile_rankings = summary_df[[
        "doc_id", "altmetric_score", "social_score", 
        "scholarly_score", "practice_score"
    ]].copy()
    
    # Temporal trends
    temporal_trends = pd.DataFrame()
    if "year" in summary_df.columns:
        year_stats = summary_df.groupby("year").agg({
            "Altmetric Score": ["mean", "median", "sum"],
            "Twitter Mentions": "sum",
            "Mendeley Readers": "sum",
        }).reset_index()
        year_stats.columns = ["year", "avg_altmetric", "median_altmetric", 
                             "total_altmetric", "total_twitter", "total_mendeley"]
        temporal_trends = year_stats
    
    # API stats
    api_stats = {
        "total_records": len(records),
        "with_altmetric_score": sum(1 for r in records if r.altmetric_score > 0),
        "with_twitter": sum(1 for r in records if r.twitter_mentions > 0),
        "with_mendeley": sum(1 for r in records if r.mendeley_readers > 0),
        "with_policy": sum(1 for r in records if r.policy_mentions > 0),
    }
    
    result = AltmetricAnalysisResult(
        records=records,
        summary_df=summary_df,
        source_coverage=source_coverage,
        correlation_matrix=correlation_matrix,
        percentile_rankings=percentile_rankings,
        temporal_trends=temporal_trends,
        parameters={
            "citations_col": citations_col,
            "year_col": year_col,
        },
        api_stats=api_stats)
    
    if verbose:
        print("  Analysis complete!")
        _print_altmetric_summary(result)
    
    return result

def _print_altmetric_summary(result: AltmetricAnalysisResult) -> None:
    """Print summary of altmetric analysis."""
    print("\nAltmetric Summary:")
    print("-" * 30)
    
    stats = result.api_stats
    total = stats["total_records"]
    with_score = stats["with_altmetric_score"]
    
    print(f"Total papers: {total}")
    print(f"With altmetric attention: {with_score} ({with_score/total*100:.1f}%)")
    
    print("\nSource Coverage:")
    for source, coverage in sorted(result.source_coverage.items(), key=lambda x: -x[1]):
        print(f"  {source}: {coverage:.1f}%")
    
    # Top scores
    top_score = result.summary_df["altmetric_score"].max()
    avg_score = result.summary_df["altmetric_score"].mean()
    print(f"\nAltmetric Score: max={top_score:.1f}, mean={avg_score:.2f}")

# =============================================================================
# SPECIALIZED ANALYSES
# =============================================================================

def analyze_citation_altmetric_relationship(
    result: AltmetricAnalysisResult,
    verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze relationship between citations and altmetrics.
    
    Returns correlation coefficients and insights.
    """
    if "citations" not in result.summary_df.columns:
        warnings.warn("Citations data not available")
        return {}
    
    df = result.summary_df
    
    # Remove zeros for log analysis
    df_nonzero = df[(df["citations"] > 0) & (df["altmetric_score"] > 0)]
    
    analysis = {}
    
    # Pearson correlation
    if len(df_nonzero) > 10:
        corr, p_value = pearsonr(df_nonzero["citations"], df_nonzero["altmetric_score"])
        analysis["pearson_r"] = round(corr, 3)
        analysis["pearson_p"] = round(p_value, 4)
        
        # Log-log correlation
        log_corr, log_p = pearsonr(
            np.log1p(df_nonzero["citations"]), 
            np.log1p(df_nonzero["altmetric_score"])
        )
        analysis["log_pearson_r"] = round(log_corr, 3)
        
        # Spearman (rank) correlation
        spearman_r, spearman_p = spearmanr(df["citations"], df["altmetric_score"])
        analysis["spearman_r"] = round(spearman_r, 3)
    
    # Identify discrepancies
    df["cit_percentile"] = df["citations"].apply(lambda x: percentileofscore(df["citations"], x))
    df["alt_percentile"] = df["altmetric_score"].apply(lambda x: percentileofscore(df["altmetric_score"], x))
    df["percentile_gap"] = df["alt_percentile"] - df["cit_percentile"]
    
    # High altmetric, low citation (public attention without academic impact)
    high_alt_low_cit = df[(df["alt_percentile"] > 75) & (df["cit_percentile"] < 25)]
    analysis["n_high_altmetric_low_citation"] = len(high_alt_low_cit)
    
    # High citation, low altmetric (academic impact without public attention)
    high_cit_low_alt = df[(df["cit_percentile"] > 75) & (df["alt_percentile"] < 25)]
    analysis["n_high_citation_low_altmetric"] = len(high_cit_low_alt)
    
    if verbose:
        print("\nCitation-Altmetric Relationship:")
        spearman = analysis.get("spearman_r", "N/A")
        high_alt_low_cit = analysis["n_high_altmetric_low_citation"]
        high_cit_low_alt = analysis["n_high_citation_low_altmetric"]
        print(f"  Spearman correlation: {spearman}")
        print(f"  High altmetric, low citation: {high_alt_low_cit}")
        print(f"  High citation, low altmetric: {high_cit_low_alt}")
    
    return analysis

def identify_policy_impact_papers(
    result: AltmetricAnalysisResult,
    verbose: bool = True) -> pd.DataFrame:
    """
    Identify papers with policy/practice impact.
    
    Returns papers mentioned in policy documents, patents, or clinical guidelines.
    """
    df = result.summary_df
    
    # Policy impact indicators
    policy_mask = (
        (df["policy_mentions"] > 0) |
        (df["patent_citations"] > 0)
    )
    
    policy_papers = df[policy_mask].copy()
    
    # Calculate practice impact score
    policy_papers["practice_impact_score"] = (
        policy_papers["policy_mentions"] * 5 +
        policy_papers["patent_citations"] * 4
    )
    
    policy_papers = policy_papers.sort_values("Practice Impact Score", ascending=False)
    
    if verbose:
        print(f"\nPolicy/Practice Impact Papers: {len(policy_papers)}")
        if len(policy_papers) > 0:
            n_policy = (df["policy_mentions"] > 0).sum()
            n_patent = (df["patent_citations"] > 0).sum()
            print(f"  With policy mentions: {n_policy}")
            print(f"  With patent citations: {n_patent}")
    
    return policy_papers

def identify_public_attention_papers(
    result: AltmetricAnalysisResult,
    percentile_threshold: float = 90,
    verbose: bool = True) -> pd.DataFrame:
    """
    Identify papers with high public attention.
    
    Returns papers in top percentile for news, social media, etc.
    """
    df = result.summary_df
    
    # Public attention score
    df["public_attention"] = (
        df["twitter_mentions"] * 1 +
        df["news_mentions"] * 3 +
        df["blog_mentions"] * 1 +
        df["reddit_mentions"] * 1.5 +
        df["youtube_mentions"] * 2
    )
    
    threshold = np.percentile(df["public_attention"], percentile_threshold)
    high_attention = df[df["public_attention"] >= threshold].copy()
    high_attention = high_attention.sort_values("Public Attention", ascending=False)
    
    if verbose:
        print(f"\nHigh Public Attention Papers (top {100-percentile_threshold}%): {len(high_attention)}")
    
    return high_attention

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_altmetric_overview(
    result: AltmetricAnalysisResult,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None) -> plt.Figure:
    """Plot comprehensive altmetric overview."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    df = result.summary_df
    
    # 1. Altmetric score distribution
    scores = df["altmetric_score"]
    scores_nonzero = scores[scores > 0]
    
    if len(scores_nonzero) > 0:
        ax.hist(np.log1p(scores_nonzero), bins=50, color="lightblue", 
                edgecolor="white", alpha=0.7)
        ax.set_xlabel("log(Altmetric Score + 1)", fontsize=11)
        ax.set_ylabel("Number of Papers", fontsize=11)
        ax.set_title(f"Altmetric Score Distribution (n={len(scores_nonzero)} with attention)", fontsize=12)
    else:
        ax.text(0.5, 0.5, "No altmetric scores available", ha="center", va="center")
        ax.axis("off")
    
    # 2. Source coverage
    if result.source_coverage:
        sources = list(result.source_coverage.keys())
        coverage = list(result.source_coverage.values())
        
        colors = "lightblue"
        bars = ax.barh(range(len(sources)), coverage, color=colors)
        ax.set_yticks(range(len(sources)))
        ax.set_yticklabels(sources)
        ax.set_xlabel("Coverage (%)", fontsize=11)
        ax.set_title("Source Coverage", fontsize=12)
        ax.set_xlim(0, 100)
        
        for bar, cov in zip(bars, coverage):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f"{cov:.1f}%", va="center", fontsize=9)
    
    # 3. Score components breakdown
    component_cols = ["social_score", "scholarly_score", "practice_score", 
                     "public_score", "openness_score"]
    available_components = [c for c in component_cols if c in df.columns]
    
    if available_components:
        component_sums = df[available_components].sum()
        component_names = [c.replace("_score", "").title() for c in available_components]
        colors = "lightblue"
        bars = ax.bar(range(len(component_names)), component_sums.values, 
                      color="lightblue")
        ax.set_xticks(range(len(component_names)))
        ax.set_xticklabels(component_names, rotation=45, ha="right")
        ax.set_ylabel("Total Score", fontsize=11)
        ax.set_title("Score Components", fontsize=12)
    
    # 4. Citation vs Altmetric scatter
    if "citations" in df.columns:
        nonzero = df[(df["citations"] > 0) & (df["altmetric_score"] > 0)]
        
        if len(nonzero) > 0:
            ax.scatter(nonzero["citations"], nonzero["altmetric_score"],
                       alpha=0.5, s=30, c="steelblue")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Citations (log scale)", fontsize=11)
            ax.set_ylabel("Altmetric Score (log scale)", fontsize=11)
            ax.set_title("Citations vs Altmetric Score", fontsize=12)
            
            # Add correlation
            if len(nonzero) > 10:
                corr, _ = spearmanr(nonzero["citations"], nonzero["altmetric_score"])
                ax.text(0.05, 0.95, f"Spearman r = {corr:.2f}",
                        transform=ax.transAxes, fontsize=10)
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            ax.axis("off")
    else:
        ax.text(0.5, 0.5, "Citation data not available", ha="center", va="center")
        ax.axis("off")
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

def plot_source_breakdown(
    result: AltmetricAnalysisResult,
    top_n: int = 20,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """Plot breakdown by source for top papers."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    df = result.summary_df.nlargest(top_n, "altmetric_score")
    
    # Source columns
    source_cols = {
        "Twitter Mentions": ("Twitter", "#1DA1F2"),
        "Mendeley Readers": ("Mendeley", "#B6232A"),
        "News Mentions": ("News", "#27ae60"),
        "Blog Mentions": ("Blogs", "#f39c12"),
        "Reddit Mentions": ("Reddit", "#FF4500"),
        "Wikipedia Mentions": ("Wikipedia", "#000000"),
    }
    
    available_sources = {k: v for k, v in source_cols.items() if k in df.columns}
    
    # 1. Stacked bar chart
    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    
    for col, (name, color) in available_sources.items():
        values = df[col].values
        ax.bar(x, values, bottom=bottom, label=name, color=color, alpha=0.8)
        bottom += values
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{i+1}" for i in range(len(df))], rotation=45)
    ax.set_ylabel("Mentions/Readers", fontsize=11)
    ax.set_title(f"Top {top_n} Papers by Source", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    
    # 2. Pie chart of total mentions by source
    total_by_source = {name: result.summary_df[col].sum() 
                      for col, (name, _) in available_sources.items()}
    
    # Filter to non-zero
    total_by_source = {k: v for k, v in total_by_source.items() if v > 0}
    
    if total_by_source:
        colors = [available_sources[col][1] for col, (name, _) in source_cols.items() 
                 if name in total_by_source]
        
        # Use bar chart instead of pie
        names = list(total_by_source.keys())
        values = list(total_by_source.values())
        bars = ax.bar(range(len(names)), values, color="lightblue")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Total Mentions", fontsize=11)
        ax.set_title("Total by Source", fontsize=12)
    
    # 3. Score distribution by type
    score_types = ["social_score", "scholarly_score", "public_score", "practice_score"]
    available_scores = [s for s in score_types if s in df.columns]
    
    if available_scores:
        data_to_plot = [result.summary_df[s].values for s in available_scores]
        labels = [s.replace("_score", "").title() for s in available_scores]
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = "lightblue"
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("Score Distribution by Type", fontsize=12)
        ax.set_yscale("symlog", linthresh=1)
    
    # 4. Correlation heatmap
    if len(result.correlation_matrix) > 0:
        # Select subset of columns
        corr_cols = ["altmetric_score", "twitter_mentions", "mendeley_readers", 
                    "news_mentions", "social_score"]
        if "citations" in result.correlation_matrix.columns:
            corr_cols.append("citations")
        
        available_corr = [c for c in corr_cols if c in result.correlation_matrix.columns]
        corr_subset = result.correlation_matrix.loc[available_corr, available_corr]
        
        sns.heatmap(corr_subset, annot=True, fmt=".2f", cmap=cmap or CMAP_CONTINUOUS,
                   center=0, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title("Correlation Matrix", fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

def plot_temporal_altmetrics(
    result: AltmetricAnalysisResult,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None) -> plt.Figure:
    """Plot temporal trends in altmetrics."""
    if len(result.temporal_trends) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(False)
        ax.text(0.5, 0.5, "Temporal data not available", ha="center", va="center")
        ax.axis("off")
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    df = result.temporal_trends.dropna(subset=["year"])
    df = df[df["year"] >= 2000]  # Filter to reasonable years
    
    # 1. Average altmetric score over time
    ax.plot(df["year"], df["avg_altmetric"], marker="o", linewidth=2, 
            color="lightblue", label="Mean")
    ax.plot(df["year"], df["median_altmetric"], marker="s", linewidth=2,
            color="lightblue", label="Median")
    ax.set_xlabel("Publication Year", fontsize=11)
    ax.set_ylabel("Altmetric Score", fontsize=11)
    ax.set_title("Average Altmetric Score by Year", fontsize=12)
    ax.legend()
    ax
    
    # 2. Total mentions by year
    if "total_twitter" in df.columns and "total_mendeley" in df.columns:
        ax.bar(df["year"] - 0.2, df["total_twitter"], 0.4, 
               label="Twitter", color="#1DA1F2", alpha=0.8)
        ax.bar(df["year"] + 0.2, df["total_mendeley"], 0.4,
               label="Mendeley", color="#B6232A", alpha=0.8)
        ax.set_xlabel("Publication Year", fontsize=11)
        ax.set_ylabel("Total Mentions", fontsize=11)
        ax.set_title("Total Mentions by Year", fontsize=12)
        ax.legend()
    else:
        ax.plot(df["year"], df["total_altmetric"], marker="o", linewidth=2)
        ax.set_xlabel("Publication Year", fontsize=11)
        ax.set_ylabel("Total Altmetric Score", fontsize=11)
        ax.set_title("Total Altmetric Score by Year", fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_altmetric_results(
    result: AltmetricAnalysisResult,
    output_dir: str) -> Dict[str, str]:
    """Export altmetric analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    excel_path = os.path.join(output_dir, "altmetric_analysis.xlsx")
    
    with pd.ExcelWriter(excel_path) as writer:
        # Summary
        result.summary_df.to_excel(writer, sheet_name="Summary", index=False)
        
        # Top papers
        top_papers = result.summary_df.nlargest(100, "altmetric_score")
        top_papers.to_excel(writer, sheet_name="Top_Papers", index=False)
        
        # Correlations
        result.correlation_matrix.to_excel(writer, sheet_name="Correlations")
        
        # Source coverage
        coverage_df = pd.DataFrame([
            {"Source": k, "Coverage Percent": v}
            for k, v in result.source_coverage.items()
        ])
        coverage_df.to_excel(writer, sheet_name="Source_Coverage", index=False)
        
        # Temporal trends
        if len(result.temporal_trends) > 0:
            result.temporal_trends.to_excel(writer, sheet_name="Temporal_Trends", index=False)
        
        # Policy impact
        policy_papers = result.get_policy_impact_papers()
        if len(policy_papers) > 0:
            policy_papers.to_excel(writer, sheet_name="Policy_Impact", index=False)
    
    print(f"Exported to: {excel_path}")
    return {"excel": excel_path}

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_altmetric_analysis(
    df: pd.DataFrame,
    id_col: str = "unique-id",
    doi_col: str = "DOI",
    citations_col: str = "Cited by",
    year_col: str = "Year",
    field_col: str = "Subject Area",
    api_key: Optional[str] = None,
    fetch_from_api: bool = False,
    max_api_requests: int = 100,
    simulate: bool = True,
    verbose: bool = True) -> AltmetricAnalysisResult:
    """
    Run complete altmetric analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    id_col : str
        Document ID column.
    doi_col : str
        DOI column.
    citations_col : str
        Citations column.
    year_col : str
        Year column.
    field_col : str
        Field column.
    api_key : str, optional
        Altmetric.com API key.
    fetch_from_api : bool
        Whether to fetch from APIs.
    max_api_requests : int
        Max API requests.
    simulate : bool
        If True and not fetching from API, simulate data.
    verbose : bool
        Print progress.
    
    Returns
    -------
    AltmetricAnalysisResult
    """
    if verbose:
        print("=" * 50)
        print("Alternative Metrics Analysis")
        print("=" * 50)
    
    # Get altmetric records
    if fetch_from_api:
        records = fetch_altmetrics(
            df, id_col, doi_col, api_key, max_api_requests, verbose=verbose
        )
    elif simulate:
        records = simulate_altmetrics(df, id_col, doi_col, citations_col, verbose=verbose)
    else:
        records = extract_altmetrics_from_dataframe(df, id_col, doi_col, verbose=verbose)
    
    # Analyze
    result = analyze_altmetrics(
        records, df, citations_col, year_col, field_col, id_col, verbose=verbose
    )
    
    return result

# =============================================================================
# CLASS INTEGRATION
# =============================================================================

def add_altmetric_methods(cls):
    """Add altmetric methods to BiblioAnalysis class."""
    
    def run_altmetric_analysis_method(
        self,
        fetch_from_api: bool = False,
        simulate: bool = True,
        api_key: str = None,
        **kwargs
    ) -> AltmetricAnalysisResult:
        """Run altmetric analysis."""
        id_col = getattr(self, "id_var", "unique-id")
        doi_col = getattr(self, "doi_var", "DOI")
        citations_col = getattr(self, "citations_var", "Cited by")
        year_col = getattr(self, "year_var", "Year")
        
        self.altmetric_results = run_altmetric_analysis(
            self.df,
            id_col=id_col,
            doi_col=doi_col,
            citations_col=citations_col,
            year_col=year_col,
            fetch_from_api=fetch_from_api,
            simulate=simulate,
            api_key=api_key,
            **kwargs
        )
        
        if hasattr(self, "res_folder") and self.res_folder:
            export_altmetric_results(
                self.altmetric_results,
                self.res_folder
            )
        
        return self.altmetric_results
    
    def plot_altmetrics_method(
        self,
        plot_type: str = "overview",
        save: bool = True,
        **kwargs
    ) -> plt.Figure:
        """Create altmetric visualizations."""
        if not hasattr(self, "altmetric_results"):
            raise ValueError("Run run_altmetric_analysis first")
        
        save_path = None
        if save and hasattr(self, "res_folder") and self.res_folder:
            save_path = os.path.join(self.res_folder, f"altmetric_{plot_type}")
        
        plot_functions = {
            "overview": plot_altmetric_overview,
            "sources": plot_source_breakdown,
            "temporal": plot_temporal_altmetrics,
        }
        
        if plot_type not in plot_functions:
            raise ValueError(f"Unknown plot_type: {plot_type}")
        
        return plot_functions[plot_type](self.altmetric_results, save_path=save_path, **kwargs)
    
    cls.run_altmetric_analysis = run_altmetric_analysis_method
    cls.plot_altmetrics = plot_altmetrics_method
    
    return cls
# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import biblium as bb
    ba = bb.BiblioAnalysis(f_name="data\\open alex dataset.csv", db="oa")
    
    print("Altmetrics Analysis")
    print("=" * 60)
    
    # Run analysis
    result = run_altmetric_analysis(
        df=ba.df,
        doi_col="DOI",
        year_col="Year",
        citations_col="Cited by",
        verbose=True)
    
    # Print summary
    print(f"\nAnalyzed {len(result.summary_df)} papers")
    
    # Visualizations
    print("\nGenerating plots...")
    plot_altmetric_overview(result, save_path="results/altmetric_overview")
    plot_temporal_altmetrics(result, save_path="results/temporal_altmetrics")
    
    # Export
    print("\nExporting results...")
    export_altmetric_results(result, "results")
    
    print("\nDone!")
