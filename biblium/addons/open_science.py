# -*- coding: utf-8 -*-
"""
Reproducibility & Open Science Module for Bibliometric Analysis

This module provides tools for analyzing and tracking open science practices
in academic publications, including data sharing, code availability,
preprint usage, and reproducibility indicators.

Features implemented:
1. Open Access Detection - Identify OA status (gold, green, hybrid, bronze)
2. Data Availability Analysis - Track data sharing statements and repositories
3. Code Availability Analysis - Detect code/software sharing (GitHub, GitLab, etc.)
4. Preprint Detection - Track preprint versions (arXiv, bioRxiv, medRxiv, SSRN)
5. Reproducibility Indicators - Methods transparency, protocol registration
6. FAIR Compliance Estimation - Findable, Accessible, Interoperable, Reusable
7. Open Science Score - Composite openness metric
8. Temporal Trends - Track open science adoption over time

Created for integration with the Biblium bibliometric toolkit.

@author: Claude (Anthropic) for Lan.Umek
"""

from __future__ import annotations

import os
import re
import json
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from datetime import datetime
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore, spearmanr, chi2_contingency

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import seaborn as sns

# Optional imports
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

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
# CONSTANTS AND PATTERNS
# =============================================================================

# Data repository patterns
DATA_REPOSITORY_PATTERNS = {
    "figshare": r"figshare\.com|doi\.org/10\.6084/m9\.figshare",
    "zenodo": r"zenodo\.org|doi\.org/10\.5281/zenodo",
    "dryad": r"datadryad\.org|doi\.org/10\.5061/dryad",
    "osf": r"osf\.io",
    "dataverse": r"dataverse\.|doi\.org/10\.7910/DVN",
    "mendeley_data": r"data\.mendeley\.com|doi\.org/10\.17632",
    "github": r"github\.com",
    "gitlab": r"gitlab\.com",
    "bitbucket": r"bitbucket\.org",
    "kaggle": r"kaggle\.com/datasets",
    "ieee_dataport": r"ieee-dataport\.org",
    "pangaea": r"pangaea\.de|doi\.org/10\.1594/PANGAEA",
    "genbank": r"ncbi\.nlm\.nih\.gov/genbank|genbank",
    "geo": r"ncbi\.nlm\.nih\.gov/geo|GSE\d+",
    "arrayexpress": r"ebi\.ac\.uk/arrayexpress|E-[A-Z]+-\d+",
    "proteomexchange": r"proteomecentral|PXD\d+",
    "clinicaltrials": r"clinicaltrials\.gov|NCT\d+",
}

# Code repository patterns
CODE_REPOSITORY_PATTERNS = {
    "github": r"github\.com/[\w-]+/[\w-]+",
    "gitlab": r"gitlab\.com/[\w-]+/[\w-]+",
    "bitbucket": r"bitbucket\.org/[\w-]+/[\w-]+",
    "sourceforge": r"sourceforge\.net/projects/[\w-]+",
    "cran": r"cran\.r-project\.org/package=\w+",
    "pypi": r"pypi\.org/project/[\w-]+",
    "bioconductor": r"bioconductor\.org/packages/\w+",
    "conda": r"anaconda\.org/[\w-]+/[\w-]+",
    "npm": r"npmjs\.com/package/[\w-]+",
    "docker": r"hub\.docker\.com/r/[\w-]+/[\w-]+",
    "code_ocean": r"codeocean\.com",
    "matlab_exchange": r"mathworks\.com/matlabcentral/fileexchange",
}

# Preprint server patterns
PREPRINT_PATTERNS = {
    "arxiv": r"arxiv\.org|arXiv:\d+\.\d+",
    "biorxiv": r"biorxiv\.org|doi\.org/10\.1101/\d{4}\.\d{2}\.\d{2}",
    "medrxiv": r"medrxiv\.org",
    "chemrxiv": r"chemrxiv\.org",
    "ssrn": r"ssrn\.com|papers\.ssrn\.com",
    "preprints_org": r"preprints\.org",
    "researchsquare": r"researchsquare\.com",
    "eartharxiv": r"eartharxiv\.org",
    "psyarxiv": r"psyarxiv\.com",
    "socarxiv": r"socarxiv\.org",
    "osf_preprints": r"osf\.io/preprints",
    "authorea": r"authorea\.com",
    "techrxiv": r"techrxiv\.org",
    "engrxiv": r"engrxiv\.org",
}

# Data availability statement patterns
DATA_STATEMENT_PATTERNS = [
    r"data\s+(are|is)\s+available",
    r"data\s+availability",
    r"available\s+(at|from|upon|on)\s+request",
    r"data\s+sharing",
    r"supplementary\s+(data|material|information)",
    r"supporting\s+information",
    r"openly\s+available",
    r"publicly\s+available",
    r"deposited\s+(in|at|to)",
    r"accessible\s+(at|from|via)",
    r"repository",
    r"can\s+be\s+(accessed|found|obtained)",
]

# Code availability patterns
CODE_STATEMENT_PATTERNS = [
    r"code\s+(is|are)\s+available",
    r"source\s+code",
    r"software\s+(is|are)\s+available",
    r"scripts?\s+(are|is)\s+available",
    r"code\s+availability",
    r"reproducible",
    r"open[\s-]?source",
    r"analysis\s+code",
    r"computational\s+code",
]

# Open access patterns
OA_PATTERNS = {
    "gold": [r"open\s+access", r"gold\s+oa", r"fully\s+open"],
    "green": [r"green\s+oa", r"self[\s-]?archived", r"repository\s+version"],
    "hybrid": [r"hybrid\s+oa", r"author\s+choice"],
    "bronze": [r"free\s+to\s+read", r"bronze\s+oa"],
}

# Methods transparency patterns
METHODS_PATTERNS = [
    r"pre[\s-]?registration",
    r"pre[\s-]?registered",
    r"registered\s+report",
    r"study\s+protocol",
    r"analysis\s+plan",
    r"prospero",
    r"osf\.io/[\w]+",
    r"aspredicted\.org",
    r"materials?\s+(are|is)\s+available",
    r"stimuli\s+(are|is)\s+available",
    r"detailed\s+methods",
    r"supplementary\s+methods",
]

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class OpenScienceRecord:
    """Container for open science indicators for a single paper."""
    doc_id: Any
    doi: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    
    # Open Access
    is_open_access: bool = False
    oa_status: str = "closed"  # gold, green, hybrid, bronze, closed
    oa_url: Optional[str] = None
    
    # Data availability
    has_data_statement: bool = False
    data_available: bool = False
    data_repositories: List[str] = field(default_factory=list)
    data_doi: Optional[str] = None
    data_on_request: bool = False
    
    # Code availability
    has_code_statement: bool = False
    code_available: bool = False
    code_repositories: List[str] = field(default_factory=list)
    code_url: Optional[str] = None
    
    # Preprints
    has_preprint: bool = False
    preprint_servers: List[str] = field(default_factory=list)
    preprint_doi: Optional[str] = None
    
    # Methods transparency
    is_preregistered: bool = False
    has_protocol: bool = False
    has_supplementary: bool = False
    methods_transparency_score: float = 0.0
    
    # FAIR indicators (estimated)
    findable_score: float = 0.0
    accessible_score: float = 0.0
    interoperable_score: float = 0.0
    reusable_score: float = 0.0
    fair_score: float = 0.0
    
    # Composite score
    open_science_score: float = 0.0
    openness_level: str = "closed"  # closed, minimal, partial, open, exemplary
    
    # Raw detection info
    detected_urls: List[str] = field(default_factory=list)
    detection_notes: List[str] = field(default_factory=list)
    
    def calculate_open_science_score(self) -> float:
        """Calculate composite open science score (0-100)."""
        score = 0.0
        
        # Open access (max 25 points)
        if self.is_open_access:
            oa_points = {"gold": 25, "green": 20, "hybrid": 22, "bronze": 15}
            score += oa_points.get(self.oa_status, 10)
        
        # Data availability (max 25 points)
        if self.data_available:
            score += 20
            if self.data_repositories:
                score += 5
        elif self.has_data_statement:
            score += 5
            if self.data_on_request:
                score += 3
        
        # Code availability (max 25 points)
        if self.code_available:
            score += 20
            if self.code_repositories:
                score += 5
        elif self.has_code_statement:
            score += 5
        
        # Preprints (max 10 points)
        if self.has_preprint:
            score += 10
        
        # Methods transparency (max 15 points)
        if self.is_preregistered:
            score += 8
        if self.has_protocol:
            score += 4
        if self.has_supplementary:
            score += 3
        
        self.open_science_score = min(score, 100)
        
        # Determine openness level
        if self.open_science_score >= 80:
            self.openness_level = "exemplary"
        elif self.open_science_score >= 60:
            self.openness_level = "open"
        elif self.open_science_score >= 40:
            self.openness_level = "partial"
        elif self.open_science_score >= 20:
            self.openness_level = "minimal"
        else:
            self.openness_level = "closed"
        
        return self.open_science_score
    
    def calculate_fair_score(self) -> float:
        """Calculate FAIR compliance score (0-100)."""
        # Findable (DOI, metadata, indexed)
        findable = 0.0
        if self.doi:
            findable += 50
        if self.data_doi:
            findable += 30
        if self.data_repositories:
            findable += 20
        self.findable_score = min(findable, 100)
        
        # Accessible (OA, data available, code available)
        accessible = 0.0
        if self.is_open_access:
            accessible += 40
        if self.data_available:
            accessible += 35
        if self.code_available:
            accessible += 25
        self.accessible_score = min(accessible, 100)
        
        # Interoperable (standard formats, repositories)
        interoperable = 0.0
        if self.data_repositories:
            interoperable += 50
        if self.code_repositories:
            interoperable += 30
        if self.has_supplementary:
            interoperable += 20
        self.interoperable_score = min(interoperable, 100)
        
        # Reusable (licenses, documentation)
        reusable = 0.0
        if self.data_available and self.data_repositories:
            reusable += 40
        if self.code_available and self.code_repositories:
            reusable += 40
        if self.is_preregistered or self.has_protocol:
            reusable += 20
        self.reusable_score = min(reusable, 100)
        
        # Overall FAIR score
        self.fair_score = (
            self.findable_score * 0.25 +
            self.accessible_score * 0.25 +
            self.interoperable_score * 0.25 +
            self.reusable_score * 0.25
        )
        
        return self.fair_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "Doc Id": self.doc_id,
            "doi": self.doi,
            "year": self.year,
            "is_open_access": self.is_open_access,
            "oa_status": self.oa_status,
            "has_data_statement": self.has_data_statement,
            "data_available": self.data_available,
            "data_repositories": ", ".join(self.data_repositories) if self.data_repositories else "",
            "has_code_statement": self.has_code_statement,
            "code_available": self.code_available,
            "code_repositories": ", ".join(self.code_repositories) if self.code_repositories else "",
            "has_preprint": self.has_preprint,
            "preprint_servers": ", ".join(self.preprint_servers) if self.preprint_servers else "",
            "is_preregistered": self.is_preregistered,
            "has_protocol": self.has_protocol,
            "methods_transparency_score": self.methods_transparency_score,
            "findable_score": self.findable_score,
            "accessible_score": self.accessible_score,
            "interoperable_score": self.interoperable_score,
            "reusable_score": self.reusable_score,
            "fair_score": self.fair_score,
            "open_science_score": self.open_science_score,
            "openness_level": self.openness_level,
        }

@dataclass
class OpenScienceAnalysisResult:
    """Container for complete open science analysis results."""
    records: List[OpenScienceRecord]
    summary_df: pd.DataFrame
    aggregate_stats: Dict[str, Any]
    temporal_trends: pd.DataFrame
    repository_usage: Dict[str, int]
    field_comparison: pd.DataFrame
    parameters: Dict[str, Any]
    
    def get_open_access_papers(self) -> pd.DataFrame:
        """Get all open access papers."""
        return self.summary_df[self.summary_df["is_open_access"] == True]
    
    def get_papers_with_data(self) -> pd.DataFrame:
        """Get papers with data availability."""
        return self.summary_df[self.summary_df["data_available"] == True]
    
    def get_papers_with_code(self) -> pd.DataFrame:
        """Get papers with code availability."""
        return self.summary_df[self.summary_df["code_available"] == True]
    
    def get_exemplary_papers(self, threshold: float = 80) -> pd.DataFrame:
        """Get papers with high open science scores."""
        return self.summary_df[self.summary_df["open_science_score"] >= threshold]
    
    def get_preregistered_papers(self) -> pd.DataFrame:
        """Get preregistered studies."""
        return self.summary_df[self.summary_df["is_preregistered"] == True]

# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

def detect_data_availability(
    text: str,
    check_repositories: bool = True) -> Dict[str, Any]:
    """
    Detect data availability indicators in text.
    
    Parameters
    ----------
    text : str
        Text to analyze (abstract, full text, etc.)
    check_repositories : bool
        Whether to check for repository mentions.
    
    Returns
    -------
    Dict with detection results.
    """
    if not text or pd.isna(text):
        return {
            "has_statement": False,
            "data_available": False,
            "repositories": [],
            "on_request": False,
            "urls": [],
        }
    
    text_lower = text.lower()
    result = {
        "has_statement": False,
        "data_available": False,
        "repositories": [],
        "on_request": False,
        "urls": [],
    }
    
    # Check for data availability statements
    for pattern in DATA_STATEMENT_PATTERNS:
        if re.search(pattern, text_lower):
            result["has_statement"] = True
            break
    
    # Check for "on request" availability
    if re.search(r"(available|provided)\s+(upon|on)\s+request", text_lower):
        result["on_request"] = True
        result["has_statement"] = True
    
    # Check for repository mentions
    if check_repositories:
        for repo_name, pattern in DATA_REPOSITORY_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                result["repositories"].append(repo_name)
                result["data_available"] = True
                result["has_statement"] = True
    
    # Extract URLs
    url_pattern = r"https?://[^\s<>\"{}|\\^`\[\]]+"
    urls = re.findall(url_pattern, text)
    result["urls"] = urls[:10]  # Limit to 10 URLs
    
    # If repositories found, data is available
    if result["repositories"]:
        result["data_available"] = True
    
    return result

def detect_code_availability(
    text: str) -> Dict[str, Any]:
    """
    Detect code/software availability indicators in text.
    
    Parameters
    ----------
    text : str
        Text to analyze.
    
    Returns
    -------
    Dict with detection results.
    """
    if not text or pd.isna(text):
        return {
            "has_statement": False,
            "code_available": False,
            "repositories": [],
            "urls": [],
        }
    
    text_lower = text.lower()
    result = {
        "has_statement": False,
        "code_available": False,
        "repositories": [],
        "urls": [],
    }
    
    # Check for code availability statements
    for pattern in CODE_STATEMENT_PATTERNS:
        if re.search(pattern, text_lower):
            result["has_statement"] = True
            break
    
    # Check for repository mentions
    for repo_name, pattern in CODE_REPOSITORY_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            result["repositories"].append(repo_name)
            result["code_available"] = True
            result["has_statement"] = True
    
    # Extract code-related URLs
    code_url_patterns = [
        r"github\.com/[\w-]+/[\w-]+",
        r"gitlab\.com/[\w-]+/[\w-]+",
        r"bitbucket\.org/[\w-]+/[\w-]+",
    ]
    
    for pattern in code_url_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        result["urls"].extend(matches)
    
    return result

def detect_preprint(
    text: str,
    doi: str = None) -> Dict[str, Any]:
    """
    Detect preprint indicators.
    
    Parameters
    ----------
    text : str
        Text to analyze.
    doi : str, optional
        DOI to check for preprint patterns.
    
    Returns
    -------
    Dict with detection results.
    """
    result = {
        "has_preprint": False,
        "servers": [],
        "preprint_doi": None,
    }
    
    # Check DOI for preprint patterns
    if doi and pd.notna(doi):
        doi_str = str(doi).lower()
        if "10.1101" in doi_str:  # bioRxiv/medRxiv
            result["has_preprint"] = True
            if "medrxiv" in doi_str:
                result["servers"].append("medrxiv")
            else:
                result["servers"].append("biorxiv")
            result["preprint_doi"] = doi
        elif "arxiv" in doi_str:
            result["has_preprint"] = True
            result["servers"].append("arxiv")
            result["preprint_doi"] = doi
        elif "ssrn" in doi_str:
            result["has_preprint"] = True
            result["servers"].append("ssrn")
            result["preprint_doi"] = doi
    
    # Check text for preprint mentions
    if text and pd.notna(text):
        for server_name, pattern in PREPRINT_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                if server_name not in result["servers"]:
                    result["servers"].append(server_name)
                result["has_preprint"] = True
    
    return result

def detect_open_access(
    text: str = None,
    oa_column_value: Any = None) -> Dict[str, Any]:
    """
    Detect open access status.
    
    Parameters
    ----------
    text : str, optional
        Text to analyze.
    oa_column_value : Any, optional
        Value from OA status column (e.g., from OpenAlex).
    
    Returns
    -------
    Dict with detection results.
    """
    result = {
        "is_open_access": False,
        "oa_status": "closed",
        "oa_url": None,
    }
    
    # Check OA column value first
    if oa_column_value is not None and pd.notna(oa_column_value):
        oa_str = str(oa_column_value).lower()
        
        if oa_str in ["gold", "green", "hybrid", "bronze"]:
            result["is_open_access"] = True
            result["oa_status"] = oa_str
        elif oa_str in ["true", "yes", "1", "open"]:
            result["is_open_access"] = True
            result["oa_status"] = "gold"  # Assume gold if just "open"
        elif "gold" in oa_str:
            result["is_open_access"] = True
            result["oa_status"] = "gold"
        elif "green" in oa_str:
            result["is_open_access"] = True
            result["oa_status"] = "green"
        elif "hybrid" in oa_str:
            result["is_open_access"] = True
            result["oa_status"] = "hybrid"
        elif "bronze" in oa_str:
            result["is_open_access"] = True
            result["oa_status"] = "bronze"
    
    # Check text for OA patterns
    if text and pd.notna(text):
        text_lower = text.lower()
        
        for oa_type, patterns in OA_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    result["is_open_access"] = True
                    if result["oa_status"] == "closed":
                        result["oa_status"] = oa_type
                    break
    
    return result

def detect_methods_transparency(
    text: str) -> Dict[str, Any]:
    """
    Detect methods transparency indicators.
    
    Parameters
    ----------
    text : str
        Text to analyze.
    
    Returns
    -------
    Dict with detection results.
    """
    if not text or pd.isna(text):
        return {
            "is_preregistered": False,
            "has_protocol": False,
            "has_supplementary": False,
            "transparency_score": 0.0,
        }
    
    text_lower = text.lower()
    result = {
        "is_preregistered": False,
        "has_protocol": False,
        "has_supplementary": False,
        "transparency_score": 0.0,
    }
    
    # Check for preregistration
    prereg_patterns = [
        r"pre[\s-]?regist",
        r"registered\s+report",
        r"aspredicted",
        r"osf\.io/[\w]+",
        r"prospero",
    ]
    for pattern in prereg_patterns:
        if re.search(pattern, text_lower):
            result["is_preregistered"] = True
            break
    
    # Check for protocol
    protocol_patterns = [
        r"study\s+protocol",
        r"analysis\s+plan",
        r"research\s+protocol",
        r"trial\s+protocol",
    ]
    for pattern in protocol_patterns:
        if re.search(pattern, text_lower):
            result["has_protocol"] = True
            break
    
    # Check for supplementary materials
    supp_patterns = [
        r"supplementary",
        r"supporting\s+information",
        r"additional\s+file",
        r"appendix",
    ]
    for pattern in supp_patterns:
        if re.search(pattern, text_lower):
            result["has_supplementary"] = True
            break
    
    # Calculate transparency score
    score = 0.0
    if result["is_preregistered"]:
        score += 50
    if result["has_protocol"]:
        score += 30
    if result["has_supplementary"]:
        score += 20
    result["transparency_score"] = score
    
    return result

# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def analyze_open_science(
    df: pd.DataFrame,
    id_col: str = "unique-id",
    doi_col: str = "DOI",
    title_col: str = "Title",
    abstract_col: str = "Abstract",
    year_col: str = "Year",
    oa_col: str = None,
    field_col: str = None,
    full_text_col: str = None,
    verbose: bool = True) -> OpenScienceAnalysisResult:
    """
    Analyze open science practices in a bibliographic dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    id_col : str
        Document ID column.
    doi_col : str
        DOI column.
    title_col : str
        Title column.
    abstract_col : str
        Abstract column.
    year_col : str
        Year column.
    oa_col : str, optional
        Open access status column.
    field_col : str, optional
        Field/subject area column.
    full_text_col : str, optional
        Full text column (if available).
    verbose : bool
        Print progress.
    
    Returns
    -------
    OpenScienceAnalysisResult
    """
    if verbose:
        print("=" * 50)
        print("Reproducibility & Open Science Analysis")
        print("=" * 50)
    
    records = []
    
    for idx, row in df.iterrows():
        doc_id = row.get(id_col, idx)
        doi = row.get(doi_col) if doi_col and doi_col in df.columns else None
        title = row.get(title_col) if title_col and title_col in df.columns else None
        year = row.get(year_col) if year_col and year_col in df.columns else None
        
        # Get text to analyze
        abstract = row.get(abstract_col, "") if abstract_col and abstract_col in df.columns else ""
        full_text = row.get(full_text_col, "") if full_text_col and full_text_col in df.columns else ""
        
        # Combine available text
        text_to_analyze = " ".join([
            str(abstract) if pd.notna(abstract) else "",
            str(full_text) if pd.notna(full_text) else "",
        ])
        
        # Create record
        record = OpenScienceRecord(
            doc_id=doc_id,
            doi=str(doi) if pd.notna(doi) else None,
            title=str(title) if pd.notna(title) else None,
            year=int(year) if pd.notna(year) else None)
        
        # Detect open access
        oa_value = row.get(oa_col) if oa_col and oa_col in df.columns else None
        oa_result = detect_open_access(text_to_analyze, oa_value)
        record.is_open_access = oa_result["is_open_access"]
        record.oa_status = oa_result["oa_status"]
        
        # Detect data availability
        data_result = detect_data_availability(text_to_analyze)
        record.has_data_statement = data_result["has_statement"]
        record.data_available = data_result["data_available"]
        record.data_repositories = data_result["repositories"]
        record.data_on_request = data_result["on_request"]
        record.detected_urls.extend(data_result["urls"])
        
        # Detect code availability
        code_result = detect_code_availability(text_to_analyze)
        record.has_code_statement = code_result["has_statement"]
        record.code_available = code_result["code_available"]
        record.code_repositories = code_result["repositories"]
        
        # Detect preprints
        preprint_result = detect_preprint(text_to_analyze, str(doi) if pd.notna(doi) else None)
        record.has_preprint = preprint_result["has_preprint"]
        record.preprint_servers = preprint_result["servers"]
        record.preprint_doi = preprint_result["preprint_doi"]
        
        # Detect methods transparency
        methods_result = detect_methods_transparency(text_to_analyze)
        record.is_preregistered = methods_result["is_preregistered"]
        record.has_protocol = methods_result["has_protocol"]
        record.has_supplementary = methods_result["has_supplementary"]
        record.methods_transparency_score = methods_result["transparency_score"]
        
        # Calculate scores
        record.calculate_open_science_score()
        record.calculate_fair_score()
        
        records.append(record)
    
    if verbose:
        print(f"  Analyzed {len(records)} papers")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame([r.to_dict() for r in records])
    
    # Add field information if available
    if field_col and field_col in df.columns:
        field_map = dict(zip(df[id_col], df[field_col]))
        summary_df["field"] = summary_df["doc_id"].map(field_map)
    
    # Calculate aggregate statistics
    aggregate_stats = _calculate_aggregate_stats(records)
    
    # Calculate temporal trends
    temporal_trends = _calculate_temporal_trends(records)
    
    # Repository usage
    repository_usage = _calculate_repository_usage(records)
    
    # Field comparison
    field_comparison = pd.DataFrame()
    if "field" in summary_df.columns:
        field_comparison = _calculate_field_comparison(summary_df)
    
    result = OpenScienceAnalysisResult(
        records=records,
        summary_df=summary_df,
        aggregate_stats=aggregate_stats,
        temporal_trends=temporal_trends,
        repository_usage=repository_usage,
        field_comparison=field_comparison,
        parameters={
            "id_col": id_col,
            "doi_col": doi_col,
            "year_col": year_col,
        })
    
    if verbose:
        _print_open_science_summary(result)
    
    return result

def _calculate_aggregate_stats(records: List[OpenScienceRecord]) -> Dict[str, Any]:
    """Calculate aggregate statistics."""
    total = len(records)
    if total == 0:
        return {}
    
    stats = {
        "total_papers": total,
        
        # Open Access
        "open_access_count": sum(1 for r in records if r.is_open_access),
        "open_access_pct": sum(1 for r in records if r.is_open_access) / total * 100,
        "oa_gold_count": sum(1 for r in records if r.oa_status == "gold"),
        "oa_green_count": sum(1 for r in records if r.oa_status == "green"),
        "oa_hybrid_count": sum(1 for r in records if r.oa_status == "hybrid"),
        "oa_bronze_count": sum(1 for r in records if r.oa_status == "bronze"),
        
        # Data
        "data_statement_count": sum(1 for r in records if r.has_data_statement),
        "data_statement_pct": sum(1 for r in records if r.has_data_statement) / total * 100,
        "data_available_count": sum(1 for r in records if r.data_available),
        "data_available_pct": sum(1 for r in records if r.data_available) / total * 100,
        
        # Code
        "code_statement_count": sum(1 for r in records if r.has_code_statement),
        "code_statement_pct": sum(1 for r in records if r.has_code_statement) / total * 100,
        "code_available_count": sum(1 for r in records if r.code_available),
        "code_available_pct": sum(1 for r in records if r.code_available) / total * 100,
        
        # Preprints
        "preprint_count": sum(1 for r in records if r.has_preprint),
        "preprint_pct": sum(1 for r in records if r.has_preprint) / total * 100,
        
        # Methods
        "preregistered_count": sum(1 for r in records if r.is_preregistered),
        "preregistered_pct": sum(1 for r in records if r.is_preregistered) / total * 100,
        
        # Scores
        "mean_open_science_score": np.mean([r.open_science_score for r in records]),
        "median_open_science_score": np.median([r.open_science_score for r in records]),
        "mean_fair_score": np.mean([r.fair_score for r in records]),
        
        # Openness levels
        "exemplary_count": sum(1 for r in records if r.openness_level == "exemplary"),
        "open_count": sum(1 for r in records if r.openness_level == "open"),
        "partial_count": sum(1 for r in records if r.openness_level == "partial"),
        "minimal_count": sum(1 for r in records if r.openness_level == "minimal"),
        "closed_count": sum(1 for r in records if r.openness_level == "closed"),
    }
    
    return stats

def _calculate_temporal_trends(records: List[OpenScienceRecord]) -> pd.DataFrame:
    """Calculate temporal trends in open science practices."""
    year_data = defaultdict(lambda: {
        "total": 0,
        "open_access": 0,
        "data_available": 0,
        "code_available": 0,
        "preprint": 0,
        "preregistered": 0,
        "total_os_score": 0,
    })
    
    for record in records:
        if record.year and record.year >= 2000:
            year_data[record.year]["total"] += 1
            year_data[record.year]["open_access"] += int(record.is_open_access)
            year_data[record.year]["data_available"] += int(record.data_available)
            year_data[record.year]["code_available"] += int(record.code_available)
            year_data[record.year]["preprint"] += int(record.has_preprint)
            year_data[record.year]["preregistered"] += int(record.is_preregistered)
            year_data[record.year]["total_os_score"] += record.open_science_score
    
    trends = []
    for year, data in sorted(year_data.items()):
        if data["total"] > 0:
            trends.append({
                "year": year,
                "total_papers": data["total"],
                "open_access_pct": data["open_access"] / data["total"] * 100,
                "data_available_pct": data["data_available"] / data["total"] * 100,
                "code_available_pct": data["code_available"] / data["total"] * 100,
                "preprint_pct": data["preprint"] / data["total"] * 100,
                "preregistered_pct": data["preregistered"] / data["total"] * 100,
                "Mean OS Score": data["total_os_score"] / data["total"],
            })
    
    return pd.DataFrame(trends)

def _calculate_repository_usage(records: List[OpenScienceRecord]) -> Dict[str, int]:
    """Calculate repository usage statistics."""
    data_repos = Counter()
    code_repos = Counter()
    preprint_servers = Counter()
    
    for record in records:
        for repo in record.data_repositories:
            data_repos[repo] += 1
        for repo in record.code_repositories:
            code_repos[repo] += 1
        for server in record.preprint_servers:
            preprint_servers[server] += 1
    
    return {
        "data_repositories": dict(data_repos),
        "code_repositories": dict(code_repos),
        "preprint_servers": dict(preprint_servers),
    }

def _calculate_field_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate open science metrics by field."""
    if "field" not in summary_df.columns:
        return pd.DataFrame()
    
    field_stats = summary_df.groupby("field").agg({
        "is_open_access": "mean",
        "data_available": "mean",
        "code_available": "mean",
        "has_preprint": "mean",
        "open_science_score": "mean",
        "Doc Id": "count",
    }).reset_index()
    
    field_stats.columns = [
        "field", "open_access_rate", "data_rate", "code_rate",
        "preprint_rate", "mean_os_score", "paper_count"
    ]
    
    # Convert to percentages
    for col in ["open_access_rate", "data_rate", "code_rate", "preprint_rate"]:
        field_stats[col] = field_stats[col] * 100
    
    return field_stats.sort_values("Mean OS Score", ascending=False)

def _print_open_science_summary(result: OpenScienceAnalysisResult) -> None:
    """Print summary of open science analysis."""
    stats = result.aggregate_stats
    
    print("\nOpen Science Summary:")
    print("-" * 40)
    print(f"Total papers analyzed: {stats['total_papers']}")
    
    print("\nOpen Access:")
    print(f"  Open access papers: {stats['open_access_count']} ({stats['open_access_pct']:.1f}%)")
    print(f"    Gold: {stats['oa_gold_count']}, Green: {stats['oa_green_count']}, "
          f"Hybrid: {stats['oa_hybrid_count']}, Bronze: {stats['oa_bronze_count']}")
    
    print("\nData Sharing:")
    print(f"  Data statement: {stats['data_statement_count']} ({stats['data_statement_pct']:.1f}%)")
    print(f"  Data available: {stats['data_available_count']} ({stats['data_available_pct']:.1f}%)")
    
    print("\nCode Sharing:")
    print(f"  Code statement: {stats['code_statement_count']} ({stats['code_statement_pct']:.1f}%)")
    print(f"  Code available: {stats['code_available_count']} ({stats['code_available_pct']:.1f}%)")
    
    print("\nPreprints & Methods:")
    print(f"  Has preprint: {stats['preprint_count']} ({stats['preprint_pct']:.1f}%)")
    print(f"  Preregistered: {stats['preregistered_count']} ({stats['preregistered_pct']:.1f}%)")
    
    print("\nOpen Science Scores:")
    print(f"  Mean score: {stats['mean_open_science_score']:.1f}")
    print(f"  Median score: {stats['median_open_science_score']:.1f}")
    print(f"  FAIR score (mean): {stats['mean_fair_score']:.1f}")
    
    print("\nOpenness Levels:")
    print(f"  Exemplary: {stats['exemplary_count']}, Open: {stats['open_count']}, "
          f"Partial: {stats['partial_count']}, Minimal: {stats['minimal_count']}, "
          f"Closed: {stats['closed_count']}")

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_open_science_overview(
    result: OpenScienceAnalysisResult,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None) -> plt.Figure:
    """Plot comprehensive open science overview."""
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    stats = result.aggregate_stats
    
    # 1. Open science practices adoption
    practices = ["Open Access", "Data Available", "Code Available", "Preprint", "Preregistered"]
    percentages = [
        stats["open_access_pct"],
        stats["data_available_pct"],
        stats["code_available_pct"],
        stats["preprint_pct"],
        stats["preregistered_pct"],
    ]
    
    colors = "lightblue"
    bars = ax.bar(range(len(practices)), percentages, color=colors)
    ax.set_xticks(range(len(practices)))
    ax.set_xticklabels(practices, rotation=45, ha="right")
    ax.set_ylabel("Percentage of Papers", fontsize=11)
    ax.set_title("Open Science Practices Adoption", fontsize=12)
    ax.set_ylim(0, 100)
    
    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=10)
    
    # 2. Open Access breakdown
    oa_types = ["Gold", "Green", "Hybrid", "Bronze", "Closed"]
    oa_counts = [
        stats["oa_gold_count"],
        stats["oa_green_count"],
        stats["oa_hybrid_count"],
        stats["oa_bronze_count"],
        stats["total_papers"] - stats["open_access_count"],
    ]
    
    oa_colors = "lightblue"
    bars = ax.bar(range(len(oa_types)), oa_counts, color=oa_colors)
    ax.set_xticks(range(len(oa_types)))
    ax.set_xticklabels(oa_types)
    ax.set_ylabel("Number of Papers", fontsize=11)
    ax.set_title("Open Access Status Distribution", fontsize=12)
    
    for bar, count in zip(bars, oa_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(count), ha="center", va="bottom", fontsize=10)
    
    # 3. Openness levels
    levels = ["Exemplary", "Open", "Partial", "Minimal", "Closed"]
    level_counts = [
        stats["exemplary_count"],
        stats["open_count"],
        stats["partial_count"],
        stats["minimal_count"],
        stats["closed_count"],
    ]
    
    level_colors = "lightblue"
    bars = ax.bar(range(len(levels)), level_counts, color=level_colors)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels)
    ax.set_ylabel("Number of Papers", fontsize=11)
    ax.set_title("Openness Level Distribution", fontsize=12)
    
    for bar, count in zip(bars, level_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(count), ha="center", va="bottom", fontsize=10)
    
    # 4. Open Science Score distribution
    scores = result.summary_df["open_science_score"]
    
    ax.hist(scores, bins=20, color="lightblue", edgecolor="white", alpha=0.7)
    ax.set_xlabel("Open Science Score", fontsize=11)
    ax.set_ylabel("Number of Papers", fontsize=11)
    ax.set_title("Open Science Score Distribution", fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

def plot_temporal_trends(
    result: OpenScienceAnalysisResult,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None) -> plt.Figure:
    """Plot temporal trends in open science adoption."""
    
    if len(result.temporal_trends) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(False)
        ax.text(0.5, 0.5, "Temporal data not available", ha="center", va="center")
        ax.axis("off")
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    df = result.temporal_trends
    
    # 1. Open access trend
    ax.bar(df["year"], df["open_access_pct"], color="lightblue")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Percentage", fontsize=11)
    ax.set_title("Open Access Adoption Over Time", fontsize=12)
    
    # 2. Data availability trend
    ax.bar(df["year"], df["data_available_pct"], color="lightblue")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Percentage", fontsize=11)
    ax.set_title("Data Availability Over Time", fontsize=12)
    
    # 3. Code availability trend
    ax.bar(df["year"], df["code_available_pct"], color="lightblue")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Percentage", fontsize=11)
    ax.set_title("Code Availability Over Time", fontsize=12)
    
    # 4. Mean open science score trend
    ax.bar(df["year"], df["mean_os_score"], color="lightblue")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Mean Score", fontsize=11)
    ax.set_title("Mean Open Science Score Over Time", fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

def plot_repository_usage(
    result: OpenScienceAnalysisResult,
    top_n: int = 10,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None) -> plt.Figure:
    """Plot repository usage statistics."""
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    repo_data = result.repository_usage
    
    # 1. Data repositories
    data_repos = repo_data.get("data_repositories", {})
    if data_repos:
        sorted_repos = sorted(data_repos.items(), key=lambda x: -x[1])[:top_n]
        names, counts = zip(*sorted_repos) if sorted_repos else ([], [])
        
        colors = "lightblue"
        ax.barh(range(len(names)), counts, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Number of Papers", fontsize=11)
        ax.set_title("Data Repositories", fontsize=12)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No data repositories detected", ha="center", va="center")
        ax.axis("off")
    
    # 2. Code repositories
    code_repos = repo_data.get("code_repositories", {})
    if code_repos:
        sorted_repos = sorted(code_repos.items(), key=lambda x: -x[1])[:top_n]
        names, counts = zip(*sorted_repos) if sorted_repos else ([], [])
        
        colors = "lightblue"
        ax.barh(range(len(names)), counts, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Number of Papers", fontsize=11)
        ax.set_title("Code Repositories", fontsize=12)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No code repositories detected", ha="center", va="center")
        ax.axis("off")
    
    # 3. Preprint servers
    preprint_servers = repo_data.get("preprint_servers", {})
    if preprint_servers:
        sorted_servers = sorted(preprint_servers.items(), key=lambda x: -x[1])[:top_n]
        names, counts = zip(*sorted_servers) if sorted_servers else ([], [])
        
        colors = "lightblue"
        ax.barh(range(len(names)), counts, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Number of Papers", fontsize=11)
        ax.set_title("Preprint Servers", fontsize=12)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No preprint servers detected", ha="center", va="center")
        ax.axis("off")
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

def plot_fair_scores(
    result: OpenScienceAnalysisResult,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None) -> plt.Figure:
    """Plot FAIR compliance scores."""
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    df = result.summary_df
    
    # 1. FAIR component means
    components = ["Findable", "Accessible", "Interoperable", "Reusable"]
    means = [
        df["findable_score"].mean(),
        df["accessible_score"].mean(),
        df["interoperable_score"].mean(),
        df["reusable_score"].mean(),
    ]
    
    colors = "lightblue"
    bars = ax.bar(range(len(components)), means, color=colors)
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(components)
    ax.set_ylabel("Mean Score", fontsize=11)
    ax.set_title("FAIR Component Scores", fontsize=12)
    ax.set_ylim(0, 100)
    
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=10)
    
    # 2. FAIR score distribution
    ax.hist(df["fair_score"], bins=20, color="lightblue", edgecolor="white", alpha=0.7)
    ax.set_xlabel("FAIR Score", fontsize=11)
    ax.set_ylabel("Number of Papers", fontsize=11)
    ax.set_title("FAIR Score Distribution", fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig

# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_open_science_results(
    result: OpenScienceAnalysisResult,
    output_dir: str) -> Dict[str, str]:
    """Export open science analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    excel_path = os.path.join(output_dir, "open_science_analysis.xlsx")
    
    with pd.ExcelWriter(excel_path) as writer:
        # Summary
        result.summary_df.to_excel(writer, sheet_name="Summary", index=False)
        
        # Aggregate stats
        stats_df = pd.DataFrame([result.aggregate_stats])
        stats_df.to_excel(writer, sheet_name="Aggregate_Stats", index=False)
        
        # Temporal trends
        if len(result.temporal_trends) > 0:
            result.temporal_trends.to_excel(writer, sheet_name="Temporal_Trends", index=False)
        
        # Repository usage
        for repo_type, repos in result.repository_usage.items():
            if repos:
                repo_df = pd.DataFrame([
                    {"repository": k, "count": v} for k, v in repos.items()
                ]).sort_values("count", ascending=False)
                sheet_name = repo_type.replace("_", " ").title().replace(" ", "_")[:31]
                repo_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Field comparison
        if len(result.field_comparison) > 0:
            result.field_comparison.to_excel(writer, sheet_name="Field_Comparison", index=False)
        
        # Exemplary papers
        exemplary = result.get_exemplary_papers(threshold=80)
        if len(exemplary) > 0:
            exemplary.to_excel(writer, sheet_name="Exemplary_Papers", index=False)
    
    print(f"Exported to: {excel_path}")
    return {"excel": excel_path}

# =============================================================================
# CLASS INTEGRATION
# =============================================================================

def add_open_science_methods(cls):
    """Add open science methods to BiblioAnalysis class."""
    
    def run_open_science_analysis_method(
        self,
        oa_col: str = None,
        full_text_col: str = None,
        **kwargs
    ) -> OpenScienceAnalysisResult:
        """Run open science analysis."""
        id_col = "unique-id" if "unique-id" in self.df.columns else "Doc ID"
        doi_col = "DOI"
        title_col = "Title"
        abstract_col = "Abstract"
        year_col = "Year"
        
        self.open_science_results = analyze_open_science(
            self.df,
            id_col=id_col,
            doi_col=doi_col,
            title_col=title_col,
            abstract_col=abstract_col,
            year_col=year_col,
            oa_col=oa_col,
            full_text_col=full_text_col,
            **kwargs
        )
        
        if hasattr(self, "res_folder") and self.res_folder:
            export_open_science_results(
                self.open_science_results,
                self.res_folder
            )
        
        return self.open_science_results
    
    def plot_open_science_method(
        self,
        plot_type: str = "overview",
        save: bool = True,
        **kwargs
    ) -> plt.Figure:
        """Create open science visualizations."""
        if not hasattr(self, "open_science_results"):
            raise ValueError("Run run_open_science_analysis first")
        
        save_path = None
        if save and hasattr(self, "res_folder") and self.res_folder:
            save_path = os.path.join(self.res_folder, f"open_science_{plot_type}")
        
        plot_functions = {
            "overview": plot_open_science_overview,
            "temporal": plot_temporal_trends,
            "repositories": plot_repository_usage,
            "fair": plot_fair_scores,
        }
        
        if plot_type not in plot_functions:
            raise ValueError(f"Unknown plot_type: {plot_type}")
        
        return plot_functions[plot_type](self.open_science_results, save_path=save_path, **kwargs)
    
    cls.run_open_science_analysis = run_open_science_analysis_method
    cls.plot_open_science = plot_open_science_method
    
    return cls
# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import biblium as bb
    ba = bb.BiblioAnalysis(f_name="data\\open alex dataset.csv", db="oa")
    
    print("Open Science Analysis")
    print("=" * 60)
    
    # Run analysis
    result = analyze_open_science(
        df=ba.df,
        doi_col="DOI",
        abstract_col="Abstract",
        year_col="Year",
        verbose=True)
    
    # Print summary
    print(f"\nAnalyzed {len(result.summary_df)} papers")
    print(f"Open access rate: {result.aggregate_stats.get('oa_rate', 0)*100:.1f}%")
    
    # Visualizations
    print("\nGenerating plots...")
    plot_open_science_overview(result, save_path="results/open_science_overview")
    plot_temporal_trends(result, save_path="results/os_temporal_trends")
    
    # Export
    print("\nExporting results...")
    export_open_science_results(result, "results")
    
    print("\nDone!")
