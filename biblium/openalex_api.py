# -*- coding: utf-8 -*-
"""
OpenAlex API Module for Biblium.

Provides a comprehensive interface to the OpenAlex API for:
- Searching works, authors, institutions, sources, concepts, publishers, funders
- Bulk downloading datasets with pagination
- Enriching existing bibliometric data
- Getting aggregate statistics
- Converting OpenAlex data to Biblium-compatible format

OpenAlex API Documentation: https://docs.openalex.org/

Usage
-----
    from biblium.openalex_api import OpenAlexClient
    
    # Create client (email recommended for polite pool)
    client = OpenAlexClient(email="your@email.com")
    
    # Search works
    works = client.search_works("machine learning climate change", max_results=100)
    
    # Get author by ORCID
    author = client.get_author(orcid="0000-0002-1234-5678")
    
    # Download full dataset
    df = client.download_works(
        query="hybrid governance",
        filters={"publication_year": "2020-2024"},
        max_results=1000
    )
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Union
from urllib.parse import quote, urlencode
from dataclasses import dataclass, field
from datetime import datetime
import json

import pandas as pd
import numpy as np

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None


# =============================================================================
# CONSTANTS
# =============================================================================

OPENALEX_BASE_URL = "https://api.openalex.org"

# Entity types
ENTITY_TYPES = Literal["works", "authors", "institutions", "sources", "concepts", "publishers", "funders", "topics"]

# Common filters for works
WORK_FILTERS = {
    "publication_year": "publication_year",
    "type": "type",
    "is_oa": "open_access.is_oa",
    "cited_by_count": "cited_by_count",
    "doi": "doi",
    "pmid": "pmid",
    "concepts": "concepts.id",
    "institutions": "authorships.institutions.id",
    "countries": "authorships.countries",
    "source": "primary_location.source.id",
    "author": "authorships.author.id",
    "funder": "grants.funder",
    "sdg": "sustainable_development_goals.id",
    "language": "language",
}

# SDG ID mapping
SDG_IDS = {
    1: "https://metadata.un.org/sdg/1",
    2: "https://metadata.un.org/sdg/2",
    3: "https://metadata.un.org/sdg/3",
    4: "https://metadata.un.org/sdg/4",
    5: "https://metadata.un.org/sdg/5",
    6: "https://metadata.un.org/sdg/6",
    7: "https://metadata.un.org/sdg/7",
    8: "https://metadata.un.org/sdg/8",
    9: "https://metadata.un.org/sdg/9",
    10: "https://metadata.un.org/sdg/10",
    11: "https://metadata.un.org/sdg/11",
    12: "https://metadata.un.org/sdg/12",
    13: "https://metadata.un.org/sdg/13",
    14: "https://metadata.un.org/sdg/14",
    15: "https://metadata.un.org/sdg/15",
    16: "https://metadata.un.org/sdg/16",
    17: "https://metadata.un.org/sdg/17",
}

SDG_NAMES = {
    1: "No Poverty",
    2: "Zero Hunger",
    3: "Good Health and Well-being",
    4: "Quality Education",
    5: "Gender Equality",
    6: "Clean Water and Sanitation",
    7: "Affordable and Clean Energy",
    8: "Decent Work and Economic Growth",
    9: "Industry, Innovation and Infrastructure",
    10: "Reduced Inequalities",
    11: "Sustainable Cities and Communities",
    12: "Responsible Consumption and Production",
    13: "Climate Action",
    14: "Life Below Water",
    15: "Life on Land",
    16: "Peace, Justice and Strong Institutions",
    17: "Partnerships for the Goals",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OpenAlexWork:
    """Represents an OpenAlex work (publication)."""
    id: str
    doi: Optional[str] = None
    title: Optional[str] = None
    publication_year: Optional[int] = None
    publication_date: Optional[str] = None
    type: Optional[str] = None
    cited_by_count: int = 0
    is_oa: bool = False
    oa_status: Optional[str] = None
    abstract: Optional[str] = None
    authors: List[Dict] = field(default_factory=list)
    institutions: List[Dict] = field(default_factory=list)
    countries: List[str] = field(default_factory=list)
    concepts: List[Dict] = field(default_factory=list)
    topics: List[Dict] = field(default_factory=list)
    keywords: List[Dict] = field(default_factory=list)
    sdgs: List[Dict] = field(default_factory=list)
    source: Optional[Dict] = None
    referenced_works: List[str] = field(default_factory=list)
    related_works: List[str] = field(default_factory=list)
    raw_data: Dict = field(default_factory=dict)
    
    @classmethod
    def from_api_response(cls, data: Dict) -> "OpenAlexWork":
        """Create from API response."""
        # Extract authors
        authors = []
        institutions = []
        countries = set()
        for auth in data.get("authorships", []):
            author_info = auth.get("author", {})
            authors.append({
                "id": author_info.get("id"),
                "name": author_info.get("display_name"),
                "orcid": author_info.get("orcid"),
                "position": auth.get("author_position"),
                "is_corresponding": auth.get("is_corresponding", False),
            })
            for inst in auth.get("institutions", []):
                institutions.append({
                    "id": inst.get("id"),
                    "name": inst.get("display_name"),
                    "country": inst.get("country_code"),
                    "type": inst.get("type"),
                })
            for country in auth.get("countries", []):
                countries.add(country)
        
        # Extract source
        source = None
        primary_loc = data.get("primary_location", {})
        if primary_loc and primary_loc.get("source"):
            src = primary_loc["source"]
            source = {
                "id": src.get("id"),
                "name": src.get("display_name"),
                "issn": src.get("issn_l"),
                "type": src.get("type"),
                "is_oa": src.get("is_oa", False),
            }
        
        # Extract OA info
        oa_info = data.get("open_access", {})
        
        return cls(
            id=data.get("id", ""),
            doi=data.get("doi"),
            title=data.get("title") or data.get("display_name"),
            publication_year=data.get("publication_year"),
            publication_date=data.get("publication_date"),
            type=data.get("type"),
            cited_by_count=data.get("cited_by_count", 0),
            is_oa=oa_info.get("is_oa", False),
            oa_status=oa_info.get("oa_status"),
            abstract=data.get("abstract"),
            authors=authors,
            institutions=institutions,
            countries=list(countries),
            concepts=data.get("concepts", []),
            topics=data.get("topics", []),
            keywords=data.get("keywords", []),
            sdgs=data.get("sustainable_development_goals", []),
            source=source,
            referenced_works=data.get("referenced_works", []),
            related_works=data.get("related_works", []),
            raw_data=data,
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame."""
        return {
            "openalex_id": self.id,
            "doi": self.doi,
            "title": self.title,
            "publication_year": self.publication_year,
            "publication_date": self.publication_date,
            "type": self.type,
            "cited_by_count": self.cited_by_count,
            "is_oa": self.is_oa,
            "oa_status": self.oa_status,
            "abstract": self.abstract,
            "authors": "|".join([a["name"] for a in self.authors if a.get("name")]),
            "author_ids": "|".join([a["id"] for a in self.authors if a.get("id")]),
            "institutions": "|".join(list(set([i["name"] for i in self.institutions if i.get("name")]))),
            "countries": "|".join(self.countries),
            "concepts": "|".join([c.get("display_name", "") for c in self.concepts[:10]]),
            "topics": "|".join([t.get("display_name", "") for t in self.topics[:5]]),
            "keywords": "|".join([k.get("display_name", "") for k in self.keywords]),
            "sdgs": "|".join([s.get("display_name", "") for s in self.sdgs]),
            "source_name": self.source.get("name") if self.source else None,
            "source_issn": self.source.get("issn") if self.source else None,
            "n_references": len(self.referenced_works),
            "n_related": len(self.related_works),
            "n_authors": len(self.authors),
        }


@dataclass
class OpenAlexAuthor:
    """Represents an OpenAlex author."""
    id: str
    name: str
    orcid: Optional[str] = None
    works_count: int = 0
    cited_by_count: int = 0
    h_index: Optional[int] = None
    i10_index: Optional[int] = None
    affiliations: List[Dict] = field(default_factory=list)
    topics: List[Dict] = field(default_factory=list)
    raw_data: Dict = field(default_factory=dict)
    
    @classmethod
    def from_api_response(cls, data: Dict) -> "OpenAlexAuthor":
        """Create from API response."""
        summary = data.get("summary_stats", {})
        
        affiliations = []
        for aff in data.get("affiliations", []):
            inst = aff.get("institution", {})
            affiliations.append({
                "id": inst.get("id"),
                "name": inst.get("display_name"),
                "country": inst.get("country_code"),
                "years": aff.get("years", []),
            })
        
        return cls(
            id=data.get("id", ""),
            name=data.get("display_name", ""),
            orcid=data.get("orcid"),
            works_count=data.get("works_count", 0),
            cited_by_count=data.get("cited_by_count", 0),
            h_index=summary.get("h_index"),
            i10_index=summary.get("i10_index"),
            affiliations=affiliations,
            topics=data.get("topics", [])[:10],
            raw_data=data,
        )


@dataclass  
class OpenAlexInstitution:
    """Represents an OpenAlex institution."""
    id: str
    name: str
    ror: Optional[str] = None
    country: Optional[str] = None
    type: Optional[str] = None
    works_count: int = 0
    cited_by_count: int = 0
    raw_data: Dict = field(default_factory=dict)
    
    @classmethod
    def from_api_response(cls, data: Dict) -> "OpenAlexInstitution":
        """Create from API response."""
        return cls(
            id=data.get("id", ""),
            name=data.get("display_name", ""),
            ror=data.get("ror"),
            country=data.get("country_code"),
            type=data.get("type"),
            works_count=data.get("works_count", 0),
            cited_by_count=data.get("cited_by_count", 0),
            raw_data=data,
        )


# =============================================================================
# MAIN CLIENT CLASS
# =============================================================================

class OpenAlexClient:
    """
    Comprehensive client for the OpenAlex API.
    
    Parameters
    ----------
    email : str, optional
        Your email address. Highly recommended for the "polite pool"
        which has faster rate limits.
    per_page : int
        Default number of results per page (max 200).
    rate_limit_delay : float
        Delay between requests in seconds.
    
    Examples
    --------
    >>> client = OpenAlexClient(email="researcher@university.edu")
    >>> 
    >>> # Search for works
    >>> works = client.search_works("machine learning", max_results=50)
    >>> 
    >>> # Get author by ORCID
    >>> author = client.get_author(orcid="0000-0002-1234-5678")
    >>> 
    >>> # Download dataset
    >>> df = client.download_works(
    ...     query="climate change adaptation",
    ...     filters={"publication_year": "2020-2024", "is_oa": True},
    ...     max_results=500
    ... )
    """
    
    def __init__(
        self,
        email: Optional[str] = None,
        per_page: int = 100,
        rate_limit_delay: float = 0.1,
    ):
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required. Install with: pip install requests")
        
        self.email = email
        self.per_page = min(per_page, 200)  # Max 200 per page
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        
        # Set up session headers
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "Biblium/2.2 (https://github.com/bibliometrics/biblium)",
        })
        
        # Stats
        self._request_count = 0
        self._last_request_time = 0
    
    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        timeout: int = 30,
    ) -> Optional[Dict]:
        """Make a request to the OpenAlex API."""
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        
        url = f"{OPENALEX_BASE_URL}/{endpoint}"
        
        if params is None:
            params = {}
        
        # Add email for polite pool
        if self.email:
            params["mailto"] = self.email
        
        try:
            response = self.session.get(url, params=params, timeout=timeout)
            self._request_count += 1
            self._last_request_time = time.time()
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited - wait and retry
                warnings.warn("Rate limited. Waiting 1 second...")
                time.sleep(1)
                return self._make_request(endpoint, params, timeout)
            else:
                warnings.warn(f"API error {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            warnings.warn(f"Request failed: {e}")
            return None
    
    def _build_filter_string(self, filters: Dict[str, Any]) -> str:
        """Build OpenAlex filter string from dictionary."""
        parts = []
        
        for key, value in filters.items():
            # Map common names to OpenAlex filter names
            filter_name = WORK_FILTERS.get(key, key)
            
            if isinstance(value, bool):
                value = str(value).lower()
            elif isinstance(value, (list, tuple)):
                value = "|".join(str(v) for v in value)
            
            parts.append(f"{filter_name}:{value}")
        
        return ",".join(parts)
    
    # =========================================================================
    # SEARCH METHODS
    # =========================================================================
    
    def search_works(
        self,
        query: str,
        filters: Optional[Dict] = None,
        sort: str = "relevance_score:desc",
        max_results: int = 100,
    ) -> List[OpenAlexWork]:
        """
        Search for works (publications).
        
        Parameters
        ----------
        query : str
            Search query (searches title, abstract).
        filters : dict, optional
            Filters to apply. Common filters:
            - publication_year: "2020-2024" or "2023"
            - type: "article", "review", "book", etc.
            - is_oa: True/False
            - countries: "US|UK|DE"
            - concepts: concept ID
        sort : str
            Sort order. Options: relevance_score, cited_by_count, 
            publication_date, etc. Add :desc or :asc.
        max_results : int
            Maximum number of results to return.
            
        Returns
        -------
        List[OpenAlexWork]
            List of work objects.
        """
        works = []
        cursor = "*"
        
        while len(works) < max_results:
            params = {
                "search": query,
                "per_page": min(self.per_page, max_results - len(works)),
                "sort": sort,
                "cursor": cursor,
            }
            
            if filters:
                params["filter"] = self._build_filter_string(filters)
            
            data = self._make_request("works", params)
            
            if not data or not data.get("results"):
                break
            
            for item in data["results"]:
                works.append(OpenAlexWork.from_api_response(item))
            
            # Get next cursor
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
        
        return works[:max_results]
    
    def search_authors(
        self,
        query: str,
        filters: Optional[Dict] = None,
        max_results: int = 50,
    ) -> List[OpenAlexAuthor]:
        """
        Search for authors.
        
        Parameters
        ----------
        query : str
            Author name to search.
        filters : dict, optional
            Filters like affiliations, works_count, etc.
        max_results : int
            Maximum results.
            
        Returns
        -------
        List[OpenAlexAuthor]
        """
        authors = []
        cursor = "*"
        
        while len(authors) < max_results:
            params = {
                "search": query,
                "per_page": min(self.per_page, max_results - len(authors)),
                "cursor": cursor,
            }
            
            if filters:
                params["filter"] = self._build_filter_string(filters)
            
            data = self._make_request("authors", params)
            
            if not data or not data.get("results"):
                break
            
            for item in data["results"]:
                authors.append(OpenAlexAuthor.from_api_response(item))
            
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
        
        return authors[:max_results]
    
    def search_institutions(
        self,
        query: str,
        filters: Optional[Dict] = None,
        max_results: int = 50,
    ) -> List[OpenAlexInstitution]:
        """
        Search for institutions.
        
        Parameters
        ----------
        query : str
            Institution name to search.
        filters : dict, optional
            Filters like country_code, type, etc.
        max_results : int
            Maximum results.
            
        Returns
        -------
        List[OpenAlexInstitution]
        """
        institutions = []
        cursor = "*"
        
        while len(institutions) < max_results:
            params = {
                "search": query,
                "per_page": min(self.per_page, max_results - len(institutions)),
                "cursor": cursor,
            }
            
            if filters:
                params["filter"] = self._build_filter_string(filters)
            
            data = self._make_request("institutions", params)
            
            if not data or not data.get("results"):
                break
            
            for item in data["results"]:
                institutions.append(OpenAlexInstitution.from_api_response(item))
            
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
        
        return institutions[:max_results]
    
    # =========================================================================
    # GET BY ID METHODS
    # =========================================================================
    
    def get_work(
        self,
        openalex_id: Optional[str] = None,
        doi: Optional[str] = None,
        pmid: Optional[str] = None,
    ) -> Optional[OpenAlexWork]:
        """
        Get a single work by ID.
        
        Parameters
        ----------
        openalex_id : str, optional
            OpenAlex ID (e.g., "W2741809807").
        doi : str, optional
            DOI (e.g., "10.1038/nature12373").
        pmid : str, optional
            PubMed ID.
            
        Returns
        -------
        OpenAlexWork or None
        """
        if openalex_id:
            endpoint = f"works/{openalex_id}"
        elif doi:
            # Clean DOI
            if doi.startswith("https://doi.org/"):
                doi = doi.replace("https://doi.org/", "")
            endpoint = f"works/https://doi.org/{doi}"
        elif pmid:
            endpoint = f"works/pmid:{pmid}"
        else:
            raise ValueError("Must provide openalex_id, doi, or pmid")
        
        data = self._make_request(endpoint)
        
        if data:
            return OpenAlexWork.from_api_response(data)
        return None
    
    def get_author(
        self,
        openalex_id: Optional[str] = None,
        orcid: Optional[str] = None,
    ) -> Optional[OpenAlexAuthor]:
        """
        Get an author by ID.
        
        Parameters
        ----------
        openalex_id : str, optional
            OpenAlex author ID.
        orcid : str, optional
            ORCID (e.g., "0000-0002-1234-5678").
            
        Returns
        -------
        OpenAlexAuthor or None
        """
        if openalex_id:
            endpoint = f"authors/{openalex_id}"
        elif orcid:
            # Clean ORCID
            if orcid.startswith("https://orcid.org/"):
                orcid = orcid.replace("https://orcid.org/", "")
            endpoint = f"authors/https://orcid.org/{orcid}"
        else:
            raise ValueError("Must provide openalex_id or orcid")
        
        data = self._make_request(endpoint)
        
        if data:
            return OpenAlexAuthor.from_api_response(data)
        return None
    
    def get_institution(
        self,
        openalex_id: Optional[str] = None,
        ror: Optional[str] = None,
    ) -> Optional[OpenAlexInstitution]:
        """
        Get an institution by ID.
        
        Parameters
        ----------
        openalex_id : str, optional
            OpenAlex institution ID.
        ror : str, optional
            ROR ID (e.g., "https://ror.org/02y3ad647").
            
        Returns
        -------
        OpenAlexInstitution or None
        """
        if openalex_id:
            endpoint = f"institutions/{openalex_id}"
        elif ror:
            endpoint = f"institutions/{ror}"
        else:
            raise ValueError("Must provide openalex_id or ror")
        
        data = self._make_request(endpoint)
        
        if data:
            return OpenAlexInstitution.from_api_response(data)
        return None
    
    # =========================================================================
    # BULK DOWNLOAD METHODS
    # =========================================================================
    
    def download_works(
        self,
        query: Optional[str] = None,
        filters: Optional[Dict] = None,
        sort: str = "publication_date:desc",
        max_results: int = 1000,
        progress: bool = True,
    ) -> pd.DataFrame:
        """
        Download works as a DataFrame.
        
        Parameters
        ----------
        query : str, optional
            Search query.
        filters : dict, optional
            Filters to apply.
        sort : str
            Sort order.
        max_results : int
            Maximum number of works to download.
        progress : bool
            Show progress.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with work metadata in Biblium-compatible format.
        """
        works = []
        cursor = "*"
        
        if progress:
            print(f"Downloading works from OpenAlex (max {max_results})...")
        
        while len(works) < max_results:
            params = {
                "per_page": min(self.per_page, max_results - len(works)),
                "sort": sort,
                "cursor": cursor,
            }
            
            if query:
                params["search"] = query
            
            if filters:
                params["filter"] = self._build_filter_string(filters)
            
            data = self._make_request("works", params)
            
            if not data or not data.get("results"):
                break
            
            for item in data["results"]:
                work = OpenAlexWork.from_api_response(item)
                works.append(work.to_dict())
            
            if progress:
                print(f"  Downloaded {len(works)} / {max_results}...", end="\r")
            
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
        
        if progress:
            print(f"  Downloaded {len(works)} works.          ")
        
        return pd.DataFrame(works)
    
    def download_author_works(
        self,
        author_id: Optional[str] = None,
        orcid: Optional[str] = None,
        max_results: int = 500,
        progress: bool = True,
    ) -> pd.DataFrame:
        """
        Download all works by a specific author.
        
        Parameters
        ----------
        author_id : str, optional
            OpenAlex author ID.
        orcid : str, optional
            Author ORCID.
        max_results : int
            Maximum works to download.
        progress : bool
            Show progress.
            
        Returns
        -------
        pd.DataFrame
        """
        if orcid:
            if not orcid.startswith("https://orcid.org/"):
                orcid = f"https://orcid.org/{orcid}"
            filter_str = f"author.orcid:{orcid}"
        elif author_id:
            filter_str = f"author.id:{author_id}"
        else:
            raise ValueError("Must provide author_id or orcid")
        
        return self.download_works(
            filters={"filter": filter_str},
            max_results=max_results,
            progress=progress,
        )
    
    def download_institution_works(
        self,
        institution_id: Optional[str] = None,
        ror: Optional[str] = None,
        year_range: Optional[Tuple[int, int]] = None,
        max_results: int = 1000,
        progress: bool = True,
    ) -> pd.DataFrame:
        """
        Download all works from a specific institution.
        
        Parameters
        ----------
        institution_id : str, optional
            OpenAlex institution ID.
        ror : str, optional
            ROR ID.
        year_range : tuple, optional
            (start_year, end_year).
        max_results : int
            Maximum works.
        progress : bool
            Show progress.
            
        Returns
        -------
        pd.DataFrame
        """
        filters = {}
        
        if ror:
            filters["institutions"] = ror
        elif institution_id:
            filters["institutions"] = institution_id
        else:
            raise ValueError("Must provide institution_id or ror")
        
        if year_range:
            filters["publication_year"] = f"{year_range[0]}-{year_range[1]}"
        
        return self.download_works(
            filters=filters,
            max_results=max_results,
            progress=progress,
        )
    
    # =========================================================================
    # AGGREGATION METHODS
    # =========================================================================
    
    def get_work_counts_by_year(
        self,
        query: Optional[str] = None,
        filters: Optional[Dict] = None,
        year_range: Tuple[int, int] = (2000, 2024),
    ) -> pd.DataFrame:
        """
        Get publication counts by year.
        
        Parameters
        ----------
        query : str, optional
            Search query to filter works.
        filters : dict, optional
            Additional filters.
        year_range : tuple
            (start_year, end_year).
            
        Returns
        -------
        pd.DataFrame
            DataFrame with Year and Count columns.
        """
        params = {
            "group_by": "publication_year",
            "per_page": 200,
        }
        
        filter_parts = [f"publication_year:{year_range[0]}-{year_range[1]}"]
        
        if query:
            params["search"] = query
        
        if filters:
            filter_parts.append(self._build_filter_string(filters))
        
        params["filter"] = ",".join(filter_parts)
        
        data = self._make_request("works", params)
        
        if not data or not data.get("group_by"):
            return pd.DataFrame(columns=["Year", "Count"])
        
        results = []
        for item in data["group_by"]:
            results.append({
                "Year": int(item["key"]),
                "Count": item["count"],
            })
        
        df = pd.DataFrame(results).sort_values("Year")
        return df
    
    def get_work_counts_by_country(
        self,
        query: Optional[str] = None,
        filters: Optional[Dict] = None,
        top_n: int = 50,
    ) -> pd.DataFrame:
        """
        Get publication counts by country.
        
        Parameters
        ----------
        query : str, optional
            Search query.
        filters : dict, optional
            Additional filters.
        top_n : int
            Number of top countries.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with Country and Count columns.
        """
        params = {
            "group_by": "authorships.countries",
            "per_page": top_n,
        }
        
        if query:
            params["search"] = query
        
        if filters:
            params["filter"] = self._build_filter_string(filters)
        
        data = self._make_request("works", params)
        
        if not data or not data.get("group_by"):
            return pd.DataFrame(columns=["Country", "Count"])
        
        results = []
        for item in data["group_by"]:
            results.append({
                "Country": item.get("key_display_name", item["key"]),
                "Country_Code": item["key"],
                "Count": item["count"],
            })
        
        return pd.DataFrame(results)
    
    def get_work_counts_by_sdg(
        self,
        query: Optional[str] = None,
        filters: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Get publication counts by Sustainable Development Goal.
        
        Parameters
        ----------
        query : str, optional
            Search query.
        filters : dict, optional
            Additional filters.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with SDG, SDG_Name, and Count columns.
        """
        params = {
            "group_by": "sustainable_development_goals.id",
            "per_page": 20,
        }
        
        if query:
            params["search"] = query
        
        if filters:
            params["filter"] = self._build_filter_string(filters)
        
        data = self._make_request("works", params)
        
        if not data or not data.get("group_by"):
            return pd.DataFrame(columns=["SDG", "SDG_Name", "Count"])
        
        results = []
        for item in data["group_by"]:
            sdg_id = item["key"]
            # Extract SDG number from URL
            try:
                sdg_num = int(sdg_id.split("/")[-1])
                sdg_name = SDG_NAMES.get(sdg_num, f"SDG {sdg_num}")
            except:
                sdg_num = sdg_id
                sdg_name = item.get("key_display_name", sdg_id)
            
            results.append({
                "SDG": sdg_num,
                "SDG_Name": sdg_name,
                "Count": item["count"],
            })
        
        return pd.DataFrame(results).sort_values("SDG")
    
    def get_work_counts_by_type(
        self,
        query: Optional[str] = None,
        filters: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Get publication counts by document type.
        
        Parameters
        ----------
        query : str, optional
            Search query.
        filters : dict, optional
            Additional filters.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with Type and Count columns.
        """
        params = {
            "group_by": "type",
            "per_page": 50,
        }
        
        if query:
            params["search"] = query
        
        if filters:
            params["filter"] = self._build_filter_string(filters)
        
        data = self._make_request("works", params)
        
        if not data or not data.get("group_by"):
            return pd.DataFrame(columns=["Type", "Count"])
        
        results = []
        for item in data["group_by"]:
            results.append({
                "Type": item.get("key_display_name", item["key"]),
                "Count": item["count"],
            })
        
        return pd.DataFrame(results).sort_values("Count", ascending=False)
    
    def get_work_counts_by_source(
        self,
        query: Optional[str] = None,
        filters: Optional[Dict] = None,
        top_n: int = 50,
    ) -> pd.DataFrame:
        """
        Get publication counts by source (journal/venue).
        
        Parameters
        ----------
        query : str, optional
            Search query.
        filters : dict, optional
            Additional filters.
        top_n : int
            Number of top sources.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with Source, Source_ID, and Count columns.
        """
        params = {
            "group_by": "primary_location.source.id",
            "per_page": top_n,
        }
        
        if query:
            params["search"] = query
        
        if filters:
            params["filter"] = self._build_filter_string(filters)
        
        data = self._make_request("works", params)
        
        if not data or not data.get("group_by"):
            return pd.DataFrame(columns=["Source", "Source_ID", "Count"])
        
        results = []
        for item in data["group_by"]:
            results.append({
                "Source": item.get("key_display_name", item["key"]),
                "Source_ID": item["key"],
                "Count": item["count"],
            })
        
        return pd.DataFrame(results)
    
    # =========================================================================
    # ENRICHMENT METHODS
    # =========================================================================
    
    def enrich_dataframe(
        self,
        df: pd.DataFrame,
        doi_column: str = "DOI",
        fields: Optional[List[str]] = None,
        progress: bool = True,
        delay: float = 0.1,
    ) -> pd.DataFrame:
        """
        Enrich a DataFrame with OpenAlex metadata.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with DOIs.
        doi_column : str
            Name of the DOI column.
        fields : list, optional
            Fields to add. Default: ["cited_by_count", "is_oa", "concepts", "sdgs"].
        progress : bool
            Show progress.
        delay : float
            Delay between requests.
            
        Returns
        -------
        pd.DataFrame
            Enriched DataFrame with new columns.
        """
        if fields is None:
            fields = ["cited_by_count", "is_oa", "oa_status", "concepts", "sdgs", "topics"]
        
        df = df.copy()
        
        # Initialize new columns
        for field in fields:
            if field not in df.columns:
                df[f"oa_{field}"] = None
        
        n_total = len(df)
        n_enriched = 0
        
        if progress:
            print(f"Enriching {n_total} records from OpenAlex...")
        
        for idx, row in df.iterrows():
            doi = row.get(doi_column)
            
            if pd.isna(doi) or not doi:
                continue
            
            work = self.get_work(doi=str(doi))
            
            if work:
                n_enriched += 1
                
                if "cited_by_count" in fields:
                    df.at[idx, "oa_cited_by_count"] = work.cited_by_count
                if "is_oa" in fields:
                    df.at[idx, "oa_is_oa"] = work.is_oa
                if "oa_status" in fields:
                    df.at[idx, "oa_oa_status"] = work.oa_status
                if "concepts" in fields:
                    df.at[idx, "oa_concepts"] = "|".join([c.get("display_name", "") for c in work.concepts[:5]])
                if "sdgs" in fields:
                    df.at[idx, "oa_sdgs"] = "|".join([s.get("display_name", "") for s in work.sdgs])
                if "topics" in fields:
                    df.at[idx, "oa_topics"] = "|".join([t.get("display_name", "") for t in work.topics[:3]])
            
            if progress and (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{n_total} ({n_enriched} enriched)...", end="\r")
            
            time.sleep(delay)
        
        if progress:
            print(f"  Enriched {n_enriched}/{n_total} records.          ")
        
        return df
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_random_works(self, n: int = 10) -> List[OpenAlexWork]:
        """Get random works (useful for sampling)."""
        params = {
            "sample": n,
            "per_page": n,
        }
        
        data = self._make_request("works", params)
        
        if not data or not data.get("results"):
            return []
        
        return [OpenAlexWork.from_api_response(item) for item in data["results"]]
    
    def get_stats(self) -> Dict:
        """Get API usage statistics."""
        return {
            "requests_made": self._request_count,
            "email_configured": self.email is not None,
            "rate_limit_delay": self.rate_limit_delay,
        }
    
    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            data = self._make_request("works", {"per_page": 1})
            return data is not None
        except:
            return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def search_openalex(
    query: str,
    entity_type: ENTITY_TYPES = "works",
    max_results: int = 100,
    email: Optional[str] = None,
    **filters,
) -> pd.DataFrame:
    """
    Quick search function for OpenAlex.
    
    Parameters
    ----------
    query : str
        Search query.
    entity_type : str
        Type of entity: "works", "authors", "institutions", etc.
    max_results : int
        Maximum results.
    email : str, optional
        Your email for polite pool.
    **filters
        Additional filters.
        
    Returns
    -------
    pd.DataFrame
    """
    client = OpenAlexClient(email=email)
    
    if entity_type == "works":
        works = client.search_works(query, filters=filters, max_results=max_results)
        return pd.DataFrame([w.to_dict() for w in works])
    elif entity_type == "authors":
        authors = client.search_authors(query, filters=filters, max_results=max_results)
        return pd.DataFrame([{
            "id": a.id,
            "name": a.name,
            "orcid": a.orcid,
            "works_count": a.works_count,
            "cited_by_count": a.cited_by_count,
            "h_index": a.h_index,
        } for a in authors])
    elif entity_type == "institutions":
        institutions = client.search_institutions(query, filters=filters, max_results=max_results)
        return pd.DataFrame([{
            "id": i.id,
            "name": i.name,
            "country": i.country,
            "type": i.type,
            "works_count": i.works_count,
        } for i in institutions])
    else:
        raise ValueError(f"Unknown entity type: {entity_type}")


def download_openalex_dataset(
    query: str,
    output_path: Optional[str] = None,
    max_results: int = 1000,
    email: Optional[str] = None,
    **filters,
) -> pd.DataFrame:
    """
    Download a dataset from OpenAlex.
    
    Parameters
    ----------
    query : str
        Search query.
    output_path : str, optional
        Path to save CSV.
    max_results : int
        Maximum results.
    email : str, optional
        Your email for polite pool.
    **filters
        Additional filters.
        
    Returns
    -------
    pd.DataFrame
    """
    client = OpenAlexClient(email=email)
    df = client.download_works(query=query, filters=filters, max_results=max_results)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} records to {output_path}")
    
    return df
