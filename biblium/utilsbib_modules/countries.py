# -*- coding: utf-8 -*-
"""
Country utilities - country name standardization, mappings, and extraction.

This module contains:
- Country name mappings and lookups
- ISO codes and continent mappings
- Corresponding author country extraction
- Affiliation parsing
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

# =============================================================================
# LOAD COUNTRY DATA
# =============================================================================

_fd = os.path.dirname(os.path.dirname(__file__))  # biblium package directory
_countries_path = os.path.join(_fd, "additional files", "countries.xlsx")

# Load country reference data
df_countries = pd.read_excel(_countries_path)

# Build lookup dictionaries
domain_dct = df_countries.set_index("Internet domain").to_dict()["Name"]
c_off_dct = df_countries.set_index("Official name").to_dict()["Name"]
code_dct = df_countries.set_index("Name").to_dict()["Code"]
code_dct_r = df_countries.set_index("Code").to_dict()["Name"]
country_iso3_dct = df_countries.set_index("Name").to_dict()["ISO-3"]
continent_dct = df_countries.set_index("Name").to_dict()["Continent"]

# Coordinate lookup
df_countries_un_iso = df_countries.drop_duplicates(subset="ISO-3")
code_to_coords = df_countries_un_iso[["ISO-3", "latitude", "longitude"]].set_index("ISO-3")[["latitude", "longitude"]].to_dict(orient="index")

# Country lists
l_countries = list(df_countries["Name"])
eu_countries = list(df_countries[df_countries["EU"] == 1]["Name"])


# =============================================================================
# COUNTRY NAME CORRECTION
# =============================================================================

def correct_country_name(s: str) -> str:
    """
    Return the corrected country name based on known lists and mappings.

    Parameters
    ----------
    s : str
        Input country name.

    Returns
    -------
    str
        Corrected country name if recognized, empty string otherwise.
    """
    if not isinstance(s, str):
        return ""
    if s in l_countries:
        return s
    return c_off_dct.get(s, "")


# =============================================================================
# CORRESPONDING AUTHOR PARSING
# =============================================================================

def split_ca(s: str) -> Tuple:
    """
    Split a Scopus corresponding author string into name, affiliation, and country.

    Parameters
    ----------
    s : str
        Raw Scopus corresponding author string.

    Returns
    -------
    tuple
        (corresponding author, affiliation, country) or (np.nan, np.nan, np.nan) if parsing fails.
    """
    try:
        ca, long_aff = s.split("; ", 1)
        parts = long_aff.split(", ")
        return ca, parts[0], parts[-1]
    except Exception:
        return np.nan, np.nan, np.nan


def parse_mail(s: str) -> Union[str, float]:
    """
    Attempt to extract the country based on the email domain.

    Parameters
    ----------
    s : str
        Full string that may contain an email.

    Returns
    -------
    str or np.nan
        Country inferred from email domain or np.nan if not found.
    """
    if "@" in s:
        domain = s.split("@")[1].split(" ")[0].split(".")[-1]
        return domain_dct.get(domain, np.nan)
    return np.nan


def get_ca_country_scopus(
    s: str,
    countries_list: Optional[List[str]] = None
) -> str:
    """
    Extract the country of the corresponding author from a Scopus entry.

    Parameters
    ----------
    s : str
        Scopus corresponding author string.
    countries_list : list, optional
        List of recognized country names. Uses default if not provided.

    Returns
    -------
    str
        Extracted country name or empty string.
    """
    if countries_list is None:
        countries_list = l_countries
        
    if not isinstance(s, str):
        return ""
    
    # Try to split the string
    ca, aff, country = split_ca(s)
    
    # Check if country is valid
    corrected = correct_country_name(country)
    if corrected:
        return corrected
    
    # Try parsing from email
    mail_country = parse_mail(s)
    if pd.notna(mail_country):
        return mail_country
    
    return ""


def get_ca_country_wos(s: str) -> str:
    """
    Extract the country from a Web of Science corresponding author field.

    Parameters
    ----------
    s : str
        WoS corresponding author string.

    Returns
    -------
    str
        Extracted country name or empty string.
    """
    if not isinstance(s, str):
        return ""
    
    # WoS format often has country at the end after a comma
    parts = s.split(", ")
    if parts:
        country = parts[-1].strip()
        corrected = correct_country_name(country)
        if corrected:
            return corrected
    
    return ""


def get_ca_country(s: str, db: str = "scopus") -> str:
    """
    Extract corresponding author country based on database type.

    Parameters
    ----------
    s : str
        Corresponding author string.
    db : str
        Database type ("scopus", "wos", etc.).

    Returns
    -------
    str
        Extracted country name.
    """
    db = db.lower()
    if db == "scopus":
        return get_ca_country_scopus(s)
    elif db in ["wos", "web of science"]:
        return get_ca_country_wos(s)
    return ""


def add_ca_country_df(
    df: pd.DataFrame,
    db: str = "scopus",
    ca_col: str = "Correspondence Address"
) -> pd.DataFrame:
    """
    Add a 'CA Country' column to the DataFrame.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    db : str
        Database type.
    ca_col : str
        Column containing correspondence address.

    Returns
    -------
    DataFrame
        DataFrame with 'CA Country' column added.
    """
    df = df.copy()
    
    if ca_col in df.columns:
        df["CA Country"] = df[ca_col].apply(lambda x: get_ca_country(x, db))
    else:
        df["CA Country"] = ""
    
    return df


# =============================================================================
# AFFILIATION PARSING
# =============================================================================

def extract_countries_from_affiliations(
    df: pd.DataFrame,
    aff_column: str = "Affiliations",
    sep: str = "; "
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract countries from affiliation strings and build collaboration matrix.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    aff_column : str
        Column containing affiliations.
    sep : str
        Separator between affiliations.

    Returns
    -------
    tuple
        (DataFrame with country columns, collaboration matrix DataFrame)
    """
    df = df.copy()
    
    if aff_column not in df.columns:
        return df, pd.DataFrame()
    
    # Extract countries from each affiliation
    def extract_countries(aff_str):
        if not isinstance(aff_str, str):
            return []
        
        countries = []
        for aff in aff_str.split(sep):
            # Country is typically the last part after comma
            parts = aff.split(", ")
            if parts:
                country = correct_country_name(parts[-1].strip())
                if country:
                    countries.append(country)
        
        return list(set(countries))  # Unique countries
    
    df["Countries"] = df[aff_column].apply(extract_countries)
    df["N Countries"] = df["Countries"].apply(len)
    
    # Build collaboration matrix
    all_countries = set()
    for countries in df["Countries"]:
        all_countries.update(countries)
    
    all_countries = sorted(all_countries)
    
    # Create binary indicators
    collab_matrix = pd.DataFrame(index=df.index, columns=all_countries)
    for country in all_countries:
        collab_matrix[country] = df["Countries"].apply(lambda x: 1 if country in x else 0)
    
    collab_matrix = collab_matrix.fillna(0).astype(int)
    
    return df, collab_matrix


# =============================================================================
# OPENALEX COUNTRY UTILITIES
# =============================================================================

def openalex_map_country_codes(
    df: pd.DataFrame,
    code_column: str = "Country Code"
) -> pd.DataFrame:
    """
    Map OpenAlex country codes to country names.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    code_column : str
        Column containing country codes.

    Returns
    -------
    DataFrame
        DataFrame with country names added.
    """
    df = df.copy()
    
    if code_column in df.columns:
        df["Country"] = df[code_column].map(code_dct_r)
    
    return df


def openalex_add_corresponding_country(
    df: pd.DataFrame,
    authors_col: str = "Authors"
) -> pd.DataFrame:
    """
    Add corresponding author country for OpenAlex data.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    authors_col : str
        Column containing author information.

    Returns
    -------
    DataFrame
        DataFrame with 'CA Country' column.
    """
    df = df.copy()
    
    # OpenAlex may have country info in author affiliations
    # This is a placeholder - actual implementation depends on data format
    if "CA Country" not in df.columns:
        df["CA Country"] = ""
    
    return df
