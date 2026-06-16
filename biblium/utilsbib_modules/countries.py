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
from typing import List, Optional, Tuple, Union
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

# ============================================================================
# Country display names (ISO2 → preferred name) + diagnostic functions
# ============================================================================

# Default display overrides — ISO 3166-1 alpha-2 → preferred name
# (uporabnik lahko razširi prek parametra)
DEFAULT_COUNTRY_DISPLAY: dict[str, str] = {
    "GB": "UK",          # United Kingdom (kolokvialno UK je razumljivejši kot GB)
    "US": "US",
    # po želji dodaj druge mappinge
}


def country_iso2_to_display(iso2: str,
                              overrides: dict[str, str] | None = None) -> str:
    """
    Pretvori ISO2 v prikazno ime, ki si ga uporabnik želi.

    Parameters
    ----------
    iso2 : str
        ISO 3166-1 alpha-2 koda (npr. "GB", "MN", "US").
    overrides : dict, optional
        Custom mapping (npr. {"GB": "UK"}). Spojen z DEFAULT_COUNTRY_DISPLAY.

    Returns
    -------
    str
        Prikazna oznaka — bodisi iz overrides bodisi originalni ISO2.
    """
    if not isinstance(iso2, str) or not iso2.strip():
        return iso2
    iso2 = iso2.strip().upper()
    omap = dict(DEFAULT_COUNTRY_DISPLAY)
    if overrides:
        omap.update({k.upper(): v for k, v in overrides.items()})
    return omap.get(iso2, iso2)


def map_country_codes_to_display(series: "pd.Series",
                                    overrides: dict[str, str] | None = None,
                                    sep: str | None = None) -> "pd.Series":
    """
    Mapira pd.Series ISO2 kod v prikazne oznake. Če je vsaka vrednost
    seznam (sep podan), se mappira vsak token posebej.
    """
    import pandas as pd
    if sep is None:
        return series.map(lambda v: country_iso2_to_display(v, overrides))
    def _map_list(s):
        if not isinstance(s, str):
            return s
        items = [country_iso2_to_display(t.strip(), overrides)
                 for t in s.split(sep) if t.strip()]
        return sep.join(items)
    return series.map(_map_list)


def diagnose_country_codes_in_corpus(
    df: "pd.DataFrame",
    col: str = "oa_institution_countries",
    sep: str = "; ",
    institution_col: str = "oa_institutions",
    institution_sep: str = "; ",
    top_n: int = 25,
    suspicious_threshold: int = 20,
    verbose: bool = True,
) -> "pd.DataFrame":
    """
    Diagnostika ISO2 country kod v korpusu — za sanity check.

    Za vsak ISO2 najde top 5 institucij, ki tej kodi pripadajo. To omogoči
    preverbo "ali je MN res Mongolia ali OpenAlex artefakt (npr. Minnesota)".

    Parameters
    ----------
    df : DataFrame
        Mora vsebovati `col` (country ISO2 kode, sep-separated) in opcijsko
        `institution_col` (display imena institucij, paralelno).
    col : str
        Stolpec z državami.
    sep : str
        Separator v `col`.
    institution_col : str
        Stolpec z institucijami.
    institution_sep : str
        Separator v `institution_col`.
    top_n : int
        Koliko top ISO2 razdrobi.
    suspicious_threshold : int
        ISO2 z manj kot toliko zapisov se označi kot "sumljiv".
    verbose : bool
        Izpiši top N.

    Returns
    -------
    DataFrame
        Vrstice = ISO2; stolpci: iso2, display, n_documents,
        top_institutions (semicolon-joined), is_suspicious.
    """
    import pandas as pd
    from collections import Counter, defaultdict

    if col not in df.columns:
        if verbose:
            print(f"  ! stolpec {col} ni v df; preskačem")
        return pd.DataFrame()

    has_inst = institution_col in df.columns

    iso_counter = Counter()
    iso_to_insts: dict[str, Counter] = defaultdict(Counter)

    for idx, row in df.iterrows():
        cs = row.get(col)
        if not isinstance(cs, str) or not cs.strip():
            continue
        iso_list = [c.strip() for c in cs.split(sep) if c.strip()]
        for c in iso_list:
            iso_counter[c] += 1
        if has_inst:
            inst_str = row.get(institution_col) or ""
            if isinstance(inst_str, str):
                inst_list = [i.strip() for i in inst_str.split(institution_sep) if i.strip()]
                # naključno: poveži vse institucije z vsemi državami v vrstici
                # (heuristika — eksaktna povezava bi rabila OpenAlex authorships)
                for c in iso_list:
                    for inst in inst_list:
                        iso_to_insts[c][inst] += 1

    rows = []
    for iso, n in iso_counter.most_common():
        top_insts = ""
        if has_inst:
            top_insts = "; ".join(
                f"{name} ({cnt})" for name, cnt in iso_to_insts[iso].most_common(5)
            )
        rows.append({
            "iso2": iso,
            "display": country_iso2_to_display(iso),
            "n_documents": n,
            "top_institutions": top_insts,
            "is_suspicious": n < suspicious_threshold,
        })
    result = pd.DataFrame(rows)

    if verbose and not result.empty:
        print(f"\n  Top {min(top_n, len(result))} držav po številu zapisov "
              f"(skupaj {len(result)} unikatnih ISO2):")
        for _, r in result.head(top_n).iterrows():
            iso = r["iso2"]; disp = r["display"]; n = r["n_documents"]
            print(f"    {iso:>3} -> {disp:<8} {n:>6}")
            if r["top_institutions"]:
                inst = r["top_institutions"][:200]
                print(f"        top institucije: {inst}")

    return result

