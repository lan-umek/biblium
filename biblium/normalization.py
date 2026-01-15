# -*- coding: utf-8 -*-
"""
Country Normalization Module for Biblium

Fetch population and researcher data from World Bank API
to normalize bibliometric country statistics.

Indicators used:
- SP.POP.TOTL: Total population
- SP.POP.SCIE.RD.P6: Researchers in R&D (per million people)

@author: Lan.Umek
"""

import pandas as pd
import numpy as np
import requests
from typing import Optional, Dict, List, Tuple
import warnings
from functools import lru_cache


# =============================================================================
# WORLD BANK API FUNCTIONS
# =============================================================================

# World Bank indicator codes
INDICATORS = {
    "population": "SP.POP.TOTL",           # Total population
    "researchers_per_million": "SP.POP.SCIE.RD.P6",  # Researchers per million people
    "rd_expenditure_pct_gdp": "GB.XPD.RSDV.GD.ZS",   # R&D expenditure (% of GDP)
    "gdp": "NY.GDP.MKTP.CD",               # GDP (current US$)
}

# Country name mapping (bibliometric databases → World Bank ISO codes)
COUNTRY_NAME_TO_ISO = {
    # Common variations
    "United States": "USA",
    "United States of America": "USA",
    "USA": "USA",
    "US": "USA",
    "United Kingdom": "GBR",
    "UK": "GBR",
    "Great Britain": "GBR",
    "England": "GBR",
    "China": "CHN",
    "People's Republic of China": "CHN",
    "Peoples Republic of China": "CHN",
    "Hong Kong": "HKG",
    "Taiwan": "TWN",
    "South Korea": "KOR",
    "Korea": "KOR",
    "Republic of Korea": "KOR",
    "North Korea": "PRK",
    "Russia": "RUS",
    "Russian Federation": "RUS",
    "Germany": "DEU",
    "France": "FRA",
    "Italy": "ITA",
    "Spain": "ESP",
    "Japan": "JPN",
    "Canada": "CAN",
    "Australia": "AUS",
    "Brazil": "BRA",
    "India": "IND",
    "Netherlands": "NLD",
    "The Netherlands": "NLD",
    "Switzerland": "CHE",
    "Sweden": "SWE",
    "Poland": "POL",
    "Belgium": "BEL",
    "Austria": "AUT",
    "Denmark": "DNK",
    "Norway": "NOR",
    "Finland": "FIN",
    "Ireland": "IRL",
    "Portugal": "PRT",
    "Greece": "GRC",
    "Czech Republic": "CZE",
    "Czechia": "CZE",
    "Hungary": "HUN",
    "Romania": "ROU",
    "Turkey": "TUR",
    "Türkiye": "TUR",
    "Mexico": "MEX",
    "Argentina": "ARG",
    "Chile": "CHL",
    "Colombia": "COL",
    "South Africa": "ZAF",
    "Egypt": "EGY",
    "Saudi Arabia": "SAU",
    "Israel": "ISR",
    "Iran": "IRN",
    "Islamic Republic of Iran": "IRN",
    "Pakistan": "PAK",
    "Indonesia": "IDN",
    "Malaysia": "MYS",
    "Singapore": "SGP",
    "Thailand": "THA",
    "Vietnam": "VNM",
    "Viet Nam": "VNM",
    "Philippines": "PHL",
    "New Zealand": "NZL",
    "Slovenia": "SVN",
    "Slovakia": "SVK",
    "Slovak Republic": "SVK",
    "Croatia": "HRV",
    "Serbia": "SRB",
    "Bulgaria": "BGR",
    "Ukraine": "UKR",
    "Nigeria": "NGA",
    "Kenya": "KEN",
    "Morocco": "MAR",
    "Tunisia": "TUN",
    "United Arab Emirates": "ARE",
    "UAE": "ARE",
    "Qatar": "QAT",
    "Kuwait": "KWT",
    "Bangladesh": "BGD",
    "Sri Lanka": "LKA",
    "Nepal": "NPL",
    "Luxembourg": "LUX",
    "Iceland": "ISL",
    "Estonia": "EST",
    "Latvia": "LVA",
    "Lithuania": "LTU",
    "Cyprus": "CYP",
    "Malta": "MLT",
}


def get_iso_code(country_name: str) -> Optional[str]:
    """
    Convert country name to ISO 3166-1 alpha-3 code.
    
    Parameters
    ----------
    country_name : str
        Country name as it appears in bibliometric data.
    
    Returns
    -------
    str or None
        ISO 3166-1 alpha-3 code, or None if not found.
    """
    if pd.isna(country_name):
        return None
    
    name = str(country_name).strip()
    
    # Direct match
    if name in COUNTRY_NAME_TO_ISO:
        return COUNTRY_NAME_TO_ISO[name]
    
    # Case-insensitive match
    name_lower = name.lower()
    for key, code in COUNTRY_NAME_TO_ISO.items():
        if key.lower() == name_lower:
            return code
    
    # If it's already a 3-letter code, return it
    if len(name) == 3 and name.isupper():
        return name
    
    return None


@lru_cache(maxsize=100)
def fetch_world_bank_indicator(
    indicator: str,
    countries: str = "all",
    year: int = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch indicator data from World Bank API.
    
    Parameters
    ----------
    indicator : str
        World Bank indicator code (e.g., "SP.POP.TOTL").
    countries : str
        Country codes (semicolon-separated) or "all".
    year : int, optional
        Specific year. If None, gets most recent available.
    timeout : int
        Request timeout in seconds.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Country', 'ISO', 'Year', 'Value'].
    """
    base_url = "https://api.worldbank.org/v2/country"
    
    # Build URL
    if year:
        date_param = f"date={year}"
    else:
        # Get last 5 years to find most recent data
        date_param = "date=2018:2023"
    
    url = f"{base_url}/{countries}/indicator/{indicator}?{date_param}&format=json&per_page=500"
    
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        
        # World Bank returns [metadata, data]
        if len(data) < 2 or data[1] is None:
            return pd.DataFrame(columns=["Country", "ISO", "Year", "Value"])
        
        records = []
        for item in data[1]:
            if item.get("value") is not None:
                records.append({
                    "Country": item.get("country", {}).get("value", "Unknown"),
                    "ISO": item.get("countryiso3code", ""),
                    "Year": int(item.get("date", 0)),
                    "Value": float(item.get("value", 0)),
                })
        
        df = pd.DataFrame(records)
        
        # Keep only most recent value per country
        if len(df) > 0:
            df = df.sort_values("Year", ascending=False).drop_duplicates("ISO").reset_index(drop=True)
        
        return df
        
    except Exception as e:
        warnings.warn(f"Failed to fetch World Bank data for {indicator}: {e}")
        return pd.DataFrame(columns=["Country", "ISO", "Year", "Value"])


def fetch_population_data(year: int = None) -> pd.DataFrame:
    """
    Fetch population data from World Bank.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Country', 'ISO', 'Year', 'Population'].
    """
    df = fetch_world_bank_indicator(INDICATORS["population"], year=year)
    if len(df) > 0:
        df = df.rename(columns={"Value": "Population"})
    return df


def fetch_researcher_data(year: int = None) -> pd.DataFrame:
    """
    Fetch researchers per million people from World Bank.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Country', 'ISO', 'Year', 'Researchers_per_Million'].
    """
    df = fetch_world_bank_indicator(INDICATORS["researchers_per_million"], year=year)
    if len(df) > 0:
        df = df.rename(columns={"Value": "Researchers_per_Million"})
    return df


def fetch_normalization_data(year: int = None) -> pd.DataFrame:
    """
    Fetch both population and researcher data, merged.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with population and researcher counts per country.
    """
    print("Fetching population data from World Bank...")
    pop_df = fetch_population_data(year)
    
    print("Fetching researcher data from World Bank...")
    res_df = fetch_researcher_data(year)
    
    if len(pop_df) == 0:
        warnings.warn("Could not fetch population data from World Bank")
        return pd.DataFrame()
    
    # Merge
    merged = pop_df[["Country", "ISO", "Population"]].copy()
    
    if len(res_df) > 0:
        merged = pd.merge(
            merged,
            res_df[["ISO", "Researchers_per_Million"]],
            on="ISO",
            how="left",
        )
        # Calculate total researchers
        merged["Total_Researchers"] = (
            merged["Population"] / 1_000_000 * merged["Researchers_per_Million"]
        ).fillna(0).astype(int)
    else:
        merged["Researchers_per_Million"] = np.nan
        merged["Total_Researchers"] = np.nan
    
    print(f"Fetched data for {len(merged)} countries")
    
    return merged


# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_country_counts(
    country_df: pd.DataFrame,
    country_col: str = "Country",
    count_col: str = "Count",
    normalize_by: str = "population",
    normalization_data: pd.DataFrame = None,
    per_million: bool = True,
) -> pd.DataFrame:
    """
    Normalize country publication counts by population or researchers.
    
    Parameters
    ----------
    country_df : pd.DataFrame
        DataFrame with country counts.
    country_col : str
        Name of country column.
    count_col : str
        Name of count column.
    normalize_by : str
        "population", "researchers", or "both".
    normalization_data : pd.DataFrame, optional
        Pre-fetched normalization data. If None, fetches from World Bank.
    per_million : bool
        If True, express as "per million". Otherwise raw ratio.
    
    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional normalized columns.
    """
    result = country_df.copy()
    
    # Fetch normalization data if not provided
    if normalization_data is None:
        normalization_data = fetch_normalization_data()
    
    if len(normalization_data) == 0:
        warnings.warn("No normalization data available. Returning original counts.")
        return result
    
    # Map country names to ISO codes
    result["_ISO"] = result[country_col].apply(get_iso_code)
    
    # Merge with normalization data
    result = pd.merge(
        result,
        normalization_data[["ISO", "Population", "Total_Researchers", "Researchers_per_Million"]],
        left_on="_ISO",
        right_on="ISO",
        how="left",
    )
    
    # Calculate normalized values
    multiplier = 1_000_000 if per_million else 1
    suffix = " per Million" if per_million else " (normalized)"
    
    # Get the actual count values
    count_values = result[count_col].values
    
    if normalize_by in ["population", "both"]:
        result[f"Per Million Pop"] = np.where(
            result["Population"] > 0,
            count_values / result["Population"].values * multiplier,
            np.nan
        )
    
    if normalize_by in ["researchers", "both"]:
        result[f"Per 1000 Researchers"] = np.where(
            result["Total_Researchers"] > 0,
            count_values / result["Total_Researchers"].values * 1000,
            np.nan
        )
    
    # Clean up
    result = result.drop(columns=["_ISO", "ISO"], errors="ignore")
    
    # Report missing
    missing_iso = result[country_col][result["Population"].isna()].tolist()
    if missing_iso:
        print(f"Note: No normalization data for {len(missing_iso)} countries: {missing_iso[:5]}{'...' if len(missing_iso) > 5 else ''}")
    
    return result


# =============================================================================
# FALLBACK DATA (when API unavailable)
# =============================================================================

# Approximate 2022 data for major research countries
FALLBACK_COUNTRY_DATA = {
    "USA": {"Population": 331900000, "Researchers_per_Million": 4412},
    "CHN": {"Population": 1412000000, "Researchers_per_Million": 1585},
    "GBR": {"Population": 67330000, "Researchers_per_Million": 4603},
    "DEU": {"Population": 83200000, "Researchers_per_Million": 5234},
    "JPN": {"Population": 125700000, "Researchers_per_Million": 5331},
    "FRA": {"Population": 67750000, "Researchers_per_Million": 4782},
    "IND": {"Population": 1417000000, "Researchers_per_Million": 253},
    "ITA": {"Population": 59110000, "Researchers_per_Million": 2407},
    "CAN": {"Population": 38930000, "Researchers_per_Million": 4648},
    "AUS": {"Population": 25690000, "Researchers_per_Million": 5041},
    "KOR": {"Population": 51740000, "Researchers_per_Million": 8714},
    "ESP": {"Population": 47420000, "Researchers_per_Million": 3050},
    "BRA": {"Population": 215300000, "Researchers_per_Million": 888},
    "NLD": {"Population": 17530000, "Researchers_per_Million": 5713},
    "RUS": {"Population": 144100000, "Researchers_per_Million": 2784},
    "CHE": {"Population": 8740000, "Researchers_per_Million": 5606},
    "SWE": {"Population": 10490000, "Researchers_per_Million": 8032},
    "POL": {"Population": 37750000, "Researchers_per_Million": 3567},
    "BEL": {"Population": 11590000, "Researchers_per_Million": 5588},
    "AUT": {"Population": 8980000, "Researchers_per_Million": 5649},
    "DNK": {"Population": 5857000, "Researchers_per_Million": 8198},
    "NOR": {"Population": 5408000, "Researchers_per_Million": 6853},
    "FIN": {"Population": 5541000, "Researchers_per_Million": 7203},
    "ISR": {"Population": 9364000, "Researchers_per_Million": 8337},
    "SGP": {"Population": 5454000, "Researchers_per_Million": 7529},
    "IRL": {"Population": 5060000, "Researchers_per_Million": 4468},
    "PRT": {"Population": 10270000, "Researchers_per_Million": 4639},
    "GRC": {"Population": 10430000, "Researchers_per_Million": 4088},
    "CZE": {"Population": 10510000, "Researchers_per_Million": 4150},
    "HUN": {"Population": 9710000, "Researchers_per_Million": 3033},
    "TUR": {"Population": 85340000, "Researchers_per_Million": 1616},
    "MEX": {"Population": 128900000, "Researchers_per_Million": 349},
    "ARG": {"Population": 45810000, "Researchers_per_Million": 1204},
    "ZAF": {"Population": 59390000, "Researchers_per_Million": 484},
    "SAU": {"Population": 36410000, "Researchers_per_Million": 423},
    "IRN": {"Population": 87920000, "Researchers_per_Million": 1073},
    "MYS": {"Population": 33940000, "Researchers_per_Million": 2397},
    "THA": {"Population": 71700000, "Researchers_per_Million": 1350},
    "IDN": {"Population": 275500000, "Researchers_per_Million": 216},
    "EGY": {"Population": 109300000, "Researchers_per_Million": 723},
    "PAK": {"Population": 231400000, "Researchers_per_Million": 335},
    "NGA": {"Population": 218500000, "Researchers_per_Million": 39},
    "VNM": {"Population": 98190000, "Researchers_per_Million": 760},
    "NZL": {"Population": 5124000, "Researchers_per_Million": 4764},
    "SVN": {"Population": 2108000, "Researchers_per_Million": 5124},
    "HRV": {"Population": 3899000, "Researchers_per_Million": 1984},
    "SVK": {"Population": 5435000, "Researchers_per_Million": 2895},
    "BGR": {"Population": 6878000, "Researchers_per_Million": 2291},
    "ROU": {"Population": 19120000, "Researchers_per_Million": 924},
    "UKR": {"Population": 43790000, "Researchers_per_Million": 1131},
    "TWN": {"Population": 23890000, "Researchers_per_Million": 8080},
    "HKG": {"Population": 7413000, "Researchers_per_Million": 4038},
    "CHL": {"Population": 19490000, "Researchers_per_Million": 493},
    "COL": {"Population": 51870000, "Researchers_per_Million": 138},
}


def get_fallback_normalization_data() -> pd.DataFrame:
    """
    Get fallback normalization data when API is unavailable.
    """
    records = []
    for iso, data in FALLBACK_COUNTRY_DATA.items():
        pop = data["Population"]
        rpm = data["Researchers_per_Million"]
        records.append({
            "ISO": iso,
            "Country": iso,  # Will be mapped later
            "Population": pop,
            "Researchers_per_Million": rpm,
            "Total_Researchers": int(pop / 1_000_000 * rpm),
        })
    return pd.DataFrame(records)


# =============================================================================
# HIGH-LEVEL INTERFACE
# =============================================================================

def get_normalization_data(use_api: bool = True, year: int = None) -> pd.DataFrame:
    """
    Get normalization data, with fallback to hardcoded data.
    
    Parameters
    ----------
    use_api : bool
        Whether to try World Bank API first.
    year : int, optional
        Specific year for API data.
    
    Returns
    -------
    pd.DataFrame
        Normalization data with Population and Researcher counts.
    """
    if use_api:
        try:
            df = fetch_normalization_data(year)
            if len(df) > 0:
                return df
        except Exception as e:
            print(f"World Bank API failed: {e}")
    
    print("Using fallback normalization data.")
    return get_fallback_normalization_data()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Test fetching data
    print("Testing World Bank API...")
    
    # Test population
    pop = fetch_population_data()
    print(f"\nPopulation data: {len(pop)} countries")
    print(pop.head())
    
    # Test researchers
    res = fetch_researcher_data()
    print(f"\nResearcher data: {len(res)} countries")
    print(res.head())
    
    # Test combined
    combined = get_normalization_data()
    print(f"\nCombined data: {len(combined)} countries")
    print(combined.head(10))
    
    # Test normalization
    test_countries = pd.DataFrame({
        "Country": ["United States", "China", "Germany", "Slovenia", "Unknown Country"],
        "Count": [1000, 800, 500, 50, 10],
    })
    
    print("\n\nTest normalization:")
    normalized = normalize_country_counts(test_countries, normalize_by="both")
    print(normalized)
