# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 12:18:25 2025

@author: Lan.Umek
"""

import pandas as pd
import os
import re
from typing import Any, Callable

fd = os.path.dirname(__file__)
df0 = pd.read_excel(fd+"\\additional files\\variable names.xlsx", sheet_name="names")


# WOS


# Regular expression to detect WoS tag lines in the .txt export
_TAG_LINE_RE = re.compile(r'^([A-Z0-9]{2,3})\s+(.*)$')


def create_name_mapper(df: pd.DataFrame, key_column: str) -> Callable[[Any], Any]:
    """
    Create a mapper function that maps values from a specified column to the 'name' column.

    Args:
        df (pd.DataFrame): DataFrame containing a 'name' column and the key_column.
        key_column (str): The name of the column whose values you want to map.

    Returns:
        Callable[[Any], Any]: A function that takes a value (from key_column) and returns the corresponding
                              'name' value from the same row, or the original value if not found.
    """
    # Build mapping from key_column values to name values
    mapping = dict(zip(df[key_column], df["name"]))

    def mapper(value: Any) -> Any:
        """
        Map a single value to its corresponding 'name', or return it unchanged if missing.
        
        Args:
            value (Any): A value to look up in key_column.
        
        Returns:
            Any: The mapped 'name' value, or the original value if not found.
        """
        return mapping.get(value, value)

    return mapper


def read_wos_xls(filepath: str, mapping_column: str = "wos") -> pd.DataFrame:
    """
    Read a Web of Science Excel (.xls or .xlsx) export and return a DataFrame
    containing the raw columns from the file, with any all-NA columns dropped,
    optionally remapped based on a mapping column.

    Parameters
    ----------
    filepath : str
        Path to the WoS Excel file export.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names. If provided,
        a mapper is constructed via create_name_mapper(df0, mapping_column)
        and applied to rename columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with raw WoS columns, no all-NA columns,
        potentially with renamed columns.
    """
    # Load the Excel file (auto-detect engine)
    df = pd.read_excel(filepath, dtype=str)
    # Drop any columns that are entirely missing
    df.dropna(axis=1, how='all', inplace=True)

    # Apply column mapping if requested
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    return df


def read_wos_txt(filepath: str, mapping_column: str = "wos-abb") -> pd.DataFrame:
    """
    Read a Web of Science plain-text export (.txt) and return a DataFrame
    with raw field tags as columns, parsing records separated by blank lines,
    skipping the first two header lines, dropping any all-NA columns,
    and optionally remapping based on a mapping column.

    Parameters
    ----------
    filepath : str
        Path to the WoS .txt file export.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names. If provided,
        a mapper is constructed via create_name_mapper(df0, mapping_column)
        and applied to rename columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with raw WoS tags as columns, no all-NA columns,
        potentially with renamed columns.
    """
    records = []
    with open(filepath, encoding='utf-8') as f:
        # Skip first two metadata lines
        next(f, None)
        next(f, None)
        record = {}
        last_tag = None
        for line in f:
            line = line.rstrip('\n')
            # Blank line indicates end of record
            if not line.strip():
                if record:
                    records.append(record)
                    record = {}
                last_tag = None
                continue
            # Match tag lines
            m = _TAG_LINE_RE.match(line)
            if m:
                tag, val = m.groups()
                record[tag] = val
                last_tag = tag
            else:
                # Continuation of previous tag
                if last_tag and last_tag in record:
                    record[last_tag] += ' ' + line.strip()
        # Append last record if present
        if record:
            records.append(record)
    df = pd.DataFrame(records)
    df.dropna(axis=1, how='all', inplace=True)

    # Apply column mapping if requested
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    return df


def read_wos_bib(filepath: str, mapping_column: str = "wos-bib") -> pd.DataFrame:
    """
    Read a Web of Science BibTeX export (.bib) and return a DataFrame
    with raw BibTeX fields as columns, all-NA columns dropped,
    and optionally remapped based on a mapping column.

    Parameters
    ----------
    filepath : str
        Path to the WoS .bib file export.
    mapping_column : str, optional
        Column name in df0 to use for mapping column names. If provided,
        a mapper is constructed via create_name_mapper(df0, mapping_column)
        and applied to rename columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with raw BibTeX fields as columns, no all-NA columns,
        potentially with renamed columns.
    """
    records = []
    entry = {}
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # New entry on '@'
            if line.startswith('@'):
                if entry:
                    records.append(entry)
                entry = {}
            elif '=' in line:
                key, rest = line.split('=', 1)
                key = key.strip().lower()
                val = rest.strip().rstrip(',').strip('{}').strip()
                entry[key] = val
    # Append last entry
    if entry:
        records.append(entry)
    df = pd.DataFrame(records)
    df.dropna(axis=1, how='all', inplace=True)

    # Apply column mapping if requested
    if mapping_column is not None:
        mapper = create_name_mapper(df0, mapping_column)
        df.rename(columns=mapper, inplace=True)

    return df

# general

def read_bibfile(f_name, db):
    if f_name is None:
        return pd.DataFrame([])
    if db.lower() == "scopus":
        if ".xlsx" in f_name:
            df = pd.read_excel(f_name)
        elif ".csv" in f_name:
            df = pd.read_csv(f_name)
    elif db.lower() == "wos":
        if ".txt" in f_name:
            df = read_wos_txt(f_name)
        elif ".xls" in f_name:
            df = read_wos_xls(f_name)
        elif ".bib" in f_name:
            df = read_wos_bib(f_name)
    return df


