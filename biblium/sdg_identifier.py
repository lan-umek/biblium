import pandas as pd
import re
from typing import List, Dict, Tuple, Optional


SDG_MAPPINGS = {
    "SDG 01": {"perspective": "life",                "dimension": "economic"},
    "SDG 02": {"perspective": "life",                "dimension": "economic"},
    "SDG 03": {"perspective": "life",                "dimension": "economic"},
    "SDG 04": {"perspective": "equality",            "dimension": "social"},
    "SDG 05": {"perspective": "equality",            "dimension": "social"},
    "SDG 06": {"perspective": "resources",           "dimension": "environmental"},
    "SDG 07": {"perspective": "resources",           "dimension": "environmental"},
    "SDG 08": {"perspective": "economic",            "dimension": "economic"},
    "SDG 09": {"perspective": "economic",            "dimension": "economic"},
    "SDG 10": {"perspective": "equality",            "dimension": "social"},
    "SDG 11": {"perspective": "social development",  "dimension": "social"},
    "SDG 12": {"perspective": "resources",           "dimension": "environmental"},
    "SDG 13": {"perspective": "natural environment", "dimension": "environmental"},
    "SDG 14": {"perspective": "natural environment", "dimension": "environmental"},
    "SDG 15": {"perspective": "natural environment", "dimension": "environmental"},
    "SDG 16": {"perspective": "social development",  "dimension": "social"},
}

class ScopusQueryMatcher:
    """
    Converts and applies Scopus TITLE-ABS-KEY queries for text matching.
    """
    
    def __init__(self, query: str):
        self.original_query = query
        self.parsed_query = self._parse_query(query)
    
    def _parse_query(self, query: str) -> str:
        if pd.isna(query):
            return ""
        # Remove TITLE-ABS-KEY prefix and cleanup
        pattern = re.sub(r'^TITLE-ABS-KEY\s*\(\s*', '', str(query))
        pattern = pattern.rstrip().rstrip(')')
        return pattern
    
    def _create_phrase_pattern(self, phrase: str) -> str:
        phrase = re.escape(phrase)
        phrase = phrase.replace(r'\ ', r'\s+')
        return r'\b' + phrase + r'\b'
    
    def _check_phrase(self, text: str, phrase: str) -> bool:
        pattern = self._create_phrase_pattern(phrase)
        return bool(re.search(pattern, text, re.IGNORECASE))
    
    def _check_wildcard(self, text: str, pattern: str) -> bool:
        # Improved wildcard handling: escape everything except the *
        # Then replace * with \w* (word characters)
        parts = pattern.split('*')
        regex_parts = [re.escape(p) for p in parts]
        regex_pattern = r'\w*'.join(regex_parts)
        regex_pattern = r'\b' + regex_pattern + r'\b'
        return bool(re.search(regex_pattern, text, re.IGNORECASE))
    
    def _evaluate_expression(self, expr: str, text: str) -> bool:
        expr = expr.strip()
        if not expr:
            return False
        
        # Remove outer parentheses
        if expr.startswith('(') and expr.endswith(')'):
            depth = 0
            is_wrapped = True
            for i, char in enumerate(expr):
                if char == '(': depth += 1
                elif char == ')': depth -= 1
                if depth == 0 and i < len(expr) - 1:
                    is_wrapped = False
                    break
            if is_wrapped:
                return self._evaluate_expression(expr[1:-1], text)
        
        # Split by OR (Top level)
        depth = 0
        or_parts = []
        last_pos = 0
        for i in range(len(expr)):
            if expr[i] == '(': depth += 1
            elif expr[i] == ')': depth -= 1
            elif depth == 0 and expr[i:i+4] == ' OR ':
                or_parts.append(expr[last_pos:i])
                last_pos = i + 4
        or_parts.append(expr[last_pos:])
        
        if len(or_parts) > 1:
            return any(self._evaluate_expression(part, text) for part in or_parts)

        # Split by AND (Top level)
        depth = 0
        and_parts = []
        last_pos = 0
        for i in range(len(expr)):
            if expr[i] == '(': depth += 1
            elif expr[i] == ')': depth -= 1
            elif depth == 0 and expr[i:i+5] == ' AND ':
                and_parts.append(expr[last_pos:i])
                last_pos = i + 5
        and_parts.append(expr[last_pos:])
        
        if len(and_parts) > 1:
            return all(self._evaluate_expression(part, text) for part in and_parts)
            
        # Terminal checks
        phrase_match = re.match(r'^\{(.+?)\}$', expr)
        if phrase_match:
            return self._check_phrase(text, phrase_match.group(1))
        
        if '*' in expr:
            return self._check_wildcard(text, expr)
        
        pattern = r'\b' + re.escape(expr) + r'\b'
        return bool(re.search(pattern, text, re.IGNORECASE))

    def matches(self, text: str) -> bool:
        if pd.isna(text): return False
        return self._evaluate_expression(self.parsed_query, str(text))


def load_sdg_queries(filepath: str) -> pd.DataFrame:
    """Loads only the goals sheet which contains queries."""
    return pd.read_excel(filepath, sheet_name='goals')


def identify_sdgs(df: pd.DataFrame, 
                  text_column: str = "Abstract",
                  sdg_queries_path: str = None,
                  return_perspectives: bool = True,
                  return_dimensions: bool = True) -> pd.DataFrame:
    
    result_df = df.copy()
    
    # Default path handling
    if sdg_queries_path is None:
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sdg_queries_path = os.path.join(script_dir, "additional files", "Scopus_SDG_metadata.xlsx")
    
    # Load goals (queries)
    goals_df = load_sdg_queries(sdg_queries_path)
    
    # 1. Identify SDGs
    print("Identifying SDGs...")
    sdg_cols_present = []
    
    for idx, row in goals_df.iterrows():
        # Handle cases where goal might be formatted differently
        goal_str = str(row['goal']).strip() # e.g., "SDG 01"
        
        # Extract number for consistent column naming
        match = re.search(r'\d+', goal_str)
        if match:
            sdg_num = int(match.group())
            col_name = f'SDG{sdg_num:02d}'
            sdg_cols_present.append(col_name)
            
            print(f"  Processing {col_name}...")
            matcher = ScopusQueryMatcher(row['query'])
            result_df[col_name] = result_df[text_column].apply(lambda x: 1 if matcher.matches(x) else 0)

    # 2. Compute Perspectives (Using Helper Dictionary)
    if return_perspectives:
        print("\nComputing perspectives...")
        # Invert the mapping: Perspective -> [List of SDG columns]
        persp_to_sdgs = {}
        for sdg_key, props in SDG_MAPPINGS.items():
            # Construct column name like "SDG01" from "SDG 01"
            sdg_col_num = int(re.search(r'\d+', sdg_key).group())
            sdg_col = f'SDG{sdg_col_num:02d}'
            
            persp = props['perspective']
            if persp not in persp_to_sdgs:
                persp_to_sdgs[persp] = []
            persp_to_sdgs[persp].append(sdg_col)

        for persp, cols in persp_to_sdgs.items():
            # Only use columns that actually exist in the result
            valid_cols = [c for c in cols if c in result_df.columns]
            if valid_cols:
                col_name = f"perspective_{persp.replace(' ', '_')}"
                print(f"  Processing {col_name} (from {', '.join(valid_cols)})...")
                result_df[col_name] = result_df[valid_cols].max(axis=1)

    # 3. Compute Dimensions (Using Helper Dictionary)
    if return_dimensions:
        print("\nComputing dimensions...")
        dim_to_sdgs = {}
        for sdg_key, props in SDG_MAPPINGS.items():
            sdg_col_num = int(re.search(r'\d+', sdg_key).group())
            sdg_col = f'SDG{sdg_col_num:02d}'
            
            dim = props['dimension']
            if dim not in dim_to_sdgs:
                dim_to_sdgs[dim] = []
            dim_to_sdgs[dim].append(sdg_col)

        for dim, cols in dim_to_sdgs.items():
            valid_cols = [c for c in cols if c in result_df.columns]
            if valid_cols:
                col_name = f"dimension_{dim}"
                print(f"  Processing {col_name} (from {', '.join(valid_cols)})...")
                result_df[col_name] = result_df[valid_cols].max(axis=1)

    # 4. Any SDG
    print("\nComputing any_SDG column...")
    all_sdg_cols = [c for c in result_df.columns if re.match(r'SDG\d+', c)]
    if all_sdg_cols:
        result_df['any_SDG'] = result_df[all_sdg_cols].max(axis=1)
    else:
        result_df['any_SDG'] = 0

    return result_df