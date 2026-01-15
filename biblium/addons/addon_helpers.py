# -*- coding: utf-8 -*-
"""
Biblium Addons - Usage Guide and Helper Functions

This module provides easy-to-use wrappers for addon functions
and documents the correct usage patterns.
"""

import pandas as pd
from typing import List, Optional


# =============================================================================
# IMPACT METRICS
# =============================================================================

def compute_disruption(df: pd.DataFrame, 
                       id_col: str = "EID",
                       refs_col: str = "References",
                       citations_col: str = "Cited by",
                       year_col: str = "Year",
                       sep: str = "; ") -> pd.DataFrame:
    """
    Compute Disruption Index for all papers.
    
    Example:
        from biblium.addons import impact_metrics
        results = impact_metrics.compute_disruption_index(
            ba.df,
            id_col="EID",
            refs_col="References",
            citations_col="Cited by",
            year_col="Year",
            sep="; "
        )
    """
    from biblium.addons import impact_metrics
    return impact_metrics.compute_disruption_index(
        df, id_col=id_col, refs_col=refs_col, 
        citations_col=citations_col, year_col=year_col, sep=sep
    )


# =============================================================================
# TEXT MINING - DataFrame wrappers
# =============================================================================

def extract_entities_from_df(df: pd.DataFrame,
                             text_col: str = "Abstract",
                             id_col: str = None) -> pd.DataFrame:
    """
    Extract named entities from all documents in a DataFrame.
    
    Example:
        from biblium.addons import text_mining_nlp
        
        # Process each row
        results = []
        for idx, row in ba.df.iterrows():
            ner = text_mining_nlp.extract_named_entities(row["Abstract"], doc_id=idx)
            results.append(ner.to_dict())
        entities_df = pd.DataFrame(results)
    """
    from biblium.addons import text_mining_nlp
    
    results = []
    for idx, row in df.iterrows():
        doc_id = row[id_col] if id_col and id_col in row else idx
        text = row.get(text_col, "")
        ner = text_mining_nlp.extract_named_entities(str(text), doc_id=doc_id)
        results.append(ner.to_dict())
    
    return pd.DataFrame(results)


def analyze_sentiment_df(df: pd.DataFrame,
                         text_col: str = "Abstract",
                         id_col: str = None) -> pd.DataFrame:
    """
    Analyze sentiment for all documents in a DataFrame.
    
    Example:
        from biblium.addons import text_mining_nlp
        
        results = []
        for idx, row in ba.df.iterrows():
            sent = text_mining_nlp.analyze_sentiment(row["Abstract"], doc_id=idx)
            results.append(sent.to_dict())
        sentiment_df = pd.DataFrame(results)
    """
    from biblium.addons import text_mining_nlp
    
    results = []
    for idx, row in df.iterrows():
        doc_id = row[id_col] if id_col and id_col in row else idx
        text = row.get(text_col, "")
        sent = text_mining_nlp.analyze_sentiment(str(text), doc_id=doc_id)
        results.append(sent.to_dict())
    
    return pd.DataFrame(results)


# =============================================================================
# CONCEPTUAL DRIFT
# =============================================================================

def analyze_concept_drift(df: pd.DataFrame,
                          target_terms: List[str],
                          text_col: str = "Processed Abstract",
                          year_col: str = "Year",
                          window_size: int = 5):
    """
    Analyze how concepts evolve over time.
    
    Example:
        from biblium.addons import conceptual_drift
        
        # Analyze drift for specific terms
        results = conceptual_drift.compute_conceptual_drift(
            ba.df,
            target_terms=["sustainability", "innovation", "governance"],
            text_col="Processed Abstract",  # Must be preprocessed text
            year_col="Year",
            window_size=5
        )
    """
    from biblium.addons import conceptual_drift
    return conceptual_drift.compute_conceptual_drift(
        df, target_terms=target_terms,
        text_col=text_col, year_col=year_col,
        window_size=window_size
    )


# =============================================================================
# GEOGRAPHIC ANALYSIS
# =============================================================================

def run_geographic(df: pd.DataFrame,
                   affiliation_col: str = "Affiliations",
                   citations_col: str = "Cited by",
                   year_col: str = "Year",
                   sep: str = "; "):
    """
    Run comprehensive geographic analysis.
    
    Example:
        from biblium.addons import geographic_analysis
        
        results = geographic_analysis.run_geographic_analysis(
            ba.df,
            affiliation_col="Affiliations",
            citations_col="Cited by",
            year_col="Year"
        )
        
        # Access results
        print(results.country_stats)
        print(results.collaboration_links)
    """
    from biblium.addons import geographic_analysis
    return geographic_analysis.run_geographic_analysis(
        df, affiliation_col=affiliation_col,
        citations_col=citations_col, year_col=year_col, sep=sep
    )


# =============================================================================
# TEMPORAL NETWORKS
# =============================================================================

def analyze_network_evolution(df: pd.DataFrame,
                              network_type: str = "keyword",
                              column: str = None,
                              year_col: str = "Year",
                              window_size: int = 5,
                              sep: str = "; "):
    """
    Analyze how networks evolve over time.
    
    Example:
        from biblium.addons import temporal_networks
        
        # Keyword co-occurrence evolution
        snapshots = temporal_networks.generate_network_snapshots(
            ba.df,
            network_type="keyword",  # or "coauthorship", "country"
            column="Author Keywords",
            year_col="Year",
            window_size=5
        )
        
        # Access snapshots
        for period, snapshot in snapshots.items():
            print(f"{period}: {snapshot.n_nodes} nodes, {snapshot.n_edges} edges")
    """
    from biblium.addons import temporal_networks
    return temporal_networks.generate_network_snapshots(
        df, network_type=network_type, column=column,
        year_col=year_col, window_size=window_size, sep=sep
    )


# =============================================================================
# ADVANCED STATISTICS
# =============================================================================

def analyze_bibliometric_laws(df: pd.DataFrame,
                              authors_col: str = "Authors",
                              sources_col: str = "Source title",
                              sep: str = "; "):
    """
    Analyze Lotka's, Bradford's, and other bibliometric laws.
    
    Example:
        from biblium.addons import advanced_statistics
        
        # Lotka's Law (author productivity)
        lotka = advanced_statistics.analyze_lotka_law(
            ba.df,
            authors_col="Authors",
            sep="; "
        )
        print(f"Lotka exponent: {lotka.exponent}")
        
        # Bradford's Law (source distribution)
        bradford = advanced_statistics.analyze_bradford_law(
            ba.df,
            source_col="Source title"
        )
        print(f"Core sources: {bradford.core_sources}")
    """
    from biblium.addons import advanced_statistics
    
    results = {}
    results['lotka'] = advanced_statistics.analyze_lotka_law(df, authors_col=authors_col, sep=sep)
    results['bradford'] = advanced_statistics.analyze_bradford_law(df, source_col=sources_col)
    
    return results


# =============================================================================
# QUICK REFERENCE
# =============================================================================

ADDON_QUICK_REFERENCE = """
BIBLIUM ADDONS - QUICK REFERENCE
================================

1. DISRUPTION INDEX
   from biblium.addons import impact_metrics
   results = impact_metrics.compute_disruption_index(df, refs_col="References")

2. NAMED ENTITY EXTRACTION (per text)
   from biblium.addons import text_mining_nlp
   ner = text_mining_nlp.extract_named_entities(text_string)

3. SENTIMENT ANALYSIS (per text)
   from biblium.addons import text_mining_nlp
   sent = text_mining_nlp.analyze_sentiment(text_string)

4. CONCEPTUAL DRIFT
   from biblium.addons import conceptual_drift
   drift = conceptual_drift.compute_conceptual_drift(
       df, target_terms=["term1", "term2"], text_col="Processed Abstract"
   )

5. GEOGRAPHIC ANALYSIS
   from biblium.addons import geographic_analysis
   geo = geographic_analysis.run_geographic_analysis(df, affiliation_col="Affiliations")

6. TEMPORAL NETWORKS
   from biblium.addons import temporal_networks
   snapshots = temporal_networks.generate_network_snapshots(
       df, network_type="keyword", column="Author Keywords"
   )

7. LOTKA'S LAW
   from biblium.addons import advanced_statistics
   lotka = advanced_statistics.analyze_lotka_law(df, authors_col="Authors")

8. BRADFORD'S LAW
   from biblium.addons import advanced_statistics
   bradford = advanced_statistics.analyze_bradford_law(df, source_col="Source title")

9. COMPARATIVE ANALYSIS
   from biblium.addons import comparative_analysis
   comp = comparative_analysis.compare_datasets(df1, df2)

10. OPEN SCIENCE METRICS
    from biblium.addons import open_science
    oa = open_science.analyze_open_access(df)

11. ALTMETRICS
    from biblium.addons import altmetrics
    alt = altmetrics.compute_attention_score(df)
"""

def print_quick_reference():
    """Print quick reference guide for all addons."""
    print(ADDON_QUICK_REFERENCE)


if __name__ == "__main__":
    print_quick_reference()
