# -*- coding: utf-8 -*-
"""
Topic Modeling Utilities.

This module provides functions for topic modeling and conceptual structure analysis:
- LDA (Latent Dirichlet Allocation)
- NMF (Non-negative Matrix Factorization)
- LSA (Latent Semantic Analysis)
- Conceptual Structure Analysis (factor analysis, clustering, MDS/PCA)
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd


# Lazy imports
def _get_vectorizers():
    """Get sklearn vectorizers (lazy import)."""
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    return CountVectorizer, TfidfVectorizer


def _get_decomposition():
    """Get sklearn decomposition classes (lazy import)."""
    from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
    return LatentDirichletAllocation, NMF, TruncatedSVD


def _get_clustering():
    """Get sklearn clustering classes (lazy import)."""
    from sklearn.cluster import KMeans, AgglomerativeClustering
    return KMeans, AgglomerativeClustering


def _get_manifold():
    """Get sklearn manifold classes (lazy import)."""
    from sklearn.manifold import MDS
    from sklearn.decomposition import PCA
    return MDS, PCA


# (2.18) topic_modeling: kanonična implementacija je v biblium.utilsbib
from biblium.utilsbib import topic_modeling  # noqa: E402,F401

def get_topic_summary(
    topics_df: pd.DataFrame,
    n_terms: int = 10,
) -> Dict[int, List[str]]:
    """
    Get summary of topics with top terms.
    
    Parameters
    ----------
    topics_df : pd.DataFrame
        Topic terms DataFrame from topic_modeling().
    n_terms : int
        Number of terms per topic.
        
    Returns
    -------
    dict
        Mapping from topic number to list of top terms.
    """
    summary = {}
    for topic in topics_df["Topic"].unique():
        terms = (
            topics_df[topics_df["Topic"] == topic]
            .nlargest(n_terms, "Weight")["Term"]
            .tolist()
        )
        summary[int(topic)] = terms
    return summary


def compute_topic_coherence(
    df: pd.DataFrame,
    text_column: str,
    topics_df: pd.DataFrame,
    n_terms: int = 10,
) -> Dict[int, float]:
    """
    Compute topic coherence scores using co-occurrence.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame with text.
    text_column : str
        Column containing text.
    topics_df : pd.DataFrame
        Topic terms DataFrame.
    n_terms : int
        Number of terms to use for coherence.
        
    Returns
    -------
    dict
        Coherence score per topic.
    """
    texts = df[text_column].fillna("").str.lower()
    
    coherence = {}
    for topic in topics_df["Topic"].unique():
        terms = (
            topics_df[topics_df["Topic"] == topic]
            .nlargest(n_terms, "Weight")["Term"]
            .str.lower()
            .tolist()
        )
        
        # Count co-occurrences
        cooc_sum = 0
        pair_count = 0
        
        for i, term1 in enumerate(terms):
            for term2 in terms[i+1:]:
                # Count documents containing both terms
                both = ((texts.str.contains(term1, regex=False)) & 
                       (texts.str.contains(term2, regex=False))).sum()
                # Count documents containing term1
                single = texts.str.contains(term1, regex=False).sum()
                
                if single > 0:
                    cooc_sum += np.log((both + 1) / single)
                    pair_count += 1
        
        coherence[int(topic)] = cooc_sum / pair_count if pair_count > 0 else 0.0
    
    return coherence


def label_topics_with_llm(
    topics_df: pd.DataFrame,
    api_key: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    n_terms: int = 10,
) -> Dict[int, str]:
    """
    Generate topic labels using an LLM.
    
    Parameters
    ----------
    topics_df : pd.DataFrame
        Topic terms DataFrame.
    api_key : str, optional
        OpenAI API key. If None, uses OPENAI_API_KEY environment variable.
    model : str
        Model to use.
    n_terms : int
        Number of terms to include in prompt.
        
    Returns
    -------
    dict
        Mapping from topic number to generated label.
    """
    import os
    
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key required. Set OPENAI_API_KEY or pass api_key.")
    
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required: pip install openai") from None
    
    client = OpenAI(api_key=api_key)
    
    labels = {}
    for topic in topics_df["Topic"].unique():
        terms = (
            topics_df[topics_df["Topic"] == topic]
            .nlargest(n_terms, "Weight")["Term"]
            .tolist()
        )
        
        prompt = f"""Given these topic terms from a bibliometric analysis:
{', '.join(terms)}

Provide a short (2-4 word) descriptive label for this topic.
Return only the label, nothing else."""
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.3,
        )
        
        labels[int(topic)] = response.choices[0].message.content.strip()
    
    return labels
