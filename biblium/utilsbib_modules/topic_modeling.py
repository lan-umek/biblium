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

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

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


def topic_modeling(
    df: pd.DataFrame,
    text_column: str,
    model_type: str = "LDA",
    n_topics: Optional[int] = None,
    max_topics: int = 10,
    max_features: int = 5000,
    stop_words: Union[str, List[str], None] = "english",
    *,
    include_doc_topic_weights: bool = True,
    topic_prefix: str = "Topic",
    weight_suffix: str = "Weight",
    normalize: Literal["row_sum", "softmax", "none"] = "row_sum",
    lsa_nonneg: Literal["abs", "square", "none"] = "abs",
    empty_weight_as_nan: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform topic modeling (LDA, NMF, or LSA) on text data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with text data.
    text_column : str
        Column containing text to model.
    model_type : str
        Type of topic model: 'LDA', 'NMF', or 'LSA'.
    n_topics : int, optional
        Number of topics. If None, uses coherence-based selection.
    max_topics : int
        Maximum number of topics to consider.
    max_features : int
        Maximum vocabulary size.
    stop_words : str, list, or None
        Stop words to remove. 'english' for default list.
    include_doc_topic_weights : bool
        If True, add topic weight columns to output.
    topic_prefix : str
        Prefix for topic columns.
    weight_suffix : str
        Suffix for weight columns.
    normalize : str
        Normalization method for weights: 'row_sum', 'softmax', 'none'.
    lsa_nonneg : str
        How to handle negative LSA values: 'abs', 'square', 'none'.
    empty_weight_as_nan : bool
        If True, empty texts get NaN weights.
        
    Returns
    -------
    tuple
        (df_with_topics, topics_df)
        - df_with_topics: Original DataFrame with Topic column and optionally weight columns
        - topics_df: DataFrame with Topic, Term, Weight columns for top terms
    """
    CountVectorizer, TfidfVectorizer = _get_vectorizers()
    LDA, NMF, TruncatedSVD = _get_decomposition()
    
    if text_column not in df.columns:
        raise KeyError(f"Column '{text_column}' not found in df.")
    
    texts = df[text_column].fillna("").astype(str).str.strip()
    mask = texts != ""
    df_out = df.copy()
    
    # Early return if nothing to model
    if mask.sum() == 0:
        df_out["Topic"] = pd.Series([pd.NA] * len(df_out), index=df_out.index, dtype="Float64")
        return df_out, pd.DataFrame(columns=["Topic", "Term", "Weight"])
    
    # Select vectorizer based on model type
    mt = model_type.upper()
    if mt in {"LDA", "LSA"}:
        vectorizer = CountVectorizer(max_features=max_features, stop_words=stop_words)
    elif mt == "NMF":
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    else:
        raise ValueError('model_type must be one of: "LDA", "NMF", "LSA".')
    
    # Vectorize valid rows
    try:
        X = vectorizer.fit_transform(texts[mask])
    except ValueError:
        df_out["Topic"] = pd.Series([pd.NA] * len(df_out), index=df_out.index, dtype="Float64")
        return df_out, pd.DataFrame(columns=["Topic", "Term", "Weight"])
    
    # Determine number of topics
    if n_topics is None:
        n_topics = min(max_topics, X.shape[1], mask.sum())
    n_topics = max(1, min(n_topics, X.shape[1], mask.sum()))
    
    # Build model
    if mt == "LDA":
        model = LDA(n_components=n_topics, random_state=42)
        doc_topic = model.fit_transform(X)
    elif mt == "NMF":
        model = NMF(n_components=n_topics, random_state=42, init="nndsvd")
        doc_topic = model.fit_transform(X)
    else:  # LSA
        model = TruncatedSVD(n_components=n_topics, random_state=42)
        doc_topic = model.fit_transform(X)
        
        # Handle negative values
        if lsa_nonneg == "abs":
            doc_topic = np.abs(doc_topic)
        elif lsa_nonneg == "square":
            doc_topic = doc_topic ** 2
    
    # Normalize if requested
    if normalize == "row_sum":
        row_sums = doc_topic.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        doc_topic = doc_topic / row_sums
    elif normalize == "softmax":
        exp_vals = np.exp(doc_topic - doc_topic.max(axis=1, keepdims=True))
        doc_topic = exp_vals / exp_vals.sum(axis=1, keepdims=True)
    
    # Assign dominant topic
    topic_assignments = doc_topic.argmax(axis=1) + 1  # 1-indexed
    
    # Create topic column
    topic_col = pd.Series([pd.NA] * len(df_out), index=df_out.index, dtype="Float64")
    topic_col.loc[mask] = topic_assignments
    df_out["Topic"] = topic_col
    
    # Add weight columns if requested
    if include_doc_topic_weights:
        weight_matrix = np.full((len(df_out), n_topics), np.nan if empty_weight_as_nan else 0.0)
        weight_matrix[mask.values] = doc_topic
        
        for i in range(n_topics):
            col_name = f"{topic_prefix} {i+1} {weight_suffix}"
            df_out[col_name] = weight_matrix[:, i]
    
    # Extract top terms per topic
    feature_names = vectorizer.get_feature_names_out()
    if mt == "LSA":
        components = np.abs(model.components_)
    else:
        components = model.components_
    
    topics_data = []
    for topic_idx in range(n_topics):
        top_indices = components[topic_idx].argsort()[-10:][::-1]
        for idx in top_indices:
            topics_data.append({
                "Topic": topic_idx + 1,
                "Term": feature_names[idx],
                "Weight": components[topic_idx, idx],
            })
    
    topics_df = pd.DataFrame(topics_data)
    
    return df_out, topics_df


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
        raise ImportError("openai package required: pip install openai")
    
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
