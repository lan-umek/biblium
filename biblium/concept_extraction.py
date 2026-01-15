# -*- coding: utf-8 -*-
"""
Concept Extraction & Key Terminology Analysis
==============================================
Extract and analyze key concepts, keywords, and terminology from bibliometric datasets.

Features:
- Parse existing concepts/keywords from datasets
- Extract terms using TF-IDF from titles/abstracts
- Enrich documents via OpenAlex API
- Analyze concept frequency, co-occurrence, and trends
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Set
from collections import Counter
import re

import numpy as np
import pandas as pd


@dataclass
class ConceptInfo:
    """Information about a single concept/keyword."""
    name: str
    count: int = 0
    doc_freq: int = 0  # Number of documents containing this concept
    avg_score: float = 0.0  # Average relevance score (if available)
    level: int = -1  # Concept hierarchy level (for OpenAlex concepts)
    parent: str = ""  # Parent concept (if hierarchical)
    
    # For OpenAlex concepts
    openalex_id: str = ""
    wikidata_id: str = ""


@dataclass 
class ConceptExtractionResult:
    """Result of concept extraction analysis."""
    # Term frequency data
    concepts: List[ConceptInfo] = field(default_factory=list)
    keywords: List[ConceptInfo] = field(default_factory=list)
    topics: List[ConceptInfo] = field(default_factory=list)
    ngrams: List[ConceptInfo] = field(default_factory=list)
    
    # DataFrames for detailed analysis
    concept_df: pd.DataFrame = None
    keyword_df: pd.DataFrame = None
    topic_df: pd.DataFrame = None
    ngram_df: pd.DataFrame = None
    
    # Co-occurrence matrix
    cooccurrence_matrix: pd.DataFrame = None
    
    # Temporal trends
    temporal_trends: pd.DataFrame = None
    
    # Statistics
    n_documents: int = 0
    n_with_concepts: int = 0
    n_with_keywords: int = 0
    n_unique_concepts: int = 0
    n_unique_keywords: int = 0
    
    # Data source info
    data_source: str = "dataset"
    api_calls_made: int = 0


def _parse_pipe_separated(value, score_value=None) -> List[Tuple[str, float]]:
    """
    Parse pipe-separated string to list of (term, score) tuples.
    
    Parameters
    ----------
    value : str
        Pipe-separated string of terms
    score_value : str, optional
        Pipe-separated string of scores
        
    Returns
    -------
    list of (term, score) tuples
    """
    if pd.isna(value) or not value:
        return []
    
    if isinstance(value, list):
        terms = value
    else:
        terms = [x.strip() for x in str(value).split('|') if x.strip()]
    
    if score_value and not pd.isna(score_value):
        if isinstance(score_value, list):
            scores = score_value
        else:
            scores = []
            for x in str(score_value).split('|'):
                try:
                    scores.append(float(x.strip()))
                except:
                    scores.append(0.0)
        
        # Pad scores if needed
        while len(scores) < len(terms):
            scores.append(0.0)
        
        return list(zip(terms, scores[:len(terms)]))
    
    return [(t, 1.0) for t in terms]


def _parse_semicolon_separated(value) -> List[str]:
    """Parse semicolon-separated string to list."""
    if pd.isna(value) or not value:
        return []
    
    return [x.strip() for x in str(value).split(';') if x.strip()]


def extract_concepts_from_dataset(
    df: pd.DataFrame,
    verbose: bool = True,
) -> Dict[str, Counter]:
    """
    Extract concepts, keywords, and topics from dataset columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliometric dataset
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        {type: Counter of terms}
    """
    results = {
        'concepts': Counter(),
        'keywords': Counter(),
        'topics': Counter(),
        'author_keywords': Counter(),
        'index_keywords': Counter(),
    }
    
    # Document frequency counters
    doc_freq = {
        'concepts': Counter(),
        'keywords': Counter(),
        'topics': Counter(),
        'author_keywords': Counter(),
        'index_keywords': Counter(),
    }
    
    # Score accumulators
    scores = {
        'concepts': {},
        'keywords': {},
        'topics': {},
    }
    
    # Find relevant columns
    cols = {c.lower(): c for c in df.columns}
    
    # OpenAlex columns
    concept_col = cols.get('concepts.display_name')
    concept_score_col = cols.get('concepts.score')
    keyword_col = cols.get('keywords.display_name')
    keyword_score_col = cols.get('keywords.score')
    topic_col = cols.get('topics.display_name') or cols.get('primary_topic.display_name')
    topic_score_col = cols.get('topics.score') or cols.get('primary_topic.score')
    
    # Scopus/WoS columns
    author_kw_col = cols.get('author keywords')
    index_kw_col = cols.get('index keywords')
    
    for idx, row in df.iterrows():
        # OpenAlex concepts
        if concept_col:
            terms = _parse_pipe_separated(
                row.get(concept_col), 
                row.get(concept_score_col) if concept_score_col else None
            )
            seen = set()
            for term, score in terms:
                term_lower = term.lower()
                results['concepts'][term] += 1
                if term_lower not in seen:
                    doc_freq['concepts'][term] += 1
                    seen.add(term_lower)
                if term not in scores['concepts']:
                    scores['concepts'][term] = []
                scores['concepts'][term].append(score)
        
        # OpenAlex keywords
        if keyword_col:
            terms = _parse_pipe_separated(
                row.get(keyword_col),
                row.get(keyword_score_col) if keyword_score_col else None
            )
            seen = set()
            for term, score in terms:
                term_lower = term.lower()
                results['keywords'][term] += 1
                if term_lower not in seen:
                    doc_freq['keywords'][term] += 1
                    seen.add(term_lower)
                if term not in scores['keywords']:
                    scores['keywords'][term] = []
                scores['keywords'][term].append(score)
        
        # OpenAlex topics
        if topic_col:
            terms = _parse_pipe_separated(
                row.get(topic_col),
                row.get(topic_score_col) if topic_score_col else None
            )
            seen = set()
            for term, score in terms:
                term_lower = term.lower()
                results['topics'][term] += 1
                if term_lower not in seen:
                    doc_freq['topics'][term] += 1
                    seen.add(term_lower)
                if term not in scores['topics']:
                    scores['topics'][term] = []
                scores['topics'][term].append(score)
        
        # Author keywords (Scopus/WoS)
        if author_kw_col:
            terms = _parse_semicolon_separated(row.get(author_kw_col))
            seen = set()
            for term in terms:
                term_lower = term.lower()
                results['author_keywords'][term] += 1
                if term_lower not in seen:
                    doc_freq['author_keywords'][term] += 1
                    seen.add(term_lower)
        
        # Index keywords (Scopus/WoS)
        if index_kw_col:
            terms = _parse_semicolon_separated(row.get(index_kw_col))
            seen = set()
            for term in terms:
                term_lower = term.lower()
                results['index_keywords'][term] += 1
                if term_lower not in seen:
                    doc_freq['index_keywords'][term] += 1
                    seen.add(term_lower)
    
    if verbose:
        print(f"  Extracted concepts: {len(results['concepts'])}")
        print(f"  Extracted keywords: {len(results['keywords'])}")
        print(f"  Extracted topics: {len(results['topics'])}")
        print(f"  Extracted author keywords: {len(results['author_keywords'])}")
        print(f"  Extracted index keywords: {len(results['index_keywords'])}")
    
    return results, doc_freq, scores


def extract_ngrams_tfidf(
    df: pd.DataFrame,
    text_cols: List[str] = None,
    ngram_range: Tuple[int, int] = (1, 3),
    max_features: int = 500,
    min_df: int = 2,
    stop_words: str = 'english',
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract n-grams from text columns using TF-IDF.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    text_cols : list, optional
        Columns to extract from. Default: title, abstract
    ngram_range : tuple
        Min and max n-gram size
    max_features : int
        Maximum number of features
    min_df : int
        Minimum document frequency
    stop_words : str
        Stop words language or list
    verbose : bool
        Print progress
        
    Returns
    -------
    pd.DataFrame
        DataFrame with term, tf, df, tfidf columns
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        if verbose:
            print("  ⚠️ scikit-learn not available for TF-IDF extraction")
        return pd.DataFrame()
    
    # Find text columns
    if text_cols is None:
        cols_lower = {c.lower(): c for c in df.columns}
        text_cols = []
        for name in ['title', 'abstract', 'document title', 'ti', 'ab']:
            if name in cols_lower:
                text_cols.append(cols_lower[name])
    
    if not text_cols:
        if verbose:
            print("  ⚠️ No text columns found for n-gram extraction")
        return pd.DataFrame()
    
    # Combine text columns
    texts = []
    for _, row in df.iterrows():
        parts = []
        for col in text_cols:
            val = row.get(col)
            if pd.notna(val):
                parts.append(str(val))
        texts.append(' '.join(parts))
    
    # Filter empty texts
    texts = [t if t.strip() else "empty" for t in texts]
    
    if verbose:
        print(f"  Extracting n-grams from {len(texts)} documents...")
        print(f"  Text columns: {text_cols}")
    
    # TF-IDF vectorization
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            stop_words=stop_words,
            lowercase=True,
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b',  # At least 2 chars
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate statistics
        data = []
        for i, term in enumerate(feature_names):
            col = tfidf_matrix.getcol(i)
            tf = col.sum()  # Total frequency
            df_count = col.nnz  # Document frequency
            avg_tfidf = col.sum() / col.nnz if col.nnz > 0 else 0
            
            data.append({
                'Term': term,
                'TF': int(tf),
                'DF': df_count,
                'Avg_TFIDF': round(avg_tfidf, 4),
            })
        
        result_df = pd.DataFrame(data)
        result_df = result_df.sort_values('Avg_TFIDF', ascending=False)
        
        if verbose:
            print(f"  ✓ Extracted {len(result_df)} n-grams")
        
        return result_df
        
    except Exception as e:
        if verbose:
            print(f"  ⚠️ TF-IDF extraction failed: {e}")
        return pd.DataFrame()


def compute_cooccurrence_matrix(
    df: pd.DataFrame,
    term_col: str,
    top_n: int = 50,
    min_cooccurrence: int = 2,
    separator: str = '|',
) -> pd.DataFrame:
    """
    Compute co-occurrence matrix for terms.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    term_col : str
        Column containing terms (pipe or semicolon separated)
    top_n : int
        Number of top terms to include
    min_cooccurrence : int
        Minimum co-occurrence count to include
    separator : str
        Term separator
        
    Returns
    -------
    pd.DataFrame
        Co-occurrence matrix
    """
    # Get top terms
    all_terms = Counter()
    for val in df[term_col].dropna():
        if separator == '|':
            terms = [t.strip() for t in str(val).split('|') if t.strip()]
        else:
            terms = [t.strip() for t in str(val).split(';') if t.strip()]
        all_terms.update(terms)
    
    top_terms = [t for t, _ in all_terms.most_common(top_n)]
    top_set = set(top_terms)
    
    # Build co-occurrence matrix
    cooc = Counter()
    for val in df[term_col].dropna():
        if separator == '|':
            terms = [t.strip() for t in str(val).split('|') if t.strip() and t.strip() in top_set]
        else:
            terms = [t.strip() for t in str(val).split(';') if t.strip() and t.strip() in top_set]
        
        for i, t1 in enumerate(terms):
            for t2 in terms[i+1:]:
                key = tuple(sorted([t1, t2]))
                cooc[key] += 1
    
    # Build matrix
    matrix = pd.DataFrame(0, index=top_terms, columns=top_terms)
    for (t1, t2), count in cooc.items():
        if count >= min_cooccurrence:
            matrix.loc[t1, t2] = count
            matrix.loc[t2, t1] = count
    
    return matrix


def compute_temporal_trends(
    df: pd.DataFrame,
    term_col: str,
    year_col: str = 'Year',
    top_n: int = 20,
    separator: str = '|',
) -> pd.DataFrame:
    """
    Compute temporal trends for top terms.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    term_col : str
        Column containing terms
    year_col : str
        Column containing publication year
    top_n : int
        Number of top terms to track
    separator : str
        Term separator
        
    Returns
    -------
    pd.DataFrame
        Pivot table with years as rows, terms as columns
    """
    # Find year column
    year_col_actual = None
    for c in df.columns:
        if c.lower() in ['year', 'publication_year', 'py']:
            year_col_actual = c
            break
    
    if year_col_actual is None:
        return pd.DataFrame()
    
    # Get top terms overall
    all_terms = Counter()
    for val in df[term_col].dropna():
        if separator == '|':
            terms = [t.strip() for t in str(val).split('|') if t.strip()]
        else:
            terms = [t.strip() for t in str(val).split(';') if t.strip()]
        all_terms.update(set(terms))  # Use set to count documents, not occurrences
    
    top_terms = [t for t, _ in all_terms.most_common(top_n)]
    
    # Build year-term counts
    data = []
    for _, row in df.iterrows():
        year = row.get(year_col_actual)
        if pd.isna(year):
            continue
        try:
            year = int(year)
        except:
            continue
        
        val = row.get(term_col)
        if pd.isna(val):
            continue
        
        if separator == '|':
            terms = set(t.strip() for t in str(val).split('|') if t.strip())
        else:
            terms = set(t.strip() for t in str(val).split(';') if t.strip())
        
        for term in terms:
            if term in top_terms:
                data.append({'Year': year, 'Term': term, 'Count': 1})
    
    if not data:
        return pd.DataFrame()
    
    trend_df = pd.DataFrame(data)
    pivot = trend_df.pivot_table(index='Year', columns='Term', values='Count', aggfunc='sum', fill_value=0)
    
    return pivot


def fetch_concepts_openalex(
    title: str,
    abstract: str = None,
    verbose: bool = False,
) -> Dict:
    """
    Fetch concepts/keywords/topics from OpenAlex /text endpoint.
    
    Parameters
    ----------
    title : str
        Document title
    abstract : str, optional
        Document abstract
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        {keywords: [...], topics: [...], concepts: [...]}
    """
    import requests
    import urllib.parse
    import time
    
    # Build query
    text = title
    if abstract:
        text = f"{title}. {abstract}"
    
    # Limit to 2000 chars
    text = text[:2000]
    
    if len(text) < 20:
        return {'keywords': [], 'topics': [], 'concepts': []}
    
    url = f"https://api.openalex.org/text?title={urllib.parse.quote(text)}"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return {
                'keywords': data.get('keywords', []),
                'topics': data.get('topics', []),
                'concepts': data.get('concepts', []),
                'primary_topic': data.get('primary_topic'),
            }
        elif response.status_code == 429:
            if verbose:
                print("  ⚠️ Rate limited by OpenAlex")
            time.sleep(1)
    except Exception as e:
        if verbose:
            print(f"  ⚠️ API error: {e}")
    
    return {'keywords': [], 'topics': [], 'concepts': []}


def analyze_concepts(
    df: pd.DataFrame,
    use_openalex: bool = False,
    extract_ngrams: bool = True,
    ngram_range: Tuple[int, int] = (1, 3),
    max_ngrams: int = 500,
    top_n: int = 100,
    verbose: bool = True,
    stop_flag: Callable[[], bool] = None,
) -> ConceptExtractionResult:
    """
    Comprehensive concept and keyword analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliometric dataset
    use_openalex : bool
        Use OpenAlex API for documents without concepts
    extract_ngrams : bool
        Extract n-grams from titles/abstracts
    ngram_range : tuple
        N-gram size range
    max_ngrams : int
        Maximum n-grams to extract
    top_n : int
        Number of top items to return
    verbose : bool
        Print progress
    stop_flag : callable, optional
        Function that returns True if analysis should stop
        
    Returns
    -------
    ConceptExtractionResult
    """
    def should_stop():
        return stop_flag is not None and stop_flag()
    
    if verbose:
        print("Analyzing concepts and keywords...")
        print(f"  Documents: {len(df)}")
    
    result = ConceptExtractionResult()
    result.n_documents = len(df)
    
    # Extract from existing columns
    term_counts, doc_freqs, score_avgs = extract_concepts_from_dataset(df, verbose=verbose)
    
    if should_stop():
        return result
    
    # Build concept info lists
    for term, count in term_counts['concepts'].most_common(top_n):
        df_count = doc_freqs['concepts'].get(term, 0)
        avg_score = np.mean(score_avgs['concepts'].get(term, [0]))
        result.concepts.append(ConceptInfo(
            name=term,
            count=count,
            doc_freq=df_count,
            avg_score=round(avg_score, 4),
        ))
    
    for term, count in term_counts['keywords'].most_common(top_n):
        df_count = doc_freqs['keywords'].get(term, 0)
        avg_score = np.mean(score_avgs['keywords'].get(term, [0]))
        result.keywords.append(ConceptInfo(
            name=term,
            count=count,
            doc_freq=df_count,
            avg_score=round(avg_score, 4),
        ))
    
    for term, count in term_counts['topics'].most_common(top_n):
        df_count = doc_freqs['topics'].get(term, 0)
        avg_score = np.mean(score_avgs['topics'].get(term, [0]))
        result.topics.append(ConceptInfo(
            name=term,
            count=count,
            doc_freq=df_count,
            avg_score=round(avg_score, 4),
        ))
    
    # Combine author + index keywords for Scopus/WoS
    combined_keywords = term_counts['author_keywords'] + term_counts['index_keywords']
    if combined_keywords and not result.keywords:
        for term, count in combined_keywords.most_common(top_n):
            result.keywords.append(ConceptInfo(
                name=term,
                count=count,
                doc_freq=doc_freqs['author_keywords'].get(term, 0) + doc_freqs['index_keywords'].get(term, 0),
            ))
    
    result.n_unique_concepts = len(term_counts['concepts'])
    result.n_unique_keywords = len(term_counts['keywords']) + len(combined_keywords)
    result.n_with_concepts = sum(1 for c in df.get('concepts.display_name', pd.Series()) if pd.notna(c) and c)
    result.n_with_keywords = sum(1 for c in df.get('keywords.display_name', df.get('Author Keywords', pd.Series())) if pd.notna(c) and c)
    
    if should_stop():
        return result
    
    # Extract n-grams
    if extract_ngrams:
        result.ngram_df = extract_ngrams_tfidf(
            df, 
            ngram_range=ngram_range, 
            max_features=max_ngrams,
            verbose=verbose
        )
        
        if result.ngram_df is not None and not result.ngram_df.empty:
            for _, row in result.ngram_df.head(top_n).iterrows():
                result.ngrams.append(ConceptInfo(
                    name=row['Term'],
                    count=int(row['TF']),
                    doc_freq=int(row['DF']),
                    avg_score=row['Avg_TFIDF'],
                ))
    
    if should_stop():
        return result
    
    # Build DataFrames
    if result.concepts:
        result.concept_df = pd.DataFrame([
            {'Concept': c.name, 'Count': c.count, 'Doc_Freq': c.doc_freq, 'Avg_Score': c.avg_score}
            for c in result.concepts
        ])
    
    if result.keywords:
        result.keyword_df = pd.DataFrame([
            {'Keyword': k.name, 'Count': k.count, 'Doc_Freq': k.doc_freq, 'Avg_Score': k.avg_score}
            for k in result.keywords
        ])
    
    if result.topics:
        result.topic_df = pd.DataFrame([
            {'Topic': t.name, 'Count': t.count, 'Doc_Freq': t.doc_freq, 'Avg_Score': t.avg_score}
            for t in result.topics
        ])
    
    # Compute co-occurrence
    cooc_col = None
    for col_name in ['concepts.display_name', 'keywords.display_name', 'Author Keywords']:
        if col_name in df.columns:
            cooc_col = col_name
            break
    
    if cooc_col:
        sep = '|' if 'display_name' in cooc_col else ';'
        result.cooccurrence_matrix = compute_cooccurrence_matrix(
            df, cooc_col, top_n=30, separator=sep
        )
        
        # Temporal trends
        result.temporal_trends = compute_temporal_trends(
            df, cooc_col, top_n=15, separator=sep
        )
    
    result.data_source = "dataset"
    
    if verbose:
        print(f"  ✓ Analysis complete")
        print(f"    Unique concepts: {result.n_unique_concepts}")
        print(f"    Unique keywords: {result.n_unique_keywords}")
        if result.ngrams:
            print(f"    N-grams extracted: {len(result.ngrams)}")
    
    return result


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_top_concepts(
    result: ConceptExtractionResult,
    n: int = 20,
    concept_type: str = "concepts",
    ax=None,
    figsize: Tuple[int, int] = (10, 8),
    color: str = '#3498db',
):
    """
    Plot top concepts/keywords as horizontal bar chart.
    
    Parameters
    ----------
    result : ConceptExtractionResult
        Analysis result.
    n : int
        Number of top items.
    concept_type : str
        "concepts", "keywords", "topics", or "ngrams"
    ax : matplotlib.axes.Axes
        Axes to plot on.
    figsize : tuple
        Figure size.
    color : str
        Bar color.
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Get data
    if concept_type == "concepts":
        items = result.concepts[:n]
        title = f"Top {n} Concepts"
    elif concept_type == "keywords":
        items = result.keywords[:n]
        title = f"Top {n} Keywords"
    elif concept_type == "topics":
        items = result.topics[:n]
        title = f"Top {n} Topics"
    else:
        items = result.ngrams[:n]
        title = f"Top {n} N-grams (TF-IDF)"
    
    if not items:
        ax.text(0.5, 0.5, f"No {concept_type} data available", 
               ha='center', va='center', transform=ax.transAxes)
        return fig
    
    names = [item.name[:40] + '...' if len(item.name) > 40 else item.name for item in items]
    counts = [item.count for item in items]
    
    y_pos = list(range(len(names)))
    ax.barh(y_pos, counts, color=color, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Frequency")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(False)
    
    plt.tight_layout()
    return fig


def plot_concept_cooccurrence(
    result: ConceptExtractionResult,
    ax=None,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'Blues',
):
    """
    Plot concept co-occurrence heatmap.
    
    Parameters
    ----------
    result : ConceptExtractionResult
        Analysis result.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    figsize : tuple
        Figure size.
    cmap : str
        Colormap.
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    
    if result.cooccurrence_matrix is None or result.cooccurrence_matrix.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No co-occurrence data available",
               ha='center', va='center', transform=ax.transAxes)
        return fig
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    matrix = result.cooccurrence_matrix
    
    # Shorten labels
    labels = [l[:25] + '...' if len(l) > 25 else l for l in matrix.columns]
    
    im = ax.imshow(matrix.values, cmap=cmap, aspect='auto')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Concept Co-occurrence Matrix")
    
    plt.colorbar(im, ax=ax, label='Co-occurrence Count')
    plt.tight_layout()
    
    return fig


def plot_temporal_trends(
    result: ConceptExtractionResult,
    ax=None,
    figsize: Tuple[int, int] = (12, 6),
    top_n: int = 10,
):
    """
    Plot temporal trends of top concepts.
    
    Parameters
    ----------
    result : ConceptExtractionResult
        Analysis result.
    ax : matplotlib.axes.Axes
        Axes to plot on.
    figsize : tuple
        Figure size.
    top_n : int
        Number of concepts to show.
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    
    if result.temporal_trends is None or result.temporal_trends.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No temporal trend data available",
               ha='center', va='center', transform=ax.transAxes)
        return fig
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    trends = result.temporal_trends
    
    # Get top columns by total
    col_sums = trends.sum().sort_values(ascending=False)
    top_cols = col_sums.head(top_n).index.tolist()
    
    for col in top_cols:
        label = col[:30] + '...' if len(col) > 30 else col
        ax.plot(trends.index, trends[col], marker='o', label=label, markersize=4)
    
    ax.set_xlabel("Year")
    ax.set_ylabel("Document Count")
    ax.set_title("Concept Trends Over Time")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(False)
    
    plt.tight_layout()
    return fig
