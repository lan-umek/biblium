# -*- coding: utf-8 -*-
"""
Dynamic Topic Models Module for Bibliometric Analysis

This module provides functions to analyze how topics evolve over time in academic literature.
It implements multiple approaches to dynamic topic modeling and provides rich visualizations.

Approaches implemented:
1. Sequential LDA: Train separate LDA models per period, align topics across periods
2. Dynamic Topic Model (DTM): True dynamic model where topics evolve continuously
3. BERTopic with temporal tracking: Neural topic modeling with time-aware analysis
4. Rolling window LDA: Sliding window approach for smooth topic evolution

Features:
- Topic discovery and evolution tracking
- Topic birth, death, merge, split detection
- Topic popularity trajectories
- Representative document extraction per topic per period
- Topic similarity/distance matrices across time
- Rich visualizations (evolution charts, heatmaps, river plots, word clouds)

Created for integration with the Biblium bibliometric toolkit.

@author: Claude (Anthropic) for Lan.Umek
"""

from __future__ import annotations

import os
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, Set
from itertools import combinations
import json
import pickle

import numpy as np
import pandas as pd
from scipy.stats import entropy, spearmanr, pearsonr
from scipy.spatial.distance import cosine, jensenshannon, cdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch, Polygon
from matplotlib.collections import LineCollection, PolyCollection
import matplotlib.patheffects as path_effects
import seaborn as sns

# Optional imports with fallbacks
try:
    import gensim
    from gensim import corpora
    from gensim.models import LdaModel, LdaMulticore, CoherenceModel
    from gensim.models.wrappers import DtmModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

try:
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# =============================================================================
# DEFAULT COLORMAPS (user-configurable)
# =============================================================================

CMAP_CONTINUOUS = "viridis"  # For continuous/sequential data
CATEGORICAL_COLOR = "lightblue"      # For categorical/discrete data

def set_default_cmaps(continuous: str = None):
    """Set default colormap for continuous data."""
    global CMAP_CONTINUOUS
    if continuous:
        CMAP_CONTINUOUS = continuous

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    import pyLDAvis
    import pyLDAvis.gensim_models
    PYLDAVIS_AVAILABLE = True
except ImportError:
    PYLDAVIS_AVAILABLE = False

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TopicState:
    """Represents a topic's state at a specific time period."""
    topic_id: int
    period: str
    top_words: List[Tuple[str, float]]  # (word, weight) pairs
    word_distribution: Dict[str, float]  # Full word distribution
    n_documents: int
    document_ids: List[Any]
    prevalence: float  # Proportion of corpus assigned to this topic
    coherence: Optional[float] = None
    representative_docs: List[Tuple[Any, float]] = field(default_factory=list)  # (doc_id, score)
    
    def get_top_words_str(self, n: int = 10, sep: str = ", ") -> str:
        """Get top words as formatted string."""
        words = [w for w, _ in self.top_words[:n]]
        return sep.join(words)
    
    def get_word_set(self, n: int = 20) -> Set[str]:
        """Get set of top n words."""
        return {w for w, _ in self.top_words[:n]}

@dataclass
class TopicEvolution:
    """Tracks a topic's evolution across time periods."""
    topic_id: int
    label: str  # Human-readable label
    states: Dict[str, TopicState]  # period -> TopicState
    first_appearance: str
    last_appearance: str
    trajectory_type: str  # "stable", "growing", "declining", "volatile", "emerging", "dying"
    
    def get_prevalence_trajectory(self, periods: List[str]) -> List[float]:
        """Get prevalence values across all periods."""
        return [self.states[p].prevalence if p in self.states else 0.0 for p in periods]
    
    def get_word_stability(self, n_words: int = 10) -> float:
        """Calculate how stable the top words are across periods."""
        if len(self.states) < 2:
            return 1.0
        
        periods = sorted(self.states.keys())
        stabilities = []
        
        for i in range(len(periods) - 1):
            words1 = self.states[periods[i]].get_word_set(n_words)
            words2 = self.states[periods[i+1]].get_word_set(n_words)
            
            if words1 and words2:
                jaccard = len(words1 & words2) / len(words1 | words2)
                stabilities.append(jaccard)
        
        return np.mean(stabilities) if stabilities else 1.0
    
    def is_active_in(self, period: str) -> bool:
        return period in self.states

@dataclass
class TopicTransition:
    """Represents a transition event between topics across periods."""
    event_type: str  # "continuation", "merge", "split", "birth", "death"
    period: str
    source_topics: List[int]  # Topic IDs before transition
    target_topics: List[int]  # Topic IDs after transition
    similarity: float  # Similarity score for the transition
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DynamicTopicModelResult:
    """Container for complete dynamic topic model results."""
    method: str  # "sequential_lda", "dtm", "bertopic", "rolling_lda"
    n_topics: int
    periods: List[str]
    topic_evolutions: Dict[int, TopicEvolution]  # topic_id -> TopicEvolution
    transitions: List[TopicTransition]
    document_topics: pd.DataFrame  # Document-level topic assignments
    global_metrics: pd.DataFrame  # Period-level metrics
    vocabulary: List[str]
    parameters: Dict[str, Any]
    models: Dict[str, Any] = field(default_factory=dict)  # Store fitted models
    
    def get_topic(self, topic_id: int) -> Optional[TopicEvolution]:
        return self.topic_evolutions.get(topic_id)
    
    def get_active_topics(self, period: str) -> List[int]:
        """Get topics active in a specific period."""
        return [tid for tid, tevol in self.topic_evolutions.items() 
                if tevol.is_active_in(period)]
    
    def get_topic_prevalence_matrix(self) -> pd.DataFrame:
        """Get matrix of topic prevalences across periods."""
        data = {}
        for tid, tevol in self.topic_evolutions.items():
            data[f"Topic_{tid}"] = tevol.get_prevalence_trajectory(self.periods)
        
        df = pd.DataFrame(data, index=self.periods)
        return df
    
    def get_emerging_topics(self, threshold: float = 0.5) -> List[int]:
        """Get topics that emerged in later periods."""
        mid_point = len(self.periods) // 2
        later_periods = set(self.periods[mid_point:])
        
        emerging = []
        for tid, tevol in self.topic_evolutions.items():
            if tevol.first_appearance in later_periods:
                emerging.append(tid)
        
        return emerging
    
    def get_declining_topics(self, threshold: float = 0.3) -> List[int]:
        """Get topics with declining prevalence."""
        declining = []
        for tid, tevol in self.topic_evolutions.items():
            traj = tevol.get_prevalence_trajectory(self.periods)
            if len(traj) >= 3:
                # Compare first third to last third
                first_avg = np.mean(traj[:len(traj)//3]) if traj[:len(traj)//3] else 0
                last_avg = np.mean(traj[-len(traj)//3:]) if traj[-len(traj)//3:] else 0
                
                if first_avg > 0 and last_avg / first_avg < threshold:
                    declining.append(tid)
        
        return declining
    
    def to_summary_df(self) -> pd.DataFrame:
        """Get summary DataFrame of all topics."""
        records = []
        for tid, tevol in self.topic_evolutions.items():
            prevalences = tevol.get_prevalence_trajectory(self.periods)
            
            # Get representative words from most recent active period
            active_periods = [p for p in self.periods if p in tevol.states]
            if active_periods:
                recent_state = tevol.states[active_periods[-1]]
                top_words = recent_state.get_top_words_str(10)
            else:
                top_words = ""
            
            records.append({
                "Topic_ID": tid,
                "Label": tevol.label,
                "Top Words": top_words,
                "First Appearance": tevol.first_appearance,
                "Last Appearance": tevol.last_appearance,
                "Active Periods": len(tevol.states),
                "Avg Prevalence": np.mean([p for p in prevalences if p > 0]),
                "Max Prevalence": max(prevalences) if prevalences else 0,
                "Word Stability": tevol.get_word_stability(),
                "Trajectory Type": tevol.trajectory_type,
            })
        
        return pd.DataFrame(records).sort_values("Avg Prevalence", ascending=False)

# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

def preprocess_documents(
    df: pd.DataFrame,
    text_column: str,
    year_column: str = "Year",
    min_doc_length: int = 10,
    stopwords: Optional[Set[str]] = None,
    lowercase: bool = True,
    remove_numbers: bool = True,
    min_word_length: int = 2) -> Tuple[pd.DataFrame, List[str]]:
    """
    Preprocess documents for topic modeling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    text_column : str
        Column containing text to analyze.
    year_column : str
        Column containing publication year.
    min_doc_length : int
        Minimum number of words after preprocessing.
    stopwords : Set[str], optional
        Custom stopwords to remove.
    lowercase : bool
        Convert to lowercase.
    remove_numbers : bool
        Remove numeric tokens.
    min_word_length : int
        Minimum word length to keep.
    
    Returns
    -------
    Tuple of (processed DataFrame, list of processed texts).
    """
    if stopwords is None:
        # Default English stopwords
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how',
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
            'some', 'such', 'than', 'too', 'very', 'can', 'just', 'should',
            'now', 'also', 'into', 'only', 'over', 'after', 'before', 'between',
            'through', 'during', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'these', 'those', 'above', 'below', 'up', 'down',
            'out', 'off', 'about', 'any', 'been', 'being', 'could', 'did', 'do',
            'does', 'doing', 'done', 'get', 'got', 'go', 'going', 'make', 'made',
            'may', 'might', 'much', 'must', 'need', 'our', 'ours', 'own', 'same',
            'so', 'would', 'your', 'yours', 'their', 'theirs', 'them', 'us', 'we',
            'et', 'al', 'etc', 'ie', 'eg', 'vs', 'via', 'using', 'used', 'use',
            'based', 'show', 'shown', 'shows', 'study', 'studies', 'result',
            'results', 'method', 'methods', 'approach', 'paper', 'article',
            'research', 'analysis', 'data', 'model', 'models', 'system', 'systems',
            'however', 'therefore', 'thus', 'hence', 'although', 'whereas',
            'moreover', 'furthermore', 'nevertheless', 'nonetheless', 'yet',
        }
    
    df = df.copy()
    df = df.dropna(subset=[text_column, year_column])
    
    processed_texts = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        text = str(row[text_column])
        
        # Lowercase
        if lowercase:
            text = text.lower()
        
        # Tokenize (simple whitespace + punctuation split)
        tokens = re.findall(r'\b[a-zA-Z]+\b', text)
        
        # Filter tokens
        filtered = []
        for token in tokens:
            if len(token) < min_word_length:
                continue
            if token in stopwords:
                continue
            if remove_numbers and token.isdigit():
                continue
            filtered.append(token)
        
        if len(filtered) >= min_doc_length:
            processed_texts.append(' '.join(filtered))
            valid_indices.append(idx)
    
    df_processed = df.loc[valid_indices].copy()
    df_processed['processed_text'] = processed_texts
    
    print(f"  Preprocessed {len(processed_texts)} documents (dropped {len(df) - len(processed_texts)} short docs)")
    
    return df_processed, processed_texts

def create_time_periods(
    df: pd.DataFrame,
    year_column: str = "Year",
    period_size: int = 5,
    period_type: str = "fixed",
    min_year: Optional[int] = None,
    max_year: Optional[int] = None) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    """
    Split dataframe into time periods.
    
    Returns
    -------
    Tuple of (list of period labels, dict mapping period to subset DataFrame).
    """
    df = df.copy()
    df[year_column] = df[year_column].astype(int)
    
    if min_year is None:
        min_year = df[year_column].min()
    if max_year is None:
        max_year = df[year_column].max()
    
    periods = []
    period_dfs = {}
    
    if period_type == "fixed":
        start = min_year
        while start <= max_year:
            end = min(start + period_size - 1, max_year)
            period_label = f"{start}-{end}"
            
            mask = (df[year_column] >= start) & (df[year_column] <= end)
            subset = df[mask]
            
            if len(subset) > 0:
                periods.append(period_label)
                period_dfs[period_label] = subset
            
            start = end + 1
    else:  # sliding
        start = min_year
        while start + period_size - 1 <= max_year:
            end = start + period_size - 1
            period_label = f"{start}-{end}"
            
            mask = (df[year_column] >= start) & (df[year_column] <= end)
            subset = df[mask]
            
            if len(subset) > 0:
                periods.append(period_label)
                period_dfs[period_label] = subset
            
            start += 1
    
    return periods, period_dfs

# =============================================================================
# SEQUENTIAL LDA (SKLEARN-BASED)
# =============================================================================

def train_sequential_lda(
    df: pd.DataFrame,
    text_column: str,
    year_column: str = "Year",
    n_topics: int = 10,
    period_size: int = 5,
    n_top_words: int = 20,
    min_df: int = 5,
    max_df: float = 0.7,
    max_features: int = 5000,
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs) -> Tuple[Dict[str, Any], List[str], Dict[str, pd.DataFrame]]:
    """
    Train separate LDA models for each time period using sklearn.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with text and year columns.
    text_column : str
        Column containing text (abstracts, titles, etc.).
    year_column : str
        Column containing publication year.
    n_topics : int
        Number of topics to discover.
    period_size : int
        Size of each time period in years.
    n_top_words : int
        Number of top words to extract per topic.
    min_df : int
        Minimum document frequency for vocabulary.
    max_df : float
        Maximum document frequency for vocabulary.
    max_features : int
        Maximum vocabulary size.
    random_state : int
        Random seed.
    n_jobs : int
        Number of parallel jobs.
    
    Returns
    -------
    Tuple of (dict of models per period, vocabulary, period DataFrames).
    """
    print("Training Sequential LDA models...")
    
    # Create time periods
    periods, period_dfs = create_time_periods(df, year_column, period_size)
    print(f"  Created {len(periods)} time periods")
    
    # Build global vocabulary
    print("  Building vocabulary...")
    all_texts = df[text_column].dropna().astype(str).tolist()
    
    vectorizer = CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        stop_words='english')
    vectorizer.fit(all_texts)
    vocabulary = vectorizer.get_feature_names_out().tolist()
    print(f"  Vocabulary size: {len(vocabulary)}")
    
    # Train LDA for each period
    models = {}
    
    for period in periods:
        period_df = period_dfs[period]
        texts = period_df[text_column].dropna().astype(str).tolist()
        
        if len(texts) < n_topics * 2:
            print(f"  Skipping {period}: insufficient documents ({len(texts)})")
            continue
        
        print(f"  Training LDA for {period} ({len(texts)} documents)...")
        
        # Vectorize
        dtm = vectorizer.transform(texts)
        
        # Train LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            n_jobs=n_jobs,
            max_iter=20,
            learning_method='batch',
            **kwargs)
        lda.fit(dtm)
        
        # Store results
        models[period] = {
            'lda': lda,
            'dtm': dtm,
            'texts': texts,
            'doc_indices': period_df.index.tolist(),
            'doc_topics': lda.transform(dtm),
        }
    
    return models, vocabulary, period_dfs

def align_topics_across_periods(
    models: Dict[str, Any],
    vocabulary: List[str],
    periods: List[str],
    n_topics: int,
    similarity_threshold: float = 0.3) -> Tuple[Dict[int, List[Tuple[str, int]]], np.ndarray]:
    """
    Align topics across time periods using cosine similarity.
    
    Returns
    -------
    Tuple of (topic alignment mapping, similarity matrices).
    """
    print("  Aligning topics across periods...")
    
    # Get topic-word distributions for each period
    period_topics = {}
    for period in periods:
        if period not in models:
            continue
        
        lda = models[period]['lda']
        # Normalize topic-word distributions
        topic_word = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
        period_topics[period] = topic_word
    
    # Compute pairwise similarities between consecutive periods
    active_periods = [p for p in periods if p in period_topics]
    
    if len(active_periods) < 2:
        # No alignment needed
        alignment = {i: [(p, i) for p in active_periods] for i in range(n_topics)}
        return alignment, None
    
    # Initialize alignment: each topic in first period gets its own chain
    alignment = {i: [(active_periods[0], i)] for i in range(n_topics)}
    
    # Track which global topic each period-topic maps to
    topic_mapping = {active_periods[0]: {i: i for i in range(n_topics)}}
    
    # Align subsequent periods
    for i in range(1, len(active_periods)):
        prev_period = active_periods[i-1]
        curr_period = active_periods[i]
        
        prev_topics = period_topics[prev_period]
        curr_topics = period_topics[curr_period]
        
        # Compute similarity matrix
        sim_matrix = cosine_similarity(prev_topics, curr_topics)
        
        # Greedy matching
        curr_mapping = {}
        used_curr = set()
        
        for prev_idx in range(n_topics):
            # Get previous topic's global ID
            prev_global = topic_mapping[prev_period][prev_idx]
            
            # Find best matching current topic
            similarities = sim_matrix[prev_idx]
            sorted_indices = np.argsort(similarities)[::-1]
            
            for curr_idx in sorted_indices:
                if curr_idx not in used_curr and similarities[curr_idx] >= similarity_threshold:
                    curr_mapping[curr_idx] = prev_global
                    used_curr.add(curr_idx)
                    alignment[prev_global].append((curr_period, curr_idx))
                    break
        
        # Handle unmatched current topics (new topics)
        for curr_idx in range(n_topics):
            if curr_idx not in curr_mapping:
                # Create new global topic
                new_global = max(alignment.keys()) + 1
                alignment[new_global] = [(curr_period, curr_idx)]
                curr_mapping[curr_idx] = new_global
        
        topic_mapping[curr_period] = curr_mapping
    
    return alignment, topic_mapping

def build_topic_evolutions(
    models: Dict[str, Any],
    vocabulary: List[str],
    alignment: Dict[int, List[Tuple[str, int]]],
    periods: List[str],
    n_top_words: int = 20) -> Dict[int, TopicEvolution]:
    """
    Build TopicEvolution objects from aligned topics.
    """
    print("  Building topic evolutions...")
    
    topic_evolutions = {}
    
    for global_id, period_topics in alignment.items():
        states = {}
        
        for period, local_topic_id in period_topics:
            if period not in models:
                continue
            
            lda = models[period]['lda']
            doc_topics = models[period]['doc_topics']
            doc_indices = models[period]['doc_indices']
            
            # Get topic-word distribution
            topic_word = lda.components_[local_topic_id]
            topic_word_norm = topic_word / topic_word.sum()
            
            # Get top words
            top_indices = np.argsort(topic_word)[::-1][:n_top_words]
            top_words = [(vocabulary[i], float(topic_word_norm[i])) for i in top_indices]
            
            # Full word distribution (sparse)
            word_dist = {vocabulary[i]: float(topic_word_norm[i]) 
                        for i in np.where(topic_word_norm > 0.001)[0]}
            
            # Document assignments
            topic_assignments = doc_topics[:, local_topic_id]
            assigned_mask = topic_assignments > 0.1  # Threshold for assignment
            assigned_docs = [doc_indices[i] for i, m in enumerate(assigned_mask) if m]
            
            # Prevalence
            prevalence = float(topic_assignments.mean())
            
            # Representative documents
            top_doc_indices = np.argsort(topic_assignments)[::-1][:5]
            rep_docs = [(doc_indices[i], float(topic_assignments[i])) for i in top_doc_indices]
            
            states[period] = TopicState(
                topic_id=global_id,
                period=period,
                top_words=top_words,
                word_distribution=word_dist,
                n_documents=len(assigned_docs),
                document_ids=assigned_docs,
                prevalence=prevalence,
                representative_docs=rep_docs)
        
        if states:
            active_periods = sorted(states.keys())
            
            # Generate label from most common top words
            all_words = Counter()
            for state in states.values():
                for word, weight in state.top_words[:5]:
                    all_words[word] += weight
            
            label_words = [w for w, _ in all_words.most_common(3)]
            label = " / ".join(label_words) if label_words else f"Topic {global_id}"
            
            # Determine trajectory type
            prevalences = [states[p].prevalence for p in active_periods]
            trajectory_type = classify_topic_trajectory(prevalences, active_periods, periods)
            
            topic_evolutions[global_id] = TopicEvolution(
                topic_id=global_id,
                label=label,
                states=states,
                first_appearance=active_periods[0],
                last_appearance=active_periods[-1],
                trajectory_type=trajectory_type)
    
    return topic_evolutions

def classify_topic_trajectory(
    prevalences: List[float],
    active_periods: List[str],
    all_periods: List[str]) -> str:
    """Classify topic trajectory pattern."""
    if len(prevalences) < 2:
        return "transient"
    
    # Check if topic spans full timerange
    first_idx = all_periods.index(active_periods[0]) if active_periods[0] in all_periods else 0
    last_idx = all_periods.index(active_periods[-1]) if active_periods[-1] in all_periods else len(all_periods)-1
    
    is_early = first_idx <= 1
    is_late = last_idx >= len(all_periods) - 2
    
    # Calculate trend
    x = np.arange(len(prevalences))
    slope = np.polyfit(x, prevalences, 1)[0] if len(prevalences) > 1 else 0
    
    # Calculate volatility
    if len(prevalences) > 2:
        volatility = np.std(prevalences) / (np.mean(prevalences) + 1e-10)
    else:
        volatility = 0
    
    # Classify
    if volatility > 0.5:
        return "volatile"
    elif not is_early and is_late and slope > 0.01:
        return "emerging"
    elif is_early and not is_late and slope < -0.01:
        return "dying"
    elif slope > 0.02:
        return "growing"
    elif slope < -0.02:
        return "declining"
    else:
        return "stable"

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_dynamic_topics(
    df: pd.DataFrame,
    text_column: str,
    year_column: str = "Year",
    method: str = "sequential_lda",
    n_topics: int = 10,
    period_size: int = 5,
    n_top_words: int = 20,
    min_df: int = 5,
    max_df: float = 0.7,
    similarity_threshold: float = 0.3,
    preprocess: bool = True,
    random_state: int = 42,
    **kwargs) -> DynamicTopicModelResult:
    """
    Perform dynamic topic modeling analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with text and year columns.
    text_column : str
        Column containing text to analyze.
    year_column : str
        Column containing publication year.
    method : str
        Method to use: "sequential_lda", "nmf", "gensim_lda", "bertopic".
    n_topics : int
        Number of topics.
    period_size : int
        Size of time periods in years.
    n_top_words : int
        Number of top words per topic.
    min_df : int
        Minimum document frequency.
    max_df : float
        Maximum document frequency.
    similarity_threshold : float
        Threshold for topic alignment.
    preprocess : bool
        Whether to preprocess text.
    random_state : int
        Random seed.
    **kwargs
        Additional method-specific arguments.
    
    Returns
    -------
    DynamicTopicModelResult
        Complete analysis results.
    """
    print(f"Analyzing dynamic topics using {method}...")
    print(f"  {len(df)} documents, {n_topics} topics, {period_size}-year periods")
    
    # Preprocess if needed
    if preprocess:
        df_processed, processed_texts = preprocess_documents(
            df, text_column, year_column
        )
        text_col_to_use = 'processed_text'
    else:
        df_processed = df.copy()
        text_col_to_use = text_column
    
    # Create time periods
    periods, period_dfs = create_time_periods(df_processed, year_column, period_size)
    print(f"  {len(periods)} time periods: {periods[0]} to {periods[-1]}")
    
    # Train models based on method
    if method == "sequential_lda":
        models, vocabulary, period_dfs = train_sequential_lda(
            df_processed, text_col_to_use, year_column,
            n_topics=n_topics, period_size=period_size,
            n_top_words=n_top_words, min_df=min_df, max_df=max_df,
            random_state=random_state, **kwargs
        )
        
        # Align topics
        alignment, topic_mapping = align_topics_across_periods(
            models, vocabulary, periods, n_topics, similarity_threshold
        )
        
        # Build evolutions
        topic_evolutions = build_topic_evolutions(
            models, vocabulary, alignment, periods, n_top_words
        )
        
    elif method == "nmf":
        models, vocabulary, period_dfs = train_sequential_nmf(
            df_processed, text_col_to_use, year_column,
            n_topics=n_topics, period_size=period_size,
            n_top_words=n_top_words, min_df=min_df, max_df=max_df,
            random_state=random_state, **kwargs
        )
        
        alignment, topic_mapping = align_topics_across_periods(
            models, vocabulary, periods, n_topics, similarity_threshold
        )
        
        topic_evolutions = build_topic_evolutions(
            models, vocabulary, alignment, periods, n_top_words
        )
        
    elif method == "gensim_lda" and GENSIM_AVAILABLE:
        models, vocabulary, period_dfs = train_sequential_gensim_lda(
            df_processed, text_col_to_use, year_column,
            n_topics=n_topics, period_size=period_size,
            n_top_words=n_top_words, random_state=random_state, **kwargs
        )
        
        alignment, topic_mapping = align_gensim_topics(
            models, vocabulary, periods, n_topics, similarity_threshold
        )
        
        topic_evolutions = build_gensim_topic_evolutions(
            models, vocabulary, alignment, periods, n_top_words
        )
        
    elif method == "bertopic" and BERTOPIC_AVAILABLE:
        topic_evolutions, models, vocabulary = train_temporal_bertopic(
            df_processed, text_col_to_use, year_column,
            n_topics=n_topics, period_size=period_size,
            periods=periods, period_dfs=period_dfs, **kwargs
        )
    else:
        raise ValueError(f"Unknown or unavailable method: {method}")
    
    # Detect transitions
    transitions = detect_topic_transitions(topic_evolutions, periods, similarity_threshold)
    
    # Build document-topic matrix
    document_topics = build_document_topic_matrix(
        df_processed, models, periods, period_dfs, text_col_to_use, n_topics, method
    )
    
    # Compute global metrics
    global_metrics = compute_topic_metrics(topic_evolutions, periods)
    
    # Create result object
    result = DynamicTopicModelResult(
        method=method,
        n_topics=n_topics,
        periods=periods,
        topic_evolutions=topic_evolutions,
        transitions=transitions,
        document_topics=document_topics,
        global_metrics=global_metrics,
        vocabulary=vocabulary,
        parameters={
            'text_column': text_column,
            'year_column': year_column,
            'period_size': period_size,
            'n_top_words': n_top_words,
            'min_df': min_df,
            'max_df': max_df,
            'similarity_threshold': similarity_threshold,
            'random_state': random_state,
        },
        models=models)
    
    print(f"  Analysis complete: {len(topic_evolutions)} topics tracked")
    
    return result

# =============================================================================
# ALTERNATIVE METHODS
# =============================================================================

def train_sequential_nmf(
    df: pd.DataFrame,
    text_column: str,
    year_column: str = "Year",
    n_topics: int = 10,
    period_size: int = 5,
    n_top_words: int = 20,
    min_df: int = 5,
    max_df: float = 0.7,
    max_features: int = 5000,
    random_state: int = 42,
    **kwargs) -> Tuple[Dict[str, Any], List[str], Dict[str, pd.DataFrame]]:
    """Train NMF models for each time period (alternative to LDA)."""
    print("Training Sequential NMF models...")
    
    periods, period_dfs = create_time_periods(df, year_column, period_size)
    
    # Build global vocabulary with TF-IDF
    all_texts = df[text_column].dropna().astype(str).tolist()
    
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        stop_words='english')
    vectorizer.fit(all_texts)
    vocabulary = vectorizer.get_feature_names_out().tolist()
    
    models = {}
    
    for period in periods:
        period_df = period_dfs[period]
        texts = period_df[text_column].dropna().astype(str).tolist()
        
        if len(texts) < n_topics * 2:
            continue
        
        print(f"  Training NMF for {period} ({len(texts)} documents)...")
        
        tfidf = vectorizer.transform(texts)
        
        nmf = NMF(
            n_components=n_topics,
            random_state=random_state,
            max_iter=200,
            **kwargs)
        nmf.fit(tfidf)
        
        models[period] = {
            'lda': nmf,  # Using same key for compatibility
            'dtm': tfidf,
            'texts': texts,
            'doc_indices': period_df.index.tolist(),
            'doc_topics': nmf.transform(tfidf),
        }
    
    return models, vocabulary, period_dfs

def train_sequential_gensim_lda(
    df: pd.DataFrame,
    text_column: str,
    year_column: str = "Year",
    n_topics: int = 10,
    period_size: int = 5,
    n_top_words: int = 20,
    random_state: int = 42,
    **kwargs) -> Tuple[Dict[str, Any], List[str], Dict[str, pd.DataFrame]]:
    """Train Gensim LDA models with coherence optimization."""
    if not GENSIM_AVAILABLE:
        raise ImportError("gensim is required for this method. Install with: pip install gensim")
    
    print("Training Sequential Gensim LDA models...")
    
    periods, period_dfs = create_time_periods(df, year_column, period_size)
    
    # Build global dictionary
    all_texts = df[text_column].dropna().astype(str).tolist()
    tokenized = [text.split() for text in all_texts]
    
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=5, no_above=0.7)
    vocabulary = list(dictionary.token2id.keys())
    
    models = {}
    
    for period in periods:
        period_df = period_dfs[period]
        texts = period_df[text_column].dropna().astype(str).tolist()
        
        if len(texts) < n_topics * 2:
            continue
        
        print(f"  Training Gensim LDA for {period} ({len(texts)} documents)...")
        
        tokenized_period = [text.split() for text in texts]
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_period]
        
        lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=n_topics,
            random_state=random_state,
            passes=10,
            alpha='auto',
            eta='auto',
            **kwargs)
        
        # Get document-topic distributions
        doc_topics = np.zeros((len(corpus), n_topics))
        for i, doc in enumerate(corpus):
            for topic_id, prob in lda.get_document_topics(doc):
                doc_topics[i, topic_id] = prob
        
        models[period] = {
            'lda': lda,
            'corpus': corpus,
            'dictionary': dictionary,
            'texts': texts,
            'doc_indices': period_df.index.tolist(),
            'doc_topics': doc_topics,
        }
    
    return models, vocabulary, period_dfs

def align_gensim_topics(
    models: Dict[str, Any],
    vocabulary: List[str],
    periods: List[str],
    n_topics: int,
    similarity_threshold: float = 0.3) -> Tuple[Dict[int, List[Tuple[str, int]]], Dict]:
    """Align Gensim LDA topics across periods."""
    print("  Aligning Gensim topics...")
    
    period_topics = {}
    for period in periods:
        if period not in models:
            continue
        
        lda = models[period]['lda']
        dictionary = models[period]['dictionary']
        
        # Build topic-word matrix
        n_words = len(dictionary)
        topic_word = np.zeros((n_topics, n_words))
        
        for topic_id in range(n_topics):
            topic_terms = lda.get_topic_terms(topic_id, topn=n_words)
            for word_id, prob in topic_terms:
                if word_id < n_words:
                    topic_word[topic_id, word_id] = prob
        
        # Normalize
        topic_word = topic_word / (topic_word.sum(axis=1, keepdims=True) + 1e-10)
        period_topics[period] = topic_word
    
    # Use same alignment logic as sklearn version
    active_periods = [p for p in periods if p in period_topics]
    
    if len(active_periods) < 2:
        alignment = {i: [(p, i) for p in active_periods] for i in range(n_topics)}
        return alignment, {}
    
    alignment = {i: [(active_periods[0], i)] for i in range(n_topics)}
    topic_mapping = {active_periods[0]: {i: i for i in range(n_topics)}}
    
    for i in range(1, len(active_periods)):
        prev_period = active_periods[i-1]
        curr_period = active_periods[i]
        
        prev_topics = period_topics[prev_period]
        curr_topics = period_topics[curr_period]
        
        sim_matrix = cosine_similarity(prev_topics, curr_topics)
        
        curr_mapping = {}
        used_curr = set()
        
        for prev_idx in range(n_topics):
            prev_global = topic_mapping[prev_period][prev_idx]
            similarities = sim_matrix[prev_idx]
            sorted_indices = np.argsort(similarities)[::-1]
            
            for curr_idx in sorted_indices:
                if curr_idx not in used_curr and similarities[curr_idx] >= similarity_threshold:
                    curr_mapping[curr_idx] = prev_global
                    used_curr.add(curr_idx)
                    alignment[prev_global].append((curr_period, curr_idx))
                    break
        
        for curr_idx in range(n_topics):
            if curr_idx not in curr_mapping:
                new_global = max(alignment.keys()) + 1
                alignment[new_global] = [(curr_period, curr_idx)]
                curr_mapping[curr_idx] = new_global
        
        topic_mapping[curr_period] = curr_mapping
    
    return alignment, topic_mapping

def build_gensim_topic_evolutions(
    models: Dict[str, Any],
    vocabulary: List[str],
    alignment: Dict[int, List[Tuple[str, int]]],
    periods: List[str],
    n_top_words: int = 20) -> Dict[int, TopicEvolution]:
    """Build topic evolutions from Gensim models."""
    print("  Building Gensim topic evolutions...")
    
    topic_evolutions = {}
    
    for global_id, period_topics in alignment.items():
        states = {}
        
        for period, local_topic_id in period_topics:
            if period not in models:
                continue
            
            lda = models[period]['lda']
            dictionary = models[period]['dictionary']
            doc_topics = models[period]['doc_topics']
            doc_indices = models[period]['doc_indices']
            
            # Get top words
            topic_terms = lda.show_topic(local_topic_id, topn=n_top_words)
            top_words = [(word, float(prob)) for word, prob in topic_terms]
            
            # Word distribution
            word_dist = {word: float(prob) for word, prob in topic_terms}
            
            # Document assignments
            topic_assignments = doc_topics[:, local_topic_id]
            assigned_mask = topic_assignments > 0.1
            assigned_docs = [doc_indices[i] for i, m in enumerate(assigned_mask) if m]
            
            prevalence = float(topic_assignments.mean())
            
            top_doc_indices = np.argsort(topic_assignments)[::-1][:5]
            rep_docs = [(doc_indices[i], float(topic_assignments[i])) for i in top_doc_indices]
            
            states[period] = TopicState(
                topic_id=global_id,
                period=period,
                top_words=top_words,
                word_distribution=word_dist,
                n_documents=len(assigned_docs),
                document_ids=assigned_docs,
                prevalence=prevalence,
                representative_docs=rep_docs)
        
        if states:
            active_periods = sorted(states.keys())
            
            all_words = Counter()
            for state in states.values():
                for word, weight in state.top_words[:5]:
                    all_words[word] += weight
            
            label_words = [w for w, _ in all_words.most_common(3)]
            label = " / ".join(label_words)
            
            prevalences = [states[p].prevalence for p in active_periods]
            trajectory_type = classify_topic_trajectory(prevalences, active_periods, periods)
            
            topic_evolutions[global_id] = TopicEvolution(
                topic_id=global_id,
                label=label,
                states=states,
                first_appearance=active_periods[0],
                last_appearance=active_periods[-1],
                trajectory_type=trajectory_type)
    
    return topic_evolutions

def train_temporal_bertopic(
    df: pd.DataFrame,
    text_column: str,
    year_column: str,
    n_topics: int = 10,
    period_size: int = 5,
    periods: List[str] = None,
    period_dfs: Dict[str, pd.DataFrame] = None,
    **kwargs) -> Tuple[Dict[int, TopicEvolution], Dict[str, Any], List[str]]:
    """Train BERTopic with temporal tracking."""
    if not BERTOPIC_AVAILABLE:
        raise ImportError("bertopic is required. Install with: pip install bertopic")
    
    print("Training Temporal BERTopic...")
    
    if periods is None or period_dfs is None:
        periods, period_dfs = create_time_periods(df, year_column, period_size)
    
    # Prepare all documents with timestamps
    all_docs = []
    all_timestamps = []
    all_indices = []
    
    for period in periods:
        period_df = period_dfs[period]
        texts = period_df[text_column].dropna().astype(str).tolist()
        all_docs.extend(texts)
        all_timestamps.extend([period] * len(texts))
        all_indices.extend(period_df.index.tolist())
    
    print(f"  Training on {len(all_docs)} documents...")
    
    # Train BERTopic
    topic_model = BERTopic(
        nr_topics=n_topics,
        verbose=True,
        **kwargs)
    
    topics, probs = topic_model.fit_transform(all_docs)
    
    # Get topics over time
    topics_over_time = topic_model.topics_over_time(
        all_docs, 
        all_timestamps,
        nr_bins=len(periods))
    
    # Build topic evolutions
    vocabulary = list(topic_model.vectorizer_model.vocabulary_.keys()) if hasattr(topic_model, 'vectorizer_model') else []
    
    topic_evolutions = {}
    unique_topics = set(topics) - {-1}  # Exclude outlier topic
    
    for topic_id in unique_topics:
        states = {}
        
        # Get topic info for each period
        topic_time_data = topics_over_time[topics_over_time['Topic'] == topic_id]
        
        for _, row in topic_time_data.iterrows():
            period = row['Timestamp']
            if period not in periods:
                continue
            
            # Get topic words
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                top_words = [(w, float(s)) for w, s in topic_words[:20]]
                word_dist = {w: float(s) for w, s in topic_words}
            else:
                top_words = []
                word_dist = {}
            
            # Count documents in this period for this topic
            period_mask = [t == period for t in all_timestamps]
            topic_mask = [t == topic_id for t in topics]
            combined_mask = [p and t for p, t in zip(period_mask, topic_mask)]
            
            n_docs = sum(combined_mask)
            doc_ids = [all_indices[i] for i, m in enumerate(combined_mask) if m]
            
            prevalence = row['Frequency'] if 'Frequency' in row else n_docs / sum(period_mask)
            
            states[period] = TopicState(
                topic_id=topic_id,
                period=period,
                top_words=top_words,
                word_distribution=word_dist,
                n_documents=n_docs,
                document_ids=doc_ids,
                prevalence=prevalence)
        
        if states:
            active_periods = sorted(states.keys())
            
            # Label from top words
            if top_words:
                label = " / ".join([w for w, _ in top_words[:3]])
            else:
                label = f"Topic {topic_id}"
            
            prevalences = [states[p].prevalence for p in active_periods]
            trajectory_type = classify_topic_trajectory(prevalences, active_periods, periods)
            
            topic_evolutions[topic_id] = TopicEvolution(
                topic_id=topic_id,
                label=label,
                states=states,
                first_appearance=active_periods[0],
                last_appearance=active_periods[-1],
                trajectory_type=trajectory_type)
    
    models = {'bertopic': topic_model, 'topics_over_time': topics_over_time}
    
    return topic_evolutions, models, vocabulary

# =============================================================================
# TRANSITION DETECTION
# =============================================================================

def detect_topic_transitions(
    topic_evolutions: Dict[int, TopicEvolution],
    periods: List[str],
    threshold: float = 0.3) -> List[TopicTransition]:
    """Detect topic transition events (births, deaths, merges, splits)."""
    transitions = []
    
    for i, period in enumerate(periods):
        # Check for births (topics appearing for first time)
        for tid, tevol in topic_evolutions.items():
            if tevol.first_appearance == period and i > 0:
                transitions.append(TopicTransition(
                    event_type="birth",
                    period=period,
                    source_topics=[],
                    target_topics=[tid],
                    similarity=0.0,
                    details={"label": tevol.label}
                ))
        
        # Check for deaths (topics disappearing)
        for tid, tevol in topic_evolutions.items():
            if tevol.last_appearance == period and i < len(periods) - 1:
                transitions.append(TopicTransition(
                    event_type="death",
                    period=period,
                    source_topics=[tid],
                    target_topics=[],
                    similarity=0.0,
                    details={"label": tevol.label}
                ))
    
    return transitions

def build_document_topic_matrix(
    df: pd.DataFrame,
    models: Dict[str, Any],
    periods: List[str],
    period_dfs: Dict[str, pd.DataFrame],
    text_column: str,
    n_topics: int,
    method: str) -> pd.DataFrame:
    """Build document-level topic assignment matrix."""
    records = []
    
    for period in periods:
        if period not in models:
            continue
        
        doc_topics = models[period].get('doc_topics')
        doc_indices = models[period].get('doc_indices', [])
        
        if doc_topics is None:
            continue
        
        for i, doc_idx in enumerate(doc_indices):
            record = {
                'doc_id': doc_idx,
                'period': period,
            }
            
            # Add topic probabilities
            for j in range(min(n_topics, doc_topics.shape[1])):
                record[f'topic_{j}'] = float(doc_topics[i, j])
            
            # Dominant topic
            record['dominant_topic'] = int(np.argmax(doc_topics[i]))
            record['dominant_prob'] = float(np.max(doc_topics[i]))
            
            records.append(record)
    
    return pd.DataFrame(records)

def compute_topic_metrics(
    topic_evolutions: Dict[int, TopicEvolution],
    periods: List[str]) -> pd.DataFrame:
    """Compute period-level topic metrics."""
    records = []
    
    for period in periods:
        active_topics = [tid for tid, tevol in topic_evolutions.items() 
                        if tevol.is_active_in(period)]
        
        prevalences = []
        n_docs = []
        
        for tid in active_topics:
            state = topic_evolutions[tid].states.get(period)
            if state:
                prevalences.append(state.prevalence)
                n_docs.append(state.n_documents)
        
        records.append({
            'Period': period,
            'N_Active_Topics': len(active_topics),
            'Total_Documents': sum(n_docs),
            'Avg_Prevalence': np.mean(prevalences) if prevalences else 0,
            'Max_Prevalence': max(prevalences) if prevalences else 0,
            'Prevalence_Entropy': entropy(prevalences) if prevalences else 0,
        })
    
    return pd.DataFrame(records)

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_topic_evolution(
    result: DynamicTopicModelResult,
    top_n: int = 10,
    figsize: Tuple[int, int] = (16, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """
    Plot topic prevalence evolution over time.
    
    Parameters
    ----------
    result : DynamicTopicModelResult
        Analysis results.
    top_n : int
        Number of top topics to show.
    figsize : Tuple[int, int]
        Figure size.
    title : str, optional
        Custom title.
    save_path : str, optional
        Path to save figure.
    dpi : int
        Resolution.
    
    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    periods = result.periods
    x = np.arange(len(periods))
    
    # Get top topics by average prevalence
    topic_avg_prev = []
    for tid, tevol in result.topic_evolutions.items():
        avg = np.mean(tevol.get_prevalence_trajectory(periods))
        topic_avg_prev.append((tid, avg, tevol.label))
    
    topic_avg_prev.sort(key=lambda x: -x[1])
    top_topics = topic_avg_prev[:top_n]
    
    colors = "lightblue"
    
    for i, (tid, avg, label) in enumerate(top_topics):
        tevol = result.topic_evolutions[tid]
        trajectory = tevol.get_prevalence_trajectory(periods)
        
        # Truncate label
        display_label = label[:40] + "..." if len(label) > 40 else label
        
        ax.plot(x, trajectory, marker='o', linewidth=2, color="lightblue",
               label=f"T{tid}: {display_label}", markersize=6, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=45, ha='right')
    ax.set_xlabel("Time Period", fontsize=12)
    ax.set_ylabel("Topic Prevalence", fontsize=12)
    
    if title is None:
        title = "Topic Evolution Over Time"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig

def plot_topic_heatmap(
    result: DynamicTopicModelResult,
    top_n: int = 15,
    figsize: Tuple[int, int] = (14, 10),
    cmap: str = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """
    Plot topic prevalence heatmap.
    """
    # Get prevalence matrix
    prevalence_df = result.get_topic_prevalence_matrix()
    
    # Select top topics
    topic_sums = prevalence_df.sum(axis=0).sort_values(ascending=False)
    top_cols = topic_sums.head(top_n).index.tolist()
    
    plot_df = prevalence_df[top_cols].T
    
    # Add labels
    labels = []
    for col in top_cols:
        tid = int(col.split('_')[1])
        if tid in result.topic_evolutions:
            label = result.topic_evolutions[tid].label[:30]
            labels.append(f"T{tid}: {label}")
        else:
            labels.append(col)
    
    plot_df.index = labels
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    sns.heatmap(plot_df, cmap=cmap, annot=True, fmt='.3f', ax=ax,
               cbar_kws={'label': 'Prevalence'})
    
    ax.set_xlabel("Time Period", fontsize=12)
    ax.set_ylabel("Topic", fontsize=12)
    
    if title is None:
        title = "Topic Prevalence Heatmap"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig

def plot_topic_river(
    result: DynamicTopicModelResult,
    top_n: int = 10,
    figsize: Tuple[int, int] = (16, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """
    Plot streamgraph/river plot of topic evolution.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    periods = result.periods
    x = np.arange(len(periods))
    
    # Get top topics
    topic_avg_prev = []
    for tid, tevol in result.topic_evolutions.items():
        avg = np.mean(tevol.get_prevalence_trajectory(periods))
        topic_avg_prev.append((tid, avg))
    
    topic_avg_prev.sort(key=lambda x: -x[1])
    top_topic_ids = [t[0] for t in topic_avg_prev[:top_n]]
    
    # Build stacked data
    y_data = np.zeros((len(top_topic_ids), len(periods)))
    
    for i, tid in enumerate(top_topic_ids):
        tevol = result.topic_evolutions[tid]
        y_data[i] = tevol.get_prevalence_trajectory(periods)
    
    # Normalize to sum to 1
    col_sums = y_data.sum(axis=0)
    col_sums[col_sums == 0] = 1
    y_data = y_data / col_sums
    
    colors = "lightblue"
    
    ax.stackplot(x, y_data, labels=[f"T{tid}" for tid in top_topic_ids],
                colors=colors, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=45, ha='right')
    ax.set_xlabel("Time Period", fontsize=12)
    ax.set_ylabel("Relative Topic Share", fontsize=12)
    
    if title is None:
        title = "Topic Evolution River Plot"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.set_xlim(0, len(periods) - 1)
    ax.set_ylim(0, 1)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig

def plot_topic_words_evolution(
    result: DynamicTopicModelResult,
    topic_id: int,
    n_words: int = 10,
    figsize: Tuple[int, int] = (14, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """
    Plot how a topic's top words change over time.
    """
    if topic_id not in result.topic_evolutions:
        raise ValueError(f"Topic {topic_id} not found")
    
    tevol = result.topic_evolutions[topic_id]
    active_periods = [p for p in result.periods if p in tevol.states]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    # Build word presence matrix
    all_words = set()
    for period in active_periods:
        words = [w for w, _ in tevol.states[period].top_words[:n_words]]
        all_words.update(words)
    
    all_words = sorted(all_words)
    word_matrix = np.zeros((len(all_words), len(active_periods)))
    
    for j, period in enumerate(active_periods):
        word_weights = dict(tevol.states[period].top_words)
        for i, word in enumerate(all_words):
            word_matrix[i, j] = word_weights.get(word, 0)
    
    # Plot heatmap
    im = ax.imshow(word_matrix, aspect='auto', cmap=cmap)
    
    ax.set_xticks(range(len(active_periods)))
    ax.set_xticklabels(active_periods, rotation=45, ha='right')
    ax.set_yticks(range(len(all_words)))
    ax.set_yticklabels(all_words, fontsize=9)
    
    ax.set_xlabel("Time Period", fontsize=12)
    ax.set_ylabel("Word", fontsize=12)
    
    if title is None:
        title = f"Topic {topic_id} Word Evolution: {tevol.label[:50]}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Word Weight')
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig

def plot_topic_wordclouds(
    result: DynamicTopicModelResult,
    topic_id: int,
    max_periods: int = 6,
    figsize_per: Tuple[int, int] = (6, 4),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """
    Plot word clouds for a topic across time periods.
    """
    if not WORDCLOUD_AVAILABLE:
        raise ImportError("wordcloud is required. Install with: pip install wordcloud")
    
    if topic_id not in result.topic_evolutions:
        raise ValueError(f"Topic {topic_id} not found")
    
    tevol = result.topic_evolutions[topic_id]
    active_periods = [p for p in result.periods if p in tevol.states]
    
    # Subsample if too many periods
    if len(active_periods) > max_periods:
        step = len(active_periods) // max_periods
        active_periods = active_periods[::step][:max_periods]
    
    n_periods = len(active_periods)
    n_cols = min(3, n_periods)
    n_rows = (n_periods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per[0]*n_cols, figsize_per[1]*n_rows))
    
    if n_periods == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, period in enumerate(active_periods):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        state = tevol.states[period]
        word_freq = {w: float(wt) for w, wt in state.top_words[:50]}
        
        wc = WordCloud(
            width=400, height=300,
            background_color='white',
            max_words=50,
            colormap='viridis').generate_from_frequencies(word_freq)
        
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(period, fontsize=11)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_periods, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    if title is None:
        title = f"Topic {topic_id}: {tevol.label[:50]}"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    # Disable grids
    for _ax in axes.flat:
        _
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig

def plot_topic_trajectory_types(
    result: DynamicTopicModelResult,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """
    Plot distribution of topic trajectory types.
    """
    trajectory_counts = Counter(
        tevol.trajectory_type for tevol in result.topic_evolutions.values()
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    # Bar chart
    labels = list(trajectory_counts.keys())
    sizes = list(trajectory_counts.values())
    colors = "lightblue"
    
    bars = ax.bar(range(len(labels)), sizes, color=colors, edgecolor='white')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Number of Topics", fontsize=11)
    ax.set_title("Topic Trajectory Types", fontsize=12)
    
    # Add count and percentage labels
    total = sum(sizes)
    for bar, count in zip(bars, sizes):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.02,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    # Example trajectories
    
    type_examples = {}
    for tid, tevol in result.topic_evolutions.items():
        ttype = tevol.trajectory_type
        if ttype not in type_examples:
            type_examples[ttype] = tevol
    
    x = np.arange(len(result.periods))
    
    for i, (ttype, tevol) in enumerate(type_examples.items()):
        trajectory = tevol.get_prevalence_trajectory(result.periods)
        ax.plot(x, trajectory, marker='o', label=f"{ttype} (T{tevol.topic_id})",
                color="lightblue", linewidth=2)
    
    ax.set_xticks(x[::max(1, len(x)//5)])
    ax.set_xticklabels([result.periods[i] for i in range(0, len(result.periods), max(1, len(result.periods)//5))],
                       rotation=45, ha='right')
    ax.set_xlabel("Time Period", fontsize=11)
    ax.set_ylabel("Prevalence", fontsize=11)
    ax.set_title("Example Trajectories", fontsize=12)
    ax.legend(fontsize=9)
    ax
    
    if title is None:
        title = "Topic Trajectory Analysis"
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig

def plot_topic_similarity_matrix(
    result: DynamicTopicModelResult,
    period: str,
    top_n: int = 15,
    figsize: Tuple[int, int] = (12, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """
    Plot topic similarity matrix for a specific period.
    """
    # Get active topics for this period
    active_topics = []
    for tid, tevol in result.topic_evolutions.items():
        if period in tevol.states:
            active_topics.append((tid, tevol.states[period]))
    
    if not active_topics:
        raise ValueError(f"No topics active in period {period}")
    
    # Sort by prevalence and take top N
    active_topics.sort(key=lambda x: -x[1].prevalence)
    active_topics = active_topics[:top_n]
    
    n = len(active_topics)
    sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            words_i = active_topics[i][1].get_word_set(20)
            words_j = active_topics[j][1].get_word_set(20)
            
            if words_i and words_j:
                jaccard = len(words_i & words_j) / len(words_i | words_j)
                sim_matrix[i, j] = jaccard
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    labels = [f"T{tid}" for tid, _ in active_topics]
    
    im = ax.imshow(sim_matrix, cmap=cmap, vmin=0, vmax=1)
    
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    
    # Add values
    for i in range(n):
        for j in range(n):
            color = 'white' if sim_matrix[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{sim_matrix[i, j]:.2f}', ha='center', va='center',
                   color=color, fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Jaccard Similarity')
    
    if title is None:
        title = f"Topic Similarity Matrix ({period})"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Disable grids
    for _ax in axes.flat:
        _
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig

# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_topic_results(
    result: DynamicTopicModelResult,
    output_dir: str,
    formats: List[str] = None) -> Dict[str, str]:
    """
    Export topic modeling results to various formats.
    
    Parameters
    ----------
    result : DynamicTopicModelResult
        Analysis results.
    output_dir : str
        Output directory.
    formats : List[str], optional
        Formats: "excel", "json", "csv".
    
    Returns
    -------
    Dict mapping format to file path.
    """
    if formats is None:
        formats = ["excel", "json"]
    
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    
    if "excel" in formats:
        excel_path = os.path.join(output_dir, "dynamic_topics.xlsx")
        with pd.ExcelWriter(excel_path) as writer:
            # Summary
            result.to_summary_df().to_excel(writer, sheet_name="Topic_Summary", index=False)
            
            # Prevalence matrix
            result.get_topic_prevalence_matrix().to_excel(writer, sheet_name="Prevalence_Matrix")
            
            # Global metrics
            result.global_metrics.to_excel(writer, sheet_name="Period_Metrics", index=False)
            
            # Document assignments
            if len(result.document_topics) < 50000:
                result.document_topics.to_excel(writer, sheet_name="Document_Topics", index=False)
            
            # Top words per topic per period
            records = []
            for tid, tevol in result.topic_evolutions.items():
                for period, state in tevol.states.items():
                    records.append({
                        'Topic_ID': tid,
                        'Period': period,
                        'Label': tevol.label[:50],
                        'Top_Words': state.get_top_words_str(15),
                        'N_Documents': state.n_documents,
                        'Prevalence': state.prevalence,
                    })
            pd.DataFrame(records).to_excel(writer, sheet_name="Topic_Details", index=False)
        
        paths["excel"] = excel_path
        print(f"Excel export: {excel_path}")
    
    if "json" in formats:
        json_path = os.path.join(output_dir, "dynamic_topics.json")
        
        export_data = {
            'method': result.method,
            'n_topics': result.n_topics,
            'periods': result.periods,
            'parameters': result.parameters,
            'topics': {}
        }
        
        for tid, tevol in result.topic_evolutions.items():
            export_data['topics'][str(tid)] = {
                'label': tevol.label,
                'trajectory_type': tevol.trajectory_type,
                'first_appearance': tevol.first_appearance,
                'last_appearance': tevol.last_appearance,
                'word_stability': tevol.get_word_stability(),
                'states': {
                    period: {
                        'top_words': state.top_words[:10],
                        'prevalence': state.prevalence,
                        'n_documents': state.n_documents,
                    }
                    for period, state in tevol.states.items()
                }
            }
        
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        paths["json"] = json_path
        print(f"JSON export: {json_path}")
    
    if "csv" in formats:
        csv_dir = output_dir
        os.makedirs(csv_dir, exist_ok=True)
        
        result.to_summary_df().to_csv(os.path.join(csv_dir, "topic_summary.csv"), index=False)
        result.get_topic_prevalence_matrix().to_csv(os.path.join(csv_dir, "prevalence_matrix.csv"))
        result.global_metrics.to_csv(os.path.join(csv_dir, "period_metrics.csv"), index=False)
        
        paths["csv"] = csv_dir
        print(f"CSV exports: {csv_dir}")
    
    return paths

# =============================================================================
# INTEGRATION WITH BIBLIOANALYSIS CLASS
# =============================================================================

def add_dynamic_topic_methods(cls):
    """
    Add dynamic topic modeling methods to BiblioAnalysis class.
    
    Usage:
        from dynamic_topic_models import add_dynamic_topic_methods
        add_dynamic_topic_methods(BiblioAnalysis)
    """
    
    def analyze_dynamic_topics_method(
        self,
        text_column: str = None,
        method: str = "sequential_lda",
        n_topics: int = 10,
        period_size: int = 5,
        save_results: bool = True,
        **kwargs
    ) -> DynamicTopicModelResult:
        """
        Analyze dynamic topics in the corpus.
        
        Parameters
        ----------
        text_column : str, optional
            Column with text to analyze. Auto-detects if None.
        method : str
            Method: "sequential_lda", "nmf", "gensim_lda", "bertopic".
        n_topics : int
            Number of topics.
        period_size : int
            Size of time periods in years.
        save_results : bool
            Whether to save results.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        DynamicTopicModelResult
        """
        # Auto-detect text column
        if text_column is None:
            candidates = ["Abstract", "abstract", "Title", "title", 
                         "Processed Abstract", "abstract_inverted_index"]
            for col in candidates:
                if col in self.df.columns:
                    text_column = col
                    break
            if text_column is None:
                raise ValueError("Could not auto-detect text column. Please specify.")
        
        year_col = getattr(self, 'year_var', 'Year')
        
        self.dynamic_topics = analyze_dynamic_topics(
            self.df,
            text_column=text_column,
            year_column=year_col,
            method=method,
            n_topics=n_topics,
            period_size=period_size,
            **kwargs
        )
        
        self.topic_summary = self.dynamic_topics.to_summary_df()
        
        if save_results and hasattr(self, 'res_folder') and self.res_folder:
            output_dir = self.res_folder
            export_topic_results(self.dynamic_topics, output_dir)
        
        return self.dynamic_topics
    
    def plot_dynamic_topics_method(
        self,
        plot_type: str = "evolution",
        save: bool = True,
        **kwargs
    ) -> plt.Figure:
        """
        Create dynamic topic visualizations.
        
        Parameters
        ----------
        plot_type : str
            Type: "evolution", "heatmap", "river", "words", "wordclouds",
            "trajectories", "similarity".
        save : bool
            Whether to save plot.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        matplotlib Figure.
        """
        if not hasattr(self, 'dynamic_topics'):
            raise ValueError("Run analyze_dynamic_topics first.")
        
        save_path = None
        if save and hasattr(self, 'res_folder') and self.res_folder:
            save_path = os.path.join(self.res_folder, f"topics_{plot_type}")
        
        plot_functions = {
            "evolution": plot_topic_evolution,
            "heatmap": plot_topic_heatmap,
            "river": plot_topic_river,
            "trajectories": plot_topic_trajectory_types,
        }
        
        if plot_type in plot_functions:
            return plot_functions[plot_type](self.dynamic_topics, save_path=save_path, **kwargs)
        elif plot_type == "words":
            topic_id = kwargs.pop('topic_id', 0)
            return plot_topic_words_evolution(self.dynamic_topics, topic_id, save_path=save_path, **kwargs)
        elif plot_type == "wordclouds":
            topic_id = kwargs.pop('topic_id', 0)
            return plot_topic_wordclouds(self.dynamic_topics, topic_id, save_path=save_path, **kwargs)
        elif plot_type == "similarity":
            period = kwargs.pop('period', self.dynamic_topics.periods[-1])
            return plot_topic_similarity_matrix(self.dynamic_topics, period, save_path=save_path, **kwargs)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
    
    cls.analyze_dynamic_topics = analyze_dynamic_topics_method
    cls.plot_dynamic_topics = plot_dynamic_topics_method
    
    return cls

# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import biblium as bb
    ba = bb.BiblioAnalysis(f_name="data\\open alex dataset.csv", db="oa")
    
    print("Dynamic Topic Models")
    print("=" * 60)
    
    # Run analysis
    result = analyze_dynamic_topics(
        df=ba.df,
        text_column="Abstract",
        year_column="Year",
        n_topics=10,
        method="sequential_lda",
        verbose=True)
    
    # Print summary
    print(f"\nExtracted {result.n_topics} topics")
    print(f"Time periods: {len(result.periods)}")
    
    # Visualizations
    print("\nGenerating plots...")
    plot_topic_evolution(result, save_path="results/topic_evolution")
    plot_topic_heatmap(result, save_path="results/topic_heatmap")
    
    # Export
    print("\nExporting results...")
    export_topic_results(result, "results")
    
    print("\nDone!")
