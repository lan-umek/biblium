# -*- coding: utf-8 -*-
"""
Predictive & Prescriptive Analytics Module for Bibliometric Analysis

This module provides predictive models and prescriptive recommendations
for bibliometric data analysis.

Features implemented:
1. Citation Prediction Models - ML models predicting future citations
2. Collaboration Recommender - Suggest potential collaborators
3. Research Gap Analysis - Detect under-researched areas
4. Expert/Reviewer Finder - Find experts based on publication history
5. Hot Topic Prediction - Identify emerging research fronts
6. Venue Recommender - Suggest journals for manuscripts

Created for integration with the Biblium bibliometric toolkit.

@author: Claude (Anthropic) for Lan.Umek
"""

from __future__ import annotations

import os
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from itertools import combinations
import json

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from scipy.spatial.distance import cosine

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Optional imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

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

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CollaboratorRecommendation:
    """Recommended collaborator with reasoning."""
    author_id: str
    author_name: str
    similarity_score: float
    shared_topics: List[str]
    shared_coauthors: List[str]
    complementary_skills: List[str]
    collaboration_potential: str
    reasoning: str

@dataclass
class ResearchGap:
    """Identified research gap."""
    gap_id: int
    description: str
    topic_combination: List[str]
    existing_papers: int
    expected_papers: float
    gap_score: float
    potential_impact: str
    related_papers: List[Any]
    suggested_research_questions: List[str]

@dataclass
class PredictiveAnalysisResult:
    """Container for complete predictive analysis results."""
    citation_predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    model_performance: Dict[str, float]
    collaborator_recommendations: Dict[str, List[CollaboratorRecommendation]]
    research_gaps: List[ResearchGap]
    expert_profiles: pd.DataFrame
    hot_topics: pd.DataFrame
    parameters: Dict[str, Any]
    models: Dict[str, Any]
    
    def get_top_predictions(self, n: int = 20) -> pd.DataFrame:
        """Get papers with highest predicted citations."""
        if len(self.citation_predictions) == 0:
            return pd.DataFrame()
        return self.citation_predictions.nlargest(n, 'predicted_citations')
    
    def get_overperforming(self, threshold: float = 2.0) -> pd.DataFrame:
        """Get papers performing above prediction."""
        if len(self.citation_predictions) == 0:
            return pd.DataFrame()
        df = self.citation_predictions.copy()
        df['ratio'] = df['current_citations'] / (df['predicted_citations'] + 1)
        return df[df['ratio'] > threshold].sort_values('ratio', ascending=False)
    
    def get_underperforming(self, threshold: float = 0.5) -> pd.DataFrame:
        """Get papers performing below prediction."""
        if len(self.citation_predictions) == 0:
            return pd.DataFrame()
        df = self.citation_predictions.copy()
        df['ratio'] = df['current_citations'] / (df['predicted_citations'] + 1)
        return df[df['ratio'] < threshold].sort_values('ratio')

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def extract_citation_features(
    df: pd.DataFrame,
    id_col: str = "unique-id",
    title_col: str = "Title",
    abstract_col: str = "Abstract",
    authors_col: str = "Authors",
    year_col: str = "Year",
    citations_col: str = "Cited by",
    refs_col: str = "References",
    journal_col: str = "Source title",
    keywords_col: str = "Author Keywords",
    field_col: str = "Subject Area",
    sep: str = "; ",
    current_year: int = None,
    verbose: bool = True) -> pd.DataFrame:
    """Extract features for citation prediction."""
    if verbose:
        print("Extracting citation prediction features...")
    
    if current_year is None:
        current_year = pd.Timestamp.now().year
    
    # Pre-compute aggregate statistics
    author_stats = _compute_author_stats(df, authors_col, citations_col, sep)
    journal_stats = _compute_journal_stats(df, journal_col, citations_col)
    
    features = []
    
    for idx, row in df.iterrows():
        doc_id = row.get(id_col, idx)
        feature_dict = {'doc_id': doc_id}
        
        # Target
        citations = row.get(citations_col, 0)
        feature_dict['citations'] = int(citations) if pd.notna(citations) else 0
        
        # Paper age
        year = row.get(year_col)
        if pd.notna(year):
            feature_dict['paper_age'] = current_year - int(year)
            feature_dict['publication_year'] = int(year)
        else:
            feature_dict['paper_age'] = np.nan
            feature_dict['publication_year'] = np.nan
        
        # Author features
        authors = row.get(authors_col, "")
        if pd.notna(authors) and authors:
            if isinstance(authors, str):
                author_list = [a.strip() for a in authors.split(sep) if a.strip()]
            else:
                author_list = []
            
            feature_dict['n_authors'] = len(author_list)
            feature_dict['is_single_author'] = 1 if len(author_list) == 1 else 0
            feature_dict['is_large_team'] = 1 if len(author_list) > 5 else 0
            
            author_citations = [author_stats.get(a, {}).get('avg_citations', 0) for a in author_list]
            feature_dict['max_author_reputation'] = max(author_citations) if author_citations else 0
            feature_dict['avg_author_reputation'] = np.mean(author_citations) if author_citations else 0
        else:
            feature_dict['n_authors'] = 0
            feature_dict['is_single_author'] = 0
            feature_dict['is_large_team'] = 0
            feature_dict['max_author_reputation'] = 0
            feature_dict['avg_author_reputation'] = 0
        
        # Title features
        title = str(row.get(title_col, "")) if pd.notna(row.get(title_col)) else ""
        feature_dict['title_length'] = len(title)
        feature_dict['title_word_count'] = len(title.split())
        feature_dict['title_has_question'] = 1 if '?' in title else 0
        
        # Abstract features
        abstract = str(row.get(abstract_col, "")) if pd.notna(row.get(abstract_col)) else ""
        feature_dict['abstract_length'] = len(abstract)
        feature_dict['abstract_word_count'] = len(abstract.split())
        feature_dict['has_abstract'] = 1 if len(abstract) > 50 else 0
        
        # Reference features
        refs = row.get(refs_col, "")
        if pd.notna(refs) and refs:
            if isinstance(refs, str):
                ref_list = [r.strip() for r in refs.split(sep) if r.strip()]
            else:
                ref_list = list(refs) if hasattr(refs, '__iter__') else []
            feature_dict['n_references'] = len(ref_list)
        else:
            feature_dict['n_references'] = 0
        
        # Keyword features
        keywords = row.get(keywords_col, "")
        if pd.notna(keywords) and keywords:
            if isinstance(keywords, str):
                kw_list = [k.strip() for k in keywords.split(sep) if k.strip()]
            else:
                kw_list = []
            feature_dict['n_keywords'] = len(kw_list)
        else:
            feature_dict['n_keywords'] = 0
        
        # Journal features
        journal = row.get(journal_col, "")
        if pd.notna(journal) and journal in journal_stats:
            js = journal_stats[journal]
            feature_dict['journal_avg_citations'] = js.get('avg_citations', 0)
            feature_dict['journal_paper_count'] = js.get('n_papers', 0)
        else:
            feature_dict['journal_avg_citations'] = 0
            feature_dict['journal_paper_count'] = 0
        
        features.append(feature_dict)
    
    feature_df = pd.DataFrame(features)
    
    if verbose:
        print(f"  Extracted {len(feature_df.columns) - 2} features for {len(feature_df)} papers")
    
    return feature_df

def _compute_author_stats(df, authors_col, citations_col, sep):
    """Compute per-author statistics."""
    author_data = defaultdict(lambda: {'citations': [], 'papers': 0})
    
    for _, row in df.iterrows():
        authors = row.get(authors_col, "")
        citations = row.get(citations_col, 0)
        
        if pd.isna(authors) or not authors:
            continue
        
        if isinstance(authors, str):
            author_list = [a.strip() for a in authors.split(sep) if a.strip()]
        else:
            continue
        
        cit_val = int(citations) if pd.notna(citations) else 0
        
        for author in author_list:
            author_data[author]['citations'].append(cit_val)
            author_data[author]['papers'] += 1
    
    return {a: {'avg_citations': np.mean(d['citations']) if d['citations'] else 0,
                'n_papers': d['papers']} 
            for a, d in author_data.items()}

def _compute_journal_stats(df, journal_col, citations_col):
    """Compute per-journal statistics."""
    if journal_col not in df.columns:
        return {}
    
    journal_data = defaultdict(lambda: {'citations': [], 'papers': 0})
    
    for _, row in df.iterrows():
        journal = row.get(journal_col, "")
        citations = row.get(citations_col, 0)
        
        if pd.isna(journal) or not journal:
            continue
        
        cit_val = int(citations) if pd.notna(citations) else 0
        journal_data[journal]['citations'].append(cit_val)
        journal_data[journal]['papers'] += 1
    
    return {j: {'avg_citations': np.mean(d['citations']) if d['citations'] else 0,
                'n_papers': d['papers']} 
            for j, d in journal_data.items()}

# =============================================================================
# CITATION PREDICTION
# =============================================================================

def train_citation_predictor(
    feature_df: pd.DataFrame,
    target_col: str = "citations",
    model_type: str = "random_forest",
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
    verbose: bool = True) -> Tuple[Any, pd.DataFrame, Dict[str, float]]:
    """Train a citation prediction model."""
    if verbose:
        print(f"Training citation prediction model ({model_type})...")
    
    feature_cols = [c for c in feature_df.columns 
                   if c not in ['doc_id', target_col, 'citations']]
    
    # Remove columns with mostly NaN
    valid_cols = [c for c in feature_cols 
                 if feature_df[c].notna().sum() > len(feature_df) * 0.5]
    
    X = feature_df[valid_cols].fillna(feature_df[valid_cols].median())
    y = feature_df[target_col].fillna(0)
    
    # Log transform target
    y_log = np.log1p(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Select model
    models = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10, 
                                               random_state=random_state, n_jobs=-1),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                                       random_state=random_state),
    }
    model = models.get(model_type, models['random_forest'])
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_test = model.predict(X_test_scaled)
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred_test)
    
    performance = {
        'test_r2': r2_score(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
        'test_mae': mean_absolute_error(y_test_orig, y_pred_orig),
        'n_features': len(valid_cols),
    }
    
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='r2')
    performance['cv_r2_mean'] = cv_scores.mean()
    performance['cv_r2_std'] = cv_scores.std()
    
    if verbose:
        print(f"  Test R²: {performance['test_r2']:.3f}")
        print(f"  Test RMSE: {performance['test_rmse']:.2f}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        importances = np.ones(len(valid_cols)) / len(valid_cols)
    
    importance_df = pd.DataFrame({
        'feature': valid_cols,
        'importance': importances,
    }).sort_values('importance', ascending=False)
    importance_df['importance_normalized'] = importance_df['importance'] / importance_df['importance'].sum()
    
    model._scaler = scaler
    model._feature_cols = valid_cols
    
    return model, importance_df.reset_index(drop=True), performance

def predict_citations(model, feature_df: pd.DataFrame) -> pd.DataFrame:
    """Generate citation predictions."""
    X = feature_df[model._feature_cols].fillna(feature_df[model._feature_cols].median())
    X_scaled = model._scaler.transform(X)
    
    y_pred_log = model.predict(X_scaled)
    y_pred = np.expm1(y_pred_log)
    
    results = pd.DataFrame({
        'doc_id': feature_df['doc_id'],
        'current_citations': feature_df.get('citations', 0),
        'predicted_citations': np.round(y_pred, 1),
    })
    
    results['performance_ratio'] = results['current_citations'] / (results['predicted_citations'] + 1)
    results['performance_class'] = results['performance_ratio'].apply(
        lambda x: 'overperforming' if x > 1.5 else ('underperforming' if x < 0.5 else 'as_expected')
    )
    
    return results

# =============================================================================
# COLLABORATION RECOMMENDER
# =============================================================================

def build_collaboration_recommender(
    df: pd.DataFrame,
    authors_col: str = "Authors",
    keywords_col: str = "Author Keywords",
    citations_col: str = "Cited by",
    sep: str = "; ",
    min_papers: int = 3,
    verbose: bool = True) -> Dict[str, Any]:
    """Build a collaboration recommendation system."""
    if verbose:
        print("Building collaboration recommender...")
    
    author_keywords = defaultdict(list)
    author_coauthors = defaultdict(set)
    author_papers = defaultdict(list)
    
    for idx, row in df.iterrows():
        authors = row.get(authors_col, "")
        keywords = row.get(keywords_col, "")
        
        if pd.isna(authors) or not authors:
            continue
        
        if isinstance(authors, str):
            author_list = [a.strip() for a in authors.split(sep) if a.strip()]
        else:
            continue
        
        kw_list = []
        if pd.notna(keywords) and keywords and isinstance(keywords, str):
            kw_list = [k.strip().lower() for k in keywords.split(sep) if k.strip()]
        
        for author in author_list:
            author_keywords[author].extend(kw_list)
            author_papers[author].append(idx)
            for coauthor in author_list:
                if coauthor != author:
                    author_coauthors[author].add(coauthor)
    
    valid_authors = {a for a, papers in author_papers.items() if len(papers) >= min_papers}
    
    if verbose:
        print(f"  Found {len(valid_authors)} authors with {min_papers}+ papers")
    
    # Build author-keyword matrix
    all_keywords = set()
    for kws in author_keywords.values():
        all_keywords.update(kws)
    
    keyword_list = sorted(all_keywords)
    author_list = sorted(valid_authors)
    
    author_kw_matrix = np.zeros((len(author_list), len(keyword_list)))
    author_to_idx = {a: i for i, a in enumerate(author_list)}
    keyword_to_idx = {k: i for i, k in enumerate(keyword_list)}
    
    for author in author_list:
        author_idx = author_to_idx[author]
        kw_counts = Counter(author_keywords[author])
        for kw, count in kw_counts.items():
            if kw in keyword_to_idx:
                author_kw_matrix[author_idx, keyword_to_idx[kw]] = count
    
    # Normalize
    row_sums = author_kw_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    author_kw_matrix = author_kw_matrix / row_sums
    
    similarity_matrix = cosine_similarity(author_kw_matrix)
    
    return {
        'author_list': author_list,
        'author_to_idx': author_to_idx,
        'similarity_matrix': similarity_matrix,
        'author_coauthors': dict(author_coauthors),
        'author_keywords': dict(author_keywords),
    }

def get_collaboration_recommendations(
    recommender: Dict[str, Any],
    author_name: str,
    n_recommendations: int = 10,
    exclude_coauthors: bool = True) -> List[CollaboratorRecommendation]:
    """Get collaboration recommendations for an author."""
    author_list = recommender['author_list']
    author_to_idx = recommender['author_to_idx']
    similarity_matrix = recommender['similarity_matrix']
    author_coauthors = recommender['author_coauthors']
    author_keywords = recommender['author_keywords']
    
    if author_name not in author_to_idx:
        matches = [a for a in author_list if author_name.lower() in a.lower()]
        if matches:
            author_name = matches[0]
        else:
            return []
    
    author_idx = author_to_idx[author_name]
    similarities = similarity_matrix[author_idx]
    existing_coauthors = author_coauthors.get(author_name, set())
    author_kws = set(author_keywords.get(author_name, []))
    
    recommendations = []
    
    for i, candidate in enumerate(author_list):
        if candidate == author_name:
            continue
        if exclude_coauthors and candidate in existing_coauthors:
            continue
        
        sim_score = similarities[i]
        if sim_score < 0.1:
            continue
        
        candidate_kws = set(author_keywords.get(candidate, []))
        shared = list(author_kws & candidate_kws)[:5]
        complementary = list(candidate_kws - author_kws)[:5]
        
        candidate_coauthors = author_coauthors.get(candidate, set())
        shared_coauthors = list(existing_coauthors & candidate_coauthors)[:3]
        
        potential = "high" if sim_score > 0.5 else ("medium" if sim_score > 0.3 else "low")
        
        reasons = []
        if shared:
            reasons.append(f"shares interest in {', '.join(shared[:3])}")
        if complementary:
            reasons.append(f"brings expertise in {', '.join(complementary[:3])}")
        
        recommendations.append(CollaboratorRecommendation(
            author_id=candidate,
            author_name=candidate,
            similarity_score=round(sim_score, 3),
            shared_topics=shared,
            shared_coauthors=shared_coauthors,
            complementary_skills=complementary,
            collaboration_potential=potential,
            reasoning="; ".join(reasons) if reasons else "topic similarity"))
    
    recommendations.sort(key=lambda x: -x.similarity_score)
    return recommendations[:n_recommendations]

# =============================================================================
# RESEARCH GAP ANALYSIS
# =============================================================================

def analyze_research_gaps(
    df: pd.DataFrame,
    keywords_col: str = "Author Keywords",
    sep: str = "; ",
    min_keyword_freq: int = 5,
    top_n_gaps: int = 20,
    verbose: bool = True) -> List[ResearchGap]:
    """Identify research gaps based on keyword co-occurrence."""
    if verbose:
        print("Analyzing research gaps...")
    
    keyword_counts = Counter()
    keyword_papers = defaultdict(list)
    pair_counts = Counter()
    
    for idx, row in df.iterrows():
        keywords = row.get(keywords_col, "")
        
        if pd.isna(keywords) or not keywords:
            continue
        
        if isinstance(keywords, str):
            kw_list = [k.strip().lower() for k in keywords.split(sep) if k.strip()]
        else:
            continue
        
        for kw in kw_list:
            keyword_counts[kw] += 1
            keyword_papers[kw].append(idx)
        
        for i, kw1 in enumerate(kw_list):
            for kw2 in kw_list[i+1:]:
                pair = tuple(sorted([kw1, kw2]))
                pair_counts[pair] += 1
    
    frequent_keywords = {kw for kw, count in keyword_counts.items() if count >= min_keyword_freq}
    
    if verbose:
        print(f"  {len(frequent_keywords)} keywords with {min_keyword_freq}+ occurrences")
    
    total_papers = len(df)
    gaps = []
    gap_id = 0
    
    for kw1, kw2 in combinations(sorted(frequent_keywords), 2):
        pair = tuple(sorted([kw1, kw2]))
        observed = pair_counts.get(pair, 0)
        
        p1 = keyword_counts[kw1] / total_papers
        p2 = keyword_counts[kw2] / total_papers
        expected = p1 * p2 * total_papers
        
        if expected > 1:
            gap_score = (expected - observed) / expected
            
            if gap_score > 0.8 and observed < 5:
                combined_freq = keyword_counts[kw1] + keyword_counts[kw2]
                potential = "high" if combined_freq > 50 else ("medium" if combined_freq > 20 else "low")
                
                related = list(set(keyword_papers[kw1][:5] + keyword_papers[kw2][:5]))
                
                gaps.append(ResearchGap(
                    gap_id=gap_id,
                    description=f"Under-explored: '{kw1}' × '{kw2}'",
                    topic_combination=[kw1, kw2],
                    existing_papers=observed,
                    expected_papers=round(expected, 1),
                    gap_score=round(gap_score, 3),
                    potential_impact=potential,
                    related_papers=related,
                    suggested_research_questions=[
                        f"How does {kw1} relate to {kw2}?",
                        f"Can {kw1} methods apply to {kw2}?",
                    ]))
                gap_id += 1
    
    impact_order = {"high": 0, "medium": 1, "low": 2}
    gaps.sort(key=lambda x: (impact_order[x.potential_impact], -x.gap_score))
    
    if verbose:
        print(f"  Identified {len(gaps)} research gaps")
    
    return gaps[:top_n_gaps]

# =============================================================================
# EXPERT FINDER
# =============================================================================

def build_expert_database(
    df: pd.DataFrame,
    authors_col: str = "Authors",
    keywords_col: str = "Author Keywords",
    citations_col: str = "Cited by",
    year_col: str = "Year",
    sep: str = "; ",
    min_papers: int = 3,
    current_year: int = None,
    verbose: bool = True) -> pd.DataFrame:
    """Build expert database from publications."""
    if verbose:
        print("Building expert database...")
    
    if current_year is None:
        current_year = pd.Timestamp.now().year
    
    author_data = defaultdict(lambda: {
        'papers': [], 'keywords': [], 'citations': 0, 'years': []
    })
    
    for idx, row in df.iterrows():
        authors = row.get(authors_col, "")
        keywords = row.get(keywords_col, "")
        citations = row.get(citations_col, 0)
        year = row.get(year_col)
        
        if pd.isna(authors) or not authors:
            continue
        
        if isinstance(authors, str):
            author_list = [a.strip() for a in authors.split(sep) if a.strip()]
        else:
            continue
        
        kw_list = []
        if pd.notna(keywords) and keywords and isinstance(keywords, str):
            kw_list = [k.strip().lower() for k in keywords.split(sep) if k.strip()]
        
        cit_val = int(citations) if pd.notna(citations) else 0
        year_val = int(year) if pd.notna(year) else None
        
        for author in author_list:
            author_data[author]['papers'].append(idx)
            author_data[author]['keywords'].extend(kw_list)
            author_data[author]['citations'] += cit_val
            if year_val:
                author_data[author]['years'].append(year_val)
    
    experts = []
    
    for author, data in author_data.items():
        n_papers = len(data['papers'])
        if n_papers < min_papers:
            continue
        
        # H-index estimate
        citation_list = []
        for paper_idx in data['papers']:
            if paper_idx < len(df):
                paper_cit = df.iloc[paper_idx].get(citations_col, 0)
                citation_list.append(int(paper_cit) if pd.notna(paper_cit) else 0)
        
        citation_list.sort(reverse=True)
        h_index = sum(1 for i, c in enumerate(citation_list) if c >= i + 1)
        
        expertise_areas = Counter(data['keywords']).most_common(10)
        recent_papers = sum(1 for y in data['years'] if y and y >= current_year - 3)
        
        experts.append({
            'author_name': author,
            'expertise_areas': expertise_areas,
            'expertise_str': ", ".join([kw for kw, _ in expertise_areas[:5]]),
            'h_index_estimate': h_index,
            'total_papers': n_papers,
            'total_citations': data['citations'],
            'recent_activity': recent_papers,
        })
    
    if verbose:
        print(f"  Built profiles for {len(experts)} experts")
    
    return pd.DataFrame(experts)

def find_experts(
    expert_db: pd.DataFrame,
    query_topics: List[str],
    n_experts: int = 10,
    min_h_index: int = 0,
    require_recent: bool = True) -> pd.DataFrame:
    """Find experts for given topics."""
    query_lower = [t.lower() for t in query_topics]
    results = []
    
    for _, row in expert_db.iterrows():
        if row['h_index_estimate'] < min_h_index:
            continue
        if require_recent and row['recent_activity'] == 0:
            continue
        
        expertise = row['expertise_areas']
        if isinstance(expertise, str):
            expert_topics = [t.strip().lower() for t in expertise.split(",")]
        else:
            expert_topics = [t.lower() for t, _ in expertise]
        
        matches = sum(1 for qt in query_lower for et in expert_topics if qt in et or et in qt)
        
        if matches > 0:
            match_score = matches / len(query_topics)
            results.append({
                'author_name': row['author_name'],
                'match_score': round(match_score, 3),
                'expertise': row['expertise_str'],
                'h_index': row['h_index_estimate'],
                'total_papers': row['total_papers'],
                'recent_papers': row['recent_activity'],
            })
    
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values('match_score', ascending=False)
    
    return results_df.head(n_experts)

# =============================================================================
# HOT TOPIC PREDICTION
# =============================================================================

def predict_hot_topics(
    df: pd.DataFrame,
    keywords_col: str = "Author Keywords",
    year_col: str = "Year",
    citations_col: str = "Cited by",
    sep: str = "; ",
    lookback_years: int = 5,
    top_n: int = 20,
    verbose: bool = True) -> pd.DataFrame:
    """Predict emerging hot topics."""
    if verbose:
        print("Predicting hot topics...")
    
    current_year = pd.Timestamp.now().year
    start_year = current_year - lookback_years
    
    keyword_year_counts = defaultdict(lambda: defaultdict(int))
    keyword_year_citations = defaultdict(lambda: defaultdict(int))
    
    for _, row in df.iterrows():
        keywords = row.get(keywords_col, "")
        year = row.get(year_col)
        citations = row.get(citations_col, 0)
        
        if pd.isna(keywords) or pd.isna(year):
            continue
        
        year = int(year)
        if year < start_year:
            continue
        
        if isinstance(keywords, str):
            kw_list = [k.strip().lower() for k in keywords.split(sep) if k.strip()]
        else:
            continue
        
        cit_val = int(citations) if pd.notna(citations) else 0
        
        for kw in kw_list:
            keyword_year_counts[kw][year] += 1
            keyword_year_citations[kw][year] += cit_val
    
    years = list(range(start_year, current_year + 1))
    hot_topics = []
    
    for kw, year_counts in keyword_year_counts.items():
        counts = [year_counts.get(y, 0) for y in years]
        total_count = sum(counts)
        
        if total_count < 5:
            continue
        
        # Growth rate
        if len([c for c in counts if c > 0]) >= 3:
            x = np.arange(len(counts))
            slope = np.polyfit(x, counts, 1)[0]
            growth_rate = slope / (np.mean(counts) + 1)
        else:
            growth_rate = 0
        
        # Momentum
        recent = sum(counts[-2:])
        earlier = sum(counts[:-2]) if len(counts) > 2 else 1
        momentum = recent / (earlier + 1)
        
        # Hot score
        hot_score = 0.4 * min(growth_rate, 2) + 0.4 * min(momentum, 5) + 0.2 * min(total_count / 50, 1)
        
        # Trend classification
        if growth_rate > 0.2 and momentum > 1.5:
            trend = "rapidly_emerging"
        elif growth_rate > 0.1 and momentum > 1:
            trend = "emerging"
        elif growth_rate > 0:
            trend = "growing"
        elif growth_rate < -0.1:
            trend = "declining"
        else:
            trend = "stable"
        
        hot_topics.append({
            'keyword': kw,
            'hot_score': round(hot_score, 3),
            'growth_rate': round(growth_rate, 3),
            'momentum': round(momentum, 2),
            'total_papers': total_count,
            'recent_papers': recent,
            'trend': trend,
        })
    
    hot_df = pd.DataFrame(hot_topics)
    if len(hot_df) > 0:
        hot_df = hot_df.sort_values('hot_score', ascending=False)
    
    if verbose:
        print(f"  Analyzed {len(hot_df)} keywords")
    
    return hot_df.head(top_n)

# =============================================================================
# VENUE RECOMMENDER
# =============================================================================

def recommend_venues(
    df: pd.DataFrame,
    target_keywords: List[str],
    journal_col: str = "Source title",
    keywords_col: str = "Author Keywords",
    citations_col: str = "Cited by",
    sep: str = "; ",
    n_recommendations: int = 10,
    verbose: bool = True) -> pd.DataFrame:
    """Recommend publication venues."""
    if verbose:
        print("Recommending venues...")
    
    target_lower = [k.lower() for k in target_keywords]
    
    venue_data = defaultdict(lambda: {'keywords': [], 'citations': [], 'papers': 0})
    
    for _, row in df.iterrows():
        journal = row.get(journal_col, "")
        keywords = row.get(keywords_col, "")
        citations = row.get(citations_col, 0)
        
        if pd.isna(journal) or not journal:
            continue
        
        venue_data[journal]['papers'] += 1
        venue_data[journal]['citations'].append(int(citations) if pd.notna(citations) else 0)
        
        if pd.notna(keywords) and keywords and isinstance(keywords, str):
            kw_list = [k.strip().lower() for k in keywords.split(sep) if k.strip()]
            venue_data[journal]['keywords'].extend(kw_list)
    
    recommendations = []
    
    for venue, data in venue_data.items():
        if data['papers'] < 3:
            continue
        
        venue_kws = set(data['keywords'])
        matches = sum(1 for tk in target_lower for vk in venue_kws if tk in vk or vk in tk)
        
        topic_score = matches / (len(target_keywords) + 1)
        avg_citations = np.mean(data['citations']) if data['citations'] else 0
        
        combined_score = 0.6 * topic_score + 0.4 * min(avg_citations / 20, 1)
        
        if combined_score > 0.1:
            top_kws = [kw for kw, _ in Counter(data['keywords']).most_common(5)]
            recommendations.append({
                'venue': venue,
                'match_score': round(combined_score, 3),
                'topic_match': round(topic_score, 3),
                'avg_citations': round(avg_citations, 1),
                'total_papers': data['papers'],
                'top_keywords': ", ".join(top_kws),
            })
    
    rec_df = pd.DataFrame(recommendations)
    if len(rec_df) > 0:
        rec_df = rec_df.sort_values('match_score', ascending=False)
    
    if verbose:
        print(f"  Found {len(rec_df)} matching venues")
    
    return rec_df.head(n_recommendations)

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_predictive_analysis(
    df: pd.DataFrame,
    id_col: str = "unique-id",
    title_col: str = "Title",
    abstract_col: str = "Abstract",
    authors_col: str = "Authors",
    year_col: str = "Year",
    citations_col: str = "Cited by",
    refs_col: str = "References",
    journal_col: str = "Source title",
    keywords_col: str = "Author Keywords",
    field_col: str = "Subject Area",
    sep: str = "; ",
    train_citation_model: bool = True,
    build_recommender: bool = True,
    find_gaps: bool = True,
    build_experts: bool = True,
    predict_topics: bool = True,
    model_type: str = "random_forest",
    verbose: bool = True) -> PredictiveAnalysisResult:
    """Run comprehensive predictive analysis."""
    if verbose:
        print("=" * 50)
        print("Predictive & Prescriptive Analytics")
        print("=" * 50)
    
    citation_predictions = pd.DataFrame()
    feature_importance = pd.DataFrame()
    model_performance = {}
    recommender = {}
    research_gaps = []
    expert_db = pd.DataFrame()
    hot_topics = pd.DataFrame()
    models = {}
    
    # Citation prediction
    if train_citation_model:
        if verbose:
            print("\n--- Citation Prediction ---")
        try:
            feature_df = extract_citation_features(
                df, id_col, title_col, abstract_col, authors_col,
                year_col, citations_col, refs_col, journal_col,
                keywords_col, field_col, sep, verbose=verbose
            )
            
            model, feature_importance, model_performance = train_citation_predictor(
                feature_df, model_type=model_type, verbose=verbose
            )
            
            citation_predictions = predict_citations(model, feature_df)
            models['citation_predictor'] = model
        except Exception as e:
            if verbose:
                print(f"  Warning: Citation prediction failed: {e}")
    
    # Recommender
    if build_recommender:
        if verbose:
            print("\n--- Collaboration Recommender ---")
        try:
            recommender = build_collaboration_recommender(
                df, authors_col, keywords_col, sep=sep, verbose=verbose
            )
            models['recommender'] = recommender
        except Exception as e:
            if verbose:
                print(f"  Warning: Recommender failed: {e}")
    
    # Gaps
    if find_gaps:
        if verbose:
            print("\n--- Research Gap Analysis ---")
        try:
            research_gaps = analyze_research_gaps(df, keywords_col, sep, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"  Warning: Gap analysis failed: {e}")
    
    # Experts
    if build_experts:
        if verbose:
            print("\n--- Expert Database ---")
        try:
            expert_db = build_expert_database(
                df, authors_col, keywords_col, citations_col, year_col, sep, verbose=verbose
            )
        except Exception as e:
            if verbose:
                print(f"  Warning: Expert database failed: {e}")
    
    # Hot topics
    if predict_topics:
        if verbose:
            print("\n--- Hot Topic Prediction ---")
        try:
            hot_topics = predict_hot_topics(
                df, keywords_col, year_col, citations_col, sep, verbose=verbose
            )
        except Exception as e:
            if verbose:
                print(f"  Warning: Hot topics failed: {e}")
    
    result = PredictiveAnalysisResult(
        citation_predictions=citation_predictions,
        feature_importance=feature_importance,
        model_performance=model_performance,
        collaborator_recommendations={},
        research_gaps=research_gaps,
        expert_profiles=expert_db,
        hot_topics=hot_topics,
        parameters={'model_type': model_type},
        models=models)
    
    if verbose:
        print("\n" + "=" * 50)
        print("Analysis complete!")
    
    return result

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_citation_prediction_performance(
    result: PredictiveAnalysisResult,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None) -> plt.Figure:
    """Plot citation prediction performance."""
    if len(result.citation_predictions) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(False)
        ax.text(0.5, 0.5, "No predictions available", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    df = result.citation_predictions
    
    # Actual vs Predicted
    ax.scatter(df['current_citations'], df['predicted_citations'], alpha=0.5, s=30)
    max_val = max(df['current_citations'].max(), df['predicted_citations'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    ax.set_xlabel("Actual Citations")
    ax.set_ylabel("Predicted Citations")
    ax.set_title("Actual vs Predicted")
    ax.set_xscale('symlog', linthresh=1)
    ax.set_yscale('symlog', linthresh=1)
    
    # Residuals
    residuals = df['current_citations'] - df['predicted_citations']
    ax.hist(residuals.clip(-100, 100), bins=50, color="lightblue", edgecolor='white')
    ax.set_xlabel("Residual")
    ax.set_title("Residual Distribution")
    
    # Feature importance
    if len(result.feature_importance) > 0:
        top_feat = result.feature_importance.head(10)
        ax.barh(range(len(top_feat)), top_feat['importance_normalized'], color="lightblue")
        ax.set_yticks(range(len(top_feat)))
        ax.set_yticklabels(top_feat['feature'])
        ax.set_xlabel("Importance")
        ax.set_title("Top 10 Features")
        ax.invert_yaxis()
    
    # Performance classes
    class_counts = df['performance_class'].value_counts()
    colors = {'overperforming': '#27ae60', 'as_expected': '#3498db', 'underperforming': '#e74c3c'}
    bar_colors = [colors.get(c, '#95a5a6') for c in class_counts.index]
    ax.bar(range(len(class_counts)), class_counts.values, color=bar_colors)
    ax.set_xticks(range(len(class_counts)))
    ax.set_xticklabels(class_counts.index, rotation=45, ha='right')
    ax.set_ylabel("Papers")
    ax.set_title("Performance Classification")
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig

def plot_research_gaps(
    result: PredictiveAnalysisResult,
    n_gaps: int = 15,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
    dpi: int = 600) -> plt.Figure:
    """Plot research gaps."""
    if not result.research_gaps:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(False)
        ax.text(0.5, 0.5, "No gaps available", ha='center', va='center')
        ax.axis('off')
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    gaps = result.research_gaps[:n_gaps]
    
    # Gap scores
    labels = [f"{g.topic_combination[0][:12]}×{g.topic_combination[1][:12]}" for g in gaps]
    scores = [g.gap_score for g in gaps]
    
    ax.barh(range(len(gaps)), scores, color="lightblue")
    ax.set_yticks(range(len(gaps)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Gap Score")
    ax.set_title("Top Research Gaps")
    ax.invert_yaxis()
    
    # Expected vs Observed
    x = np.arange(len(gaps))
    ax.bar(x - 0.2, [g.expected_papers for g in gaps], 0.4, label='Expected', color="lightblue")
    ax.bar(x + 0.2, [g.existing_papers for g in gaps], 0.4, label='Observed', color="lightblue")
    ax.set_xticks(x)
    ax.set_xticklabels([f"G{i+1}" for i in range(len(gaps))], rotation=45)
    ax.set_ylabel("Papers")
    ax.set_title("Expected vs Observed")
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig

def plot_hot_topics(
    result: PredictiveAnalysisResult,
    n_topics: int = 15,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None) -> plt.Figure:
    """Plot hot topics analysis."""
    if len(result.hot_topics) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(False)
        ax.text(0.5, 0.5, "No hot topics available", ha='center', va='center')
        ax.axis('off')
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    df = result.hot_topics.head(n_topics)
    
    trend_colors = {
        'rapidly_emerging': '#e74c3c', 'emerging': '#f39c12',
        'growing': '#27ae60', 'stable': '#3498db', 'declining': '#95a5a6'
    }
    
    # Hot scores
    colors = [trend_colors.get(t, '#95a5a6') for t in df['trend']]
    ax.barh(range(len(df)), df['hot_score'], color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([k[:25] for k in df['keyword']], fontsize=9)
    ax.set_xlabel("Hot Score")
    ax.set_title("Topic Hot Scores")
    ax.invert_yaxis()
    
    # Growth vs Momentum
    scatter = ax.scatter(df['growth_rate'], df['momentum'], c=df['hot_score'], 
                         cmap=cmap, s=df['total_papers']*2, alpha=0.7)
    ax.set_xlabel("Growth Rate")
    ax.set_ylabel("Momentum")
    ax.set_title("Growth vs Momentum")
    plt.colorbar(scatter, ax=ax, label='Hot Score')
    
    # Trend distribution
    trend_counts = result.hot_topics['trend'].value_counts()
    bar_colors = [trend_colors.get(t, '#95a5a6') for t in trend_counts.index]
    ax.bar(range(len(trend_counts)), trend_counts.values, color=bar_colors)
    ax.set_xticks(range(len(trend_counts)))
    ax.set_xticklabels(trend_counts.index, rotation=45, ha='right')
    ax.set_ylabel("Topics")
    ax.set_title("Trend Distribution")
    
    # Emerging topics detail
    emerging = df[df['trend'].isin(['rapidly_emerging', 'emerging'])].head(8)
    if len(emerging) > 0:
        ax.barh(range(len(emerging)), emerging['recent_papers'], color="lightblue", label='Recent')
        ax.barh(range(len(emerging)), emerging['total_papers'], color="lightblue", alpha=0.3, label='Total')
        ax.set_yticks(range(len(emerging)))
        ax.set_yticklabels([k[:20] for k in emerging['keyword']], fontsize=9)
        ax.set_xlabel("Papers")
        ax.set_title("Emerging Topics")
        ax.legend()
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No emerging topics", ha='center', va='center')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches='tight')
    
    return fig

# =============================================================================
# EXPORT
# =============================================================================

def export_predictive_results(
    result: PredictiveAnalysisResult,
    output_dir: str) -> Dict[str, str]:
    """Export results to Excel."""
    os.makedirs(output_dir, exist_ok=True)
    
    excel_path = os.path.join(output_dir, "predictive_analysis.xlsx")
    
    with pd.ExcelWriter(excel_path) as writer:
        if len(result.citation_predictions) > 0:
            result.citation_predictions.to_excel(writer, sheet_name="Predictions", index=False)
        
        if len(result.feature_importance) > 0:
            result.feature_importance.to_excel(writer, sheet_name="Features", index=False)
        
        if result.research_gaps:
            gaps_df = pd.DataFrame([{
                'topics': f"{g.topic_combination[0]} × {g.topic_combination[1]}",
                'gap_score': g.gap_score,
                'existing': g.existing_papers,
                'expected': g.expected_papers,
                'impact': g.potential_impact,
            } for g in result.research_gaps])
            gaps_df.to_excel(writer, sheet_name="Gaps", index=False)
        
        if len(result.expert_profiles) > 0:
            cols = ['author_name', 'expertise_str', 'h_index_estimate', 
                   'total_papers', 'total_citations', 'recent_activity']
            cols = [c for c in cols if c in result.expert_profiles.columns]
            result.expert_profiles[cols].to_excel(writer, sheet_name="Experts", index=False)
        
        if len(result.hot_topics) > 0:
            cols = ['keyword', 'hot_score', 'growth_rate', 'momentum', 'total_papers', 'trend']
            result.hot_topics[cols].to_excel(writer, sheet_name="HotTopics", index=False)
    
    print(f"Exported to: {excel_path}")
    return {"excel": excel_path}

# =============================================================================
# CLASS INTEGRATION
# =============================================================================

def add_predictive_methods(cls):
    """Add predictive methods to BiblioAnalysis class."""
    
    def run_predictive_method(self, **kwargs) -> PredictiveAnalysisResult:
        id_col = getattr(self, 'id_var', 'unique-id')
        authors_col = getattr(self, 'authors_var', 'Authors')
        year_col = getattr(self, 'year_var', 'Year')
        citations_col = getattr(self, 'citations_var', 'Cited by')
        keywords_col = getattr(self, 'kw_var', 'Author Keywords')
        sep = getattr(self, 'default_separator', '; ')
        
        self.predictive_results = run_predictive_analysis(
            self.df, id_col=id_col, authors_col=authors_col,
            year_col=year_col, citations_col=citations_col,
            keywords_col=keywords_col, sep=sep, **kwargs
        )
        
        if hasattr(self, 'res_folder') and self.res_folder:
            export_predictive_results(
                self.predictive_results,
                self.res_folder
            )
        
        return self.predictive_results
    
    def get_recommendations_method(self, author: str, n: int = 10):
        if not hasattr(self, 'predictive_results'):
            raise ValueError("Run run_predictive_analysis first")
        recommender = self.predictive_results.models.get('recommender')
        if not recommender:
            raise ValueError("Recommender not built")
        return get_collaboration_recommendations(recommender, author, n)
    
    def find_experts_method(self, topics: List[str], n: int = 10):
        if not hasattr(self, 'predictive_results'):
            raise ValueError("Run run_predictive_analysis first")
        return find_experts(self.predictive_results.expert_profiles, topics, n)
    
    cls.run_predictive_analysis = run_predictive_method
    cls.get_collaborator_recommendations = get_recommendations_method
    cls.find_experts = find_experts_method
    
    return cls
# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import biblium as bb
    ba = bb.BiblioAnalysis(f_name="data\\open alex dataset.csv", db="oa")
    
    print("Predictive Analytics")
    print("=" * 60)
    
    # Run analysis
    result = run_predictive_analysis(
        df=ba.df,
        citations_col="Cited by",
        year_col="Year",
        abstract_col="Abstract",
        verbose=True)
    
    # Print summary
    print(f"\nModel trained successfully")
    
    # Visualizations
    print("\nGenerating plots...")
    plot_citation_prediction_performance(result, save_path="results/prediction_performance")
    plot_research_gaps(result, save_path="results/research_gaps")
    
    # Export
    print("\nExporting results...")
    export_predictive_results(result, "results")
    
    print("\nDone!")
