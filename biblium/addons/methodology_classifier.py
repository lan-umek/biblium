# -*- coding: utf-8 -*-
"""
Methodology Classifier Module

This module provides enhanced methodology detection and classification
for bibliometric analysis, identifying research methods, study designs,
and data sources used in scientific papers.

Features:
1. Method Classification - Quantitative/Qualitative/Mixed
2. Study Design Detection - Experimental, observational, case study, etc.
3. Data Type Identification - Primary/Secondary data
4. Sample Size Extraction - Extract sample sizes from text
5. Temporal Design - Cross-sectional vs longitudinal
6. Analysis Method Detection - Statistical tests, qualitative approaches
7. Tool/Software Detection - R, Python, SPSS, NVivo, etc.
8. SDG-Methodology Mapping - Methods used for each SDG
9. ML-Based Classification - Optional machine learning classifier

Created for integration with the Biblium bibliometric toolkit.

@author: Claude (Anthropic) for Lan.Umek
"""

from __future__ import annotations

import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# ML imports (optional)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# =============================================================================
# METHODOLOGY TAXONOMIES
# =============================================================================

PARADIGM_KEYWORDS = {
    "quantitative": [
        "quantitative", "statistical analysis", "regression", "correlation",
        "survey data", "numerical data", "measurement", "hypothesis testing",
        "descriptive statistics", "inferential", "sample size", "n =", "n=",
        "anova", "t-test", "chi-square", "statistical significant"
    ],
    "qualitative": [
        "qualitative", "interview", "focus group", "ethnograph", "phenomenolog",
        "grounded theory", "thematic analysis", "content analysis", "narrative",
        "discourse analysis", "participant observation", "in-depth", "semi-structured",
        "open-ended", "purposive sampling", "saturation", "coding", "themes"
    ],
    "mixed_methods": [
        "mixed method", "mixed-method", "multi-method", "triangulation",
        "sequential explanatory", "sequential exploratory", "concurrent",
        "embedded design", "convergent design"
    ]
}

STUDY_DESIGN_KEYWORDS = {
    "experimental": [
        "experiment", "randomized controlled", "rct", "randomised", "treatment group",
        "control group", "intervention", "placebo", "blind", "double-blind"
    ],
    "quasi_experimental": [
        "quasi-experiment", "natural experiment", "difference-in-difference",
        "propensity score", "regression discontinuity"
    ],
    "observational": [
        "observational", "cohort study", "prospective", "retrospective",
        "longitudinal", "panel data", "time series", "cross-sectional"
    ],
    "survey": [
        "survey", "questionnaire", "likert", "self-report", "online survey"
    ],
    "case_study": [
        "case study", "case-study", "single case", "multiple case", "comparative case"
    ],
    "systematic_review": [
        "systematic review", "meta-analysis", "scoping review", "literature review",
        "prisma", "narrative review"
    ],
    "simulation": [
        "simulation", "monte carlo", "agent-based model", "system dynamics"
    ],
    "action_research": [
        "action research", "participatory action", "community-based", "co-design"
    ],
    "delphi": [
        "delphi", "expert panel", "consensus method"
    ]
}

DATA_SOURCE_KEYWORDS = {
    "primary_survey": ["primary data", "original survey", "administered questionnaire"],
    "primary_interview": ["conducted interview", "interview data", "in-depth interview"],
    "primary_experiment": ["experimental data", "laboratory", "field experiment"],
    "secondary_database": ["secondary data", "existing database", "archival data", "world bank", "eurostat"],
    "secondary_literature": ["literature search", "database search", "scopus", "web of science", "pubmed"]
}

STATISTICAL_METHODS = {
    "descriptive": ["descriptive statistics", "mean", "median", "standard deviation"],
    "regression": ["regression", "ols", "logistic regression", "linear regression", "multilevel"],
    "correlation": ["correlation", "pearson", "spearman"],
    "comparison_tests": ["t-test", "anova", "mann-whitney", "chi-square"],
    "factor_analysis": ["factor analysis", "pca", "principal component", "sem", "structural equation"],
    "cluster_analysis": ["cluster analysis", "k-means", "hierarchical cluster"],
    "time_series": ["time series", "arima", "var", "cointegration"],
    "machine_learning": ["machine learning", "random forest", "neural network", "deep learning"],
    "spatial_analysis": ["spatial analysis", "gis", "geospatial"],
    "network_analysis": ["network analysis", "social network", "centrality"],
    "text_analysis": ["text analysis", "text mining", "nlp", "topic model"]
}

QUALITATIVE_METHODS = {
    "thematic_analysis": ["thematic analysis", "theme", "thematic coding"],
    "content_analysis": ["content analysis", "coding scheme"],
    "grounded_theory": ["grounded theory", "theoretical sampling", "constant comparative"],
    "discourse_analysis": ["discourse analysis", "critical discourse"],
    "narrative_analysis": ["narrative analysis", "story", "biographical"],
    "phenomenology": ["phenomenolog", "lived experience"],
    "ethnography": ["ethnograph", "field work", "participant observation"]
}

SOFTWARE_KEYWORDS = {
    "spss": ["spss"], "stata": ["stata"], "r": ["r software", "rstudio", "cran"],
    "python": ["python", "pandas", "scikit-learn"], "sas": ["sas software"],
    "nvivo": ["nvivo"], "atlas_ti": ["atlas.ti"], "maxqda": ["maxqda"],
    "excel": ["excel"], "arcgis": ["arcgis"], "matlab": ["matlab"]
}

SAMPLE_SIZE_PATTERNS = [
    r'n\s*=\s*(\d+[\d,]*)',
    r'sample\s+(?:size|of)\s+(?:was\s+)?(\d+[\d,]*)',
    r'(\d+[\d,]*)\s+(?:participants|respondents|subjects)',
]

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MethodologyResult:
    """Methodology classification for a single document."""
    doc_id: Any
    paradigm: str
    paradigm_confidence: float
    study_designs: List[str]
    primary_design: str
    data_sources: List[str]
    is_primary_data: bool
    is_secondary_data: bool
    statistical_methods: List[str]
    qualitative_methods: List[str]
    software_tools: List[str]
    sample_sizes: List[int]
    is_longitudinal: bool
    is_cross_sectional: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "Doc Id": self.doc_id,
            "paradigm": self.paradigm,
            "Paradigm Confidence": self.paradigm_confidence,
            "Study Designs": "; ".join(self.study_designs),
            "Primary Design": self.primary_design,
            "Is Primary Data": self.is_primary_data,
            "Is Secondary Data": self.is_secondary_data,
            "Statistical Methods": "; ".join(self.statistical_methods),
            "Qualitative Methods": "; ".join(self.qualitative_methods),
            "software_tools": "; ".join(self.software_tools),
            "sample_sizes": self.sample_sizes,
        }


@dataclass
class MethodologyAnalysis:
    """Complete methodology analysis results."""
    document_results: List[MethodologyResult]
    paradigm_distribution: Dict[str, int]
    design_distribution: Dict[str, int]
    method_distribution: Dict[str, int]
    software_distribution: Dict[str, int]
    sdg_methodology_matrix: pd.DataFrame
    temporal_trends: Dict[str, pd.DataFrame]
    
    def get_results_df(self) -> pd.DataFrame:
        records = [r.to_dict() for r in self.document_results]
        return pd.DataFrame(records)
    
    def get_paradigm_summary(self, include_unknown: bool = False) -> pd.DataFrame:
        filtered = {k: v for k, v in self.paradigm_distribution.items() 
                    if include_unknown or k != 'unknown'}
        total = sum(filtered.values())
        records = [{"Paradigm": p, "Count": c, "Percentage": c/total*100 if total else 0}
                   for p, c in filtered.items()]
        return pd.DataFrame(records).sort_values("Count", ascending=False)
    
    def get_method_summary(self) -> pd.DataFrame:
        total = len(self.document_results)
        records = [{"Method": m, "Count": c, "Percentage": c/total*100 if total else 0}
                   for m, c in self.method_distribution.items()]
        return pd.DataFrame(records).sort_values("Count", ascending=False)


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def classify_paradigm(text: str) -> Tuple[str, float]:
    text_lower = text.lower()
    scores = {}
    for paradigm, keywords in PARADIGM_KEYWORDS.items():
        scores[paradigm] = sum(1 for kw in keywords if kw in text_lower)
    
    total = sum(scores.values())
    if total == 0:
        return "unknown", 0.0
    
    if scores.get("mixed_methods", 0) > 0:
        return "mixed_methods", scores["mixed_methods"] / total
    
    if scores.get("quantitative", 0) > scores.get("qualitative", 0):
        return "quantitative", scores["quantitative"] / total
    elif scores.get("qualitative", 0) > scores.get("quantitative", 0):
        return "qualitative", scores["qualitative"] / total
    return "unknown", 0.0


def detect_study_designs(text: str) -> List[str]:
    text_lower = text.lower()
    return [d for d, kws in STUDY_DESIGN_KEYWORDS.items() if any(kw in text_lower for kw in kws)]


def detect_data_sources(text: str) -> Tuple[List[str], bool, bool]:
    text_lower = text.lower()
    detected = [s for s, kws in DATA_SOURCE_KEYWORDS.items() if any(kw in text_lower for kw in kws)]
    return detected, any(s.startswith("primary") for s in detected), any(s.startswith("secondary") for s in detected)


def detect_statistical_methods(text: str) -> List[str]:
    text_lower = text.lower()
    return [m for m, kws in STATISTICAL_METHODS.items() if any(kw in text_lower for kw in kws)]


def detect_qualitative_methods(text: str) -> List[str]:
    text_lower = text.lower()
    return [m for m, kws in QUALITATIVE_METHODS.items() if any(kw in text_lower for kw in kws)]


def detect_software(text: str) -> List[str]:
    text_lower = text.lower()
    return [s for s, kws in SOFTWARE_KEYWORDS.items() if any(kw in text_lower for kw in kws)]


def extract_sample_sizes(text: str) -> List[int]:
    sizes = []
    for pattern in SAMPLE_SIZE_PATTERNS:
        for match in re.findall(pattern, text, re.IGNORECASE):
            try:
                size = int(match.replace(",", ""))
                if 1 <= size <= 1000000:
                    sizes.append(size)
            except:
                pass
    return sorted(set(sizes))


def classify_methodology(text: str, doc_id: Any = None) -> MethodologyResult:
    if not text or pd.isna(text):
        return MethodologyResult(doc_id, "unknown", 0.0, [], "unknown", [], False, False, [], [], [], [], False, False)
    
    text = str(text)
    paradigm, confidence = classify_paradigm(text)
    designs = detect_study_designs(text)
    sources, is_primary, is_secondary = detect_data_sources(text)
    stat_methods = detect_statistical_methods(text)
    qual_methods = detect_qualitative_methods(text)
    software = detect_software(text)
    samples = extract_sample_sizes(text)
    
    is_long = any(kw in text.lower() for kw in ["longitudinal", "panel", "follow-up", "prospective"])
    is_cross = any(kw in text.lower() for kw in ["cross-sectional", "snapshot"])
    
    return MethodologyResult(
        doc_id=doc_id, paradigm=paradigm, paradigm_confidence=confidence,
        study_designs=designs, primary_design=designs[0] if designs else "unknown",
        data_sources=sources, is_primary_data=is_primary, is_secondary_data=is_secondary,
        statistical_methods=stat_methods, qualitative_methods=qual_methods,
        software_tools=software, sample_sizes=samples,
        is_longitudinal=is_long, is_cross_sectional=is_cross
    )


def analyze_methodology_corpus(
    df: pd.DataFrame,
    text_col: str = "Abstract",
    id_col: str = None,
    sdg_cols: List[str] = None,
    year_col: str = "Year",
    verbose: bool = True
) -> MethodologyAnalysis:
    if verbose:
        print("="*60)
        print("METHODOLOGY CLASSIFICATION")
        print("="*60)
        print(f"Classifying {len(df)} documents...")
    
    results = []
    for idx, row in df.iterrows():
        doc_id = row.get(id_col, idx) if id_col else idx
        text = row.get(text_col, "")
        results.append(classify_methodology(str(text), doc_id))
    
    paradigm_dist = Counter(r.paradigm for r in results)
    design_dist = Counter(d for r in results for d in r.study_designs)
    method_dist = Counter(m for r in results for m in r.statistical_methods + r.qualitative_methods)
    software_dist = Counter(s for r in results for s in r.software_tools)
    
    sdg_matrix = pd.DataFrame()
    if sdg_cols:
        matrix_data = defaultdict(lambda: defaultdict(int))
        for i, (_, row) in enumerate(df.iterrows()):
            r = results[i]
            sdgs = [int(re.search(r'\d+', c).group()) for c in sdg_cols if row.get(c, 0) == 1 and re.search(r'\d+', c)]
            methods = r.statistical_methods + r.qualitative_methods
            for sdg in sdgs:
                for m in methods:
                    matrix_data[sdg][m] += 1
        if matrix_data:
            sdg_matrix = pd.DataFrame(matrix_data).T.fillna(0)
    
    temporal = {}
    if year_col in df.columns:
        df_t = df.copy()
        df_t["_paradigm"] = [r.paradigm for r in results]
        temporal["paradigm"] = df_t.groupby([year_col, "_paradigm"]).size().unstack(fill_value=0)
    
    if verbose:
        print("\nParadigm Distribution:")
        for p, c in paradigm_dist.most_common():
            print(f"  {p}: {c} ({c/len(results)*100:.1f}%)")
    
    return MethodologyAnalysis(results, dict(paradigm_dist), dict(design_dist), 
                               dict(method_dist), dict(software_dist), sdg_matrix, temporal)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_paradigm_distribution(analysis: MethodologyAnalysis, figsize=(10, 6), save_path=None, dpi=300):
    """Plot paradigm distribution as bar chart (excludes unknown)."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    # Filter out 'unknown'
    filtered = {k: v for k, v in analysis.paradigm_distribution.items() if k != 'unknown'}
    
    if not filtered:
        ax.text(0.5, 0.5, "No classified paradigms", ha='center', va='center', fontsize=12)
        return fig
    
    # Clean paradigm names for display (replace underscores with spaces, title case)
    paradigm_labels = [p.replace('_', ' ').title() for p in filtered.keys()]
    counts = list(filtered.values())
    colors = {'quantitative': '#1f77b4', 'qualitative': '#2ca02c', 'mixed_methods': '#ff7f0e'}
    bar_colors = [colors.get(p, '#888888') for p in filtered.keys()]
    
    bars = ax.bar(paradigm_labels, counts, color=bar_colors)
    ax.set_ylabel("Number of Papers")
    ax.set_xlabel("Research Paradigm")
    ax.set_title("Research Paradigm Distribution", fontsize=12, fontweight='bold')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom', fontsize=10)
    
    # Remove spines and grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    
    plt.xticks(rotation=0)  # No rotation needed for clean labels
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig


def plot_method_distribution(analysis: MethodologyAnalysis, top_n=15, figsize=(12, 6), save_path=None, dpi=300):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    methods = dict(Counter(analysis.method_distribution).most_common(top_n))
    ax.barh(list(methods.keys()), list(methods.values()), color='steelblue')
    ax.set_xlabel("Papers")
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig


def plot_sdg_methodology_heatmap(analysis: MethodologyAnalysis, figsize=(14, 10), save_path=None, dpi=300):
    if analysis.sdg_methodology_matrix.empty:
        print("No SDG-methodology data")
        return None
    
    matrix = analysis.sdg_methodology_matrix.div(analysis.sdg_methodology_matrix.sum(axis=1), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    sns.heatmap(matrix, cmap="YlOrRd", annot=True, fmt=".2f", ax=ax)
    ax.set_title("SDG-Methodology Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig


# =============================================================================
# ML-BASED CLASSIFIER
# =============================================================================

class MethodologyMLClassifier:
    """
    Machine Learning-based methodology classifier.
    
    Uses TF-IDF features and supervised learning to classify research
    methodology when labeled training data is available.
    
    Parameters
    ----------
    model_type : str
        Type of ML model: "logistic", "naive_bayes", "random_forest".
    
    Example
    -------
    >>> # Create and train classifier
    >>> clf = MethodologyMLClassifier(model_type="logistic")
    >>> clf.fit(train_texts, train_labels)
    >>> 
    >>> # Predict on new data
    >>> predictions = clf.predict(new_texts)
    >>> 
    >>> # Get prediction with confidence
    >>> pred, conf = clf.predict_with_confidence(new_texts)
    """
    
    def __init__(self, model_type: str = "logistic"):
        if not ML_AVAILABLE:
            raise ImportError("scikit-learn is required for ML classification")
        
        self.model_type = model_type
        self.is_fitted = False
        
        # Create pipeline
        if model_type == "logistic":
            classifier = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == "naive_bayes":
            classifier = MultinomialNB()
        elif model_type == "random_forest":
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                stop_words="english"
            )),
            ("clf", classifier)
        ])
    
    def fit(
        self,
        texts: List[str],
        labels: List[str],
        validate: bool = True,
    ) -> "MethodologyMLClassifier":
        """
        Train the classifier.
        
        Parameters
        ----------
        texts : list
            Training texts (abstracts).
        labels : list
            Training labels (paradigm names).
        validate : bool
            Whether to run cross-validation.
        
        Returns
        -------
        self
        """
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have same length")
        
        # Filter out empty texts
        valid_idx = [i for i, t in enumerate(texts) if t and len(str(t)) > 10]
        texts = [texts[i] for i in valid_idx]
        labels = [labels[i] for i in valid_idx]
        
        if len(texts) < 10:
            raise ValueError("Need at least 10 training samples")
        
        # Cross-validation
        if validate:
            scores = cross_val_score(self.pipeline, texts, labels, cv=min(5, len(texts)//3))
            print(f"Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
        
        # Fit on all data
        self.pipeline.fit(texts, labels)
        self.classes_ = self.pipeline.classes_
        self.is_fitted = True
        
        return self
    
    def predict(self, texts: List[str]) -> List[str]:
        """Predict labels for texts."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        return self.pipeline.predict(texts)
    
    def predict_with_confidence(
        self,
        texts: List[str],
    ) -> Tuple[List[str], List[float]]:
        """
        Predict labels with confidence scores.
        
        Returns
        -------
        tuple
            (predictions, confidences)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        predictions = self.pipeline.predict(texts)
        probas = self.pipeline.predict_proba(texts)
        confidences = probas.max(axis=1)
        
        return predictions.tolist(), confidences.tolist()
    
    def classify_with_fallback(
        self,
        texts: List[str],
        confidence_threshold: float = 0.5,
    ) -> List[Tuple[str, float, str]]:
        """
        Classify with fallback to keyword-based when confidence is low.
        
        Returns
        -------
        list
            List of (prediction, confidence, method) tuples.
        """
        results = []
        
        ml_preds, ml_confs = self.predict_with_confidence(texts)
        
        for text, pred, conf in zip(texts, ml_preds, ml_confs):
            if conf >= confidence_threshold:
                results.append((pred, conf, "ml"))
            else:
                # Fallback to keyword-based
                kw_pred, kw_conf = classify_paradigm(str(text))
                results.append((kw_pred, kw_conf, "keyword"))
        
        return results
    
    @staticmethod
    def create_training_data(
        df: pd.DataFrame,
        text_col: str = "Abstract",
        paradigm_col: str = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Create training data from DataFrame.
        
        If paradigm_col is None, uses keyword-based classification
        to generate pseudo-labels.
        
        Returns
        -------
        tuple
            (texts, labels)
        """
        texts = []
        labels = []
        
        for _, row in df.iterrows():
            text = str(row.get(text_col, ""))
            if len(text) < 50:
                continue
            
            if paradigm_col and paradigm_col in row:
                label = str(row[paradigm_col])
            else:
                # Use keyword-based classification for pseudo-labels
                label, conf = classify_paradigm(text)
                if conf < 0.3:  # Skip low-confidence examples
                    continue
            
            if label and label != "unknown":
                texts.append(text)
                labels.append(label)
        
        return texts, labels


def train_methodology_classifier(
    ba,
    text_col: str = "Abstract",
    paradigm_col: str = None,
    model_type: str = "logistic",
    min_confidence: float = 0.3,
) -> MethodologyMLClassifier:
    """
    Train an ML classifier using a BiblioAnalysis dataset.
    
    Parameters
    ----------
    ba : BiblioStats
        BiblioAnalysis object with data.
    text_col : str
        Column containing text (abstracts).
    paradigm_col : str
        Column with paradigm labels (if available).
        If None, uses keyword-based pseudo-labels.
    model_type : str
        ML model type.
    min_confidence : float
        Minimum confidence for pseudo-labels.
    
    Returns
    -------
    MethodologyMLClassifier
        Trained classifier.
    
    Example
    -------
    >>> clf = train_methodology_classifier(ba, model_type="logistic")
    >>> predictions = clf.predict(ba.df["Abstract"].tolist())
    """
    if not ML_AVAILABLE:
        raise ImportError("scikit-learn required for ML classification")
    
    # Find text column
    if text_col not in ba.df.columns:
        for c in ["Abstract", "Processed Abstract", "Title"]:
            if c in ba.df.columns:
                text_col = c
                break
        else:
            raise ValueError("No text column found")
    
    # Create training data
    texts, labels = MethodologyMLClassifier.create_training_data(
        ba.df, text_col=text_col, paradigm_col=paradigm_col
    )
    
    print(f"Training on {len(texts)} documents")
    print(f"Label distribution: {Counter(labels)}")
    
    # Train classifier
    clf = MethodologyMLClassifier(model_type=model_type)
    clf.fit(texts, labels)
    
    return clf


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_methodology_analysis(ba, text_col=None, verbose=True, use_ml=False) -> MethodologyAnalysis:
    """
    Run methodology analysis on a BiblioAnalysis object.
    
    Parameters
    ----------
    ba : BiblioStats
        BiblioAnalysis object with data.
    text_col : str
        Column containing text (abstracts).
    verbose : bool
        Print progress.
    use_ml : bool
        Whether to use ML-based classification (requires training data).
    
    Returns
    -------
    MethodologyAnalysis
        Analysis results.
    
    Usage:
        from biblium.addons.methodology_classifier import run_methodology_analysis
        methods = run_methodology_analysis(ba)
        print(methods.get_paradigm_summary())
    """
    if text_col is None:
        for c in ["Abstract", "Processed Abstract", "Title"]:
            if c in ba.df.columns:
                text_col = c
                break
    
    sdg_cols = [c for c in ba.df.columns if c.startswith("SDG") and any(d.isdigit() for d in c)]
    return analyze_methodology_corpus(ba.df, text_col=text_col, sdg_cols=sdg_cols or None, verbose=verbose)
