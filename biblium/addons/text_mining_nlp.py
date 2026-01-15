# -*- coding: utf-8 -*-
"""
Text Mining & NLP Advanced Module for Bibliometric Analysis

This module provides advanced natural language processing capabilities
for extracting insights from scientific text in bibliometric datasets.

Features implemented:
1. Named Entity Recognition (NER) - Methods, datasets, software, chemicals
2. Claim Extraction - Identify key claims and findings from abstracts
3. Research Question Identification - Extract research questions/objectives
4. Methodology Classification - Classify research methods used
5. Sentiment Analysis - Analyze tone and certainty in scientific writing
6. Novelty Detection - Detect novel contributions via text similarity
7. Concept Extraction - Extract key concepts and terminology
8. Readability Analysis - Assess text complexity and accessibility

Created for integration with the Biblium bibliometric toolkit.

@author: Claude (Anthropic) for Lan.Umek
"""

from __future__ import annotations

import os
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from itertools import combinations
import json
import string

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import percentileofscore

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import seaborn as sns

# Import core biblium bridge
try:
    from biblium.addons.core_utils import get_core_function, CORE_AVAILABLE, use_core_or_fallback
except ImportError:
    CORE_AVAILABLE = False
    def get_core_function(name): return None
    def use_core_or_fallback(name, fallback, *args, **kwargs): return fallback(*args, **kwargs)

# Optional imports with fallbacks
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# =============================================================================
# DEFAULT COLORMAPS (user-configurable)
# =============================================================================

CMAP_CONTINUOUS = "viridis"  # For continuous/sequential data
CMAP_DISCRETE = "tab10"      # For categorical/discrete data

def set_default_cmaps(continuous: str = None, discrete: str = None):
    """Set default colormaps for all plots in this module."""
    global CMAP_CONTINUOUS, CMAP_DISCRETE
    if continuous:
        CMAP_CONTINUOUS = continuous
    if discrete:
        CMAP_DISCRETE = discrete

# =============================================================================
# CONSTANTS AND PATTERNS
# =============================================================================

# Method/technique patterns
METHOD_PATTERNS = {
    "machine_learning": [
        r"machine\s+learning", r"deep\s+learning", r"neural\s+network",
        r"random\s+forest", r"support\s+vector", r"SVM", r"CNN", r"RNN",
        r"LSTM", r"transformer", r"BERT", r"GPT", r"gradient\s+boosting",
        r"XGBoost", r"LightGBM", r"ensemble", r"classification",
        r"regression\s+model", r"clustering", r"k-means", r"PCA",
        r"dimensionality\s+reduction", r"feature\s+selection",
    ],
    "statistical": [
        r"regression\s+analysis", r"ANOVA", r"t-test", r"chi-square",
        r"correlation", r"meta-analysis", r"systematic\s+review",
        r"statistical\s+analysis", r"hypothesis\s+test", r"p-value",
        r"confidence\s+interval", r"Bayesian", r"Monte\s+Carlo",
        r"bootstrap", r"cross-validation", r"significance",
    ],
    "experimental": [
        r"experiment", r"randomized\s+control", r"RCT", r"clinical\s+trial",
        r"blind\s+study", r"placebo", r"control\s+group", r"treatment\s+group",
        r"in\s+vitro", r"in\s+vivo", r"laboratory", r"field\s+study",
    ],
    "qualitative": [
        r"interview", r"focus\s+group", r"ethnograph", r"case\s+study",
        r"grounded\s+theory", r"thematic\s+analysis", r"content\s+analysis",
        r"discourse\s+analysis", r"phenomenolog", r"narrative\s+analysis",
    ],
    "survey": [
        r"survey", r"questionnaire", r"likert", r"sampling",
        r"respondent", r"participant", r"cross-sectional",
    ],
    "simulation": [
        r"simulation", r"agent-based", r"finite\s+element", r"CFD",
        r"molecular\s+dynamics", r"discrete\s+event", r"system\s+dynamics",
    ],
    "nlp": [
        r"natural\s+language", r"NLP", r"text\s+mining", r"sentiment",
        r"named\s+entity", r"NER", r"part-of-speech", r"POS\s+tag",
        r"word\s+embedding", r"word2vec", r"GloVe", r"topic\s+model",
    ],
    "image_analysis": [
        r"image\s+processing", r"computer\s+vision", r"object\s+detection",
        r"segmentation", r"image\s+classification", r"convolutional",
    ],
}

# Software/tool patterns
SOFTWARE_PATTERNS = {
    "programming_languages": [
        r"\bPython\b", r"\bR\b(?!\s+&)", r"\bJava\b", r"\bC\+\+\b",
        r"\bMATLAB\b", r"\bJulia\b", r"\bScala\b", r"\bPerl\b",
    ],
    "statistical_software": [
        r"\bSPSS\b", r"\bStata\b", r"\bSAS\b", r"\bMinitab\b",
        r"\bJMP\b", r"\bEViews\b", r"\bGraphPad\b",
    ],
    "ml_frameworks": [
        r"\bTensorFlow\b", r"\bPyTorch\b", r"\bKeras\b", r"\bscikit-learn\b",
        r"\bCaffe\b", r"\bMXNet\b", r"\bXGBoost\b", r"\bLightGBM\b",
    ],
    "visualization": [
        r"\bTableau\b", r"\bPower\s*BI\b", r"\bggplot\b", r"\bmatplotlib\b",
        r"\bseaborn\b", r"\bD3\.js\b", r"\bPlotly\b",
    ],
    "databases": [
        r"\bMySQL\b", r"\bPostgreSQL\b", r"\bMongoDB\b", r"\bOracle\b",
        r"\bSQL\s+Server\b", r"\bSQLite\b", r"\bRedis\b",
    ],
    "domain_specific": [
        r"\bImageJ\b", r"\bFIJI\b", r"\bCellProfiler\b", r"\bBLAST\b",
        r"\bGaussian\b", r"\bANSYS\b", r"\bAbaqus\b", r"\bCOMSOL\b",
        r"\bAutoCAD\b", r"\bRevit\b", r"\bArcGIS\b", r"\bQGIS\b",
    ],
}

# Dataset patterns
DATASET_PATTERNS = [
    r"dataset[s]?\s+(?:from|of|called|named)?\s*[\"']?([A-Z][A-Za-z0-9\-_]+)[\"']?",
    r"(?:used|using|from)\s+(?:the\s+)?([A-Z][A-Za-z0-9\-_]+)\s+dataset",
    r"([A-Z][A-Za-z0-9\-_]+)\s+(?:dataset|database|corpus|benchmark)",
    r"(?:ImageNet|MNIST|CIFAR|COCO|VOC|UCI|Kaggle|OpenML)",
    r"(?:PubMed|Scopus|Web\s+of\s+Science|Google\s+Scholar)",
    r"(?:GenBank|UniProt|PDB|KEGG|GO|Ensembl)",
    r"(?:World\s+Bank|IMF|OECD|Eurostat|Census)",
]

# Claim/finding patterns
CLAIM_PATTERNS = [
    r"(?:we|this\s+study|our\s+(?:results|findings|analysis))\s+(?:show|demonstrate|reveal|indicate|suggest|find|found|confirm|establish)",
    r"(?:results|findings|data|evidence)\s+(?:show|indicate|suggest|demonstrate|reveal|support)",
    r"(?:significantly|substantially|markedly)\s+(?:increase|decrease|improve|reduce|enhance)",
    r"(?:there\s+(?:is|was|were)\s+(?:a\s+)?(?:significant|strong|positive|negative|clear))",
    r"(?:compared\s+to|relative\s+to|in\s+contrast\s+to)",
    r"(?:outperform|exceed|surpass|better\s+than)",
]

# Research question patterns
RESEARCH_QUESTION_PATTERNS = [
    r"(?:this\s+study|we|this\s+paper|this\s+research)\s+(?:aim|seek|attempt|investigate|examine|explore|analyze|assess|evaluate)[s]?\s+to",
    r"(?:the\s+)?(?:aim|goal|objective|purpose)\s+(?:of\s+this\s+(?:study|paper|research))?\s+(?:is|was)\s+to",
    r"(?:research|study)\s+question[s]?",
    r"(?:we\s+)?(?:hypothesize|propose|argue|suggest)\s+that",
    r"(?:how|what|why|whether|to\s+what\s+extent)",
    r"(?:this\s+paper\s+)?address(?:es)?\s+(?:the\s+)?(?:question|issue|problem)",
]

# Novelty/contribution patterns
NOVELTY_PATTERNS = [
    r"(?:novel|new|first|innovative|original|unique|pioneering)",
    r"(?:contribution|advance|breakthrough|improvement)",
    r"(?:for\s+the\s+first\s+time|never\s+before|unprecedented)",
    r"(?:we\s+introduce|we\s+propose|we\s+present|we\s+develop)",
    r"(?:unlike\s+previous|in\s+contrast\s+to\s+existing|different\s+from)",
    r"(?:state-of-the-art|cutting-edge|leading-edge)",
    r"(?:extend|improve\s+upon|build\s+on|go\s+beyond)",
]

# Hedging/certainty patterns
CERTAINTY_MARKERS = {
    "high_certainty": [
        r"\bdemonstrate[sd]?\b", r"\bprove[sd]?\b", r"\bconfirm[sed]?\b",
        r"\bestablish(?:es|ed)?\b", r"\bclearly\b", r"\bdefinitely\b",
        r"\bundoubtedly\b", r"\bcertainly\b", r"\bevident\b",
    ],
    "moderate_certainty": [
        r"\bindicate[sd]?\b", r"\bshow[sed]?\b", r"\bfind[s]?\b", r"\bfound\b",
        r"\bsuggest[sed]?\b", r"\bimply\b", r"\bimplies\b",
    ],
    "low_certainty": [
        r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bpossibly\b",
        r"\bperhaps\b", r"\bpotentially\b", r"\bappear[s]?\b",
        r"\bseem[s]?\b", r"\btend[s]?\b", r"\blikely\b",
    ],
    "hedging": [
        r"\bsomewhat\b", r"\brelatively\b", r"\bpartially\b",
        r"\bto\s+some\s+extent\b", r"\bin\s+part\b", r"\bgenerally\b",
        r"\btypically\b", r"\busually\b", r"\boften\b",
    ],
}

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class NERResult:
    """Named Entity Recognition results for a document."""
    doc_id: Any
    methods: List[Tuple[str, str]]  # (entity, category)
    software: List[Tuple[str, str]]
    datasets: List[str]
    chemicals: List[str]
    organisms: List[str]
    metrics: List[str]
    all_entities: Dict[str, List[str]]

@dataclass
class ClaimResult:
    """Extracted claims from a document."""
    doc_id: Any
    claims: List[str]
    claim_types: List[str]  # "finding", "comparison", "improvement"
    claim_strength: List[str]  # "strong", "moderate", "weak"
    n_claims: int

@dataclass
class ResearchQuestionResult:
    """Extracted research questions/objectives."""
    doc_id: Any
    questions: List[str]
    objectives: List[str]
    hypotheses: List[str]
    question_type: str  # "descriptive", "comparative", "causal", "exploratory"

@dataclass
class MethodologyResult:
    """Methodology classification results."""
    doc_id: Any
    primary_method: str
    all_methods: List[str]
    method_categories: Dict[str, float]  # category -> confidence
    is_empirical: bool
    is_quantitative: bool
    is_qualitative: bool
    study_design: str

@dataclass
class SentimentResult:
    """Sentiment and certainty analysis results."""
    doc_id: Any
    polarity: float  # -1 to 1
    subjectivity: float  # 0 to 1
    certainty_score: float  # 0 to 1
    hedging_score: float  # 0 to 1
    tone: str  # "positive", "negative", "neutral"
    certainty_markers: Dict[str, int]

@dataclass
class NoveltyResult:
    """Novelty detection results."""
    doc_id: Any
    novelty_score: float  # 0 to 1
    novelty_claims: List[str]
    similarity_to_corpus: float
    unique_terms: List[str]
    contribution_type: str  # "methodological", "empirical", "theoretical"

@dataclass
class TextAnalysisResult:
    """Complete text analysis results for a document."""
    doc_id: Any
    title: str
    ner: NERResult
    claims: ClaimResult
    research_questions: ResearchQuestionResult
    methodology: MethodologyResult
    sentiment: SentimentResult
    novelty: NoveltyResult
    readability: Dict[str, float]
    word_count: int
    sentence_count: int

@dataclass
class CorpusAnalysisResult:
    """Aggregated analysis results for entire corpus."""
    document_results: List[TextAnalysisResult]
    method_distribution: Dict[str, int]
    software_usage: Dict[str, int]
    dataset_usage: Dict[str, int]
    avg_sentiment: Dict[str, float]
    avg_certainty: float
    avg_novelty: float
    methodology_breakdown: pd.DataFrame
    temporal_trends: pd.DataFrame
    summary_statistics: Dict[str, Any]

# =============================================================================
# NAMED ENTITY RECOGNITION
# =============================================================================

def extract_named_entities(
    text: str,
    doc_id: Any = None,
    use_spacy: bool = True,
    use_patterns: bool = True,
) -> NERResult:
    """
    Extract named entities from scientific text.
    
    Parameters
    ----------
    text : str
        Text to analyze.
    doc_id : Any
        Document identifier.
    use_spacy : bool
        Use spaCy NER if available.
    use_patterns : bool
        Use regex patterns for domain-specific entities.
    
    Returns
    -------
    NERResult
    """
    if not text or pd.isna(text):
        return NERResult(
            doc_id=doc_id, methods=[], software=[], datasets=[],
            chemicals=[], organisms=[], metrics=[], all_entities={}
        )
    
    text = str(text)
    methods = []
    software = []
    datasets = []
    chemicals = []
    organisms = []
    metrics = []
    
    # Pattern-based extraction for methods
    if use_patterns:
        for category, patterns in METHOD_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, str) and len(match) > 2:
                        methods.append((match.lower(), category))
                    elif match:
                        methods.append((pattern.replace(r"\s+", " ").replace("\\b", ""), category))
        
        # Software extraction
        for category, patterns in SOFTWARE_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if match:
                        software.append((match, category))
        
        # Dataset extraction
        for pattern in DATASET_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            datasets.extend([m for m in matches if isinstance(m, str) and len(m) > 2])
    
    # SpaCy NER for general entities
    if use_spacy and SPACY_AVAILABLE:
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text[:100000])  # Limit text length
            
            for ent in doc.ents:
                if ent.label_ == "ORG" and any(kw in ent.text.lower() for kw in ["university", "institute", "lab"]):
                    pass  # Skip organizations
                elif ent.label_ == "PRODUCT":
                    software.append((ent.text, "product"))
                elif ent.label_ in ["GPE", "LOC"]:
                    pass  # Skip locations
        except:
            pass
    
    # Extract metrics/measures
    metric_patterns = [
        r"(?:accuracy|precision|recall|F1|AUC|ROC|RMSE|MAE|R-squared|R²)",
        r"(?:p\s*[<>=]\s*0\.\d+)",
        r"(?:\d+\.?\d*\s*%)",
        r"(?:CI\s*[:=]?\s*\[?\d+\.?\d*\s*[-–]\s*\d+\.?\d*\]?)",
    ]
    for pattern in metric_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        metrics.extend(matches)
    
    # Chemical patterns
    chemical_patterns = [
        r"\b[A-Z][a-z]?(?:\d+[A-Z][a-z]?\d*)+\b",  # Chemical formulas
        r"\b(?:sodium|potassium|calcium|magnesium|chloride|sulfate|nitrate)\b",
    ]
    for pattern in chemical_patterns:
        matches = re.findall(pattern, text)
        chemicals.extend(matches[:10])  # Limit
    
    # Deduplicate
    methods = list(set(methods))
    software = list(set(software))
    datasets = list(set(datasets))
    chemicals = list(set(chemicals))
    metrics = list(set(metrics))
    
    all_entities = {
        "methods": [m[0] for m in methods],
        "software": [s[0] for s in software],
        "datasets": datasets,
        "chemicals": chemicals,
        "metrics": metrics,
    }
    
    return NERResult(
        doc_id=doc_id,
        methods=methods,
        software=software,
        datasets=datasets,
        chemicals=chemicals,
        organisms=organisms,
        metrics=metrics,
        all_entities=all_entities,
    )

# =============================================================================
# CLAIM EXTRACTION
# =============================================================================

def extract_claims(
    text: str,
    doc_id: Any = None,
) -> ClaimResult:
    """
    Extract key claims and findings from text.
    
    Parameters
    ----------
    text : str
        Text to analyze.
    doc_id : Any
        Document identifier.
    
    Returns
    -------
    ClaimResult
    """
    if not text or pd.isna(text):
        return ClaimResult(
            doc_id=doc_id, claims=[], claim_types=[], claim_strength=[], n_claims=0
        )
    
    text = str(text)
    claims = []
    claim_types = []
    claim_strength = []
    
    # Split into sentences
    if NLTK_AVAILABLE:
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = re.split(r"[.!?]+", text)
    else:
        sentences = re.split(r"[.!?]+", text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:
            continue
        
        # Check for claim patterns
        is_claim = False
        c_type = "general"
        strength = "moderate"
        
        # Finding claims
        finding_patterns = [
            r"(?:we|this\s+study|our\s+(?:results|findings))\s+(?:show|demonstrate|reveal|find|found)",
            r"(?:results|findings|data)\s+(?:show|indicate|suggest|demonstrate)",
        ]
        for pattern in finding_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                is_claim = True
                c_type = "finding"
                break
        
        # Comparison claims
        comparison_patterns = [
            r"(?:compared\s+to|relative\s+to|versus|vs\.?)",
            r"(?:outperform|better\s+than|worse\s+than|superior|inferior)",
            r"(?:significant(?:ly)?\s+(?:higher|lower|more|less|greater|fewer))",
        ]
        for pattern in comparison_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                is_claim = True
                c_type = "comparison"
                break
        
        # Improvement claims
        improvement_patterns = [
            r"(?:improve|increase|enhance|boost|accelerate|optimize)",
            r"(?:reduce|decrease|minimize|eliminate|mitigate)",
            r"(?:\d+\.?\d*\s*%\s+(?:improvement|increase|decrease|reduction))",
        ]
        for pattern in improvement_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                is_claim = True
                c_type = "improvement"
                break
        
        # Determine strength
        if is_claim:
            strong_markers = [r"\bsignificant", r"\bdramatic", r"\bsubstantial", r"\bclear", r"\bstrong"]
            weak_markers = [r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bpossibly", r"\bsuggest"]
            
            if any(re.search(p, sentence, re.IGNORECASE) for p in strong_markers):
                strength = "strong"
            elif any(re.search(p, sentence, re.IGNORECASE) for p in weak_markers):
                strength = "weak"
            
            claims.append(sentence)
            claim_types.append(c_type)
            claim_strength.append(strength)
    
    return ClaimResult(
        doc_id=doc_id,
        claims=claims[:20],  # Limit to 20 claims
        claim_types=claim_types[:20],
        claim_strength=claim_strength[:20],
        n_claims=len(claims),
    )

# =============================================================================
# RESEARCH QUESTION IDENTIFICATION
# =============================================================================

def extract_research_questions(
    text: str,
    doc_id: Any = None,
) -> ResearchQuestionResult:
    """
    Extract research questions, objectives, and hypotheses.
    
    Parameters
    ----------
    text : str
        Text to analyze.
    doc_id : Any
        Document identifier.
    
    Returns
    -------
    ResearchQuestionResult
    """
    if not text or pd.isna(text):
        return ResearchQuestionResult(
            doc_id=doc_id, questions=[], objectives=[], hypotheses=[],
            question_type="unknown"
        )
    
    text = str(text)
    questions = []
    objectives = []
    hypotheses = []
    
    # Split into sentences
    if NLTK_AVAILABLE:
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = re.split(r"[.!?]+", text)
    else:
        sentences = re.split(r"[.!?]+", text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:
            continue
        
        sentence_lower = sentence.lower()
        
        # Research questions (explicit)
        if "?" in sentence or re.search(r"\b(?:what|how|why|whether|which|who|when)\b", sentence_lower):
            if re.search(r"(?:research|study|paper|investigate|examine|explore)", sentence_lower):
                questions.append(sentence)
        
        # Objectives/aims
        objective_patterns = [
            r"(?:aim|goal|objective|purpose)\s+(?:of\s+(?:this|the)\s+(?:study|paper|research))?\s+(?:is|was|are|were)\s+to",
            r"(?:this\s+(?:study|paper|research))\s+(?:aims?|seeks?|attempts?)\s+to",
            r"(?:we|this\s+study)\s+(?:aim|seek|attempt|investigate|examine|explore|analyze)[s]?\s+to",
        ]
        for pattern in objective_patterns:
            if re.search(pattern, sentence_lower):
                objectives.append(sentence)
                break
        
        # Hypotheses
        hypothesis_patterns = [
            r"(?:we\s+)?hypothesiz",
            r"(?:the\s+)?hypothesis",
            r"(?:we\s+)?(?:propose|predict|expect)\s+that",
            r"it\s+is\s+(?:hypothesized|predicted|expected)\s+that",
        ]
        for pattern in hypothesis_patterns:
            if re.search(pattern, sentence_lower):
                hypotheses.append(sentence)
                break
    
    # Determine question type
    question_type = "exploratory"
    all_text = " ".join(questions + objectives + hypotheses).lower()
    
    if re.search(r"(?:cause|effect|impact|influence|affect)", all_text):
        question_type = "causal"
    elif re.search(r"(?:compare|comparison|difference|versus|vs)", all_text):
        question_type = "comparative"
    elif re.search(r"(?:describe|identify|characterize|explore|understand)", all_text):
        question_type = "descriptive"
    elif re.search(r"(?:relationship|correlation|association|link)", all_text):
        question_type = "relational"
    
    return ResearchQuestionResult(
        doc_id=doc_id,
        questions=questions[:10],
        objectives=objectives[:10],
        hypotheses=hypotheses[:10],
        question_type=question_type,
    )

# =============================================================================
# METHODOLOGY CLASSIFICATION
# =============================================================================

def classify_methodology(
    text: str,
    doc_id: Any = None,
) -> MethodologyResult:
    """
    Classify the methodology used in research.
    
    Parameters
    ----------
    text : str
        Text to analyze.
    doc_id : Any
        Document identifier.
    
    Returns
    -------
    MethodologyResult
    """
    if not text or pd.isna(text):
        return MethodologyResult(
            doc_id=doc_id, primary_method="unknown", all_methods=[],
            method_categories={}, is_empirical=False, is_quantitative=False,
            is_qualitative=False, study_design="unknown"
        )
    
    text = str(text).lower()
    
    # Count method category matches
    method_scores = {}
    all_methods = []
    
    for category, patterns in METHOD_PATTERNS.items():
        score = 0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            score += len(matches)
            all_methods.extend(matches)
        method_scores[category] = score
    
    # Normalize scores
    total_score = sum(method_scores.values())
    if total_score > 0:
        method_categories = {k: v / total_score for k, v in method_scores.items()}
    else:
        method_categories = {k: 0 for k in method_scores.keys()}
    
    # Determine primary method
    if total_score > 0:
        primary_method = max(method_scores, key=method_scores.get)
    else:
        primary_method = "unknown"
    
    # Determine if empirical
    empirical_patterns = [
        r"(?:data|dataset|sample|participant|respondent|subject)",
        r"(?:collect|gather|measure|observe|record)",
        r"(?:experiment|survey|interview|observation)",
    ]
    is_empirical = any(re.search(p, text) for p in empirical_patterns)
    
    # Determine quantitative vs qualitative
    quant_patterns = [
        r"(?:statistical|regression|correlation|significance|p-value)",
        r"(?:sample\s+size|n\s*=\s*\d+|\d+\s*participants)",
        r"(?:quantitative|numerical|measurement|scale)",
    ]
    qual_patterns = [
        r"(?:interview|focus\s+group|ethnograph|narrative)",
        r"(?:thematic|discourse|content\s+analysis)",
        r"(?:qualitative|interpretive|phenomenolog)",
    ]
    
    is_quantitative = any(re.search(p, text) for p in quant_patterns)
    is_qualitative = any(re.search(p, text) for p in qual_patterns)
    
    # Determine study design
    study_design = "unknown"
    design_patterns = {
        "experimental": [r"experiment", r"randomized", r"RCT", r"control\s+group"],
        "quasi-experimental": [r"quasi-experiment", r"non-random"],
        "observational": [r"observational", r"cohort", r"case-control", r"cross-sectional"],
        "survey": [r"survey", r"questionnaire", r"cross-sectional\s+survey"],
        "case_study": [r"case\s+study", r"single\s+case"],
        "review": [r"systematic\s+review", r"meta-analysis", r"literature\s+review"],
        "simulation": [r"simulation", r"computational\s+model", r"agent-based"],
        "mixed_methods": [r"mixed\s+method", r"multi-method"],
    }
    
    for design, patterns in design_patterns.items():
        if any(re.search(p, text) for p in patterns):
            study_design = design
            break
    
    return MethodologyResult(
        doc_id=doc_id,
        primary_method=primary_method,
        all_methods=list(set(all_methods))[:20],
        method_categories=method_categories,
        is_empirical=is_empirical,
        is_quantitative=is_quantitative,
        is_qualitative=is_qualitative,
        study_design=study_design,
    )

# =============================================================================
# SENTIMENT AND CERTAINTY ANALYSIS
# =============================================================================

def analyze_sentiment(
    text: str,
    doc_id: Any = None,
) -> SentimentResult:
    """
    Analyze sentiment and certainty in scientific text.
    
    Parameters
    ----------
    text : str
        Text to analyze.
    doc_id : Any
        Document identifier.
    
    Returns
    -------
    SentimentResult
    """
    if not text or pd.isna(text):
        return SentimentResult(
            doc_id=doc_id, polarity=0, subjectivity=0, certainty_score=0.5,
            hedging_score=0, tone="neutral", certainty_markers={}
        )
    
    text = str(text)
    
    # Use TextBlob if available
    polarity = 0
    subjectivity = 0
    
    if TEXTBLOB_AVAILABLE:
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
        except:
            pass
    else:
        # Simple polarity estimation
        positive_words = ["good", "excellent", "significant", "important", "novel", "effective", "successful", "improved", "better", "best"]
        negative_words = ["poor", "bad", "failed", "limited", "weak", "worse", "worst", "problem", "issue", "challenge"]
        
        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        
        if pos_count + neg_count > 0:
            polarity = (pos_count - neg_count) / (pos_count + neg_count)
    
    # Count certainty markers
    certainty_markers = {}
    certainty_counts = {"high": 0, "moderate": 0, "low": 0, "hedging": 0}
    
    for level, patterns in CERTAINTY_MARKERS.items():
        count = 0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            count += len(matches)
        certainty_markers[level] = count
        
        if level == "high_certainty":
            certainty_counts["high"] = count
        elif level == "moderate_certainty":
            certainty_counts["moderate"] = count
        elif level == "low_certainty":
            certainty_counts["low"] = count
        elif level == "hedging":
            certainty_counts["hedging"] = count
    
    # Calculate certainty score (0-1)
    total_markers = sum(certainty_counts.values())
    if total_markers > 0:
        certainty_score = (
            certainty_counts["high"] * 1.0 +
            certainty_counts["moderate"] * 0.7 +
            certainty_counts["low"] * 0.3 +
            certainty_counts["hedging"] * 0.2
        ) / total_markers
    else:
        certainty_score = 0.5
    
    # Hedging score
    hedging_score = certainty_counts["hedging"] / max(1, total_markers)
    
    # Determine tone
    if polarity > 0.1:
        tone = "positive"
    elif polarity < -0.1:
        tone = "negative"
    else:
        tone = "neutral"
    
    return SentimentResult(
        doc_id=doc_id,
        polarity=round(polarity, 3),
        subjectivity=round(subjectivity, 3),
        certainty_score=round(certainty_score, 3),
        hedging_score=round(hedging_score, 3),
        tone=tone,
        certainty_markers=certainty_markers,
    )

# =============================================================================
# NOVELTY DETECTION
# =============================================================================

def detect_novelty(
    text: str,
    corpus_texts: List[str] = None,
    doc_id: Any = None,
    tfidf_vectorizer: Any = None,
) -> NoveltyResult:
    """
    Detect novelty claims and measure uniqueness.
    
    Parameters
    ----------
    text : str
        Text to analyze.
    corpus_texts : List[str], optional
        Corpus for comparison.
    doc_id : Any
        Document identifier.
    tfidf_vectorizer : Any, optional
        Pre-fitted TF-IDF vectorizer.
    
    Returns
    -------
    NoveltyResult
    """
    if not text or pd.isna(text):
        return NoveltyResult(
            doc_id=doc_id, novelty_score=0, novelty_claims=[],
            similarity_to_corpus=0, unique_terms=[], contribution_type="unknown"
        )
    
    text = str(text)
    novelty_claims = []
    
    # Extract novelty claims
    if NLTK_AVAILABLE:
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = re.split(r"[.!?]+", text)
    else:
        sentences = re.split(r"[.!?]+", text)
    
    for sentence in sentences:
        for pattern in NOVELTY_PATTERNS:
            if re.search(pattern, sentence, re.IGNORECASE):
                novelty_claims.append(sentence.strip())
                break
    
    # Calculate novelty score based on claims
    novelty_score = min(1.0, len(novelty_claims) * 0.15)
    
    # Contribution type
    contribution_type = "unknown"
    text_lower = text.lower()
    
    if re.search(r"(?:new\s+method|novel\s+approach|propose\s+(?:a\s+)?method|algorithm)", text_lower):
        contribution_type = "methodological"
    elif re.search(r"(?:empirical|data|experiment|survey|study\s+(?:of|on))", text_lower):
        contribution_type = "empirical"
    elif re.search(r"(?:theory|theoretical|framework|model|concept)", text_lower):
        contribution_type = "theoretical"
    elif re.search(r"(?:review|meta-analysis|synthesis|overview)", text_lower):
        contribution_type = "review"
    
    # Compare to corpus if provided
    similarity_to_corpus = 0
    unique_terms = []
    
    if corpus_texts and SKLEARN_AVAILABLE and len(corpus_texts) > 1:
        try:
            if tfidf_vectorizer is None:
                tfidf_vectorizer = TfidfVectorizer(
                    max_features=5000,
                    stop_words="english",
                    ngram_range=(1, 2),
                )
                corpus_matrix = tfidf_vectorizer.fit_transform(corpus_texts)
            else:
                corpus_matrix = tfidf_vectorizer.transform(corpus_texts)
            
            text_vector = tfidf_vectorizer.transform([text])
            
            # Calculate similarity to corpus
            similarities = cosine_similarity(text_vector, corpus_matrix)[0]
            similarity_to_corpus = np.mean(similarities)
            
            # Novelty is inverse of similarity
            novelty_score = max(novelty_score, 1 - similarity_to_corpus)
            
            # Find unique terms (high TF-IDF in this doc, low in corpus)
            feature_names = tfidf_vectorizer.get_feature_names_out()
            text_scores = text_vector.toarray()[0]
            corpus_avg = corpus_matrix.mean(axis=0).A1
            
            uniqueness = text_scores - corpus_avg
            top_indices = np.argsort(uniqueness)[-10:][::-1]
            unique_terms = [feature_names[i] for i in top_indices if uniqueness[i] > 0]
            
        except Exception as e:
            pass
    
    return NoveltyResult(
        doc_id=doc_id,
        novelty_score=round(novelty_score, 3),
        novelty_claims=novelty_claims[:10],
        similarity_to_corpus=round(similarity_to_corpus, 3),
        unique_terms=unique_terms[:10],
        contribution_type=contribution_type,
    )

# =============================================================================
# READABILITY ANALYSIS
# =============================================================================

def analyze_readability(text: str) -> Dict[str, float]:
    """
    Calculate readability metrics.
    
    Parameters
    ----------
    text : str
        Text to analyze.
    
    Returns
    -------
    Dict of readability metrics.
    """
    if not text or pd.isna(text):
        return {
            "flesch_reading_ease": 0,
            "flesch_kincaid_grade": 0,
            "avg_sentence_length": 0,
            "avg_word_length": 0,
            "complex_word_ratio": 0,
        }
    
    text = str(text)
    
    # Tokenize
    if NLTK_AVAILABLE:
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
        except:
            sentences = re.split(r"[.!?]+", text)
            words = text.split()
    else:
        sentences = re.split(r"[.!?]+", text)
        words = text.split()
    
    sentences = [s for s in sentences if s.strip()]
    words = [w for w in words if w.isalpha()]
    
    if len(words) == 0 or len(sentences) == 0:
        return {
            "flesch_reading_ease": 0,
            "flesch_kincaid_grade": 0,
            "avg_sentence_length": 0,
            "avg_word_length": 0,
            "complex_word_ratio": 0,
        }
    
    # Count syllables (simple approximation)
    def count_syllables(word):
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith("e"):
            count -= 1
        return max(1, count)
    
    total_syllables = sum(count_syllables(w) for w in words)
    n_words = len(words)
    n_sentences = len(sentences)
    
    # Flesch Reading Ease
    flesch_ease = 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (total_syllables / n_words)
    flesch_ease = max(0, min(100, flesch_ease))
    
    # Flesch-Kincaid Grade Level
    fk_grade = 0.39 * (n_words / n_sentences) + 11.8 * (total_syllables / n_words) - 15.59
    fk_grade = max(0, fk_grade)
    
    # Average lengths
    avg_sentence_length = n_words / n_sentences
    avg_word_length = sum(len(w) for w in words) / n_words
    
    # Complex words (3+ syllables)
    complex_words = sum(1 for w in words if count_syllables(w) >= 3)
    complex_word_ratio = complex_words / n_words
    
    return {
        "flesch_reading_ease": round(flesch_ease, 1),
        "flesch_kincaid_grade": round(fk_grade, 1),
        "avg_sentence_length": round(avg_sentence_length, 1),
        "avg_word_length": round(avg_word_length, 2),
        "complex_word_ratio": round(complex_word_ratio, 3),
    }

# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def analyze_single_document(
    text: str,
    title: str = "",
    doc_id: Any = None,
    corpus_texts: List[str] = None,
    tfidf_vectorizer: Any = None,
) -> TextAnalysisResult:
    """
    Perform complete text analysis on a single document.
    
    Parameters
    ----------
    text : str
        Text to analyze.
    title : str
        Document title.
    doc_id : Any
        Document identifier.
    corpus_texts : List[str], optional
        Corpus for novelty comparison.
    tfidf_vectorizer : Any, optional
        Pre-fitted TF-IDF vectorizer.
    
    Returns
    -------
    TextAnalysisResult
    """
    combined_text = f"{title} {text}" if title else text
    
    # Run all analyses
    ner = extract_named_entities(combined_text, doc_id)
    claims = extract_claims(combined_text, doc_id)
    research_questions = extract_research_questions(combined_text, doc_id)
    methodology = classify_methodology(combined_text, doc_id)
    sentiment = analyze_sentiment(combined_text, doc_id)
    novelty = detect_novelty(combined_text, corpus_texts, doc_id, tfidf_vectorizer)
    readability = analyze_readability(combined_text)
    
    # Word and sentence counts
    words = combined_text.split()
    word_count = len(words)
    
    if NLTK_AVAILABLE:
        try:
            sentences = sent_tokenize(combined_text)
        except:
            sentences = re.split(r"[.!?]+", combined_text)
    else:
        sentences = re.split(r"[.!?]+", combined_text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    return TextAnalysisResult(
        doc_id=doc_id,
        title=title,
        ner=ner,
        claims=claims,
        research_questions=research_questions,
        methodology=methodology,
        sentiment=sentiment,
        novelty=novelty,
        readability=readability,
        word_count=word_count,
        sentence_count=sentence_count,
    )

def analyze_corpus(
    df: pd.DataFrame,
    text_col: str = "Abstract",
    title_col: str = "Title",
    id_col: str = "DOI",
    year_col: str = "Year",
    max_docs: int = None,
    verbose: bool = True,
) -> CorpusAnalysisResult:
    """
    Analyze entire corpus of documents.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bibliographic dataframe.
    text_col : str
        Text column (abstract).
    title_col : str
        Title column.
    id_col : str
        Document ID column.
    year_col : str
        Year column.
    max_docs : int, optional
        Maximum documents to analyze.
    verbose : bool
        Print progress.
    
    Returns
    -------
    CorpusAnalysisResult
    """
    if verbose:
        print("=" * 50)
        print("Text Mining & NLP Analysis")
        print("=" * 50)
    
    # Prepare corpus
    texts = df[text_col].fillna("").astype(str).tolist()
    
    if max_docs:
        texts = texts[:max_docs]
        df = df.head(max_docs)
    
    # Pre-fit TF-IDF for novelty detection
    tfidf_vectorizer = None
    if SKLEARN_AVAILABLE and len(texts) > 10:
        try:
            tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words="english",
                ngram_range=(1, 2),
            )
            tfidf_vectorizer.fit(texts)
        except:
            pass
    
    if verbose:
        print(f"  Analyzing {len(df)} documents...")
    
    # Analyze each document
    document_results = []
    method_counts = Counter()
    software_counts = Counter()
    dataset_counts = Counter()
    
    for idx, row in df.iterrows():
        text = str(row.get(text_col, ""))
        title = str(row.get(title_col, ""))
        doc_id = row.get(id_col, idx)
        
        result = analyze_single_document(
            text, title, doc_id,
            corpus_texts=texts if len(texts) > 10 else None,
            tfidf_vectorizer=tfidf_vectorizer,
        )
        document_results.append(result)
        
        # Aggregate counts
        for method, category in result.ner.methods:
            method_counts[category] += 1
        for software, category in result.ner.software:
            software_counts[software] += 1
        for dataset in result.ner.datasets:
            dataset_counts[dataset] += 1
    
    if verbose:
        print(f"  Completed analysis of {len(document_results)} documents")
    
    # Calculate averages
    avg_sentiment = {
        "polarity": np.mean([r.sentiment.polarity for r in document_results]),
        "subjectivity": np.mean([r.sentiment.subjectivity for r in document_results]),
        "certainty": np.mean([r.sentiment.certainty_score for r in document_results]),
    }
    avg_certainty = avg_sentiment["certainty"]
    avg_novelty = np.mean([r.novelty.novelty_score for r in document_results])
    
    # Methodology breakdown
    method_data = []
    for r in document_results:
        method_data.append({
            "Doc Id": r.doc_id,
            "primary_method": r.methodology.primary_method,
            "study_design": r.methodology.study_design,
            "is_empirical": r.methodology.is_empirical,
            "is_quantitative": r.methodology.is_quantitative,
            "is_qualitative": r.methodology.is_qualitative,
        })
    methodology_breakdown = pd.DataFrame(method_data)
    
    # Temporal trends
    temporal_data = []
    if year_col in df.columns:
        df_copy = df.copy()
        df_copy["_result_idx"] = range(len(df_copy))
        
        for year in df_copy[year_col].dropna().unique():
            year_mask = df_copy[year_col] == year
            year_indices = df_copy.loc[year_mask, "_result_idx"].tolist()
            year_results = [document_results[i] for i in year_indices if i < len(document_results)]
            
            if year_results:
                temporal_data.append({
                    "year": int(year),
                    "n_docs": len(year_results),
                    "avg_certainty": np.mean([r.sentiment.certainty_score for r in year_results]),
                    "avg_novelty": np.mean([r.novelty.novelty_score for r in year_results]),
                    "avg_claims": np.mean([r.claims.n_claims for r in year_results]),
                })
    
    temporal_trends = pd.DataFrame(temporal_data).sort_values("year") if temporal_data else pd.DataFrame()
    
    # Summary statistics
    summary_statistics = {
        "total_documents": len(document_results),
        "avg_word_count": np.mean([r.word_count for r in document_results]),
        "avg_sentence_count": np.mean([r.sentence_count for r in document_results]),
        "avg_claims_per_doc": np.mean([r.claims.n_claims for r in document_results]),
        "avg_methods_per_doc": np.mean([len(r.ner.methods) for r in document_results]),
        "pct_empirical": np.mean([r.methodology.is_empirical for r in document_results]) * 100,
        "pct_quantitative": np.mean([r.methodology.is_quantitative for r in document_results]) * 100,
        "pct_qualitative": np.mean([r.methodology.is_qualitative for r in document_results]) * 100,
        "top_methods": dict(method_counts.most_common(10)),
        "top_software": dict(software_counts.most_common(10)),
        "top_datasets": dict(dataset_counts.most_common(10)),
    }
    
    if verbose:
        print("\nSummary:")
        print(f"  Average certainty score: {avg_certainty:.3f}")
        print(f"  Average novelty score: {avg_novelty:.3f}")
        print(f"  Empirical studies: {summary_statistics['pct_empirical']:.1f}%")
        print(f"  Top method category: {method_counts.most_common(1)[0][0] if method_counts else 'N/A'}")
    
    return CorpusAnalysisResult(
        document_results=document_results,
        method_distribution=dict(method_counts),
        software_usage=dict(software_counts),
        dataset_usage=dict(dataset_counts),
        avg_sentiment=avg_sentiment,
        avg_certainty=avg_certainty,
        avg_novelty=avg_novelty,
        methodology_breakdown=methodology_breakdown,
        temporal_trends=temporal_trends,
        summary_statistics=summary_statistics,
    )


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

# Default color for categorical/discrete data
CATEGORICAL_COLOR = "lightblue"


def plot_method_categories(
    result: CorpusAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot method categories distribution."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    methods = result.method_distribution
    if methods:
        sorted_methods = sorted(methods.items(), key=lambda x: -x[1])[:10]
        names, counts = zip(*sorted_methods)
        
        ax.barh(range(len(names)), counts, color=CATEGORICAL_COLOR)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Count", fontsize=11)
        ax.set_title("Method Categories", fontsize=12)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No methods detected", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Method Categories", fontsize=12)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_methods.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_study_design(
    result: CorpusAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot study design distribution."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    design_counts = result.methodology_breakdown["study_design"].value_counts()
    if len(design_counts) > 0:
        ax.barh(range(len(design_counts)), design_counts.values, color=CATEGORICAL_COLOR)
        ax.set_yticks(range(len(design_counts)))
        ax.set_yticklabels(design_counts.index)
        ax.set_xlabel("Count", fontsize=11)
        ax.set_title("Study Design Distribution", fontsize=12)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No study designs detected", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Study Design Distribution", fontsize=12)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_design.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_research_approach(
    result: CorpusAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot research approach (empirical/quantitative/qualitative)."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    breakdown = result.methodology_breakdown
    categories = ["Empirical", "Quantitative", "Qualitative"]
    percentages = [
        breakdown["is_empirical"].mean() * 100,
        breakdown["is_quantitative"].mean() * 100,
        breakdown["is_qualitative"].mean() * 100,
    ]
    
    bars = ax.bar(categories, percentages, color=CATEGORICAL_COLOR)
    ax.set_ylabel("Percentage of Studies", fontsize=11)
    ax.set_title("Research Approach", fontsize=12)
    ax.set_ylim(0, 100)
    
    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=10)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_approach.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_software_usage(
    result: CorpusAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None,
) -> plt.Figure:
    """Plot software/tool usage."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    software = result.software_usage
    if software:
        sorted_sw = sorted(software.items(), key=lambda x: -x[1])[:10]
        names, counts = zip(*sorted_sw)
        
        ax.barh(range(len(names)), counts, color=CATEGORICAL_COLOR)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Mentions", fontsize=11)
        ax.set_title("Software/Tool Usage", fontsize=12)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No software detected", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Software/Tool Usage", fontsize=12)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_software.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_methodology_distribution(
    result: CorpusAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> Dict[str, plt.Figure]:
    """
    Plot methodology distribution as separate figures.
    
    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary with keys: 'methods', 'design', 'approach', 'software'
    """
    figures = {}
    figures["methods"] = plot_method_categories(result, figsize, save_path, dpi)
    figures["design"] = plot_study_design(result, figsize, save_path, dpi)
    figures["approach"] = plot_research_approach(result, figsize, save_path, dpi)
    figures["software"] = plot_software_usage(result, figsize, save_path, dpi)
    return figures


def plot_polarity_distribution(
    result: CorpusAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None,
) -> plt.Figure:
    """Plot sentiment polarity distribution."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    doc_results = result.document_results
    polarities = [r.sentiment.polarity for r in doc_results]
    ax.hist(polarities, bins=30, color="lightblue", edgecolor="white", alpha=0.7)
    ax.set_xlabel("Polarity", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Sentiment Polarity Distribution", fontsize=12)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_polarity.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_certainty_distribution(
    result: CorpusAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None,
) -> plt.Figure:
    """Plot certainty score distribution."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    doc_results = result.document_results
    certainties = [r.sentiment.certainty_score for r in doc_results]
    ax.hist(certainties, bins=30, color="lightblue", edgecolor="white", alpha=0.7)
    ax.set_xlabel("Certainty Score", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Certainty Score Distribution", fontsize=12)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_certainty.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_tone_distribution(
    result: CorpusAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot tone distribution."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    doc_results = result.document_results
    tones = [r.sentiment.tone for r in doc_results]
    tone_counts = Counter(tones)
    
    tone_names = list(tone_counts.keys())
    tone_values = list(tone_counts.values())
    
    bars = ax.bar(range(len(tone_names)), tone_values, color=CATEGORICAL_COLOR)
    ax.set_xticks(range(len(tone_names)))
    ax.set_xticklabels([t.title() for t in tone_names])
    ax.set_ylabel("Number of Documents", fontsize=11)
    ax.set_title("Tone Distribution", fontsize=12)
    
    for bar, val in zip(bars, tone_values):
        pct = val / sum(tone_values) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_tone.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_sentiment_temporal(
    result: CorpusAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None,
) -> plt.Figure:
    """Plot sentiment temporal trends."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    if len(result.temporal_trends) > 0:
        trends = result.temporal_trends
        ax.plot(trends["year"], trends["avg_certainty"], marker="o", linewidth=2,
                label="Certainty", color="lightblue")
        ax.plot(trends["year"], trends["avg_novelty"], marker="s", linewidth=2,
                label="Novelty", color="lightblue")
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("Certainty & Novelty Over Time", fontsize=12)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Temporal data not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Certainty & Novelty Over Time", fontsize=12)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_temporal.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_sentiment_analysis(
    result: CorpusAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None,
) -> Dict[str, plt.Figure]:
    """
    Plot sentiment analysis as separate figures.
    
    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary with keys: 'polarity', 'certainty', 'tone', 'temporal'
    """
    figures = {}
    figures["polarity"] = plot_polarity_distribution(result, figsize, save_path, dpi, cmap)
    figures["certainty"] = plot_certainty_distribution(result, figsize, save_path, dpi, cmap)
    figures["tone"] = plot_tone_distribution(result, figsize, save_path, dpi)
    figures["temporal"] = plot_sentiment_temporal(result, figsize, save_path, dpi, cmap)
    return figures


def plot_novelty_scores(
    result: CorpusAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None,
) -> plt.Figure:
    """Plot novelty score distribution."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    doc_results = result.document_results
    novelty_scores = [r.novelty.novelty_score for r in doc_results]
    ax.hist(novelty_scores, bins=30, color="lightblue", edgecolor="white", alpha=0.7)
    ax.set_xlabel("Novelty Score", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Novelty Score Distribution", fontsize=12)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_novelty.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_contribution_types(
    result: CorpusAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot contribution types distribution."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    doc_results = result.document_results
    contribution_types = [r.novelty.contribution_type for r in doc_results]
    type_counts = Counter(contribution_types)
    
    if type_counts:
        sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])
        names, counts = zip(*sorted_types)
        
        bars = ax.barh(range(len(names)), counts, color=CATEGORICAL_COLOR)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([n.replace("_", " ").title() for n in names])
        ax.set_xlabel("Count", fontsize=11)
        ax.set_title("Contribution Types", fontsize=12)
        ax.invert_yaxis()
        
        for bar, count in zip(bars, counts):
            pct = count / sum(counts) * 100
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{pct:.1f}%", ha="left", va="center", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No contribution types detected", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Contribution Types", fontsize=12)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_contributions.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_novelty_vs_claims(
    result: CorpusAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot novelty vs claims scatter."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    doc_results = result.document_results
    novelty_scores = [r.novelty.novelty_score for r in doc_results]
    n_claims = [r.claims.n_claims for r in doc_results]
    ax.scatter(novelty_scores, n_claims, alpha=0.5, c=CATEGORICAL_COLOR, s=30)
    ax.set_xlabel("Novelty Score", fontsize=11)
    ax.set_ylabel("Number of Claims", fontsize=11)
    ax.set_title("Novelty vs Claims", fontsize=12)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_novelty_claims.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_claim_strength(
    result: CorpusAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot claim strength distribution."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    doc_results = result.document_results
    all_strengths = []
    for r in doc_results:
        all_strengths.extend(r.claims.claim_strength)
    
    strength_counts = Counter(all_strengths)
    if strength_counts:
        bars = ax.bar(range(len(strength_counts)), strength_counts.values(), color=CATEGORICAL_COLOR)
        ax.set_xticks(range(len(strength_counts)))
        ax.set_xticklabels(strength_counts.keys())
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Claim Strength Distribution", fontsize=12)
    else:
        ax.text(0.5, 0.5, "No claims detected", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Claim Strength Distribution", fontsize=12)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_strength.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_novelty_analysis(
    result: CorpusAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
    cmap: str = None,
) -> Dict[str, plt.Figure]:
    """
    Plot novelty analysis as separate figures.
    
    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary with keys: 'novelty', 'contributions', 'novelty_claims', 'strength'
    """
    figures = {}
    figures["novelty"] = plot_novelty_scores(result, figsize, save_path, dpi, cmap)
    figures["contributions"] = plot_contribution_types(result, figsize, save_path, dpi)
    figures["novelty_claims"] = plot_novelty_vs_claims(result, figsize, save_path, dpi)
    figures["strength"] = plot_claim_strength(result, figsize, save_path, dpi)
    return figures


def plot_methods_detected(
    result: CorpusAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot methods/techniques detected."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    methods = result.method_distribution
    if methods:
        sorted_methods = sorted(methods.items(), key=lambda x: -x[1])[:10]
        names, counts = zip(*sorted_methods)
        
        ax.barh(range(len(names)), counts, color=CATEGORICAL_COLOR)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([n.replace("_", " ").title() for n in names])
        ax.set_xlabel("Count", fontsize=11)
        ax.set_title("Methods/Techniques Detected", fontsize=12)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No methods detected", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Methods/Techniques Detected", fontsize=12)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_ner_methods.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_software_detected(
    result: CorpusAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot software/tools detected."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    software = result.software_usage
    if software:
        sorted_sw = sorted(software.items(), key=lambda x: -x[1])[:10]
        names, counts = zip(*sorted_sw)
        
        ax.barh(range(len(names)), counts, color=CATEGORICAL_COLOR)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Mentions", fontsize=11)
        ax.set_title("Software/Tools Detected", fontsize=12)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No software detected", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Software/Tools Detected", fontsize=12)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_ner_software.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_datasets_detected(
    result: CorpusAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot datasets detected."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    datasets = result.dataset_usage
    if datasets:
        sorted_ds = sorted(datasets.items(), key=lambda x: -x[1])[:10]
        names, counts = zip(*sorted_ds)
        
        ax.barh(range(len(names)), counts, color=CATEGORICAL_COLOR)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Mentions", fontsize=11)
        ax.set_title("Datasets Detected", fontsize=12)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No datasets detected", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Datasets Detected", fontsize=12)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_ner_datasets.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_entity_counts(
    result: CorpusAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """Plot average entity counts per document."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.grid(False)
    
    entity_counts = {
        "Methods": np.mean([len(r.ner.methods) for r in result.document_results]),
        "Software": np.mean([len(r.ner.software) for r in result.document_results]),
        "Datasets": np.mean([len(r.ner.datasets) for r in result.document_results]),
        "Metrics": np.mean([len(r.ner.metrics) for r in result.document_results]),
    }
    
    bars = ax.bar(entity_counts.keys(), entity_counts.values(), color=CATEGORICAL_COLOR)
    ax.set_ylabel("Avg. Entities per Document", fontsize=11)
    ax.set_title("Average Entity Counts", fontsize=12)
    
    for bar, val in zip(bars, entity_counts.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    
    
    plt.tight_layout()
    
    if save_path:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(f"{save_path}_ner_counts.{ext}", dpi=dpi, bbox_inches="tight")
    
    return fig


def plot_ner_results(
    result: CorpusAnalysisResult,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 600,
) -> Dict[str, plt.Figure]:
    """
    Plot NER results as separate figures.
    
    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary with keys: 'methods', 'software', 'datasets', 'counts'
    """
    figures = {}
    figures["methods"] = plot_methods_detected(result, figsize, save_path, dpi)
    figures["software"] = plot_software_detected(result, figsize, save_path, dpi)
    figures["datasets"] = plot_datasets_detected(result, figsize, save_path, dpi)
    figures["counts"] = plot_entity_counts(result, figsize, save_path, dpi)
    return figures


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_text_analysis_results(
    result: CorpusAnalysisResult,
    output_dir: str = "results",
    prefix: str = "text_analysis",
) -> Dict[str, str]:
    """
    Export text analysis results to various formats.
    
    Parameters
    ----------
    result : CorpusAnalysisResult
        Results from analyze_corpus.
    output_dir : str
        Directory for output files.
    prefix : str
        Prefix for output filenames.
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping output type to filepath.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    exported = {}
    
    # Export methodology breakdown
    if result.methodology_breakdown is not None and len(result.methodology_breakdown) > 0:
        methodology_path = os.path.join(output_dir, f"{prefix}_methodology.csv")
        result.methodology_breakdown.to_csv(methodology_path, index=False)
        exported["methodology"] = methodology_path
    
    # Export method distribution
    if result.method_distribution:
        methods_df = pd.DataFrame([
            {"method": k, "count": v} 
            for k, v in sorted(result.method_distribution.items(), key=lambda x: -x[1])
        ])
        methods_path = os.path.join(output_dir, f"{prefix}_methods.csv")
        methods_df.to_csv(methods_path, index=False)
        exported["methods"] = methods_path
    
    # Export software usage
    if result.software_usage:
        software_df = pd.DataFrame([
            {"software": k, "count": v}
            for k, v in sorted(result.software_usage.items(), key=lambda x: -x[1])
        ])
        software_path = os.path.join(output_dir, f"{prefix}_software.csv")
        software_df.to_csv(software_path, index=False)
        exported["software"] = software_path
    
    # Export dataset usage
    if result.dataset_usage:
        datasets_df = pd.DataFrame([
            {"dataset": k, "count": v}
            for k, v in sorted(result.dataset_usage.items(), key=lambda x: -x[1])
        ])
        datasets_path = os.path.join(output_dir, f"{prefix}_datasets.csv")
        datasets_df.to_csv(datasets_path, index=False)
        exported["datasets"] = datasets_path
    
    # Export temporal trends
    if result.temporal_trends is not None and len(result.temporal_trends) > 0:
        temporal_path = os.path.join(output_dir, f"{prefix}_temporal.csv")
        result.temporal_trends.to_csv(temporal_path, index=False)
        exported["temporal"] = temporal_path
    
    # Export summary statistics
    if result.summary_statistics:
        summary_df = pd.DataFrame([result.summary_statistics])
        summary_path = os.path.join(output_dir, f"{prefix}_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        exported["summary"] = summary_path
    
    # Export document-level results
    doc_data = []
    for i, doc in enumerate(result.document_results):
        doc_data.append({
            "Doc Id": i,
            "polarity": doc.sentiment.polarity,
            "subjectivity": doc.sentiment.subjectivity,
            "tone": doc.sentiment.tone,
            "certainty_score": doc.sentiment.certainty_score,
            "novelty_score": doc.novelty.novelty_score,
            "contribution_type": doc.novelty.contribution_type,
            "n_claims": doc.claims.n_claims,
            "n_methods": len(doc.ner.methods),
            "n_software": len(doc.ner.software),
            "n_datasets": len(doc.ner.datasets),
        })
    
    if doc_data:
        docs_df = pd.DataFrame(doc_data)
        docs_path = os.path.join(output_dir, f"{prefix}_documents.csv")
        docs_df.to_csv(docs_path, index=False)
        exported["documents"] = docs_path
    
    print(f"Exported {len(exported)} files to {output_dir}/")
    for key, path in exported.items():
        print(f"  - {key}: {os.path.basename(path)}")
    
    return exported


if __name__ == "__main__":
    import biblium as bb
    ba = bb.BiblioAnalysis(f_name="data\\open alex dataset.csv", db="oa")
    
    print("Text Mining & NLP Analysis")
    print("=" * 60)
    
    # Run analysis
    result = analyze_corpus(
        df=ba.df,
        text_col="Abstract",
        title_col="Title",
        year_col="Year",
        max_docs=500,
        verbose=True,
    )
    
    # Print summary
    print(f"\nAnalyzed {result.summary_statistics['total_documents']} documents")
    print(f"Average certainty: {result.avg_certainty:.3f}")
    
    # Visualizations
    print("\nGenerating plots...")
    plot_methodology_distribution(result, save_path="results/methodology_distribution")
    plot_sentiment_analysis(result, save_path="results/sentiment_analysis")
    plot_novelty_analysis(result, save_path="results/novelty_analysis")
    
    # Export
    print("\nExporting results...")
    export_text_analysis_results(result, "results")
    
    print("\nDone!")
