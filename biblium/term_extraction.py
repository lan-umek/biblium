# -*- coding: utf-8 -*-
"""
biblium.term_extraction
=======================

VOSviewer-style term extraction from titles and abstracts (new in 2.19),
following van Eck & Waltman (2011, "Text mining and visualization using
VOSviewer"). The pipeline is

1. candidate noun phrases (``extract_candidate_phrases``),
2. a binary document x term matrix with a minimum number of occurrences
   (``build_term_document_matrix``),
3. a *relevance score* per term -- the Kullback-Leibler divergence between
   the term's co-occurrence distribution and the overall distribution of
   term occurrences (``relevance_scores``) -- and
4. selection of the most relevant share of terms (``select_terms``;
   VOSviewer's default is 60 %).

VOSviewer identifies noun phrases with a part-of-speech tagger. Without a
tagger this module approximates noun phrases by *stopword chunking*: the
token stream is split at stopwords (function words and common academic
verbs) and at tokens containing digits, and every contiguous n-gram inside
a chunk is a candidate phrase. Plurals of the last word are singularised
with a small conservative rule set.

Contents
--------
``STOPWORDS``, ``GENERIC_TERMS``
``extract_candidate_phrases``, ``singularize``
``build_term_document_matrix``, ``relevance_scores``, ``select_terms``
``extract_terms``                 full pipeline
``term_cooccurrence``             term x term matrix for network plotting
``add_terms_column``              per-document "; "-joined terms
``plot_term_relevance``           horizontal bars (matplotlib)

All functions are pure and deterministic; no external NLP dependencies.

Author: Lan Umek
Version: 2.19.0
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    "STOPWORDS",
    "GENERIC_TERMS",
    "singularize",
    "extract_candidate_phrases",
    "build_term_document_matrix",
    "relevance_scores",
    "select_terms",
    "extract_terms",
    "term_cooccurrence",
    "add_terms_column",
    "plot_term_relevance",
]

# ---------------------------------------------------------------------------
# Word lists
# ---------------------------------------------------------------------------

#: English function words plus frequent academic verbs / participles (the
#: latter stand in for the part-of-speech tagger VOSviewer uses to drop
#: non-noun tokens). Chunks are split at these tokens.
STOPWORDS: FrozenSet[str] = frozenset("""
a an the and or but nor so yet for of in on at to from by with without within
into onto upon about above below under over between among through during before
after across along around against toward towards via per
is are was were be been being am
has have had having do does did doing done
can could may might must shall should will would
this that these those such same other another each every either neither both all
any some few many much more most less least several various
i me my mine we us our ours you your yours he him his she her hers it its they them
their theirs who whom whose which what where when why how whether while
as than then there here thus hence therefore however moreover furthermore also
not no nor only own very too quite rather almost already still just even
if unless until because since although though despite whereas
one two three first second third
new further recent current present
use used uses using based based-on
show shows showed shown found find finds finding
propose proposed proposes provide provides provided
suggest suggests suggested indicate indicates indicated
examine examines examined investigate investigates investigated
explore explores explored identify identifies identified
analyze analyzes analyzed analyse analyses analysed
develop develops developed apply applies applied
conduct conducted consider considered include includes including
compare compared discuss discussed describe described present presented
obtain obtained perform performed require required
make makes made take takes taken give gives given
et al ie eg etc vs
""".split())

#: Generic research vocabulary that carries no topical information. Candidate
#: phrases consisting *only* of these words are dropped when
#: ``drop_generic=True``.
GENERIC_TERMS: FrozenSet[str] = frozenset("""
study studies paper papers article articles research work works
result results finding findings analysis approach approaches method methods
methodology methodologies data dataset datasets sample samples
aim aims objective objectives purpose goal goals
implication implications contribution contributions limitation limitations
literature review reviews overview background introduction conclusion conclusions
discussion section chapter table figure
author authors researcher researchers reader readers
evidence insight insights understanding knowledge
issue issues problem problems question questions topic topics field fields area areas
case cases context contexts example examples
effect effects impact impacts role roles importance influence
number total part parts way ways type types kind kinds level levels
year years time times period periods
process processes practice practices framework frameworks model models
relationship relationships difference differences factor factors
term terms respect order addition
""".split())

_WORD_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_SENT_SPLIT_RE = re.compile(r"[.;:?!,()\[\]{}\"“”‘’/]+")
_DIGIT_RE = re.compile(r"\d")

_IRREGULAR_SINGULAR: Dict[str, str] = {
    "analyses": "analysis", "hypotheses": "hypothesis", "theses": "thesis",
    "crises": "crisis", "diagnoses": "diagnosis", "bases": "basis", "axes": "axis",
    "indices": "index", "matrices": "matrix", "vertices": "vertex", "appendices": "appendix",
    "criteria": "criterion", "phenomena": "phenomenon", "media": "medium",
    "children": "child", "people": "people", "men": "man", "women": "woman",
    "feet": "foot", "teeth": "tooth", "mice": "mouse", "data": "data",
    "series": "series", "species": "species", "news": "news", "means": "means",
}
_INVARIANT_SUFFIXES = ("ss", "us", "is", "ics", "ous")


# ---------------------------------------------------------------------------
# Phrase extraction
# ---------------------------------------------------------------------------

def singularize(word: str) -> str:
    """
    Conservative English singularisation of one (lower-case) word.

    Rules: a small irregular table; ``-ies`` -> ``-y`` (``studies`` ->
    ``study``); ``-sses``/``-xes``/``-ches``/``-shes``/``-zes`` -> strip
    ``es`` (``processes`` -> ``process``); otherwise strip a trailing ``s``
    when the word is longer than 3 characters and does not end in ``ss``,
    ``us``, ``is``, ``ics`` or ``ous`` (``cases`` -> ``case``,
    ``bibliometrics`` unchanged). Hyphenated words are singularised on their
    last part.
    """
    if not word:
        return word
    if "-" in word:
        head, _, tail = word.rpartition("-")
        return f"{head}-{singularize(tail)}"
    if word in _IRREGULAR_SINGULAR:
        return _IRREGULAR_SINGULAR[word]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "xes", "ches", "shes", "zes")) and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and len(word) > 3 and not word.endswith(_INVARIANT_SUFFIXES):
        return word[:-1]
    return word


def _chunks(text: object, stopwords: FrozenSet[str]) -> List[List[str]]:
    """Token chunks of ``text``: sentences are split on ``.;:?!`` (and on
    commas / brackets / quotes / slashes), tokenised on non-alphanumerics
    (hyphens inside words are kept) and cut at stopwords, at tokens that
    contain a digit and at single-character tokens."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return []
    s = str(text).lower()
    out: List[List[str]] = []
    for sent in _SENT_SPLIT_RE.split(s):
        cur: List[str] = []
        for tok in _WORD_RE.findall(sent):
            if (tok in stopwords or len(tok) < 2 or _DIGIT_RE.search(tok)
                    or all(p in stopwords for p in tok.split("-"))):
                if cur:
                    out.append(cur)
                cur = []
            else:
                cur.append(tok)
        if cur:
            out.append(cur)
    return out


def _ngrams_of_chunk(chunk: Sequence[str], max_len: int, min_len: int) -> List[Tuple[int, int, str]]:
    """``(start, length, phrase)`` for every contiguous n-gram of the chunk
    (last word singularised)."""
    out = []
    n = len(chunk)
    for i in range(n):
        for L in range(min_len, max_len + 1):
            if i + L > n:
                break
            words = list(chunk[i:i + L])
            words[-1] = singularize(words[-1])
            out.append((i, L, " ".join(words)))
    return out


def extract_candidate_phrases(
    text: object,
    max_len: int = 3,
    min_len: int = 1,
    stopwords: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Candidate noun phrases of one text (approximation of VOSviewer's noun
    phrase identification; see the module docstring).

    The text is lower-cased, split into sentences on ``.;:?!`` (and on
    commas, brackets, quotes and slashes), tokenised on non-alphanumeric
    characters with hyphens kept inside words, and the token stream is cut
    at stopwords, at tokens containing digits and at single-character
    tokens. Every contiguous n-gram of
    length ``min_len``..``max_len`` inside a chunk is emitted (in order of
    appearance, with repetitions); the last word of each phrase is
    singularised.

    >>> extract_candidate_phrases("Bibliometric analysis of public management reform "
    ...                           "using the Scopus database", max_len=2)
    ['bibliometric', 'bibliometric analysis', 'analysis', 'public', 'public management',
     'management', 'management reform', 'reform', 'scopus', 'scopus database', 'database']
    """
    if min_len < 1 or max_len < min_len:
        raise ValueError("need 1 <= min_len <= max_len")
    sw = STOPWORDS if stopwords is None else frozenset(str(w).lower() for w in stopwords)
    phrases: List[str] = []
    for ch in _chunks(text, sw):
        phrases.extend(p for _, _, p in _ngrams_of_chunk(ch, max_len, min_len))
    return phrases


def _is_generic(phrase: str) -> bool:
    return all(w in GENERIC_TERMS for w in phrase.split())


# ---------------------------------------------------------------------------
# Term-document matrix
# ---------------------------------------------------------------------------

_COPYRIGHT_RE = re.compile(r"(©|\(c\)\s*\d{4}|copyright\s*©?\s*\d{4}).*$", re.I | re.S)
_NO_ABSTRACT_RE = re.compile(r"^\s*\[?no abstract available\]?\s*\.?\s*$", re.I)


def _clean_field(text: str, strip_copyright: bool) -> str:
    t = text.strip()
    if _NO_ABSTRACT_RE.match(t):
        return ""
    if strip_copyright:
        t = _COPYRIGHT_RE.sub("", t).strip()
    return t


def _doc_text(row: pd.Series, columns: Sequence[str], strip_copyright: bool = True) -> str:
    """Concatenate the text columns of one document (joined by ". " so that
    phrases never span two fields). Scopus artefacts -- the copyright trailer
    ("© 2020 Elsevier B.V. ...") and "[No abstract available]" -- are
    removed when ``strip_copyright`` is True."""
    parts = []
    for c in columns:
        if c in row.index and not pd.isna(row[c]):
            t = _clean_field(str(row[c]), strip_copyright)
            if t:
                parts.append(t)
    return ". ".join(parts)


def build_term_document_matrix(
    df: pd.DataFrame,
    columns: Sequence[str] = ("Title", "Abstract"),
    max_len: int = 3,
    min_occurrences: int = 10,
    min_docs: Optional[int] = None,
    stopwords: Optional[Iterable[str]] = None,
    drop_generic: bool = True,
    binary: bool = True,
    merge_nested: bool = True,
    nested_share: float = 0.9,
    min_len: int = 1,
    strip_copyright: bool = True,
) -> Tuple[sp.csr_matrix, List[str]]:
    """
    Document x term matrix of candidate phrases.

    Parameters
    ----------
    columns : sequence of str
        Text columns concatenated per document (missing ones are ignored).
    min_occurrences : int
        Keep terms with at least this many occurrences (with ``binary=True``
        an occurrence is a document, as in VOSviewer).
    min_docs : int, optional
        Additional minimum number of documents (useful with
        ``binary=False``).
    drop_generic : bool
        Drop phrases consisting only of :data:`GENERIC_TERMS`.
    binary : bool
        True: 1 if the term occurs in the document (VOSviewer counts a term
        once per document); False: raw counts.
    merge_nested : bool
        Drop a shorter term whose occurrences are, in at least
        ``nested_share`` of the cases, part of an occurrence of a longer
        kept term (e.g. "network" when it almost only appears inside
        "citation network").
    strip_copyright : bool
        Remove Scopus copyright trailers ("© 2020 Elsevier ...") and
        "[No abstract available]" placeholders before extraction.

    Returns
    -------
    X : scipy.sparse.csr_matrix, shape (n_docs, n_terms)
    terms : list of str
        Column labels, sorted alphabetically.
    """
    sw = STOPWORDS if stopwords is None else frozenset(str(w).lower() for w in stopwords)
    cols = [c for c in columns if c in df.columns]
    if not cols:
        raise ValueError(f"None of the columns {tuple(columns)} found")

    doc_chunks: List[List[List[str]]] = []
    doc_counts: List[Counter] = []
    for _, row in df.iterrows():
        chs = _chunks(_doc_text(row, cols, strip_copyright), sw)
        cnt: Counter = Counter()
        for ch in chs:
            for _, _, p in _ngrams_of_chunk(ch, max_len, min_len):
                if drop_generic and _is_generic(p):
                    continue
                cnt[p] += 1
        doc_chunks.append(chs)
        doc_counts.append(cnt)

    occ: Counter = Counter()
    dfreq: Counter = Counter()
    for cnt in doc_counts:
        for p, k in cnt.items():
            occ[p] += 1 if binary else k
            dfreq[p] += 1
    kept: Set[str] = {p for p, k in occ.items() if k >= min_occurrences}
    if min_docs is not None:
        kept = {p for p in kept if dfreq[p] >= min_docs}

    if merge_nested and kept:
        kept_by_len: Dict[int, Set[str]] = {}
        for p in kept:
            kept_by_len.setdefault(len(p.split()), set()).add(p)
        max_kept_len = max(kept_by_len)
        total: Counter = Counter()
        covered: Counter = Counter()
        for chs in doc_chunks:
            for ch in chs:
                grams = _ngrams_of_chunk(ch, max_len, min_len)
                long_occ = [(i, L) for i, L, p in grams if L > 1 and p in kept]
                for i, L, p in grams:
                    if p not in kept or L >= max_kept_len:
                        continue
                    total[p] += 1
                    if any(j <= i and i + L <= j + M and M > L for j, M in long_occ):
                        covered[p] += 1
        for p in list(kept):
            if total[p] > 0 and covered[p] / total[p] >= nested_share:
                kept.discard(p)

    terms = sorted(kept)
    col = {t: j for j, t in enumerate(terms)}
    rows, cols_idx, vals = [], [], []
    for i, cnt in enumerate(doc_counts):
        for p, k in cnt.items():
            j = col.get(p)
            if j is not None:
                rows.append(i)
                cols_idx.append(j)
                vals.append(1 if binary else k)
    X = sp.csr_matrix((np.asarray(vals, dtype=float), (rows, cols_idx)),
                      shape=(len(df), len(terms)))
    return X, terms


# ---------------------------------------------------------------------------
# Relevance
# ---------------------------------------------------------------------------

def relevance_scores(X, terms: Sequence[str]) -> pd.DataFrame:
    """
    VOSviewer relevance score of every term.

    With document-level co-occurrence counts ``c_ij`` (``X.T @ X`` with a
    zero diagonal), ``p_i(j) = c_ij / sum_j c_ij`` is the co-occurrence
    distribution of term *i* and ``q(j) = sum_i c_ij / sum_ij c_ij`` the
    overall distribution. The relevance of *i* is the Kullback-Leibler
    divergence ``sum_j p_i(j) log(p_i(j) / q(j))`` (zero-probability terms
    are skipped; terms without any co-occurrence get 0). Generic terms that
    co-occur with everything have a distribution close to ``q`` and thus a
    low relevance; specific terms have a high one.

    Returns
    -------
    pd.DataFrame
        ``Term``, ``Occurrences`` (column sums of ``X``), ``Relevance``,
        ``Relevance rank`` (1 = most relevant), sorted by rank.
    """
    X = sp.csr_matrix(X)
    terms = list(terms)
    if X.shape[1] != len(terms):
        raise ValueError("number of columns of X must equal len(terms)")
    n = len(terms)
    occ = np.asarray(X.sum(axis=0)).ravel()
    if n == 0:
        return pd.DataFrame(columns=["Term", "Occurrences", "Relevance", "Relevance rank"])
    B = X.copy()
    B.data = np.ones_like(B.data)  # binary presence for co-occurrence
    C = (B.T @ B).toarray().astype(float)
    np.fill_diagonal(C, 0.0)
    row_tot = C.sum(axis=1)
    grand = C.sum()
    q = C.sum(axis=0) / grand if grand > 0 else np.zeros(n)
    rel = np.zeros(n)
    for i in range(n):
        if row_tot[i] <= 0:
            continue
        p = C[i] / row_tot[i]
        mask = p > 0
        rel[i] = float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
    out = pd.DataFrame({"Term": terms, "Occurrences": occ, "Relevance": rel})
    out = out.sort_values(["Relevance", "Occurrences", "Term"], ascending=[False, False, True])
    out["Relevance rank"] = np.arange(1, n + 1)
    out["Occurrences"] = out["Occurrences"].astype(int) if np.allclose(occ, np.round(occ)) else out["Occurrences"]
    return out.reset_index(drop=True)


def select_terms(
    scores: pd.DataFrame,
    share: float = 0.6,
    min_relevance: Optional[float] = None,
) -> pd.DataFrame:
    """
    Keep the most relevant ``share`` of terms (``ceil(share * n)``,
    VOSviewer's default 60 %) and, optionally, only terms with
    ``Relevance >= min_relevance``. Returns the filtered table ordered by
    ``Relevance rank``.
    """
    if not 0 < share <= 1:
        raise ValueError("share must be in (0, 1]")
    s = scores.sort_values("Relevance rank") if "Relevance rank" in scores.columns else scores
    k = int(math.ceil(share * len(s)))
    out = s.head(k)
    if min_relevance is not None:
        out = out[out["Relevance"] >= min_relevance]
    return out.reset_index(drop=True)


def extract_terms(
    df: pd.DataFrame,
    columns: Sequence[str] = ("Title", "Abstract"),
    max_len: int = 3,
    min_occurrences: int = 10,
    share: float = 0.6,
    min_relevance: Optional[float] = None,
    **kw,
) -> Tuple[pd.DataFrame, sp.csr_matrix, List[str]]:
    """
    Full VOSviewer-style pipeline: candidate phrases -> term-document matrix
    (``min_occurrences``) -> relevance scores -> the most relevant ``share``.

    Extra keyword arguments go to :func:`build_term_document_matrix`.

    Returns
    -------
    selected : pd.DataFrame
        Term table (``Term, Occurrences, Relevance, Relevance rank``) of the
        selected terms.
    X_sel : scipy.sparse.csr_matrix
        Document x selected-term matrix (columns in the order of ``terms``).
    terms : list of str
        Selected terms (alphabetical).
    """
    X, terms = build_term_document_matrix(df, columns=columns, max_len=max_len,
                                          min_occurrences=min_occurrences, **kw)
    scores = relevance_scores(X, terms)
    selected = select_terms(scores, share=share, min_relevance=min_relevance)
    sel_terms = sorted(selected["Term"].tolist())
    idx = [terms.index(t) for t in sel_terms]
    X_sel = sp.csr_matrix(X[:, idx]) if idx else sp.csr_matrix((X.shape[0], 0))
    return selected, X_sel, sel_terms


# ---------------------------------------------------------------------------
# Co-occurrence and document column
# ---------------------------------------------------------------------------

def term_cooccurrence(X, terms: Sequence[str]) -> pd.DataFrame:
    """
    Symmetric term x term DataFrame of document co-occurrence counts; the
    diagonal holds the number of documents containing the term. Feed it to
    ``biblium.utilsbib_modules.network.matrix_to_network``.
    """
    X = sp.csr_matrix(X)
    terms = list(terms)
    if X.shape[1] != len(terms):
        raise ValueError("number of columns of X must equal len(terms)")
    B = X.copy()
    B.data = np.ones_like(B.data)
    C = (B.T @ B).toarray()
    return pd.DataFrame(C.astype(int), index=terms, columns=terms)


def add_terms_column(
    df: pd.DataFrame,
    X,
    terms: Sequence[str],
    col: str = "Extracted Terms",
    sep: str = "; ",
) -> pd.DataFrame:
    """Copy of ``df`` with a column of the document's terms (alphabetical,
    joined by ``sep``; empty string when none)."""
    X = sp.csr_matrix(X)
    terms = list(terms)
    if X.shape[0] != len(df) or X.shape[1] != len(terms):
        raise ValueError("X must have shape (len(df), len(terms))")
    out = df.copy()
    vals = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        js = row.indices[row.data > 0]
        vals.append(sep.join(sorted(terms[j] for j in js)))
    out[col] = pd.Series(vals, index=df.index)
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _finish(fig, ax, title, f_name, dpi, show):
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if f_name:
        fig.savefig(f_name, dpi=dpi, bbox_inches="tight")
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return fig, ax


def plot_term_relevance(
    scores: pd.DataFrame,
    top_n: int = 30,
    ax=None,
    figsize: Optional[Tuple[float, float]] = None,
    color: str = "#1f77b4",
    annotate_occurrences: bool = True,
    title: Optional[str] = "Term relevance (VOSviewer score)",
    f_name: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
):
    """Horizontal bars of the ``top_n`` most relevant terms (most relevant on
    top); the number of occurrences is written at the bar end."""
    import matplotlib.pyplot as plt
    s = scores.sort_values("Relevance", ascending=False).head(int(top_n))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (7, max(3, 0.28 * len(s) + 1)))
    else:
        fig = ax.figure
    y = np.arange(len(s))[::-1]
    ax.barh(y, s["Relevance"].to_numpy(dtype=float), color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(s["Term"].tolist(), fontsize=8)
    ax.set_xlabel("Relevance")
    if annotate_occurrences and "Occurrences" in s.columns:
        for yi, (r, o) in zip(y, zip(s["Relevance"], s["Occurrences"])):
            ax.text(r, yi, f" {int(o)}", va="center", fontsize=7)
    return _finish(fig, ax, title, f_name, dpi, show)
