# -*- coding: utf-8 -*-
"""
Embedding Landscape Addon - paper-level semantic landscapes
=============================================================

Functions
---------
- embed_corpus(df, text_col, method, model_name, n_components, output_dir)
    Embed papers either via sentence-transformers (SBERT) or TF-IDF +
    TruncatedSVD fallback. Always returns a dense (n, d) ndarray. The
    `method='auto'` setting picks SBERT when available, otherwise TF-IDF.
- reduce_to_2d(embeddings, method, n_neighbors, min_dist, random_state)
    UMAP -> t-SNE -> PCA cascade depending on availability.
- cluster_landscape(coords, method, n_clusters, min_cluster_size)
    HDBSCAN (sklearn.cluster.HDBSCAN if available) with KMeans fallback;
    -1 indicates noise (only for HDBSCAN).
- plot_paper_landscape(coords, labels, sizes, hover_labels,
                        concept_overlays, out, title, figsize, max_labels)
    Scatter without grid, NAVY single-color when no labels, tab20 when
    labels are given. Concept overlays draw outline contours per concept.
    Labels are wrapped via biblium.utilsbib_modules.plotting_helpers.wrap_label.
- find_similar_papers(query_idx, embeddings, n, df)
    Cosine similarity ranking; returns DataFrame with ranks, sim and any
    extra columns from `df`.

Design notes
------------
- Honours user memory rules: no grid lines (ax.set_axis_off()),
  no truncated labels, single-series scatter uses one color (NAVY).
- biblium-first: this is a pure library module; the orchestrator script
  lives in the project _scripts folder.
- Sandbox-safe: SBERT import is guarded; sklearn TruncatedSVD covers the
  pipeline when torch is unavailable.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# safe imports
# ---------------------------------------------------------------------------
def _safe_import_sbert():
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        return True
    except Exception:
        return False


def _safe_import_umap():
    try:
        import umap  # noqa: F401
        return True
    except Exception:
        return False


def _safe_import_hdbscan():
    """Prefer sklearn.cluster.HDBSCAN (sklearn>=1.3) then standalone hdbscan."""
    try:
        from sklearn.cluster import HDBSCAN  # noqa: F401
        return "sklearn"
    except Exception:
        pass
    try:
        import hdbscan  # noqa: F401
        return "hdbscan"
    except Exception:
        return None


def _resolve_navy() -> str:
    try:
        from biblium.config import plot_config  # type: ignore

        palette = list(plot_config.categorical_palette)
        if palette:
            return palette[0]
    except Exception:
        pass
    return "#1F3864"


def _wrap(label: Any, width: int = 18) -> str:
    try:
        from biblium.utilsbib_modules.plotting_helpers import wrap_label

        return wrap_label(label, width=width)
    except Exception:
        import textwrap

        s = str(label).replace(" and ", " & ")
        if len(s) <= width:
            return s
        return "\n".join(textwrap.wrap(s, width=width)) or s


# ---------------------------------------------------------------------------
# embedding
# ---------------------------------------------------------------------------
def _tfidf_svd(
    texts: List[str],
    n_components: int = 256,
    random_state: int = 2026,
) -> Tuple[np.ndarray, str]:
    """TF-IDF (1-2 ngrams) + TruncatedSVD -> dense float32."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
        max_features=80_000,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    n_comp = min(n_components, max(2, min(X.shape) - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    Z = svd.fit_transform(X)
    # L2-normalise so cosine = dot
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Z = (Z / norms).astype(np.float32)
    return Z, f"tfidf+svd(n_comp={n_comp})"


def _sbert(
    texts: List[str],
    model_name: Optional[str] = None,
    batch_size: int = 32,
) -> Tuple[np.ndarray, str]:
    from sentence_transformers import SentenceTransformer

    if model_name is None:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    return emb, model_name


def embed_corpus(
    df: pd.DataFrame,
    text_col: str = "Processed Combined Text",
    method: str = "auto",
    model_name: Optional[str] = None,
    n_components: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, str, str]:
    """Embed papers.

    Returns
    -------
    embeddings : ndarray (n, d)
    method_used : str   one of {"sbert", "tfidf"}
    model_name_used : str (e.g. "tfidf+svd(n_comp=256)" or HF id)
    """
    if text_col not in df.columns:
        raise ValueError(
            f"text_col '{text_col}' not in dataframe; "
            f"available: {list(df.columns)[:10]}..."
        )
    texts = df[text_col].fillna("").astype(str).tolist()
    if not texts:
        raise ValueError("no texts to embed")

    if method == "sbert":
        if not _safe_import_sbert():
            raise ImportError("sentence-transformers not available")
        emb, used = _sbert(texts, model_name=model_name)
        used_method = "sbert"
    elif method == "tfidf":
        n_comp = n_components or 256
        emb, used = _tfidf_svd(texts, n_components=n_comp)
        used_method = "tfidf"
    elif method == "auto":
        if _safe_import_sbert():
            try:
                emb, used = _sbert(texts, model_name=model_name)
                used_method = "sbert"
            except Exception as e:
                warnings.warn(f"SBERT failed ({e}); falling back to TF-IDF")
                n_comp = n_components or 256
                emb, used = _tfidf_svd(texts, n_components=n_comp)
                used_method = "tfidf"
        else:
            n_comp = n_components or 256
            emb, used = _tfidf_svd(texts, n_components=n_comp)
            used_method = "tfidf"
    else:
        raise ValueError(f"unknown method='{method}'")

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "embeddings.npy", emb)
        with (out / "embedding_meta.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "method": used_method,
                    "model_name": used,
                    "shape": list(emb.shape),
                    "text_col": text_col,
                },
                fh,
                indent=2,
            )
    return emb, used_method, used


# ---------------------------------------------------------------------------
# 2-D reduction
# ---------------------------------------------------------------------------
def reduce_to_2d(
    embeddings: np.ndarray,
    method: str = "umap",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 2026,
) -> np.ndarray:
    """Reduce embeddings to 2D. Cascade: requested -> tsne -> pca."""
    if embeddings.shape[0] < 3:
        raise ValueError("need >= 3 rows to reduce")

    if method == "umap" and _safe_import_umap():
        import umap

        reducer = umap.UMAP(
            n_neighbors=min(n_neighbors, embeddings.shape[0] - 1),
            min_dist=min_dist,
            n_components=2,
            metric="cosine",
            random_state=random_state,
        )
        return reducer.fit_transform(embeddings)

    if method == "tsne" or method == "umap":
        try:
            from sklearn.manifold import TSNE

            perplexity = max(5, min(30, embeddings.shape[0] // 4))
            ts = TSNE(
                n_components=2,
                metric="cosine",
                init="pca",
                perplexity=perplexity,
                random_state=random_state,
            )
            return ts.fit_transform(embeddings)
        except Exception as e:
            warnings.warn(f"t-SNE failed ({e}); falling back to PCA")

    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=random_state).fit_transform(embeddings)


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------
def cluster_landscape(
    coords: np.ndarray,
    method: str = "kmeans",
    n_clusters: int = 10,
    min_cluster_size: int = 20,
    random_state: int = 2026,
) -> np.ndarray:
    """Cluster 2-D coords. method 'hdbscan' falls back to kmeans if missing."""
    if method == "hdbscan":
        which = _safe_import_hdbscan()
        if which == "sklearn":
            from sklearn.cluster import HDBSCAN

            model = HDBSCAN(min_cluster_size=min_cluster_size)
            return model.fit_predict(coords)
        if which == "hdbscan":
            import hdbscan

            model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            return model.fit_predict(coords)
        warnings.warn("hdbscan unavailable; using kmeans")
        method = "kmeans"

    from sklearn.cluster import KMeans

    n_clusters = max(2, min(n_clusters, coords.shape[0] - 1))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    return km.fit_predict(coords)


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _label_top_clusters(
    ax,
    coords: np.ndarray,
    labels: np.ndarray,
    hover_labels: Optional[List[str]],
    max_labels: int,
) -> None:
    """Annotate the centroid of the largest clusters."""
    if labels is None:
        return
    uniq, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    placed = 0
    for ui in order:
        if placed >= max_labels:
            break
        lbl = uniq[ui]
        if lbl == -1:
            continue  # noise
        mask = labels == lbl
        if mask.sum() < 3:
            continue
        cx = float(np.median(coords[mask, 0]))
        cy = float(np.median(coords[mask, 1]))
        if hover_labels is not None:
            # pick most central paper title
            cluster_pts = coords[mask]
            d = np.linalg.norm(cluster_pts - np.array([cx, cy]), axis=1)
            idxs = np.where(mask)[0]
            rep = idxs[int(np.argmin(d))]
            text = _wrap(hover_labels[rep], width=24)
        else:
            text = f"C{int(lbl)}"
        ax.annotate(
            text,
            xy=(cx, cy),
            ha="center",
            va="center",
            fontsize=8,
            weight="bold",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="0.7",
                alpha=0.85,
            ),
            zorder=10,
        )
        placed += 1


def filter_language(
    df: pd.DataFrame,
    text_col: str = "Processed Combined Text",
    lang: str = "en",
    threshold: float = 0.85,
) -> np.ndarray:
    """Return boolean mask True where text is in ``lang``.

    Strategy (in order, best-effort):
    1. If a ``Language`` column exists, match by string.
    2. If ``langdetect`` is installed, run detection (returns the
       most-probable language per row; mask = detected == lang).
    3. Fallback heuristic: ratio of English stop-words ('the', 'of', 'and',
       'to', 'in') vs total tokens >= 0.02 means English.
    """
    n = len(df)
    if "Language of Original Document" in df.columns:
        col = df["Language of Original Document"].fillna("").astype(str).str.lower()
        return col.str.contains(lang, regex=False).values
    if "Language" in df.columns:
        col = df["Language"].fillna("").astype(str).str.lower()
        return col.str.contains(lang, regex=False).values
    try:  # pragma: no cover - optional dep
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        out = np.zeros(n, dtype=bool)
        texts = df[text_col].fillna("").astype(str).tolist()
        for i, t in enumerate(texts):
            t = t.strip()
            if len(t) < 30:
                out[i] = True  # too short to judge -> keep
                continue
            try:
                out[i] = detect(t[:600]) == lang
            except Exception:
                out[i] = True
        return out
    except Exception:
        # Heuristic fallback (English only).
        if lang != "en":
            return np.ones(n, dtype=bool)
        stops = {"the", "of", "and", "to", "in", "a", "is", "for", "on", "with"}
        out = np.zeros(n, dtype=bool)
        texts = df[text_col].fillna("").astype(str).tolist()
        for i, t in enumerate(texts):
            tokens = [w for w in t.lower().split() if w.isalpha()]
            if len(tokens) < 5:
                out[i] = True
                continue
            ratio = sum(1 for w in tokens if w in stops) / len(tokens)
            out[i] = ratio >= 0.02
        return out


def plot_paper_landscape(
    coords: np.ndarray,
    labels: Optional[np.ndarray] = None,
    sizes: Optional[Iterable[float]] = None,
    hover_labels: Optional[List[str]] = None,
    concept_overlays: Optional[Dict[str, np.ndarray]] = None,
    out: Optional[Union[str, Path]] = None,
    title: str = "Paper landscape",
    figsize: Tuple[float, float] = (12, 10),
    max_labels: int = 30,
    cmap_categorical: str = "tab20",
    annotate_top_papers: Optional[Dict[str, int]] = None,
    paper_metrics: Optional[Dict[str, np.ndarray]] = None,
    paper_titles: Optional[List[str]] = None,
) -> None:
    """Plot a 2-D paper landscape scatter.

    Parameters
    ----------
    coords : (n, 2) ndarray
    labels : optional cluster id per point (-1 = noise)
    sizes : optional per-point marker size (default 14)
    hover_labels : optional per-point string (used to annotate clusters)
    concept_overlays : optional dict {concept_name: bool_mask(n)}; drawn
        as outline scatter on top.
    """
    coords = np.asarray(coords)
    n = coords.shape[0]
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    if sizes is None:
        s_arr = np.full(n, 14.0)
    else:
        s_arr = np.asarray(list(sizes), dtype=float)
        if s_arr.shape[0] != n:
            s_arr = np.full(n, 14.0)

    if labels is None:
        navy = _resolve_navy()
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=s_arr,
            c=navy,
            alpha=0.55,
            linewidths=0,
            zorder=2,
        )
    else:
        labels = np.asarray(labels)
        uniq = np.unique(labels)
        cmap = matplotlib.colormaps.get_cmap(cmap_categorical)
        # noise first in grey
        noise = labels == -1
        if noise.any():
            ax.scatter(
                coords[noise, 0],
                coords[noise, 1],
                s=s_arr[noise] * 0.7,
                c="#BBBBBB",
                alpha=0.3,
                linewidths=0,
                zorder=1,
            )
        non_noise = [u for u in uniq if u != -1]
        for i, lbl in enumerate(non_noise):
            mask = labels == lbl
            color = cmap(i % cmap.N)
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=s_arr[mask],
                color=color,
                alpha=0.65,
                linewidths=0,
                label=f"C{int(lbl)}",
                zorder=2,
            )
        _label_top_clusters(ax, coords, labels, hover_labels, max_labels)

    # concept overlays
    if concept_overlays:
        # restricted categorical palette; outline strokes for distinguishability
        overlay_cmap = matplotlib.colormaps.get_cmap("tab10")
        for i, (name, mask) in enumerate(concept_overlays.items()):
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() == 0:
                continue
            color = overlay_cmap(i % overlay_cmap.N)
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=s_arr[mask] * 2.5,
                facecolor="none",
                edgecolor=color,
                linewidths=0.9,
                alpha=0.8,
                zorder=4,
                label=_wrap(name, width=18),
            )
        ax.legend(
            loc="best",
            frameon=False,
            fontsize=8,
            handletextpad=0.4,
            labelspacing=0.4,
        )

    # Annotate top papers (reviewer feedback C8): label e.g. the 20 most-cited
    # plus the 10 most cluster-characteristic papers with short wrapped titles.
    if annotate_top_papers and paper_titles is not None:
        try:
            from matplotlib import patheffects as _pe
        except Exception:
            _pe = None
        chosen: list[int] = []
        # (1) Top-cited
        if "cited" in annotate_top_papers and paper_metrics is not None:
            arr = paper_metrics.get("cited")
            if arr is not None and len(arr) == n:
                k = int(annotate_top_papers["cited"])
                order = np.argsort(-np.asarray(arr, dtype=float))
                chosen.extend(int(i) for i in order[:k])
        # (2) Top characteristic: closest to its cluster centroid (lower L2 = more
        # characteristic). Falls back to silently doing nothing when labels are
        # absent.
        if "characteristic" in annotate_top_papers and labels is not None:
            k = int(annotate_top_papers["characteristic"])
            lbl_arr = np.asarray(labels)
            ranks: list[tuple[int, float]] = []
            for u in np.unique(lbl_arr):
                if u == -1:
                    continue
                mask = lbl_arr == u
                if mask.sum() == 0:
                    continue
                cx = coords[mask, 0].mean()
                cy = coords[mask, 1].mean()
                d = np.hypot(coords[mask, 0] - cx, coords[mask, 1] - cy)
                idxs = np.where(mask)[0]
                # Closest few per cluster
                for j in np.argsort(d)[:max(1, k // max(len(np.unique(lbl_arr)), 1))]:
                    ranks.append((int(idxs[j]), float(d[j])))
            ranks.sort(key=lambda t: t[1])
            chosen.extend(idx for idx, _ in ranks[:k])
        # Deduplicate while preserving order
        seen = set(); ordered = []
        for i in chosen:
            if i not in seen and 0 <= i < n:
                seen.add(i); ordered.append(i)
        for idx in ordered[: max_labels]:
            txt = str(paper_titles[idx])[:80] if idx < len(paper_titles) else ""
            if not txt:
                continue
            t = ax.text(
                coords[idx, 0], coords[idx, 1],
                _wrap(txt, width=22),
                fontsize=7, ha="center", va="center",
                zorder=11,
            )
            if _pe is not None:
                t.set_path_effects([
                    _pe.withStroke(linewidth=2.5, foreground="white"),
                ])

    ax.set_title(_wrap(title, width=60), fontsize=12)
    fig.tight_layout()

    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# similarity
# ---------------------------------------------------------------------------
def find_similar_papers(
    query_idx: int,
    embeddings: np.ndarray,
    n: int = 10,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return top-n cosine-similar papers to the query index."""
    q = embeddings[query_idx]
    # assume L2-normalised; fall back to explicit cosine
    qn = q / max(np.linalg.norm(q), 1e-12)
    en = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
    )
    sims = en @ qn
    order = np.argsort(-sims)
    order = [i for i in order if i != query_idx][:n]
    rows = []
    for rank, idx in enumerate(order, start=1):
        row = {"rank": rank, "doc_idx": int(idx), "cosine": float(sims[idx])}
        if df is not None and idx < len(df):
            for col in ("Title", "Year", "DOI"):
                if col in df.columns:
                    row[col] = df.iloc[idx][col]
        rows.append(row)
    return pd.DataFrame(rows)



# ---------------------------------------------------------------------------
# cluster terms (TF-IDF top-n per cluster)
# ---------------------------------------------------------------------------
def compute_cluster_terms(
    texts: List[str],
    labels: np.ndarray,
    top_n: int = 10,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 3,
    max_df: float = 0.85,
    max_features: int = 30_000,
    drop_noise: bool = True,
    min_cluster_size: int = 2,
) -> pd.DataFrame:
    """Compute top-N TF-IDF terms per cluster (or any categorical label).

    For each unique label in `labels`, fits a TF-IDF on all texts then
    averages tf-idf weights within the cluster, ranks terms by mean weight.

    Parameters
    ----------
    texts : list of str
        Texts (length n).
    labels : ndarray (n,)
        Cluster id per text. Use -1 for noise / unassigned.
    top_n : int
        Number of top terms per cluster.
    ngram_range, min_df, max_df, max_features : passed to TfidfVectorizer.
    drop_noise : bool
        If True (default), skip label == -1.
    min_cluster_size : int
        Skip clusters smaller than this.

    Returns
    -------
    DataFrame
        Columns: cluster, size, top_terms (semicolon-joined string).
        Sorted by size desc.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    if len(texts) != len(labels):
        raise ValueError(
            f"len(texts)={len(texts)} != len(labels)={len(labels)}"
        )

    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())

    rows = []
    labels_arr = np.asarray(labels)
    for lbl in sorted({int(l) for l in labels_arr}):
        if drop_noise and lbl == -1:
            continue
        mask = labels_arr == lbl
        if mask.sum() < min_cluster_size:
            continue
        sub = X[mask].mean(axis=0)
        arr = np.asarray(sub).ravel()
        top = np.argsort(-arr)[:top_n]
        rows.append({
            "cluster": int(lbl),
            "size": int(mask.sum()),
            "top_terms": "; ".join(vocab[top]),
        })
    if not rows:
        return pd.DataFrame(columns=["cluster", "size", "top_terms"])
    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)


__all__ = [
    "embed_corpus",
    "reduce_to_2d",
    "cluster_landscape",
    "plot_paper_landscape",
    "find_similar_papers",
    "compute_cluster_terms",
]
