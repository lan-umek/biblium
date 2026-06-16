# -*- coding: utf-8 -*-
"""
Group pair analysis methods for BiblioGroup.

Provides analysis of relationships BETWEEN groups (concept × concept):
- PMI / log-lift matrix (symmetric, signed association strength)
- Conditional probability P(g2|g1) (asymmetric)
- Citation impact matrix (mean citations of papers in both groups)
- Pair year trends (top pairs over time, lead-lag)
- Group-level bibliographic coupling (Jaccard over reference sets per group)

These complement plot_group_overlapping (which provides venn/upset/heatmap/
dendrogram of overlaps using Jaccard/raw counts) by adding:
  - signed association (PMI/lift) for "more/less than chance"
  - asymmetric direction (conditional probability)
  - quality outcome (citation impact)
  - temporal dynamics for pairs
  - reference-base similarity (different lens than text overlap)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class GroupPairsMixin:
    """Mixin providing concept × concept pair analysis methods."""

    def compute_group_pmi(
        self,
        smoothing: float = 1e-9,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Compute group × group co-occurrence association matrices.

        Returns three DataFrames (square, indexed by group names):
          - raw_counts: number of documents in BOTH groups
          - lift:      P(g1 ∩ g2) / (P(g1) * P(g2)); >1 = positive assoc
          - pmi:       log(lift); centered at 0, useful for diverging heatmap

        Sets attributes:
          self.group_pair_raw, self.group_pair_lift, self.group_pair_pmi

        Parameters
        ----------
        smoothing : float
            Small constant to avoid divide-by-zero in lift/PMI when a pair
            has zero co-occurrence.

        Notes
        -----
        Requires ``self.group_matrix`` (built by BiblioGroupAnalysis init
        with ``group_desc=...``). Each row is a document, each column a group.
        """
        if not hasattr(self, "group_matrix") or self.group_matrix is None:
            raise RuntimeError(
                "group_matrix not built. Pass `group_desc` to "
                "BiblioGroupAnalysis(...) constructor."
            )
        M = self.group_matrix.astype(int)
        N = len(M)
        raw = M.T.dot(M).astype(int)
        sizes = pd.Series(np.diag(raw.values), index=raw.index).astype(float)
        p = sizes.values / max(N, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = (raw.values / max(N, 1)) / (
                (p[:, None] * p[None, :]) + smoothing
            )
            lift = np.where(np.isfinite(lift), lift, 0.0)
            pmi = np.log(np.where(lift > 0, lift, 1.0))
        lift_df = pd.DataFrame(lift, index=raw.index, columns=raw.columns)
        pmi_df = pd.DataFrame(pmi, index=raw.index, columns=raw.columns)

        self.group_pair_raw = raw
        self.group_pair_lift = lift_df
        self.group_pair_pmi = pmi_df
        return raw, lift_df, pmi_df

    def compute_group_conditional(self) -> pd.DataFrame:
        """
        Compute asymmetric conditional probability matrix P(col | row).

        ``cond.loc[g1, g2]`` = probability that a document in g1 is also
        in g2. Useful to identify directional dependencies (e.g.
        P(Performance | NPM) vs P(NPM | Performance)).

        Returns
        -------
        pd.DataFrame (square, indexed by group names)

        Sets attributes:
          self.group_pair_conditional
        """
        if not hasattr(self, "group_matrix") or self.group_matrix is None:
            raise RuntimeError(
                "group_matrix not built. Pass `group_desc` to "
                "BiblioGroupAnalysis(...) constructor."
            )
        M = self.group_matrix.astype(int)
        raw = M.T.dot(M).astype(int)
        sizes = pd.Series(np.diag(raw.values), index=raw.index).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            cond = raw.values / sizes.values[:, None].clip(min=1)
        cond_df = pd.DataFrame(cond, index=raw.index, columns=raw.columns)
        self.group_pair_conditional = cond_df
        return cond_df

    def get_group_citation_impact(
        self,
        citations_col: str = "Cited by",
        min_papers: int = 5,
    ) -> pd.DataFrame:
        """
        Mean citations per pair of groups.

        ``impact.loc[g1, g2]`` = mean of ``citations_col`` over documents
        that are in BOTH g1 and g2 (diagonal = mean within group g1).
        Pairs with fewer than ``min_papers`` joint documents are NaN.

        Parameters
        ----------
        citations_col : str
            Column with citation counts (default ``"Cited by"`` — Scopus).
            Use ``"oa_cited_by_count"`` for OpenAlex.
        min_papers : int
            Minimum number of joint documents to compute a mean.

        Returns
        -------
        pd.DataFrame
            Square matrix (group × group) of mean citations.

        Sets attributes:
          self.group_pair_citation_impact
        """
        if not hasattr(self, "group_matrix") or self.group_matrix is None:
            raise RuntimeError(
                "group_matrix not built. Pass `group_desc` to "
                "BiblioGroupAnalysis(...) constructor."
            )
        if citations_col not in self.df.columns:
            raise KeyError(f"Citation column '{citations_col}' not in df.")
        cits = pd.to_numeric(self.df[citations_col], errors="coerce").fillna(0).values
        M = self.group_matrix.astype(bool).values
        names = list(self.group_matrix.columns)
        n = len(names)
        impact = np.full((n, n), np.nan, dtype=float)
        for i in range(n):
            m1 = M[:, i]
            for j in range(n):
                m2 = M[:, j]
                both = m1 & m2
                k = int(both.sum())
                if k >= min_papers:
                    impact[i, j] = float(round(cits[both].mean(), 2))
        df_out = pd.DataFrame(impact, index=names, columns=names)
        self.group_pair_citation_impact = df_out
        return df_out

    def analyze_pair_year_trends(
        self,
        top_n_pairs: int = 10,
        year_col: str = "Year",
        n_windows: int = 4,
        rank_by: str = "pmi",
        current_year_exclude: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        For the top-N group pairs (ranked by PMI, lift, or raw count),
        compute their co-occurrence rate across time windows.

        Parameters
        ----------
        top_n_pairs : int
            Number of top pairs to track over time.
        year_col : str
            Column with publication year.
        n_windows : int
            Number of equal-width time windows.
        rank_by : {"pmi", "lift", "raw"}
            Metric to rank pairs by when picking the top-N.
        current_year_exclude : int or None
            If set, drop years >= this value (e.g. 2026 for partial year).

        Returns
        -------
        pd.DataFrame
            Long-form: one row per (pair, window) with co-occurrence share.

        Sets attributes:
          self.group_pair_year_trends
        """
        # Ensure pair metrics computed
        if not hasattr(self, "group_pair_pmi") or self.group_pair_pmi is None:
            self.compute_group_pmi()

        # Pick top pairs
        if rank_by == "pmi":
            rank_M = self.group_pair_pmi
        elif rank_by == "lift":
            rank_M = self.group_pair_lift
        elif rank_by == "raw":
            rank_M = self.group_pair_raw
        else:
            raise ValueError(f"rank_by must be 'pmi', 'lift', or 'raw'")
        rows = []
        names = list(rank_M.index)
        for i, c1 in enumerate(names):
            for j, c2 in enumerate(names):
                if i >= j:
                    continue
                rows.append({"c1": c1, "c2": c2,
                             "score": float(rank_M.iloc[i, j])})
        pairs_df = (pd.DataFrame(rows).sort_values("score", ascending=False)
                       .head(top_n_pairs))

        # Time bins
        if year_col not in self.df.columns:
            raise KeyError(f"Year column '{year_col}' not in df.")
        years = pd.to_numeric(self.df[year_col], errors="coerce")
        mask = years.notna()
        if current_year_exclude is not None:
            mask &= (years < current_year_exclude)
        sub_df = self.df[mask].copy()
        sub_gm = self.group_matrix.loc[sub_df.index]
        sub_y = pd.to_numeric(sub_df[year_col], errors="coerce").astype(int)
        ymin, ymax = int(sub_y.min()), int(sub_y.max())
        cuts = np.linspace(ymin, ymax + 1, n_windows + 1).astype(int)
        bin_labels = [f"{cuts[i]}-{cuts[i+1]-1}" for i in range(n_windows)]
        bin_idx = pd.cut(sub_y, bins=list(cuts), labels=bin_labels,
                          include_lowest=True, right=False)

        out_rows = []
        for _, p in pairs_df.iterrows():
            c1, c2 = p["c1"], p["c2"]
            for b in bin_labels:
                bmask = (bin_idx == b).values
                if bmask.sum() == 0:
                    continue
                gm_b = sub_gm.loc[bmask]
                m1 = gm_b[c1].astype(bool)
                m2 = gm_b[c2].astype(bool)
                n12 = int((m1 & m2).sum())
                n1 = int(m1.sum())
                n_tot = int(bmask.sum())
                out_rows.append({
                    "pair": f"{c1} × {c2}",
                    "c1": c1, "c2": c2,
                    "bin": str(b),
                    "n_total_in_bin": n_tot,
                    "n_c1": n1,
                    "n_c1_and_c2": n12,
                    "share_c1_in_bin_pct":
                        round(100 * n1 / max(n_tot, 1), 2),
                    "share_c1c2_of_bin_pct":
                        round(100 * n12 / max(n_tot, 1), 2),
                    "share_c1c2_of_c1_pct":
                        round(100 * n12 / max(n1, 1), 2),
                })
        out = pd.DataFrame(out_rows)
        self.group_pair_year_trends = out
        return out

    def plot_pair_year_trends(
        self,
        metric: str = "share_c1c2_of_bin_pct",
        cmap: str = "tab20",
        figsize: Tuple[float, float] = (14, 6),
        filename: Optional[str] = None,
    ):
        """
        Plot pair co-occurrence trends over time (must call
        ``analyze_pair_year_trends`` first).

        Parameters
        ----------
        metric : str
            Column name in ``self.group_pair_year_trends`` to plot
            (default: % of papers in bin that are in both groups).
        cmap : str
            Categorical colormap for distinguishing pairs.
        filename : str or None
            If given, save plot to res_folder/plots/<filename>.png.
        """
        import matplotlib.pyplot as plt

        if not hasattr(self, "group_pair_year_trends") \
                or self.group_pair_year_trends is None \
                or self.group_pair_year_trends.empty:
            raise RuntimeError(
                "Call analyze_pair_year_trends() first."
            )
        d = self.group_pair_year_trends
        pairs = list(d["pair"].unique())
        bin_order = list(d["bin"].unique())
        cmap_f = plt.get_cmap(cmap, max(len(pairs), 1))

        # Use textwrap-aware labels and an outside legend; allocate extra
        # horizontal room so labels are never clipped.
        import textwrap as _tw
        fig, ax = plt.subplots(figsize=figsize)
        for i, pair in enumerate(pairs):
            sub = d[d["pair"] == pair]
            xs = [bin_order.index(b) for b in sub["bin"]]
            ys = sub[metric].values
            label = _tw.fill(str(pair), width=42)
            ax.plot(xs, ys, marker="o", ms=4, lw=1.6,
                    color=cmap_f(i % cmap_f.N), label=label)
        ax.set_xticks(range(len(bin_order)))
        ax.set_xticklabels(bin_order, rotation=20)
        ax.set_ylabel(metric)
        ax.set_xlabel("Time window")
        ax.set_title(f"Top {len(pairs)} group-pair co-occurrence over time")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, fontsize=7)
        ax.grid(False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        # Leave 32% of width on the right for the legend (avoids cut labels).
        fig.subplots_adjust(right=0.68)
        if filename and getattr(self, "res_folder", None):
            out = Path(self.res_folder) / "plots" / f"{filename}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"Saved to {out}")
        return fig

    def get_group_bibliographic_coupling(
        self,
        refs_col: str = "oa_referenced_works",
        sep: str = "|",
    ) -> pd.DataFrame:
        """
        Group-level bibliographic coupling: similarity of groups by their
        aggregate reference base. For each group, take the UNION of
        ``refs_col`` (semicolon/pipe-separated reference IDs) across its
        documents; compute Jaccard similarity between groups.

        Parameters
        ----------
        refs_col : str
            Column with reference IDs (default ``"oa_referenced_works"``).
            For Scopus text refs use ``"References"`` with ``sep="; "``.
        sep : str
            Delimiter between reference IDs in each cell.

        Returns
        -------
        pd.DataFrame
            Square Jaccard similarity matrix (group × group) over reference sets.

        Sets attributes:
          self.group_bibliographic_coupling
        """
        if not hasattr(self, "group_matrix") or self.group_matrix is None:
            raise RuntimeError(
                "group_matrix not built. Pass `group_desc` to "
                "BiblioGroupAnalysis(...) constructor."
            )
        if refs_col not in self.df.columns:
            raise KeyError(f"Reference column '{refs_col}' not in df.")

        # Union of refs per group
        group_refs: Dict[str, set] = {}
        refs_series = self.df[refs_col].fillna("").astype(str)
        for g in self.group_matrix.columns:
            m = self.group_matrix[g].astype(bool).values
            all_r: set[str] = set()
            for s in refs_series[m]:
                if not s:
                    continue
                for r in s.split(sep):
                    r = r.strip()
                    if r:
                        all_r.add(r)
            group_refs[g] = all_r

        # Pairwise Jaccard
        names = list(group_refs.keys())
        n = len(names)
        J = np.zeros((n, n), dtype=float)
        for i, c1 in enumerate(names):
            s1 = group_refs[c1]
            for j, c2 in enumerate(names):
                s2 = group_refs[c2]
                if not s1 and not s2:
                    J[i, j] = 0.0
                    continue
                inter = len(s1 & s2)
                union = len(s1 | s2)
                J[i, j] = inter / union if union > 0 else 0.0
        out = pd.DataFrame(J, index=names, columns=names)
        self.group_bibliographic_coupling = out
        # Store ref-set sizes too
        self.group_ref_set_sizes = pd.Series(
            {g: len(s) for g, s in group_refs.items()}, name="n_unique_refs"
        )
        return out

    # =================================================================
    # SCATTER / PCA helperji za vizualne predstavitve skupin
    # =================================================================
    def plot_group_scatter(
        self,
        x_metric: str = "n_papers",
        y_metric: str = "mean_citations",
        size_metric: Optional[str] = "pct_multicountry",
        color_metric: Optional[str] = "mean_year",
        citation_col: str = "Cited by",
        year_col: str = "Year",
        country_col: str = "oa_institution_countries",
        country_sep: str = "; ",
        min_papers: int = 30,
        x_scale: str = "log",
        y_scale: str = "log",
        filename: Optional[str] = "group_scatter",
        figsize: Tuple[float, float] = (11, 7.5),
    ):
        """
        Scatter koncept x velikost x impact x internacionalizacija x cas.

        Za vsak koncept v ``self.group_matrix`` izracuna n_papers,
        mean_citations, h_index, mean_year, pct_multicountry in narise
        scatter (log-log) z labeli vseh konceptov, barvno skalo (mean_year)
        in size legendo (pct_multicountry).
        """
        import matplotlib.pyplot as plt
        from matplotlib import patheffects

        if not hasattr(self, "group_matrix") or self.group_matrix is None:
            raise RuntimeError(
                "group_matrix not built. Pass `group_desc=` to "
                "BiblioGroupAnalysis(...) constructor."
            )

        df = self.df
        if citation_col in df.columns:
            df_c = pd.to_numeric(df[citation_col], errors="coerce").fillna(0)
        else:
            df_c = pd.Series(0, index=df.index)
        if year_col in df.columns:
            df_y = pd.to_numeric(df[year_col], errors="coerce")
        else:
            df_y = pd.Series(np.nan, index=df.index)
        if country_col in df.columns:
            n_countries = (
                df[country_col].fillna("").astype(str)
                .apply(lambda s: len({c.strip()
                                      for c in s.split(country_sep)
                                      if c.strip()}))
            )
        else:
            n_countries = pd.Series(1, index=df.index)

        rows = []
        for g in self.group_matrix.columns:
            m = self.group_matrix[g].astype(bool).values
            if m.sum() < min_papers:
                continue
            cits = df_c[m].values
            yrs = df_y[m].dropna().values
            s = sorted(cits, reverse=True)
            h = 0
            for i, c in enumerate(s, 1):
                if c >= i:
                    h = i
                else:
                    break
            multi_ct = float((n_countries[m] >= 2).mean()) * 100
            mean_y = float(round(yrs.mean(), 1)) if len(yrs) else float("nan")
            rows.append({
                "group": g,
                "n_papers": int(m.sum()),
                "total_citations": int(cits.sum()),
                "mean_citations": float(round(cits.mean(), 2)),
                "median_citations": float(np.median(cits)),
                "h_index": int(h),
                "mean_year": mean_y,
                "pct_multicountry": float(round(multi_ct, 1)),
            })
        gsc = (pd.DataFrame(rows)
                  .sort_values("n_papers", ascending=False)
                  .reset_index(drop=True))
        self.group_scatter_df = gsc

        if gsc.empty:
            return gsc, None

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        x = gsc[x_metric]
        y = gsc[y_metric]
        if size_metric and size_metric in gsc.columns:
            sizes = 60 + 12 * gsc[size_metric]
        else:
            sizes = 100
        if color_metric and color_metric in gsc.columns:
            colors = gsc[color_metric]
            sc = ax.scatter(x, y, s=sizes, c=colors, cmap="viridis",
                             alpha=0.78, edgecolors="white", linewidths=0.8)
            cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label(color_metric, fontsize=9)
        else:
            ax.scatter(x, y, s=sizes, c="#1F3864", alpha=0.78,
                        edgecolors="white", linewidths=0.8)
        for _, r in gsc.iterrows():
            ax.text(r[x_metric], r[y_metric], "  " + str(r["group"]),
                     fontsize=8, ha="left", va="center",
                     path_effects=[patheffects.withStroke(
                         linewidth=2, foreground="white")])
        if x_scale == "log":
            ax.set_xscale("log")
        if y_scale == "log":
            ax.set_yscale("log")
        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        ax.set_title(
            f"Group scatter ({len(gsc)} groups; "
            f"size={size_metric}, color={color_metric})"
        )
        if size_metric and size_metric in gsc.columns:
            vmax = float(gsc[size_metric].max())
            for v in [vmax * 0.1, vmax * 0.4, vmax]:
                ax.scatter([], [], s=60 + 12 * v, c="#888", alpha=0.5,
                            label=f"{size_metric}={v:.0f}")
            ax.legend(loc="lower right", frameon=False, fontsize=7,
                       handletextpad=1.5, borderaxespad=0.5)

        if filename and getattr(self, "res_folder", None):
            from pathlib import Path
            out = Path(self.res_folder) / "plots" / f"{filename}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=getattr(self, "dpi", 200),
                         bbox_inches="tight")
            print(f"Saved to {out}")

        return gsc, fig

    def plot_entity_concept_pca(
        self,
        entity_col: str = "Author full names",
        sep: str = "; ",
        top_n: int = 60,
        min_count: int = 5,
        clean_pattern: Optional[str] = r"\s*\(\d+\)\s*$",
        label_top_n: int = 25,
        filename: Optional[str] = "entity_concept_pca",
        figsize: Tuple[float, float] = (13, 9),
    ):
        """
        PCA na entiteta x koncept profilu (avtorji/viri/institucije).

        Za top N entitet (po pogostosti v entity_col) zgradi vektor
        normaliziranih frekvenc po skupinah iz self.group_matrix,
        izvede 2D PCA in narise scatter (barva = dominantni koncept,
        velikost = sqrt(n_papers)).

        Returns (DataFrame s coords + dominant concept, matplotlib Figure).
        """
        import re as _re
        import matplotlib.pyplot as plt
        from matplotlib import patheffects
        from matplotlib.patches import Patch
        from sklearn.decomposition import PCA

        if not hasattr(self, "group_matrix") or self.group_matrix is None:
            raise RuntimeError("group_matrix not built.")
        if entity_col not in self.df.columns:
            raise KeyError(f"Stolpec '{entity_col}' ne obstaja.")

        cleaner = _re.compile(clean_pattern) if clean_pattern else None

        def clean_one(s):
            if not isinstance(s, str):
                return ""
            s = s.strip()
            if cleaner:
                s = cleaner.sub("", s)
            return s.strip()

        rows = []
        for idx, val in self.df[entity_col].items():
            if not isinstance(val, str) or not val.strip():
                continue
            for ent in val.split(sep):
                ent = clean_one(ent)
                if ent:
                    rows.append({"entity": ent, "doc": idx})
        ed = pd.DataFrame(rows)
        if ed.empty:
            return pd.DataFrame(), None
        counts = ed["entity"].value_counts()
        top_ent = counts[counts >= min_count].head(top_n).index.tolist()
        if not top_ent:
            return pd.DataFrame(), None

        ec = pd.DataFrame(0.0, index=top_ent,
                          columns=self.group_matrix.columns)
        for e in top_ent:
            doc_idxs = ed.loc[ed["entity"] == e, "doc"].unique()
            for c in self.group_matrix.columns:
                ec.loc[e, c] = float(self.group_matrix.loc[doc_idxs, c].sum())
        ec_norm = ec.div(ec.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

        pca = PCA(n_components=2, random_state=2026)
        coords = pca.fit_transform(ec_norm.values)
        ev_ratio = pca.explained_variance_ratio_

        dom = ec_norm.idxmax(axis=1)
        unique_dom = sorted(dom.unique())
        cmap = plt.get_cmap("tab20", max(len(unique_dom), 1))
        color_map = {c: cmap(i % cmap.N) for i, c in enumerate(unique_dom)}
        colors = [color_map[c] for c in dom]
        n_p = [int(counts[e]) for e in top_ent]
        max_n = max(n_p)
        sizes = [60 + 250 * (np.sqrt(n) / np.sqrt(max_n)) for n in n_p]

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax.scatter(coords[:, 0], coords[:, 1], s=sizes, c=colors,
                    alpha=0.78, edgecolors="white", linewidths=0.8)
        top_label_idx = sorted(range(len(top_ent)),
                                key=lambda i: -n_p[i])[:label_top_n]
        for i in top_label_idx:
            ax.text(coords[i, 0], coords[i, 1],
                     "  " + str(top_ent[i])[:25],
                     fontsize=7, ha="left", va="center",
                     path_effects=[patheffects.withStroke(
                         linewidth=2, foreground="white")])
        ax.set_xlabel(f"PC1 ({ev_ratio[0] * 100:.1f} % variance)")
        ax.set_ylabel(f"PC2 ({ev_ratio[1] * 100:.1f} % variance)")
        ax.set_title(
            f"PCA on {entity_col} x concept profile "
            f"(top {len(top_ent)})"
        )
        legend_doms = unique_dom[:12]
        handles = [Patch(color=color_map[c], label=str(c)[:30])
                    for c in legend_doms]
        ax.legend(handles=handles, loc="center left",
                   bbox_to_anchor=(1.02, 0.5),
                   frameon=False, fontsize=7,
                   title="Dominant concept", title_fontsize=8)

        result_df = pd.DataFrame({
            "entity": top_ent,
            "n_records": n_p,
            "dominant_concept": dom.values,
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
        })
        self.entity_concept_pca_df = result_df
        self.entity_concept_pca_variance = list(ev_ratio)

        if filename and getattr(self, "res_folder", None):
            from pathlib import Path
            out = Path(self.res_folder) / "plots" / f"{filename}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=getattr(self, "dpi", 200),
                         bbox_inches="tight")
            print(f"Saved to {out}")

        return result_df, fig
