# -*- coding: utf-8 -*-
"""
GroupContentAnalysisMixin — vsebinske razširitve za BiblioGroup.

Pet novih analiz, ki sledijo BiblioGroupAnalysis (ovojnica nad BiblioGroup):

A. crosstab_with_column(other_col) — crosstab koncept × kategorialna spr.
   (npr. Document Type, Open Access, Language)
B. analyze_citations_by_group_year — povprečni/mediana citati per group × year
C. analyze_oa_share_by_group — OA share per group + lift vs overall + Phi
E. compute_group_entropy — Shannon entropija entitet znotraj koncepta
   (visoka = razpršen koncept, nizka = specializiran)
H. analyze_group_overlap_over_time — Jaccard koncept × koncept v
   več časovnih oknih

D (top cited per group) je že implementiran kot get_group_top_cited_documents
v analysis.py; F (concept × top entitete heatmap) je plot_group_metric_heatmap.

Vse metode shranijo tabelo v <res>/tables/ in opcijsko sliko v <res>/plots/.

@author: Lan + biblium 2.16
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Helpers
# =============================================================================

def _save_xlsx(df: pd.DataFrame, res_folder: Optional[str], filename: str) -> None:
    if res_folder is None or filename is None:
        return
    tables_dir = os.path.join(res_folder, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    df.to_excel(os.path.join(tables_dir, f"{filename}.xlsx"), index=True)


def _save_fig(fig, res_folder: Optional[str], filename: str, dpi: int = 200) -> None:
    if res_folder is None or filename is None:
        return
    plots_dir = os.path.join(res_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, f"{filename}.png"),
                dpi=dpi, bbox_inches="tight")


def _shannon_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _gini(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts >= 0]
    if counts.sum() == 0 or len(counts) < 2:
        return 0.0
    counts = np.sort(counts)
    n = len(counts)
    cum = np.cumsum(counts)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def _herfindahl(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    if counts.sum() == 0:
        return 0.0
    p = counts / counts.sum()
    return float((p ** 2).sum())


def _phi_2x2(a: int, b: int, c: int, d: int) -> float:
    """Pearson phi za 2×2 kontingenčno tabelo."""
    n = a + b + c + d
    denom = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denom == 0 or n == 0:
        return 0.0
    return float((a * d - b * c) / denom)


# =============================================================================
# Mixin
# =============================================================================

class GroupContentAnalysisMixin:
    """Razširitve za vsebinske analize konceptov / skupin."""

    # ----------------------------------------------------------------
    # A) crosstab koncept × katerakoli kategorialna spr.
    # ----------------------------------------------------------------
    def crosstab_with_column(
        self,
        other_col: str,
        normalize: bool = False,
        plot: bool = True,
        filename: str = "concept_crosstab",
        figsize: Tuple[float, float] = (10, 6),
    ) -> pd.DataFrame:
        """
        Crosstab vsake skupine v group_matrix proti drugemu kategorialnemu
        stolpcu (npr. "Document Type", "Open Access", "Language of Original
        Document").

        Vrstice = kategorije iz other_col. Stolpci = skupine. Celice = št.
        dokumentov, ki so v skupini IN imajo to vrednost v other_col.

        Parametri
        ---------
        normalize : bool ali {"row", "col"}
            False = surovi count. "col" = column-share (% znotraj skupine).
            True ali "row" = row-share (% znotraj kategorije).
        """
        if not hasattr(self, "group_matrix") or self.group_matrix is None:
            raise AttributeError("self.group_matrix manjka.")
        if other_col not in self.df.columns:
            raise ValueError(f"Stolpec {other_col!r} ne obstaja.")

        gm = self.group_matrix.astype(int)
        groups = list(gm.columns)
        cats = self.df[other_col].fillna("[missing]").astype(str)

        rows = []
        for cat, mask in cats.groupby(cats):
            idx = self.df.index[cats == cat]
            row = gm.loc[idx].sum().rename(cat)
            rows.append(row)
        ct = pd.concat(rows, axis=1).T
        ct.index.name = other_col

        # Normalize
        if normalize == "col" or normalize is True and False:
            ct_n = ct.div(ct.sum(axis=0).replace(0, np.nan), axis=1) * 100
        elif normalize == "row" or normalize is True:
            ct_n = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100
        else:
            ct_n = None

        out = ct if ct_n is None else ct_n.round(2)

        res_folder = getattr(self, "res_folder", None)
        _save_xlsx(out, res_folder, filename)

        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            data = out.values
            im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
            ax.set_xticks(range(len(out.columns)))
            ax.set_xticklabels(out.columns, rotation=30, ha="right")
            ax.set_yticks(range(len(out.index)))
            ax.set_yticklabels(out.index)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    v = data[i, j]
                    if pd.notna(v) and v > 0:
                        col = "white" if v > np.nanmax(data) * 0.5 else "black"
                        fmt = "{:.0f}" if normalize is False else "{:.1f}%"
                        ax.text(j, i, fmt.format(v), ha="center", va="center",
                                fontsize=8, color=col)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Concept × {other_col}"
                         + ("" if normalize is False else f" ({normalize}-normalized)"))
            ax.set_xlabel("Concept")
            ax.set_ylabel(other_col)
            fig.tight_layout()
            _save_fig(fig, res_folder, filename, dpi=getattr(self, "dpi", 200))
            plt.close(fig)

        return out

    # ----------------------------------------------------------------
    # B) Citati per group × year — mean + median + total
    # ----------------------------------------------------------------
    def analyze_citations_by_group_year(
        self,
        citation_col: str = "Cited by",
        year_col: str = "Year",
        year_range: Optional[Tuple[int, int]] = None,
        plot: bool = True,
        filename: str = "citations_by_group_year",
        figsize: Tuple[float, float] = (12, 6),
    ) -> pd.DataFrame:
        """
        Per koncept × leto: število zapisov + povprečni/mediana citati.

        Vrne long DataFrame: group, year, n_papers, mean_cit, median_cit, total_cit.
        Plot: line chart povprečnih citatov per group skozi leta.
        """
        if not hasattr(self, "group_matrix") or self.group_matrix is None:
            raise AttributeError("self.group_matrix manjka.")
        for c in (citation_col, year_col):
            if c not in self.df.columns:
                raise ValueError(f"Stolpec {c!r} ne obstaja.")

        years = pd.to_numeric(self.df[year_col], errors="coerce")
        valid = years.notna()
        if year_range:
            valid &= (years >= year_range[0]) & (years <= year_range[1])
        cits = pd.to_numeric(self.df[citation_col], errors="coerce").fillna(0)

        gm = self.group_matrix.astype(int)
        rows = []
        for g in gm.columns:
            in_g = gm[g].astype(bool) & valid
            sub = pd.DataFrame({
                "year": years[in_g].astype(int),
                "cit": cits[in_g],
            })
            agg = sub.groupby("year")["cit"].agg(
                n_papers="count", mean_cit="mean",
                median_cit="median", total_cit="sum"
            ).reset_index()
            agg["group"] = g
            rows.append(agg)

        long = pd.concat(rows, ignore_index=True)
        long = long[["group", "year", "n_papers", "mean_cit", "median_cit", "total_cit"]]
        long = long.sort_values(["group", "year"]).reset_index(drop=True)

        res_folder = getattr(self, "res_folder", None)
        _save_xlsx(long.set_index(["group", "year"]),
                   res_folder, filename)

        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            for g, sub in long.groupby("group"):
                ax.plot(sub["year"], sub["mean_cit"],
                        marker="o", linewidth=2, label=g)
            ax.set_xlabel("Year")
            ax.set_ylabel("Mean citations per document")
            ax.set_title("Mean citations per document, by concept × year")
            ax.legend(loc="best", fontsize=9, ncol=2)
            ax.grid(False)
            fig.tight_layout()
            _save_fig(fig, res_folder, filename, dpi=getattr(self, "dpi", 200))
            plt.close(fig)

        self.citations_by_group_year_df = long
        return long

    # ----------------------------------------------------------------
    # C) Open Access share per group + lift vs overall + Phi
    # ----------------------------------------------------------------
    def analyze_oa_share_by_group(
        self,
        oa_col: str = "Open Access",
        plot: bool = True,
        filename: str = "oa_share_by_group",
        figsize: Tuple[float, float] = (10, 5),
    ) -> pd.DataFrame:
        """
        Per koncept: delež OA + lift vs overall corpus + Phi korelacija
        (group_membership × OA_membership).

        OA = "any non-empty value in oa_col". Closed = empty.
        """
        if not hasattr(self, "group_matrix") or self.group_matrix is None:
            raise AttributeError("self.group_matrix manjka.")
        if oa_col not in self.df.columns:
            raise ValueError(f"Stolpec {oa_col!r} ne obstaja.")

        is_oa = self.df[oa_col].notna() & (self.df[oa_col].astype(str).str.strip() != "")
        n = len(self.df)
        n_oa_overall = int(is_oa.sum())
        oa_overall = n_oa_overall / n if n else 0.0

        gm = self.group_matrix.astype(bool)
        rows = []
        for g in gm.columns:
            in_g = gm[g]
            n_g = int(in_g.sum())
            n_oa_g = int((in_g & is_oa).sum())
            share_g = n_oa_g / n_g if n_g else 0.0
            lift = share_g / oa_overall if oa_overall else 0.0
            # Phi 2×2: rows = in_g/not, cols = oa/closed
            a = int((in_g & is_oa).sum())
            b = int((in_g & ~is_oa).sum())
            c = int((~in_g & is_oa).sum())
            d = int((~in_g & ~is_oa).sum())
            phi = _phi_2x2(a, b, c, d)
            rows.append({
                "group": g, "n_papers": n_g, "n_oa": n_oa_g,
                "oa_share_pct": round(share_g * 100, 2),
                "lift_vs_overall": round(lift, 3),
                "phi": round(phi, 3),
            })

        df_out = pd.DataFrame(rows).sort_values("oa_share_pct", ascending=False)
        df_out.attrs["overall_oa_share_pct"] = round(oa_overall * 100, 2)

        res_folder = getattr(self, "res_folder", None)
        _save_xlsx(df_out.set_index("group"), res_folder, filename)

        if plot:
            import matplotlib.pyplot as plt
            # Horizontalne palice, sortirano po OA share. Veliko bolj
            # berljivo pri >10 konceptih kot vertikalne s prelomljenimi labeli.
            df_plot = df_out.sort_values("oa_share_pct", ascending=True).reset_index(drop=True)
            n = len(df_plot)
            row_height = 0.35
            fh = max(figsize[1], row_height * n + 1.5)
            fig, ax = plt.subplots(figsize=(figsize[0], fh))

            colors = ["#3b82c4" if l >= 1 else "#c25450"
                      for l in df_plot["lift_vs_overall"]]
            y_pos = np.arange(n)
            ax.barh(y_pos, df_plot["oa_share_pct"], color=colors,
                    edgecolor="white", height=0.78)
            ax.axvline(oa_overall * 100, color="#333333",
                       linestyle="--", linewidth=1.2,
                       label=f"Overall OA share = {oa_overall*100:.1f}%")

            ax.set_yticks(y_pos)
            ax.set_yticklabels(df_plot["group"], fontsize=9)
            ax.set_xlabel("OA share (%)", fontsize=10)
            ax.set_title("Open Access share per concept "
                         "(blue ≥ overall, red < overall)", fontsize=11)
            ax.set_xlim(0, max(100.0, float(df_plot["oa_share_pct"].max()) * 1.1))

            # Share % cleanly to the right of each bar
            xmax = ax.get_xlim()[1]
            for i, (sp, lp, ph) in enumerate(zip(
                df_plot["oa_share_pct"],
                df_plot["lift_vs_overall"],
                df_plot["phi"],
            )):
                ax.text(sp + xmax * 0.005, i, f"{sp:.1f}%",
                        va="center", ha="left", fontsize=8.5,
                        color="#222222")

            # Annotate only the top-3 and bottom-3 with lift/phi
            extremes = list(range(max(0, n - 3), n)) + list(range(0, min(3, n)))
            for i in extremes:
                row = df_plot.iloc[i]
                ax.text(0.5, i, f"  ×{row['lift_vs_overall']:.2f}, "
                        f"φ={row['phi']:+.2f}",
                        va="center", ha="left", fontsize=7.5,
                        color="white", fontweight="bold")

            ax.legend(loc="lower right", fontsize=9, frameon=False)
            ax.grid(False)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            fig.tight_layout()
            _save_fig(fig, res_folder, filename, dpi=getattr(self, "dpi", 200))
            plt.close(fig)

        self.oa_share_by_group_df = df_out
        return df_out

    # ----------------------------------------------------------------
    # E) Entropija/Gini/Herfindahl entitet znotraj vsakega koncepta
    # ----------------------------------------------------------------
    def compute_group_entropy(
        self,
        entity_col: str,
        value_type: str = "list",
        sep: Optional[str] = None,
        plot: bool = True,
        filename: str = "concept_entropy",
        figsize: Tuple[float, float] = (10, 5),
    ) -> pd.DataFrame:
        """
        Shannon entropija + Gini + Herfindahl distribucije entitet znotraj
        vsakega koncepta. Zelo uporabno za primerjavo "raznolikosti" tem.

        Parametri
        ---------
        entity_col : str
            Stolpec z entitetami (npr. "Source title", "Author full names",
            "Countries of Authors").
        value_type : {"single", "list"}
            "list" = razdeli s sep; "single" = ena vrednost na zapis.
        sep : str, optional
            Ločilo za list-tip; če None, uporabi self.default_separator.
        """
        if not hasattr(self, "group_matrix") or self.group_matrix is None:
            raise AttributeError("self.group_matrix manjka.")
        if entity_col not in self.df.columns:
            raise ValueError(f"Stolpec {entity_col!r} ne obstaja.")

        if sep is None:
            sep = getattr(self, "default_separator", "; ")

        def _entities_in(rows: pd.DataFrame) -> List[str]:
            vals = rows[entity_col].dropna().astype(str)
            if value_type == "list":
                exploded = vals.str.split(sep).explode().str.strip()
                exploded = exploded[exploded != ""]
                return exploded.tolist()
            return [v.strip() for v in vals if v.strip()]

        gm = self.group_matrix.astype(bool)
        rows = []
        # Overall za primerjavo
        total = pd.Series(_entities_in(self.df)).value_counts()
        overall = {
            "group": "[OVERALL]",
            "n_unique": int(total.size),
            "n_total": int(total.sum()),
            "shannon": _shannon_entropy(total.values),
            "shannon_max": float(np.log2(total.size)) if total.size > 1 else 0.0,
            "gini": _gini(total.values),
            "herfindahl": _herfindahl(total.values),
        }
        overall["evenness"] = (
            overall["shannon"] / overall["shannon_max"] if overall["shannon_max"] > 0 else 0.0
        )
        rows.append(overall)

        for g in gm.columns:
            sub = self.df[gm[g]]
            ents = _entities_in(sub)
            if not ents:
                rows.append({"group": g, "n_unique": 0, "n_total": 0,
                             "shannon": 0.0, "shannon_max": 0.0,
                             "evenness": 0.0, "gini": 0.0, "herfindahl": 0.0})
                continue
            counts = pd.Series(ents).value_counts()
            n_unique = int(counts.size)
            shannon = _shannon_entropy(counts.values)
            shannon_max = float(np.log2(n_unique)) if n_unique > 1 else 0.0
            evenness = shannon / shannon_max if shannon_max > 0 else 0.0
            rows.append({
                "group": g,
                "n_unique": n_unique,
                "n_total": int(counts.sum()),
                "shannon": round(shannon, 3),
                "shannon_max": round(shannon_max, 3),
                "evenness": round(evenness, 3),
                "gini": round(_gini(counts.values), 3),
                "herfindahl": round(_herfindahl(counts.values), 3),
            })

        df_out = pd.DataFrame(rows)

        res_folder = getattr(self, "res_folder", None)
        _save_xlsx(df_out.set_index("group"), res_folder,
                   f"{filename}_{entity_col.replace(' ', '_').lower()}")

        if plot:
            import matplotlib.pyplot as plt
            plot_df = df_out[df_out["group"] != "[OVERALL]"].copy()
            plot_df = plot_df.sort_values("evenness", ascending=False)
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            ax1, ax2 = axes
            ax1.bar(plot_df["group"], plot_df["evenness"],
                    color="steelblue", edgecolor="white")
            ax1.set_ylim(0, 1.05)
            ax1.set_xticks(range(len(plot_df)))
            ax1.set_xticklabels(plot_df["group"], rotation=30, ha="right")
            ax1.set_ylabel("Evenness  (Shannon / log₂N_unique)")
            ax1.set_title(f"Within-concept evenness over {entity_col}")
            ax1.grid(False)

            ax2.bar(plot_df["group"], plot_df["gini"],
                    color="indianred", edgecolor="white")
            ax2.set_ylim(0, 1.05)
            ax2.set_xticks(range(len(plot_df)))
            ax2.set_xticklabels(plot_df["group"], rotation=30, ha="right")
            ax2.set_ylabel("Gini coefficient")
            ax2.set_title(f"Within-concept inequality over {entity_col}")
            ax2.grid(False)

            fig.tight_layout()
            _save_fig(fig, res_folder,
                      f"{filename}_{entity_col.replace(' ', '_').lower()}",
                      dpi=getattr(self, "dpi", 200))
            plt.close(fig)

        return df_out

    # ----------------------------------------------------------------
    # H) Concept × concept overlap v več časovnih oknih (Jaccard)
    # ----------------------------------------------------------------
    def analyze_group_overlap_over_time(
        self,
        time_windows: Optional[List[Tuple[int, int]]] = None,
        year_col: str = "Year",
        plot: bool = True,
        filename: str = "concept_overlap_over_time",
        figsize_per_window: Tuple[float, float] = (5.5, 4.5),
    ) -> dict:
        """
        Za vsako časovno okno izračuna Jaccard koncept × koncept matriko.
        Plot: small multiples (en heatmap na okno).

        Privzeto okna: (2010, 2014), (2015, 2019), (2020, 2025).
        """
        if not hasattr(self, "group_matrix") or self.group_matrix is None:
            raise AttributeError("self.group_matrix manjka.")
        if year_col not in self.df.columns:
            raise ValueError(f"Stolpec {year_col!r} ne obstaja.")

        if time_windows is None:
            time_windows = [(2010, 2014), (2015, 2019), (2020, 2025)]

        years = pd.to_numeric(self.df[year_col], errors="coerce")
        gm = self.group_matrix.astype(bool)
        groups = list(gm.columns)

        results = {}
        for lo, hi in time_windows:
            mask = (years >= lo) & (years <= hi)
            sub = gm.loc[mask].astype(int)
            M = sub.values
            inter = M.T @ M
            counts = M.sum(axis=0)
            union = counts[:, None] + counts[None, :] - inter
            with np.errstate(divide="ignore", invalid="ignore"):
                jacc = np.where(union > 0, inter / union, 0.0)
            jacc_df = pd.DataFrame(jacc, index=groups, columns=groups)
            label = f"{lo}-{hi}"
            results[label] = jacc_df

        res_folder = getattr(self, "res_folder", None)
        if res_folder is not None:
            tables_dir = os.path.join(res_folder, "tables")
            os.makedirs(tables_dir, exist_ok=True)
            with pd.ExcelWriter(os.path.join(tables_dir, f"{filename}.xlsx")) as w:
                for label, df in results.items():
                    df.round(3).to_excel(w, sheet_name=label)

        if plot:
            import matplotlib.pyplot as plt
            n_w = len(results)
            fig, axes = plt.subplots(
                1, n_w,
                figsize=(figsize_per_window[0] * n_w, figsize_per_window[1]),
                squeeze=False,
            )
            vmax = max(df.values.max() for df in results.values())
            for ax, (label, df) in zip(axes[0], results.items()):
                im = ax.imshow(df.values, vmin=0, vmax=vmax,
                               cmap="YlOrRd", aspect="auto")
                ax.set_xticks(range(len(df.columns)))
                ax.set_xticklabels(df.columns, rotation=45, ha="right",
                                   fontsize=8)
                ax.set_yticks(range(len(df.index)))
                ax.set_yticklabels(df.index, fontsize=8)
                ax.set_title(f"Jaccard, {label}")
                for i in range(df.shape[0]):
                    for j in range(df.shape[1]):
                        v = df.values[i, j]
                        if i != j and v > 0:
                            color = "white" if v > vmax * 0.5 else "black"
                            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                    fontsize=7, color=color)
            fig.suptitle("Concept overlap (Jaccard) over time windows",
                         fontsize=12, y=1.02)
            fig.tight_layout()
            _save_fig(fig, res_folder, filename, dpi=getattr(self, "dpi", 200))
            plt.close(fig)

        self.group_overlap_over_time_dict = results
        return results

    # =============================================================================
    # I. analyze_concept_pair_trends — per-pair Mann-Kendall nad Jaccard skozi okna
    # =============================================================================
    def analyze_concept_pair_trends(
        self,
        time_windows: Optional[List[Tuple[int, int]]] = None,
        metric: str = "jaccard",
        year_col: str = "Year",
        min_jaccard: float = 0.01,
        rising_alpha: float = 0.05,
        emerging_baseline_max: float = 0.01,
        emerging_recent_min: float = 0.05,
        plot: bool = True,
        filename: str = "concept_pair_trends",
        figsize: Tuple[float, float] = (12, 9),
    ) -> dict:
        """
        Per-pair (Ci, Cj) trend skozi casovna okna.

        Najprej zazene `analyze_group_overlap_over_time(time_windows=...)` da
        dobi Jaccard matrike per okno. Nato za vsak par aplicira Mann-Kendall
        tau nad zaporedjem Jaccard vrednosti skozi okna.

        Klasifikacija parov:
          - rising:    tau > 0, p < rising_alpha
          - falling:   tau < 0, p < rising_alpha
          - persistent: vse Jaccard >= min_jaccard, |tau| nizek
          - emerging:  prvi 1-2 okni pod emerging_baseline_max, zadnje okno >= emerging_recent_min
          - other:    ostali

        Outputs:
          - df_pair_evolution: vrstice = pari (i, j), stolpci = okna + tau/p/category
          - plot: heatmap top N rising/emerging parov skozi okna

        Returns
        -------
        dict
            {"pair_evolution": DataFrame, "by_category": dict, "windows": list}
        """
        from biblium.bibgroup_modules.year_trend import _kendall_tau

        if not hasattr(self, "group_matrix") or self.group_matrix is None:
            raise AttributeError("self.group_matrix manjka.")

        # Re-uporabi analyze_group_overlap_over_time za Jaccard per okno
        overlap = self.analyze_group_overlap_over_time(
            time_windows=time_windows,
            year_col=year_col,
            plot=False,
            filename=None,
        )
        windows = list(overlap.keys())
        groups = list(self.group_matrix.columns)
        n = len(groups)

        # Za vsak par (i<j) zberi seznam Jaccard vrednosti skozi okna
        rows = []
        x = np.arange(len(windows), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                ci, cj = groups[i], groups[j]
                vals = np.array([overlap[w].iloc[i, j] for w in windows], dtype=float)
                if np.all(vals < min_jaccard):
                    # vsi okni pod pragom — nezanimivo
                    continue
                tau, pval = _kendall_tau(x, vals)
                row = {"concept_A": ci, "concept_B": cj}
                for w, v in zip(windows, vals):
                    row[w] = round(float(v), 4)
                row["mean"] = round(float(vals.mean()), 4)
                row["last"] = round(float(vals[-1]), 4)
                row["delta"] = round(float(vals[-1] - vals[0]), 4)
                row["tau"] = round(float(tau), 3)
                row["p_value"] = round(float(pval), 4)

                # Klasifikacija
                baseline = vals[:max(1, len(vals) // 2)].mean()
                recent = vals[-1]
                cat = "other"
                if tau > 0 and pval < rising_alpha:
                    cat = "rising"
                elif tau < 0 and pval < rising_alpha:
                    cat = "falling"
                elif (baseline <= emerging_baseline_max) and (recent >= emerging_recent_min):
                    cat = "emerging"
                elif (vals >= min_jaccard).all() and abs(tau) < 0.3:
                    cat = "persistent"
                row["category"] = cat
                rows.append(row)

        df_evo = pd.DataFrame(rows)
        if not df_evo.empty:
            df_evo = df_evo.sort_values(["category", "last"], ascending=[True, False])

        by_cat = {c: df_evo[df_evo["category"] == c].copy()
                  for c in ("rising", "falling", "emerging", "persistent", "other")}

        # Save
        res_folder = getattr(self, "res_folder", None)
        if res_folder is not None and filename:
            tables_dir = os.path.join(res_folder, "tables")
            os.makedirs(tables_dir, exist_ok=True)
            out_xlsx = os.path.join(tables_dir, f"{filename}.xlsx")
            with pd.ExcelWriter(out_xlsx) as w:
                df_evo.to_excel(w, sheet_name="all_pairs", index=False)
                for c, d in by_cat.items():
                    if not d.empty:
                        d.to_excel(w, sheet_name=c, index=False)

        # Plot: heatmap of top N rising + emerging pairs across windows
        if plot and not df_evo.empty:
            import matplotlib.pyplot as plt

            picks = pd.concat([
                by_cat["rising"].head(8),
                by_cat["emerging"].head(8),
                by_cat["falling"].head(6),
            ])
            if picks.empty:
                picks = df_evo.head(15)

            fig, ax = plt.subplots(figsize=figsize)
            mat = picks[windows].values.astype(float)
            ylabels = [f"{r['concept_A']} ↔ {r['concept_B']} [{r['category']}]"
                       for _, r in picks.iterrows()]
            im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0,
                            vmax=max(0.05, float(mat.max())))
            ax.set_xticks(range(len(windows)))
            ax.set_xticklabels(windows, rotation=0, fontsize=9)
            ax.set_yticks(range(len(ylabels)))
            ax.set_yticklabels(ylabels, fontsize=8)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat[i, j]
                    if v > 0:
                        color = "white" if v > 0.5 * mat.max() else "black"
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                fontsize=7, color=color)
            ax.set_title("Concept-pair Jaccard over time windows "
                          "(rising / emerging / falling)", fontsize=11)
            ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="Jaccard")
            fig.tight_layout()
            _save_fig(fig, res_folder, filename, dpi=getattr(self, "dpi", 200))
            plt.close(fig)

        self.concept_pair_trends = {
            "pair_evolution": df_evo,
            "by_category": by_cat,
            "windows": windows,
        }
        return self.concept_pair_trends

