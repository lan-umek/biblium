# -*- coding: utf-8 -*-
"""
GroupFieldDynamicsMixin — analize "field-level" dinamike.

Vse so na ravni group_matrix (koncepti) ali corpus-wide z razčlenitvijo po
konceptih. Privzeto BREZ pomožnih črt v grafih (ax.grid(False)).

Vključene metode:

  4) compute_country_concept_phi      — Phi koncept × država (vse z ≥10 dok.)
  7) analyze_methods_vs_applications_ratio — letna ratio Methods/(Methods+Apps)
  5) analyze_internationalization     — % multi-country zapisov per leto
  6) analyze_publisher_share          — % zapisov per publisher pattern + citati
  8) analyze_topic_shock              — share + citati za poljuben topic (npr. COVID)
 10) compute_citation_half_life_per_group — leta do polovice kumulativnih citatov
  3) compute_software_adoption_curves — letna proporcija vsakega softver-keyworda
  9) analyze_method_domain_overlap    — entiteta kot METODA vs DOMENA
 11) compute_self_citation_share_per_group — % "self-cite" referenc (najden EID/DOI v korpusu) per group
 12) analyze_conditional_term_share  — med papirji z A: delež z/brez B skozi
                                       leta (npr. buttonology: software brez metod)

@author: Lan + biblium 2.16
"""

from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Helpers
# =============================================================================

def _save_xlsx(df: pd.DataFrame, res_folder: Optional[str], filename: str,
               sheet_name: str = "Sheet1") -> None:
    if res_folder is None or filename is None:
        return
    tables_dir = os.path.join(res_folder, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    df.to_excel(os.path.join(tables_dir, f"{filename}.xlsx"),
                sheet_name=sheet_name, index=True)


def _save_fig(fig, res_folder: Optional[str], filename: str, dpi: int = 200) -> None:
    if res_folder is None or filename is None:
        return
    plots_dir = os.path.join(res_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, f"{filename}.png"),
                dpi=dpi, bbox_inches="tight")


def _phi_2x2(a: int, b: int, c: int, d: int) -> float:
    n = a + b + c + d
    denom = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denom == 0 or n == 0:
        return 0.0
    return float((a * d - b * c) / denom)


# =============================================================================
# Mixin
# =============================================================================

class GroupFieldDynamicsMixin:
    """Razširitve za field-level dinamike na BiblioGroup (konceptih)."""

    # ----------------------------------------------------------------
    # 4) Phi koncept × država — za vse države z >= min_docs
    # ----------------------------------------------------------------
    def compute_country_concept_phi(
        self,
        country_col: str = "Countries of Authors",
        min_docs: int = 10,
        top_n_visualize: int = 25,
        plot: bool = True,
        filename: str = "country_concept_phi",
        figsize: Tuple[float, float] = (11, 9),
    ) -> Dict[str, pd.DataFrame]:
        """
        Za vsako državo (z ≥ min_docs zapisi) izračuna Phi korelacijo z vsakim
        konceptom (binarni × binarni). Vrne dict z dvema tabelama:
          full_phi  — vse države, vsi koncepti (DataFrame)
          full_n    — n_papers za vsako državo (Series)

        Plot: heatmap top_n_visualize držav po MAX(|Phi|) za berljivost.
        """
        if not hasattr(self, "group_matrix") or self.group_matrix is None:
            raise AttributeError("self.group_matrix manjka.")
        if country_col not in self.df.columns:
            raise ValueError(f"Stolpec {country_col!r} ne obstaja.")

        sep = getattr(self, "default_separator", "; ")
        gm = self.group_matrix.astype(bool)
        groups = list(gm.columns)

        # Eksplodiraj države
        ct_series = (
            self.df[country_col].fillna("").astype(str)
            .str.split(sep)
            .apply(lambda L: sorted({x.strip() for x in L if x.strip()}))
        )
        # Country counts (po dokumentih, ne po pojavnostih)
        country_doc_count: dict[str, int] = {}
        for ctrs in ct_series:
            for c in ctrs:
                country_doc_count[c] = country_doc_count.get(c, 0) + 1

        countries = sorted([c for c, n in country_doc_count.items()
                            if n >= min_docs])
        if not countries:
            raise ValueError(f"Nobena država nima >= {min_docs} dokumentov.")

        # Za vsako državo zgradi indikator
        phi_data = {g: {} for g in groups}
        n_data = {}
        for c in countries:
            in_c = ct_series.apply(lambda L: c in L)
            n_data[c] = int(in_c.sum())
            for g in groups:
                in_g = gm[g]
                a = int((in_c & in_g).sum())
                b = int((in_c & ~in_g).sum())
                cc = int((~in_c & in_g).sum())
                d = int((~in_c & ~in_g).sum())
                phi_data[g][c] = round(_phi_2x2(a, b, cc, d), 4)

        phi_df = pd.DataFrame(phi_data, index=countries)  # rows=country, cols=group
        phi_df.index.name = "Country"
        n_series = pd.Series(n_data, name="n_papers").sort_values(ascending=False)

        # Save full table
        res_folder = getattr(self, "res_folder", None)
        if res_folder is not None:
            tables_dir = os.path.join(res_folder, "tables")
            os.makedirs(tables_dir, exist_ok=True)
            with pd.ExcelWriter(os.path.join(tables_dir, f"{filename}.xlsx")) as w:
                phi_df.to_excel(w, sheet_name="phi_full")
                n_series.to_frame().to_excel(w, sheet_name="n_papers")

        if plot:
            import matplotlib.pyplot as plt
            # Top-N po max abs Phi
            score = phi_df.abs().max(axis=1)
            top = score.sort_values(ascending=False).head(top_n_visualize).index
            sub = phi_df.loc[top].reindex(top)

            fig, ax = plt.subplots(figsize=figsize)
            vmax = float(np.nanmax(np.abs(sub.values)))
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 0.05
            im = ax.imshow(sub.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           aspect="auto")
            ax.set_xticks(range(len(sub.columns)))
            ax.set_xticklabels(sub.columns, rotation=30, ha="right")
            ax.set_yticks(range(len(sub.index)))
            ax.set_yticklabels(sub.index)
            for i in range(sub.shape[0]):
                for j in range(sub.shape[1]):
                    v = sub.values[i, j]
                    if pd.notna(v) and abs(v) > vmax * 0.15:
                        col = "white" if abs(v) > vmax * 0.6 else "black"
                        ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                                fontsize=7.5, color=col)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Phi")
            ax.set_title(f"Phi(concept × country) — top {top_n_visualize} "
                         f"countries by max |Phi|, n≥{min_docs}")
            ax.grid(False)
            fig.tight_layout()
            _save_fig(fig, res_folder, filename, dpi=getattr(self, "dpi", 200))
            plt.close(fig)

        self.country_concept_phi_df = phi_df
        self.country_concept_n_series = n_series
        return {"full_phi": phi_df, "full_n": n_series}

    # ----------------------------------------------------------------
    # 7) Methods vs Applications ratio over time
    # ----------------------------------------------------------------
    def analyze_methods_vs_applications_ratio(
        self,
        methods_concept: str = "Methods",
        apps_concept: str = "Applications",
        year_col: str = "Year",
        year_range: Optional[Tuple[int, int]] = None,
        min_year_n: int = 10,
        plot: bool = True,
        filename: str = "methods_vs_applications_ratio",
        figsize: Tuple[float, float] = (12, 5),
    ) -> pd.DataFrame:
        """
        Per leto: ratio = n_methods_only / (n_methods_only + n_apps_only).
        Visok ratio = polje raziskuje SAMO sebe; nizek = polje uporablja
        bibliometrijo na drugih domenah ("vending machine").
        """
        if methods_concept not in self.group_matrix.columns:
            raise ValueError(f"Koncept {methods_concept!r} manjka v group_matrix.")
        if apps_concept not in self.group_matrix.columns:
            raise ValueError(f"Koncept {apps_concept!r} manjka v group_matrix.")
        if year_col not in self.df.columns:
            raise ValueError(f"Stolpec {year_col!r} ne obstaja.")

        years = pd.to_numeric(self.df[year_col], errors="coerce")
        valid = years.notna()
        if year_range:
            valid &= (years >= year_range[0]) & (years <= year_range[1])

        gm = self.group_matrix.astype(bool)
        is_m = gm[methods_concept] & valid
        is_a = gm[apps_concept] & valid
        is_both = is_m & is_a

        df_y = pd.DataFrame({
            "year": years[valid].astype(int),
            "methods": is_m[valid].astype(int),
            "applications": is_a[valid].astype(int),
            "both": is_both[valid].astype(int),
        })
        agg = df_y.groupby("year").sum()
        agg = agg[agg.sum(axis=1) >= min_year_n]
        agg["methods_only"] = agg["methods"] - agg["both"]
        agg["apps_only"] = agg["applications"] - agg["both"]
        denom = agg["methods_only"] + agg["apps_only"]
        agg["methods_share_of_M_or_A_only"] = (
            agg["methods_only"] / denom.replace(0, np.nan)
        ).round(4)
        agg["apps_share_of_M_or_A_only"] = (
            agg["apps_only"] / denom.replace(0, np.nan)
        ).round(4)

        res_folder = getattr(self, "res_folder", None)
        _save_xlsx(agg, res_folder, filename)

        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(agg.index, agg["methods_share_of_M_or_A_only"] * 100,
                    marker="o", linewidth=2, color="steelblue",
                    label=f"{methods_concept} only")
            ax.plot(agg.index, agg["apps_share_of_M_or_A_only"] * 100,
                    marker="s", linewidth=2, color="darkorange",
                    label=f"{apps_concept} only")
            ax.axhline(50, color="grey", linestyle=":", linewidth=1)
            ax.set_xlabel("Year")
            ax.set_ylabel(f"% of ({methods_concept}-only ∪ {apps_concept}-only)")
            ax.set_title(f"{methods_concept} vs {apps_concept} share over time "
                         f"(papers belonging to exactly one of the two)")
            ax.legend(loc="best", fontsize=10)
            ax.grid(False)
            fig.tight_layout()
            _save_fig(fig, res_folder, filename, dpi=getattr(self, "dpi", 200))
            plt.close(fig)

        return agg

    # ----------------------------------------------------------------
    # 5) Internacionalizacija skozi čas
    # ----------------------------------------------------------------
    def analyze_internationalization(
        self,
        country_col: str = "Countries of Authors",
        year_col: str = "Year",
        year_range: Optional[Tuple[int, int]] = None,
        min_year_n: int = 20,
        plot: bool = True,
        filename: str = "internationalization_over_time",
        figsize: Tuple[float, float] = (12, 5),
    ) -> pd.DataFrame:
        """
        Per leto: % zapisov z ≥2 različnimi državami avtorjev. Mednarodne
        kolaboracije so znak "trans-nacionalizacije" polja.
        """
        if country_col not in self.df.columns:
            raise ValueError(f"Stolpec {country_col!r} ne obstaja.")
        if year_col not in self.df.columns:
            raise ValueError(f"Stolpec {year_col!r} ne obstaja.")

        sep = getattr(self, "default_separator", "; ")
        years = pd.to_numeric(self.df[year_col], errors="coerce")
        n_countries = (
            self.df[country_col].fillna("").astype(str)
            .str.split(sep)
            .apply(lambda L: len({x.strip() for x in L if x.strip()}))
        )

        valid = years.notna()
        if year_range:
            valid &= (years >= year_range[0]) & (years <= year_range[1])

        df_y = pd.DataFrame({
            "year": years[valid].astype(int),
            "n_countries": n_countries[valid].astype(int),
            "is_intl": (n_countries[valid] >= 2).astype(int),
        })
        agg = df_y.groupby("year").agg(
            n_papers=("n_countries", "size"),
            n_with_country=("n_countries", lambda s: int((s >= 1).sum())),
            n_intl=("is_intl", "sum"),
            mean_countries=("n_countries", "mean"),
        )
        agg = agg[agg["n_papers"] >= min_year_n]
        agg["intl_share_pct"] = (
            agg["n_intl"] / agg["n_with_country"].replace(0, np.nan) * 100
        ).round(2)

        res_folder = getattr(self, "res_folder", None)
        _save_xlsx(agg, res_folder, filename)

        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(agg.index, agg["intl_share_pct"], marker="o",
                    linewidth=2, color="steelblue",
                    label="% multi-country papers")
            ax.set_xlabel("Year")
            ax.set_ylabel("% of papers with ≥ 2 countries")
            ax.set_title("Internationalisation of bibliometrics over time")
            ax.set_ylim(0, max(agg["intl_share_pct"].max() * 1.15, 5))
            ax.grid(False)
            fig.tight_layout()
            _save_fig(fig, res_folder, filename, dpi=getattr(self, "dpi", 200))
            plt.close(fig)

        return agg

    # ----------------------------------------------------------------
    # 6) Publisher share trend (npr. MDPI)
    # ----------------------------------------------------------------
    def analyze_publisher_share(
        self,
        patterns: Dict[str, Iterable[str]],
        publisher_col: Optional[str] = None,
        source_col: str = "Source title",
        year_col: str = "Year",
        citation_col: str = "Cited by",
        year_range: Optional[Tuple[int, int]] = None,
        min_year_n: int = 50,
        plot: bool = True,
        filename: str = "publisher_share",
        figsize: Tuple[float, float] = (12, 5),
    ) -> pd.DataFrame:
        """
        Za vsak publisher pattern (dict ime → seznam regex/podnizov), izračuna
        letni delež papirjev + povprečne citate. Pattern matchanje:
        - Najprej iščemo v publisher_col, če obstaja.
        - Sicer v source_col.
        - Ujemanje case-insensitive substring na katerem koli iz patterns.

        Primer patterns:
            {"MDPI": ["mdpi"], "Elsevier": ["elsevier"], "Springer": ["springer", "nature"]}
        """
        years = pd.to_numeric(self.df[year_col], errors="coerce")
        cits = pd.to_numeric(self.df[citation_col], errors="coerce").fillna(0)

        col = publisher_col if (publisher_col and publisher_col in self.df.columns) \
              else source_col
        text = self.df[col].fillna("").astype(str).str.lower()

        valid = years.notna()
        if year_range:
            valid &= (years >= year_range[0]) & (years <= year_range[1])

        rows = []
        for label, terms in patterns.items():
            mask = pd.Series(False, index=text.index)
            for t in terms:
                mask = mask | text.str.contains(re.escape(t.lower()), regex=True)
            mask = mask & valid
            yr = years[valid].astype(int).rename("year")
            sub = pd.DataFrame({
                "year": yr,
                "is_pub": mask[valid].astype(int),
                "cit": cits[valid].values,
            })
            agg = sub.groupby("year").agg(
                n_papers=("is_pub", "size"),
                n_pub=("is_pub", "sum"),
                mean_cit_pub=("cit", lambda s: s[sub.loc[s.index, "is_pub"] == 1].mean()),
                mean_cit_nonpub=("cit", lambda s: s[sub.loc[s.index, "is_pub"] == 0].mean()),
            )
            agg = agg[agg["n_papers"] >= min_year_n]
            agg["share_pct"] = (agg["n_pub"] / agg["n_papers"] * 100).round(2)
            agg["publisher"] = label
            rows.append(agg.reset_index())
        long = pd.concat(rows, ignore_index=True)

        res_folder = getattr(self, "res_folder", None)
        _save_xlsx(long.set_index(["publisher", "year"]),
                   res_folder, filename)

        if plot:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
            ax1, ax2 = axes
            for label, sub in long.groupby("publisher"):
                ax1.plot(sub["year"], sub["share_pct"], marker="o",
                         linewidth=2, label=label)
                ax2.plot(sub["year"], sub["mean_cit_pub"], marker="s",
                         linewidth=2, label=label)
            ax1.set_xlabel("Year"); ax1.set_ylabel("% of papers")
            ax1.set_title("Publisher share over time")
            ax1.legend(loc="best", fontsize=9); ax1.grid(False)
            ax2.set_xlabel("Year"); ax2.set_ylabel("Mean citations (publisher subset)")
            ax2.set_title("Citation impact of publisher subset")
            ax2.legend(loc="best", fontsize=9); ax2.grid(False)
            fig.tight_layout()
            _save_fig(fig, res_folder, filename, dpi=getattr(self, "dpi", 200))
            plt.close(fig)

        return long

    # ----------------------------------------------------------------
    # 8) Topic shock (npr. COVID)
    # ----------------------------------------------------------------
    def analyze_topic_shock(
        self,
        terms: Iterable[str],
        topic_label: str = "Topic",
        text_col: str = "Processed Combined Text",
        year_col: str = "Year",
        citation_col: str = "Cited by",
        year_range: Optional[Tuple[int, int]] = None,
        min_year_n: int = 50,
        plot: bool = True,
        filename: str = "topic_shock",
        figsize: Tuple[float, float] = (12, 5),
    ) -> pd.DataFrame:
        """
        Za poljuben topic (seznam regex/podnizov), izračuna letni delež +
        povprečne citate. Tipičen primer: COVID, AI.
        """
        if text_col not in self.df.columns:
            raise ValueError(f"Stolpec {text_col!r} ne obstaja.")

        text = self.df[text_col].fillna("").astype(str).str.lower()
        years = pd.to_numeric(self.df[year_col], errors="coerce")
        cits = pd.to_numeric(self.df[citation_col], errors="coerce").fillna(0)

        # token-level wildcard → regex
        def _term_to_re(t: str) -> str:
            t = t.strip().lower()
            if t.endswith("*"):
                return rf"\b{re.escape(t[:-1])}\w*"
            return rf"\b{re.escape(t)}\b"

        pattern = "|".join(_term_to_re(t) for t in terms)
        mask = text.str.contains(pattern, regex=True, na=False)

        valid = years.notna()
        if year_range:
            valid &= (years >= year_range[0]) & (years <= year_range[1])

        df_y = pd.DataFrame({
            "year": years[valid].astype(int),
            "is_topic": mask[valid].astype(int),
            "cit": cits[valid].values,
        })
        agg = df_y.groupby("year").agg(
            n_papers=("is_topic", "size"),
            n_topic=("is_topic", "sum"),
            mean_cit_topic=("cit", lambda s: s[df_y.loc[s.index, "is_topic"] == 1].mean()),
            mean_cit_nontopic=("cit", lambda s: s[df_y.loc[s.index, "is_topic"] == 0].mean()),
        )
        agg = agg[agg["n_papers"] >= min_year_n]
        agg["share_pct"] = (agg["n_topic"] / agg["n_papers"] * 100).round(2)
        agg.attrs["topic_label"] = topic_label

        res_folder = getattr(self, "res_folder", None)
        _save_xlsx(agg, res_folder, filename)

        if plot:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
            ax1, ax2 = axes
            ax1.bar(agg.index, agg["share_pct"], color="indianred",
                    edgecolor="white")
            ax1.set_xlabel("Year"); ax1.set_ylabel("% of yearly papers")
            ax1.set_title(f"'{topic_label}' share of bibliometrics output")
            ax1.grid(False)

            ax2.plot(agg.index, agg["mean_cit_topic"], marker="o",
                     linewidth=2, color="indianred", label=f"{topic_label}")
            ax2.plot(agg.index, agg["mean_cit_nontopic"], marker="s",
                     linewidth=2, color="grey", label="other")
            ax2.set_xlabel("Year"); ax2.set_ylabel("Mean citations")
            ax2.set_title(f"Mean citations: '{topic_label}' vs other")
            ax2.legend(loc="best", fontsize=9); ax2.grid(False)
            fig.tight_layout()
            _save_fig(fig, res_folder, filename, dpi=getattr(self, "dpi", 200))
            plt.close(fig)

        return agg

    # ----------------------------------------------------------------
    # 10) Citation half-life per group (eksponentni razpad)
    # ----------------------------------------------------------------
    def compute_citation_half_life_per_group(
        self,
        year_col: str = "Year",
        citation_col: str = "Cited by",
        current_year: Optional[int] = None,
        min_papers: int = 50,
        plot: bool = True,
        filename: str = "citation_half_life_per_group",
        figsize: Tuple[float, float] = (10, 5),
    ) -> pd.DataFrame:
        """
        Approx half-life: za vsak koncept fitnemo y = b * exp(-r * age) na
        povprečnih citatih per starost (current_year - year). Half-life =
        ln(2) / r. Je APPROXIMATION (citati so kumulativni, ne yearly).

        Vrne tabelo: group, n_papers, half_life_years, r, b.
        """
        if not hasattr(self, "group_matrix") or self.group_matrix is None:
            raise AttributeError("self.group_matrix manjka.")
        if year_col not in self.df.columns or citation_col not in self.df.columns:
            raise ValueError("Manjka year ali citation stolpec.")

        cy = current_year or int(pd.to_numeric(self.df[year_col],
                                               errors="coerce").max())
        years = pd.to_numeric(self.df[year_col], errors="coerce")
        cits = pd.to_numeric(self.df[citation_col], errors="coerce").fillna(0)

        gm = self.group_matrix.astype(bool)
        rows = []
        traces: dict[str, pd.Series] = {}

        for g in gm.columns:
            mask = gm[g] & years.notna()
            if mask.sum() < min_papers:
                rows.append({"group": g, "n_papers": int(mask.sum()),
                             "half_life_years": np.nan, "r": np.nan, "b": np.nan})
                continue
            sub = pd.DataFrame({
                "age": (cy - years[mask]).astype(int),
                "cit": cits[mask].values,
            })
            mean_by_age = sub.groupby("age")["cit"].mean()
            mean_by_age = mean_by_age[mean_by_age.index >= 1]
            mean_by_age = mean_by_age[mean_by_age > 0]
            if len(mean_by_age) < 5:
                rows.append({"group": g, "n_papers": int(mask.sum()),
                             "half_life_years": np.nan, "r": np.nan, "b": np.nan})
                continue
            # log-linearna regresija: log(c) = log(b) - r*age
            x = mean_by_age.index.values.astype(float)
            y = np.log(mean_by_age.values)
            r = -np.polyfit(x, y, 1)[0]
            b = float(np.exp(np.polyfit(x, y, 1)[1]))
            half = float(np.log(2) / r) if r > 0 else np.nan
            rows.append({"group": g, "n_papers": int(mask.sum()),
                         "half_life_years": round(half, 2) if not np.isnan(half) else np.nan,
                         "r": round(r, 4),
                         "b": round(b, 2)})
            traces[g] = mean_by_age

        df_out = pd.DataFrame(rows).sort_values("half_life_years",
                                                ascending=False, na_position="last")

        res_folder = getattr(self, "res_folder", None)
        _save_xlsx(df_out.set_index("group"), res_folder, filename)

        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            for g, ts in traces.items():
                ax.semilogy(ts.index, ts.values, marker="o",
                            linewidth=1.5, label=g)
            ax.set_xlabel("Years since publication")
            ax.set_ylabel("Mean citations (log scale)")
            ax.set_title("Citation accumulation by concept "
                         "(slope ≈ exponential decay rate)")
            ax.legend(loc="best", fontsize=9, ncol=2)
            ax.grid(False)
            fig.tight_layout()
            _save_fig(fig, res_folder, filename, dpi=getattr(self, "dpi", 200))
            plt.close(fig)

        return df_out

    # ----------------------------------------------------------------
    # 3) Software adoption curves
    # ----------------------------------------------------------------
    def compute_software_adoption_curves(
        self,
        software_terms: Optional[Dict[str, Iterable[str]]] = None,
        text_col: str = "Processed Combined Text",
        year_col: str = "Year",
        year_range: Optional[Tuple[int, int]] = None,
        min_year_n: int = 50,
        fit_logistic: bool = True,
        plot: bool = True,
        filename: str = "software_adoption_curves",
        figsize: Tuple[float, float] = (12, 6),
    ) -> pd.DataFrame:
        """
        Za vsak software term: letni delež zapisov, ki ga omenjajo. Logistični
        fit (S-krivulja) z napovedjo midpoint, growth rate, plateau.
        """
        if software_terms is None:
            software_terms = {
                "VOSviewer":     ["vosviewer", "vos viewer"],
                "Bibliometrix":  ["bibliometrix", "biblioshiny"],
                "CiteSpace":     ["citespace", "cite space"],
                "BibExcel":      ["bibexcel"],
                "Pajek":         ["pajek"],
                "Gephi":         ["gephi"],
                "Sci2":          ["sci2"],
                "HistCite":      ["histcite"],
            }
        if text_col not in self.df.columns:
            raise ValueError(f"Stolpec {text_col!r} ne obstaja.")

        text = self.df[text_col].fillna("").astype(str).str.lower()
        years = pd.to_numeric(self.df[year_col], errors="coerce")
        valid = years.notna()
        if year_range:
            valid &= (years >= year_range[0]) & (years <= year_range[1])

        all_years = sorted(years[valid].dropna().astype(int).unique())
        n_year = pd.Series(0, index=all_years)
        for y in all_years:
            n_year[y] = int((years[valid] == y).sum())
        n_year = n_year[n_year >= min_year_n]

        # Per software, yearly count
        rows = []
        share_df = pd.DataFrame(0.0, index=n_year.index, columns=list(software_terms.keys()))
        for sw, terms in software_terms.items():
            pat = "|".join(re.escape(t.lower()) for t in terms)
            mask = text.str.contains(pat, regex=True, na=False) & valid
            counts = years[mask].astype(int).value_counts().reindex(n_year.index).fillna(0)
            share = (counts / n_year * 100).astype(float)
            share_df[sw] = share.values

        # Logistic fit per softver
        fit_rows = []
        if fit_logistic:
            try:
                from scipy.optimize import curve_fit
                def logistic(t, L, k, t0):
                    return L / (1 + np.exp(-k * (t - t0)))

                t = np.array(n_year.index, dtype=float)
                for sw in software_terms.keys():
                    y = share_df[sw].values
                    if (y > 0).sum() < 4:
                        fit_rows.append({"software": sw, "L_pct": np.nan,
                                         "k": np.nan, "t0": np.nan})
                        continue
                    try:
                        L0 = max(float(y.max()) * 1.2, 1.0)
                        t0_0 = float(t[np.argmax(np.cumsum(y) >= y.sum() / 2)])
                        popt, _ = curve_fit(
                            logistic, t, y, p0=[L0, 0.3, t0_0], maxfev=5000
                        )
                        fit_rows.append({"software": sw,
                                         "L_pct": round(float(popt[0]), 2),
                                         "k": round(float(popt[1]), 4),
                                         "t0": round(float(popt[2]), 1)})
                    except Exception:
                        fit_rows.append({"software": sw, "L_pct": np.nan,
                                         "k": np.nan, "t0": np.nan})
            except ImportError:
                pass

        fit_df = pd.DataFrame(fit_rows) if fit_rows else None

        res_folder = getattr(self, "res_folder", None)
        if res_folder is not None:
            tables_dir = os.path.join(res_folder, "tables")
            os.makedirs(tables_dir, exist_ok=True)
            with pd.ExcelWriter(os.path.join(tables_dir, f"{filename}.xlsx")) as w:
                share_df.to_excel(w, sheet_name="share_pct_per_year")
                if fit_df is not None:
                    fit_df.to_excel(w, sheet_name="logistic_fit", index=False)

        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            for sw in software_terms.keys():
                ax.plot(share_df.index, share_df[sw], marker="o",
                        linewidth=2, label=sw)
            ax.set_xlabel("Year")
            ax.set_ylabel("% of yearly papers mentioning software")
            ax.set_title("Software adoption curves in bibliometrics literature")
            ax.legend(loc="best", fontsize=9, ncol=2)
            ax.grid(False)
            fig.tight_layout()
            _save_fig(fig, res_folder, filename, dpi=getattr(self, "dpi", 200))
            plt.close(fig)

        return share_df

    # ----------------------------------------------------------------
    # 9) Method-vs-domain razpodelitev za poljuben pojem (npr. AI)
    # ----------------------------------------------------------------
    def analyze_method_domain_overlap(
        self,
        topic_terms: Iterable[str],
        topic_label: str = "AI",
        method_concept: str = "Methods",
        application_concept: str = "Applications",
        text_col: str = "Processed Combined Text",
        year_col: str = "Year",
        year_range: Optional[Tuple[int, int]] = None,
        plot: bool = True,
        filename: str = "method_vs_domain_overlap",
        figsize: Tuple[float, float] = (10, 6),
    ) -> pd.DataFrame:
        """
        Za poljuben pojem (npr. "AI") razdeli ujetnike v 4 razrede:
          - method-only: v Methods konceptu, NE v Applications
          - domain-only: v Applications, NE v Methods
          - both: v obeh konceptih hkrati
          - neither: v nobenem od obeh konceptov

        Vrne letno tabelo + skupno vsoto. Plot: stacked area po letih.
        """
        text = self.df[text_col].fillna("").astype(str).str.lower()
        # Zgradi pattern
        def _term_to_re(t: str) -> str:
            t = t.strip().lower()
            if t.endswith("*"):
                return rf"\b{re.escape(t[:-1])}\w*"
            return rf"\b{re.escape(t)}\b"
        pattern = "|".join(_term_to_re(t) for t in topic_terms)
        is_topic = text.str.contains(pattern, regex=True, na=False)

        gm = self.group_matrix.astype(bool)
        is_m = gm[method_concept]
        is_a = gm[application_concept]

        years = pd.to_numeric(self.df[year_col], errors="coerce")
        valid = years.notna() & is_topic
        if year_range:
            valid &= (years >= year_range[0]) & (years <= year_range[1])

        cls = pd.Series("neither", index=self.df.index, dtype=object)
        cls.loc[is_m & ~is_a] = "method-only"
        cls.loc[~is_m & is_a] = "domain-only"
        cls.loc[is_m & is_a] = "both"

        df_y = pd.DataFrame({
            "year": years[valid].astype(int),
            "class": cls[valid].values,
        })
        ct = pd.crosstab(df_y["year"], df_y["class"])
        for col in ("method-only", "domain-only", "both", "neither"):
            if col not in ct.columns:
                ct[col] = 0
        ct = ct[["method-only", "domain-only", "both", "neither"]]

        res_folder = getattr(self, "res_folder", None)
        _save_xlsx(ct, res_folder, filename)

        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            cmap = {"method-only": "steelblue", "domain-only": "darkorange",
                    "both": "purple", "neither": "lightgrey"}
            ax.stackplot(ct.index, ct["method-only"], ct["domain-only"],
                         ct["both"], ct["neither"],
                         labels=["method-only", "domain-only",
                                 "both (method × domain)", "neither"],
                         colors=[cmap[c] for c in
                                 ["method-only", "domain-only", "both", "neither"]],
                         alpha=0.85)
            ax.set_xlabel("Year")
            ax.set_ylabel(f"# papers mentioning '{topic_label}'")
            ax.set_title(f"'{topic_label}' as METHOD vs DOMAIN within bibliometrics")
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(False)
            fig.tight_layout()
            _save_fig(fig, res_folder, filename, dpi=getattr(self, "dpi", 200))
            plt.close(fig)

        return ct

    # ----------------------------------------------------------------
    # 11) Self-citation share per group — približek
    # ----------------------------------------------------------------
    def compute_self_citation_share_per_group(
        self,
        eid_col: str = "EID",
        refs_col: str = "References",
        plot: bool = True,
        filename: str = "self_citation_share_per_group",
        figsize: Tuple[float, float] = (10, 5),
    ) -> pd.DataFrame:
        """
        Approx: za vsak zapis prešteje, koliko njegovih referenc se ujema z
        EID drugih zapisov v korpusu (within-corpus self-cite). Per group
        vrne povprečno število teh ujemanj in delež zapisov z >= 1.
        """
        if eid_col not in self.df.columns or refs_col not in self.df.columns:
            raise ValueError("EID ali References stolpec manjka.")

        all_eids = set(self.df[eid_col].dropna().astype(str))
        # Strip Scopus "2-s2.0-" prefix so the bare numeric id is also a hit
        # (Scopus References text typically contains the numeric EID without prefix).
        eid_short = {e.split("-")[-1] for e in all_eids if e}
        all_lookup = all_eids | {e for e in eid_short if e and e.isdigit() and len(e) >= 8}
        sep = getattr(self, "default_separator", "; ")
        # ujemanja: koliko EID-jev iz all_lookup se pojavi v References vsakega zapisa
        def _count_internal(refs):
            if not isinstance(refs, str) or not refs.strip():
                return 0
            return sum(1 for eid in all_lookup if eid and eid in refs)
        n_internal = self.df[refs_col].apply(_count_internal)

        # If Scopus EID-in-references heuristic yields zero matches, try the
        # OpenAlex reference set when available (oa_referenced_works × oa_id).
        if n_internal.sum() == 0 and {"oa_referenced_works", "oa_id"}.issubset(self.df.columns):
            oa_ids = set(self.df["oa_id"].dropna().astype(str))
            def _count_oa(refs):
                if not isinstance(refs, str) or not refs.strip():
                    return 0
                tokens = [t.strip() for t in refs.replace(";", "|").split("|") if t.strip()]
                return sum(1 for t in tokens if t in oa_ids)
            n_internal = self.df["oa_referenced_works"].apply(_count_oa)

        gm = self.group_matrix.astype(bool)
        rows = []
        for g in gm.columns:
            mask = gm[g]
            n_g = int(mask.sum())
            mean_internal = float(n_internal[mask].mean()) if n_g else 0.0
            share_with_any = float((n_internal[mask] >= 1).mean() * 100) if n_g else 0.0
            rows.append({
                "group": g, "n_papers": n_g,
                "mean_internal_refs": round(mean_internal, 2),
                "share_with_any_internal_pct": round(share_with_any, 2),
            })
        df_out = pd.DataFrame(rows).sort_values("mean_internal_refs",
                                                ascending=False)

        res_folder = getattr(self, "res_folder", None)
        _save_xlsx(df_out.set_index("group"), res_folder, filename)

        # Skip plot if all groups have zero internal references — empty plot
        # is more confusing than helpful.
        if df_out["mean_internal_refs"].sum() == 0 and \
                df_out["share_with_any_internal_pct"].sum() == 0:
            print(f"  [WARN] {filename}: no within-corpus references detected "
                  "(EID/oa_id not found in any reference list); plot skipped.")
            return df_out

        if plot:
            import matplotlib.pyplot as plt
            # Use single NAVY color (single-series bars, color-encodes-data rule)
            NAVY = "#1f3a93"
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            ax1, ax2 = axes
            ax1.bar(df_out["group"], df_out["mean_internal_refs"],
                    color=NAVY, edgecolor="white")
            ax1.set_xticklabels(df_out["group"], rotation=30, ha="right")
            ax1.set_ylabel("Mean within-corpus refs per paper")
            ax1.set_title("Field self-referentiality, mean")
            ax1.grid(False)

            ax2.bar(df_out["group"], df_out["share_with_any_internal_pct"],
                    color=NAVY, edgecolor="white")
            ax2.set_xticklabels(df_out["group"], rotation=30, ha="right")
            ax2.set_ylabel("% papers with >=1 internal ref")
            ax2.set_title("Field self-referentiality, prevalence")
            ax2.grid(False)

            fig.tight_layout()
            _save_fig(fig, res_folder, filename, dpi=getattr(self, "dpi", 200))
            plt.close(fig)

        return df_out

    # ----------------------------------------------------------------
    # 12) Conditional term share — med papirji z A, delež z/brez B skozi leta
    # ----------------------------------------------------------------
    def analyze_conditional_term_share(
        self,
        given_terms: Iterable[str],
        condition_terms: Iterable[str],
        given_label: str = "A",
        condition_label: str = "B",
        text_col: str = "Processed Combined Text",
        year_col: str = "Year",
        year_range: Optional[Tuple[int, int]] = None,
        min_year_n: int = 10,
        plot: bool = True,
        filename: str = "conditional_term_share",
        figsize: Tuple[float, float] = (12, 5),
    ) -> pd.DataFrame:
        """
        Med papirji, ki ujamejo `given_terms`, izračuna letni delež tistih, ki
        ujamejo TUDI `condition_terms`, in tistih, ki je NE.

        Klasična uporaba — "buttonology": given=software orodja,
        condition=metodološke besede. Delež brez metodološke besede = papirji,
        ki orodje uporabijo "na gumb" brez metodološke refleksije.

        Terms podpirajo wildcard `*` na koncu tokena.

        Vrne letni DataFrame: year, n_given, n_given_and_condition,
        share_with_condition_pct, share_without_condition_pct.
        """
        if text_col not in self.df.columns:
            raise ValueError(f"Stolpec {text_col!r} ne obstaja.")

        def _term_to_re(t: str) -> str:
            t = t.strip().lower()
            if t.endswith("*"):
                return rf"\b{re.escape(t[:-1])}\w*"
            return rf"\b{re.escape(t)}\b"

        text = self.df[text_col].fillna("").astype(str).str.lower()
        years = pd.to_numeric(self.df[year_col], errors="coerce")

        given_pat = "|".join(_term_to_re(t) for t in given_terms)
        cond_pat = "|".join(_term_to_re(t) for t in condition_terms)
        has_given = text.str.contains(given_pat, regex=True, na=False)
        has_cond = text.str.contains(cond_pat, regex=True, na=False)

        valid = years.notna() & has_given
        if year_range:
            valid &= (years >= year_range[0]) & (years <= year_range[1])

        df_y = pd.DataFrame({
            "year": years[valid].astype(int),
            "both": (has_cond[valid]).astype(int),
        })
        agg = df_y.groupby("year").agg(
            n_given=("both", "size"),
            n_given_and_condition=("both", "sum"),
        )
        agg = agg[agg["n_given"] >= min_year_n]
        agg["share_with_condition_pct"] = (
            agg["n_given_and_condition"] / agg["n_given"] * 100
        ).round(2)
        agg["share_without_condition_pct"] = (
            100 - agg["share_with_condition_pct"]
        ).round(2)

        res_folder = getattr(self, "res_folder", None)
        _save_xlsx(agg, res_folder, filename)

        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(agg.index, agg["share_with_condition_pct"],
                    marker="o", linewidth=2, color="steelblue",
                    label=f"with {condition_label}")
            ax.plot(agg.index, agg["share_without_condition_pct"],
                    marker="s", linewidth=2, color="indianred",
                    label=f"without {condition_label}")
            ax.axhline(50, color="grey", linestyle=":", linewidth=1)
            ax.set_xlabel("Year")
            ax.set_ylabel(f"% of papers mentioning {given_label}")
            ax.set_title(f"Among '{given_label}' papers: share with vs "
                         f"without '{condition_label}'")
            ax.legend(loc="best", fontsize=10)
            ax.grid(False)
            fig.tight_layout()
            _save_fig(fig, res_folder, filename, dpi=getattr(self, "dpi", 200))
            plt.close(fig)

        return agg

    # =============================================================================
    # plot_thematic_evolution — streamgraph: per period, n_papers per concept
    # =============================================================================
    def plot_thematic_evolution(
        self,
        time_windows: Optional[List[Tuple[int, int]]] = None,
        year_col: str = "Year",
        top_n: int = 12,
        normalize: bool = True,
        filename: str = "thematic_evolution",
        figsize: Tuple[float, float] = (12, 7),
    ) -> pd.DataFrame:
        """
        Streamgraph-style: za vsako okno izracunaj n_papers per koncept,
        narisi smooth-stacked area med okni. Pokae rast/upad konceptov.

        Parameters
        ----------
        time_windows : list of (lo, hi) tuples
            Casovna okna. Privzeto: (1979, 1999), (2000, 2009), (2010, 2017), (2018, 2026).
        top_n : int
            Stevilo top konceptov (po skupnem n) za prikaz. Ostali padejo v "Other".
        normalize : bool
            Ce True, plotaj share (%); sicer raw counts.

        Returns
        -------
        pd.DataFrame
            Okno × koncept matrika n_papers.
        """
        if not hasattr(self, "group_matrix") or self.group_matrix is None:
            raise AttributeError("self.group_matrix manjka.")

        if time_windows is None:
            time_windows = [(1979, 1999), (2000, 2009), (2010, 2017), (2018, 2026)]

        years = pd.to_numeric(self.df[year_col], errors="coerce")
        gm = self.group_matrix.astype(bool).astype(int)

        rows = []
        labels = []
        x_centers = []
        for lo, hi in time_windows:
            mask = (years >= lo) & (years <= hi)
            sub = gm.loc[mask].sum(axis=0)
            rows.append(sub)
            labels.append(f"{lo}-{hi}")
            x_centers.append((lo + hi) / 2.0)
        mat = pd.DataFrame(rows, index=labels)

        # Pick top_n concepts by total
        col_totals = mat.sum(axis=0).sort_values(ascending=False)
        top_cols = col_totals.head(top_n).index.tolist()
        rest_cols = [c for c in mat.columns if c not in top_cols]
        if rest_cols:
            mat_top = mat[top_cols].copy()
            mat_top["Other"] = mat[rest_cols].sum(axis=1)
        else:
            mat_top = mat[top_cols].copy()

        # Normalize per period
        if normalize:
            row_sums = mat_top.sum(axis=1).replace(0, 1)
            mat_plot = mat_top.div(row_sums, axis=0) * 100.0
        else:
            mat_plot = mat_top.copy()

        # Save table
        res_folder = getattr(self, "res_folder", None)
        if res_folder is not None and filename:
            tables_dir = os.path.join(res_folder, "tables")
            os.makedirs(tables_dir, exist_ok=True)
            with pd.ExcelWriter(os.path.join(tables_dir, f"{filename}.xlsx")) as w:
                mat.to_excel(w, sheet_name="counts")
                mat_plot.to_excel(w, sheet_name="plot_values")

        # Plot
        import matplotlib.pyplot as plt
        from matplotlib import cm

        fig, ax = plt.subplots(figsize=figsize)
        n_concepts = mat_plot.shape[1]
        colors = cm.tab20(np.linspace(0, 1, n_concepts))

        x = np.array(x_centers, dtype=float)
        ys = mat_plot.T.values

        ax.stackplot(x, ys, labels=mat_plot.columns, colors=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel("Time period")
        ax.set_ylabel("Share (%)" if normalize else "Number of papers")
        ax.set_title("Thematic evolution — concept share over time", fontsize=11)
        ax.grid(False)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8,
                  frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        if res_folder is not None and filename:
            plots_dir = os.path.join(res_folder, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            fig.savefig(os.path.join(plots_dir, f"{filename}.png"),
                        dpi=getattr(self, "dpi", 200), bbox_inches="tight")
        plt.close(fig)

        self.thematic_evolution_df = mat
        return mat

    # =============================================================================
    # plot_concept_strategic_diagram — Callon (centrality x density) za koncepte
    # =============================================================================
    def plot_concept_strategic_diagram(
        self,
        time_window: Optional[Tuple[int, int]] = None,
        year_col: str = "Year",
        filename: str = "concept_strategic_diagram",
        figsize: Tuple[float, float] = (10, 8),
    ) -> pd.DataFrame:
        """
        Callon strategic diagram za 26 konceptov.

        Definicije (po Callon, 1991, prilagojeno za koncepte):
          - **Centrality** (x): mean Jaccard med tem konceptom in vsemi
            ostalimi (kako povezan je z drugimi).
          - **Density** (y): self-intensity = n_papers v konceptu / total_papers
            (recimo koncentracija oz. internal cohesion proxy).

        Quadrants (relativno na median):
          - top-right    (high C, high D): MOTOR themes (central + dense)
          - top-left     (low C, high D):  NICHE themes
          - bottom-right (high C, low D):  EMERGING / BASIC themes
          - bottom-left  (low C, low D):   DECLINING / PERIPHERAL

        Parameters
        ----------
        time_window : (lo, hi), optional
            Ce dano, izracun samo na zapisih v tem oknu.

        Returns
        -------
        pd.DataFrame
            concept, centrality, density, n_papers, quadrant
        """
        if not hasattr(self, "group_matrix") or self.group_matrix is None:
            raise AttributeError("self.group_matrix manjka.")

        gm = self.group_matrix.astype(int)
        if time_window is not None:
            years = pd.to_numeric(self.df[year_col], errors="coerce")
            lo, hi = time_window
            mask = (years >= lo) & (years <= hi)
            gm = gm.loc[mask]

        if gm.empty:
            return pd.DataFrame()

        groups = list(gm.columns)
        n_total = len(gm)

        # Compute Jaccard concept x concept
        M = gm.values
        inter = M.T @ M
        counts = M.sum(axis=0)
        union = counts[:, None] + counts[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            jacc = np.where(union > 0, inter / union, 0.0)
        np.fill_diagonal(jacc, 0.0)

        # Centrality: mean Jaccard with all others
        centrality = jacc.mean(axis=1)

        # Density: share of corpus
        density = counts / max(1, n_total)

        df_out = pd.DataFrame({
            "concept": groups,
            "n_papers": counts,
            "centrality": np.round(centrality, 4),
            "density": np.round(density, 4),
        })

        # Quadrant labels relative to medians
        med_c = float(np.median(df_out["centrality"]))
        med_d = float(np.median(df_out["density"]))

        def _quadrant(row):
            hi_c = row["centrality"] >= med_c
            hi_d = row["density"] >= med_d
            if hi_c and hi_d:    return "Motor"
            if (not hi_c) and hi_d: return "Niche"
            if hi_c and (not hi_d): return "Emerging/Basic"
            return "Declining/Peripheral"
        df_out["quadrant"] = df_out.apply(_quadrant, axis=1)

        # Save
        res_folder = getattr(self, "res_folder", None)
        if res_folder is not None and filename:
            tables_dir = os.path.join(res_folder, "tables")
            os.makedirs(tables_dir, exist_ok=True)
            df_out.sort_values("centrality", ascending=False).to_excel(
                os.path.join(tables_dir, f"{filename}.xlsx"), index=False
            )

        # Plot
        import matplotlib.pyplot as plt
        try:
            from adjustText import adjust_text
            _have_adjust_text = True
        except Exception:
            _have_adjust_text = False

        fig, ax = plt.subplots(figsize=figsize)
        # Normalizirano scaling: razpni v [40, 400] pt^2, da ni
        # ogromnih krogov za en koncept in mikro-pik za drugega.
        cnt = df_out["n_papers"].astype(float).clip(lower=1)
        s_min, s_max = 40.0, 380.0
        rng = cnt.max() - cnt.min()
        if rng <= 0:
            sizes = pd.Series([(s_min + s_max) / 2.0] * len(df_out),
                              index=df_out.index)
        else:
            sizes = s_min + (cnt - cnt.min()) / rng * (s_max - s_min)

        quad_colors = {
            "Motor":                "#2ca02c",
            "Niche":                "#ff7f0e",
            "Emerging/Basic":       "#1f77b4",
            "Declining/Peripheral": "#9aa0a6",
        }
        for q, grp in df_out.groupby("quadrant"):
            ax.scatter(
                grp["centrality"], grp["density"],
                s=sizes.loc[grp.index], color=quad_colors.get(q, "#888"),
                alpha=0.78, edgecolor="white", linewidth=1.0,
                label=q, zorder=3,
            )

        # Labeli — uporabi adjustText, ce na voljo
        texts = [
            ax.text(r["centrality"], r["density"], r["concept"],
                    fontsize=8.5, va="center", ha="center", zorder=5,
                    color="#111111")
            for _, r in df_out.iterrows()
        ]
        if _have_adjust_text and texts:
            try:
                adjust_text(
                    texts, ax=ax,
                    expand_points=(1.2, 1.4),
                    expand_text=(1.05, 1.2),
                    force_points=(0.5, 0.6),
                    force_text=(0.5, 0.6),
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
                )
            except Exception:
                pass

        ax.axvline(med_c, color="#aaaaaa", lw=0.6, linestyle="--", zorder=1)
        ax.axhline(med_d, color="#aaaaaa", lw=0.6, linestyle="--", zorder=1)

        # Quadrant pripise v vogale (faint)
        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        for label, x, y, ha, va in [
            ("Motor",                x_hi, y_hi, "right", "top"),
            ("Niche",                x_lo, y_hi, "left",  "top"),
            ("Emerging/Basic",       x_hi, y_lo, "right", "bottom"),
            ("Declining/Peripheral", x_lo, y_lo, "left",  "bottom"),
        ]:
            ax.text(x, y, label, fontsize=9, color=quad_colors.get(label, "#aaa"),
                    ha=ha, va=va, alpha=0.55, fontweight="bold", zorder=2)

        ax.set_xlabel("Centrality (mean Jaccard with others)", fontsize=10)
        ax.set_ylabel("Density (share of corpus)", fontsize=10)
        title_extra = f" — {time_window[0]}-{time_window[1]}" if time_window else ""
        ax.set_title(f"Concept strategic diagram (Callon){title_extra}",
                     fontsize=12, pad=10)
        ax.grid(False)
        # Fiksne velikosti markerjev v legendi (ne podedujejo iz scatter)
        leg = ax.legend(loc="upper left", fontsize=9, frameon=False,
                        title="Quadrant", title_fontsize=9,
                        bbox_to_anchor=(0.01, 0.99))
        for handle in leg.legend_handles if hasattr(leg, "legend_handles") else leg.legendHandles:
            try:
                handle.set_sizes([60])
            except Exception:
                pass
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        if res_folder is not None and filename:
            plots_dir = os.path.join(res_folder, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            fig.savefig(os.path.join(plots_dir, f"{filename}.png"),
                        dpi=getattr(self, "dpi", 200), bbox_inches="tight")
        plt.close(fig)

        return df_out

