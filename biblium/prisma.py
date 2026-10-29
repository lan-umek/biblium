# -*- coding: utf-8 -*-
"""
biblium.prisma
==============

PRISMA 2020 flow diagram (Page et al. 2021) for systematic reviews and
bibliometric studies. New in 2.19.

The module is deliberately light: a small dataclass that holds the counts
of each PRISMA stage, arithmetic helpers that derive missing downstream
numbers, and a matplotlib renderer that draws the standard three-phase
layout (Identification / Screening / Included).

Contents
--------
``PrismaCounts``                   dataclass with the stage counts
``PrismaCounts.fill_derived``      derive missing counts from the balance equations
``PrismaCounts.to_dataframe``      Stage / Count table
``prisma_counts_from_pipeline``    build counts from typical biblium numbers
``plot_prisma``                    draw the PRISMA 2020 diagram (matplotlib)

Author: Lan Umek
Version: 2.19.0
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field, fields as dc_fields
from typing import Dict, List, Optional, Tuple

import pandas as pd

__all__ = [
    "PrismaCounts",
    "prisma_counts_from_pipeline",
    "plot_prisma",
    "PRISMA_STAGES",
]

# Order and human-readable description of every count field.
PRISMA_STAGES: Tuple[Tuple[str, str], ...] = (
    ("db_records", "Records identified from databases"),
    ("register_records", "Records identified from registers"),
    ("other_records", "Records identified from other sources"),
    ("duplicates_removed", "Duplicate records removed"),
    ("removed_automation", "Records marked as ineligible by automation tools"),
    ("removed_other", "Records removed for other reasons"),
    ("records_screened", "Records screened"),
    ("records_excluded", "Records excluded"),
    ("reports_sought", "Reports sought for retrieval"),
    ("reports_not_retrieved", "Reports not retrieved"),
    ("reports_assessed", "Reports assessed for eligibility"),
    ("reports_excluded", "Reports excluded"),
    ("studies_included", "Studies included in review"),
    ("reports_included", "Reports of included studies"),
)

# Default box texts; ``{n}`` placeholders are replaced by the counts.
DEFAULT_LABELS: Dict[str, str] = {
    "identification": "Identification",
    "screening": "Screening",
    "included": "Included",
    "identified": "Records identified from:",
    "identified_db": "Databases (n = {n})",
    "identified_registers": "Registers (n = {n})",
    "identified_other": "Other sources (n = {n})",
    "removed": "Records removed before screening:",
    "removed_duplicates": "Duplicate records removed (n = {n})",
    "removed_automation": "Records marked as ineligible by automation tools (n = {n})",
    "removed_other": "Records removed for other reasons (n = {n})",
    "screened": "Records screened (n = {n})",
    "excluded": "Records excluded (n = {n})",
    "sought": "Reports sought for retrieval (n = {n})",
    "not_retrieved": "Reports not retrieved (n = {n})",
    "assessed": "Reports assessed for eligibility (n = {n})",
    "reports_excluded": "Reports excluded (n = {n}):",
    "reason": "{reason} (n = {n})",
    "included_studies": "Studies included in review (n = {n})",
    "included_reports": "Reports of included studies (n = {n})",
}


def _fmt_n(n: Optional[int]) -> str:
    return "?" if n is None else f"{int(n):,}"


@dataclass
class PrismaCounts:
    """Counts of the PRISMA 2020 stages.

    All counts are optional; ``fill_derived`` completes the ones that follow
    from the balance equations of the flow diagram:

    * records_screened  = db + registers + other − duplicates − automation − other removals
    * reports_sought    = records_screened − records_excluded
    * reports_assessed  = reports_sought − reports_not_retrieved
    * reports_excluded  = Σ exclusion_reasons (if reasons are given)
    * studies_included  = reports_assessed − reports_excluded
    * reports_included  = studies_included (if not given)

    Each equation is also solved "backwards" for a single missing term, so
    e.g. ``records_excluded`` is derived when ``records_screened`` and
    ``reports_sought`` are both known.

    Parameters
    ----------
    exclusion_reasons : dict, optional
        Mapping reason → number of reports excluded for that reason; listed
        inside the "Reports excluded" box.
    labels : dict, optional
        Overrides for the box texts (keys as in ``DEFAULT_LABELS``).
    """

    db_records: Optional[int] = None
    register_records: Optional[int] = None
    other_records: Optional[int] = None
    duplicates_removed: Optional[int] = None
    removed_automation: Optional[int] = None
    removed_other: Optional[int] = None
    records_screened: Optional[int] = None
    records_excluded: Optional[int] = None
    reports_sought: Optional[int] = None
    reports_not_retrieved: Optional[int] = None
    reports_assessed: Optional[int] = None
    reports_excluded: Optional[int] = None
    exclusion_reasons: Optional[Dict[str, int]] = None
    studies_included: Optional[int] = None
    reports_included: Optional[int] = None
    labels: Optional[Dict[str, str]] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    def _identification_total(self) -> Optional[int]:
        parts = [self.db_records, self.register_records, self.other_records]
        if all(p is None for p in parts):
            return None
        return sum(int(p) for p in parts if p is not None)

    def _removed_total(self) -> int:
        parts = [self.duplicates_removed, self.removed_automation, self.removed_other]
        return sum(int(p) for p in parts if p is not None)

    def fill_derived(self) -> "PrismaCounts":
        """Fill missing counts that follow from the flow arithmetic (in place).

        Returns ``self`` for chaining. Already-set values are never changed.
        """
        if self.exclusion_reasons and self.reports_excluded is None:
            self.reports_excluded = int(sum(int(v) for v in self.exclusion_reasons.values()))

        def _solve(total: Optional[int], a: Optional[int], b: Optional[int]):
            """total = a + b; return (total, a, b) with one missing term solved."""
            known = sum(x is not None for x in (total, a, b))
            if known == 2:
                if total is None:
                    total = int(a) + int(b)
                elif a is None:
                    a = int(total) - int(b)
                else:
                    b = int(total) - int(a)
            return total, a, b

        changed = True
        while changed:
            changed = False
            before = self.as_dict()

            ident = self._identification_total()
            if ident is not None and self.records_screened is None:
                self.records_screened = ident - self._removed_total()
            elif ident is not None and self.records_screened is not None:
                # one missing removal term can be solved
                removals = [self.duplicates_removed, self.removed_automation, self.removed_other]
                if sum(r is None for r in removals) == 1:
                    missing = ident - self.records_screened - self._removed_total()
                    if self.duplicates_removed is None:
                        self.duplicates_removed = missing
                    elif self.removed_automation is None:
                        self.removed_automation = missing
                    else:
                        self.removed_other = missing

            (self.records_screened, self.reports_sought, self.records_excluded) = _solve(
                self.records_screened, self.reports_sought, self.records_excluded)
            (self.reports_sought, self.reports_assessed, self.reports_not_retrieved) = _solve(
                self.reports_sought, self.reports_assessed, self.reports_not_retrieved)
            (self.reports_assessed, self.studies_included, self.reports_excluded) = _solve(
                self.reports_assessed, self.studies_included, self.reports_excluded)

            if self.reports_included is None and self.studies_included is not None:
                self.reports_included = self.studies_included

            changed = self.as_dict() != before
        return self

    def as_dict(self) -> Dict[str, Optional[int]]:
        """Plain dict of the count fields (no labels)."""
        out = {f.name: getattr(self, f.name) for f in dc_fields(self)
               if f.name not in ("labels", "exclusion_reasons")}
        out["exclusion_reasons"] = dict(self.exclusion_reasons) if self.exclusion_reasons else None
        return out

    def to_dataframe(self) -> pd.DataFrame:
        """Two-column ``Stage`` / ``Count`` table (exclusion reasons as
        separate rows ``Reports excluded: <reason>``)."""
        rows: List[Dict[str, object]] = []
        for key, label in PRISMA_STAGES:
            rows.append({"Stage": label, "Count": getattr(self, key)})
            if key == "reports_excluded" and self.exclusion_reasons:
                for reason, n in self.exclusion_reasons.items():
                    rows.append({"Stage": f"Reports excluded: {reason}", "Count": n})
        df = pd.DataFrame(rows, columns=["Stage", "Count"])
        df["Count"] = pd.to_numeric(df["Count"], errors="coerce").astype("Int64")
        return df

    def validate(self) -> List[str]:
        """Return a list of arithmetic inconsistencies (empty if consistent)."""
        problems: List[str] = []
        ident = self._identification_total()
        if ident is not None and self.records_screened is not None:
            if ident - self._removed_total() != self.records_screened:
                problems.append("identified − removed ≠ records_screened")
        for tot, a, b, name in (
            (self.records_screened, self.reports_sought, self.records_excluded, "records_screened"),
            (self.reports_sought, self.reports_assessed, self.reports_not_retrieved, "reports_sought"),
            (self.reports_assessed, self.studies_included, self.reports_excluded, "reports_assessed"),
        ):
            if None not in (tot, a, b) and int(tot) != int(a) + int(b):
                problems.append(f"{name} ≠ sum of its downstream parts")
        if self.exclusion_reasons and self.reports_excluded is not None:
            if sum(self.exclusion_reasons.values()) != self.reports_excluded:
                problems.append("Σ exclusion_reasons ≠ reports_excluded")
        return problems


def prisma_counts_from_pipeline(
    n_identified: int,
    n_after_dedup: int,
    n_final: int,
    n_screened_out: Optional[int] = None,
    n_register_records: Optional[int] = None,
    n_other_records: Optional[int] = None,
    n_removed_automation: Optional[int] = None,
    n_removed_other: Optional[int] = None,
    n_not_retrieved: int = 0,
    exclusion_reasons: Optional[Dict[str, int]] = None,
    labels: Optional[Dict[str, str]] = None,
) -> PrismaCounts:
    """Build ``PrismaCounts`` from the numbers a typical biblium pipeline yields.

    Parameters
    ----------
    n_identified : int
        Records retrieved from the database(s) (``db_records``).
    n_after_dedup : int
        Records left after de-duplication (and automation / other removals,
        if given) = records screened.
    n_final : int
        Documents in the final corpus (``studies_included``).
    n_screened_out : int, optional
        Records excluded at title/abstract screening. If omitted, every
        record removed between ``n_after_dedup`` and ``n_final`` is counted
        as excluded at screening, i.e. no full-text stage took place.
    n_register_records, n_other_records : int, optional
        Additional identification sources (added to the identification total).
    n_removed_automation, n_removed_other : int, optional
        Removals before screening besides duplicates; ``duplicates_removed``
        is derived as the remainder.
    n_not_retrieved : int
        Reports that could not be retrieved (default 0).
    exclusion_reasons : dict, optional
        Reasons for full-text exclusion (sum must equal reports excluded).
    """
    ident_total = int(n_identified) + int(n_register_records or 0) + int(n_other_records or 0)
    dup = ident_total - int(n_after_dedup) - int(n_removed_automation or 0) - int(n_removed_other or 0)
    c = PrismaCounts(
        db_records=int(n_identified),
        register_records=n_register_records,
        other_records=n_other_records,
        duplicates_removed=dup,
        removed_automation=n_removed_automation,
        removed_other=n_removed_other,
        records_screened=int(n_after_dedup),
        reports_not_retrieved=int(n_not_retrieved),
        studies_included=int(n_final),
        reports_included=int(n_final),
        exclusion_reasons=dict(exclusion_reasons) if exclusion_reasons else None,
        labels=labels,
    )
    if n_screened_out is not None:
        c.records_excluded = int(n_screened_out)
    else:
        reasons_sum = sum(exclusion_reasons.values()) if exclusion_reasons else 0
        c.reports_excluded = int(reasons_sum)
        c.reports_assessed = int(n_final) + int(reasons_sum)
        c.reports_sought = c.reports_assessed + int(n_not_retrieved)
        c.records_excluded = int(n_after_dedup) - c.reports_sought
    return c.fill_derived()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.fill(line, width) for line in text.split("\n"))


def _box(ax, x, y, w, h, text, *, box_color, edge_color, font_size, wrap, ha="center", bold=False):
    """Draw a rounded box centred at (x, y) with wrapped text; returns (x, y, w, h)."""
    from matplotlib.patches import FancyBboxPatch
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.012",
        linewidth=1.2, edgecolor=edge_color, facecolor=box_color, zorder=2,
    )
    ax.add_patch(patch)
    tx = x if ha == "center" else x - w / 2 + 0.015
    ax.text(tx, y, _wrap(text, wrap), ha=ha, va="center", fontsize=font_size,
            zorder=3, linespacing=1.3, fontweight="bold" if bold else "normal")
    return x, y, w, h


def _arrow(ax, p_from, p_to, color):
    ax.annotate("", xy=p_to, xytext=p_from,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                                shrinkA=0, shrinkB=0, mutation_scale=12), zorder=1)


def plot_prisma(
    counts: PrismaCounts,
    title: Optional[str] = None,
    ax=None,
    f_name: Optional[str] = None,
    dpi: int = 300,
    show: bool = False,
    box_color: str = "#eef3fb",
    edge_color: str = "#1f3b73",
    font_size: float = 9,
    phase_color: str = "#d9e2f3",
    fill_derived: bool = True,
):
    """Draw a PRISMA 2020 flow diagram.

    Layout: three phase labels on the left (Identification / Screening /
    Included), the main column of boxes connected by downward arrows and a
    right-hand column with the records/reports removed at each step.
    Exclusion reasons (``counts.exclusion_reasons``) are listed inside the
    "Reports excluded" box. Boxes are placed on a fixed grid in axes
    coordinates so they never overlap.

    Parameters
    ----------
    counts : PrismaCounts
    fill_derived : bool
        Run ``counts.fill_derived()`` on a copy before drawing (default).

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt
    from dataclasses import replace

    c = replace(counts)
    if fill_derived:
        c.fill_derived()
    L = dict(DEFAULT_LABELS)
    if c.labels:
        L.update(c.labels)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 12))
    else:
        fig = ax.figure
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    # grid --------------------------------------------------------------
    x_phase, w_phase = 0.055, 0.05
    x_main, w_main = 0.36, 0.40
    x_right, w_right = 0.80, 0.34
    y_ident = 0.86
    y_screen, y_sought, y_assess = 0.62, 0.48, 0.33
    y_incl = 0.11
    h_std = 0.075
    wrap_main, wrap_right = 44, 38
    kw = dict(box_color=box_color, edge_color=edge_color, font_size=font_size)

    # identification -----------------------------------------------------
    id_lines = [L["identified"]]
    if c.db_records is not None or (c.register_records is None and c.other_records is None):
        id_lines.append("  " + L["identified_db"].format(n=_fmt_n(c.db_records)))
    if c.register_records is not None:
        id_lines.append("  " + L["identified_registers"].format(n=_fmt_n(c.register_records)))
    if c.other_records is not None:
        id_lines.append("  " + L["identified_other"].format(n=_fmt_n(c.other_records)))
    h_ident = 0.035 + 0.022 * len(id_lines)
    _box(ax, x_main, y_ident, w_main, h_ident, "\n".join(id_lines), wrap=wrap_main, ha="left", **kw)

    rm_lines = [L["removed"],
                "  " + L["removed_duplicates"].format(n=_fmt_n(c.duplicates_removed))]
    if c.removed_automation is not None:
        rm_lines.append("  " + L["removed_automation"].format(n=_fmt_n(c.removed_automation)))
    if c.removed_other is not None:
        rm_lines.append("  " + L["removed_other"].format(n=_fmt_n(c.removed_other)))
    rm_text = "\n".join(rm_lines)
    n_rm_lines = _wrap(rm_text, wrap_right).count("\n") + 1
    h_rm = max(h_ident, 0.03 + 0.022 * n_rm_lines)
    _box(ax, x_right, y_ident, w_right, h_rm, rm_text, wrap=wrap_right, ha="left", **kw)
    _arrow(ax, (x_main + w_main / 2, y_ident), (x_right - w_right / 2, y_ident), edge_color)

    # screening ----------------------------------------------------------
    _box(ax, x_main, y_screen, w_main, h_std, L["screened"].format(n=_fmt_n(c.records_screened)), wrap=wrap_main, **kw)
    _box(ax, x_right, y_screen, w_right, h_std, L["excluded"].format(n=_fmt_n(c.records_excluded)), wrap=wrap_right, **kw)
    _arrow(ax, (x_main + w_main / 2, y_screen), (x_right - w_right / 2, y_screen), edge_color)

    _box(ax, x_main, y_sought, w_main, h_std, L["sought"].format(n=_fmt_n(c.reports_sought)), wrap=wrap_main, **kw)
    _box(ax, x_right, y_sought, w_right, h_std, L["not_retrieved"].format(n=_fmt_n(c.reports_not_retrieved)), wrap=wrap_right, **kw)
    _arrow(ax, (x_main + w_main / 2, y_sought), (x_right - w_right / 2, y_sought), edge_color)

    _box(ax, x_main, y_assess, w_main, h_std, L["assessed"].format(n=_fmt_n(c.reports_assessed)), wrap=wrap_main, **kw)
    ex_lines = [L["reports_excluded"].format(n=_fmt_n(c.reports_excluded))]
    if c.exclusion_reasons:
        for reason, n in c.exclusion_reasons.items():
            ex_lines.append("  " + L["reason"].format(reason=reason, n=_fmt_n(n)))
    ex_text = "\n".join(ex_lines)
    n_ex_lines = _wrap(ex_text, wrap_right).count("\n") + 1
    h_ex = max(h_std, 0.03 + 0.022 * n_ex_lines)
    # keep the exclusion box inside the screening band (above the included box)
    y_ex_center = y_assess
    y_ex_bottom_limit = y_incl + 0.06 + 0.02
    if y_ex_center - h_ex / 2 < y_ex_bottom_limit:
        y_ex_center = y_ex_bottom_limit + h_ex / 2
    _box(ax, x_right, y_ex_center, w_right, h_ex, ex_text, wrap=wrap_right,
         ha="left" if c.exclusion_reasons else "center", **kw)
    _arrow(ax, (x_main + w_main / 2, y_assess), (x_right - w_right / 2, y_assess), edge_color)

    # included -----------------------------------------------------------
    inc_text = (L["included_studies"].format(n=_fmt_n(c.studies_included)) + "\n"
                + L["included_reports"].format(n=_fmt_n(c.reports_included)))
    h_inc = 0.085
    _box(ax, x_main, y_incl, w_main, h_inc, inc_text, wrap=wrap_main, **kw)

    # vertical arrows between main boxes
    _arrow(ax, (x_main, y_ident - h_ident / 2), (x_main, y_screen + h_std / 2), edge_color)
    _arrow(ax, (x_main, y_screen - h_std / 2), (x_main, y_sought + h_std / 2), edge_color)
    _arrow(ax, (x_main, y_sought - h_std / 2), (x_main, y_assess + h_std / 2), edge_color)
    _arrow(ax, (x_main, y_assess - h_std / 2), (x_main, y_incl + h_inc / 2), edge_color)

    # phase labels (rotated, left) --------------------------------------
    top_ident = y_ident + max(h_ident, h_rm) / 2 + 0.01
    bot_ident = y_ident - max(h_ident, h_rm) / 2 - 0.01
    top_screen = y_screen + h_std / 2 + 0.01
    bot_screen = y_assess - max(h_std, h_ex) / 2 - 0.01
    top_incl = y_incl + h_inc / 2 + 0.01
    bot_incl = y_incl - h_inc / 2 - 0.01
    for (top, bot, key) in ((top_ident, bot_ident, "identification"),
                            (top_screen, bot_screen, "screening"),
                            (top_incl, bot_incl, "included")):
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch((x_phase - w_phase / 2, bot), w_phase, top - bot,
                                    boxstyle="round,pad=0.0,rounding_size=0.01",
                                    linewidth=1.0, edgecolor=edge_color, facecolor=phase_color, zorder=2))
        ax.text(x_phase, (top + bot) / 2, L[key], rotation=90, ha="center", va="center",
                fontsize=font_size + 1, fontweight="bold", zorder=3)

    if title:
        ax.set_title(title)
    fig.tight_layout()
    if f_name:
        fig.savefig(f_name, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax
