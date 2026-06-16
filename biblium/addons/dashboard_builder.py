# -*- coding: utf-8 -*-
"""
dashboard_builder.py - self-contained HTML report builder
==========================================================

Generates a single-file HTML dashboard from a project's ``results/``
folder. All images are base64-embedded, all CSS/JS is inline (only
DataTables + Plotly are CDN), so the resulting file is portable -
e-mail it, drop it on a USB stick, open it in any modern browser.

Conventions (memory rules):
  * no pie / donut (none here)
  * single-series bar = NAVY (handled by upstream plots)
  * labels never truncated (we use textwrap.fill where needed)
  * no decorative grid

Public entry point:
    build_pipeline_report(project_folder, out_html, ...)

Helpers:
    embed_image_b64(image_path) -> str
    dataframe_to_html(df, table_id, ...) -> str
    make_landscape_plotly(coords, labels, hover_text, ...) -> str
    scan_results_folder(results_root) -> dict

Reviewer feedback widget
------------------------
Each figure, table and the Plotly landscape gets a small inline
feedback control with three radios (Exclude / Include / Modify) plus
an on-demand textarea. No radio is checked by default -- items with
no choice are treated as "no opinion" and omitted from the output.
Choices are persisted to ``localStorage``, a sticky bottom-right
button collects them into a Markdown report (grouped by category:
Include / Modify / Exclude) and opens a ``mailto:`` to the configured
reviewer recipient.
"""

from __future__ import annotations

import base64
import datetime as _dt
import html as _html
import json as _json
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

# -- Constants ---------------------------------------------------------------

NAVY = "#1f3a68"
TEAL = "#2a9d8f"
ACCENT = "#e76f51"

# Submit button uses the brand NAVY requested in user memory (#1F3864).
FEEDBACK_NAVY = "#1F3864"

# Default reviewer recipient (overridable via build_pipeline_report kwarg).
DEFAULT_FEEDBACK_EMAIL = "lan.umek@fu.uni-lj.si"

_DEFAULT_TABLE_PRIORITY = (
    "main info",
    "summary",
    "top",
    "stats",
    "counts",
)


# =============================================================================
# 1) low-level helpers
# =============================================================================

def embed_image_b64(image_path: str | Path) -> str:
    """Read PNG/JPG file, return ``data:image/...;base64,...`` data URI."""
    p = Path(image_path)
    ext = p.suffix.lower().lstrip(".")
    if ext in {"jpg", "jpeg"}:
        mime = "image/jpeg"
    elif ext == "svg":
        mime = "image/svg+xml"
    elif ext == "gif":
        mime = "image/gif"
    else:
        mime = "image/png"
    with open(p, "rb") as f:
        b = f.read()
    return f"data:{mime};base64,{base64.b64encode(b).decode('ascii')}"


def _format_section_title(folder_name: str) -> str:
    """``21_knowledge_flows`` -> ``21 Knowledge Flows``."""
    s = str(folder_name).strip()
    m = re.match(r"^(\d+)[_\- ]+(.+)$", s)
    if m:
        num, rest = m.group(1), m.group(2)
        rest = rest.replace("_", " ").replace("-", " ")
        return f"{num} {rest.title()}"
    return s.replace("_", " ").title()


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", str(name)).strip("-").lower()
    return s or "section"


def _wrap_label(text: str, width: int = 60) -> str:
    """Memory rule: no truncated labels. Wrap with textwrap.fill."""
    if text is None:
        return ""
    return textwrap.fill(str(text), width=width, break_long_words=False,
                         break_on_hyphens=False)


# =============================================================================
# 2) DataFrame -> HTML table
# =============================================================================

def dataframe_to_html(
    df: pd.DataFrame,
    table_id: str,
    max_rows: int = 50,
    sortable: bool = True,
    caption: Optional[str] = None,
) -> str:
    """Render a DataFrame as a DataTables-ready HTML table.

    If the frame has more than ``max_rows``, we show the first ``max_rows``
    and emit a "showing N of M" note (no elipsis truncation - memory rule).
    Float columns are rounded to 2 decimals to keep tables readable
    (memory rule: numeric tables use 2-decimal precision).
    """
    if df is None or len(df) == 0:
        return '<p class="muted">Empty table.</p>'

    n_total = len(df)
    truncated = n_total > max_rows
    show_df = df.head(max_rows) if truncated else df

    # Round float columns to 2 decimals without disturbing int/bool/object cols.
    try:
        show_df = show_df.copy()
        float_cols = [
            c for c in show_df.columns
            if pd.api.types.is_float_dtype(show_df[c])
        ]
        for c in float_cols:
            show_df[c] = show_df[c].round(2)
    except Exception:  # noqa: BLE001
        pass

    classes = "display compact bib-table"
    if sortable:
        classes += " sortable"
    table_html = show_df.to_html(
        table_id=table_id,
        classes=classes,
        index=False,
        border=0,
        na_rep="",
        escape=True,
        float_format=lambda x: f"{x:.2f}",
    )

    parts: List[str] = []
    if caption:
        parts.append(f'<p class="table-caption">{_html.escape(caption)}</p>')
    if truncated:
        parts.append(
            f'<p class="muted"><em>(showing first {max_rows:,} of '
            f'{n_total:,} rows; download xlsx for full data)</em></p>'
        )
    parts.append(table_html)
    return "\n".join(parts)


def _safe_excel_to_html(
    xlsx_path: str | Path,
    max_rows: int = 50,
    table_id: Optional[str] = None,
) -> Tuple[str, int]:
    """Read an xlsx, return ``(html_string, n_total_rows)``.

    Skips empty sheets. If multiple non-empty sheets exist, concatenates
    them with a per-sheet header.
    """
    p = Path(xlsx_path)
    tid = table_id or _slugify(p.stem)
    try:
        sheets = pd.read_excel(p, sheet_name=None)
    except Exception as e:  # noqa: BLE001
        return (f'<p class="muted">Could not read {p.name}: {e}</p>', 0)

    if not sheets:
        return ('<p class="muted">No sheets.</p>', 0)

    nonempty = [(name, sdf) for name, sdf in sheets.items()
                if sdf is not None and len(sdf) > 0]
    if not nonempty:
        return (f'<p class="muted">{p.name}: all sheets empty.</p>', 0)

    if len(nonempty) == 1:
        name, sdf = nonempty[0]
        html_str = dataframe_to_html(
            sdf, table_id=tid, max_rows=max_rows,
        )
        return (html_str, len(sdf))

    parts: List[str] = []
    total_rows = 0
    for i, (name, sdf) in enumerate(nonempty):
        sub_tid = f"{tid}-s{i}"
        parts.append(f'<h4 class="sheet-header">Sheet: {_html.escape(str(name))}</h4>')
        parts.append(dataframe_to_html(
            sdf, table_id=sub_tid, max_rows=max_rows,
        ))
        total_rows += len(sdf)
    return ("\n".join(parts), total_rows)


def _pick_priority_tables(
    table_files: List[Path],
    n_pick: int = 2,
) -> List[Path]:
    """Heuristic: rank tables that look summary-ish first."""
    def score(p: Path) -> Tuple[int, int]:
        name = p.stem.lower()
        for i, key in enumerate(_DEFAULT_TABLE_PRIORITY):
            if key in name:
                return (0, i)
        if any(k in name for k in ("authors counts", "all countries counts",
                                   "index keywords counts", "author keywords counts")):
            return (2, 0)
        return (1, 0)

    ranked = sorted(table_files, key=score)
    return ranked[:n_pick]


# =============================================================================
# 3) Plotly landscape (interactive)
# =============================================================================

def make_landscape_plotly(
    coords: pd.DataFrame,
    labels: Sequence[Any],
    hover_text: Sequence[str],
    sizes: Optional[Sequence[float]] = None,
    title: str = "Paper landscape",
    colorscale: str = "tab20",  # unused, kept for API
    x_col: str = "x",
    y_col: str = "y",
    div_id: str = "landscape-plot",
) -> str:
    """Build a Plotly scatter HTML chunk for the paper landscape.

    Returns an HTML string (no <html>/<head>), including a CDN reference
    to plotly.js.
    """
    try:
        import plotly.express as px
    except Exception:
        return ('<p class="muted">Plotly not installed; install '
                '<code>plotly</code> to enable interactive landscape.</p>')

    df_plot = pd.DataFrame({
        "x": pd.to_numeric(coords[x_col], errors="coerce"),
        "y": pd.to_numeric(coords[y_col], errors="coerce"),
        "cluster": [str(v) for v in labels],
        "hover": [str(v) for v in hover_text],
    })
    df_plot = df_plot.dropna(subset=["x", "y"]).reset_index(drop=True)

    if sizes is not None and len(sizes) == len(df_plot):
        df_plot["size"] = list(sizes)
    else:
        df_plot["size"] = 5

    cluster_counts = df_plot["cluster"].value_counts()
    ordered = cluster_counts.index.tolist()
    df_plot["cluster"] = pd.Categorical(df_plot["cluster"], categories=ordered)

    fig = px.scatter(
        df_plot,
        x="x", y="y",
        color="cluster",
        hover_name="hover",
        size="size",
        size_max=8,
        title=title,
        opacity=0.75,
        height=650,
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.update_layout(
        template="simple_white",
        legend_title_text="Cluster",
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_title=None,
        yaxis_title=None,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

    html_chunk = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        div_id=div_id,
        config={"displaylogo": False, "responsive": True},
    )
    return html_chunk


# =============================================================================
# 4) scan results
# =============================================================================

def scan_results_folder(results_root: str | Path) -> Dict[str, Dict[str, Any]]:
    """Walk ``results/*/`` and collect plots + tables per section.

    Returns dict keyed by section folder name, with keys:
      ``plots``  - list of .png paths (sorted)
      ``tables`` - list of .xlsx paths (sorted)
      ``root``   - section root Path
    """
    root = Path(results_root)
    out: Dict[str, Dict[str, Any]] = {}
    if not root.exists():
        return out
    for section_dir in sorted(root.iterdir()):
        if not section_dir.is_dir():
            continue
        name = section_dir.name
        if name.startswith("."):
            continue

        plots: List[Path] = []
        tables: List[Path] = []

        plots_dir = section_dir / "plots"
        if plots_dir.exists():
            plots.extend(sorted(plots_dir.glob("*.png")))
        figs_dir = section_dir / "figures"
        if figs_dir.exists():
            plots.extend(sorted(figs_dir.glob("*.png")))

        tables_dir = section_dir / "tables"
        if tables_dir.exists():
            tables.extend(sorted(tables_dir.glob("*.xlsx")))
        for x in sorted(section_dir.glob("*.xlsx")):
            tables.append(x)

        if plots or tables:
            out[name] = {
                "plots": plots,
                "tables": tables,
                "root": section_dir,
            }
    return out


# =============================================================================
# 5) Feedback widget (per figure + table) + reviewer panel + submit button
# =============================================================================

def _feedback_box_html(
    slug: str,
    caption: str,
    section_title: str,
) -> str:
    """Render the per-item feedback control (figures and tables).

    Layout: a single tidy row of three radios; the textarea below is hidden
    until "Modify" is selected. Wrapping div carries data attributes used
    by the JS collector. No radio is checked by default -- if the
    reviewer makes no choice the item is treated as "no opinion" and
    omitted from the Markdown output.
    """
    cap_attr = _html.escape(caption, quote=True)
    sec_attr = _html.escape(section_title, quote=True)
    # Radio name MUST be unique per item so the browser does not group
    # them across the page.
    name = f"fb-{slug}"
    return (
        f'<div class="feedback-box" data-feedback-item="{slug}" '
        f'data-caption="{cap_attr}" data-section="{sec_attr}">'
        '<div class="fb-row">'
        '<span class="fb-label">Review:</span>'
        f'<label class="fb-opt fb-exclude"><input type="radio" name="{name}" '
        'value="exclude"> Exclude</label>'
        f'<label class="fb-opt fb-include"><input type="radio" name="{name}" '
        'value="include"> Include</label>'
        f'<label class="fb-opt fb-modify"><input type="radio" name="{name}" '
        'value="modify"> Modify</label>'
        '</div>'
        '<div class="fb-modify-wrap">'
        '<label class="fb-textarea-label" '
        f'for="fb-text-{slug}">What would you change?</label>'
        f'<textarea class="fb-textarea" id="fb-text-{slug}" '
        'rows="3" placeholder="Describe the change you would like..."></textarea>'
        '</div>'
        '</div>'
    )


_FEEDBACK_CSS = r"""
/* Reviewer panel */
.reviewer-panel {
  background: #fff; border: 1px solid #d6deee; border-radius: 8px;
  padding: 14px 16px; margin: 0 0 22px;
  display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
  box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
.reviewer-panel .rp-title {
  font-weight: 600; color: #1F3864; font-size: 13px;
  text-transform: uppercase; letter-spacing: .06em;
  margin-right: 6px;
}
.reviewer-panel label {
  display: inline-flex; flex-direction: column; font-size: 11px;
  color: #666; gap: 2px;
}
.reviewer-panel input[type="text"],
.reviewer-panel input[type="email"] {
  font-size: 13px; padding: 5px 8px; border: 1px solid #c7cfdd;
  border-radius: 4px; min-width: 200px; background: #fafbfd;
  color: #1c1c1c;
}
.reviewer-panel input:focus { outline: 2px solid #1F3864; outline-offset: -1px; }
.reviewer-panel .rp-progress {
  margin-left: auto; font-size: 13px; color: #1F3864; font-weight: 600;
}

/* Per-item feedback box (figures + tables) */
.feedback-box {
  margin-top: 10px; padding: 8px 12px;
  background: #f5f7fb; border: 1px solid #dbe2ef; border-radius: 6px;
  font-size: 13px;
}
.feedback-box .fb-row {
  display: flex; align-items: center; flex-wrap: wrap; gap: 14px;
}
.feedback-box .fb-label {
  color: #4a5670; font-weight: 600; font-size: 12px;
  text-transform: uppercase; letter-spacing: .04em;
}
.feedback-box .fb-opt {
  display: inline-flex; align-items: center; gap: 5px;
  cursor: pointer; user-select: none; color: #1c1c1c;
  padding: 2px 6px; border-radius: 4px;
}
.feedback-box .fb-opt input { margin: 0; cursor: pointer; }
.feedback-box .fb-opt:hover { background: rgba(31,56,100,.07); }
.feedback-box.status-include  .fb-include  { background: rgba(42,157,143,.18); }
.feedback-box.status-modify   .fb-modify   { background: rgba(231,111,81,.18); }
.feedback-box.status-exclude  .fb-exclude  { background: rgba(120,120,120,.14); }

.feedback-box .fb-modify-wrap { display: none; margin-top: 8px; }
.feedback-box.modify .fb-modify-wrap { display: block; }
.feedback-box .fb-textarea-label {
  font-size: 11px; color: #555; display: block; margin-bottom: 3px;
}
.feedback-box .fb-textarea {
  width: 100%; min-height: 60px; padding: 6px 8px;
  border: 1px solid #c7cfdd; border-radius: 4px;
  font-family: inherit; font-size: 13px; resize: vertical;
  background: #fff; color: #1c1c1c;
}
.feedback-box .fb-textarea:focus { outline: 2px solid #1F3864; outline-offset: -1px; }

/* Floating submit button */
.fb-submit {
  position: fixed; right: 22px; bottom: 22px; z-index: 9999;
  background: #1F3864; color: #fff; border: none;
  padding: 12px 20px; border-radius: 6px;
  font-size: 14px; font-weight: 600; cursor: pointer;
  box-shadow: 0 4px 14px rgba(31,56,100,.35);
  font-family: inherit;
}
.fb-submit:hover { background: #16294a; }
.fb-submit:active { transform: translateY(1px); }
.fb-submit[disabled] { background: #6b7a96; cursor: default; box-shadow: none; }

/* Toast */
.fb-toast {
  position: fixed; right: 22px; bottom: 78px; z-index: 10000;
  background: #1F3864; color: #fff;
  padding: 10px 16px; border-radius: 6px;
  font-size: 13px; max-width: 340px;
  box-shadow: 0 4px 14px rgba(0,0,0,.25);
  opacity: 0; transform: translateY(8px);
  transition: opacity .2s ease, transform .2s ease;
  pointer-events: none;
}
.fb-toast.show { opacity: 1; transform: translateY(0); }
"""


def _feedback_js(
    feedback_email: str,
    project_title: str,
) -> str:
    """Return a <script> block implementing persistence, progress and submit.

    The JS is wholly self-contained (no jQuery required) so it works
    regardless of when DataTables initialises.
    """
    email_js = _json.dumps(feedback_email)
    title_js = _json.dumps(project_title)
    # Triple-quoted Python string -- braces inside the JS are literal.
    return r"""
<script>
(function () {
  var FEEDBACK_EMAIL = """ + email_js + r""";
  var PROJECT_TITLE  = """ + title_js + r""";
  var STORAGE_PREFIX = "dashboard_feedback_";
  var REVIEWER_NAME_KEY = "dashboard_feedback__reviewer_name";
  var REVIEWER_MAIL_KEY = "dashboard_feedback__reviewer_email";

  function slugify(s) {
    return String(s || "").toLowerCase()
      .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "report";
  }

  function safeGet(k) {
    try { return window.localStorage.getItem(k); } catch (e) { return null; }
  }
  function safeSet(k, v) {
    try { window.localStorage.setItem(k, v); } catch (e) {}
  }

  function readBox(box) {
    var slug = box.getAttribute("data-feedback-item");
    var radios = box.querySelectorAll('input[type="radio"]');
    // status === "" means no radio selected = "no opinion"
    var status = "";
    for (var i = 0; i < radios.length; i++) {
      if (radios[i].checked) { status = radios[i].value; break; }
    }
    var ta = box.querySelector("textarea");
    var text = ta ? ta.value : "";
    return {
      slug: slug,
      caption: box.getAttribute("data-caption") || slug,
      section: box.getAttribute("data-section") || "",
      status: status,
      revisions: text
    };
  }

  function applyClasses(box, status) {
    box.classList.remove("status-include", "status-modify",
                         "status-exclude", "modify");
    if (!status) return;
    box.classList.add("status-" + status);
    if (status === "modify") box.classList.add("modify");
  }

  function persistBox(box) {
    var d = readBox(box);
    var payload = { status: d.status, revisions: d.revisions };
    safeSet(STORAGE_PREFIX + d.slug, JSON.stringify(payload));
  }

  function restoreBox(box) {
    var slug = box.getAttribute("data-feedback-item");
    var raw = safeGet(STORAGE_PREFIX + slug);
    if (!raw) {
      applyClasses(box, "");
      return;
    }
    var p;
    try { p = JSON.parse(raw); } catch (e) { p = null; }
    if (!p) { applyClasses(box, ""); return; }
    var status = p.status || "";
    var radios = box.querySelectorAll('input[type="radio"]');
    for (var i = 0; i < radios.length; i++) {
      radios[i].checked = (status !== "" && radios[i].value === status);
    }
    var ta = box.querySelector("textarea");
    if (ta && typeof p.revisions === "string") ta.value = p.revisions;
    applyClasses(box, status);
  }

  function isReviewed(d) {
    return d.status === "include" ||
           d.status === "modify"  ||
           d.status === "exclude";
  }

  function updateProgress() {
    var boxes = document.querySelectorAll(".feedback-box");
    var total = boxes.length;
    var reviewed = 0;
    boxes.forEach(function (b) { if (isReviewed(readBox(b))) reviewed++; });
    var progEl = document.getElementById("fb-progress");
    if (progEl) {
      progEl.textContent = reviewed + " of " + total + " items reviewed";
    }
    var btn = document.getElementById("fb-submit-btn");
    if (btn) {
      btn.textContent = "Submit feedback (" + reviewed +
                        " reviewed of " + total + " total)";
    }
  }

  function buildMarkdown(reviewerName, reviewerEmail) {
    var boxes = document.querySelectorAll(".feedback-box");
    var groups = { include: [], modify: [], exclude: [] };
    var counts = { include: 0, modify: 0, exclude: 0 };
    boxes.forEach(function (b) {
      var d = readBox(b);
      if (!isReviewed(d)) return;
      counts[d.status]++;
      groups[d.status].push(d);
    });

    var reviewedTotal = counts.include + counts.modify + counts.exclude;
    var iso = new Date().toISOString().slice(0, 10);
    var lines = [];
    lines.push("# Feedback on: " + PROJECT_TITLE);
    lines.push("");
    lines.push("**Reviewer:** " + (reviewerName || "_(not provided)_"));
    lines.push("**Email:** " + (reviewerEmail || "_(not provided)_"));
    lines.push("**Date:** " + iso);
    lines.push("**Summary:** " + counts.include + " include, " +
               counts.modify + " modify, " + counts.exclude +
               " exclude (total reviewed: " + reviewedTotal +
               "; items with no opinion are omitted)");
    lines.push("");
    lines.push("---");
    lines.push("");

    lines.push("## ✓ Include (" + counts.include + " items)");
    lines.push("");
    if (groups.include.length === 0) {
      lines.push("_(none)_");
    } else {
      groups.include.forEach(function (d) {
        var sec = d.section ? "[" + d.section + "] " : "";
        lines.push("- " + sec + d.caption);
      });
    }
    lines.push("");

    lines.push("## ⚠ Modify (" + counts.modify + " items)");
    lines.push("");
    if (groups.modify.length === 0) {
      lines.push("_(none)_");
      lines.push("");
    } else {
      groups.modify.forEach(function (d) {
        var sec = d.section ? "[" + d.section + "] " : "";
        lines.push("### " + sec + d.caption);
        var note = (d.revisions || "").trim();
        if (note) {
          var quoted = note.split(/\r?\n/).map(function (ln) {
            return "> " + ln;
          }).join("\n");
          lines.push(quoted);
        } else {
          lines.push("> _(no detail provided)_");
        }
        lines.push("");
      });
    }

    lines.push("## ✗ Exclude (" + counts.exclude + " items)");
    lines.push("");
    if (groups.exclude.length === 0) {
      lines.push("_(none)_");
    } else {
      groups.exclude.forEach(function (d) {
        var sec = d.section ? "[" + d.section + "] " : "";
        lines.push("- " + sec + d.caption);
      });
    }
    lines.push("");

    return {
      markdown: lines.join("\n"),
      counts: counts,
      reviewedTotal: reviewedTotal,
      total: boxes.length
    };
  }

  function downloadText(filename, text) {
    var blob = new Blob([text], { type: "text/markdown;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url; a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(function () {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 200);
  }

  function showToast(msg) {
    var t = document.getElementById("fb-toast");
    if (!t) return;
    t.textContent = msg;
    t.classList.add("show");
    setTimeout(function () { t.classList.remove("show"); }, 4500);
  }

  function onSubmit() {
    var nameEl = document.getElementById("fb-reviewer-name");
    var mailEl = document.getElementById("fb-reviewer-email");
    var reviewerName  = nameEl ? nameEl.value.trim()  : "";
    var reviewerEmail = mailEl ? mailEl.value.trim() : "";

    var built = buildMarkdown(reviewerName, reviewerEmail);
    var iso = new Date().toISOString().slice(0, 10);
    var fname = "feedback_" + slugify(PROJECT_TITLE) + "_" + iso + ".md";
    downloadText(fname, built.markdown);

    var subject = "Feedback: " + PROJECT_TITLE;
    var intro = "Hello,\n\n" +
      "Please find my feedback on the report '" + PROJECT_TITLE + "'.\n\n" +
      "Reviewer : " + (reviewerName || "(anonymous)") + "\n" +
      "Email    : " + (reviewerEmail || "(not provided)") + "\n" +
      "Date     : " + iso + "\n" +
      "Summary  : " + built.counts.include + " include, " +
      built.counts.modify + " modify, " + built.counts.exclude + " exclude " +
      "(total reviewed: " + built.reviewedTotal +
      " of " + built.total + ")\n\n" +
      "The full feedback file has been downloaded to your computer; " +
      "please attach it to this email before sending.\n\n" +
      "---- FEEDBACK (preview) ----\n\n";
    var maxLen = 1500;
    var preview = built.markdown;
    if (preview.length > maxLen) {
      preview = preview.slice(0, maxLen) +
        "\n\n[... truncated, see attached file ...]";
    }
    var body = intro + preview;
    var mailto = "mailto:" + encodeURIComponent(FEEDBACK_EMAIL) +
      "?subject=" + encodeURIComponent(subject) +
      "&body=" + encodeURIComponent(body);
    window.location.href = mailto;
    showToast("Saved! Email opened - please attach the downloaded file.");
  }

  function wireBox(box) {
    var radios = box.querySelectorAll('input[type="radio"]');
    var ta = box.querySelector("textarea");
    radios.forEach(function (r) {
      r.addEventListener("change", function () {
        applyClasses(box, r.value);
        persistBox(box);
        updateProgress();
      });
    });
    if (ta) {
      ta.addEventListener("input", function () {
        persistBox(box);
        updateProgress();
      });
    }
  }

  function init() {
    var boxes = document.querySelectorAll(".feedback-box");
    boxes.forEach(function (b) { restoreBox(b); wireBox(b); });

    var nameEl = document.getElementById("fb-reviewer-name");
    var mailEl = document.getElementById("fb-reviewer-email");
    if (nameEl) {
      nameEl.value = safeGet(REVIEWER_NAME_KEY) || "";
      nameEl.addEventListener("input", function () {
        safeSet(REVIEWER_NAME_KEY, nameEl.value);
      });
    }
    if (mailEl) {
      mailEl.value = safeGet(REVIEWER_MAIL_KEY) || "";
      mailEl.addEventListener("input", function () {
        safeSet(REVIEWER_MAIL_KEY, mailEl.value);
      });
    }
    var btn = document.getElementById("fb-submit-btn");
    if (btn) btn.addEventListener("click", onSubmit);

    updateProgress();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
</script>
"""


def _reviewer_panel_html() -> str:
    return (
        '<div class="reviewer-panel">'
        '<div class="rp-title">Reviewer</div>'
        '<label>Your name (optional)'
        '<input type="text" id="fb-reviewer-name" autocomplete="name" '
        'placeholder="e.g. Jane Doe"></label>'
        '<label>Your email (optional)'
        '<input type="email" id="fb-reviewer-email" autocomplete="email" '
        'placeholder="e.g. jane@example.org"></label>'
        '<div class="rp-progress" id="fb-progress">0 of 0 items reviewed</div>'
        '</div>'
    )


def _submit_button_html(feedback_email: str) -> str:
    safe = _html.escape(feedback_email, quote=True)
    return (
        '<button type="button" id="fb-submit-btn" class="fb-submit">'
        'Submit feedback (0 reviewed of 0 total)</button>'
        '<div id="fb-toast" class="fb-toast" role="status" aria-live="polite">'
        '</div>'
        '<noscript>'
        f'<a class="fb-noscript-mailto" href="mailto:{safe}">'
        f'Send feedback to {safe}</a>'
        '</noscript>'
    )


# =============================================================================
# 6) Top-level dashboard
# =============================================================================

_CSS = r"""
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               Helvetica, Arial, sans-serif;
  color: #1c1c1c;
  background: #fafafa;
  line-height: 1.5;
  scroll-behavior: smooth;
}
a { color: #1f3a68; text-decoration: none; }
a:hover { text-decoration: underline; }

.layout { display: grid; grid-template-columns: 240px 1fr; min-height: 100vh; }

aside.toc {
  position: sticky; top: 0; align-self: start;
  height: 100vh; overflow-y: auto;
  background: #1f3a68; color: #fff;
  padding: 18px 14px;
}
aside.toc h1 {
  font-size: 14px; text-transform: uppercase; letter-spacing: .08em;
  margin: 0 0 12px 0; color: #cfd9ea;
}
aside.toc ul { list-style: none; padding: 0; margin: 0; }
aside.toc li { margin: 0; }
aside.toc a {
  display: block; padding: 6px 8px; border-radius: 6px;
  color: #e5ecf7; font-size: 13px; line-height: 1.3;
}
aside.toc a:hover { background: rgba(255,255,255,.10); text-decoration: none; }

main { padding: 28px 36px 80px; max-width: 1280px; }
header.report-header {
  border-bottom: 2px solid #1f3a68; padding-bottom: 16px; margin-bottom: 28px;
}
header.report-header h1 {
  margin: 0 0 6px 0; font-size: 28px; color: #1f3a68;
}
header.report-header .subtitle { color: #555; font-size: 16px; margin: 0; }
header.report-header .meta {
  margin-top: 10px; color: #777; font-size: 12px;
}

section.report-section { margin-bottom: 56px; }
section.report-section h2 {
  font-size: 22px; color: #1f3a68; margin: 32px 0 14px;
  border-bottom: 1px solid #d6deee; padding-bottom: 6px;
}
section.report-section h3 {
  font-size: 16px; color: #2a3a52; margin: 24px 0 10px;
}
section.report-section h4 {
  font-size: 14px; color: #2a3a52; margin: 16px 0 6px;
}

.fig {
  margin: 18px 0 26px;
  background: #fff; border: 1px solid #e3e7ef; border-radius: 8px;
  padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
.fig img { width: 100%; height: auto; display: block; border-radius: 4px; }
.fig .caption {
  font-size: 12px; color: #555; margin-top: 8px; text-align: center;
}

.tablewrap {
  background: #fff; border: 1px solid #e3e7ef; border-radius: 8px;
  padding: 14px; margin: 18px 0 26px;
  box-shadow: 0 1px 2px rgba(0,0,0,.04);
  overflow-x: auto;
}
.tablewrap h3.tbl-title { margin-top: 0; }

table.bib-table {
  border-collapse: collapse; width: 100%;
  font-size: 13px;
}
table.bib-table thead th {
  background: #1f3a68; color: #fff; padding: 8px 10px; text-align: left;
  font-weight: 600; border: 0;
}
table.bib-table tbody td {
  padding: 6px 10px; border-top: 1px solid #eef1f7;
  vertical-align: top; word-break: break-word; max-width: 480px;
}
table.bib-table tbody tr:nth-child(even) td { background: #f6f8fc; }
table.bib-table tbody tr:hover td { background: #eaf0fa; }

.muted { color: #888; font-size: 12px; }
.table-caption { font-size: 13px; color: #444; margin: 0 0 6px; }
.sheet-header { color: #555; }
.kpi-row { display: flex; gap: 14px; flex-wrap: wrap; margin: 8px 0 18px; }
.kpi {
  background: #fff; border: 1px solid #e3e7ef; border-radius: 8px;
  padding: 12px 14px; min-width: 140px;
}
.kpi .label { color: #888; font-size: 11px; text-transform: uppercase;
              letter-spacing: .06em; }
.kpi .value { font-size: 20px; color: #1f3a68; font-weight: 600; }

@media (max-width: 880px) {
  .layout { grid-template-columns: 1fr; }
  aside.toc { position: static; height: auto; }
  main { padding: 18px; }
}
"""

_DATATABLES_JS = """
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
<script>
$(document).ready(function() {
  $('table.bib-table').each(function() {
    if ($(this).find('tbody tr').length > 10) {
      try {
        $(this).DataTable({
          paging: true, pageLength: 15, lengthChange: true,
          searching: true, info: true, order: [],
          autoWidth: false
        });
      } catch (e) { console.warn('DataTables init failed', e); }
    }
  });
});
</script>
"""


def _render_kpis(kpis: Dict[str, Any]) -> str:
    if not kpis:
        return ""
    parts = ['<div class="kpi-row">']
    for label, val in kpis.items():
        parts.append(
            f'<div class="kpi"><div class="label">{_html.escape(str(label))}</div>'
            f'<div class="value">{_html.escape(str(val))}</div></div>'
        )
    parts.append("</div>")
    return "\n".join(parts)


def _try_load_landscape(
    section_root: Path,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    parquet = section_root / "tables" / "landscape_coords.parquet"
    if not parquet.exists():
        return (None, None)
    try:
        df = pd.read_parquet(parquet)
        return (df, str(parquet))
    except Exception as e:  # noqa: BLE001
        return (None, f"read error: {e}")


def _project_paper_count(project_folder: Path) -> Optional[int]:
    candidates = [
        project_folder / "data" / "derived_full" / "clean_enriched.parquet",
        project_folder / "data" / "derived" / "clean_enriched.parquet",
    ]
    for c in candidates:
        if c.exists():
            try:
                df = pd.read_parquet(c)
                return len(df)
            except Exception:
                continue
    return None


def build_pipeline_report(
    project_folder: str | Path,
    out_html: str | Path,
    project_title: str = "Bibliometric Report",
    project_subtitle: Optional[str] = None,
    sections: Optional[Sequence[str]] = None,
    max_table_rows: int = 50,
    embed_landscape_plotly: bool = True,
    max_tables_per_section: int = 2,
    feedback_email: Optional[str] = None,
    feedback_enabled: bool = True,
) -> Path:
    """Assemble a single-file HTML dashboard."""
    project_folder = Path(project_folder)
    out_html = Path(out_html)
    results_root = project_folder / "results"
    found = scan_results_folder(results_root)

    if sections:
        wanted = set(sections)
        found = {k: v for k, v in found.items() if k in wanted}

    section_order = sorted(found.keys())

    # Resolve feedback recipient
    fb_email = (feedback_email or DEFAULT_FEEDBACK_EMAIL).strip()

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    paper_n = _project_paper_count(project_folder)
    total_plots = sum(len(v["plots"]) for v in found.values())
    total_tables = sum(len(v["tables"]) for v in found.values())
    kpis: Dict[str, Any] = {
        "Sections": len(section_order),
        "Plots": total_plots,
        "Tables": total_tables,
    }
    if paper_n is not None:
        kpis = {"Papers": f"{paper_n:,}", **kpis}

    toc_items: List[str] = []
    for sec in section_order:
        slug = _slugify(sec)
        toc_items.append(
            f'<li><a href="#sec-{slug}">'
            f'{_html.escape(_format_section_title(sec))}</a></li>'
        )
    toc_html = (
        '<aside class="toc">'
        f'<h1>{_html.escape(project_title)}</h1>'
        '<ul>' + "\n".join(toc_items) + '</ul>'
        '</aside>'
    )

    slug_counts: Dict[str, int] = {}

    def _unique_slug(base: str) -> str:
        n = slug_counts.get(base, 0) + 1
        slug_counts[base] = n
        return base if n == 1 else f"{base}-{n}"

    section_blocks: List[str] = []
    for sec in section_order:
        info = found[sec]
        slug = _slugify(sec)
        title = _format_section_title(sec)
        parts: List[str] = [f'<section class="report-section" id="sec-{slug}">']
        parts.append(f'<h2>{_html.escape(title)}</h2>')

        if (embed_landscape_plotly
                and ("paper_landscape" in sec or sec.endswith("paper_landscape"))):
            df_land, _ = _try_load_landscape(info["root"])
            if df_land is not None and len(df_land):
                hover_col = "Title" if "Title" in df_land.columns else None
                cluster_col = ("cluster" if "cluster" in df_land.columns
                               else ("Cluster" if "Cluster" in df_land.columns else None))
                if hover_col and cluster_col:
                    hover_text = [
                        _wrap_label(str(t), width=80).replace("\n", "<br>")
                        for t in df_land[hover_col].fillna("").tolist()
                    ]
                    landscape_slug = _unique_slug(f"{slug}-landscape")
                    parts.append('<h3>Interactive Paper Landscape</h3>')
                    parts.append(
                        '<p class="muted">Hover over points to see paper '
                        'titles. Colour encodes cluster membership.</p>'
                    )
                    parts.append(make_landscape_plotly(
                        df_land,
                        labels=df_land[cluster_col].astype(str).tolist(),
                        hover_text=hover_text,
                        title=f"Paper landscape ({len(df_land):,} papers)",
                        div_id=f"landscape-{slug}",
                    ))
                    if feedback_enabled:
                        parts.append(_feedback_box_html(
                            slug=landscape_slug,
                            caption=f"Interactive paper landscape "
                                    f"({len(df_land):,} papers)",
                            section_title=title,
                        ))

        if info["plots"]:
            parts.append('<h3>Figures</h3>')
            for img_path in info["plots"]:
                try:
                    data_uri = embed_image_b64(img_path)
                except Exception as e:  # noqa: BLE001
                    parts.append(
                        f'<p class="muted">Could not embed {img_path.name}: {e}</p>'
                    )
                    continue
                cap = _html.escape(img_path.stem.replace("_", " "))
                fig_slug = _unique_slug(f"{slug}-{_slugify(img_path.stem)}")
                parts.append(
                    f'<div class="fig">'
                    f'<img src="{data_uri}" alt="{cap}" loading="lazy">'
                    f'<div class="caption">{cap}</div>'
                    f'</div>'
                )
                if feedback_enabled:
                    parts.append(_feedback_box_html(
                        slug=fig_slug,
                        caption=img_path.stem.replace("_", " "),
                        section_title=title,
                    ))

        if info["tables"]:
            picks = _pick_priority_tables(
                info["tables"], n_pick=max_tables_per_section
            )
            if picks:
                parts.append('<h3>Tables</h3>')
            for xlsx_path in picks:
                tid = f"tbl-{slug}-{_slugify(xlsx_path.stem)}"
                table_html, _n_rows = _safe_excel_to_html(
                    xlsx_path, max_rows=max_table_rows, table_id=tid,
                )
                pretty_name = xlsx_path.stem.replace("_", " ").title()
                tbl_slug = _unique_slug(f"{slug}-table-{_slugify(xlsx_path.stem)}")
                parts.append('<div class="tablewrap">')
                parts.append(f'<h3 class="tbl-title">{_html.escape(pretty_name)}</h3>')
                parts.append(
                    f'<p class="muted">Source: {_html.escape(xlsx_path.name)}</p>'
                )
                parts.append(table_html)
                parts.append('</div>')
                if feedback_enabled:
                    parts.append(_feedback_box_html(
                        slug=tbl_slug,
                        caption=pretty_name,
                        section_title=title,
                    ))

        parts.append('</section>')
        section_blocks.append("\n".join(parts))

    subtitle_html = (
        f'<p class="subtitle">{_html.escape(project_subtitle)}</p>'
        if project_subtitle else ""
    )

    header_html = (
        '<header class="report-header">'
        f'<h1>{_html.escape(project_title)}</h1>'
        f'{subtitle_html}'
        f'<div class="meta">Generated {now}'
        + (f' &middot; {paper_n:,} papers' if paper_n else '')
        + f' &middot; {total_plots} figures &middot; {total_tables} tables'
        + '</div>'
        '</header>'
    )

    css_full = _CSS + ("\n" + _FEEDBACK_CSS if feedback_enabled else "")
    fb_panel  = _reviewer_panel_html()  if feedback_enabled else ""
    fb_button = _submit_button_html(fb_email) if feedback_enabled else ""
    fb_script = (
        _feedback_js(fb_email, project_title) if feedback_enabled else ""
    )

    full_html = (
        '<!DOCTYPE html>\n'
        '<html lang="en"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{_html.escape(project_title)}</title>'
        f'<style>{css_full}</style>'
        '</head><body>'
        '<div class="layout">'
        + toc_html
        + '<main>'
        + header_html
        + _render_kpis(kpis)
        + fb_panel
        + "\n".join(section_blocks)
        + '</main></div>'
        + fb_button
        + _DATATABLES_JS
        + fb_script
        + '</body></html>'
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(full_html, encoding="utf-8")
    return out_html


__all__ = [
    "embed_image_b64",
    "dataframe_to_html",
    "make_landscape_plotly",
    "scan_results_folder",
    "build_pipeline_report",
    "DEFAULT_FEEDBACK_EMAIL",
]

