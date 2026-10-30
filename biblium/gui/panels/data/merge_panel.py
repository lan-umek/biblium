# -*- coding: utf-8 -*-
"""
Merge Databases Panel (novo v 2.19)
===================================
Združi izvoze iz več baz (Scopus, WoS, OpenAlex, PubMed, Dimensions, Lens …)
v en nabor brez podvojenih zapisov (ujemanje po DOI / naslovu) in ga naloži
kot aktivni dataset.
"""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from biblium.gui.config import FONTS, DATABASE_OPTIONS
from biblium.gui.core.events import event_bus, EventBus
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import LabeledCombobox, LabeledCheckbox, LabeledSpinbox
from biblium.gui.widgets.tables import DataTable

try:
    from biblium.gui.widgets.plots import PlotFrame
    HAS_PLOTFRAME = True
except Exception:  # pragma: no cover
    HAS_PLOTFRAME = False

MAX_SOURCES = 5
DB_CHOICES = [k for k in DATABASE_OPTIONS if DATABASE_OPTIONS[k]] + ["Lens", "COBISS"]
DB_CODES = {**{k: v for k, v in DATABASE_OPTIONS.items() if v}, "Lens": "lens", "COBISS": "cobiss"}
KEY_OPTIONS = {"DOI (default)": "doi", "DOI, title when no DOI": "doi_title", "Title only": "title"}
COLUMN_OPTIONS = {"Union of columns (all fields, missing values allowed)": "union",
                  "Intersection of columns (only fields common to all sources)": "intersection"}


class MergeDatabasesPanel(BasePanel):
    """Combine several database exports into one de-duplicated dataset."""

    title = "Merge Databases"
    icon = "🔗"
    description = "Combine exports from different databases, match records by DOI or title, remove duplicates and load the result as the active dataset."
    requires_data = False

    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._rows = []
        self._result = None
        self._running = False
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._run_merge

    # ------------------------------------------------------------------ options
    def _create_options(self):
        self._add_title()
        card = Card(self.options_content, title="📂 Sources (file + database)", theme=self.theme_name)
        card.pack(fill=tk.X, padx=8, pady=8)
        self._rows_frame = tk.Frame(card.content, bg=self.theme["bg_card"])
        self._rows_frame.pack(fill=tk.X)
        for _ in range(2):
            self._add_row()
        ThemedButton(card.content, text="+ Add source", style="secondary", command=self._add_row,
                     theme=self.theme_name).pack(anchor=tk.W, pady=(6, 0))
        self.include_current_cb = LabeledCheckbox(card.content, label="Include currently loaded dataset as a source",
                                                  default=False, theme=self.theme_name)
        self.include_current_cb.pack(fill=tk.X, pady=(6, 0))

        opt = Card(self.options_content, title="⚙️ Matching", theme=self.theme_name)
        opt.pack(fill=tk.X, padx=8, pady=8)
        self.key_combo = LabeledCombobox(opt.content, label="Match records by:", values=list(KEY_OPTIONS),
                                         default="DOI (default)", theme=self.theme_name, label_width=18, width=28)
        self.key_combo.pack(fill=tk.X, pady=3)
        self.columns_combo = LabeledCombobox(opt.content, label="Columns:", values=list(COLUMN_OPTIONS),
                                             default=list(COLUMN_OPTIONS)[0], theme=self.theme_name, label_width=18, width=28)
        self.columns_combo.pack(fill=tk.X, pady=3)
        self.source_cols_cb = LabeledCheckbox(opt.content, label="Keep per-source 'Cited by (<db>)' columns",
                                              default=True, theme=self.theme_name)
        self.source_cols_cb.pack(fill=tk.X, pady=3)
        self.priority_combo = LabeledCombobox(opt.content, label="Preferred source:", values=["(default order)"] + DB_CHOICES,
                                              default="(default order)", theme=self.theme_name, label_width=18, width=28)
        self.priority_combo.pack(fill=tk.X, pady=3)
        self.preprocess_spin = LabeledSpinbox(opt.content, label="Preprocess level:", from_=0, to=4, default=0,
                                              theme=self.theme_name, label_width=18)
        self.preprocess_spin.pack(fill=tk.X, pady=3)

        btns = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btns.pack(fill=tk.X, padx=8, pady=12)
        self.run_btn = ActionButton(btns, text="Merge & Load", icon="▶", command=self._run_merge, theme=self.theme_name)
        self.run_btn.pack(fill=tk.X)
        ThemedButton(btns, text="Export merged dataset…", icon="📥", style="secondary", command=self._export,
                     theme=self.theme_name).pack(fill=tk.X, pady=(8, 0))

    def _add_row(self):
        if len(self._rows) >= MAX_SOURCES:
            return
        row = tk.Frame(self._rows_frame, bg=self.theme["bg_card"])
        row.pack(fill=tk.X, pady=2)
        path_var = tk.StringVar()
        entry = tk.Entry(row, textvariable=path_var, font=FONTS.get_font("body"), width=26)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ThemedButton(row, text="…", style="secondary", command=lambda v=path_var: self._browse(v),
                     theme=self.theme_name).pack(side=tk.LEFT, padx=3)
        db_var = tk.StringVar(value=DB_CHOICES[min(len(self._rows), len(DB_CHOICES) - 1)])
        combo = ttk.Combobox(row, textvariable=db_var, values=DB_CHOICES, width=12, state="readonly")
        combo.pack(side=tk.LEFT)
        self._rows.append((path_var, db_var))

    def _browse(self, var):
        p = filedialog.askopenfilename(filetypes=[("Bibliographic exports", "*.csv *.txt *.xlsx *.xls *.bib *.ris *.json"), ("All", "*.*")])
        if p:
            var.set(p)

    # ------------------------------------------------------------------ results
    def _create_results(self):
        self.notebook = ttk.Notebook(self.results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.results_content = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.results_content, text="📋 Summary")
        self.overlap_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.overlap_frame, text="📊 Source overlap")
        self.coverage_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.coverage_frame, text="📊 Column coverage")
        self.plot_frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.plot_frame, text="📈 Overlap plot")
        info = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(info, text="ℹ️ Info")
        txt = tk.Text(info, font=FONTS.get_font("monospace"), bg=self.theme["bg_card"], fg=self.theme["text_primary"],
                      wrap=tk.WORD, padx=16, pady=16, relief=tk.FLAT)
        txt.insert("1.0", INFO_TEXT.strip("\n")); txt.config(state=tk.DISABLED); txt.pack(fill=tk.BOTH, expand=True)
        self._show_initial_message("Add two or more exports, choose the matching key and click Merge & Load.")

    # ------------------------------------------------------------------ run
    def _collect_sources(self):
        sources = []
        for path_var, db_var in self._rows:
            p = path_var.get().strip()
            if p:
                sources.append((p, DB_CODES.get(db_var.get(), db_var.get().lower())))
        if self.include_current_cb.get() and self.bib is not None:
            sources.insert(0, (self.bib.df, getattr(self.bib, "db", "scopus") or "scopus"))
        return sources

    def _run_merge(self):
        if self._running:
            return
        sources = self._collect_sources()
        if len(sources) < 2:
            messagebox.showwarning("Merge", "Add at least two sources (or one file plus the loaded dataset).")
            return
        key = KEY_OPTIONS.get(self.key_combo.get(), "doi")
        columns = COLUMN_OPTIONS.get(self.columns_combo.get(), "union")
        pref = self.priority_combo.get()
        priority = None
        if pref and pref != "(default order)":
            from biblium.merge import DEFAULT_PRIORITY
            code = DB_CODES.get(pref, pref.lower())
            priority = [code] + [d for d in DEFAULT_PRIORITY if d != code]
        keep_src = self.source_cols_cb.get()
        pre = self.preprocess_spin.get()
        self._running = True
        self.run_btn.config(state=tk.DISABLED, text="⏳ Merging...")
        self._show_loading("Reading sources and merging records…")

        def work():
            try:
                from biblium.bibstats import BiblioStats
                bib = BiblioStats.from_databases(sources, key=key, columns=columns, priority=priority,
                                                 keep_source_columns=keep_src, preprocess_level=pre, verbose=False)
                self._safe_after(lambda: self._on_success(bib))
            except Exception as e:  # noqa: BLE001
                import traceback
                tb = traceback.format_exc()
                self._safe_after(lambda msg=str(e), t=tb: self._on_error(msg, t))

        threading.Thread(target=work, daemon=True).start()

    def _on_error(self, msg, tb):
        self._running = False
        self.run_btn.config(state=tk.NORMAL, text="▶ Merge & Load")
        self._show_error(msg)
        try:
            messagebox.showerror("Merge error", f"{msg}\n\n{tb[-1200:]}")
        except Exception:
            pass

    def _on_success(self, bib):
        self._running = False
        self.run_btn.config(state=tk.NORMAL, text="▶ Merge & Load")
        self._result = bib
        self.bib = bib
        rep = bib.merge_report
        self._stop_active_spinners()
        self._safe_clear_results()
        grid = tk.Frame(self.results_content, bg=self.theme["bg_card"])
        grid.pack(fill=tk.X, padx=8, pady=8)
        cells = [("Input records", rep["n_input_total"]), ("Unique records", rep["n_output"]),
                 ("Duplicates removed", rep["n_removed"]), ("In ≥ 2 sources", int((bib.df["N sources"] > 1).sum()))]
        for i, (lab, val) in enumerate(cells):
            c = tk.Frame(grid, bg=self.theme["bg_secondary"], padx=12, pady=8)
            c.grid(row=0, column=i, padx=4, pady=4, sticky="nsew"); grid.columnconfigure(i, weight=1)
            tk.Label(c, text=lab.upper(), font=FONTS.get_font("caption"), bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"]).pack()
            tk.Label(c, text=str(val), font=FONTS.get_font("h2"), bg=self.theme["bg_secondary"], fg=self.theme["accent_primary"]).pack()
        per_src = ", ".join(f"{k}: {v}" for k, v in rep["n_input"].items())
        msg = (f"Sources: {per_src}\nMatch key: {rep['key']} · Columns: {rep['columns']} "
               f"({len(rep['columns_kept'])} kept, {len(rep['columns_dropped'])} dropped)\n"
               f"Records without a usable key: {rep['n_without_key']} · Within-source duplicates: {rep['n_within_source_duplicates']}\n"
               "The merged dataset is now the active dataset (column 'Source DB' gives the provenance).")
        tk.Label(self.results_content, text=msg, font=FONTS.get_font("body"), bg=self.theme["bg_card"],
                 fg=self.theme["text_secondary"], justify=tk.LEFT, wraplength=720).pack(anchor=tk.W, padx=12, pady=8)

        for frame, df, name in ((self.overlap_frame, rep["overlap"], "Source overlap"),
                                (self.coverage_frame, rep["column_coverage"], "Column coverage")):
            for w in frame.winfo_children():
                w.destroy()
            t = DataTable(frame, theme=self.theme_name, max_rows=1000); t.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
            try:
                t.set_table_title(f"Merge databases — {name}")
            except Exception:
                pass
            t.set_data(df)
        if HAS_PLOTFRAME:
            for w in self.plot_frame.winfo_children():
                w.destroy()
            try:
                from biblium.merge import plot_source_overlap
                pf = PlotFrame(self.plot_frame, theme=self.theme_name, figsize=(8, 5)); pf.pack(fill=tk.BOTH, expand=True)
                pf.figure.clear(); plot_source_overlap(rep["overlap"], ax=pf.figure.add_subplot(111)); pf.refresh()
            except Exception as e:  # noqa: BLE001
                tk.Label(self.plot_frame, text=f"Plot error: {e}", bg=self.theme["bg_card"]).pack()

        years = bib.df["Year"].dropna() if "Year" in bib.df.columns else []
        year_range = (int(years.min()), int(years.max())) if len(years) else (None, None)
        try:
            event_bus.emit(EventBus.RESET_PANELS, {})
        except Exception:
            pass
        event_bus.emit(EventBus.DATASET_LOADED, {
            "bib": bib, "original_df": bib.df.copy(), "path": "merged:" + "+".join(rep["priority"]),
            "n_documents": bib.n, "database": "merged (" + ", ".join(rep["priority"]) + ")", "year_range": year_range,
        })
        self.notebook.select(0)

    def _export(self):
        if self._result is None:
            messagebox.showinfo("Export", "Run the merge first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv")],
                                         initialfile="merged_dataset.xlsx")
        if not p:
            return
        try:
            if p.lower().endswith(".csv"):
                self._result.df.to_csv(p, index=False)
            else:
                import pandas as pd
                with pd.ExcelWriter(p) as xw:
                    self._result.df.to_excel(xw, sheet_name="merged", index=False)
                    self._result.merge_report["overlap"].to_excel(xw, sheet_name="source overlap", index=False)
                    self._result.merge_report["column_coverage"].to_excel(xw, sheet_name="column coverage", index=False)
            messagebox.showinfo("Export", f"Saved to {p}")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Export", str(e))


INFO_TEXT = """
MERGING RECORDS FROM SEVERAL DATABASES
──────────────────────────────────────
Each export is read with biblium's readers (column names are mapped to the
common Scopus-like names), separators and DOIs are harmonised, and records are
matched across sources:

• DOI (default) – exact match of normalised DOIs; records without a DOI stay
  separate.
• DOI, title when no DOI – DOI first; records without a DOI are matched on the
  normalised title.
• Title only – normalised title (lower case, punctuation removed).

Duplicate groups are merged into one record: citations = maximum over sources
(per-source counts kept as 'Cited by (<db>)'), keywords and references = union,
abstract = longest, other fields from the preferred source (default order:
Scopus > WoS > Dimensions > OpenAlex > Lens > PubMed).

Columns: 'Union' keeps every field of every source (fields missing in a source
are empty for its records); 'Intersection' keeps only fields present in all
sources, so there is no structural missingness.

Provenance columns: Source DB ("scopus; wos"), N sources, Merge key, Match type.
The 'Source overlap' table counts records per combination of sources.
"""


__all__ = ["MergeDatabasesPanel"]
