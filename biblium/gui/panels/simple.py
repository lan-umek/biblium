# -*- coding: utf-8 -*-
"""
SimpleAnalysisPanel (novo v 2.19)
=================================
Deklarativna osnova za analizne panele: podrazred poda seznam opcij
(``OPTIONS``) in metodo ``compute(bib, opts) -> dict`` (teče v niti), rezultat
pa se prikaže kot zavihki s tabelami (``DataTable``), slikami (``PlotFrame``)
in besedilom. S tem je nov panel dolg ~60–100 vrstic namesto ~800.

Rezultat ``compute`` je slovar z neobveznimi ključi:
- ``"tables"``: dict {ime_zavihka: DataFrame}
- ``"plots"``:  dict {ime_zavihka: callable(fig) -> None}  (risanje v podano
                 matplotlib figuro; funkcija naj sama doda osi)
- ``"text"``:   dict {ime_zavihka: str}
- ``"summary"``: list[(label, value)] — kartice z metrikami na vrhu zavihka
                 "Summary"
- ``"message"``: str (status vrstica)
"""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Callable, Dict, List, Optional

from biblium.gui.config import FONTS
from biblium.gui.panels.base import BasePanel
from biblium.gui.widgets.cards import Card
from biblium.gui.widgets.buttons import ThemedButton, ActionButton
from biblium.gui.widgets.forms import (
    LabeledSpinbox, LabeledCheckbox, LabeledCombobox, LabeledEntry,
)
from biblium.gui.widgets.tables import DataTable

try:
    from biblium.gui.widgets.plots import PlotFrame
    HAS_PLOTFRAME = True
except Exception:  # pragma: no cover
    HAS_PLOTFRAME = False


class Opt:
    """Opis ene opcije v ``OPTIONS``.

    kind: "spin" (int), "check" (bool), "combo" (str), "entry" (str),
          "columns" (combo z imeni stolpcev datasetov; ``values`` so
          privzeti kandidati, dopolnijo se s stolpci ``bib.df``).
    """

    def __init__(self, key: str, label: str, kind: str = "entry", default: Any = None,
                 values: Optional[List[str]] = None, from_: int = 0, to: int = 10_000,
                 card: str = "⚙️ Settings", width: int = 22, help: str = ""):
        self.key, self.label, self.kind, self.default = key, label, kind, default
        self.values, self.from_, self.to, self.card, self.width, self.help = (
            values or [], from_, to, card, width, help)


class SimpleAnalysisPanel(BasePanel):
    """Osnova za preproste analizne panele (glej dokumentacijo modula)."""

    title = "Analysis"
    icon = "🧪"
    description = ""
    requires_data = True
    OPTIONS: List[Opt] = []
    INFO_TEXT: str = ""
    RUN_LABEL = "Run Analysis"
    SHOW_LLM_CARD = False

    def __init__(self, parent, theme: str = "light", bib=None, **kwargs):
        self._widgets: Dict[str, Any] = {}
        self._result: Optional[Dict[str, Any]] = None
        self._tables: Dict[str, DataTable] = {}
        super().__init__(parent, theme=theme, bib=bib, **kwargs)
        self._primary_action = self._run_clicked_simple

    # ------------------------------------------------------------------ options
    def _create_options(self):
        self._add_title()
        cards: Dict[str, Card] = {}
        for opt in self.OPTIONS:
            if opt.card not in cards:
                cards[opt.card] = Card(self.options_content, title=opt.card, theme=self.theme_name)
                cards[opt.card].pack(fill=tk.X, padx=8, pady=8)
            parent = cards[opt.card].content
            w = self._make_widget(parent, opt)
            w.pack(fill=tk.X, pady=3)
            self._widgets[opt.key] = (opt, w)
        if self.SHOW_LLM_CARD:
            try:
                self._create_llm_config_card()
            except Exception:
                pass
        btn_frame = tk.Frame(self.options_content, bg=self.theme["bg_secondary"])
        btn_frame.pack(fill=tk.X, padx=8, pady=16)
        self.run_btn = ActionButton(btn_frame, text=self.RUN_LABEL, icon="▶",
                                    command=self._run_clicked_simple, theme=self.theme_name)
        self.run_btn.pack(fill=tk.X)
        self.export_btn = ThemedButton(btn_frame, text="Export Results", command=self._export_results,
                                       style="secondary", icon="📥", theme=self.theme_name)
        self.export_btn.pack(fill=tk.X, pady=(8, 0))

    def _column_values(self, opt: Opt) -> List[str]:
        vals = list(opt.values)
        try:
            if self.bib is not None and getattr(self.bib, "df", None) is not None:
                for c in self.bib.df.columns:
                    if c not in vals:
                        vals.append(str(c))
        except Exception:
            pass
        return vals

    def _make_widget(self, parent, opt: Opt):
        kw = dict(theme=self.theme_name, label_width=18)
        if opt.kind == "spin":
            return LabeledSpinbox(parent, label=opt.label, from_=opt.from_, to=opt.to,
                                  default=opt.default if opt.default is not None else opt.from_, **kw)
        if opt.kind == "check":
            return LabeledCheckbox(parent, label=opt.label, default=bool(opt.default), theme=self.theme_name)
        if opt.kind in ("combo", "columns"):
            values = self._column_values(opt) if opt.kind == "columns" else list(opt.values)
            default = opt.default if opt.default in values or not values else values[0]
            return LabeledCombobox(parent, label=opt.label, values=values, default=default,
                                   width=opt.width, **kw)
        return LabeledEntry(parent, label=opt.label, default="" if opt.default is None else str(opt.default),
                            width=opt.width, **kw)

    def get_options(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, (opt, w) in self._widgets.items():
            try:
                v = w.get()
            except Exception:
                v = opt.default
            if opt.kind == "entry" and isinstance(v, str):
                v = v.strip()
                if v == "":
                    v = None
            out[key] = v
        return out

    # ------------------------------------------------------------------ results
    def _create_results(self):
        self.notebook = ttk.Notebook(self.results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.results_content = tk.Frame(self.notebook, bg=self.theme["bg_card"])
        self.notebook.add(self.results_content, text="📋 Results")
        if self.INFO_TEXT:
            info = tk.Frame(self.notebook, bg=self.theme["bg_card"])
            self.notebook.add(info, text="ℹ️ Info")
            txt = tk.Text(info, font=FONTS.get_font("monospace"), bg=self.theme["bg_card"],
                          fg=self.theme["text_primary"], wrap=tk.WORD, padx=16, pady=16, relief=tk.FLAT)
            txt.insert("1.0", self.INFO_TEXT.strip("\n"))
            txt.config(state=tk.DISABLED)
            txt.pack(fill=tk.BOTH, expand=True)
        self._show_initial_message()

    def _clear_result_tabs(self):
        keep = {str(self.results_content)}
        for tab_id in list(self.notebook.tabs()):
            if tab_id in keep or self.notebook.tab(tab_id, "text") == "ℹ️ Info":
                continue
            try:
                self.notebook.forget(tab_id)
            except Exception:
                pass
        self._tables = {}
        self._safe_clear_results()

    # ------------------------------------------------------------------ run
    def _run_clicked_simple(self):
        if self._is_running:
            return
        if self.requires_data and self.bib is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        opts = self.get_options()
        self._is_running = True
        self.run_btn.config(state=tk.DISABLED, text="⏳ Running...")
        self._clear_result_tabs()
        self._show_loading(f"Running {self.title}...")

        def work():
            try:
                res = self.compute(self.bib, opts)
                self._safe_after(lambda: self._on_success(res))
            except Exception as e:  # noqa: BLE001
                import traceback
                tb = traceback.format_exc()
                self._safe_after(lambda msg=str(e), t=tb: self._on_error(msg, t))

        threading.Thread(target=work, daemon=True).start()

    def compute(self, bib, opts: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - override
        raise NotImplementedError

    def _on_error(self, msg, tb):
        self._is_running = False
        self.run_btn.config(state=tk.NORMAL, text=f"▶ {self.RUN_LABEL}")
        self._show_error(msg)
        try:
            messagebox.showerror("Analysis Error", f"{msg}\n\n{tb[-1200:]}")
        except Exception:
            pass

    def _on_success(self, res: Dict[str, Any]):
        self._is_running = False
        self._result = res or {}
        self.run_btn.config(state=tk.NORMAL, text=f"▶ {self.RUN_LABEL}")
        self._render(self._result)

    # ------------------------------------------------------------------ render
    def _render(self, res: Dict[str, Any]):
        self._stop_active_spinners()
        self._safe_clear_results()
        summary = res.get("summary") or []
        message = res.get("message")
        if summary:
            grid = tk.Frame(self.results_content, bg=self.theme["bg_card"])
            grid.pack(fill=tk.X, padx=8, pady=8)
            ncol = min(4, max(1, len(summary)))
            for i, (label, value) in enumerate(summary):
                cell = tk.Frame(grid, bg=self.theme["bg_secondary"], padx=12, pady=8)
                cell.grid(row=i // ncol, column=i % ncol, padx=4, pady=4, sticky="nsew")
                grid.columnconfigure(i % ncol, weight=1)
                tk.Label(cell, text=str(label).upper(), font=FONTS.get_font("caption"),
                         bg=self.theme["bg_secondary"], fg=self.theme["text_secondary"]).pack()
                tk.Label(cell, text=str(value), font=FONTS.get_font("h2"),
                         bg=self.theme["bg_secondary"], fg=self.theme["accent_primary"]).pack()
        if message:
            tk.Label(self.results_content, text=message, font=FONTS.get_font("body"),
                     bg=self.theme["bg_card"], fg=self.theme["text_secondary"],
                     wraplength=700, justify=tk.LEFT).pack(anchor=tk.W, padx=12, pady=8)
        if not summary and not message:
            tk.Label(self.results_content, text="Done — see the other tabs.", font=FONTS.get_font("body"),
                     bg=self.theme["bg_card"], fg=self.theme["text_muted"]).pack(expand=True)

        for name, df in (res.get("tables") or {}).items():
            frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
            self.notebook.add(frame, text=f"📊 {name}")
            table = DataTable(frame, theme=self.theme_name, max_rows=1000)
            table.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
            try:
                table.set_table_title(f"{self.title} — {name}")
            except Exception:
                pass
            table.set_data(df)
            self._tables[name] = table

        if HAS_PLOTFRAME:
            for name, drawer in (res.get("plots") or {}).items():
                frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
                self.notebook.add(frame, text=f"📈 {name}")
                try:
                    plot = PlotFrame(frame, theme=self.theme_name, figsize=(9, 6))
                    plot.pack(fill=tk.BOTH, expand=True)
                    plot.figure.clear()
                    drawer(plot.figure)
                    try:
                        plot.figure.tight_layout()
                    except Exception:
                        pass
                    plot.refresh()
                except Exception as e:  # noqa: BLE001
                    tk.Label(frame, text=f"Plot error: {e}", bg=self.theme["bg_card"],
                             fg=self.theme["danger"]).pack(expand=True)

        for name, text in (res.get("text") or {}).items():
            frame = tk.Frame(self.notebook, bg=self.theme["bg_card"])
            self.notebook.add(frame, text=f"📝 {name}")
            txt = tk.Text(frame, font=FONTS.get_font("monospace"), bg=self.theme["bg_card"],
                          fg=self.theme["text_primary"], wrap=tk.WORD, padx=16, pady=16, relief=tk.FLAT)
            txt.insert("1.0", text)
            txt.config(state=tk.DISABLED)
            txt.pack(fill=tk.BOTH, expand=True)
        try:
            self.notebook.select(0)
        except Exception:
            pass

    # ------------------------------------------------------------------ export
    def _export_results(self):
        if not self._result or not self._result.get("tables"):
            messagebox.showinfo("Export", "Run the analysis first.")
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                            filetypes=[("Excel", "*.xlsx")],
                                            initialfile=f"{self.title.lower().replace(' ', '_')}.xlsx")
        if not path:
            return
        try:
            import pandas as pd
            with pd.ExcelWriter(path) as xw:
                for name, df in self._result["tables"].items():
                    df.to_excel(xw, sheet_name=str(name)[:31], index=False)
            messagebox.showinfo("Export", f"Saved to {path}")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Export", str(e))
