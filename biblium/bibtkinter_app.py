# -*- coding: utf-8 -*-
"""
Biblium Tkinter GUI Application
================================
Modern UI with proper scrolling and display
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import traceback
import threading
import os

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from biblium import BiblioAnalysis, BiblioPlot
    HAS_BIBLIUM = True
except ImportError:
    HAS_BIBLIUM = False


# Mappings
DB_MAP = {"OpenAlex": "oa", "Scopus": "scopus", "Web of Science": "wos", "PubMed": "pubmed"}
PREPROCESS_MAP = {
    "0 - No preprocessing": 0,
    "1 - Basic (country info, labels)": 1,
    "2 - Keywords & text processing": 2,
    "3 - Science mappings": 3,
    "4 - Full (with interdisciplinarity)": 4
}
KEYWORD_MAP = {"Author Keywords": "author", "Index Keywords": "index", "Both (combined)": "both"}
LANG_MAP = {"English": "en", "Slovenian": "sl", "German": "de", "Spanish": "es", "French": "fr"}
CMAP_LIST = ["viridis", "plasma", "inferno", "magma", "cividis", "tab10", "tab20", "Set1", "Set2"]


class ModernStyle:
    """Modern color scheme"""
    BG_SIDEBAR = "#1e293b"
    BG_CONTENT = "#f1f5f9"
    BG_CARD = "#ffffff"
    BG_INPUT = "#f0fdf4"
    
    ACCENT = "#3b82f6"
    ACCENT_HOVER = "#2563eb"
    SUCCESS = "#10b981"
    WARNING = "#f59e0b"
    DANGER = "#ef4444"
    
    TEXT_LIGHT = "#ffffff"
    TEXT_DARK = "#1e293b"
    TEXT_MUTED = "#64748b"
    TEXT_SIDEBAR = "#94a3b8"
    
    BORDER = "#e2e8f0"
    
    FONT_TITLE = ("Segoe UI", 18, "bold")
    FONT_SUBTITLE = ("Segoe UI", 10)
    FONT_HEADING = ("Segoe UI", 11, "bold")
    FONT_BODY = ("Segoe UI", 10)
    FONT_SMALL = ("Segoe UI", 9)
    FONT_MONO = ("Consolas", 10)


class ToolTip:
    def __init__(self, widget, text):
        self.widget, self.text, self.tw = widget, text, None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        x, y = self.widget.winfo_rootx() + 25, self.widget.winfo_rooty() + 25
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tw, text=self.text, background="#334155", foreground="#fff",
                font=ModernStyle.FONT_SMALL, relief="flat", padx=8, pady=4).pack()

    def hide(self, event=None):
        if self.tw: self.tw.destroy(); self.tw = None


class LoadingSpinner(tk.Canvas):
    def __init__(self, parent, size=20, **kwargs):
        super().__init__(parent, width=size, height=size, highlightthickness=0, **kwargs)
        self.size = size
        self.angle, self.running = 0, False
        self.arc = self.create_arc(2, 2, size-2, size-2, start=0, extent=90, 
                                   outline=ModernStyle.ACCENT, width=2, style="arc")

    def start(self): self.running = True; self._animate()
    def stop(self): self.running = False

    def _animate(self):
        if self.running:
            self.angle = (self.angle - 12) % 360
            self.itemconfig(self.arc, start=self.angle)
            self.after(40, self._animate)


class BibliumApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Biblium")
        self.geometry("1450x900")
        self.minsize(1100, 700)
        self.configure(bg=ModernStyle.BG_CONTENT)
        
        self.bib = None
        self.original_df = None
        self.dataset_path = None
        
        self._setup_styles()
        self._create_layout()
        self._create_sidebar()
        self._create_welcome()
        
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure("TNotebook", background=ModernStyle.BG_CONTENT, borderwidth=0)
        style.configure("TNotebook.Tab", padding=[15, 8], font=ModernStyle.FONT_BODY,
                       background=ModernStyle.BG_CONTENT, foreground=ModernStyle.TEXT_MUTED)
        style.map("TNotebook.Tab", 
                 background=[("selected", ModernStyle.BG_CARD)],
                 foreground=[("selected", ModernStyle.ACCENT)])
        
        style.configure("Treeview", font=ModernStyle.FONT_BODY, rowheight=26,
                       background=ModernStyle.BG_CARD, fieldbackground=ModernStyle.BG_CARD)
        style.configure("Treeview.Heading", font=ModernStyle.FONT_BODY, 
                       background="#f8fafc", foreground=ModernStyle.TEXT_DARK)
        style.map("Treeview", background=[("selected", "#dbeafe")])
        
        style.configure("TCombobox", font=ModernStyle.FONT_BODY, padding=5)
        style.configure("Horizontal.TScrollbar", background=ModernStyle.BORDER, troughcolor="#f8fafc")
        style.configure("Vertical.TScrollbar", background=ModernStyle.BORDER, troughcolor="#f8fafc")
        
    def _create_layout(self):
        self.main_container = tk.Frame(self, bg=ModernStyle.BG_CONTENT)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        self.sidebar = tk.Frame(self.main_container, bg=ModernStyle.BG_SIDEBAR, width=220)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)
        
        self.content = tk.Frame(self.main_container, bg=ModernStyle.BG_CONTENT)
        self.content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.notebook = ttk.Notebook(self.content)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(12, 8))
        
        self.status_frame = tk.Frame(self, bg=ModernStyle.BG_CARD, height=32)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_frame.pack_propagate(False)
        
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.status_frame, textvariable=self.status_var, bg=ModernStyle.BG_CARD,
                fg=ModernStyle.TEXT_MUTED, font=ModernStyle.FONT_SMALL, padx=15).pack(side=tk.LEFT, fill=tk.Y)
        
        self.doc_var = tk.StringVar(value="")
        tk.Label(self.status_frame, textvariable=self.doc_var, bg=ModernStyle.BG_CARD,
                fg=ModernStyle.ACCENT, font=ModernStyle.FONT_SMALL, padx=15).pack(side=tk.RIGHT, fill=tk.Y)
        
    def _create_sidebar(self):
        logo_frame = tk.Frame(self.sidebar, bg=ModernStyle.BG_SIDEBAR)
        logo_frame.pack(fill=tk.X, pady=20)
        tk.Label(logo_frame, text="📚 Biblium", font=("Segoe UI", 20, "bold"),
                bg=ModernStyle.BG_SIDEBAR, fg=ModernStyle.TEXT_LIGHT).pack()
        tk.Label(logo_frame, text="Bibliometric Analysis", font=ModernStyle.FONT_SMALL,
                bg=ModernStyle.BG_SIDEBAR, fg=ModernStyle.TEXT_SIDEBAR).pack()
        
        canvas = tk.Canvas(self.sidebar, bg=ModernStyle.BG_SIDEBAR, highlightthickness=0)
        self.menu_frame = tk.Frame(canvas, bg=ModernStyle.BG_SIDEBAR)
        
        self.menu_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.menu_frame, anchor="nw", width=210)
        canvas.pack(side="left", fill="both", expand=True, padx=5)
        
        def _scroll(e): canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _scroll))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        
        self._menu_section("DATA")
        self._menu_btn("Load Dataset", self.tab_load_data, "📂")
        self._menu_btn("View Data", self.tab_data_table, "📋")
        self._menu_btn("Filter", self.tab_filter, "🔍")
        
        self._menu_section("ANALYSIS")
        self._menu_btn("Overview", self.tab_main_info, "ℹ️")
        self._menu_btn("Production", self.tab_production, "📈")
        self._menu_btn("Top Items", self.tab_top_items, "📊")
        self._menu_btn("Statistics", self.tab_statistics, "📉")
        
        self._menu_section("NETWORKS")
        self._menu_btn("Keywords", self.tab_keyword_net, "🔗")
        self._menu_btn("Co-authorship", self.tab_coauthorship, "👥")
        self._menu_btn("Countries", self.tab_country_collab, "🌍")
        self._menu_btn("Historiograph", self.tab_historiograph, "📜")
        
        self._menu_section("MAPPING")
        self._menu_btn("Conceptual", self.tab_wordmap, "🗺️")
        self._menu_btn("Thematic", self.tab_thematic, "📊")
        self._menu_btn("Trends", self.tab_trend, "📈")
        
        self._menu_section("LAWS")
        self._menu_btn("Lotka", self.tab_lotka, "📉")
        self._menu_btn("Bradford", self.tab_bradford, "📚")
        
    def _menu_section(self, title):
        f = tk.Frame(self.menu_frame, bg=ModernStyle.BG_SIDEBAR)
        f.pack(fill=tk.X, pady=(18, 6), padx=8)
        tk.Label(f, text=title, font=("Segoe UI", 9, "bold"), bg=ModernStyle.BG_SIDEBAR,
                fg=ModernStyle.TEXT_SIDEBAR).pack(anchor=tk.W)
        
    def _menu_btn(self, text, cmd, icon=""):
        btn = tk.Button(self.menu_frame, text=f"{icon}  {text}" if icon else text,
                       anchor=tk.W, bg=ModernStyle.BG_SIDEBAR, fg=ModernStyle.TEXT_LIGHT,
                       activebackground=ModernStyle.ACCENT, activeforeground=ModernStyle.TEXT_LIGHT,
                       relief=tk.FLAT, font=ModernStyle.FONT_BODY, padx=18, pady=8,
                       cursor="hand2", command=cmd, bd=0)
        btn.pack(fill=tk.X, padx=3, pady=1)
        btn.bind("<Enter>", lambda e: btn.config(bg="#334155"))
        btn.bind("<Leave>", lambda e: btn.config(bg=ModernStyle.BG_SIDEBAR))

    def _create_welcome(self):
        f = tk.Frame(self.notebook, bg=ModernStyle.BG_CONTENT)
        self.notebook.add(f, text="  Home  ")
        
        center = tk.Frame(f, bg=ModernStyle.BG_CONTENT)
        center.place(relx=0.5, rely=0.45, anchor=tk.CENTER)
        
        tk.Label(center, text="📚", font=("Segoe UI", 56), bg=ModernStyle.BG_CONTENT).pack()
        tk.Label(center, text="Biblium", font=("Segoe UI", 32, "bold"), 
                bg=ModernStyle.BG_CONTENT, fg=ModernStyle.TEXT_DARK).pack(pady=(8, 4))
        tk.Label(center, text="Bibliometric Analysis Platform",
                font=ModernStyle.FONT_SUBTITLE, bg=ModernStyle.BG_CONTENT, 
                fg=ModernStyle.TEXT_MUTED).pack()
        
        tk.Frame(center, height=35, bg=ModernStyle.BG_CONTENT).pack()
        
        btn = tk.Button(center, text="  📂  Load Dataset  ", font=("Segoe UI", 11, "bold"),
                       bg=ModernStyle.ACCENT, fg=ModernStyle.TEXT_LIGHT, relief=tk.FLAT,
                       padx=25, pady=10, cursor="hand2", command=self.tab_load_data)
        btn.pack()
        btn.bind("<Enter>", lambda e: btn.config(bg=ModernStyle.ACCENT_HOVER))
        btn.bind("<Leave>", lambda e: btn.config(bg=ModernStyle.ACCENT))

    def set_status(self, m): self.status_var.set(m); self.update_idletasks()
    def _update_doc(self):
        self.doc_var.set(f"📄 {self.bib.n:,} documents" if self.bib else "")
        
    def _check(self):
        if not self.bib:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return False
        return True
        
    def _tab(self, title, bg=None):
        bg = bg or ModernStyle.BG_CONTENT
        tab = tk.Frame(self.notebook, bg=bg)
        self.notebook.add(tab, text=f"  {title}  ")
        self.notebook.select(tab)
        
        header = tk.Frame(tab, bg=bg)
        header.pack(fill=tk.X, padx=15, pady=(12, 5))
        
        close_btn = tk.Button(header, text="✕", font=ModernStyle.FONT_BODY,
                             bg=bg, fg=ModernStyle.TEXT_MUTED, relief=tk.FLAT,
                             cursor="hand2", command=lambda: self._close(tab))
        close_btn.pack(side=tk.RIGHT)
        close_btn.bind("<Enter>", lambda e: close_btn.config(fg=ModernStyle.DANGER))
        close_btn.bind("<Leave>", lambda e: close_btn.config(fg=ModernStyle.TEXT_MUTED))
        
        container = tk.Frame(tab, bg=bg)
        container.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(container, bg=bg, highlightthickness=0)
        sb = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        content = tk.Frame(canvas, bg=bg)
        
        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        
        def _mw(e): canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _mw))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        
        return content, tab
        
    def _close(self, tab):
        self.notebook.forget(tab)
        tab.destroy()

    def _card(self, parent, title=""):
        bg = parent.cget('bg')
        outer = tk.Frame(parent, bg=bg)
        outer.pack(fill=tk.X, padx=15, pady=8)
        
        card = tk.Frame(outer, bg=ModernStyle.BG_CARD, highlightbackground=ModernStyle.BORDER, highlightthickness=1)
        card.pack(fill=tk.X)
        
        if title:
            tk.Label(card, text=title, font=ModernStyle.FONT_HEADING, bg=ModernStyle.BG_CARD,
                    fg=ModernStyle.TEXT_DARK, pady=8, padx=12).pack(anchor=tk.W)
            ttk.Separator(card, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12)
        
        inner = tk.Frame(card, bg=ModernStyle.BG_CARD, padx=12, pady=12)
        inner.pack(fill=tk.X)
        return inner
        
    def _title(self, p, t, sub=""):
        bg = p.cget('bg')
        f = tk.Frame(p, bg=bg)
        f.pack(fill=tk.X, padx=15, pady=(8, 12))
        tk.Label(f, text=t, font=ModernStyle.FONT_TITLE, bg=bg, fg=ModernStyle.TEXT_DARK).pack(anchor=tk.W)
        if sub:
            tk.Label(f, text=sub, font=ModernStyle.FONT_SUBTITLE, bg=bg, fg=ModernStyle.TEXT_MUTED).pack(anchor=tk.W, pady=(2, 0))
        
    def _opt(self, p, label, wtype, vals=None, default="", tip="", w=22):
        bg = p.cget('bg')
        row = tk.Frame(p, bg=bg)
        row.pack(fill=tk.X, pady=5)
        
        lbl = tk.Label(row, text=label, bg=bg, font=ModernStyle.FONT_BODY, 
                      fg=ModernStyle.TEXT_DARK, width=20, anchor=tk.W)
        lbl.pack(side=tk.LEFT)
        
        var = tk.StringVar(value=default)
        if wtype == "combo":
            widget = ttk.Combobox(row, textvariable=var, values=list(vals) if vals else [], 
                                 state="readonly", width=w, font=ModernStyle.FONT_BODY)
            if vals: widget.current(0)
        else:
            widget = tk.Entry(row, textvariable=var, width=w, font=ModernStyle.FONT_BODY,
                            relief=tk.FLAT, bg="#fff", highlightbackground=ModernStyle.BORDER, highlightthickness=1)
        widget.pack(side=tk.LEFT, padx=5)
        
        if tip: ToolTip(lbl, tip)
        return var
        
    def _check_opt(self, p, label, default=False, tip=""):
        bg = p.cget('bg')
        row = tk.Frame(p, bg=bg)
        row.pack(fill=tk.X, pady=3)
        var = tk.BooleanVar(value=default)
        cb = tk.Checkbutton(row, text=label, variable=var, bg=bg, font=ModernStyle.FONT_BODY,
                           fg=ModernStyle.TEXT_DARK, activebackground=bg, selectcolor=ModernStyle.BG_CARD)
        cb.pack(side=tk.LEFT)
        if tip: ToolTip(cb, tip)
        return var
        
    def _more(self, p, title="Advanced Options"):
        bg = p.cget('bg')
        container = tk.Frame(p, bg=bg)
        container.pack(fill=tk.X, pady=6)
        
        inner = tk.Frame(container, bg=bg, padx=12)
        exp = tk.BooleanVar(value=False)
        
        def toggle():
            if exp.get():
                inner.pack_forget()
                btn.config(text=f"▸ {title}")
                exp.set(False)
            else:
                inner.pack(fill=tk.X, pady=(6, 0))
                btn.config(text=f"▾ {title}")
                exp.set(True)
        
        btn = tk.Button(container, text=f"▸ {title}", command=toggle, bg=bg, relief=tk.FLAT,
                       font=ModernStyle.FONT_BODY, fg=ModernStyle.ACCENT, cursor="hand2", anchor=tk.W)
        btn.pack(fill=tk.X)
        return inner
        
    def _action_btn(self, p, text, cmd):
        bg = p.cget('bg')
        f = tk.Frame(p, bg=bg)
        f.pack(pady=12)
        
        btn = tk.Button(f, text=f"  ▶  {text}  ", font=("Segoe UI", 10, "bold"),
                       bg=ModernStyle.ACCENT, fg=ModernStyle.TEXT_LIGHT, relief=tk.FLAT,
                       padx=20, pady=8, cursor="hand2", command=cmd)
        btn.pack()
        btn.bind("<Enter>", lambda e: btn.config(bg=ModernStyle.ACCENT_HOVER))
        btn.bind("<Leave>", lambda e: btn.config(bg=ModernStyle.ACCENT))
        return f
        
    def _results_card(self, p, title="Results"):
        bg = p.cget('bg')
        outer = tk.Frame(p, bg=bg)
        outer.pack(fill=tk.BOTH, expand=True, padx=15, pady=8)
        
        card = tk.Frame(outer, bg=ModernStyle.BG_CARD, highlightbackground=ModernStyle.BORDER, highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=True)
        
        header = tk.Frame(card, bg=ModernStyle.BG_CARD)
        header.pack(fill=tk.X, padx=12, pady=(10, 0))
        tk.Label(header, text=title, font=ModernStyle.FONT_HEADING, bg=ModernStyle.BG_CARD,
                fg=ModernStyle.TEXT_DARK).pack(side=tk.LEFT)
        
        ttk.Separator(card, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12, pady=(8, 0))
        
        inner = tk.Frame(card, bg=ModernStyle.BG_CARD)
        inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        return inner
        
    def _show_table(self, p, df, maxr=100):
        """Display DataFrame with WORKING horizontal scrollbar"""
        for w in p.winfo_children(): w.destroy()
        
        if df is None or df.empty:
            tk.Label(p, text="No data available", bg=ModernStyle.BG_CARD, 
                    fg=ModernStyle.TEXT_MUTED, font=ModernStyle.FONT_BODY).pack(pady=40)
            return
        
        # Main container
        main_frame = tk.Frame(p, bg=ModernStyle.BG_CARD)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Create canvas for horizontal scrolling
        h_canvas = tk.Canvas(main_frame, bg=ModernStyle.BG_CARD, highlightthickness=0)
        h_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=h_canvas.xview)
        v_scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=h_canvas.yview)
        
        # Frame inside canvas
        tree_frame = tk.Frame(h_canvas, bg=ModernStyle.BG_CARD)
        
        # Configure canvas
        h_canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Pack scrollbars and canvas
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create window in canvas
        canvas_window = h_canvas.create_window((0, 0), window=tree_frame, anchor="nw")
        
        # Treeview
        cols = list(df.columns)
        tree = ttk.Treeview(tree_frame, columns=cols, show='headings', height=min(20, len(df)))
        tree.pack(fill=tk.BOTH, expand=True)
        
        # Calculate total width needed
        total_width = 0
        for col in cols:
            col_data = df[col].astype(str)
            max_len = max(len(str(col)), col_data.str.len().max() if len(df) > 0 else 10)
            col_width = min(max(90, max_len * 8), 350)
            tree.heading(col, text=str(col))
            tree.column(col, width=col_width, minwidth=70, stretch=False)
            total_width += col_width
        
        # Add data
        for _, row in df.head(maxr).iterrows():
            tree.insert("", tk.END, values=[str(v)[:60] if v is not None else "" for v in row.values])
        
        # Update canvas scroll region when tree_frame changes
        def configure_scroll(event):
            h_canvas.configure(scrollregion=h_canvas.bbox("all"))
            # Set minimum width for the canvas window
            h_canvas.itemconfig(canvas_window, width=max(total_width, h_canvas.winfo_width()))
        
        tree_frame.bind("<Configure>", configure_scroll)
        
        # Mouse wheel for horizontal scroll (Shift + scroll)
        def h_scroll(event):
            h_canvas.xview_scroll(int(-1*(event.delta/120)), "units")
        
        def v_scroll(event):
            h_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        h_canvas.bind("<Shift-MouseWheel>", h_scroll)
        h_canvas.bind("<MouseWheel>", v_scroll)
        
        # Info
        info = tk.Label(p, text=f"Showing {min(len(df), maxr):,} of {len(df):,} rows  |  Use Shift+Scroll for horizontal", 
                       bg=ModernStyle.BG_CARD, fg=ModernStyle.TEXT_MUTED, font=ModernStyle.FONT_SMALL)
        info.pack(pady=(0, 8))
        
    def _show_plot(self, p, fig):
        for w in p.winfo_children(): w.destroy()
        
        canvas = FigureCanvasTkAgg(fig, master=p)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        toolbar_frame = tk.Frame(p, bg=ModernStyle.BG_CARD)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(canvas, toolbar_frame).update()
        
    def _show_text(self, p, text):
        for w in p.winfo_children(): w.destroy()
        
        # Use Text widget with scrollbar
        frame = tk.Frame(p, bg=ModernStyle.BG_CARD)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        v_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        h_scroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)
        
        txt = tk.Text(frame, font=ModernStyle.FONT_MONO, wrap=tk.NONE, relief=tk.FLAT,
                     bg=ModernStyle.BG_CARD, fg=ModernStyle.TEXT_DARK, padx=12, pady=12,
                     yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        v_scroll.config(command=txt.yview)
        h_scroll.config(command=txt.xview)
        
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        txt.pack(fill=tk.BOTH, expand=True)
        
        txt.insert(tk.END, text)
        txt.config(state=tk.DISABLED)

    # =========================================================================
    # TAB: LOAD DATA
    # =========================================================================
    def tab_load_data(self):
        c, tab = self._tab("Load Dataset", ModernStyle.BG_INPUT)
        self._title(c, "Load Bibliometric Dataset", "Import data from OpenAlex, Scopus, Web of Science, or PubMed")
        
        file_card = self._card(c, "📂 Select File")
        ff = tk.Frame(file_card, bg=ModernStyle.BG_CARD)
        ff.pack(fill=tk.X, pady=3)
        
        file_var = tk.StringVar(value="No file selected")
        tk.Label(ff, textvariable=file_var, bg="#f8fafc", fg=ModernStyle.TEXT_DARK,
                font=ModernStyle.FONT_BODY, width=38, anchor=tk.W, padx=8, pady=6).pack(side=tk.LEFT, padx=(0, 8))
        
        def browse():
            p = filedialog.askopenfilename(filetypes=[("All Supported", "*.csv *.xlsx *.xls *.txt"), ("All", "*.*")])
            if p: self.dataset_path = p; file_var.set(os.path.basename(p))
            
        tk.Button(ff, text="Browse...", font=ModernStyle.FONT_BODY, bg=ModernStyle.BG_CARD, 
                 fg=ModernStyle.ACCENT, relief=tk.FLAT, cursor="hand2", padx=12, command=browse).pack(side=tk.LEFT)
        
        settings_card = self._card(c, "⚙️ Settings")
        db_var = self._opt(settings_card, "Data Source", "combo", list(DB_MAP.keys()))
        pp_var = self._opt(settings_card, "Preprocessing Level", "combo", list(PREPROCESS_MAP.keys()))
        
        adv = self._more(settings_card, "Advanced Options")
        res_var = self._opt(adv, "Results Folder", "entry", default="results")
        kw_var = self._opt(adv, "Default Keywords", "combo", list(KEYWORD_MAP.keys()))
        lang_var = self._opt(adv, "Document Language", "combo", list(LANG_MAP.keys()))
        out_var = self._opt(adv, "Output Language", "combo", list(LANG_MAP.keys()))
        lemma_var = self._check_opt(adv, "Lemmatize keywords")
        combine_var = self._check_opt(adv, "Combine author & index keywords")
        label_var = self._check_opt(adv, "Add document labels", True)
        
        vis = self._more(settings_card, "Visualization")
        cmap_var = self._opt(vis, "Color Palette", "combo", CMAP_LIST)
        dpi_var = self._opt(vis, "Plot DPI", "entry", default="600")
        
        kwp = self._more(settings_card, "Keyword Processing")
        tk.Label(kwp, text="Keywords to Remove (one per line):", bg=ModernStyle.BG_CARD,
                font=ModernStyle.FONT_BODY, fg=ModernStyle.TEXT_DARK).pack(anchor=tk.W, pady=(3,2))
        excl_txt = tk.Text(kwp, height=3, width=42, font=ModernStyle.FONT_BODY, relief=tk.FLAT,
                          highlightbackground=ModernStyle.BORDER, highlightthickness=1)
        excl_txt.pack(anchor=tk.W, pady=2)
        
        tk.Label(kwp, text="Synonyms (term1=term2):", bg=ModernStyle.BG_CARD,
                font=ModernStyle.FONT_BODY, fg=ModernStyle.TEXT_DARK).pack(anchor=tk.W, pady=(8,2))
        syn_txt = tk.Text(kwp, height=3, width=42, font=ModernStyle.FONT_BODY, relief=tk.FLAT,
                         highlightbackground=ModernStyle.BORDER, highlightthickness=1)
        syn_txt.pack(anchor=tk.W, pady=2)
        
        btn_card = self._card(c)
        bf = tk.Frame(btn_card, bg=ModernStyle.BG_CARD)
        bf.pack(fill=tk.X)
        
        spinner = LoadingSpinner(bf, bg=ModernStyle.BG_CARD)
        spinner.pack(side=tk.LEFT, padx=(0, 12))
        stat_var = tk.StringVar(value="")
        
        def load():
            if not self.dataset_path:
                messagebox.showwarning("No File", "Please select a file first.")
                return
            spinner.start()
            stat_var.set("Loading...")
            self.set_status("Loading dataset...")
            
            def do_load():
                try:
                    kw = {
                        'f_name': self.dataset_path,
                        'db': DB_MAP[db_var.get()],
                        'preprocess_level': PREPROCESS_MAP[pp_var.get()],
                        'res_folder': res_var.get().strip() or "results",
                        'default_keywords': KEYWORD_MAP[kw_var.get()],
                        'lang_of_docs': LANG_MAP[lang_var.get()],
                        'output_lang': LANG_MAP[out_var.get()],
                        'lemmatize_kw': lemma_var.get(),
                        'combine_with_index_keywords': combine_var.get(),
                        'label_docs': label_var.get(),
                        'cmap': cmap_var.get(),
                        'dpi': int(dpi_var.get()),
                    }
                    excl = excl_txt.get("1.0", tk.END).strip()
                    if excl: kw['exclude_list_kw'] = [x.strip() for x in excl.split('\n') if x.strip()]
                    syn = syn_txt.get("1.0", tk.END).strip()
                    if syn:
                        sd = {}
                        for line in syn.split('\n'):
                            if '=' in line: k, v = line.split('=', 1); sd[k.strip()] = v.strip()
                        if sd: kw['synonyms_kw'] = sd
                    
                    try: self.bib = BiblioPlot(**kw)
                    except: self.bib = BiblioAnalysis(**kw)
                    self.original_df = self.bib.df.copy()
                    self.after(0, lambda: done(True, f"Loaded {self.bib.n:,} documents"))
                except Exception as e:
                    traceback.print_exc()
                    self.after(0, lambda: done(False, str(e)))
            
            def done(ok, msg):
                spinner.stop()
                self._update_doc()
                if ok:
                    stat_var.set(f"✓ {msg}")
                    self.set_status(msg)
                    messagebox.showinfo("Success", msg)
                else:
                    stat_var.set("✗ Failed")
                    messagebox.showerror("Error", msg)
            
            threading.Thread(target=do_load, daemon=True).start()
        
        load_btn = tk.Button(bf, text="  📂  Load Dataset  ", font=("Segoe UI", 10, "bold"),
                            bg=ModernStyle.SUCCESS, fg=ModernStyle.TEXT_LIGHT, relief=tk.FLAT,
                            padx=18, pady=8, cursor="hand2", command=load)
        load_btn.pack(side=tk.LEFT)
        load_btn.bind("<Enter>", lambda e: load_btn.config(bg="#059669"))
        load_btn.bind("<Leave>", lambda e: load_btn.config(bg=ModernStyle.SUCCESS))
        
        tk.Label(bf, textvariable=stat_var, bg=ModernStyle.BG_CARD, fg=ModernStyle.TEXT_MUTED,
                font=ModernStyle.FONT_BODY).pack(side=tk.LEFT, padx=15)

    # =========================================================================
    # TAB: DATA TABLE
    # =========================================================================
    def tab_data_table(self):
        if not self._check(): return
        c, tab = self._tab("Data Table")
        self._title(c, "Dataset Table", f"{self.bib.n:,} documents loaded")
        res = self._results_card(c, "Data Preview")
        self._show_table(res, self.bib.df, 500)

    # =========================================================================
    # TAB: FILTER
    # =========================================================================
    def tab_filter(self):
        if not self._check(): return
        c, tab = self._tab("Filter", ModernStyle.BG_INPUT)
        self._title(c, "Filter Dataset", f"Current: {len(self.bib.df):,} | Original: {len(self.original_df):,}")
        
        rules_card = self._card(c, "🔍 Filter Rules")
        ri = tk.Frame(rules_card, bg=ModernStyle.BG_CARD)
        ri.pack(fill=tk.X)
        
        rows = []
        cols = list(self.bib.df.columns)
        
        def add_rule():
            r = tk.Frame(ri, bg=ModernStyle.BG_CARD)
            r.pack(fill=tk.X, pady=3)
            cv = tk.StringVar()
            ttk.Combobox(r, textvariable=cv, values=cols, width=20, font=ModernStyle.FONT_BODY).pack(side=tk.LEFT, padx=2)
            ov = tk.StringVar(value="contains")
            ttk.Combobox(r, textvariable=ov, values=["contains","equals","not contains",">","<",">=","<="], 
                        state="readonly", width=11, font=ModernStyle.FONT_BODY).pack(side=tk.LEFT, padx=2)
            vv = tk.StringVar()
            tk.Entry(r, textvariable=vv, width=18, font=ModernStyle.FONT_BODY, relief=tk.FLAT,
                    highlightbackground=ModernStyle.BORDER, highlightthickness=1).pack(side=tk.LEFT, padx=2)
            rule = {"col": cv, "op": ov, "val": vv, "frame": r}
            rows.append(rule)
            tk.Button(r, text="✕", font=ModernStyle.FONT_SMALL, bg=ModernStyle.BG_CARD,
                     fg=ModernStyle.DANGER, relief=tk.FLAT, cursor="hand2",
                     command=lambda: (r.destroy(), rows.remove(rule))).pack(side=tk.LEFT, padx=6)
        
        tk.Button(rules_card, text="+ Add Rule", font=ModernStyle.FONT_BODY, bg=ModernStyle.BG_CARD, 
                 fg=ModernStyle.ACCENT, relief=tk.FLAT, cursor="hand2", command=add_rule).pack(anchor=tk.W, pady=(8, 3))
        add_rule()
        
        btn_card = self._card(c)
        bf = tk.Frame(btn_card, bg=ModernStyle.BG_CARD)
        bf.pack(fill=tk.X)
        stat = tk.StringVar(value="")
        
        def apply_f():
            try:
                self.bib.df = self.original_df.copy()
                for r in rows:
                    col, op, val = r["col"].get(), r["op"].get(), r["val"].get()
                    if not col or not val or col not in self.bib.df.columns: continue
                    s = self.bib.df[col]
                    if op == "contains": m = s.astype(str).str.contains(val, case=False, na=False)
                    elif op == "equals": m = s.astype(str) == val
                    elif op == "not contains": m = ~s.astype(str).str.contains(val, case=False, na=False)
                    elif op in [">","<",">=","<="]:
                        ns = pd.to_numeric(s, errors='coerce'); nv = float(val)
                        m = eval(f"ns {op} nv")
                    else: continue
                    self.bib.df = self.bib.df[m]
                self.bib.n = len(self.bib.df)
                self._update_doc()
                stat.set(f"✓ Filtered to {self.bib.n:,} documents")
            except Exception as e:
                stat.set(f"✗ {e}")
        
        def reset_f():
            self.bib.df = self.original_df.copy()
            self.bib.n = len(self.bib.df)
            self._update_doc()
            stat.set(f"✓ Reset to {self.bib.n:,} documents")
        
        tk.Button(bf, text="Apply Filters", font=ModernStyle.FONT_BODY, bg=ModernStyle.SUCCESS, 
                 fg=ModernStyle.TEXT_LIGHT, relief=tk.FLAT, padx=16, pady=6, cursor="hand2", command=apply_f).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(bf, text="Reset", font=ModernStyle.FONT_BODY, bg=ModernStyle.WARNING, 
                 fg=ModernStyle.TEXT_DARK, relief=tk.FLAT, padx=16, pady=6, cursor="hand2", command=reset_f).pack(side=tk.LEFT)
        tk.Label(bf, textvariable=stat, bg=ModernStyle.BG_CARD, fg=ModernStyle.TEXT_MUTED, font=ModernStyle.FONT_BODY).pack(side=tk.LEFT, padx=15)

    # =========================================================================
    # TAB: MAIN INFO (FIXED)
    # =========================================================================
    def tab_main_info(self):
        if not self._check(): return
        c, tab = self._tab("Overview", ModernStyle.BG_INPUT)
        self._title(c, "Dataset Overview", "Summary statistics and metrics")
        
        settings_card = self._card(c, "⚙️ Options")
        desc_var = self._check_opt(settings_card, "Descriptive statistics", True)
        perf_var = self._check_opt(settings_card, "Performance indicators", True)
        ts_var = self._check_opt(settings_card, "Time series analysis", True)
        extra_var = self._check_opt(settings_card, "Extra statistics", False)
        
        adv = self._more(settings_card, "Advanced")
        perf_mode = self._opt(adv, "Performance Mode", "combo", ["full", "basic", "extended"])
        excl_last = self._check_opt(adv, "Exclude last year for growth", True)
        
        res = self._results_card(c, "Analysis Results")
        
        def run():
            for w in res.winfo_children(): w.destroy()
            self.set_status("Computing overview...")
            
            # Show loading indicator
            loading_lbl = tk.Label(res, text="⏳ Computing analysis...", bg=ModernStyle.BG_CARD,
                                   fg=ModernStyle.ACCENT, font=ModernStyle.FONT_BODY)
            loading_lbl.pack(pady=30)
            self.update_idletasks()
            
            try:
                include = []
                if desc_var.get(): include.append("descriptives")
                if perf_var.get(): include.append("performance")
                if ts_var.get(): include.append("time series")
                
                # Call get_main_info
                self.bib.get_main_info(
                    include=include,
                    performance_mode=perf_mode.get(),
                    extra_stats=extra_var.get(),
                    exclude_last_year_for_growth=excl_last.get()
                )
                
                # Remove loading indicator
                loading_lbl.destroy()
                
                # Build output text
                txt = "\n"
                txt += "═" * 70 + "\n"
                txt += "                        DATASET OVERVIEW\n"
                txt += "═" * 70 + "\n\n"
                
                # Descriptives
                if hasattr(self.bib, 'descriptives_df') and self.bib.descriptives_df is not None and len(self.bib.descriptives_df) > 0:
                    txt += "─── DESCRIPTIVE STATISTICS " + "─" * 43 + "\n\n"
                    df = self.bib.descriptives_df
                    for _, row in df.iterrows():
                        vals = [str(v) for v in row.values if pd.notna(v) and str(v).strip()]
                        if len(vals) >= 2:
                            txt += f"  {vals[0]:48s} {vals[1]}\n"
                        elif len(vals) == 1:
                            txt += f"  {vals[0]}\n"
                    txt += "\n"
                
                # Performance
                if hasattr(self.bib, 'performances_df') and self.bib.performances_df is not None and len(self.bib.performances_df) > 0:
                    txt += "─── PERFORMANCE INDICATORS " + "─" * 43 + "\n\n"
                    df = self.bib.performances_df
                    for _, row in df.iterrows():
                        vals = [str(v) for v in row.values if pd.notna(v) and str(v).strip()]
                        if len(vals) >= 2:
                            # Usually: Variable, Indicator, Value
                            name = vals[-2] if len(vals) >= 2 else vals[0]
                            value = vals[-1]
                            txt += f"  {name:48s} {value}\n"
                    txt += "\n"
                
                # Time series
                if hasattr(self.bib, 'time_series_stats_df') and self.bib.time_series_stats_df is not None and len(self.bib.time_series_stats_df) > 0:
                    txt += "─── TIME SERIES ANALYSIS " + "─" * 45 + "\n\n"
                    df = self.bib.time_series_stats_df
                    for _, row in df.iterrows():
                        vals = [str(v) for v in row.values if pd.notna(v) and str(v).strip()]
                        if len(vals) >= 2:
                            txt += f"  {vals[0]:48s} {vals[1]}\n"
                    txt += "\n"
                
                txt += "═" * 70 + "\n"
                
                # Display the text
                self._show_text(res, txt)
                self.set_status("Overview completed")
                
            except Exception as e:
                traceback.print_exc()
                loading_lbl.destroy()
                tk.Label(res, text=f"Error: {e}", fg=ModernStyle.DANGER, 
                        bg=ModernStyle.BG_CARD, font=ModernStyle.FONT_BODY, wraplength=500).pack(pady=30)
                self.set_status("Error")
        
        self._action_btn(c, "Compute Overview", run)

    # =========================================================================
    # TAB: PRODUCTION
    # =========================================================================
    def tab_production(self):
        if not self._check(): return
        c, tab = self._tab("Production", ModernStyle.BG_INPUT)
        self._title(c, "Annual Scientific Production", "Documents published per year")
        
        settings_card = self._card(c, "⚙️ Options")
        rel_var = self._check_opt(settings_card, "Include relative counts", True)
        cum_var = self._check_opt(settings_card, "Include cumulative counts", True)
        pred_var = self._check_opt(settings_card, "Predict last year", False)
        
        res = self._results_card(c, "Production Chart")
        
        def run():
            for w in res.winfo_children(): w.destroy()
            self.set_status("Generating chart...")
            
            try:
                self.bib.get_production(relative_counts=rel_var.get(), cumulative=cum_var.get(), predict_last_year=pred_var.get())
                df = self.bib.production_df
                if df is not None and not df.empty and HAS_MATPLOTLIB:
                    fig = Figure(figsize=(11, 5), facecolor='white')
                    ax = fig.add_subplot(111)
                    ax.bar(df.iloc[:, 0], df.iloc[:, 1], color=ModernStyle.ACCENT, edgecolor='white', linewidth=0.5)
                    ax.set_xlabel("Year", fontsize=10)
                    ax.set_ylabel("Documents", fontsize=10)
                    ax.set_title("Annual Scientific Production", fontsize=12, fontweight='bold')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_facecolor('white')
                    fig.tight_layout()
                    self._show_plot(res, fig)
                self.set_status("Chart generated")
            except Exception as e:
                traceback.print_exc()
                tk.Label(res, text=f"Error: {e}", fg=ModernStyle.DANGER, bg=ModernStyle.BG_CARD).pack(pady=30)
        
        self._action_btn(c, "Generate Chart", run)

    # =========================================================================
    # TAB: TOP ITEMS (No pie, added scatter and more)
    # =========================================================================
    def tab_top_items(self):
        if not self._check(): return
        c, tab = self._tab("Top Items", ModernStyle.BG_INPUT)
        self._title(c, "Top Items Analysis", "Most frequent sources, authors, keywords...")
        
        items_map = {"Sources (Journals)": "sources", "Authors": "authors", "Author Keywords": "author_keywords",
                    "Index Keywords": "index_keywords", "Affiliations": "affiliations", "Countries (CA)": "ca_countries",
                    "All Countries": "all_countries", "References": "references", "Document Types": "document_types"}
        
        settings_card = self._card(c, "⚙️ Options")
        item_var = self._opt(settings_card, "Analyze", "combo", list(items_map.keys()))
        topn_var = self._opt(settings_card, "Number of Items", "entry", default="10")
        plot_var = self._opt(settings_card, "Chart Type", "combo", 
                            ["Horizontal Bar", "Vertical Bar", "Lollipop", "Scatter (Rank vs Count)", "Line", "Area"])
        
        adv = self._more(settings_card, "Chart Options")
        color_var = self._opt(adv, "Color", "combo", ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"])
        grid_var = self._check_opt(adv, "Show grid", False)
        
        res = self._results_card(c, "Analysis Results")
        
        def run():
            for w in res.winfo_children(): w.destroy()
            items = items_map.get(item_var.get(), "sources")
            topn = int(topn_var.get())
            ptype = plot_var.get()
            color = color_var.get()
            
            self.set_status(f"Analyzing {items}...")
            
            try:
                m = f"count_{items}"
                if hasattr(self.bib, m): getattr(self.bib, m)(top_n=topn)
                df = getattr(self.bib, f"{items}_counts_df", None)
                
                if df is not None and not df.empty and HAS_MATPLOTLIB:
                    ic = df.columns[0]
                    cc = [x for x in df.columns if 'count' in x.lower() or x == 'Count' or 'Number' in x]
                    cc = cc[0] if cc else df.columns[1]
                    d = df.head(topn)
                    
                    fig = Figure(figsize=(10, 6), facecolor='white')
                    ax = fig.add_subplot(111)
                    
                    x_vals = range(len(d))
                    y_vals = d[cc].values
                    labels = [str(x)[:35] for x in d[ic]]
                    
                    if ptype == "Horizontal Bar":
                        ax.barh(x_vals, y_vals, color=color)
                        ax.set_yticks(x_vals)
                        ax.set_yticklabels(labels)
                        ax.invert_yaxis()
                        ax.set_xlabel("Count")
                        
                    elif ptype == "Vertical Bar":
                        ax.bar(x_vals, y_vals, color=color)
                        ax.set_xticks(x_vals)
                        ax.set_xticklabels([l[:15] for l in labels], rotation=45, ha='right')
                        ax.set_ylabel("Count")
                        
                    elif ptype == "Lollipop":
                        ax.hlines(y=x_vals, xmin=0, xmax=y_vals, color=color, linewidth=2)
                        ax.scatter(y_vals, x_vals, color=color, s=80, zorder=3)
                        ax.set_yticks(x_vals)
                        ax.set_yticklabels(labels)
                        ax.invert_yaxis()
                        ax.set_xlabel("Count")
                        
                    elif ptype == "Scatter (Rank vs Count)":
                        ranks = list(range(1, len(d)+1))
                        ax.scatter(ranks, y_vals, color=color, s=100, alpha=0.7)
                        for i, txt in enumerate(labels[:10]):
                            ax.annotate(txt[:20], (ranks[i], y_vals[i]), fontsize=8, 
                                       xytext=(5, 5), textcoords='offset points')
                        ax.set_xlabel("Rank")
                        ax.set_ylabel("Count")
                        
                    elif ptype == "Line":
                        ax.plot(x_vals, y_vals, marker='o', color=color, linewidth=2, markersize=6)
                        ax.set_xticks(x_vals)
                        ax.set_xticklabels([l[:12] for l in labels], rotation=45, ha='right')
                        ax.set_ylabel("Count")
                        
                    elif ptype == "Area":
                        ax.fill_between(x_vals, y_vals, color=color, alpha=0.5)
                        ax.plot(x_vals, y_vals, color=color, linewidth=2)
                        ax.set_xticks(x_vals)
                        ax.set_xticklabels([l[:12] for l in labels], rotation=45, ha='right')
                        ax.set_ylabel("Count")
                    
                    if grid_var.get():
                        ax.grid(True, alpha=0.3)
                    
                    ax.set_title(f"Top {topn} {item_var.get()}", fontsize=12, fontweight='bold')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_facecolor('white')
                    fig.tight_layout()
                    self._show_plot(res, fig)
                    
                self.set_status("Analysis completed")
            except Exception as e:
                traceback.print_exc()
                tk.Label(res, text=f"Error: {e}", fg=ModernStyle.DANGER, bg=ModernStyle.BG_CARD).pack(pady=30)
        
        self._action_btn(c, "Run Analysis", run)

    # =========================================================================
    # TAB: STATISTICS
    # =========================================================================
    def tab_statistics(self):
        if not self._check(): return
        c, tab = self._tab("Statistics", ModernStyle.BG_INPUT)
        self._title(c, "Detailed Statistics", "Citation metrics, h-index, productivity")
        
        items_map = {"Sources": "sources", "Authors": "authors", "Author Keywords": "author_keywords", "References": "references"}
        settings_card = self._card(c, "⚙️ Options")
        item_var = self._opt(settings_card, "Entity", "combo", list(items_map.keys()))
        topn_var = self._opt(settings_card, "Number of Items", "entry", default="15")
        
        res = self._results_card(c, "Statistics Table")
        
        def run():
            for w in res.winfo_children(): w.destroy()
            items = items_map.get(item_var.get(), "sources")
            topn = int(topn_var.get())
            self.set_status(f"Computing {items} statistics...")
            
            try:
                m = f"get_{items}_stats"
                if hasattr(self.bib, m): getattr(self.bib, m)(top_n=topn)
                df = getattr(self.bib, f"{items}_stats_df", None)
                if df is not None and not df.empty:
                    self._show_table(res, df.head(topn))
                else:
                    tk.Label(res, text="No statistics available", bg=ModernStyle.BG_CARD, fg=ModernStyle.TEXT_MUTED).pack(pady=30)
                self.set_status("Statistics computed")
            except Exception as e:
                traceback.print_exc()
                tk.Label(res, text=f"Error: {e}", fg=ModernStyle.DANGER, bg=ModernStyle.BG_CARD).pack(pady=30)
        
        self._action_btn(c, "Compute Statistics", run)

    # =========================================================================
    # NETWORK TABS
    # =========================================================================
    def _tab_network(self, title, sub, get_m, plot_m):
        if not self._check(): return
        c, tab = self._tab(title, ModernStyle.BG_INPUT)
        self._title(c, title, sub)
        
        settings_card = self._card(c, "⚙️ Options")
        topn_var = self._opt(settings_card, "Number of Nodes", "entry", default="30")
        
        res = self._results_card(c, "Network Visualization")
        
        def run():
            for w in res.winfo_children(): w.destroy()
            topn = int(topn_var.get())
            self.set_status("Building network...")
            
            try:
                if hasattr(self.bib, get_m): getattr(self.bib, get_m)(top_n=topn)
                if HAS_MATPLOTLIB and hasattr(self.bib, plot_m):
                    fig = plt.figure(figsize=(11, 8), facecolor='white')
                    getattr(self.bib, plot_m)()
                    self._show_plot(res, fig)
                    plt.close(fig)
                self.set_status("Network generated")
            except Exception as e:
                traceback.print_exc()
                tk.Label(res, text=f"Error: {e}", fg=ModernStyle.DANGER, bg=ModernStyle.BG_CARD).pack(pady=30)
        
        self._action_btn(c, "Generate Network", run)
    
    def tab_keyword_net(self): self._tab_network("Keyword Network", "Co-occurrence relationships", "get_author_keyword_cooccurrence", "plot_keyword_coocurrence_network")
    def tab_coauthorship(self): self._tab_network("Co-authorship", "Author collaboration", "get_coauthorship", "plot_co_authorship_network")
    def tab_country_collab(self): self._tab_network("Country Collaboration", "International partnerships", "get_country_collaboration", "plot_country_collaboration")

    def tab_historiograph(self):
        if not self._check(): return
        c, tab = self._tab("Historiograph", ModernStyle.BG_INPUT)
        self._title(c, "Citation Historiograph", "Historical citation network")
        
        res = self._results_card(c, "Visualization")
        
        def run():
            for w in res.winfo_children(): w.destroy()
            self.set_status("Building historiograph...")
            try:
                if hasattr(self.bib, 'build_historiograph'): self.bib.build_historiograph()
                if HAS_MATPLOTLIB and hasattr(self.bib, 'plot_historiograph'):
                    fig = plt.figure(figsize=(14, 8), facecolor='white')
                    self.bib.plot_historiograph()
                    self._show_plot(res, fig)
                    plt.close(fig)
                self.set_status("Historiograph generated")
            except Exception as e:
                traceback.print_exc()
                tk.Label(res, text=f"Error: {e}", fg=ModernStyle.DANGER, bg=ModernStyle.BG_CARD).pack(pady=30)
        
        self._action_btn(c, "Build Historiograph", run)

    # =========================================================================
    # SIMPLE PLOT TABS
    # =========================================================================
    def _tab_simple(self, title, sub, method):
        if not self._check(): return
        c, tab = self._tab(title, ModernStyle.BG_INPUT)
        self._title(c, title, sub)
        
        res = self._results_card(c, "Visualization")
        
        def run():
            for w in res.winfo_children(): w.destroy()
            self.set_status(f"Generating {title}...")
            try:
                if HAS_MATPLOTLIB and hasattr(self.bib, method):
                    fig = plt.figure(figsize=(12, 8), facecolor='white')
                    getattr(self.bib, method)()
                    self._show_plot(res, fig)
                    plt.close(fig)
                self.set_status("Completed")
            except Exception as e:
                traceback.print_exc()
                tk.Label(res, text=f"Error: {e}", fg=ModernStyle.DANGER, bg=ModernStyle.BG_CARD).pack(pady=30)
        
        self._action_btn(c, "Generate", run)
    
    def tab_wordmap(self): self._tab_simple("Conceptual Map", "MCA-based keyword mapping", "plot_word_map")
    def tab_thematic(self): self._tab_simple("Thematic Map", "Strategic diagram of themes", "plot_thematic_map")
    def tab_trend(self): self._tab_simple("Trending Topics", "Topic evolution over time", "plot_trend_topics")

    # =========================================================================
    # LAW TABS
    # =========================================================================
    def tab_lotka(self):
        if not self._check(): return
        c, tab = self._tab("Lotka's Law", ModernStyle.BG_INPUT)
        self._title(c, "Lotka's Law", "Author productivity distribution")
        
        settings_card = self._card(c, "⚙️ Options")
        obs_col = self._opt(settings_card, "Observed Color", "combo", ["blue", "red", "green", "orange"])
        exp_col = self._opt(settings_card, "Expected Color", "combo", ["orange", "blue", "red", "gray"])
        
        res = self._results_card(c, "Lotka's Law Analysis")
        
        def run():
            for w in res.winfo_children(): w.destroy()
            self.set_status("Computing Lotka's Law...")
            try:
                if hasattr(self.bib, 'lotka_law'):
                    fig = plt.figure(figsize=(10, 6), facecolor='white')
                    self.bib.lotka_law(observed_color=obs_col.get(), expected_color=exp_col.get())
                    self._show_plot(res, fig)
                    plt.close(fig)
                self.set_status("Lotka's Law computed")
            except Exception as e:
                traceback.print_exc()
                tk.Label(res, text=f"Error: {e}", fg=ModernStyle.DANGER, bg=ModernStyle.BG_CARD).pack(pady=30)
        
        self._action_btn(c, "Analyze", run)

    def tab_bradford(self):
        if not self._check(): return
        c, tab = self._tab("Bradford's Law", ModernStyle.BG_INPUT)
        self._title(c, "Bradford's Law", "Journal concentration analysis")
        
        settings_card = self._card(c, "⚙️ Options")
        color_var = self._opt(settings_card, "Line Color", "combo", ["blue", "red", "green", "orange"])
        grid_var = self._check_opt(settings_card, "Show grid", False)
        
        res = self._results_card(c, "Bradford's Law Analysis")
        
        def run():
            for w in res.winfo_children(): w.destroy()
            self.set_status("Computing Bradford's Law...")
            try:
                if hasattr(self.bib, 'bradford_law'):
                    fig = plt.figure(figsize=(10, 6), facecolor='white')
                    self.bib.bradford_law(color=color_var.get(), show_grid=grid_var.get())
                    self._show_plot(res, fig)
                    plt.close(fig)
                self.set_status("Bradford's Law computed")
            except Exception as e:
                traceback.print_exc()
                tk.Label(res, text=f"Error: {e}", fg=ModernStyle.DANGER, bg=ModernStyle.BG_CARD).pack(pady=30)
        
        self._action_btn(c, "Analyze", run)


def main():
    if not HAS_BIBLIUM:
        try:
            root = tk.Tk(); root.withdraw()
            messagebox.showerror("Biblium Not Found", "Install with: pip install -e .")
        except: pass
        return
    BibliumApp().mainloop()


if __name__ == "__main__":
    main()
