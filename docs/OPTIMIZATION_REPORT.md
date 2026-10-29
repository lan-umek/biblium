# Biblium 2.16 — poročilo o optimizaciji kode

**Datum:** 30. april 2026
**Obseg:** 210 datotek `.py`, ~212.000 vrstic kode
**Orodja:** ruff 0.15.12, pyflakes, ročno ciljano popravljanje

---

## Povzetek

Pred posegom je `ruff` (s pravili F, E, W, B, UP, SIM, PERF, RUF) javil **8.290 problemov**, od tega **64 napak F821 — uporaba neobstoječih imen** (resnične `NameError` napake ob izvajanju). Po popravkih je problemov **6.976**, F821 pa **0**. Vseh 210 datotek se uspešno prevede s `python -m py_compile`.

| Metrika              | Pred | Po       | Sprememba |
|----------------------|------|----------|-----------|
| Vsi ruff problemi    | 8290 | 6976     | -1314     |
| F821 (resnični bugi) | 64   | **0**    | -64       |
| F811 (podvojeno def) | 27   | 4 / 16\* | -11       |
| F401 (neuporabljeni importi) | 1270 | 378  | -892      |
| Sintaktične napake (`py_compile`) | n/a | **0/210** | — |

\*Ostale F811 so namerne (npr. lokalni `import np` v telesu funkcije zaradi lazy loadinga).

---

## Kaj je bilo popravljeno

### 1. Resnični bugi (F821 — 64 popravkov)

Te napake bi se manifestirale šele med izvajanjem in povzročile `NameError`:

**a) `lambda: ... str(e)` v `except` bloku — 26 lokacij**
Klasičen Python 3 bug: spremenljivka izjeme `e` se izbriše ob izhodu iz `except` bloka, lambda pa se zaradi `self.after(0, ...)` izvede *kasneje* na main threadu. Rezultat: namesto pravega sporočila o napaki uporabnik vidi `NameError: name 'e' is not defined`.

```python
# PRED (bug):
self.after(0, lambda: self._on_error(str(e)))   # str(e) se sproži, ko 'e' ne obstaja več

# PO (popravek):
self.after(0, lambda msg=str(e): self._on_error(msg))  # vrednost ujamemo ob ustvarjanju lambde
```

Popravljeno v 22 GUI panelih (`gui/panels/...`) ter `bibtkinter_app.py`.

**b) Podvojeno mrtvo kodo — `for _ax in axes.flat: _` (7 lokacij)**
V addonih `comparative_analysis.py`, `dynamic_topic_models.py`, `impact_metrics.py`, `temporal_networks.py` je bila zanka, ki je iterirala po nedefiniranem `axes` in v telesu imela samo `_` (useless expression). Odstranjeno.

**c) Manjkajoči importi**
- `dashboard/generator.py` → `from collections import defaultdict`
- `utilsbib_modules/stats.py` → dodan `Any` v `typing` import
- `correlation.py`, `main_path.py`, `biblium_main.py` → dodan `TYPE_CHECKING` blok za forward reference type hints (`"matplotlib.figure.Figure"`, `"LifeCycleResult"`), kar izogne hard dependency
- `utilsbib.py` (3 lokacije): `stopwords`, `nlp` (spaCy), `spla` (`scipy.sparse.linalg`) — zdaj se nalagajo lazy znotraj funkcije in ne kot globalna zahteva

**d) `cmap` undefined v plot funkcijah (5 lokacij)**
V `addons/altmetrics.py`, `dynamic_topic_models.py`, `impact_metrics.py`, `temporal_networks.py` se je sklicevalo na lokalni `cmap`, ki ni nikjer definiran. Zamenjano z `CMAP_CONTINUOUS` (modulski default), oziroma z `plt.get_cmap('tab10')` v `temporal_networks.create_animated_network` (community colors).

**e) Lokalne spremenljivke s prefiksi `_` (typo)**
- `concept_extraction_panel.py` — `_self._canvas_widget` → `self._canvas_widget`, odstranjen sklic na `canvas_widget` (neobstoječ)
- `correlation_panel.py`, `main_path_panel.py` — enako popravljeno (`_canvas_widget` / nedefiniran `canvas_widget` združeno v en `canvas_widget`)
- `concept_builder_panel.py` — odstranjena 4 vrstice mrtve kode v `_export_filtered_data`, ki so se sklicevale na `summary_df` in `data_df` (kopirane iz `_on_success`, kjer ti spremenljivki obstajata kot parametra)

### 2. Avtomatsko očiščeno z `ruff --fix` (1164 popravkov)

- Odstranjeni neuporabljeni importi (~900 vrstic)
- Odstranjeni podvojeni importi znotraj funkcij (PEP 8: F811)
- Posodobljene type annotations (PEP 604: `Optional[X]` → `X | None`, deprecated forward refs)
- Odstranjene odvečne `()`, redundantni `open(file, "r")`, ipd.

### 3. Ročno odstranjena podvojena koda (72 vrstic)

V addonih je bilo 72 zaporednih podvojenih klicev `ax.grid(False)` — popravljeno v 13 datotekah (`addons/text_mining_nlp.py` jih je imel kar 16, `addons/comparative_analysis.py` 10).

---

## Kar **ostaja** za naprej (priporočila)

Sledeče so še vedno v kodi, a niso nujni bugi — zahtevajo več konteksta in/ali oceno. Predlogi po prioriteti:

### Visoka prioriteta

1. **F811 — `plotting/interface.py:657` `doctypes_bar` redefiniran** — funkcija je definirana dvakrat v isti datoteki; prva definicija je izgubljena. Treba pogledati, katera je pravilna.
2. **F811 — `utilsbib.py:16793` `build_links_from_matrix`** — enako, funkcija definirana dvakrat. Preverite, ali sta implementaciji enaki ali se razlikujeta.
3. **F841 (140) — neuporabljene spremenljivke** — pri nekaterih je morda izgubljen `return` ali `yield`. Vredno pregledati ročno.
4. **B006 (25) — mutable default arguments** (`def f(x=[])`) — klasičen bug, ki povzroča deljenje stanja med klici. Treba zamenjati z `None` in inicializirati znotraj funkcije.
5. **B028 (38) — `warnings.warn` brez `stacklevel`** — uporabnik ne vidi, od kod opozorilo prihaja. Dodati `stacklevel=2`.
6. **B904 (17) — `raise X` znotraj `except` brez `from e`** — izgubi se izvorna napaka v traceback-u.

### Srednja prioriteta

7. **PERF401 (86) — manual list comprehension** — ročne zanke, kjer bi `[x for x in y]` bil občutno hitrejši (do 30%).
8. **PERF203 (85) — `try/except` v zanki** — drago, zlasti pri dolgih iteracijah. Premakniti `try/except` izven zanke ali uporabiti `dict.get()` namesto `try`.
9. **RUF012 (56) — mutable class default** — slično kot B006, le na razredih.
10. **B007 (121) — neuporabljena loop control variable** — uporabite `_` namesto pravega imena.
11. **UP007 (120) — staromodne `Union[X, Y]` anotacije** — Python 3.10+ omogoča `X | Y`. Ker projekt predvidoma cilja na 3.10+, je to varno posodobiti.

### Nizka prioriteta (kozmetika)

12. **E501 / W291 / W293** — dolžina vrstic in trailing whitespace (ignorirano v audit-u, lahko z `ruff format`).
13. **F401** — še 378 neuporabljenih importov, večinoma v `__init__.py` (kjer so namerno re-exported, vendar manjka `__all__`). Predlog: dodati `__all__` v `__init__.py`.

### Strukturne opazke

- **Velikost datotek:** `utilsbib.py` ima 17.443 vrstic, `plotbib.py` 12.723, `bibstats.py` 9.177. To je **veliko** — nemogoče za maintain. Že obstajajo `utilsbib_modules/`, `bibplot_modules/` — predlog je nadaljevati ekstrakcijo v module.
- **`addons/`** — vsak addon ima ponovljen `CMAP_CONTINUOUS = "viridis"` na vrhu. Lahko se ekstrahira v `addons/base.py` ali `colors.py`.
- **GUI paneli kažejo veliko ponovljene kode** (npr. ~25 paneli imajo identičen `_handle_data_loaded`, identično strukturo `_run_analysis`). Vredno bi bilo razmisliti o nadrejenem `BasePanel` z mixini.
- **Ujema se z izvajanjem:** za pravo verifikacijo je treba zagnati testno suito (`pytest tests/`). V tem okolju manjkajo težke odvisnosti (`squarify`, `bokeh` ipd.), zato je bilo verifikacija omejena na `py_compile` + ruff statično analizo.

---

## Varnost in rollback

- Pred posegom je narejen `tar.gz` snapshot v `outputs/biblium_backup/before_fixes.tar.gz` (8.2 MB).
- OneDrive ima dodatno verzioniranje za rollback per-file.
- Nobena datoteka ni bila izbrisana; vse spremembe so editi.
- Vseh 210 `.py` datotek se prevede brez napak.
