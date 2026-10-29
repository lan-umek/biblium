# Biblium 2.18 — poročilo o čiščenju in konsolidaciji modulov

**Datum:** 5. avgust 2026
**Izhodišče:** kopija bibliuma 2.16 (511 datotek), po konsolidaciji 214 .py datotek

## 1. Čiščenje

Izbrisanih 7 `.bak` datotek in vse cache mape (`__pycache__`, `.pytest_cache`, `.ruff_cache`, `pytest-cache-files-*`), odstranjena prazna mapa `biblium/`, 34 svg ikon in `icons_preview.html` premaknjenih v `assets/` (koda jih ne referencira), dodan `.gitignore`, verzija v `__init__.py` dvignjena na 2.18.0.

## 2. Popravljeni bugi

Odstranjene podvojene definicije, pri katerih je druga ob importu tiho povozila prvo (vedenje se ne spremeni, ker je obdržana aktivna verzija): `build_links_from_matrix` v `utilsbib.py`, `doctypes_bar` in `sources_scatter` v `plotting/interface.py`. Popravljenih 25 mutable default argumentov (B006), 19 `raise` brez `from` (B904) in nedefiniran `np` v anotaciji `bibgroup.py` (TYPE_CHECKING).

## 3. Konsolidacija utilsbib_modules (kriterij: moduli samo za specializirane knjižnice)

**Izbrisani zastareli dvojniki** (vse funkcije obstajajo v novejši, bogatejši obliki v `utilsbib.py`; nihče jih ni uvažal): `core.py`, `counting.py`, `excel_io.py` (funkcija `save_plot` prenesena), `groups.py`, `text.py`.

**Integrirano v `utilsbib.py`** (brez specializiranih knjižnic): `countries.py` (nalaganje podatkov o državah + 3 unikatne funkcije; 10 zastarelih dvojnikov izbrisanih), `caching.py`, `preprocessing.py`, `time_series.py`, `permutation.py`, `firth.py`. Sekcije so označene na koncu `utilsbib.py`.

**Ostanejo moduli** (specializirane knjižnice): `stats.py` (nltk, textblob), `network.py` (igraph, community), `citations.py` in `data_quality.py` (fuzzywuzzy), `topic_modeling.py` (openai, sklearn), `plotting_helpers.py` (networkx), `cobiss_*.py` (bs4, XML).

**Preusmerjeni uvozi:** `utilsbib_modules/__init__.py` (re-exporti integriranih imen zdaj kažejo na `biblium.utilsbib`), `bibclass.py`, `bibgroup_modules/associations.py`, glavni `__init__.py`, `tests/test_firth.py`, `tests/test_permutation.py`, `tests/test_associations_perm.py`.

## 4. Verifikacija

Vseh 214 datotek se prevede (`py_compile`), ruff F821/B006/B904 čist, celoten paket se uvozi (`import biblium` → 2.18.0), testi firth/permutation/associations: 51/52 (edini padec `test_profile_ci_finite` obstaja tudi v 2.16 — razlika v verziji scipy, ne posledica konsolidacije).

## 5. Poenotenje divergentnih implementacij (stats, network, topic_modeling)

V `utilsbib.py` in modulih je bilo 30 istoimenskih funkcij z **različnimi rezultati**. Poenoteno po literaturi oz. po pravilu "kanonična je razvitejša verzija, ki jo uporablja pipeline":

**stats (19 funkcij)** — kanonične so utilsbib verzije; iz `stats.py` izbrisane in nadomeščene z uvozom iz `biblium.utilsbib` (re-exporti delujejo naprej). Ob tem popravljeni definiciji v `utilsbib.py`:

- `tapered_h_index` — obe stari verziji sta bili napačni; nova sledi Anderson, Hankin & Killworth (2008): prispevek članka na rangu j je c/(2j−1) za c ≤ j, sicer j/(2j−1)+Σ 1/(2i−1). Kontrola: popoln h-kvadrat vrne točno h.
- `pi_index` — po Vinklerju (2009): π = 1/100 citatov elitnega seta velikosti √N (stara utilsbib verzija ni delila s 100, modulska je uporabljala drugačen elitni set).
- `chi_index` (√max(rang·cit), Fenner et al. 2018) in `h2_index` (Kosmulski 2006) — utilsbib verziji sta bili pravilni; napačni modulski (povprečje h-jedra oz. h²/ΣC) izbrisani.

Pozor: rezultati `tapered_h_index` in `pi_index` se zato razlikujejo od 2.16 (prej napačni/nestandardni).

**network (10 funkcij)** — kanonične implementacije so zdaj v `utilsbib_modules/network.py`: `normalize_symmetric_matrix` združena (metode association, salton/cosine, inclusion, jaccard, row, none; n_i z diagonale, sicer vsote vrstic), `add_partitions` združena (12 metod z per-metodo kwargs + parameter `methods`, alias "greedy"; vrne dict particij in anotira G), `save_network`/`save_to_pajek` prenesena bogatejša utilsbib verzija (Pajek .net/.clu/.vec, GraphML, GEXF). Enostavnejše modulske kopije in utilsbib dvojniki izbrisani; v `utilsbib.py` so ohranjeni tanki delegacijski ovoji za 4 imena, ki jih kliče `bibstats.py`. Prilagojeno klicno mesto v `bibplot_modules/thematic.py`.

**topic_modeling (1)** — verziji funkcionalno enakovredni (preverjeno na vzorcu); obdržana utilsbib (kliče jo bibstats), modulska nadomeščena z uvozom.

**Verifikacija poenotenja:** numerične kontrole (tapered h na znanih primerih, π = 0,38 za testni vektor, ujemanje ovojev z modulom za vseh 7 normalizacij), 214/214 prevedb, celotna testna suita: 212/213 (edini padec `test_profile_ci_finite` obstaja že v 2.16 — verzija scipy).

## 6. Novosti 2.18 (drugi paket, 5. avgust 2026)

**Packaging:** `pyproject.toml` z metapodatki paketa in popisanimi odvisnostmi (jedrne + opcijske skupine plots/nlp/topic/llm/network/geo/stats/cobiss), `requirements.txt`.

**Ponovljivost:** `louvain_partition` dobil `random_state=42`; `add_partitions` dobil `seed` (poseje tudi igraph prek Python RNG). Preverjeno deterministično. LDA/NMF sta že imela random_state.

**Regresijski testi:** `tests/test_indices_regression.py` — 22 ročno izračunanih kontrol za indekse (h, g, hg, a, r, chi, h(2), tapered h s Ferrersovimi kvadrati, π po Vinklerju, gini, robni primeri). Ti testi bi ujeli napačni implementaciji iz 2.16.

**Nove metode:**
- `BiblioStats.rpys()` — prava RPYS (Marx idr. 2014): odklon od 5-letne drseče mediane + najpogostejše reference za vrhove (`rpys_top_references_df`); risanje `plotbib.plot_rpys`.
- `BiblioStats.local_citations()` — LCS/GCS: ujemanje po DOI s samodejnim preklopom na normalizirane naslove, kadar reference nimajo DOI-jev; doda stolpca "Local Citations" in "LCS/GCS".
- `BiblioStats.thematic_evolution()` — sobesedne mreže po obdobjih (kvantilna ali eksplicitna), louvain klastri s semenom, omejitev na top ključne besede; s `plot=True` nariše obstoječi Sankey (`plotbib.plot_thematic_evolution`). S tem GUI panel "thematic evolution" prvič dejansko deluje (metoda prej ni obstajala).
- `llm_utils.llm_name_clusters()` — LLM poimenovanje tematskih klastrov z varnim fallbackom brez API ključa.
- `methods_text.py` + `BiblioStats.methods_text(language="en"|"sl")` — deterministični generator "Methods" odstavka iz dejansko izvedenih analiz in parametrov, z metodološkimi referencami.

**Verifikacija:** testna suita 234/235 (edini padec je znani scipy iz 2.16); nove metode preizkušene na priloženem Scopus datasetu (RPYS vrhovi 2005/2017/2012; LCS max 8 — "Toward a definition of bibliometrics", 1987).

**Odprto za naslednjo sejo:** konsolidacija štirih plotting sistemov (plotbib.py, bibplot.py, bibplot_modules/, plotting/).
