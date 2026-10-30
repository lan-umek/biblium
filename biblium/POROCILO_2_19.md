# Biblium 2.19 — poročilo o nadgradnji

**Datum:** 29. oktober 2026
**Izhodišče:** biblium 2.18 (konsolidirana koda, 5.–7. 8. 2026) + spremembe, ki so bile po odcepitvi 2.18 narejene še v mapi 2.16.

## 1. Združitev vej 2.16 in 2.18

Mapa 2.16 je bila po 5. 8. (ko je nastala kopija 2.18) še dopolnjena; te spremembe v 2.18 niso bile prenesene. V 2.19 so združene:

| Datoteka | Vir | Kaj |
|---|---|---|
| `bibclass.py` | 2.16 (27. 8.) | `statsmodels` je opcijski: uvoz je zaščiten (Windows Smart App Control blokira prevedene razširitve), `sm.add_constant` ima ročni nadomestek. Uvoz `fit_logit` preusmerjen na `biblium.utilsbib` (2.18 konsolidacija). |
| `addons/edition_fingerprints.py` | 2.16 (27. 8.) — novo | Starostno prilagojeni »edition fingerprints« (logistična regresija po konceptih; Firth/statsmodels/sklearn). Uvoz Firtha preusmerjen na `biblium.utilsbib`. |
| `utilsbib_modules/editions.py` | 2.16 (18. 8.) — novo | Stopenjska atribucija referenc k izdajam (ISBN → podnaslov → naslov+leto → založnik+leto → okno let). Dopolnjuje `editions.py` iz 2.18, ki je ostal nespremenjen. |
| `addons/dashboard_builder.py` | **združeno** | Osnova je novejša 2.16 verzija (rekurzivno skeniranje `results/`, `section_titles`, `section_notes`, `captions`, `table_picks`, `tables_per_section`, `exclude_patterns`, vrstni red sekcij). Nanjo je dodano JPEG pomanjševanje iz 2.18: `embed_image_b64(max_px=, jpeg_quality=)` in `build_pipeline_report(max_image_px=1600, image_jpeg_quality=85)`. Parameter `max_width`/`max_image_width` (2.16, brezizgubni PNG) ostaja in se uporabi, kadar je `max_image_px=None`. |

Odstranjene odvečne datoteke v `addons/`: `_fresh_test.py`, `_test_sync.txt`, `dashboard_builder.py.fresh`, `dashboard_builder.py.test_50k`, `test_big.py`.

## 2. Novi modul `biblium/indicators.py`

Brez novih odvisnosti (numpy, pandas, matplotlib), vse funkcije deterministične (naključni procesi sprejmejo `seed`). Vsaka ima ovoj v `BiblioStats`, ki rezultat shrani kot atribut, zabeleži parametre v `_analysis_params` (za `methods_text`) in ob nastavljenem `res_folder` zapiše xlsx (in sliko pri `plot=True`).

| `BiblioStats` metoda | Funkcija v `indicators` | Vsebina | Shrani |
|---|---|---|---|
| `citation_css(by_year=False, plot=False)` | `characteristic_scores_and_scales` | Characteristic Scores and Scales (Glänzel & Schubert 1988): β1 = povprečje, β2 = povprečje nad β1, β3 = povprečje nad β2 → razredi *poorly / fairly / remarkably / outstandingly cited*; po letih z globalnim nadomestkom za majhna leta (`min_year_size`). | `df["CSS Class"]`, `css_summary_df` |
| `percentile_indicators(entity="Source title")` | `percentile_indicators` | Leidenski indikatorji po entitetah: P, TCS, MCS, MNCS (normalizacija po letu objave), P/PP(top 10 %), P/PP(top 1 %), h. Dokument je v top X %, če je njegov percentilni rang v letu strogo nad 100−X. | `percentile_indicators_df`, `percentile_indicators_by_entity[entity]` |
| `uncitedness(group=None, plot=False)` | `uncitedness` | Delež necitiranih po letu objave (in skupini, npr. `Document Type`), starost, kumulativni delež. | `uncitedness_df` |
| `indicator_ci(statistic="mean", by=None)` | `bootstrap_ci`, `bootstrap_ci_by_group` | Bootstrap IZ (percentilni ali »basic«) za mean, median, h_index, total, uncited_share ali poljubno funkcijo; po skupinah z `by`. | `indicator_ci_result` / `indicator_ci_df` |
| `reference_age(price_window=5, plot=False)` | `reference_ages`, `reference_age_profile` | Priceov indeks (delež referenc ≤ 5 let), povprečna in mediana starost referenc, citing half-life — na dokument in po letih. Negativne starosti → 0, starosti > `max_age` izločene. | stolpci v `df`, `reference_age_df`, `reference_age_stats` |
| `cited_sources(top_n=30)` | `cited_sources` | »Knowledge base«: najpogosteje citirane revije v referencah (hevristika za Scopus nize; `extractor` za druge formate), delež, kumulativa, ali korpus v reviji objavlja; HHI, delež top 10, Jaccard prekrivanje citiranih in objavnih revij, self-reliance. | `cited_sources_df`, `cited_sources_stats` |
| `concentration(entity="Authors", value_col=None, plot=False)` | `concentration`, `lorenz_curve`, `gini_coefficient`, `theil_index` | Koncentracija produkcije (ali citatov) po entitetah: Gini, Theil, HHI, delež top 10/20 %, delež entitet za polovico vsote, Lorenzova krivulja. | `concentration_df`, `concentration_stats[entity]` |
| `author_careers(min_docs=2)` | `author_careers` | Prvo/zadnje leto, akademska starost v korpusu, razpon, aktivna leta, dokumenti na aktivno leto, lokalni h, m-kvocient (Hirsch 2005), delež prvoavtorstev. | `author_careers_df` |
| `fit_heaps_law(field="Author Keywords", plot=False)` | `fit_heaps_law` | Heapsov zakon V(n) = K·n^β (log-log regresija), vrstni red po letu (vezi premešane s semenom) ali naključen. | `heaps_df`, `heaps_params` |

Risanje: `plot_lorenz_curve`, `plot_css_classes`, `plot_reference_age_profile`, `plot_heaps_law`, `plot_uncitedness` (vse sprejmejo `ax=` za sestavljene slike).

## 3. Spremljajoče spremembe

- `methods_text.py`: stavki (en/sl) in literatura za vseh devet novih analiz (Glänzel & Schubert 1988; Waltman idr. 2012; Price 1970; Gini 1921; Hirsch 2005; Heaps 1978; Efron & Tibshirani 1993).
- `enhancements.get_available_analyses`: novih devet vnosov (ime, metoda, zahtevani stolpci).
- `__init__.py`: `from biblium import indicators`; verzija **2.19.0** (tudi `pyproject.toml`, `requirements.txt`).
- `tests/test_2_16_fixes.py`: preverjanje verzije je zdaj `>= 2.16` (test je padal že v 2.18).
- `tests/test_indicators_2_19.py` (novo): 33 testov z ročno izračunanimi kontrolami (CSS pragovi na znanih primerih, MNCS in PP(top 10 %) na konstruiranem letu, Gini 0,25 za 1..4 in 0,75 za [0,0,0,1], Theil, Lorenzove točke, Priceov indeks, Scopus ekstraktor revij, m-kvocient, Heaps β=1 za sam-nove žetone, determinističnost bootstrapa) ter integracijski test ovojev in `methods_text`.

## 4. Verifikacija

- `compileall`: vse datoteke se prevedejo.
- Celotna testna suita: **279 prehaja**; 3 testi v `test_firth.py` (primerjava s `statsmodels`) v razvojnem okolju niso mogli teči, ker tam statsmodels ni na voljo — niso povezani s spremembami in jih je treba enkrat pognati lokalno.
- Nove metode preizkušene na vgrajenem Scopus datasetu (200 zapisov): CSS 143/38/11/8, Priceov indeks 47,3 %, citing half-life 6 let, najbolj citirana revija v referencah *Scientometrics* (105 omemb, 50 dokumentov), Gini revij 0,34, Heaps β = 0,91.

## 5. Opombe in odprto (prvi paket)

- `cited_sources` je hevristika za Scopusov format referenc (`Avtor, I., Naslov, Revija, vol, (leto)`); za WoS/OpenAlex podaj lasten `extractor`.
- PP(top X %) uporablja povprečni rang pri vezeh in strogo neenakost; pri množičnih vezeh (npr. vsi 0 citatov) noben dokument ni »top« — za Leidensko frakcijsko štetje vezi bi bil potreben dodaten način.
- Še vedno odprto iz 2.18: konsolidacija štirih sistemov risanja (`plotbib.py`, `bibplot.py`, `bibplot_modules/`, `plotting/`).


---

# Drugi paket 2.19 (29. 10. 2026, zvečer) — vrzeli glede na bibliometrix / CiteSpace / VOSviewer

Izhodišče je bila primerjava z bibliometrix 5.x, CiteSpace, VOSviewer, pyBibX in pyBiblioNet. Implementirano je vse iz skupin 1–3 in analiza financiranja; vsaka novost ima samostojen modul (uporaben brez `BiblioStats`), ovoj v `BiblioStats` (mixin `extensions_219.Extensions219Mixin`), stavek v `methods_text` (en/sl z literaturo), vnos v `enhancements.get_available_analyses` in **Tkinter panel**.

## 6. Novi moduli in metode

| Modul | `BiblioStats` metoda | Vsebina | Vzor |
|---|---|---|---|
| `prisma.py` | `prisma()` | PRISMA 2020 diagram poteka (`PrismaCounts.fill_derived()` izpelje manjkajoče številke, risanje z matplotlib) | bibliometrix 5.3 |
| `metadata_completion.py` | `audit_missing()`, `complete_metadata()`, `undo_complete_metadata()` | Revizija manjkajočih polj; dopolnjevanje po DOI iz OpenAlex/Crossref samo v prazne celice, s tabelo provenience (vrstica, polje, stara/nova vrednost, vir) in razveljavitvijo | bibliometrix 5.4 `completeMetadata` |
| `journal_rankings.py` | `add_journal_ranking()`, `filter_by_journal_ranking()` | Lestvice revij (SCImago CSV z `;`, JCR, ABS, ABDC …) po ISSN, nato po normaliziranem naslovu; kvartili, povzetek (MNCS po kvartilu), kvartili po letih, filter | bibliometrix 5.1/5.5 |
| `funding.py` | `funding_analysis()` | Parsanje Scopus »Funding Details« (stari in novi format), zastavice `Funded`/`Funders`, financerji (polno štetje, MNCS), trend po letih, financirani vs. nefinancirani (Mann–Whitney), so-financiranje, profil ključnih besed po financerjih | — |
| `reference_patterns.py` | `rpys_patterns()` | RPYS s centrirano ali vzvratno mediano (`Deviation %`); citatne zgodovine citiranih referenc in klasifikacija: Hot Paper, Sleeping Beauty, Constant Performer, Life Cycle, Other, Insufficient data (pravila v docstringu) | bibliometrix 5.1 |
| `network_dynamics.py` | `diachronic_networks()`, `fuzzy_clusters()`, `cluster_timeline()`, `structural_variation()` | Sobesedne mreže po obdobjih s stabilno množico vozlišč in skupno razporeditvijo + **animirani GIF**; mehka pripadnost dokumentov klastrom; CiteSpace **timeline** pogled klastrov; **structural variation** (Chen 2012: sprememba modularnosti, cluster linkage, centrality divergence na sociacijski mreži referenc) | bibliometrix 5.1/5.4, CiteSpace |
| `dual_map.py` | `dual_map()` | **Dual-map overlay** na podlagi Scopus ASJC (vgrajen `sources_data_short.xlsx`): discipline citirajočih revij ↔ discipline citiranih revij v referencah, frakcijsko/polno štetje, Bezier trakovi, pokritost, samodisciplinarnost, entropija ciljev | CiteSpace (Chen & Leydesdorff 2014) |
| `snowball.py` | `snowball()` | Citatno snežno kepanje prek OpenAlex (reference + citirajoči članki, krogi, prag povezav na semenski set, letni filter); `works_to_biblium_df` za nalaganje razširjenega korpusa; ves HTTP prek injiciranega `fetch_json` (testi hermetični) | CiteSpace cascading expansion, pyBiblioNet |
| `term_extraction.py` | `extract_terms()` | VOSviewer-slog izrazov iz naslovov/povzetkov: samostalniške fraze (približek z razbijanjem ob stopbesedah), term-document matrika, **relevance score** (KL divergenca), izbor 60 %, matrika so-pojavljanj za obstoječe mrežne funkcije, stolpec `Extracted Terms` | VOSviewer (van Eck & Waltman 2011) |

## 7. Tkinter aplikacija

- **Obnovljen uvoz GUI paketa.** V 2.16 in 2.18 `gui/panels/__init__.py` in `gui/app.py` uvažata `biblium.gui.panels.reports` (`ReportBuilderPanel`, `CustomReportPanel`) in `TopicsPanel` iz `gui.panels.mapping`, ki v mapah ne obstajata — `import biblium.gui.panels` je zato padel in `gui.app.BibliumApp` se ni mogel zagnati. Dodan je `gui/panels/reports.py` (Custom Report iz čakalne vrste »Add to report« → HTML/DOCX/XLSX; Report Builder → Excel vseh tabel, HTML nadzorna plošča, PDF, methods text) in alias `TopicsPanel` v `mapping/__init__.py`.
- **`gui/panels/simple.py` — `SimpleAnalysisPanel`:** deklarativni panel (seznam `Opt` opcij + `compute(bib, opts)` v niti → zavihki s tabelami, slikami in besedilom, kartice z metrikami, izvoz v Excel). Nov panel je dolg 40–100 vrstic.
- **22 novih panelov** (`gui/panels/v219/`), registriranih v `app.py`, v meniju *Analysis* in v stranskem meniju (sekcije INDICATORS, REFERENCES, DYNAMICS & TEXT, REVIEW & QUALITY): CSS, Percentile Indicators, Concentration, Uncitedness, Bootstrap CI, Author Careers, Heaps' Law, Funding; Reference Age, Cited Sources, RPYS & Patterns, Dual-Map, Snowball; Diachronic Networks, Cluster Timeline, Structural Variation, Fuzzy Doc Clusters, Term Extraction; PRISMA, Metadata Completion, Journal Rankings.
- **Test** `tests/test_gui_panels_2_19.py`: uvoz GUI paketa, konstrukcija `BibliumApp` in instanciranje vseh registriranih panelov (≥ 90), zaokrožen tek `SimpleAnalysisPanel`. Teče samo, kjer je Tk (Windows); brez Tk se preskoči. V razvojnem okolju brez Tk je bil izveden z nadomestnim headless Tk (vseh 93 panelov se instancira, vsi `compute()` novih panelov tečejo na demo podatkih).

## 8. Verifikacija drugega paketa

- Testna suita: **494 prehaja** (+ 3 GUI testi v headless okolju); 3 znani padci `test_firth.py` (statsmodels ni na voljo v razvojnem okolju). Novih testov: prisma 13, metadata 27, rankings 36, funding 21, reference_patterns 20, network_dynamics 22, dual_map 15, snowball 19, term_extraction 42.
- Demo Scopus (200 zapisov): financiranih 71/129 (MNCS 0,97 vs 1,01; p = 0,37); RPYS vzorci: Other 34, Life Cycle 4, Hot Paper 1; dual-map pokritost dokumentov 90 %, referenc 59 %, samodisciplinarnost 35 %; SVA 50 dokumentov v 0,3 s; ekstrakcija izrazov 382 izrazov; diakrone mreže 3 obdobja + GIF.

## 9. Opombe (drugi paket)

- `complete_metadata` in `snowball` zahtevata mrežo; v testih in GUI sta ločena (audit brez mreže; snowball javi napako brez povezave).
- `dual_map` pokriva le revije iz Scopusovega seznama ASJC; reference brez prepoznavne revije ostanejo nepreslikane (pokritost je v naslovu slike).
- Ekstrakcija izrazov uporablja približek samostalniških fraz brez POS-označevalnika (VOSviewer uporablja OpenNLP); za natančnejše fraze je mogoče podati lasten seznam stopbesed.
- `cluster_timeline` privzeto skrije klastre z manj kot 3 vozlišči (louvain na redkih mrežah ustvari veliko samostojnih vozlišč).
- Odprto: spletni vmesnik (dogovorjeno za kasneje), frakcijsko štetje vezi pri PP(top X %), dokumentacijska stran in CI.

## 10. Mape z rezultati so opcijske (29. 10. 2026, pozno zvečer)

`BiblioStats`, `BiblioBase` in `BiblioGroupAnalysis` imajo zdaj `res_folder=None` kot privzeto vrednost: brez izrecno podane mape se na disk ne zapiše nič (ovoji preverjajo `res_folder` pred vsakim `_save_table`/`_save_plot`). Mape `results/<plots,tables,reports,networks,relations>` se ustvarijo samo, kadar uporabnik `res_folder` poda (npr. `BiblioStats(..., res_folder="results")`). Enako v obeh Tkinter aplikacijah: polje »Results Folder« je privzeto prazno (prazno = brez shranjevanja), v nastavitvah (`gui/core/state.py`) je privzeta mapa prazna, API nalaganje ne shranjuje. Mimogrede popravljen `utilsbib.collaboration_index` (NaN v stolpcu avtorjev je pri pandas 3 podrl `get_main_info`).

## 11. Združevanje zapisov iz več baz (30. 10. 2026, 2.19.1)

Nov modul `merge.py` (+ `BiblioStats.from_databases`, `BiblioStats.merge_with`, panel *Data → Merge Databases*):

- **Viri:** seznam `(pot ali DataFrame, baza)`; datoteke bere `readbib.read_bibfile` (imena stolpcev so tam že preslikana v skupna), `harmonize` poenoti ločila večvrednostnih polj (OpenAlex `|` → `; `), DOI (brez `https://doi.org/`, male črke), `Year` in `Cited by` kot števila, doda `Source DB`.
- **Ključ ujemanja:** `doi` (privzeto), `title` (normaliziran naslov) ali `doi_title` (DOI, brez DOI po naslovu). Zapisi brez uporabnega ključa ostanejo ločeni (`Match type = none`).
- **Stolpci:** `union` (vsi stolpci vseh baz; za zapise iz ene baze so tuja polja prazna) ali `intersection` (samo stolpci, ki jih imajo vse baze — brez strukturnega manjka). Poročilo `column_coverage` pokaže delež izpolnjenosti po stolpcih.
- **Združevanje podvojenih:** citati = maksimum (plus `Cited by (<baza>)` po virih), ključne besede/reference = unija, povzetek = najdaljši, ostala polja iz najbolj zaupanja vredne baze (`priority`, privzeto Scopus > WoS > Dimensions > OpenAlex > Lens > PubMed). Odstranijo se tudi podvojeni zapisi znotraj iste baze.
- **Poročilo:** `n_input` po virih, `n_output`, `n_removed`, tabela kombinacij virov (`overlap`, UpSet-slog) in parna matrika prekrivanja, `plot_source_overlap`.
- **Demo:** Scopus (200) + WoS (501) + OpenAlex (613) → 1223 enoličnih (91 podvojenih; 11 v vseh treh bazah), 2 s.
- 14 novih testov (`tests/test_merge.py`); verzija dvignjena na 2.19.1 (za naslednjo objavo).

## 12. Domenski testni nabori iz OpenAlex (30. 10. 2026, 2.19.1)

Nov modul `datasets.py` z **katalogom 47 naborov** (`DATASETS`, po ~200 del): vsak vnos je opis poizvedbe na OpenAlex (iskalni niz za naslov/povzetek, filtri `type:article, has_abstract:true, language:en, publication_year`, pri revijah `primary_location.source.issn`; razvrstitev `cited_by_count:desc`). Domene: javna uprava (8: e-uprava, NPM, koprodukcija, sodelovalno upravljanje, PSM, administrativno breme, pametna mesta, slovenski avtorji), medicina (5), sociologija (3), politologija, ekonomija (3), poslovne vede, izobraževanje, psihologija, pravo (3: GDPR, regulacija UI, podnebne tožbe), računalništvo (3), matematika (3), okoljske in biološke vede, informacijska znanost; **12 revijskih naborov** (PAR, JPART, PMR, CEPAR, Scientometrics, JOI, ASR, AER, Lancet, Harvard Law Review, CACM, Annals of Mathematics).

- `fetch_works(spec)` → `flatten_work` → `.csv.gz` v **enakem stolpčnem formatu kot izvoz s spletne strani OpenAlex** (`authorships.raw_author_name`, `keywords.display_name`, `referenced_works` …, seznami ločeni z `|`, povzetek sestavljen iz obrnjenega indeksa), zato gredo datoteke skozi obstoječi `read_oa_csv` in preslikavo `variable names.xlsx` brez sprememb.
- `load_dataset(name)` → `BiblioStats(db="oa")` z atributoma `dataset_name`, `dataset_spec`; `methods_text()` doda stavek o izvoru (ime nabora, domena, iskalni niz).
- CLI: `python -m biblium.datasets --list | --fetch ime … | --fetch-all [--domain …] --email …`. Datoteke gredo v `biblium/data/datasets/` (dodano v `package-data`), zato so po prenosu vključene v paket.
- GUI: *Load Data* ima novo kartico **Domain Datasets (OpenAlex)** — izbira domene in nabora, gumb *Use dataset* (nastavi pot in bazo OpenAlex) in *Download missing…* (prenos v niti; e-pošta iz spremenljivke okolja `OPENALEX_MAILTO`).
- Iz tega okolja OpenAlex API ni dosegljiv (proxy vrne 403), zato je prenos treba zagnati na računalniku uporabnika; testi (10, `tests/test_datasets.py`) uporabljajo ponarejen JSON del in preverijo celotno pot JSON → CSV → `read_bibfile` → `BiblioStats` → `methods_text`.
- Mimogrede: pretvorba v števila v `read_oa_csv` (in drugih bralnikih) pri pandas 3 ni delovala (`dtype == "object"`), zato sta bila `Cited by`/`Year` nizi — popravljeno.

## 13. Vrstni red zadetkov pri prenosu prek API (30. 10. 2026, 2.19.1)

Dosedaj je bil vrstni red pri delnem prenosu skrit (`cited_by_count:desc` v GUI, `publication_date:desc` v `download_works`, `relevance_score:desc` v `search_works`). Zdaj je enoten in izbirljiv: `openalex_api.SORT_OPTIONS` = `most_cited` (privzeto), `least_cited`, `newest`, `oldest`, `relevance` (samo ob iskalnem nizu; sicer se vrne na citiranost), `random` (naključni vzorec z OpenAlex `sample`+`seed`, do 10.000 zapisov, ponovljiv). `resolve_sort` sprejme ključ, GUI oznako ali surov OpenAlex niz (npr. `fwci:desc`); `paging_params` izbere kurzorsko ali stransko paginacijo. Vgrajeno v `OpenAlexClient.search_works/download_works`, `search_openalex`, `download_openalex_dataset`, `datasets.fetch_works/fetch_dataset` (CLI `--sort`), v GUI pa kot polje **Order** v *API Data* (PubMed podpira le relevantnost/najnovejše, Dimensions citiranost/datum; o neizvedljivi izbiri se izpiše opozorilo). `methods_text` pri vgrajenih naborih navede izbrani vrstni red. Testi: +3 (`test_datasets.py`).
