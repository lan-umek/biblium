# Popravki bibliuma 2.18 — najdbe iz aplikacije 004_Public_Management_Reform (5. 8. 2026)

Celoten PMR-pipeline (01→25, N = 5 758) je bil ponovno izveden z bibliumom 2.18.
Spodnje hrošče je razkrila aplikacija; večina jih je obstajala že v 2.16
(v run-logih 2.16 so bili isti FAILi), popravljeni so samo v 2.18.

## 1. `main_path.py::plot_main_path` — obešanje na velikih omrežjih
- Kamada-Kawai layout se je računal na CELOTNEM citatnem omrežju (3 723 vozlišč
  → O(n²)+; korak 05 se je obesil).
- Novo: parameter `max_layout_nodes=1500`. Če je graf večji, se za PRIKAZ
  uporabi podgraf: glavna pot + njena soseščina (analitični rezultati nespremenjeni).
- Bonus: goli `except:` → `except Exception:`.

## 2. `utilsbib.py::get_combined_stopwords` — izgubljena pri konsolidaciji
- Funkcija je bila v 2.16 v `utilsbib_modules/text.py`; ob izbrisu modula ni
  bila prenesena. Zdaj je v `utilsbib.py` (prilagojena pot do `additional
  files/stopwords.xlsx`). Skripti 11 in 19 v PMR jo uvažata iz `biblium.utilsbib`.

## 3. `representation.py` — zavajajoča diagnostika in SDG-hack
- Prazna referenčna distribucija (npr. OpenAlex fetch brez mreže) zdaj takoj
  javi jasno napako, prej "No matching categories found …".
- Pred SDG-preoblikovanjem labelov se najprej poskusi dtype-agnostična
  normalizacija v niz (najpogostejši vzrok praznega merge-a: Year int vs str).
- SDG-preoblikovanje ("2005" → "SDG 2005"!) se uporabi le, če labele res
  izgledajo kot SDG.

## 4. `bibstats.py::sequential_topic_modeling` — API neusklajenost
- Dodana `v=` (zgodovinski biblium alias za `text_column`) in `n_windows=`
  (enakomerna razdelitev razpona let; ignorirano ob podanem `time_slices`).

## 5. `bibplot.py::plot_associations_bipartite_network` — DataFrame `or`
- `getattr(R,"rm",None) or getattr(R,"contingency",None)` → ValueError
  (truth value ambiguous). Zamenjano z eksplicitnima None-testoma.

## 6. `bibplot.py` — leni predpogoji za country plote
- `plot_all_countries_map` sam pokliče `count_all_countries()`;
  `plot_country_collaboration` sam pokliče `get_country_collaboration()`;
  normalizirani heatmap se izriše le, če matrika obstaja.

## 7. `bibplot.py::plot_coocurrence` — None-graf
- Shranjeni atribut omrežja, ki obstaja a je None, je prej zdrsnil do
  kriptične napake v plotbib; zdaj fallback na getter in jasna napaka.

## Dodatni popravki 6. 8. 2026 (vsebinska analiza PMR)

**8. `utilsbib.py::sequential_topic_modeling` — time_slices semantika.**
`time_slices` je po docstringu seznam MEJA intervalov, implementacija pa je
vsako mejo obravnavala kot točno leto (`Year == 2006`) in tiho izpustila vse
dokumente med mejami. Zdaj meje tvorijo intervale (zadnji vključujoč);
`period_bounds` mapira ključ obdobja (začetno leto) → (start, end).

**9. `utilsbib.py::_align_topics_across_periods` — prazna evolucija.**
Če prvo obdobje ni bilo fitano (premalo dokumentov), je funkcija vrnila prazen
DataFrame; manjkajoče vmesno obdobje je prekinilo verigo. Zdaj se poravnava
izvede nad dejansko fitanimi obdobji.

**10. `bibgroup_modules/field_dynamics.py::compute_citation_half_life_per_group`
— half-life vedno NaN.** Eksponentni razpad se je fitiral na KUMULATIVNE citate,
ki s starostjo rastejo → r < 0 za vse skupine. Zdaj se fitira na citate na leto
(Cited by / starost). Negativen r je zdaj informativen (citatni tok še narašča —
npr. Post-NPM/NPG).

## Nova funkcionalnost 6. 8. 2026

**11. `bibgroup_modules/field_dynamics.py::compute_group_citation_flows` (novo).**
Tokovi citatov skupina → skupina: vsak znotraj-korpusni citat (citing → cited)
se preslika v koncepte obeh dokumentov (frakcijsko ali polno uteževanje);
rezultat je matrika `group_citation_flow_matrix` (cited × citing), long tabela
in Sankey diagram ("iz katerih konceptov so se razvili kateri"). Parametri:
`exclude_self`, `top_n`, `min_flow`, `weighting`.

**12. `field_dynamics.py::_plot_sankey_matplotlib` (novo).** Čisti matplotlib
dvostolpčni Sankey (Bezier trakovi) kot samodejni fallback, kadar plotly PNG
izvoz ni mogoč (kaleido zahteva Chrome, ki ga ni v sandbox okoljih);
`compute_group_citation_flows` ga uporabi, če PNG po plotly poskusu ne obstaja.

**13. `field_dynamics.py::compute_group_citation_flows_dynamic` (novo).**
Dinamični koncept→koncept citatni tokovi po obdobjih: `time_slices` kot meje
intervalov (ista semantika kot sequential_topic_modeling), `link_periods=
"collapse"` (viri obdobja p = vse citirano pred njim) ali `"strict"` (samo
neposredno predhodno obdobje). Rezultat: matrike prehodov (xlsx, po listih),
long tabela, atribut `group_citation_flow_transitions` in večstolpčni Sankey.

**14. `field_dynamics.py::_plot_sankey_matplotlib_multi` (novo).** Posplošitev
matplotlib Sankey fallbacka na k stolpcev (vozlišča v vmesnih stolpcih =
max(inflow, outflow), konsistentne barve konceptov čez stolpce).

**15. `utilsbib.py::compute_group_similarity_matrices` — metoda "lift" (novo).**
lift = P(A∩B) / (P(A)·P(B)) — mera nadnaključnega so-pojavljanja, primerna za
prekrivajoče skupine; na voljo tudi v `plot_group_heatmap`.

**16. `plotbib.py::plot_group_heatmap` — `mask_diagonal=True` (privzeto).**
Diagonala (samopodobnost = 1 oz. n) je trivialno maksimalna in požre barvno
skalo — vse ostale vrednosti so bile vizualno nerazločljive. Privzeto se zdaj
maskira (NaN); izklop z `mask_diagonal=False`.

**17. `addons/dashboard_builder.py` — pomanjšanje vgrajenih slik (novo).**
`embed_image_b64(max_px=, jpeg_quality=)` in `build_pipeline_report(
max_image_px=1600, image_jpeg_quality=85)`: rastrske slike se pred base64
vgradnjo pomanjšajo in re-enkodirajo v JPEG (RGBA na belo podlago). PMR
REPORT.html: 108,6 MB → 28,5 MB. `max_image_px=None` vgradi originale.

**18. `editions.py` (nov modul, 6. 8. 2026 zvečer).** Analiza večizdajnih
citiranih del ("biografija knjige"): `attribute_cited_editions` (atribucija
citirane izdaje iz referenčnih nizov; okna izdaj, frakcijsko uteževanje),
`edition_migration_table` + `plot_edition_migration` (stacked-area deleži
izdaj po letu citiranja), `edition_group_matrix` (izdaje kot group_matrix
za fingerprints prek BiblioGroupAnalysis). Validirano na PMR: 98,6 %
pripisljivih, 18,4 % z več izdajami. OPOZORILO: napisano brez sandbox
testa (disk poln) — prvi lokalni zagon skripte 28 je hkrati test.

## Popravki in novosti 7. 8. 2026

**19. `utilsbib.py` — mrtvi lazy globali `ig`/`community_louvain` (hrošč).**
Globala sta bila inicializirana na None s komentarjem "will be set on first
call", a ju ni nihče nikoli nastavil — VSE igraph particije (leiden,
infomap, spinglass, walktrap …) so padale z "'NoneType' object has no
attribute 'Graph'". Dodani `_require_igraph()`/`_require_louvain()` z jasno
napako; vsa mesta rabe preusmerjena.

**20. `bibstats.py::compute_bursts` — NaN ključne besede.** Dokumenti brez
ključnih besed so po explode ustvarjali navidezni "NaN keyword", ki je
burstal vsako leto z najvišjimi utežmi (25 lažnih vrstic na PMR korpusu).
Po explode se zdaj odstranijo NaN/prazni/"nan" žetoni.

**21. `utilsbib_modules/citations.py` — novi funkciji.**
`build_author_cocitation_network` (ACA iz referenčnih nizov: prvi avtorji,
normalizacija imen, izključitev poljubnih avtorjev, top-N + min so-citiranj)
in `compute_network_centralities` (degree, weighted degree, betweenness,
eigenvector, PageRank, HITS za usmerjena omrežja — en DataFrame).
Validirano na PMR: top 5 po PageRank = Hood, Peters, Christensen, Lægreid,
Osborne; Leiden po popravku (19) deluje.

**22. `utilsbib.py::semantic_interdisciplinarity` — HF ob importu (že 6.8.
zvečer).** Modul-level `SentenceTransformer(...)` je ob VSAKEM importu
bibliuma klical HuggingFace (HEAD + retry) — ob slabi mreži je vse viselo.
Zdaj lazy-load ob prvi rabi (`_get_sentence_model`).

## Opažanja (brez sprememb kode)
- Korak 07 (permutacijske asociacije, 999 perm): 21 176 s (2.16, lokalno) →
  82 s (2.18, sandbox). Vektorizirani `permutation_test` v utilsbib je
  bistveno hitrejši od stare poti.
- Pi-index se od 2.16 razlikuje NAMERNO (Vinkler 2009, /100 — popravek iz
  konsolidacije). Rank/Percentrank se razlikujeta le pri tie-breakih.
- `prince` je nova efektivna odvisnost za korespondenčno analizo v
  `associate_items` (dodati v requirements/pyproject opcije).
