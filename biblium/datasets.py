# -*- coding: utf-8 -*-
"""
biblium.datasets
================

Catalogue of bundled / downloadable demo datasets for many scientific domains
(novo v 2.19.1). All samples come from **OpenAlex** (CC0 metadata), so they can
be redistributed with the package without licensing problems.

Each catalogue entry describes an OpenAlex query (search phrase, filters, sort
order, sample size). ``fetch_dataset`` downloads the works through the OpenAlex
API and flattens them into the same column layout as the OpenAlex website CSV
export, so ``readbib.read_bibfile(path, db="oa")`` / ``BiblioStats`` read them
like any other OpenAlex export. ``load_dataset`` returns a ``BiblioStats``
object for a bundled sample (or downloads it on first use).

Command line (run on a machine with internet access)::

    python -m biblium.datasets --list
    python -m biblium.datasets --fetch pa_egovernment --email you@uni.si
    python -m biblium.datasets --fetch-all --email you@uni.si --out biblium/data/datasets

Version: 2.19.1
"""
from __future__ import annotations

import gzip
import io
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

__all__ = [
    "DatasetSpec", "DATASETS", "DOMAINS", "list_datasets", "dataset_path", "fetch_works",
    "flatten_work", "works_to_frame", "fetch_dataset", "fetch_all", "load_dataset", "load_dataset_df",
]

OPENALEX_WORKS = "https://api.openalex.org/works"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "datasets")
DEFAULT_N = 200


@dataclass
class DatasetSpec:
    """One catalogue entry = one OpenAlex query."""
    name: str                      # file stem, e.g. "pa_egovernment"
    domain: str                    # e.g. "Public administration"
    title: str                     # human readable
    description: str
    search: Optional[str] = None   # OpenAlex ``search=`` (title+abstract+fulltext relevance)
    filters: Dict[str, str] = field(default_factory=dict)  # OpenAlex filter key -> value
    sort: str = "most_cited"       # key of openalex_api.SORT_OPTIONS or raw OpenAlex sort
    n: int = DEFAULT_N
    kind: str = "topic"            # "topic" | "journal" | "country"
    require_abstract: bool = True  # journal samples: False (many publishers do not deposit abstracts)

    def filter_string(self) -> str:
        base = {"type": "article", "language": "en"}
        if self.require_abstract:
            base["has_abstract"] = "true"
        base.update(self.filters)
        return ",".join(f"{k}:{v}" for k, v in base.items())


def _topic(name, domain, title, description, search, years="2010-2025", **filters):
    f = {"publication_year": years}
    f.update(filters)
    return DatasetSpec(name=name, domain=domain, title=title, description=description, search=search, filters=f)


def _journal(name, domain, title, issn, description, years="2010-2025", **filters):
    f = {"primary_location.source.issn": issn, "publication_year": years}
    f.update(filters)
    return DatasetSpec(name=name, domain=domain, title=title, description=description, filters=f, kind="journal",
                       require_abstract=False)


DATASETS: Dict[str, DatasetSpec] = {s.name: s for s in [
    # ------------------------------------------------------------------ Public administration
    _topic("pa_egovernment", "Public administration", "E-government & digital transformation of the public sector",
           "Digital government, e-services, digital transformation of public administrations.",
           '"e-government" OR "digital government" OR "digital transformation" "public sector"'),
    _topic("pa_npm_reform", "Public administration", "New Public Management and public sector reform",
           "NPM, post-NPM, public management reform trajectories.",
           '"new public management" OR "public management reform" OR "post-NPM"'),
    _topic("pa_coproduction", "Public administration", "Co-production and co-creation of public services",
           "Citizen co-production / co-creation of public services and public value.",
           '"co-production" OR "co-creation" "public services"'),
    _topic("pa_collaborative_governance", "Public administration", "Collaborative governance & networks",
           "Collaborative governance, governance networks, cross-sector collaboration.",
           '"collaborative governance" OR "governance networks" OR "network governance"'),
    _topic("pa_public_service_motivation", "Public administration", "Public service motivation",
           "PSM, motivation and behaviour of public employees.",
           '"public service motivation"'),
    _topic("pa_red_tape", "Public administration", "Red tape and administrative burden",
           "Administrative burden, red tape, bureaucratic rules and citizens' experience.",
           '"administrative burden" OR "red tape"'),
    _topic("pa_smart_cities", "Public administration", "Smart cities and urban governance",
           "Smart city governance, urban data platforms, city administration.",
           '"smart city" OR "smart cities" governance'),
    _topic("pa_slovenia", "Public administration", "Public administration research from Slovenia",
           "Public administration / public policy papers with at least one Slovenian institution.",
           '"public administration" OR "public sector" OR "public policy"', **{"institutions.country_code": "si"}),
    # ------------------------------------------------------------------ Medicine
    _topic("med_telemedicine", "Medicine", "Telemedicine and digital health",
           "Telemedicine, telehealth, remote patient monitoring.",
           '"telemedicine" OR "telehealth" OR "remote patient monitoring"'),
    _topic("med_vaccine_hesitancy", "Medicine", "Vaccine hesitancy",
           "Vaccine hesitancy, confidence and uptake (incl. COVID-19).",
           '"vaccine hesitancy"'),
    _topic("med_antimicrobial_resistance", "Medicine", "Antimicrobial resistance & stewardship",
           "AMR surveillance, antibiotic stewardship programmes.",
           '"antimicrobial resistance" "stewardship"'),
    _topic("med_ai_imaging", "Medicine", "Artificial intelligence in medical imaging",
           "Deep learning / AI for radiology and diagnostic imaging.",
           '"deep learning" "medical imaging" OR "radiology" "artificial intelligence"'),
    _topic("med_long_covid", "Medicine", "Long COVID",
           "Post-acute sequelae of SARS-CoV-2 infection.",
           '"long COVID" OR "post-COVID" OR "post-acute sequelae"', years="2020-2025"),
    # ------------------------------------------------------------------ Social sciences
    _topic("soc_social_capital", "Sociology", "Social capital",
           "Social capital, trust and civic participation.",
           '"social capital" trust'),
    _topic("soc_polarization", "Sociology", "Social media and political polarization",
           "Echo chambers, filter bubbles, polarization on social media.",
           '"political polarization" "social media" OR "echo chamber"'),
    _topic("soc_gender_gap", "Sociology", "Gender inequality in the labour market",
           "Gender pay gap, occupational segregation, motherhood penalty.",
           '"gender pay gap" OR "gender wage gap" OR "occupational segregation"'),
    _topic("pol_populism", "Political science", "Populism",
           "Populist parties, leaders and discourse.",
           '"populism" OR "populist"'),
    _topic("econ_basic_income", "Economics", "Universal basic income",
           "UBI experiments, cash transfers and labour supply.",
           '"universal basic income" OR "basic income"'),
    _topic("econ_minimum_wage", "Economics", "Minimum wage and employment",
           "Employment effects of minimum wage policies.",
           '"minimum wage" employment'),
    _topic("econ_nudging", "Economics", "Behavioural economics and nudging",
           "Nudges, choice architecture, behavioural public policy.",
           '"nudge" OR "nudging" "behavioral economics" OR "behavioural economics"'),
    _topic("bus_csr", "Business", "Corporate social responsibility & ESG",
           "CSR, ESG performance and firm value.",
           '"corporate social responsibility" OR "ESG" "firm performance"'),
    _topic("edu_flipped_classroom", "Education", "Flipped classroom",
           "Flipped / inverted classroom in higher education.",
           '"flipped classroom" OR "inverted classroom"'),
    _topic("psy_mindfulness", "Psychology", "Mindfulness interventions",
           "Mindfulness-based interventions and mental health.",
           '"mindfulness-based" intervention'),
    # ------------------------------------------------------------------ Law
    _topic("law_gdpr", "Law", "Data protection and the GDPR",
           "GDPR, privacy law, data protection regulation.",
           '"General Data Protection Regulation" OR "GDPR"', years="2016-2025"),
    _topic("law_ai_regulation", "Law", "Regulation of artificial intelligence",
           "AI governance, AI Act, algorithmic accountability and liability.",
           '"artificial intelligence" regulation OR "AI Act" OR "algorithmic accountability"', years="2016-2025"),
    _topic("law_climate_litigation", "Law", "Climate change litigation",
           "Climate litigation, environmental rights, human rights and climate.",
           '"climate litigation" OR "climate change litigation"'),
    # ------------------------------------------------------------------ Computer science & mathematics
    _topic("cs_llm", "Computer science", "Large language models",
           "LLMs, transformers, prompting, evaluation.",
           '"large language model" OR "large language models"', years="2019-2025"),
    _topic("cs_federated_learning", "Computer science", "Federated learning",
           "Federated / decentralised machine learning and privacy.",
           '"federated learning"', years="2017-2025"),
    _topic("cs_explainable_ai", "Computer science", "Explainable AI",
           "XAI, interpretability of machine-learning models.",
           '"explainable artificial intelligence" OR "explainable AI" OR "interpretable machine learning"'),
    _topic("math_optimal_transport", "Mathematics", "Optimal transport",
           "Optimal transport, Wasserstein distances and applications.",
           '"optimal transport" Wasserstein'),
    _topic("math_fractional_calculus", "Mathematics", "Fractional differential equations",
           "Fractional calculus and fractional differential equations.",
           '"fractional differential equations" OR "fractional calculus"'),
    _topic("math_graph_theory", "Mathematics", "Spectral graph theory",
           "Graph spectra, Laplacians, expander graphs.",
           '"spectral graph theory" OR "graph Laplacian" OR "expander graphs"'),
    # ------------------------------------------------------------------ Natural / environmental sciences
    _topic("env_microplastics", "Environmental science", "Microplastics",
           "Microplastic pollution in water, soil and organisms.",
           '"microplastics" OR "microplastic"'),
    _topic("bio_crispr", "Life sciences", "CRISPR genome editing",
           "CRISPR-Cas9 genome editing and applications.",
           '"CRISPR" "genome editing"'),
    _topic("info_altmetrics", "Information science", "Altmetrics",
           "Altmetrics, social media mentions of research.",
           '"altmetrics" OR "altmetric"'),
    # ------------------------------------------------------------------ Journal-specific samples
    _journal("j_public_administration_review", "Public administration", "Public Administration Review", "0033-3352",
             "Most cited PAR articles 2010-2025."),
    _journal("j_jpart", "Public administration", "Journal of Public Administration Research and Theory", "1053-1858",
             "Most cited JPART articles 2010-2025."),
    _journal("j_public_management_review", "Public administration", "Public Management Review", "1471-9037",
             "Most cited PMR articles 2010-2025."),
    _journal("j_cepar", "Public administration", "Central European Public Administration Review", "2591-2240",
             "CEPAR (University of Ljubljana, Faculty of Public Administration) 2018-2025.", years="2018-2025"),
    _journal("j_scientometrics", "Information science", "Scientometrics", "0138-9130",
             "Most cited Scientometrics articles 2010-2025."),
    _journal("j_journal_of_informetrics", "Information science", "Journal of Informetrics", "1751-1577",
             "Most cited Journal of Informetrics articles 2010-2025."),
    _journal("j_american_sociological_review", "Sociology", "American Sociological Review", "0003-1224",
             "Most cited ASR articles 2010-2025."),
    _journal("j_american_economic_review", "Economics", "American Economic Review", "0002-8282",
             "Most cited AER articles 2010-2025."),
    _journal("j_lancet", "Medicine", "The Lancet", "0140-6736",
             "Most cited Lancet articles 2015-2025.", years="2015-2025"),
    _journal("j_harvard_law_review", "Law", "Harvard Law Review", "0017-811X",
             "Harvard Law Review articles 2010-2025."),
    _journal("j_communications_acm", "Computer science", "Communications of the ACM", "0001-0782",
             "Most cited CACM articles 2010-2025."),
    _journal("j_annals_of_mathematics", "Mathematics", "Annals of Mathematics", "0003-486X",
             "Most cited Annals of Mathematics articles 2010-2025."),
]}

DOMAINS: List[str] = sorted({s.domain for s in DATASETS.values()})


def list_datasets(domain: Optional[str] = None) -> pd.DataFrame:
    """Catalogue as a DataFrame (name, domain, kind, title, n, bundled)."""
    rows = []
    for s in DATASETS.values():
        if domain and s.domain != domain:
            continue
        rows.append({"name": s.name, "domain": s.domain, "kind": s.kind, "title": s.title, "n": s.n,
                     "bundled": os.path.exists(dataset_path(s.name)), "description": s.description})
    return pd.DataFrame(rows)


def dataset_path(name: str, data_dir: Optional[str] = None) -> str:
    d = data_dir or DATA_DIR
    p = os.path.join(d, f"{name}.csv.gz")
    if os.path.exists(p):
        return p
    return os.path.join(d, f"{name}.csv") if os.path.exists(os.path.join(d, f"{name}.csv")) else p


# ---------------------------------------------------------------------------
# OpenAlex fetching
# ---------------------------------------------------------------------------

def fetch_works(
    spec: DatasetSpec,
    email: Optional[str] = None,
    n: Optional[int] = None,
    per_page: int = 200,
    session=None,
    delay: float = 0.2,
    fetch_json=None,
    sort: Optional[str] = None,
    seed: int = 42,
) -> List[dict]:
    """Download up to ``n`` works for a spec. ``fetch_json(url)`` is injectable for tests.

    ``sort`` decides *which* works are returned when the query matches more than ``n``:
    "most_cited" (default), "newest", "oldest", "least_cited", "relevance", "random" (seeded
    random sample via OpenAlex ``sample``) or a raw OpenAlex sort such as "fwci:desc".
    """
    from biblium.openalex_api import resolve_sort, paging_params
    n = n or spec.n
    sort_value = resolve_sort(sort or spec.sort, has_search=bool(spec.search))
    if fetch_json is None:
        import requests
        sess = session or requests.Session()

        def fetch_json(url):
            r = sess.get(url, timeout=60, headers={"User-Agent": f"biblium (mailto:{email})" if email else "biblium"})
            r.raise_for_status()
            return r.json()

    works: List[dict] = []
    cursor: Optional[str] = "*"
    page = 0
    while len(works) < n and (cursor or sort_value == "random"):
        page += 1
        params = {"filter": spec.filter_string()}
        params.update(paging_params(sort_value, n, len(works), cursor, page, per_page, seed))
        if spec.search:
            params["search"] = spec.search
        if email:
            params["mailto"] = email
        from urllib.parse import urlencode
        url = f"{OPENALEX_WORKS}?{urlencode(params)}"
        data = fetch_json(url)
        results = data.get("results") or []
        works.extend(results)
        cursor = (data.get("meta") or {}).get("next_cursor")
        if not results:
            break
        time.sleep(delay)
    return works[:n]


# ---------------------------------------------------------------------------
# flattening to the OpenAlex CSV-export layout
# ---------------------------------------------------------------------------

def _abstract(inv: Optional[dict]) -> Optional[str]:
    if not inv:
        return None
    pos = {}
    for w, ps in inv.items():
        for p in ps:
            pos[p] = w
    return " ".join(pos[i] for i in sorted(pos)) if pos else None


def _join(items: Iterable[Any]) -> Optional[str]:
    vals = [str(x) for x in items if x is not None and str(x) != ""]
    return "|".join(vals) if vals else None


def _get(d: Optional[dict], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur


def flatten_work(w: dict) -> Dict[str, Any]:
    """Flatten one OpenAlex work JSON into the website-export column layout (lists joined with ``|``)."""
    auths = w.get("authorships") or []
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    boa = w.get("best_oa_location") or {}
    oa = w.get("open_access") or {}
    pt = w.get("primary_topic") or {}
    topics = w.get("topics") or []
    cby = w.get("counts_by_year") or []
    biblio = w.get("biblio") or {}
    cnp = w.get("citation_normalized_percentile") or {}
    ids = w.get("ids") or {}
    rec = {
        "id": w.get("id"), "doi": w.get("doi"), "title": w.get("title"), "display_name": w.get("display_name"),
        "publication_year": w.get("publication_year"), "publication_date": w.get("publication_date"),
        "language": w.get("language"), "type": w.get("type"), "type_crossref": w.get("type_crossref"),
        "indexed_in": _join(w.get("indexed_in") or []),
        "countries_distinct_count": w.get("countries_distinct_count"),
        "institutions_distinct_count": w.get("institutions_distinct_count"),
        "fwci": w.get("fwci"), "has_fulltext": w.get("has_fulltext"), "cited_by_count": w.get("cited_by_count"),
        "is_retracted": w.get("is_retracted"), "is_paratext": w.get("is_paratext"),
        "locations_count": w.get("locations_count"),
        "referenced_works_count": w.get("referenced_works_count"),
        "referenced_works": _join(w.get("referenced_works") or []),
        "related_works": _join(w.get("related_works") or []),
        "cited_by_api_url": w.get("cited_by_api_url"), "updated_date": w.get("updated_date"),
        "created_date": w.get("created_date"),
        "ids.openalex": ids.get("openalex"), "ids.doi": ids.get("doi"), "ids.mag": ids.get("mag"),
        "ids.pmid": ids.get("pmid"), "ids.pmcid": ids.get("pmcid"),
        "primary_location.is_oa": loc.get("is_oa"), "primary_location.landing_page_url": loc.get("landing_page_url"),
        "primary_location.pdf_url": loc.get("pdf_url"),
        "primary_location.source.id": src.get("id"), "primary_location.source.display_name": src.get("display_name"),
        "primary_location.source.issn_l": src.get("issn_l"), "primary_location.source.issn": _join(src.get("issn") or []),
        "primary_location.source.is_oa": src.get("is_oa"), "primary_location.source.is_in_doaj": src.get("is_in_doaj"),
        "primary_location.source.is_indexed_in_scopus": src.get("is_indexed_in_scopus"),
        "primary_location.source.is_core": src.get("is_core"),
        "primary_location.source.host_organization": src.get("host_organization"),
        "primary_location.source.host_organization_name": src.get("host_organization_name"),
        "primary_location.source.type": src.get("type"),
        "primary_location.license": loc.get("license"), "primary_location.version": loc.get("version"),
        "primary_location.is_accepted": loc.get("is_accepted"), "primary_location.is_published": loc.get("is_published"),
        "open_access.is_oa": oa.get("is_oa"), "open_access.oa_status": oa.get("oa_status"),
        "open_access.oa_url": oa.get("oa_url"),
        "open_access.any_repository_has_fulltext": oa.get("any_repository_has_fulltext"),
        "citation_normalized_percentile.value": cnp.get("value"),
        "citation_normalized_percentile.is_in_top_1_percent": cnp.get("is_in_top_1_percent"),
        "citation_normalized_percentile.is_in_top_10_percent": cnp.get("is_in_top_10_percent"),
        "cited_by_percentile_year.min": _get(w, "cited_by_percentile_year", "min"),
        "cited_by_percentile_year.max": _get(w, "cited_by_percentile_year", "max"),
        "biblio.volume": biblio.get("volume"), "biblio.issue": biblio.get("issue"),
        "biblio.first_page": biblio.get("first_page"), "biblio.last_page": biblio.get("last_page"),
        "primary_topic.id": pt.get("id"), "primary_topic.display_name": pt.get("display_name"),
        "primary_topic.score": pt.get("score"),
        "primary_topic.subfield.display_name": _get(pt, "subfield", "display_name"),
        "primary_topic.field.display_name": _get(pt, "field", "display_name"),
        "primary_topic.domain.display_name": _get(pt, "domain", "display_name"),
        "best_oa_location.is_oa": boa.get("is_oa"), "best_oa_location.landing_page_url": boa.get("landing_page_url"),
        "best_oa_location.pdf_url": boa.get("pdf_url"),
        "best_oa_location.source.display_name": _get(boa, "source", "display_name"),
        "best_oa_location.license": boa.get("license"), "best_oa_location.version": boa.get("version"),
        "abstract": _abstract(w.get("abstract_inverted_index")),
        "authorships.author_position": _join(a.get("author_position") for a in auths),
        "authorships.countries": _join(c for a in auths for c in (a.get("countries") or [])),
        "authorships.is_corresponding": _join(a.get("is_corresponding") for a in auths),
        "authorships.raw_author_name": _join(a.get("raw_author_name") or _get(a, "author", "display_name") for a in auths),
        "authorships.raw_affiliation_strings": _join("; ".join(a.get("raw_affiliation_strings") or []) for a in auths),
        "authorships.author.id": _join(_get(a, "author", "id") for a in auths),
        "authorships.author.display_name": _join(_get(a, "author", "display_name") for a in auths),
        "authorships.author.orcid": _join(_get(a, "author", "orcid") or "" for a in auths),
        "authorships.institutions.display_name": _join("; ".join(i.get("display_name") or "" for i in (a.get("institutions") or [])) for a in auths),
        "authorships.institutions.country_code": _join("; ".join(i.get("country_code") or "" for i in (a.get("institutions") or [])) for a in auths),
        "topics.id": _join(t.get("id") for t in topics),
        "topics.display_name": _join(t.get("display_name") for t in topics),
        "topics.score": _join(t.get("score") for t in topics),
        "topics.subfield.display_name": _join(_get(t, "subfield", "display_name") for t in topics),
        "topics.field.display_name": _join(_get(t, "field", "display_name") for t in topics),
        "topics.domain.display_name": _join(_get(t, "domain", "display_name") for t in topics),
        "keywords.id": _join(k.get("id") for k in (w.get("keywords") or [])),
        "keywords.display_name": _join(k.get("display_name") for k in (w.get("keywords") or [])),
        "keywords.score": _join(k.get("score") for k in (w.get("keywords") or [])),
        "concepts.id": _join(c.get("id") for c in (w.get("concepts") or [])),
        "concepts.display_name": _join(c.get("display_name") for c in (w.get("concepts") or [])),
        "concepts.level": _join(c.get("level") for c in (w.get("concepts") or [])),
        "concepts.score": _join(c.get("score") for c in (w.get("concepts") or [])),
        "mesh.descriptor_ui": _join(m.get("descriptor_ui") for m in (w.get("mesh") or [])),
        "mesh.descriptor_name": _join(m.get("descriptor_name") for m in (w.get("mesh") or [])),
        "mesh.is_major_topic": _join(m.get("is_major_topic") for m in (w.get("mesh") or [])),
        "sustainable_development_goals.id": _join(s.get("id") for s in (w.get("sustainable_development_goals") or [])),
        "sustainable_development_goals.display_name": _join(s.get("display_name") for s in (w.get("sustainable_development_goals") or [])),
        "sustainable_development_goals.score": _join(s.get("score") for s in (w.get("sustainable_development_goals") or [])),
        "grants.funder": _join(g.get("funder") for g in (w.get("grants") or [])),
        "grants.funder_display_name": _join(g.get("funder_display_name") for g in (w.get("grants") or [])),
        "grants.award_id": _join(g.get("award_id") for g in (w.get("grants") or [])),
        "counts_by_year.year": _join(c.get("year") for c in cby),
        "counts_by_year.cited_by_count": _join(c.get("cited_by_count") for c in cby),
    }
    return rec


def works_to_frame(works: Sequence[dict]) -> pd.DataFrame:
    return pd.DataFrame([flatten_work(w) for w in works])


def fetch_dataset(
    name: str,
    email: Optional[str] = None,
    out_dir: Optional[str] = None,
    n: Optional[int] = None,
    overwrite: bool = False,
    compress: bool = True,
    fetch_json=None,
    verbose: bool = True,
    sort: Optional[str] = None,
) -> str:
    """Download one catalogue dataset to ``out_dir`` (default: the package's ``data/datasets``). Returns the path.

    ``sort`` overrides the catalogue ordering ("most_cited", "newest", "oldest", "random", ...).
    """
    spec = DATASETS[name]
    out_dir = out_dir or DATA_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.csv.gz" if compress else f"{name}.csv")
    if os.path.exists(path) and not overwrite:
        if verbose:
            print(f"[skip] {name}: exists ({path})")
        return path
    works = fetch_works(spec, email=email, n=n, fetch_json=fetch_json, sort=sort)
    df = works_to_frame(works)
    df.to_csv(path, index=False, compression="gzip" if compress else None)
    if verbose:
        print(f"[ok] {name}: {len(df)} works -> {path}")
    return path


def fetch_all(email: Optional[str] = None, out_dir: Optional[str] = None, names: Optional[Sequence[str]] = None,
              overwrite: bool = False, verbose: bool = True, **kw) -> Dict[str, str]:
    """Download every catalogue dataset (or ``names``). Returns {name: path}; failures are reported, not raised."""
    out = {}
    for name in (names or list(DATASETS)):
        try:
            out[name] = fetch_dataset(name, email=email, out_dir=out_dir, overwrite=overwrite, verbose=verbose, **kw)
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {name}: {e}")
    return out


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_dataset_df(name: str, data_dir: Optional[str] = None, fetch_if_missing: bool = False,
                    email: Optional[str] = None) -> pd.DataFrame:
    """Raw (standardised) DataFrame of a bundled dataset via ``readbib.read_bibfile(db='oa')``."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}; see list_datasets()")
    path = dataset_path(name, data_dir)
    if not os.path.exists(path):
        if not fetch_if_missing:
            raise FileNotFoundError(f"Dataset {name!r} is not bundled; run fetch_dataset({name!r}, email=...) first")
        path = fetch_dataset(name, email=email, out_dir=data_dir)
    from biblium import readbib
    return readbib.read_bibfile(path, "oa")


def load_dataset(name: str, data_dir: Optional[str] = None, fetch_if_missing: bool = False,
                 email: Optional[str] = None, **bib_kwargs):
    """``BiblioStats`` for a bundled dataset (``db='oa'``). ``bib_kwargs`` go to the constructor."""
    df = load_dataset_df(name, data_dir=data_dir, fetch_if_missing=fetch_if_missing, email=email)
    from biblium.bibstats import BiblioStats
    import inspect
    allowed = set(inspect.signature(BiblioStats.__init__).parameters)
    bib = BiblioStats(df=df, db="oa", **{k: v for k, v in bib_kwargs.items() if k in allowed})
    bib.dataset_name = name
    bib.dataset_spec = DATASETS[name]
    return bib


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m biblium.datasets", description="Fetch biblium demo datasets from OpenAlex")
    ap.add_argument("--list", action="store_true", help="list the catalogue")
    ap.add_argument("--fetch", nargs="*", metavar="NAME", help="fetch the named datasets")
    ap.add_argument("--fetch-all", action="store_true", help="fetch every dataset in the catalogue")
    ap.add_argument("--domain", help="restrict --list / --fetch-all to one domain")
    ap.add_argument("--email", help="contact e-mail for the OpenAlex polite pool (recommended)")
    ap.add_argument("--out", help=f"output folder (default: {DATA_DIR})")
    ap.add_argument("--n", type=int, help="override sample size")
    ap.add_argument("--sort", choices=["most_cited", "least_cited", "newest", "oldest", "relevance", "random"],
                    help="which records to keep when more match (default: catalogue setting, most_cited)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-compress", action="store_true", help="write .csv instead of .csv.gz")
    a = ap.parse_args(argv)
    if a.list or not (a.fetch or a.fetch_all):
        with pd.option_context("display.max_colwidth", 60, "display.width", 200):
            print(list_datasets(a.domain).to_string(index=False))
        return 0
    names = a.fetch if a.fetch else [s.name for s in DATASETS.values() if not a.domain or s.domain == a.domain]
    fetch_all(email=a.email, out_dir=a.out, names=names, overwrite=a.overwrite, n=a.n, compress=not a.no_compress,
              sort=a.sort)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
