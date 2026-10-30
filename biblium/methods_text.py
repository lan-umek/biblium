# -*- coding: utf-8 -*-
"""
Generator metodološkega odstavka (novo v 2.18; razširjeno za indikatorje 2.19).

Iz stanja BiblioStats objekta deterministično sestavi "Methods" odstavek za
članek: podatkovni vir, obdobje, obseg, izvedene analize z dejansko
uporabljenimi parametri ter navedbo programske opreme. Namen: točna,
reproducibilna metodologija brez ročnega prepisovanja parametrov.
"""
from __future__ import annotations

import platform
import sys

DB_NAMES = {
    "scopus": "Scopus",
    "wos": "Web of Science",
    "oa": "OpenAlex",
    "openalex": "OpenAlex",
    "cobiss": "COBISS",
}

_REFS = {
    "rpys": "Marx, W., Bornmann, L., Barth, A., & Leydesdorff, L. (2014). Detecting the historical roots of research fields by reference publication year spectroscopy (RPYS). JASIST, 65(4), 751–764.",
    "louvain": "Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. J. Stat. Mech., P10008.",
    "tapered_h": "Anderson, T. R., Hankin, R. K. S., & Killworth, P. D. (2008). Beyond the Durfee square: Enhancing the h-index to score total publication output. Scientometrics, 76(3), 577–588.",
    "pi": "Vinkler, P. (2009). The π-index: a new indicator for assessing scientific impact. Journal of Information Science, 35(5), 602–612.",
    "css": "Glänzel, W., & Schubert, A. (1988). Characteristic scores and scales in assessing citation impact. Journal of Information Science, 14(2), 123–127.",
    "leiden": "Waltman, L., Calero-Medina, C., Kosten, J., Noyons, E. C. M., Tijssen, R. J. W., van Eck, N. J., … Wouters, P. (2012). The Leiden ranking 2011/2012: Data collection, indicators, and interpretation. JASIST, 63(12), 2419–2432.",
    "price": "Price, D. J. de Solla (1970). Citation measures of hard science, soft science, technology, and nonscience. In C. E. Nelson & D. K. Pollock (Eds.), Communication among scientists and engineers (pp. 3–22). Heath.",
    "gini": "Gini, C. (1921). Measurement of inequality of incomes. The Economic Journal, 31(121), 124–126.",
    "hirsch_m": "Hirsch, J. E. (2005). An index to quantify an individual's scientific research output. PNAS, 102(46), 16569–16572.",
    "heaps": "Heaps, H. S. (1978). Information retrieval: Computational and theoretical aspects. Academic Press.",
    "bootstrap": "Efron, B., & Tibshirani, R. J. (1993). An introduction to the bootstrap. Chapman & Hall.",
    "prisma": "Page, M. J., McKenzie, J. E., Bossuyt, P. M., et al. (2021). The PRISMA 2020 statement: An updated guideline for reporting systematic reviews. BMJ, 372, n71.",
    "sva": "Chen, C. (2012). Predictive effects of structural variation on citation counts. JASIST, 63(3), 431–449.",
    "dual_map": "Chen, C., & Leydesdorff, L. (2014). Patterns of connections and movements in dual-map overlays: A new method of publication portfolio analysis. JASIST, 65(2), 334–351.",
    "vos_terms": "van Eck, N. J., & Waltman, L. (2011). Text mining and visualization using VOSviewer. ISSI Newsletter, 7(3), 50–54.",
    "sleeping": "van Raan, A. F. J. (2004). Sleeping Beauties in science. Scientometrics, 59(3), 467–472.",
    "openalex": "Priem, J., Piwowar, H., & Orr, R. (2022). OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts. arXiv:2205.01833.",
}


def _dataset_sentence(bib, language):
    db = DB_NAMES.get(getattr(bib, "db", ""), getattr(bib, "db", "an unknown database"))
    srcs = getattr(bib, "source_dbs", None)
    if srcs and len(srcs) > 1:  # 2.19: združeni nabor iz več baz
        db = ", ".join(DB_NAMES.get(x, x) for x in srcs)
    n = len(bib.df) if getattr(bib, "df", None) is not None else None
    year_range = ""
    try:
        import pandas as pd
        years = pd.to_numeric(bib.df["Year"], errors="coerce").dropna()
        if len(years):
            year_range = (int(years.min()), int(years.max()))
    except Exception:
        pass
    spec = getattr(bib, "dataset_spec", None)  # 2.19.1: vgrajeni domenski nabor (biblium.datasets)
    if language == "sl":
        s = f"Bibliografski podatki so bili pridobljeni iz baz {db}" if (srcs and len(srcs) > 1) else f"Bibliografski podatki so bili pridobljeni iz baze {db}"
        if n:
            s += f" in obsegajo {n} dokumentov"
        if year_range:
            s += f", objavljenih med letoma {year_range[0]} in {year_range[1]}"
        s += "."
        if spec is not None:
            from biblium.openalex_api import sort_description
            q = f"revije {spec.title}" if spec.kind == "journal" else f"tematike »{spec.title}«"
            s += (f" Gre za demonstracijski nabor knjižnice biblium ({spec.name}) s področja {spec.domain}: "
                  f"{sort_description(spec.sort, 'sl')} {q}, pridobljeni prek OpenAlex API"
                  + (f" (iskalni niz: {spec.search})." if spec.search else "."))
        return s
    s = f"Bibliographic records were retrieved from {db}"
    if n:
        s += f", comprising {n} documents"
    if year_range:
        s += f" published between {year_range[0]} and {year_range[1]}"
    s += "."
    if spec is not None:
        from biblium.openalex_api import sort_description
        q = f"the journal {spec.title}" if spec.kind == "journal" else f"the topic '{spec.title}'"
        s += (f" The data are the biblium demonstration dataset '{spec.name}' ({spec.domain}): "
              f"{sort_description(spec.sort, 'en')} from {q}, retrieved through the OpenAlex API"
              + (f" (search string: {spec.search})." if spec.search else "."))
    return s


def _analysis_sentences(bib, language):
    out, refs = [], []
    p = getattr(bib, "_analysis_params", {}) or {}

    if hasattr(bib, "rpys_df") and getattr(bib, "rpys_df", None) is not None:
        w = p.get("rpys", {}).get("median_window", 5)
        if language == "sl":
            out.append(
                "Zgodovinske korenine področja smo identificirali z metodo RPYS "
                "(Reference Publication Year Spectroscopy; Marx idr., 2014), pri čemer "
                f"smo odklone računali od drseče mediane čez {w} let."
            )
        else:
            out.append(
                "The historical roots of the field were identified using Reference "
                "Publication Year Spectroscopy (RPYS; Marx et al., 2014), with deviations "
                f"computed from a {w}-year rolling median."
            )
        refs.append(_REFS["rpys"])

    has_lcs = getattr(bib, "df", None) is not None and "Local Citations" in bib.df.columns
    if has_lcs:
        mb = p.get("local_citations", {}).get("match_by", "auto")
        how = {"doi": "DOI", "title": "normalized-title", "auto": "DOI (with normalized-title fallback)"}[mb] if mb in ("doi", "title", "auto") else mb
        how_sl = {"doi": "DOI", "title": "normaliziranih naslovov", "auto": "DOI (z nadomestnim ujemanjem naslovov)"}.get(mb, mb)
        if language == "sl":
            out.append(
                "Lokalne citacije (LCS) — citiranost znotraj analizirane zbirke — smo "
                f"izračunali z ujemanjem {how_sl} v referencah in jih primerjali z globalnimi citacijami (GCS)."
            )
        else:
            out.append(
                "Local citation scores (LCS) — citations received from within the analysed "
                f"collection — were computed via {how} matching of cited references and "
                "compared with global citation scores (GCS)."
            )

    if hasattr(bib, "thematic_evolution_df") and getattr(bib, "thematic_evolution_df", None) is not None:
        tp = p.get("thematic_evolution", {})
        field = tp.get("field", "Author Keywords")
        method = tp.get("partition_method", "louvain")
        labels = getattr(bib, "thematic_evolution_labels", [])
        per = ", ".join(labels) if labels else f"{tp.get('n_slices', 3)} periods"
        if language == "sl":
            out.append(
                f"Tematsko evolucijo smo analizirali s sobesednimi mrežami polja »{field}« "
                f"po obdobjih ({per}); teme smo določili z algoritmom {method} "
                "(Blondel idr., 2008) s fiksnim semenom za ponovljivost."
            )
        else:
            out.append(
                f"Thematic evolution was analysed through co-word networks of the "
                f"'{field}' field across time slices ({per}); themes were detected with "
                f"the {method} community-detection algorithm (Blondel et al., 2008) "
                "using a fixed random seed for reproducibility."
            )
        refs.append(_REFS["louvain"])

    # --- novo v 2.19 -------------------------------------------------------
    if getattr(bib, "css_summary_df", None) is not None:
        cp = p.get("citation_css", {})
        by_year = cp.get("by_year", False)
        if language == "sl":
            out.append(
                "Dokumente smo po citiranosti razvrstili z metodo značilnih vrednosti in "
                "lestvic (Characteristic Scores and Scales; Glänzel & Schubert, 1988), "
                "pri kateri so pragovi določeni iterativno s povprečji"
                + (" ločeno za vsako leto objave." if by_year else ".")
            )
        else:
            out.append(
                "Documents were classified by citation impact using Characteristic Scores "
                "and Scales (Glänzel & Schubert, 1988), whose thresholds are derived "
                "iteratively from successive means"
                + (", separately for each publication year." if by_year else ".")
            )
        refs.append(_REFS["css"])

    if getattr(bib, "percentile_indicators_df", None) is not None:
        pp = p.get("percentile_indicators", {})
        ent = pp.get("entity", "entities")
        thr = pp.get("thresholds", [10, 1])
        thr_txt = ", ".join(f"{t:g} %" for t in thr)
        if language == "sl":
            out.append(
                f"Za {ent} smo izračunali percentilne indikatorje po vzoru Leidenske "
                f"lestvice (Waltman idr., 2012): delež dokumentov med zgornjimi {thr_txt} "
                "po citiranosti v letu objave (PP(top X %)) ter povprečno normalizirano "
                "citiranost (MNCS), normalizirano glede na povprečje leta objave."
            )
        else:
            out.append(
                f"For {ent}, percentile-based indicators in the spirit of the Leiden Ranking "
                f"(Waltman et al., 2012) were computed: the share of documents among the top "
                f"{thr_txt} most cited of their publication year (PP(top X%)) and the mean "
                "normalised citation score (MNCS), normalised by the publication-year mean."
            )
        refs.append(_REFS["leiden"])

    if getattr(bib, "reference_age_df", None) is not None:
        w = p.get("reference_age", {}).get("price_window", 5)
        if language == "sl":
            out.append(
                "Starost citirane literature smo opisali s Priceovim indeksom (delež referenc, "
                f"starih največ {w} let; Price, 1970), mediano starosti referenc in "
                "citirajočo razpolovno dobo (citing half-life) po letu objave."
            )
        else:
            out.append(
                "The age of the cited literature was described by Price's index (the share of "
                f"references at most {w} years old; Price, 1970), the median reference age and "
                "the citing half-life by publication year."
            )
        refs.append(_REFS["price"])

    if getattr(bib, "concentration_df", None) is not None:
        cp = p.get("concentration", {})
        ent = cp.get("entity", "entities")
        what = "citatov" if cp.get("value_col") else "produkcije"
        what_en = "citations" if cp.get("value_col") else "output"
        if language == "sl":
            out.append(
                f"Koncentracijo {what} po {ent} smo ocenili z Ginijevim koeficientom "
                "(Gini, 1921), Theilovim indeksom, Herfindahl-Hirschmanovim indeksom in "
                "Lorenzovo krivuljo."
            )
        else:
            out.append(
                f"The concentration of {what_en} across {ent} was assessed with the Gini "
                "coefficient (Gini, 1921), the Theil index, the Herfindahl–Hirschman index "
                "and the Lorenz curve."
            )
        refs.append(_REFS["gini"])

    if getattr(bib, "author_careers_df", None) is not None:
        if language == "sl":
            out.append(
                "Kariere avtorjev smo opisali z akademsko starostjo znotraj korpusa (od prvega "
                "dokumenta v zbirki), produktivnostjo na aktivno leto, lokalnim h-indeksom in "
                "m-kvocientom (Hirsch, 2005)."
            )
        else:
            out.append(
                "Author careers were characterised by within-corpus academic age (since the "
                "first document in the collection), productivity per active year, the local "
                "h-index and the m-quotient (Hirsch, 2005)."
            )
        refs.append(_REFS["hirsch_m"])

    if getattr(bib, "heaps_params", None) is not None:
        hp = p.get("fit_heaps_law", {})
        field = hp.get("field", "keywords")
        beta = bib.heaps_params.get("beta")
        b_txt = f" (β = {beta:.2f})" if beta is not None and beta == beta else ""
        if language == "sl":
            out.append(
                f"Rast besedišča polja »{field}« smo opisali s Heapsovim zakonom V(n) = K·n^β"
                f"{b_txt} (Heaps, 1978), ocenjenim z linearno regresijo na log-log skali."
            )
        else:
            out.append(
                f"Vocabulary growth of the '{field}' field was modelled with Heaps' law "
                f"V(n) = K·n^β{b_txt} (Heaps, 1978), estimated by log-log linear regression."
            )
        refs.append(_REFS["heaps"])

    if getattr(bib, "indicator_ci_df", None) is not None or getattr(bib, "indicator_ci_result", None) is not None:
        ip = p.get("indicator_ci", {})
        nb, ci = ip.get("n_boot", 1000), ip.get("ci", 0.95)
        stat = ip.get("statistic", "mean")
        if language == "sl":
            out.append(
                f"Negotovost indikatorja ({stat}) smo ocenili z neparametričnim bootstrapom "
                f"({nb} ponovitev, {int(ci * 100)} % percentilni interval zaupanja; "
                "Efron & Tibshirani, 1993)."
            )
        else:
            out.append(
                f"Indicator uncertainty ({stat}) was quantified with a non-parametric bootstrap "
                f"({nb} resamples, {int(ci * 100)}% percentile confidence intervals; "
                "Efron & Tibshirani, 1993)."
            )
        refs.append(_REFS["bootstrap"])

    if getattr(bib, "uncitedness_df", None) is not None:
        if language == "sl":
            out.append("Delež necitiranih dokumentov smo spremljali po letu objave.")
        else:
            out.append("The share of uncited documents was tracked by publication year.")

    # --- 2.19, drugi paket -------------------------------------------------
    if getattr(bib, "prisma_df", None) is not None:
        if language == "sl":
            out.append("Postopek iskanja in izbora dokumentov je prikazan z diagramom poteka PRISMA 2020 (Page idr., 2021).")
        else:
            out.append("The search and selection process is reported in a PRISMA 2020 flow diagram (Page et al., 2021).")
        refs.append(_REFS["prisma"])

    cm = p.get("complete_metadata")
    if cm:
        src = " and ".join(s.capitalize() if s != "openalex" else "OpenAlex" for s in cm.get("sources", []))
        if language == "sl":
            out.append(f"Manjkajoče metapodatke smo dopolnili po DOI iz {src} ({cm.get('n_changed', 0)} dopolnjenih celic; obstoječe vrednosti niso bile prepisane).")
        else:
            out.append(f"Missing metadata were completed by DOI lookup in {src} ({cm.get('n_changed', 0)} cells filled; existing values were never overwritten).")
        refs.append(_REFS["openalex"])

    jr = p.get("journal_ranking")
    if jr:
        if language == "sl":
            out.append(f"Revije smo uvrstili po lestvici revij ({jr.get('n_matched', 0)} dokumentov ujetih po ISSN ali naslovu).")
        else:
            out.append(f"Sources were matched to a journal ranking list ({jr.get('n_matched', 0)} documents matched by ISSN or title).")

    if getattr(bib, "funders_df", None) is not None:
        if language == "sl":
            out.append("Financiranje smo analizirali iz podatkov o financerjih (Scopus Funding Details): najpogostejši financerji, delež financiranih dokumentov po letih in primerjava citiranosti financiranih in nefinanciranih dokumentov (MNCS, Mann–Whitneyjev U-test).")
        else:
            out.append("Funding was analysed from funding acknowledgements (Scopus Funding Details): leading funders, the share of funded documents by year, and the citation impact of funded versus unfunded documents (MNCS, Mann–Whitney U test).")

    if getattr(bib, "reference_patterns_df", None) is not None:
        rp = p.get("rpys_patterns", {})
        if language == "sl":
            out.append(f"Citirane reference smo na podlagi njihovih citatnih zgodovin razvrstili v vzorce (hot paper, constant performer, life cycle, sleeping beauty; van Raan, 2004), RPYS pa smo računali z {'vzvratno' if rp.get('median_mode') == 'backward' else 'centrirano'} drsečo mediano čez {rp.get('median_window', 5)} let.")
        else:
            out.append(f"Cited references were classified by their citation histories into patterns (hot paper, constant performer, life cycle, sleeping beauty; van Raan, 2004); RPYS used a {rp.get('median_mode', 'centered')} {rp.get('median_window', 5)}-year rolling median.")
        refs.append(_REFS["sleeping"]); refs.append(_REFS["rpys"])

    if getattr(bib, "diachronic_graphs", None):
        dp = p.get("diachronic_networks", {})
        if language == "sl":
            out.append(f"Diakrono dinamiko smo prikazali s sobesednimi mrežami polja »{dp.get('field')}« po obdobjih ({', '.join(getattr(bib, 'diachronic_labels', []))}) s fiksno množico {dp.get('top_n')} najpogostejših ključnih besed in skupno razporeditvijo vozlišč.")
        else:
            out.append(f"Diachronic dynamics were shown with co-word networks of the '{dp.get('field')}' field per period ({', '.join(getattr(bib, 'diachronic_labels', []))}) using a fixed set of the {dp.get('top_n')} most frequent keywords and a shared layout.")
        refs.append(_REFS["louvain"])

    if getattr(bib, "cluster_timeline_df", None) is not None:
        if language == "sl":
            out.append("Klastre ključnih besed smo prikazali v časovnici (vsak klaster na svojem pasu, vozlišča po povprečnem letu pojavljanja).")
        else:
            out.append("Keyword clusters were visualised as a timeline (one lane per cluster, nodes placed at their mean year of occurrence).")

    if getattr(bib, "structural_variation_df", None) is not None:
        if language == "sl":
            out.append("Strukturno variacijo (Chen, 2012) smo ocenili za vsak dokument: spremembo modularnosti, cluster linkage in centrality divergence sociacijske mreže referenc, zgrajene iz prej objavljenih dokumentov.")
        else:
            out.append("Structural variation (Chen, 2012) was computed for each document: modularity change rate, cluster linkage and centrality divergence of the reference co-citation network built from earlier documents.")
        refs.append(_REFS["sva"])

    if getattr(bib, "dual_map_flows_df", None) is not None:
        dm = p.get("dual_map", {})
        if language == "sl":
            out.append(f"Tokove znanja med disciplinami smo prikazali z dual-map prekrivanjem (Chen & Leydesdorff, 2014) na podlagi klasifikacije Scopus ASJC (raven »{dm.get('level', 'Area')}«, {'frakcijsko' if dm.get('fractional', True) else 'polno'} štetje).")
        else:
            out.append(f"Knowledge flows between disciplines were shown with a dual-map overlay (Chen & Leydesdorff, 2014) based on the Scopus ASJC classification ('{dm.get('level', 'Area')}' level, {'fractional' if dm.get('fractional', True) else 'full'} counting).")
        refs.append(_REFS["dual_map"])

    sb = p.get("snowball")
    if sb:
        if language == "sl":
            out.append(f"Korpus smo razširili s citatnim snežnim kepanjem prek OpenAlex ({sb.get('direction')}, {sb.get('rounds')} krog(ov), najmanj {sb.get('min_links')} povezav na semenski set; Priem idr., 2022).")
        else:
            out.append(f"The corpus was expanded by citation snowballing through OpenAlex ({sb.get('direction')}, {sb.get('rounds')} round(s), at least {sb.get('min_links')} link(s) to the seed set; Priem et al., 2022).")
        refs.append(_REFS["openalex"])

    te = p.get("extract_terms")
    if te:
        if language == "sl":
            out.append(f"Izraze smo iz polj {', '.join(te.get('columns', []))} izluščili po metodi VOSviewer (samostalniške fraze do {te.get('max_len')} besed, najmanj {te.get('min_occurrences')} pojavitev, obdržanih {int(100 * te.get('share', 0.6))} % najrelevantnejših izrazov; van Eck & Waltman, 2011).")
        else:
            out.append(f"Terms were extracted from {', '.join(te.get('columns', []))} following the VOSviewer approach (noun phrases up to {te.get('max_len')} words, at least {te.get('min_occurrences')} occurrences, the {int(100 * te.get('share', 0.6))} % most relevant terms retained; van Eck & Waltman, 2011).")
        refs.append(_REFS["vos_terms"])

    mg = p.get("merge_databases")
    if mg:
        srcs = ", ".join(DB_NAMES.get(x, x) for x in mg.get("sources", []))
        how = {"doi": "DOI", "title": "normaliziranih naslovov", "doi_title": "DOI (brez DOI po naslovu)"}.get(mg.get("match_key"), mg.get("match_key"))
        how_en = {"doi": "DOI", "title": "normalised titles", "doi_title": "DOI (title when no DOI)"}.get(mg.get("match_key"), mg.get("match_key"))
        pol = "unija stolpcev" if mg.get("columns") == "union" else "presek stolpcev"
        pol_en = "union of columns" if mg.get("columns") == "union" else "intersection of columns"
        if language == "sl":
            out.append(f"Zapise iz baz {srcs} smo združili po {how} ({pol}); od {mg.get('n_input')} zapisov je po odstranitvi {mg.get('n_removed')} podvojenih ostalo {mg.get('n_output')} enoličnih dokumentov.")
        else:
            out.append(f"Records from {srcs} were merged by {how_en} ({pol_en}); of {mg.get('n_input')} records, {mg.get('n_removed')} duplicates were removed, leaving {mg.get('n_output')} unique documents.")

    if getattr(bib, "cited_sources_df", None) is not None:
        if language == "sl":
            out.append(
                "Znanjsko osnovo področja smo opisali z najpogosteje citiranimi revijami v "
                "referencah ter njihovim prekrivanjem z revijami, v katerih so objavljeni "
                "analizirani dokumenti."
            )
        else:
            out.append(
                "The knowledge base of the field was described by the most frequently cited "
                "sources in the reference lists and their overlap with the publishing sources "
                "of the analysed documents."
            )

    return out, refs


def _software_sentence(language):
    try:
        from biblium import __version__ as bv
    except Exception:
        bv = "2.18"
    py = platform.python_version()
    if language == "sl":
        return (f"Vse analize so bile izvedene s knjižnico biblium v{bv} "
                f"(Python {py}).")
    return (f"All analyses were performed with the biblium library v{bv} "
            f"(Python {py}).")


def generate_methods_text(bib, language="en", include_references=True):
    """
    Sestavi Methods odstavek iz stanja BiblioStats objekta.

    Parameters
    ----------
    bib : BiblioStats
        Objekt z izvedenimi analizami.
    language : {"en", "sl"}
    include_references : bool
        Če True, doda seznam metodoloških referenc.

    Returns
    -------
    str
    """
    parts = [_dataset_sentence(bib, language)]
    sentences, refs = _analysis_sentences(bib, language)
    parts.extend(sentences)
    parts.append(_software_sentence(language))
    text = " ".join(parts)
    if include_references and refs:
        head = "Metodološke reference:" if language == "sl" else "Methodological references:"
        text += "\n\n" + head + "\n" + "\n".join(f"- {r}" for r in sorted(set(refs)))
    return text
