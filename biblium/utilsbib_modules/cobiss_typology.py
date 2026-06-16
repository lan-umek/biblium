# -*- coding: utf-8 -*-
"""
biblium.utilsbib_modules.cobiss_typology
========================================

Mapping of COBISS typology codes (e.g. ``"1.01"``, ``"2.04"``) to their
official Slovenian and English labels and to broad biblium-canonical
``Document Type`` categories.

The codes are taken from IZUM's *Typology of Documents/Works for
Bibliography Management in COBISS* (revision of 29 September 2025).
Source:
https://home.izum.si/COBISS/bibliografije/Tipologija_eng.pdf

Public components
-----------------
TYPOLOGY
    Dictionary keyed by code (str), with values
    ``(label_sl, label_en, document_type)``.

typology_label
    Look up a label by code and language.

typology_to_document_type
    Map a code to the closest match among the canonical biblium
    ``Document Type`` categories: ``"Article"``, ``"Review"``,
    ``"Conference Paper"``, ``"Book"``, ``"Book Chapter"``,
    ``"Editorial"``, ``"Other"``.

Author: Lan Umek
Version: 2.16.0
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

# ----------------------------------------------------------------------------
# Code -> (Slovenian label, English label, biblium Document Type)
#
# document_type is chosen to align with the existing biblium naming used in
# the Scopus reader (Authors, Title, Year, Document Type, ...). Mapping rules:
#   * 1.01, 1.02, 1.03, 1.04 (review-like article)        -> Article / Review
#   * 1.06, 1.08, 1.10, 1.12, 1.13                        -> Conference Paper
#   * 1.16, 1.17, 1.20                                    -> Book Chapter
#   * 1.21, 1.22                                          -> Article  (popular)
#   * 1.05 editorial / preface / addendum                 -> Editorial
#   * 2.01..2.03, 2.05, 2.06                              -> Book
#   * 2.04 conference proceedings                         -> Book
#   * 2.13 dictionary / encyclopedia                      -> Book
#   * 2.20 research data                                  -> Other
#   * 2.08 doctoral, 2.09 master's                        -> Other
# ----------------------------------------------------------------------------

# Type alias for the canonical biblium Document Type
DocumentType = Literal[
    "Article",
    "Review",
    "Conference Paper",
    "Book",
    "Book Chapter",
    "Editorial",
    "Other",
]


TYPOLOGY: Dict[str, Tuple[str, str, str]] = {
    # ----- 1.x ARTICLES AND OTHER COMPONENT PARTS -----
    "1.01": ("Izvirni znanstveni članek",
             "Original Scientific Article",
             "Article"),
    "1.02": ("Pregledni znanstveni članek",
             "Review Article",
             "Review"),
    "1.03": ("Drugi znanstveni članki",
             "Other Scientific Articles",
             "Article"),
    "1.04": ("Strokovni članek",
             "Professional Article",
             "Article"),
    "1.05": ("Poljudni članek",
             "Popular Article",
             "Article"),
    "1.06": ("Objavljeni znanstveni prispevek na konferenci (vabljeno predavanje)",
             "Published Scientific Conference Contribution (Invited Lecture)",
             "Conference Paper"),
    "1.07": ("Objavljeni strokovni prispevek na konferenci (vabljeno predavanje)",
             "Published Professional Conference Contribution (Invited Lecture)",
             "Conference Paper"),
    "1.08": ("Objavljeni znanstveni prispevek na konferenci",
             "Published Scientific Conference Contribution",
             "Conference Paper"),
    "1.09": ("Objavljeni strokovni prispevek na konferenci",
             "Published Professional Conference Contribution",
             "Conference Paper"),
    "1.10": ("Objavljeni povzetek znanstvenega prispevka na konferenci (vabljeno predavanje)",
             "Published Scientific Conference Contribution Abstract (Invited Lecture)",
             "Conference Paper"),
    "1.11": ("Objavljeni povzetek strokovnega prispevka na konferenci (vabljeno predavanje)",
             "Published Professional Conference Contribution Abstract (Invited Lecture)",
             "Conference Paper"),
    "1.12": ("Objavljeni povzetek znanstvenega prispevka na konferenci",
             "Published Scientific Conference Contribution Abstract",
             "Conference Paper"),
    "1.13": ("Objavljeni povzetek strokovnega prispevka na konferenci",
             "Published Professional Conference Contribution Abstract",
             "Conference Paper"),
    "1.16": ("Samostojni znanstveni sestavek ali poglavje v monografski publikaciji",
             "Independent Scientific Component Part or a Chapter in a Monograph",
             "Book Chapter"),
    "1.17": ("Samostojni strokovni sestavek ali poglavje v monografski publikaciji",
             "Independent Professional Component Part or a Chapter in a Monograph",
             "Book Chapter"),
    "1.18": ("Strokovni sestavek v slovarju, enciklopediji ali leksikonu",
             "Professional Component Part in a Dictionary, Encyclopaedia or Lexicon",
             "Book Chapter"),
    "1.19": ("Recenzija, prikaz knjige, kritika",
             "Book Review, Critique",
             "Editorial"),
    "1.20": ("Predgovor, spremna beseda",
             "Preface, Afterword",
             "Editorial"),
    "1.21": ("Polemika, diskusijski prispevek, komentar",
             "Polemic, Discussion, Commentary",
             "Editorial"),
    "1.22": ("Intervju",
             "Interview",
             "Article"),
    "1.25": ("Drugi sestavni deli",
             "Other Component Parts",
             "Other"),
    # ----- 2.x MONOGRAPHS AND OTHER COMPLETED WORKS -----
    "2.01": ("Znanstvena monografija",
             "Scientific Monograph",
             "Book"),
    "2.02": ("Strokovna monografija",
             "Professional Monograph",
             "Book"),
    "2.03": ("Univerzitetni, visokošolski ali višješolski učbenik z recenzijo",
             "University, Higher Education or Higher Vocational Textbook with Peer Review",
             "Book"),
    "2.04": ("Srednješolski, osnovnošolski ali drugi učbenik z recenzijo",
             "Secondary, Primary School or Other Textbook with Peer Review",
             "Book"),
    "2.05": ("Drugo učno gradivo",
             "Other Educational Material",
             "Book"),
    "2.06": ("Enciklopedija, slovar, leksikon, priročnik, atlas, zemljevid",
             "Encyclopaedia, Dictionary, Lexicon, Manual, Atlas, Map",
             "Book"),
    "2.07": ("Bibliografija, kazalo ipd.",
             "Bibliography, Index, etc.",
             "Other"),
    "2.08": ("Doktorska disertacija",
             "Doctoral Dissertation",
             "Other"),
    "2.09": ("Magistrsko delo",
             "Master's Thesis",
             "Other"),
    "2.11": ("Diplomsko delo",
             "Undergraduate Thesis",
             "Other"),
    "2.12": ("Končno poročilo o rezultatih raziskav",
             "Final Research Report",
             "Other"),
    "2.13": ("Elaborat, predštudija, študija",
             "Treatise, Preliminary Study, Study",
             "Other"),
    "2.14": ("Projektna dokumentacija (idejni projekt, izvedbeni projekt)",
             "Project Documentation (Conceptual, Executive)",
             "Other"),
    "2.15": ("Izvedensko mnenje, arbitražna odločba",
             "Expert Opinion, Arbitration Decision",
             "Other"),
    "2.16": ("Umetniško delo",
             "Artistic Work",
             "Other"),
    "2.17": ("Katalog razstave",
             "Exhibition Catalogue",
             "Other"),
    "2.18": ("Strokovni standard, priporočilo, navodilo, pravilnik, predpis",
             "Professional Standard, Recommendation, Instruction, Regulation",
             "Other"),
    "2.19": ("Radijska ali televizijska oddaja, podkast, intervju, novinarska konferenca",
             "Radio or Television Broadcast, Podcast, Interview, Press Conference",
             "Other"),
    "2.20": ("Zaključena znanstvena zbirka raziskovalnih podatkov",
             "Research Data",
             "Other"),
    "2.21": ("Programska oprema",
             "Software",
             "Other"),
    "2.24": ("Patent",
             "Patent",
             "Other"),
    "2.25": ("Druge monografije in druga zaključena dela",
             "Other Monographs and Other Completed Works",
             "Other"),
    # ----- 3.x PERFORMED WORKS (EVENTS) -----
    "3.10": ("Umetniška poustvaritev",
             "Artistic Re-Creation",
             "Other"),
    "3.11": ("Radijski ali tv dogodek",
             "Radio or Television Event",
             "Other"),
    "3.12": ("Razstava",
             "Exhibition",
             "Other"),
    "3.14": ("Predavanje na tuji univerzi",
             "Lecture at a Foreign University",
             "Other"),
    "3.15": ("Prispevek na konferenci brez natisa",
             "Conference Contribution without a Printed Version",
             "Other"),
    "3.16": ("Vabljeno predavanje na konferenci brez natisa",
             "Invited Conference Lecture without a Printed Version",
             "Other"),
    "3.25": ("Druga izvedena dela",
             "Other Performed Works",
             "Other"),
}


def typology_label(
    code: str,
    lang: Literal["sl", "en"] = "en",
) -> Optional[str]:
    """
    Return the official typology label for a given COBISS code.

    Parameters
    ----------
    code : str
        Typology code, e.g. ``"1.01"``. Whitespace is stripped.
    lang : {"sl", "en"}, default "en"
        Which language to return.

    Returns
    -------
    str or None
        The label, or ``None`` if the code is not in the table.
    """
    entry = TYPOLOGY.get(code.strip())
    if entry is None:
        return None
    return entry[0] if lang == "sl" else entry[1]


def typology_to_document_type(code: str) -> str:
    """
    Map a COBISS typology code to a canonical biblium ``Document Type``.

    Falls back to ``"Other"`` for unknown codes so downstream group-by
    operations always succeed.
    """
    entry = TYPOLOGY.get(code.strip())
    return entry[2] if entry is not None else "Other"


# Reverse lookup: Slovenian label -> code (used when parsing HTML headings
# that show only the label, not the code, e.g. "1.01 Izvirni znanstveni članek")
_LABEL_SL_TO_CODE: Dict[str, str] = {v[0].lower(): k for k, v in TYPOLOGY.items()}
_LABEL_EN_TO_CODE: Dict[str, str] = {v[1].lower(): k for k, v in TYPOLOGY.items()}


def code_from_label(label: str) -> Optional[str]:
    """
    Best-effort reverse lookup: typology code from a human-readable label.

    Tries Slovenian first, then English. Matches are case-insensitive.
    Returns ``None`` if the label does not appear in the canonical table.
    """
    needle = label.strip().lower()
    return _LABEL_SL_TO_CODE.get(needle) or _LABEL_EN_TO_CODE.get(needle)


__all__ = [
    "DocumentType",
    "TYPOLOGY",
    "typology_label",
    "typology_to_document_type",
    "code_from_label",
]
