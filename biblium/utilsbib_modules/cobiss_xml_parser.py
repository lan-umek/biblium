# -*- coding: utf-8 -*-
"""
biblium.utilsbib_modules.cobiss_xml_parser
==========================================

XML parser for COBISS+ personal bibliographies in the structured
``format=X`` representation.

Background
----------
COBISS+ direct links accept ``?format=X`` and respond with structured
XML conforming to IZUM's published schema:

    https://home.izum.si/cobiss/xml/bibliography.xsd

This is the *recommended* parsing path because it is robust against
HTML/UI redesigns. The surface known from public documentation:

- Root element: ``<Bibliography>`` with attributes
  ``biblioType``, ``teamType``, ``represent``, ``code``, ``period``.
- Top-level children: ``<Name>``, ``<OrgName>``, ``<Title>``.
- Per-record children sit under one of several typology buckets;
  the *exact* element / attribute names for individual records,
  authors, citations, etc. require a real sample to confirm.

Status
------
**Skeleton awaiting a real XML sample.** This module exposes the same
public surface as :mod:`biblium.utilsbib_modules.cobiss_parser`
(``parse_cobiss_xml(...)`` returns ``(records, metadata)`` of the
exact same dataclass and metadata types) so callers can switch
between parsers transparently. The actual element-traversal logic
will be filled in once an XML response has been validated against
the schema.

Calling :func:`parse_cobiss_xml` today raises ``NotImplementedError``
with a precise message identifying what is needed; this is intentional
so we never silently produce bogus results.

The skeleton already implements:

- a permissive XML loader that tolerates BOM, namespaces, and surrounding
  whitespace,
- top-level ``<Bibliography>`` attribute extraction (``code``, ``period``,
  etc.) and ``<Name>`` / ``<OrgName>`` / ``<Title>`` text extraction,
- a deferred ``_parse_unit(...)`` hook that callers can monkey-patch
  in tests and that the production code will fill in.

Author: Lan Umek
Version: 2.16.0
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

from biblium.utilsbib_modules.cobiss_parser import ParsedCobissRecord


# =============================================================================
# DOCUMENT-LEVEL METADATA (mirror of the HTML parser's _ParseMetadata)
# =============================================================================


@dataclass
class _XmlParseMetadata:
    """Document-level metadata extracted from a ``<Bibliography>`` root."""
    researcher_name: Optional[str] = None
    researcher_code: Optional[str] = None
    period: Optional[str] = None
    biblio_type: Optional[str] = None
    org_name: Optional[str] = None
    title: Optional[str] = None
    n_records: int = 0
    n_records_per_typology: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# XML LOADING
# =============================================================================


def _load_xml(source: str, *, is_path: Optional[bool] = None):
    """
    Load XML from a string, bytes, or file path.

    Returns the root ``Element``. The function strips an XML declaration's
    BOM if present and tolerates leading whitespace.

    Notes
    -----
    Uses Python's ``xml.etree.ElementTree`` from the standard library.
    We do *not* use ``lxml`` to keep biblium's dependency tree small;
    if the schema turns out to need XSD validation, we can add it
    optionally later.
    """
    # Lazy import so this module is importable even without an XML payload
    import xml.etree.ElementTree as ET

    # Resolve source -> raw text
    if is_path is None:
        is_path = (
            isinstance(source, str)
            and len(source) < 4096
            and "\n" not in source
            and "<" not in source
            and os.path.exists(source)
        )
    if is_path:
        with open(source, "rb") as fh:
            data = fh.read()
        try:
            text = data.decode("utf-8-sig")  # strips a UTF-8 BOM
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
    else:
        text = source.lstrip("\ufeff")

    text = text.strip()
    return ET.fromstring(text)


def _local_tag(elem) -> str:
    """Return the element's local name without namespace prefix."""
    tag = elem.tag
    if isinstance(tag, str) and "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _strip_ns(tag: str) -> str:
    """Helper to compare tags without namespace noise."""
    return tag.split("}", 1)[-1]


# =============================================================================
# DOCUMENT-LEVEL METADATA EXTRACTION
# =============================================================================


def _extract_document_metadata(root) -> _XmlParseMetadata:
    """
    Extract document-level metadata from the ``<Bibliography>`` root.

    According to the published schema, the root element carries the
    attributes ``biblioType``, ``teamType``, ``represent``, ``code``,
    ``period`` and contains ``<Name>``, ``<OrgName>``, ``<Title>``
    children (potentially among others). We pull whatever is present
    and ignore the rest.
    """
    meta = _XmlParseMetadata()

    if _local_tag(root).lower() != "bibliography":
        warnings.warn(
            f"Root element is <{_local_tag(root)}>, expected <Bibliography>. "
            f"Continuing with best-effort extraction.",
            UserWarning, stacklevel=2,
        )

    # Attributes -- read case-insensitively
    attrs = {k.lower(): v for k, v in root.attrib.items()}
    meta.researcher_code = attrs.get("code")
    meta.period = attrs.get("period")
    meta.biblio_type = attrs.get("bibliotype")

    # Children
    for child in root:
        local = _local_tag(child).lower()
        text = (child.text or "").strip()
        if not text:
            continue
        if local == "name":
            meta.researcher_name = text
        elif local == "orgname":
            meta.org_name = text
        elif local == "title":
            meta.title = text

    return meta


# =============================================================================
# RECORD-LEVEL EXTRACTION (deferred until we have a real sample)
# =============================================================================


def _parse_unit(unit_elem) -> Optional[ParsedCobissRecord]:
    """
    Convert a single bibliographic-unit XML element into a ``ParsedCobissRecord``.

    **Status:** placeholder. Calling this raises ``NotImplementedError``
    until we have a real ``format=X`` XML sample to validate against.
    The implementation will read elements documented in
    ``bibliography.xsd`` (we still need a real response to know the
    exact element / attribute names for fields like authors, ISSN,
    DOI, citation counts, ...).

    Test code may monkey-patch this function with a custom parser
    to exercise the high-level dispatcher (``parse_cobiss_xml``).
    """
    raise NotImplementedError(
        "XML record parsing is not yet implemented. "
        "Run a request against a CRIS or CONOR direct link with "
        "?format=X from a Slovenian IP, then update this module with "
        "the actual element / attribute names from the response."
    )


def _iter_unit_candidates(root):
    """
    Yield XML elements that *might* be bibliographic-unit records.

    Until the schema is confirmed, we use a permissive heuristic:
    any descendant element whose local name suggests a record-level
    container is yielded. The set of candidate names mirrors what
    appears in COBISS HTML / SICRIS APIs:

    - ``bibUnit``, ``bibliographicUnit``, ``unit``  (most likely)
    - ``record``, ``item``                            (generic fallbacks)
    - ``Unit``, ``BibliographicUnit``, ``Record``     (PascalCase variants)

    The real candidate name will be selected once a response sample
    is available. Anything that is *not* a unit element is silently
    skipped.
    """
    record_tag_set = {
        "bibunit", "bibliographicunit", "unit",
        "record", "item",
    }
    for elem in root.iter():
        if _local_tag(elem).lower() in record_tag_set:
            yield elem


# =============================================================================
# PUBLIC API (mirrors cobiss_parser.parse_cobiss_html)
# =============================================================================


def parse_cobiss_xml(
    source: str,
    *,
    default_citation_source: Literal["wos", "scopus"] = "wos",
    is_path: Optional[bool] = None,
) -> Tuple[List[ParsedCobissRecord], _XmlParseMetadata]:
    """
    Parse a COBISS+ personal bibliography in the ``format=X`` XML form.

    Returns ``(records, metadata)`` matching the shape of
    :func:`biblium.utilsbib_modules.cobiss_parser.parse_cobiss_html`,
    so callers can choose between the HTML and XML paths transparently.

    Parameters
    ----------
    source : str
        Either an XML string (with or without BOM / declaration) or
        a path to an XML file.
    default_citation_source : {"wos", "scopus"}, default "wos"
        Forwarded to ``ParsedCobissRecord`` construction.
    is_path : bool, optional
        Force interpretation of ``source`` as a path. Auto-detected
        by default.

    Raises
    ------
    NotImplementedError
        Currently raised when at least one bibliographic-unit element
        is found, because record-level parsing is not yet wired up
        (awaiting a real ``format=X`` response). Document-level
        metadata extraction still runs and is included in the warning.
    """
    root = _load_xml(source, is_path=is_path)
    meta = _extract_document_metadata(root)

    records: List[ParsedCobissRecord] = []
    candidates_seen = 0
    for unit in _iter_unit_candidates(root):
        candidates_seen += 1
        rec = _parse_unit(unit)  # will raise NotImplementedError
        if rec is not None:
            records.append(rec)

    if candidates_seen == 0:
        # No record candidates means we either have an empty bibliography
        # OR our heuristic missed the unit elements. Both are noteworthy.
        warnings.warn(
            "No bibliographic-unit candidate elements were found in the "
            "XML payload. Either the bibliography is empty, or the unit "
            "element name differs from the heuristic set "
            "(bibUnit, unit, record, item, ...). Document metadata was "
            "still extracted.",
            UserWarning, stacklevel=2,
        )

    meta.n_records = len(records)
    return records, meta


__all__ = [
    "parse_cobiss_xml",
]
