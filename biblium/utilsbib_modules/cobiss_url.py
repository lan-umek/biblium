# -*- coding: utf-8 -*-
"""
biblium.utilsbib_modules.cobiss_url
===================================

Pure-logic helpers for classifying and rewriting COBISS+ URLs.

This module knows about three URL families:

1. **CRIS direct link** — based on a 5-digit ARIS researcher's code:

    ``https://bib.cobiss.net/biblioweb/direct/<country>/<lang>/cris/<code>``

   Documented parameters: ``fromyear``, ``toyear``, ``formatbib``,
   ``format``, ``citation``, ``idlist``, ``sort``, ``abstract``.

2. **CONOR direct link** — based on a CONOR.SI authority record id:

    ``https://bib.cobiss.net/biblioweb/direct/<country>/<lang>/conor/<conor_id>``

   Same parameter set as CRIS.

3. **Pre-rendered HTML report** — a static file produced by COBISS+ after
   the user has filled in the personal-bibliography form:

    ``https://bib.cobiss.net/bibliographies/<country>/webBiblio/bib201_<ts>_<code>.html``

   The format is *fixed* (HTML); query-string parameters are ignored.

The first two families accept ``?format=X`` and return XML conforming
to IZUM's published schema (``home.izum.si/cobiss/xml/bibliography.xsd``);
this is much more robust to parse than HTML scraping. The third family
must be parsed as HTML.

Public API
----------
classify_cobiss_url(url) -> CobissUrlInfo
    Tell which kind of URL we have.

prepare_request_url(url, *, format='auto', citation=True) -> str
    Rewrite a CRIS/CONOR direct link to request XML (or HTML/XML
    explicitly) without dropping any user-supplied query parameters.

The module is pure: no network, no I/O. It is exercised by
``test_cobiss_url`` in the test suite.

Author: Lan Umek
Version: 2.16.0
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlparse,
    urlunparse,
)


# =============================================================================
# CLASSIFICATION
# =============================================================================


# Type alias for the URL families we support
UrlKind = Literal["cris_direct", "conor_direct", "prerendered_html", "unknown"]


@dataclass(frozen=True)
class CobissUrlInfo:
    """
    Structured information about a COBISS+ URL.

    Attributes
    ----------
    kind : {"cris_direct", "conor_direct", "prerendered_html", "unknown"}
        Which family the URL belongs to.
    code : str or None
        For CRIS links: the 5-digit researcher's code.
        For CONOR links: the CONOR.SI-ID (numeric).
        For pre-rendered HTML: the trailing numeric suffix from the file
        name (informational only).
        ``None`` if the URL does not match any known pattern.
    country : str or None
        Country segment, e.g. ``"si"`` for Slovenia. ``None`` for
        pre-rendered HTML URLs (where the country sits in a different
        position).
    lang : str or None
        UI language code, e.g. ``"slv"`` or ``"eng"``.
    supports_format_param : bool
        ``True`` if appending ``?format=X|H|...`` will affect what the
        server returns. Pre-rendered HTML reports return ``False`` here.
    """
    kind: UrlKind
    code: Optional[str]
    country: Optional[str]
    lang: Optional[str]
    supports_format_param: bool


# Regex for the *direct* path:
#   /biblioweb/direct/<country>/<lang>/(cris|conor)/<code>
_DIRECT_PATH_RE = re.compile(
    r"^/biblioweb/direct/(?P<country>[a-z]{2})/(?P<lang>[a-z]{2,3})/"
    r"(?P<which>cris|conor)/(?P<code>[A-Za-z0-9]+)/?$"
)

# Regex for the pre-rendered HTML path:
#   /bibliographies/<country>/webBiblio/bib201_<timestamp_parts>_<code>.html
# Real-world filenames have the form bib201_<YYYYMMDD>_<HHMMSS>_<code>.html
# (date and time are separate underscore-delimited segments). We accept any
# sequence of numeric-and-underscore segments before the trailing code so the
# regex stays robust to small generator changes.
_PRERENDERED_PATH_RE = re.compile(
    r"^/bibliographies/(?P<country>[a-z]{2})/webBiblio/"
    r"bib\d+(?:_\d+)+_(?P<code>\d+)\.html?$",
    re.IGNORECASE,
)


def classify_cobiss_url(url: str) -> CobissUrlInfo:
    """
    Classify a COBISS+ URL.

    Returns ``CobissUrlInfo(kind="unknown", ...)`` for URLs that do
    not match any known pattern; this lets callers decide what to do
    (e.g. raise, fall back to plain fetch + HTML parser, ...).
    """
    try:
        parsed = urlparse(url)
    except (ValueError, TypeError):
        return CobissUrlInfo("unknown", None, None, None, False)

    # Match against the direct-link pattern first (most informative)
    m_direct = _DIRECT_PATH_RE.match(parsed.path)
    if m_direct:
        which = m_direct.group("which")
        kind: UrlKind = "cris_direct" if which == "cris" else "conor_direct"
        return CobissUrlInfo(
            kind=kind,
            code=m_direct.group("code"),
            country=m_direct.group("country"),
            lang=m_direct.group("lang"),
            supports_format_param=True,
        )

    m_pre = _PRERENDERED_PATH_RE.match(parsed.path)
    if m_pre:
        return CobissUrlInfo(
            kind="prerendered_html",
            code=m_pre.group("code"),
            country=m_pre.group("country"),
            lang=None,
            supports_format_param=False,
        )

    return CobissUrlInfo("unknown", None, None, None, False)


# =============================================================================
# QUERY-STRING REWRITING
# =============================================================================


# Single source of truth for the COBISS ``format=`` parameter codes
# (see https://bib.cobiss.net/biblioweb/info/si/eng/help#links).
_FORMAT_LETTER = {
    "html": "H",
    "xml": "X",
    "latex": "L",
    "pdf": "P",
    "txt": "T",
}


def prepare_request_url(
    url: str,
    *,
    format: Literal["auto", "xml", "html", "latex", "pdf", "txt"] = "auto",
    citation: Optional[bool] = True,
) -> str:
    """
    Return a (possibly rewritten) URL ready to be GET-fetched.

    Behaviour by URL kind:

    - **CRIS / CONOR direct link** — query parameters are merged.
      User-provided values *win* over our suggestions; we add
      ``format=...`` and ``citation=true`` only if they are missing
      and the user did not pass ``format="auto"`` with the request to
      leave the URL alone.
    - **Pre-rendered HTML** — returned unchanged. The format is fixed
      and any query parameters are ignored by the server.
    - **Unknown** — returned unchanged; the caller can still try a
      plain GET (the host allowlist is enforced separately).

    Parameters
    ----------
    url : str
        The URL the user provided.
    format : {"auto", "xml", "html", "latex", "pdf", "txt"}, default "auto"
        Which display format to request. ``"auto"`` defers to the user:
        if their URL already has ``format=``, we leave it; if not, we
        add ``format=X`` (XML) for direct links because XML is much
        easier to parse reliably than HTML.
    citation : bool or None, default True
        If ``True`` (default), add ``citation=true`` when missing so
        WoS/Scopus citation counts are included. ``False`` strips
        any existing ``citation=`` parameter. ``None`` leaves the
        parameter alone.

    Examples
    --------
    >>> prepare_request_url(
    ...     "https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519"
    ... )
    'https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519?format=X&citation=true'

    >>> prepare_request_url(
    ...     "https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519?fromyear=2021",
    ...     format="xml",
    ... )
    'https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519?fromyear=2021&format=X&citation=true'

    >>> # Pre-rendered HTML is left alone
    >>> prepare_request_url(
    ...     "https://bib.cobiss.net/bibliographies/si/webBiblio/bib201_x.html",
    ...     format="xml",
    ... )
    'https://bib.cobiss.net/bibliographies/si/webBiblio/bib201_x.html'
    """
    info = classify_cobiss_url(url)
    if not info.supports_format_param:
        return url

    parsed = urlparse(url)
    # Use parse_qsl to preserve order and duplicate-key behaviour
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    keys_present = {k.lower() for k, _ in pairs}

    # ---- format= ----
    if format != "auto":
        letter = _FORMAT_LETTER[format]
        # Replace any existing format= entries (case-insensitive) with our value
        pairs = [(k, v) for (k, v) in pairs if k.lower() != "format"]
        pairs.append(("format", letter))
    else:
        # auto: add format=X only if missing
        if "format" not in keys_present:
            pairs.append(("format", _FORMAT_LETTER["xml"]))

    # ---- citation= ----
    if citation is True:
        if "citation" not in keys_present:
            pairs.append(("citation", "true"))
    elif citation is False:
        pairs = [(k, v) for (k, v) in pairs if k.lower() != "citation"]
    # citation is None -> leave as-is

    new_query = urlencode(pairs, doseq=True)
    rebuilt = parsed._replace(query=new_query)
    return urlunparse(rebuilt)


# =============================================================================
# CONVENIENCE
# =============================================================================


def is_xml_format(url: str) -> bool:
    """
    True if the URL's query string explicitly requests XML (``format=X``).

    Used by the client to decide whether to dispatch the response
    bytes to the XML parser or the HTML parser.
    """
    try:
        q = dict(parse_qsl(urlparse(url).query, keep_blank_values=True))
    except (ValueError, TypeError):
        return False
    fmt = q.get("format") or q.get("FORMAT") or q.get("Format")
    if not fmt:
        return False
    return fmt.upper() == "X"


__all__ = [
    "CobissUrlInfo",
    "UrlKind",
    "classify_cobiss_url",
    "prepare_request_url",
    "is_xml_format",
]
