# -*- coding: utf-8 -*-
"""
biblium.cobiss_api
==================

Client for fetching personal bibliographies from COBISS+ (Slovenian
Co-operative Online Bibliographic System and Services).

Two URL patterns are supported in 2.16:

1. **Pre-rendered HTML** (the primary, fully working path):
   ``https://bib.cobiss.net/bibliographies/si/webBiblio/bib201_<timestamp>_<code>.html``
   The user generates this URL by filling in the personal bibliography
   form on COBISS+ and copying the resulting URL.

2. **CRIS direct link** (best-effort, since IZUM may need to render the
   report on the fly):
   ``https://bib.cobiss.net/biblioweb/direct/si/eng/cris/<code>?fromyear=YYYY&toyear=YYYY``
   We fetch as-is and follow any redirect to a webBiblio HTML page.

Citation counts (Web of Science / Scopus TC, CI, CIAu) are visible in
COBISS+ output **only when the request originates from a Slovenian IP**
(``.si`` domain). When run from outside Slovenia, the parser will simply
record ``None`` for these columns and a ``UserWarning`` is emitted.

Public components
-----------------
CobissClient
    HTTP client with rate limiting, retry/backoff and on-disk caching.

fetch_personal_bibliography_to_csv
    Top-level convenience function: URL in, CSV file out.

Author: Lan Umek
Version: 2.16.0
"""

from __future__ import annotations

import hashlib
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from biblium.utilsbib_modules.cobiss_parser import (
    parse_cobiss_html,
    records_to_dataframe,
)


# =============================================================================
# CONSTANTS
# =============================================================================

COBISS_HOSTS = (
    "bib.cobiss.net",
    "plus.cobiss.net",
    "plus-legacy.cobiss.net",
    "cobiss.net",
    "cobiss.si",
    "home.izum.si",
)

DEFAULT_USER_AGENT = (
    "Biblium/2.16 (https://github.com/lan-umek/biblium; "
    "Python bibliometric analysis library) "
    "Mozilla/5.0 (compatible)"
)


# =============================================================================
# RESULT DATACLASS
# =============================================================================


@dataclass
class CobissFetchResult:
    """Container for the result of a personal-bibliography fetch."""
    url: str
    final_url: str
    n_records: int
    researcher_name: Optional[str]
    researcher_code: Optional[str]
    period: Optional[str]
    typology_counts: Dict[str, int]
    csv_path: Optional[str]
    elapsed_seconds: float
    # Which parser actually produced the records ("html" or "xml").
    # Set by ``CobissClient.fetch_personal_bibliography``; ``None``
    # when the fetch did not run a parser.
    format_used: Optional[str] = None


# =============================================================================
# CLIENT
# =============================================================================


class CobissClient:
    """
    HTTP client for the COBISS+ system.

    Parameters
    ----------
    user_agent : str, optional
        Custom User-Agent header. Defaults to a polite identifier that
        mentions Biblium and includes a contact URL.
    rate_limit_delay : float, default 1.0
        Minimum seconds between successive HTTP requests. Defaults to 1
        second to keep load on IZUM's small public service polite.
    timeout : int, default 30
        Per-request timeout in seconds.
    max_retries : int, default 3
        Number of retries on transient errors (5xx, connection errors).
    backoff_base : float, default 1.5
        Base for exponential backoff: wait = ``backoff_base ** attempt``
        seconds between retries.
    cache_dir : str, optional
        Directory for on-disk caching of fetched HTML responses. If
        ``None`` (default) caching is disabled.
    cache_ttl_seconds : int, default 86400
        Time-to-live for cached responses (default: 1 day).
    verbose : bool, default False
        If True, print progress messages.

    Examples
    --------
    >>> client = CobissClient(cache_dir="/tmp/cobiss_cache")
    >>> html = client.fetch(
    ...     "https://bib.cobiss.net/bibliographies/si/webBiblio/"
    ...     "bib201_20260430_111142_28519.html"
    ... )
    """

    def __init__(
        self,
        user_agent: Optional[str] = None,
        rate_limit_delay: float = 1.0,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_base: float = 1.5,
        cache_dir: Optional[str] = None,
        cache_ttl_seconds: int = 86400,
        verbose: bool = False,
    ):
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "The 'requests' library is required for the COBISS HTTP "
                "client. Install with: pip install requests"
            )
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.cache_dir = cache_dir
        self.cache_ttl_seconds = cache_ttl_seconds
        self.verbose = verbose

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            ),
            "Accept-Language": "sl,en;q=0.9",
        })
        self._last_request_time: float = 0.0
        self._request_count: int = 0

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_cobiss_url(url: str) -> bool:
        try:
            host = urlparse(url).hostname or ""
        except Exception:
            return False
        host = host.lower()
        return any(host == h or host.endswith("." + h) for h in COBISS_HOSTS)

    def _cache_path(self, url: str) -> Optional[str]:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
        return os.path.join(self.cache_dir, f"cobiss_{digest}.html")

    def _try_cache_read(self, url: str) -> Optional[Tuple[str, str]]:
        path = self._cache_path(url)
        if path is None or not os.path.exists(path):
            return None
        if (time.time() - os.path.getmtime(path)) > self.cache_ttl_seconds:
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                # First line stores the final URL (after redirects)
                first = fh.readline().rstrip("\n")
                final_url = first.removeprefix("# final_url: ")
                html = fh.read()
            if self.verbose:
                print(f"[cobiss] cache hit: {url} -> {path}")
            return html, final_url
        except OSError:
            return None

    def _cache_write(self, url: str, html: str, final_url: str) -> None:
        path = self._cache_path(url)
        if path is None:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(f"# final_url: {final_url}\n")
                fh.write(html)
            if self.verbose:
                print(f"[cobiss] cached: {path}")
        except OSError:
            pass  # silently ignore cache write failures

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)

    # ------------------------------------------------------------------
    # Public fetch
    # ------------------------------------------------------------------

    def fetch(self, url: str) -> Tuple[str, str]:
        """
        Fetch a COBISS+ URL and return ``(html, final_url)``.

        Honors the cache (if configured), rate limit, and retry policy.
        Raises ``ValueError`` if the URL is not on a COBISS host, or
        ``RuntimeError`` if all retries fail.
        """
        if not self._is_cobiss_url(url):
            raise ValueError(
                f"Refusing to fetch non-COBISS URL: {url}. "
                f"Allowed hosts: {COBISS_HOSTS}"
            )

        cached = self._try_cache_read(url)
        if cached is not None:
            return cached

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            self._rate_limit()
            try:
                if self.verbose:
                    print(f"[cobiss] GET (attempt {attempt}): {url}")
                response = self.session.get(
                    url,
                    timeout=self.timeout,
                    allow_redirects=True,
                )
                self._request_count += 1
                self._last_request_time = time.time()

                if 200 <= response.status_code < 300:
                    response.encoding = response.encoding or "utf-8"
                    html = response.text
                    final_url = response.url
                    self._cache_write(url, html, final_url)
                    return html, final_url

                # 5xx -> retry
                if 500 <= response.status_code < 600:
                    raise RuntimeError(
                        f"Server error {response.status_code} from {url}"
                    )

                # 4xx -> don't retry
                raise RuntimeError(
                    f"Client error {response.status_code} from {url}: "
                    f"{response.text[:200]!r}"
                )
            except (requests.exceptions.RequestException, RuntimeError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                wait = self.backoff_base ** attempt
                if self.verbose:
                    print(
                        f"[cobiss] attempt {attempt} failed: {exc}. "
                        f"Retrying in {wait:.1f}s..."
                    )
                time.sleep(wait)

        raise RuntimeError(
            f"Failed to fetch {url} after {self.max_retries} attempts. "
            f"Last error: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Personal bibliography
    # ------------------------------------------------------------------

    def fetch_personal_bibliography(
        self,
        url: str,
        *,
        default_citation_source: Literal["wos", "scopus"] = "wos",
        prefer_format: Literal["auto", "xml", "html"] = "auto",
    ) -> Tuple[pd.DataFrame, CobissFetchResult]:
        """
        Fetch and parse a personal bibliography from the given URL.

        Returns ``(dataframe, result)`` where ``dataframe`` is a
        biblium-shaped DataFrame and ``result`` carries metadata
        about the fetch.

        Parameters
        ----------
        url : str
            One of:

            - Pre-rendered HTML report URL (``.../webBiblio/bib201_*.html``).
              Format is fixed; parsed as HTML.
            - CRIS direct link (``.../biblioweb/direct/<country>/<lang>/cris/<code>``).
              Accepts ``?format=X`` for structured XML output.
            - CONOR direct link (``.../biblioweb/direct/<country>/<lang>/conor/<id>``).
              Same as CRIS in terms of supported parameters.

        default_citation_source : {"wos", "scopus"}, default "wos"
            Determines which TC counts populate the canonical
            ``Cited by`` column. Both raw counts are kept in
            ``cobiss_wos_*`` / ``cobiss_scopus_*`` regardless.
        prefer_format : {"auto", "xml", "html"}, default "auto"
            For URLs that support the ``format=`` parameter (CRIS / CONOR
            direct links):

            - ``"auto"`` — request XML (more robust against UI changes),
              and silently fall back to HTML if XML parsing is not yet
              supported in this build of biblium.
            - ``"xml"`` — request XML and raise on parse failure.
            - ``"html"`` — request HTML.

            Pre-rendered HTML URLs always parse as HTML; this flag is
            ignored in that case.
        """
        from biblium.utilsbib_modules.cobiss_url import (
            classify_cobiss_url,
            is_xml_format,
            prepare_request_url,
        )

        t0 = time.time()

        # ---- URL routing ----
        info = classify_cobiss_url(url)
        request_url = url
        if info.supports_format_param:
            # Map our 'auto'|'xml'|'html' onto prepare_request_url's vocabulary
            # ('auto' here means "prefer XML when nothing was specified by user")
            if prefer_format == "html":
                request_url = prepare_request_url(
                    url, format="html", citation=True,
                )
            elif prefer_format == "xml":
                request_url = prepare_request_url(
                    url, format="xml", citation=True,
                )
            else:  # auto: prefer XML if user didn't already pick a format
                request_url = prepare_request_url(
                    url, format="auto", citation=True,
                )
        # else: pre-rendered HTML or unknown -> request_url == url

        # ---- HTTP fetch ----
        body, final_url = self.fetch(request_url)

        # ---- Decide which parser to use based on the *requested* URL ----
        use_xml = is_xml_format(request_url)
        records, meta_obj, used_format = self._dispatch_parsing(
            body, use_xml, default_citation_source, prefer_format,
        )

        df = records_to_dataframe(records)

        # ---- Outside-Slovenia detection (only meaningful when we actually
        #      got records back) ----
        if records:
            n_with_citations = sum(
                1 for r in records
                if r.cobiss_wos_tc is not None or r.cobiss_scopus_tc is not None
            )
            if n_with_citations == 0:
                warnings.warn(
                    "No WoS/Scopus citation counts were found in the "
                    "fetched response. This is normal when the COBISS+ "
                    "page was generated for or fetched from a non-.si IP "
                    "(citation counts are visible only from Slovenia). "
                    "All records will have Cited by = NaN.",
                    UserWarning, stacklevel=2,
                )

        # ---- Build result ----
        # Both parsers return objects with the same metadata attribute names
        result = CobissFetchResult(
            url=url,
            final_url=final_url,
            n_records=meta_obj.n_records,
            researcher_name=getattr(meta_obj, "researcher_name", None),
            researcher_code=getattr(meta_obj, "researcher_code", None),
            period=getattr(meta_obj, "period", None),
            typology_counts=dict(
                getattr(meta_obj, "n_records_per_typology", {}) or {}
            ),
            csv_path=None,
            elapsed_seconds=time.time() - t0,
        )
        # Tag the result with which parser actually ran (handy for debugging)
        result.format_used = used_format  # type: ignore[attr-defined]
        return df, result

    # ------------------------------------------------------------------
    # Parser dispatch (HTML vs XML, with fallback)
    # ------------------------------------------------------------------

    def _dispatch_parsing(
        self,
        body: str,
        prefer_xml: bool,
        default_citation_source: Literal["wos", "scopus"],
        prefer_format: Literal["auto", "xml", "html"],
    ):
        """
        Choose the right parser and return ``(records, metadata, format_used)``.

        - If ``prefer_xml`` is True, try the XML parser. On
          ``NotImplementedError`` (parser not yet wired up) or
          ``xml.etree.ElementTree.ParseError`` (server returned HTML
          despite the format=X request, which some endpoints do),
          fall back to HTML when ``prefer_format == "auto"``. Re-raise
          when the user explicitly asked for ``"xml"``.
        - Otherwise, use the HTML parser.
        """
        if prefer_xml:
            try:
                from biblium.utilsbib_modules.cobiss_xml_parser import (
                    parse_cobiss_xml,
                )
                records, meta = parse_cobiss_xml(
                    body, default_citation_source=default_citation_source,
                )
                return records, meta, "xml"
            except NotImplementedError:
                if prefer_format == "xml":
                    raise
                if self.verbose:
                    print(
                        "[cobiss] XML parsing not yet supported; "
                        "falling back to HTML parser."
                    )
            except Exception as exc:
                # ParseError (malformed XML) or any other XML-level failure
                # (e.g. namespace mismatch, encoding error). In auto mode
                # we treat this as "server returned HTML even though we
                # asked for XML"; in explicit mode we surface the error.
                if prefer_format == "xml":
                    raise
                exc_name = type(exc).__name__
                if self.verbose:
                    print(
                        f"[cobiss] XML parser failed ({exc_name}: {exc}); "
                        f"falling back to HTML parser."
                    )

        records, meta = parse_cobiss_html(
            body, default_citation_source=default_citation_source,
        )
        return records, meta, "html"


# =============================================================================
# HIGH-LEVEL CONVENIENCE
# =============================================================================


def fetch_personal_bibliography_to_csv(
    url: str,
    output_csv: str,
    *,
    default_citation_source: Literal["wos", "scopus"] = "wos",
    prefer_format: Literal["auto", "xml", "html"] = "auto",
    cache_dir: Optional[str] = None,
    verbose: bool = False,
    encoding: str = "utf-8",
    sep: str = ",",
) -> CobissFetchResult:
    """
    One-shot convenience: COBISS URL -> parsed -> written to CSV.

    Parameters
    ----------
    url : str
        Direct webBiblio URL or CRIS direct link.
    output_csv : str
        Path where the CSV will be written.
    default_citation_source : {"wos", "scopus"}, default "wos"
    prefer_format : {"auto", "xml", "html"}, default "auto"
        See :meth:`CobissClient.fetch_personal_bibliography` for details.
        ``"auto"`` requests XML for direct links (more robust against UI
        changes) and silently falls back to HTML if XML parsing is not
        available in this build.
    cache_dir : str, optional
        Local cache directory for downloaded responses (off by default).
    verbose : bool, default False
    encoding : str, default "utf-8"
    sep : str, default ","

    Returns
    -------
    CobissFetchResult
        Includes the path to the saved CSV in ``csv_path`` and the
        format used for parsing in ``format_used`` (``"html"`` or
        ``"xml"``).

    Examples
    --------
    >>> from biblium.cobiss_api import fetch_personal_bibliography_to_csv
    >>> result = fetch_personal_bibliography_to_csv(
    ...     "https://bib.cobiss.net/biblioweb/direct/si/eng/cris/28519",
    ...     "umek_bibliography.csv",
    ... )
    >>> print(f"Saved {result.n_records} records for {result.researcher_name}")
    """
    client = CobissClient(cache_dir=cache_dir, verbose=verbose)
    df, result = client.fetch_personal_bibliography(
        url,
        default_citation_source=default_citation_source,
        prefer_format=prefer_format,
    )
    # Ensure parent directory exists
    parent = os.path.dirname(os.path.abspath(output_csv))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    df.to_csv(output_csv, index=False, encoding=encoding, sep=sep)
    result.csv_path = output_csv

    if verbose:
        print(
            f"[cobiss] Wrote {len(df)} records to {output_csv} "
            f"(researcher: {result.researcher_name})"
        )

    return result


__all__ = [
    "CobissClient",
    "CobissFetchResult",
    "fetch_personal_bibliography_to_csv",
]
