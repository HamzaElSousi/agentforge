"""SSRF-guarded web tools: ``read_url`` and ``web_search``.

SSRF blocklist (``assert_safe_url``):
- Non-HTTP(S) schemes are rejected (file://, ftp://, etc.).
- The hostname is resolved via ``socket.getaddrinfo`` and every returned IP is
  checked against Python's ``ipaddress`` module:
    - ``is_loopback``      — 127.0.0.0/8, ::1
    - ``is_private``       — RFC-1918 (10/8, 172.16/12, 192.168/16), ULA (fc00::/7)
    - ``is_link_local``    — 169.254.0.0/16, fe80::/10
    - ``is_reserved``      — IANA reserved blocks
    - ``is_multicast``     — 224.0.0.0/4, ff00::/8
  The cloud metadata service ``169.254.169.254`` is also explicitly blocked
  by its canonical IP to guard against DNS rebinding tricks.
- After any redirect hop, the redirected host is re-checked so an attacker
  cannot chain public → private via a 302.

``read_url``:
    Fetches the URL with ``httpx``, extracts readable text via ``trafilatura``
    (fallback: stripped HTML), and truncates the result to 6 000 characters.

``web_search``:
    Runs a DuckDuckGo text search via ``duckduckgo_search.DDGS`` and formats
    the top results as a numbered list with title, URL, and snippet.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Optional
from urllib.parse import urlparse

from agentforge.tools.registry import ToolContext, tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_OUTPUT_CHARS = 6_000
# Cloud metadata endpoint — also caught by is_link_local, but explicit is safer.
_METADATA_IP = ipaddress.ip_address("169.254.169.254")


# ---------------------------------------------------------------------------
# Truncation helper
# ---------------------------------------------------------------------------


def truncate_text(s: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    """Truncate *s* to *limit* characters, appending a marker if trimmed.

    Parameters
    ----------
    s:
        The string to truncate.
    limit:
        Maximum number of characters to keep (default 6 000).

    Returns
    -------
    str
        Either *s* unchanged (if within limit) or the first *limit* chars
        followed by ``" [... N chars omitted ...]"``.
    """
    if len(s) <= limit:
        return s
    omitted = len(s) - limit
    return s[:limit] + f" [... {omitted} chars omitted ...]"


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------


def is_safe_url(url: str) -> bool:
    """Return ``True`` if *url* passes all SSRF safety checks.

    Checks applied (in order):
    1. The scheme must be ``http`` or ``https``.
    2. The hostname must be resolvable.
    3. Every IP the hostname resolves to must not be loopback, private,
       link-local, reserved, multicast, or the cloud metadata address.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    # Rule 1: scheme must be http or https.
    if parsed.scheme.lower() not in ("http", "https"):
        return False

    hostname = parsed.hostname
    if not hostname:
        return False

    # Rule 2 + 3: resolve and check every returned address.
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False

    for info in infos:
        raw_ip = info[4][0]
        try:
            addr = ipaddress.ip_address(raw_ip)
        except ValueError:
            return False  # unparseable — block it

        if (
            addr.is_loopback
            or addr.is_private
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
            or addr == _METADATA_IP
        ):
            return False

    return True


def assert_safe_url(url: str) -> None:
    """Raise ``ValueError`` with a safe, non-leaking message if the URL is blocked.

    Deliberately does NOT echo the resolved IP in the exception message to
    avoid information leakage to the LLM (which would forward it to output).

    Raises
    ------
    ValueError
        If the URL fails any SSRF safety check.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        raise ValueError(f"Malformed URL (could not parse): {url!r}.")

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(
            f"URL scheme {scheme!r} is not allowed. Only 'http' and 'https' are permitted."
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL has no hostname.")

    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve hostname {hostname!r}: {exc}.") from exc

    for info in infos:
        raw_ip = info[4][0]
        try:
            addr = ipaddress.ip_address(raw_ip)
        except ValueError:
            raise ValueError("URL resolves to an unrecognised IP format and is blocked.")

        if addr == _METADATA_IP:
            raise ValueError("URL resolves to the cloud metadata endpoint and is blocked.")
        if addr.is_loopback:
            raise ValueError("URL resolves to a loopback address and is blocked.")
        if addr.is_private:
            raise ValueError("URL resolves to a private/internal address and is blocked.")
        if addr.is_link_local:
            raise ValueError("URL resolves to a link-local address and is blocked.")
        if addr.is_reserved:
            raise ValueError("URL resolves to a reserved address and is blocked.")
        if addr.is_multicast:
            raise ValueError("URL resolves to a multicast address and is blocked.")


# ---------------------------------------------------------------------------
# ``read_url`` tool
# ---------------------------------------------------------------------------


@tool(risk="read_only", needs_network=True)
def read_url(ctx: ToolContext, url: str) -> str:
    """Fetch a web page and return its main readable text content.

    The URL is checked against the SSRF blocklist before any network request is
    made. Redirects are followed, but each intermediate URL is re-checked.
    Output is extracted with ``trafilatura`` (falls back to plain text) and
    truncated to 6 000 characters.

    Parameters
    ----------
    url:
        The HTTP(S) URL to fetch.

    Returns
    -------
    str
        Extracted text content, or an error description if the URL is blocked
        or the request fails.
    """
    if not ctx.network:
        return "[read_url] Network access is disabled for this agent (network=False)."

    # Guard: block SSRF before touching the wire.
    try:
        assert_safe_url(url)
    except ValueError as exc:
        return f"[read_url blocked] {exc}"

    try:
        import httpx
    except ImportError:
        return "[read_url error] httpx is not installed. Run: pip install httpx"

    # Custom transport that re-checks SSRF on every redirect hop.
    class _SSRFCheckTransport(httpx.HTTPTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            try:
                assert_safe_url(str(request.url))
            except ValueError as exc:
                raise httpx.HTTPStatusError(
                    f"SSRF block on redirect: {exc}",
                    request=request,
                    response=httpx.Response(403),
                )
            return super().handle_request(request)

    html_content = ""
    try:
        with httpx.Client(
            transport=_SSRFCheckTransport(),
            follow_redirects=True,
            timeout=15.0,
            headers={"User-Agent": "AgentForge/1.0 (SSRF-guarded fetch)"},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            html_content = response.text
    except httpx.HTTPStatusError as exc:
        return f"[read_url error] HTTP {exc.response.status_code} for {url}."
    except httpx.RequestError as exc:
        return f"[read_url error] Request failed: {exc}."
    except ValueError as exc:
        # Re-raised SSRF block during redirect.
        return f"[read_url blocked] {exc}"
    except Exception as exc:
        return f"[read_url error] Unexpected error: {exc}."

    # Extract main text with trafilatura; fall back to raw text.
    text = _extract_text(html_content, url)
    return truncate_text(text)


def _extract_text(html: str, url: str = "") -> str:
    """Extract readable text from *html*, falling back to stripped plain text."""
    try:
        import trafilatura  # type: ignore[import]
        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
        if extracted and extracted.strip():
            return extracted
    except Exception:
        pass

    # Fallback: strip all HTML tags manually.
    import re
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# ``web_search`` tool
# ---------------------------------------------------------------------------


@tool(risk="read_only", needs_network=True)
def web_search(ctx: ToolContext, query: str, max_results: int = 5) -> str:
    """Search the web via DuckDuckGo and return the top results.

    Each result is formatted as::

        1. <title>
           <url>
           <snippet>

    Parameters
    ----------
    query:
        The search query string.
    max_results:
        Maximum number of results to return (default 5).

    Returns
    -------
    str
        Formatted search results, or an error message if the search fails.
    """
    if not ctx.network:
        return "[web_search] Network access is disabled for this agent (network=False)."

    try:
        from duckduckgo_search import DDGS  # type: ignore[import]
    except ImportError:
        return (
            "[web_search error] duckduckgo_search is not installed. "
            "Run: pip install duckduckgo-search"
        )

    try:
        results = list(DDGS().text(query, max_results=max(1, max_results)))
    except Exception as exc:
        return f"[web_search error] DuckDuckGo search failed: {exc}."

    if not results:
        return f"[web_search] No results found for query: {query!r}."

    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "(no title)")
        href = r.get("href", r.get("url", "(no url)"))
        body = r.get("body", r.get("snippet", "(no snippet)"))
        lines.append(f"{i}. {title}\n   {href}\n   {body}")

    return truncate_text("\n\n".join(lines))
