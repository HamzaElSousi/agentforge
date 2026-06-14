"""Tests for agentforge/tools/web.py — SSRF guard and truncation helper.

All tests are hermetic: no real network calls are made. The is_safe_url /
assert_safe_url functions call socket.getaddrinfo internally, so for private/
loopback addresses the function short-circuits before resolving (or returns the
literal IP immediately). For the public-URL test we monkeypatch
socket.getaddrinfo to return a known public IP so the test works offline.
"""

from __future__ import annotations

import socket

import pytest

from agentforge.tools.web import assert_safe_url, is_safe_url, truncate_text


# ---------------------------------------------------------------------------
# truncate_text
# ---------------------------------------------------------------------------


class TestTruncateText:
    def test_short_string_returned_unchanged(self):
        s = "hello world"
        assert truncate_text(s, limit=100) == s

    def test_exact_limit_not_truncated(self):
        s = "x" * 100
        assert truncate_text(s, limit=100) == s

    def test_long_string_is_shortened_to_limit(self):
        s = "A" * 10_000
        result = truncate_text(s, limit=200)
        # Result should be limit chars + the omission marker suffix
        assert len(result) > 200  # marker adds chars
        assert result[:200] == "A" * 200

    def test_long_string_contains_omitted_marker(self):
        s = "Z" * 500
        result = truncate_text(s, limit=100)
        assert "omitted" in result, "Truncated result must contain 'omitted' marker"

    def test_omitted_marker_states_correct_count(self):
        s = "x" * 150
        result = truncate_text(s, limit=100)
        assert "50" in result, "Marker should report 50 chars omitted (150 - 100)"

    def test_empty_string_returned_unchanged(self):
        assert truncate_text("", limit=10) == ""

    def test_default_limit_is_6000(self):
        # Just under 6000 — should not be truncated
        s = "y" * 5999
        assert truncate_text(s) == s

        # Over 6000 — should be truncated
        s2 = "y" * 6001
        assert "omitted" in truncate_text(s2)


# ---------------------------------------------------------------------------
# SSRF: blocked addresses — is_safe_url returns False
# ---------------------------------------------------------------------------


class TestIsSafeUrlBlocked:
    def test_loopback_127_0_0_1_blocked(self):
        assert is_safe_url("http://127.0.0.1") is False

    def test_loopback_localhost_blocked(self):
        # localhost should resolve to 127.0.0.1 (loopback) — blocked
        assert is_safe_url("http://localhost") is False

    def test_cloud_metadata_169_254_169_254_blocked(self):
        assert is_safe_url("http://169.254.169.254") is False

    def test_private_rfc1918_10_blocked(self):
        assert is_safe_url("http://10.0.0.5") is False

    def test_private_rfc1918_192_168_blocked(self):
        assert is_safe_url("http://192.168.1.1") is False

    def test_file_scheme_blocked(self):
        assert is_safe_url("file:///etc/passwd") is False

    def test_ftp_scheme_blocked(self):
        assert is_safe_url("ftp://example.com") is False

    def test_no_scheme_blocked(self):
        assert is_safe_url("example.com") is False

    def test_empty_string_blocked(self):
        assert is_safe_url("") is False

    def test_loopback_ipv6_blocked(self):
        assert is_safe_url("http://[::1]") is False


# ---------------------------------------------------------------------------
# SSRF: blocked addresses — assert_safe_url raises ValueError
# ---------------------------------------------------------------------------


class TestAssertSafeUrlBlocked:
    def test_loopback_raises(self):
        with pytest.raises(ValueError, match="(?i)loopback|blocked"):
            assert_safe_url("http://127.0.0.1")

    def test_localhost_raises(self):
        with pytest.raises(ValueError):
            assert_safe_url("http://localhost")

    def test_cloud_metadata_raises(self):
        with pytest.raises(ValueError, match="(?i)metadata|blocked"):
            assert_safe_url("http://169.254.169.254")

    def test_private_10_raises(self):
        with pytest.raises(ValueError, match="(?i)private|blocked"):
            assert_safe_url("http://10.0.0.5")

    def test_private_192_168_raises(self):
        with pytest.raises(ValueError, match="(?i)private|blocked"):
            assert_safe_url("http://192.168.1.1")

    def test_file_scheme_raises(self):
        with pytest.raises(ValueError, match="(?i)scheme|not allowed"):
            assert_safe_url("file:///etc/passwd")

    def test_ftp_scheme_raises(self):
        with pytest.raises(ValueError, match="(?i)scheme|not allowed"):
            assert_safe_url("ftp://example.com")

    def test_link_local_raises(self):
        # 169.254.x.x is link-local; use another address in that block
        with pytest.raises(ValueError):
            assert_safe_url("http://169.254.0.1")


# ---------------------------------------------------------------------------
# SSRF: public URL is allowed (monkeypatched to avoid real DNS)
# ---------------------------------------------------------------------------


def _fake_getaddrinfo_public(host, port, *args, **kwargs):
    """Return a fake public IP (93.184.216.34 — example.com) for any hostname."""
    # Format: (family, type, proto, canonname, sockaddr)
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]


def test_public_https_url_is_safe(monkeypatch):
    """A URL that resolves to a public IP passes the SSRF check.

    We monkeypatch socket.getaddrinfo so this test works offline and is
    deterministic regardless of DNS availability.
    """
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_public)
    assert is_safe_url("https://example.com") is True


def test_assert_safe_url_public_does_not_raise(monkeypatch):
    """assert_safe_url must not raise for a public URL."""
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_public)
    # Should complete without raising
    assert_safe_url("https://example.com")


def test_public_http_url_is_safe(monkeypatch):
    """Plain http:// to a public IP also passes."""
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_public)
    assert is_safe_url("http://example.com") is True


# ---------------------------------------------------------------------------
# SSRF: unresolvable hostname
# ---------------------------------------------------------------------------


def _fake_getaddrinfo_fail(host, port, *args, **kwargs):
    raise socket.gaierror("Name or service not known")


def test_unresolvable_hostname_blocked(monkeypatch):
    """A hostname that cannot be resolved must be blocked."""
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_fail)
    assert is_safe_url("http://nonexistent.invalid.tld") is False


def test_assert_safe_url_unresolvable_raises(monkeypatch):
    """assert_safe_url must raise for an unresolvable hostname."""
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo_fail)
    with pytest.raises(ValueError, match="(?i)resolve|hostname"):
        assert_safe_url("http://nonexistent.invalid.tld")
