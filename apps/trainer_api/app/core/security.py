# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path, PureWindowsPath
from urllib.parse import parse_qs

from fastapi import HTTPException, UploadFile

from .config import MAX_UPLOAD_BYTES


def _sanitize_filename(name: str) -> str:
    """Strip directory components and leading/trailing whitespace from a user-supplied filename."""
    # PureWindowsPath treats both `/` and `\` as separators on every platform;
    # Path on POSIX would let `..\..\` segments through untouched.
    return PureWindowsPath(name).name.strip()


def _safe_child(parent: Path, child_name: str) -> Path:
    """Resolve child_name under parent and ensure it stays within parent."""
    resolved = (parent / child_name).resolve()
    if not resolved.is_relative_to(parent.resolve()):
        raise HTTPException(status_code=400, detail="invalid path")
    return resolved


def _safe_dir(base: Path, user_path: str) -> Path:
    """Validate that user_path resolves inside base."""
    resolved = Path(user_path).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise HTTPException(status_code=400, detail="invalid directory path")
    return resolved


async def _read_upload(file: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    """Read an uploaded file with a streaming size cap to prevent DoS."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(65536)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"file too large (max {max_bytes // (1024*1024)} MB)",
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def _stream_upload_to_disk(file: UploadFile, dest: Path, max_bytes: int = MAX_UPLOAD_BYTES) -> int:
    """Stream uploaded file directly to disk. Returns bytes written."""
    import os
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
    total = 0
    try:
        while True:
            chunk = await file.read(262144)  # 256 KB chunks
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                os.close(tmp_fd)
                os.unlink(tmp_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"file too large (max {max_bytes // (1024*1024)} MB)",
                )
            os.write(tmp_fd, chunk)
        os.close(tmp_fd)
        Path(tmp_path).replace(dest)
    except HTTPException:
        raise
    except Exception:
        try:
            os.close(tmp_fd)
        except OSError:
            pass
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return total


# ---------------------------------------------------------------------------
# Archive-import safety: reject decompression bombs and path escapes before
# an archive is trusted. Defaults are generous for real dataset ZIPs (many
# thousands of images) yet bounded so one request cannot exhaust the host.
# ---------------------------------------------------------------------------
MAX_ZIP_UNCOMPRESSED_BYTES = 5 * 1024 * 1024 * 1024   # 5 GiB total expanded
MAX_ZIP_ENTRIES = 200_000                              # member count ceiling
MAX_ZIP_COMPRESSION_RATIO = 200                        # expanded / compressed


def _check_zip_bounds(
    zf,
    *,
    max_uncompressed: int = MAX_ZIP_UNCOMPRESSED_BYTES,
    max_entries: int = MAX_ZIP_ENTRIES,
    max_ratio: int = MAX_ZIP_COMPRESSION_RATIO,
) -> None:
    """Reject an archive whose central directory describes a bomb.

    Reads only the directory metadata (no member bytes), so it is cheap and
    runs before any extraction or per-member read. Callers that read members
    themselves (rather than extracting) should call this first.
    """
    infos = zf.infolist()
    if len(infos) > max_entries:
        raise HTTPException(
            status_code=400,
            detail=f"archive has too many entries ({len(infos)} > {max_entries})",
        )
    total_uncompressed = 0
    total_compressed = 0
    for info in infos:
        total_uncompressed += info.file_size
        total_compressed += info.compress_size
        if total_uncompressed > max_uncompressed:
            raise HTTPException(
                status_code=400,
                detail=(
                    "archive expands to more than "
                    f"{max_uncompressed // (1024*1024)} MB"
                ),
            )
        # Per-entry ratio guard catches a single hyper-compressed member even
        # when the total is modest.
        if info.compress_size > 0 and info.file_size / info.compress_size > max_ratio:
            raise HTTPException(
                status_code=400,
                detail=f"archive entry compression ratio exceeds {max_ratio}:1",
            )
    if total_compressed > 0 and total_uncompressed / total_compressed > max_ratio:
        raise HTTPException(
            status_code=400,
            detail=f"archive compression ratio exceeds {max_ratio}:1",
        )


def _safe_extract_zip(
    zf,
    dest_dir: Path,
    *,
    max_uncompressed: int = MAX_ZIP_UNCOMPRESSED_BYTES,
    max_entries: int = MAX_ZIP_ENTRIES,
    max_ratio: int = MAX_ZIP_COMPRESSION_RATIO,
) -> int:
    """Extract every member under dest_dir, enforcing size and path safety.

    Refuses path traversal, absolute paths and symlinks, and stops the moment
    the running total of extracted bytes would exceed the ceiling — so a bomb
    whose declared sizes lie is still bounded by what actually lands on disk.
    Returns the total bytes written.
    """
    import stat as _stat

    _check_zip_bounds(zf, max_uncompressed=max_uncompressed,
                       max_entries=max_entries, max_ratio=max_ratio)
    dest_resolved = dest_dir.resolve()
    written = 0
    for info in zf.infolist():
        # Symlinks in a zip are an escape primitive; refuse them outright.
        mode = info.external_attr >> 16
        if _stat.S_ISLNK(mode):
            raise HTTPException(status_code=400,
                                detail=f"archive contains a symlink: {info.filename}")
        target = (dest_dir / info.filename).resolve()
        if not target.is_relative_to(dest_resolved):
            raise HTTPException(status_code=400,
                                detail=f"unsafe archive entry: {info.filename}")
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info, "r") as src, open(target, "wb") as dst:
            while True:
                chunk = src.read(262144)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_uncompressed:
                    dst.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "archive expands to more than "
                            f"{max_uncompressed // (1024*1024)} MB"
                        ),
                    )
                dst.write(chunk)
    return written


# ---------------------------------------------------------------------------
# Public aliases — routers and other callers should use the un-underscored
# names. The underscored variants remain as the canonical definitions so
# in-module references and ``app.main.__getattr__`` lookups keep working.
# ---------------------------------------------------------------------------
sanitize_filename = _sanitize_filename
safe_child = _safe_child
safe_dir = _safe_dir
read_upload = _read_upload
stream_upload_to_disk = _stream_upload_to_disk
check_zip_bounds = _check_zip_bounds
safe_extract_zip = _safe_extract_zip
# evaluate_request_guard / is_allowed_host are already module-level.



# ---------------------------------------------------------------------------
# CSRF + DNS-rebinding guard. CORS restricts reading a response, not sending a
# request, so a tokenless loopback bind must still reject cross-origin state
# changes and rebound Host headers. Pure logic lives in evaluate_request_guard
# so the decision table is unit-testable without a server.
# ---------------------------------------------------------------------------
import hashlib as _hashlib
import hmac as _hmac
import os as _os
from urllib.parse import urlsplit as _urlsplit

_LOOPBACK_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "::1", "[::1]", ""})
_MUTATING_METHODS = frozenset({"POST", "PUT", "DELETE", "PATCH"})

#: Name of the browser session cookie issued by POST /api/v1/auth/session.
SESSION_COOKIE_NAME = "seg_session"
_SESSION_DERIVATION = b"seg-studio session cookie v1"


#: Headers a proxy adds when it relays someone else's request. Their presence
#: means the loopback peer is a front door, not the user's own browser.
_FORWARDING_HEADERS = ("x-forwarded-for", "x-forwarded-host", "x-real-ip", "forwarded")

_LOOPBACK_PEERS = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1"})


def secrets_equal(supplied: str, expected: str) -> bool:
    """Constant-time comparison of two credentials.

    hmac.compare_digest refuses str arguments containing non-ASCII and raises
    TypeError, which in middleware turns a wrong password into a 500 instead of
    a 401 -- and tells the caller their guess was unusual. Comparing the UTF-8
    bytes keeps the timing property and treats every input as simply wrong.
    """
    return _hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def is_same_origin(origin_header: str, host_header: str) -> bool:
    """True when a browser request did not come from another site.

    An absent Origin means a same-origin GET or a non-browser client; both are
    fine. A present one must be the exact origin being served -- host AND port.

    Deliberately stricter than the CSRF check in evaluate_request_guard, which
    also accepts anything named in SEG_ALLOWED_HOSTS and compares that entry by
    hostname only. That is the right latitude for "may this request change
    state", because the operator listed those names. It is the wrong latitude
    for handing out the shared secret: an allow-listed name matches on any
    port, so a page on a different port of the same machine would qualify, and
    whoever reads the token can then replay it as X-API-Token from anywhere on
    the network.
    """
    origin = (origin_header or "").strip()
    if not origin:
        return True
    origin_netloc = _urlsplit(origin).netloc.lower()
    return bool(origin_netloc) and origin_netloc == (host_header or "").strip().lower()


def is_local_peer(client_host: str | None, forwarding_headers: Iterable[str] = ()) -> bool:
    """True when the TCP peer is this machine and nothing relayed the request.

    Unlike the Host header, the peer address cannot be forged by the client, so
    this survives a bind to 0.0.0.0: a LAN caller claiming ``Host: localhost``
    still connects from its own address. Any forwarding header disqualifies the
    peer, because a reverse proxy on this host also connects from loopback and
    the request behind it is not local at all.
    """
    if not client_host:
        return False
    if any(h.lower() in _FORWARDING_HEADERS for h in forwarding_headers):
        return False
    return client_host.strip().strip("[]") in _LOOPBACK_PEERS


def session_cookie_value(configured_token: str) -> str:
    """The cookie value that proves knowledge of ``configured_token``.

    A hash rather than the token itself, so the secret never appears in a
    browser's cookie jar, a proxy log, or an HAR export. It is derived
    deterministically so sessions survive a server restart — there is no
    server-side session store to keep in sync.
    """
    if not configured_token:
        return ""
    return _hmac.new(_SESSION_DERIVATION, configured_token.encode("utf-8"), _hashlib.sha256).hexdigest()


def _hostname_only(netloc: str) -> str:
    """Lowercase hostname (no port) from a Host header or an Origin netloc."""
    s = (netloc or "").strip().lower()
    if "://" in s:
        s = _urlsplit(s).netloc
    if s.startswith("["):  # bracketed IPv6, e.g. [::1]:8002
        return s[: s.index("]") + 1] if "]" in s else s
    return s.rsplit(":", 1)[0] if ":" in s else s


def _extra_allowed_hosts() -> frozenset[str]:
    raw = _os.getenv("SEG_ALLOWED_HOSTS", "")
    return frozenset(h.strip().lower() for h in raw.split(",") if h.strip())


def is_allowed_host(host_header: str) -> bool:
    """True when the Host header is loopback or an explicitly allowed name.

    A DNS-rebinding request arrives with the attacker's domain in Host, which
    is neither loopback nor allow-listed, so it is rejected.
    """
    h = _hostname_only(host_header)
    return h in _LOOPBACK_HOSTNAMES or h in _extra_allowed_hosts()


def _origin_matches_host(origin_header: str, host_header: str) -> bool:
    """True when the request's Origin is the same origin as its Host (or allowed)."""
    origin_netloc = _urlsplit((origin_header or "").strip()).netloc.lower()
    if not origin_netloc:
        return False
    if origin_netloc == (host_header or "").strip().lower():
        return True
    return _hostname_only(origin_netloc) in _extra_allowed_hosts()


def evaluate_request_guard(
    *,
    method: str,
    host_header: str,
    origin_header: str,
    supplied_token: str,
    configured_token: str,
    supplied_cookie: str = "",
    local_peer: bool = False,
) -> tuple[str, str]:
    """Decide a guarded request: returns (verdict, reason).

    verdict is one of "allow", "unauthorized" (401), "forbidden" (403). The
    caller is responsible for only invoking this on guarded paths.
    """
    m = (method or "GET").upper()
    if m == "OPTIONS":
        return "allow", ""
    if configured_token:
        if supplied_token and secrets_equal(supplied_token, configured_token):
            return "allow", ""
        if supplied_cookie and secrets_equal(supplied_cookie, session_cookie_value(configured_token)):
            # A session cookie is an *ambient* credential: the browser attaches
            # it to any request aimed at this origin, including one triggered by
            # another site. So the CSRF check that guards the tokenless bind
            # applies here too. The DNS-rebinding host allowlist deliberately
            # does not: cookies are scoped to the hostname the user actually
            # browsed to, so a rebound attacker origin is never sent this cookie
            # to begin with, and requiring loopback here would break the LAN
            # deployment this cookie exists to serve.
            if m in _MUTATING_METHODS and origin_header and not _origin_matches_host(origin_header, host_header):
                return "forbidden", "Cross-origin request rejected (CSRF protection)."
            return "allow", ""
        if not local_peer:
            return "unauthorized", "Missing or invalid X-API-Token header."
        # A request from this machine itself. The token exists to authenticate
        # the *network*, and this peer is not on it: binding to 0.0.0.0 should
        # not make the operator's own browser log in to their own desktop app.
        # Falling through applies exactly the rules that protect the default
        # loopback install, so this is no weaker than a local-only deployment.
    # Tokenless (or a local peer on a token-protected server): only defensible
    # for a loopback, same-origin browser client.
    if not is_allowed_host(host_header):
        return "forbidden", "Host header not allowed (possible DNS rebinding)."
    if m in _MUTATING_METHODS and origin_header and not _origin_matches_host(origin_header, host_header):
        return "forbidden", "Cross-origin request rejected (CSRF protection)."
    return "allow", ""


def _session_cookie_from_header(cookie_header: str) -> str:
    """Extract the session cookie from a raw ``Cookie:`` header value.

    Starlette parses cookies for HTTP requests, but a WebSocket handshake is
    judged in raw-ASGI territory where only the byte headers are available.
    """
    for part in (cookie_header or "").split(";"):
        name, _, value = part.strip().partition("=")
        if name == SESSION_COOKIE_NAME:
            return value.strip()
    return ""


class WebSocketTokenGate:
    """Reject unauthenticated WebSocket handshakes when a shared token is set.

    Starlette's ``@app.middleware("http")`` only sees HTTP scopes, so the
    ``X-API-Token`` check for the REST surface never fires for WebSocket
    connects. This pure-ASGI middleware closes guarded WebSocket handshakes
    with code 4401 unless the client supplies the token via an
    ``X-API-Token`` header or an ``api_token`` query parameter (browsers
    cannot set custom headers on WebSocket connections).
    """

    def __init__(self, app, token: str = "", guard: Callable[[str], bool] | None = None):
        self.app = app
        self.token = token
        self.guard = guard or (lambda _path: False)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket" and self.guard(scope.get("path", "")):
            headers = {k: v for k, v in (scope.get("headers") or [])}
            supplied = headers.get(b"x-api-token", b"").decode("latin-1")
            if not supplied:
                query = parse_qs((scope.get("query_string") or b"").decode("latin-1"))
                supplied = (query.get("api_token") or [""])[0]
            origin = headers.get(b"origin", b"").decode("latin-1")
            host = headers.get(b"host", b"").decode("latin-1")
            cookie = _session_cookie_from_header(headers.get(b"cookie", b"").decode("latin-1"))
            peer = (scope.get("client") or ("", 0))[0]
            local = is_local_peer(peer, [k.decode("latin-1") for k in headers])
            # A WS handshake changes/streams state, so judge it like a mutating
            # request: browsers always send Origin on WS, so the same-origin
            # check is reliable for the tokenless case.
            verdict, _reason = evaluate_request_guard(
                method="POST", host_header=host, origin_header=origin,
                supplied_token=supplied, configured_token=self.token,
                supplied_cookie=cookie, local_peer=local,
            )
            if verdict != "allow":
                await receive()  # consume the websocket.connect event
                await send({"type": "websocket.close", "code": 4401})
                return
        await self.app(scope, receive, send)
