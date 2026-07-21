# Security Policy

> **WARNING -- DO NOT EXPOSE TO THE PUBLIC INTERNET**
>
> Seg-Studio ships with **no built-in user accounts or RBAC**. It is
> designed for local workstation use or trusted private networks.
> Exposing the API or UI to the public internet will allow **anyone**
> (unless a reverse proxy is in front) to:
>
> - Read, modify, and delete all project data (images, masks, models)
> - Execute training jobs that consume GPU/CPU resources
> - Upload arbitrary files to the server
>
> **If you need network access** (e.g., sharing with teammates on a LAN),
> you MUST place the application behind a reverse proxy (nginx, Caddy, etc.)
> with TLS and authentication enabled. See [docs/deployment.md](docs/deployment.md)
> for a complete reverse-proxy setup guide.
>
> By default, all start scripts bind to `127.0.0.1` (localhost only). To
> enable LAN access, set the environment variable `SEG_HOST=0.0.0.0` --
> but only after configuring a reverse proxy with authentication.

## Optional shared-secret API token

For deployments that need a minimal defense-in-depth layer (e.g., a
reverse proxy in front of multiple users on a LAN), set:

```
SEG_API_TOKEN=<a long random secret>
```

in your `.env`. When present, every request to `/api/v1/*`, `/v2/*`,
and `/ws/v2/*` must carry the header `X-API-Token: <value>` or it is
rejected with `401`; WebSocket handshakes without a valid token are
closed with code `4401`. Browsers cannot set custom WebSocket headers,
so WebSocket clients may pass the token as an `api_token` query
parameter instead. Leave
the variable empty (default) to disable the check — this is the expected
configuration for localhost-only use. The shipped UI does not yet wire
up the token header automatically (tracked for v1.1); until then,
`SEG_API_TOKEN` is primarily useful for scripted / proxy-fronted
deployments where the proxy injects the header.

## Supported Versions

| Version      | Supported |
|--------------|-----------|
| 0.9.x-beta   | Yes       |
| < 0.9        | No        |

Only the latest released version receives security updates.

## Scope

Seg-Studio is designed as a **local desktop application** for Windows workstations
with NVIDIA GPUs and macOS machines with Apple Silicon (MPS). It is **not intended to be exposed to the public internet**.

- The Trainer API listens on `localhost` by default.
- CORS is restricted to private network ranges (127.0.0.1, 192.168.x.x, 10.x.x.x, 172.16–31.x.x).
- No authentication is enforced unless `SEG_API_TOKEN` is configured; the
  application is otherwise assumed to run in a trusted local environment.

If you deploy this application on a network-accessible server, you are responsible for
adding appropriate authentication, TLS, and network-level access controls.

## Reporting a Vulnerability

If you discover a security vulnerability, please report it privately:

1. **GitHub Security Advisories:** Use the [Report a vulnerability](https://github.com/segmen-pixel/seg-studio/security/advisories/new) feature on GitHub.
2. Include steps to reproduce, affected versions, and potential impact.

We will acknowledge receipt within 3 business days and aim to provide a fix or mitigation
within 14 business days for critical issues.

**Please do not open public GitHub issues for security vulnerabilities.**

## Dependency Auditing

Run the included audit script to check for known vulnerabilities in dependencies:

```bash
# Windows
scripts\audit.bat
# macOS / Linux
bash scripts/audit.sh
```

This runs `npm audit` for the UI and `pip-audit` for the API.

## Security Measures

- **Path traversal protection:** All file operations use `_safe_child()` / `_safe_dir()` to prevent directory escape.
- **Upload size limits:** 200 MB per file, enforced on both client and server.
- **Input sanitization:** Filenames are sanitized to prevent injection attacks.
- **Security headers:** `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY` (generated report HTML is served `SAMEORIGIN` for the in-app preview iframe), `Referrer-Policy: strict-origin-when-cross-origin`.
- **No secrets in error responses:** The global exception handler returns sanitized 500 errors without stack traces.
- **Structured logging:** Errors are logged server-side with timestamps; no sensitive data is sent to the client.
