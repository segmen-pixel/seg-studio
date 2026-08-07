# Deployment Guide

## Important Security Notice

Seg-Studio is designed for **local or trusted network use**, and it ships that
way: every start script under `scripts/` defaults to `127.0.0.1`, so a fresh
install is reachable only from the machine it runs on. LAN access is opt-in — Settings → **Allow access
from LAN**, or `SEG_HOST=0.0.0.0` — and takes effect at the next restart.

On that loopback default the trainer API needs **no token** (a Host
allowlist and a same-origin check still apply, so a page you merely visit
cannot drive it). For LAN /
reverse-proxy deployments the shared secret `SEG_API_TOKEN` requires an
`X-API-Token` header on every `/api/v1/*`, `/v2/*`, and `/ws/v2/*` request
(WebSockets included — browsers can pass `?api_token=<value>` instead of the
header); the Web UI signs in with it once and then uses a session cookie. The
two settings are tied together: the trainer API **refuses to start** when
`SEG_HOST` (or the persisted LAN toggle) puts it on a non-loopback address while
`SEG_API_TOKEN` is empty. That check reads `SEG_HOST`, not uvicorn's `--host`,
so if you pass `--host 0.0.0.0` by hand, set `SEG_HOST=0.0.0.0` as well.
Skipping it does not expose anything: a server with no token answers only
clients on the machine it runs on, so the port opens and every remote
request is refused. It does mean the failure arrives as a 401 per request
instead of a message at startup.

The token covers the trainer API only. The serving API on port 8001 has no
authentication at all — read [The serving API (port 8001) is
unauthenticated](#the-serving-api-port-8001-is-unauthenticated) before you open
that port. Do not expose either API to the public internet without a reverse
proxy providing TLS and real authentication.

### Browser sign-in

A browser cannot attach a custom header to an `<img src>`, a stylesheet, or a
download link, so a header-only credential would leave the bundled Web UI
unable to render a single overlay. Instead the UI signs in once:

1. Start the server with `SEG_HOST=0.0.0.0`, or enable LAN access in Settings
   and restart with the launcher for your platform
   (`scripts/windows/start_local_windows.bat`, `scripts/macos/start_api.sh`,
   `scripts/start_local.sh`). On the first LAN start the launcher generates
   `SEG_API_TOKEN`, prints it, and stores it as `api_token` in
   `projects/runtime_settings.json`; later starts reuse it. Set the environment
   variable yourself to use your own secret instead.
2. Open `http://<server>:8002/ui/` from another machine. The UI asks for the
   token before it loads.
3. On success the server issues a session cookie — `HttpOnly` (page scripts
   cannot read the secret), `SameSite=Strict` (another site cannot ride it),
   and carrying a hash of the token rather than the token itself. The browser
   then attaches it to every same-origin request, images and downloads
   included.

The cookie is derived deterministically from the token, so sessions survive a
server restart, and rotating `SEG_API_TOKEN` invalidates every existing one.
Because a cookie is an *ambient* credential, cookie-authenticated requests are
still subject to the same-origin (CSRF) check; an `X-API-Token` header — used
by scripts and `curl` — is not. (The bundled `seg-sdk` client does not send the
token yet, so run it on the machine hosting the server — whose requests are
token-exempt, see the next paragraph — or point it at a server with
`SEG_API_TOKEN` unset.) **No reverse proxy is required for this**: it
is only needed if you additionally want TLS or a second authentication layer.

Requests from the machine running the server are exempt from the token: binding
to the LAN should not make the operator's own browser sign in to its own desktop
app. The exemption keys off the TCP peer address, which — unlike the Host header
— a remote client cannot forge, and it drops only the token: a local request is
still held to the same-origin and Host-allowlist rules that protect the default
loopback install. A request carrying `X-Forwarded-For`, `X-Forwarded-Host`,
`X-Real-IP` or `Forwarded` is never treated as local, because a reverse proxy on
this host also connects from loopback while the request behind it did not.

The sign-in routes (`/api/v1/auth/status`, `/auth/session`, `/auth/logout`) are
the one part of `/api/v1/*` that is not token-guarded — they are how a client
with no credential obtains one. `/auth/status` reveals only whether a token is
required; `/auth/session` issues nothing unless the caller already knows it.

## The serving API (port 8001) is unauthenticated

`apps/serving_api` is a second, deliberately minimal FastAPI app. It loads the
ONNX model the trainer activated in `models/registry/` and answers
`POST /segment` and `POST /count`. **It does not read `SEG_API_TOKEN`, and it has
no token, cookie, Host or origin check of any kind.** Anything that can reach
port 8001 can run inference, list the registered model ids (`GET /models`), and
make the process re-read the registry pointer (`POST /reload`).

That is fine on the loopback default, and only there. The launchers start the
serving API with the same `SEG_HOST` as the trainer
(`scripts/windows/start_local_windows.bat`, `scripts/macos/start_local_macos.sh`,
`scripts/start_local.sh`, `scripts/windows/start_serving_api.bat`), so turning on
Settings → **Allow access from LAN** publishes port 8001 to the LAN too —
without the token gate the trainer gets. Before you do that, pick one:

- Leave the serving API on loopback and keep its callers on the same machine.
  This is the default and needs no configuration.
- Put it behind a reverse proxy that terminates TLS and authenticates (the nginx
  block below, with `proxy_pass http://127.0.0.1:8001;`), **and** block port 8001
  at the host firewall so the proxy is the only route in.

`docker compose` publishes it on `127.0.0.1:8001` only.

## Recommended Architecture

```
[Internet] → [Reverse Proxy (nginx)] → [Seg-Studio API :8002]
                  ↓
            TLS termination
            Basic/OAuth auth
            Rate limiting
```

## Reverse Proxy Setup (nginx)

```nginx
server {
    listen 443 ssl;
    server_name seg-studio.example.com;

    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;

    # Basic authentication
    auth_basic "Seg-Studio";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        proxy_pass http://127.0.0.1:8002;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        # WebSocket support (train logs /api/v1/ws/..., streaming /ws/v2/...)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        client_max_body_size 2G;  # for ZIP uploads
    }
}
```

The `Host` header this config forwards is the proxy's `server_name`, not
`localhost`, and that name is not in the DNS-rebinding allowlist by default. On
a server with no `SEG_API_TOKEN` every proxied request then comes back 403 `Host
header not allowed`, so add it:

```bash
SEG_ALLOWED_HOSTS=seg-studio.example.com
```

(Requests carrying a valid `X-API-Token` — or the session cookie the Web UI
gets after signing in — are allowed before the host check runs, so a
token-protected deployment does not need the entry.)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SEG_PROJECTS_DIR` | `./projects` | Project data directory |
| `SEG_DB_PATH` | `./projects/app.db` | SQLite database path |
| `SEG_MODELS_DIR` | `./models` | Directory for exported/registered models |
| `SEG_TORCH_DEVICE` | `auto` | PyTorch device (`cuda:0`, `mps`, `cpu`, or `auto`) |
| `SEG_API_TOKEN` | (empty) | Optional shared secret. When set, `/api/v1/*`, `/v2/*`, and `/ws/v2/*` require `X-API-Token` (or `?api_token=` for browser WebSockets). The Web UI signs in with it once and then uses a session cookie — see [Browser sign-in](#browser-sign-in) |
| `SEG_PREPARED_IMAGE_FORMAT` | `lossless` | Format for prepared training copies. The default writes PNG. Set `jpeg` to trade fidelity for ~3.6x smaller prepared sets and faster decode; see the handbook for the measured trade-off |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `text` | Log format (`text` or `json`) |
| `SEG_HOST` | `127.0.0.1` | Bind address. Set to `0.0.0.0` for LAN access — requires `SEG_API_TOKEN`, which the server enforces at startup |
| `SEG_ALLOWED_HOSTS` | (empty) | Comma-separated extra hostnames accepted in the `Host` header (DNS-rebinding allowlist; loopback names are always accepted). Needed when a reverse proxy forwards its own hostname to a server that has no `SEG_API_TOKEN` |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | CUDA allocator config for better VRAM management. Set by the API at import unless the variable is already present in the environment |

## Docker (Optional)

A `docker-compose.yml` ships with the repository (trainer API on `127.0.0.1:8002`, serving API on `127.0.0.1:8001`, UI on `127.0.0.1:5173` — ports are published on loopback only).

**Set `SEG_API_TOKEN` first.** Each container binds `0.0.0.0` inside its own
network namespace — that is how a published port reaches it — which the startup
check reads as a non-loopback bind, so the trainer container exits immediately
while the secret is empty. Put one in a `.env` file next to `docker-compose.yml`
(the repo root is mounted into the container and the app loads that file at
startup):

```bash
# Windows: use `python` instead of `python3`
python3 -c "import secrets; print('SEG_API_TOKEN=' + secrets.token_urlsafe(24))" >> .env
docker compose up -d
```

The compose stack requests no GPU (there is no
`deploy.resources.reservations.devices` entry), so the trainer container runs on
CPU: annotation, inference and the UI work, training does not. Training needs a
native install on a machine with an NVIDIA CUDA GPU — see [GPU
Requirements](#gpu-requirements).

Otherwise, run directly:

```bash
pip install -r apps/trainer_api/requirements.txt
# Localhost only (default):
python -m uvicorn apps.trainer_api.app.main:app --port 8002
# LAN access (use with reverse proxy + auth):
SEG_HOST=0.0.0.0 python -m uvicorn apps.trainer_api.app.main:app --host 0.0.0.0 --port 8002
```

## Data Backup

Project data is stored in `$SEG_PROJECTS_DIR`:
- `app.db` — Project metadata (SQLite)
- `{project_id}/` — Images, masks, training runs, models

Back up the entire `projects/` directory regularly.

## GPU Requirements

**NVIDIA (Windows / Linux):**
- **Minimum**: GTX 1650 (4GB VRAM) — training with batch_size=4-8
- **Recommended**: RTX 3060+ (8GB+ VRAM) — larger batch sizes, SAM2 models

**Apple Silicon (macOS):**
- **M1/M2/M3/M4**: MPS acceleration for both training and inference. `auto`
  selects MPS when no CUDA device is present. Mixed precision is disabled on
  MPS, so a run is slower than on a comparable NVIDIA card, and MPS shares
  unified memory with the rest of the system — see
  [Troubleshooting](troubleshooting.md) if a run runs out of memory.
- **Intel Mac**: CPU only. The installer detects the architecture and says so.
  There is no automated coverage for this configuration, so treat it as
  untested.

**CPU-only**: Annotation, inference, and training work on every platform. CPU
training is exercised by the test suite, but it is far slower than a GPU run.
Set `SEG_TORCH_DEVICE=cpu`.

**Object counting (instance segmentation) requires an NVIDIA GPU.** The VRAM
auto-fit only runs on CUDA devices, and the measured requirement for the
`small` model is 8 GiB at the default batch 8, auto-reducing to batch 4
(5.5 GiB) and batch 2 (3.5 GiB); below 3.5 GiB it is unsupported.
