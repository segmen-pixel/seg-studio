# Deployment Guide

## Important Security Notice

Seg-Studio is designed for **local or trusted network use**. By default the API has **no authentication**; for LAN / reverse-proxy deployments you can set the optional shared secret `SEG_API_TOKEN`, which requires an `X-API-Token` header on every `/api/v1/*`, `/v2/*`, and `/ws/v2/*` request (WebSockets included — browsers can pass `?api_token=<value>` instead of the header). Do not expose the API to the public internet without a reverse proxy providing TLS and real authentication.

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

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SEG_PROJECTS_DIR` | `./projects` | Project data directory |
| `SEG_DB_PATH` | `./projects/app.db` | SQLite database path |
| `SEG_MODELS_DIR` | `./models` | Directory for exported/registered models |
| `SEG_TORCH_DEVICE` | `auto` | PyTorch device (`cuda:0`, `mps`, `cpu`, or `auto`) |
| `SEG_API_TOKEN` | (empty) | Optional shared secret. When set, `/api/v1/*`, `/v2/*`, and `/ws/v2/*` require `X-API-Token` (or `?api_token=` for browser WebSockets) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `text` | Log format (`text` or `json`) |
| `SEG_HOST` | `127.0.0.1` | Bind address. Set to `0.0.0.0` for LAN access (use with reverse proxy) |
| `PYTORCH_CUDA_ALLOC_CONF` | - | Set to `expandable_segments:True` for better VRAM management (set automatically by the API) |

## Docker (Optional)

A `docker-compose.yml` ships with the repository (trainer API on `127.0.0.1:8002`, serving API on `127.0.0.1:8001`, UI on `127.0.0.1:5173` — ports are published on loopback only):

```bash
docker compose up -d
```

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
- **M1/M2/M3/M4**: MPS acceleration for inference. Training requires an NVIDIA CUDA GPU.
- **Intel Mac**: CPU-only — annotation and inference only.

**CPU-only**: Annotation and inference on all platforms. Set `SEG_TORCH_DEVICE=cpu`. Training is not supported on CPU — it requires an NVIDIA CUDA GPU.
