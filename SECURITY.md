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
> enable LAN access, set `SEG_HOST=0.0.0.0` (or turn on Settings → **Allow
> access from LAN**) and restart with a launcher. The Trainer API refuses to
> start on a non-loopback bind while `SEG_API_TOKEN` is empty, and the
> launchers mint, persist and print a token on the first LAN start -- so a
> Trainer API published to a LAN *by a launcher* is always token-protected.
> (Starting uvicorn by hand with `--host 0.0.0.0` but no `SEG_HOST` binds the
> LAN without tripping that check. Always set both.) A reverse proxy is what
> adds TLS and real user accounts on top, and is required for anything beyond
> a trusted LAN.
>
> **The Serving API on port 8001 has none of that.** It does not read
> `SEG_API_TOKEN` and performs no authentication, yet the launchers bind it to
> the same `SEG_HOST` as the Trainer API. Keep it on loopback, or firewall port
> 8001 and front it with an authenticating proxy -- see
> [docs/deployment.md](docs/deployment.md#the-serving-api-port-8001-is-unauthenticated).

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
configuration for localhost-only use.

The shipped UI now wires this up: it asks for the token once and exchanges it
for an `HttpOnly`, `SameSite=Strict` session cookie via
`POST /api/v1/auth/session`, because a browser cannot put a custom header on an
`<img>` or a download link. Scripts and `curl` send the `X-API-Token` header
directly. Requests whose TCP peer is the server's own machine are exempt from
the token entirely. The bundled `seg-sdk` client does **not** send the header
yet, so run it on the server's own machine, or point it at a server with
`SEG_API_TOKEN` unset. The full rules are in
[docs/deployment.md](docs/deployment.md#browser-sign-in).

`SEG_API_TOKEN` guards `/api/v1/*`, `/v2/*` and `/ws/v2/*` on the **Trainer
API** only. The Serving API (port 8001) does not implement it.

## Labelling with a vision model (AI support)

The AI support panel (Assist) on the Annotate screen sends a project's data
to the model server that is configured: the pictures the model is shown, your
instruction, the tool results, image ids and the project's name. The
pictures are scaled or cropped copies of the project's images, and some
carry masks drawn over the image: a person's own masks (with single objects
cut out below them), SAM's proposals and the specks `spot_detect` found, and
the masks the run wrote. Treat that server as something that sees your data.

- **Where the server is, is configuration on the machine.** It comes from
  `SEG_VLM_BACKEND`, `SEG_VLM_BASE_URL`, `SEG_VLM_MODEL` and
  `SEG_VLM_API_KEY_ENV` in the environment, or from the `vlm` block of
  `projects/runtime_settings.json` (the environment wins). No API endpoint
  writes it, so neither a browser nor anyone holding the API token can point
  the trainer at another address. A `base_url` that is not `http://` or
  `https://` is ignored, and the backend's default address is used instead.
- **The defaults are local.** A run cannot start until at least one of
  `backend`, `base_url` or `model` is set. Every backend except `openai`
  defaults to a server on `127.0.0.1` (Ollama port 11434, LM Studio 1234,
  vLLM 8000, MLX and llama.cpp 8080); `openai` defaults to
  `https://api.openai.com/v1`.
- **A remote server should be reached over `https://`, on a host you
  trust.** Over `http://` the images, the prompts and the key cross the
  network in the clear.
- **The key is never stored in settings.** `api_key_env` holds the *name* of
  an environment variable. The trainer reads the key from that variable when
  it talks to the model server and sends it, as `Authorization: Bearer`, to
  that server only.
- **A run can write masks, and delete nothing.** The MCP bridge a run starts
  runs with `--policy write`, never `full` (see below). Every call is held to
  the project the run was started on and to that project's images, and a run
  cannot replace a mask a person drew or an image a person marked clean.
- The AI support panel's endpoints (`/api/v1/projects/{project_id}/agent/*` and
  `/api/v1/agent/runs`) are behind the same token rules as the rest of
  `/api/v1`. A project's conversation is kept in
  `projects/<project id>/agent_thread.json` and
  `projects/<project id>/agent_logs/`, without the pictures.

## The MCP bridge

`scripts/mcp_server.py` gives an MCP client -- a desktop assistant, an agent
of your own, or the AI support panel -- tools over the Trainer API. Every tool
is READ, WRITE or DESTRUCTIVE, and `--policy` decides which run:

| `--policy` | What runs |
|---|---|
| `read` (default) | READ tools only: look at projects, images, masks, runs and results, and change nothing |
| `write` | READ and WRITE tools: write masks, mark images clean or for review, upload images, create projects, change splits and classes, start and stop training, export models, generate reports, draft from a finished run. Some of these change what everyone using the trainer shares: `hardware_set_device` switches the compute device for every project and saves the choice; `classes_set` renames, recolours, deactivates or adds a project's classes; `assistant_context_set` replaces a project's assistant notes; `dataset_prepare_annotate` and `dataset_set_split` move images between splits; `train_start` and `assistant_command` (`/train start`) start training on the compute device; `train_stop` stops a run whoever started it |
| `full` | also the DESTRUCTIVE tools (`train_run_delete`, `clear_class`), and `overwrite=true` on a person's work |

- **Under `write`, no tool writes over a mask a person drew or an image a
  person marked clean.** That is all the rule protects: a project's classes,
  its splits, its notes, the compute device and training change under
  `write` as the table above says. `mask_put`, `write_kept`, `spot_write`,
  `mark_clean`, `recipe_apply` and `prelabel_run` leave a mask a person drew
  and an image a person marked clean as they are.
  `overwrite=true` over such an image needs `--policy full`, because what it
  replaces is not kept: the first five keep no copy, and `prelabel_run`
  keeps a copy of the mask file in `projects/<project id>/masks_replaced/`
  only until the next draft over that image, and none of a clean mark.
  `prelabel_run` with `overwrite=true` is checked against every image it
  would draft over -- the images named, or every image in the project when
  none are named -- and is refused as a whole under `write` if any of them
  carries a person's work. The AI support panel runs the bridge with `write`.
- **The trainer checks too.** The bridge names itself on every request
  (`X-Seg-Agent`). The trainer records the agent as the author (`by`) of what
  it writes through the mask, mark-clean, unmark-clean and recipe-apply
  routes, and answers 409 when an agent's write through one of them would
  replace a person's work, unless `overwrite=1` is passed. The draft route
  (`prelabel`) does neither: a draft written through `prelabel_run` is marked
  `draft: true` with the run that drafted it (`draftRun`), the route records
  no agent (`by`), and it does not ask who made the mask it replaces; for
  `prelabel_run` the bridge's own check is the guard. The header is the
  caller's own statement, so this guards against an agent's mistakes; it is
  not access control, and any client holding the API's credentials can
  write as a person.
- **Token.** The bridge sends `--token` (or `SEG_API_TOKEN`) as
  `X-API-Token`. A Trainer API with a token configured needs it from every
  client that is not on the server's own machine; a request whose TCP peer is
  loopback and that carries no forwarding header is exempt (see above), so a
  bridge on the same machine talking to `http://127.0.0.1:8002` needs none.
  On a shared machine, prefer `SEG_API_TOKEN` to `--token`, which the process
  list shows. The bridge warns at startup when it has no token for an API
  that is not local. The AI support panel hands its bridge `SEG_API_TOKEN`
  through the environment, never the command line, together with only a
  short list of ordinary environment variables.
- **Ids are checked before they reach a URL.** Every id that goes into a
  request path (`item_id`, `like_item_id`, `run_id`, `filename`) is refused
  if it contains `/`, `\` or `?` or is `.` or `..`, and every path segment is
  percent-encoded. A project may be named instead of given by id; a WRITE or
  DESTRUCTIVE tool takes an id, an exact name or a name ignoring case, never
  a fragment of one.
- **Text from the API is sanitised.** Strings in API responses that look
  like instructions to a model are replaced with a placeholder, and the tools
  that write text back (`classes_set`, `assistant_context_set`) refuse text
  that carries it. Every tool call is logged to stderr.

## The example chat page

`scripts/examples/qwen_mcp_chat.py` is an example, not part of the product
UI, and it has **no login**. Its bridge runs with `--policy write`, so
whoever can send it a request can write masks into the trainer's projects,
read the conversation and its pictures, and use the configured model server.
So:

- **It listens on this machine only.** `--host` must be `127.0.0.1`,
  `localhost` or `::1`. Reach it from another machine through an SSH tunnel
  (for example `ssh -L 8765:127.0.0.1:8765 <host>`), not over the network.
- **It refuses other origins.** A request whose `Host` header does not name
  the loopback address the page listens on (or `localhost`) is refused with
  403, so a web page whose name has been pointed at `127.0.0.1` gets nothing.
  A POST must be `application/json` (415 otherwise), and a POST whose
  `Origin` is not the page's own is refused with 403 before it reaches a
  route. A POST with no `Origin` is accepted: browsers send one, so it comes
  from a program on this machine.
- **The model server's address and key are set on the command line only.**
  `--base-url` and `--api-key-env` (the name of the variable holding the key)
  cannot be changed from the page, and a request that names either is refused
  with 400. The page can switch between the known servers, reached at their
  default addresses with no key, and pick a model. The key is sent only to
  the server that `--backend` names (without `--backend`, the server the page
  starts on: the one used last, else Ollama), at the `--base-url` address or
  that server's default address. The page remembers only the server's name
  and the model, and ignores an address saved by an earlier version.
- Conversations are kept in `~/.seg-studio/vlm-chat` (or the directory
  `SEG_VLM_CHAT_STATE` names).

## Supported Versions

| Version      | Supported |
|--------------|-----------|
| 0.9.x        | Yes       |
| < 0.9        | No        |

Only the latest released version receives security updates.

## Scope

Seg-Studio is designed as a **local desktop application** for Windows and Linux
workstations with NVIDIA GPUs and macOS machines with Apple Silicon (MPS). It is **not intended to be exposed to the public internet**.

- The Trainer API (port 8002) and the Serving API (port 8001) both listen on
  `localhost` by default. The example chat page (port 8765) listens on
  loopback only and refuses any other address.
- CORS is restricted to private network ranges (127.0.0.1, 192.168.x.x, 10.x.x.x, 172.16–31.x.x).
- Even with no token configured, the Trainer API's guarded paths (`/api/v1/*`,
  `/v2/*`, `/ws/v2/*`) reject cross-origin state changes and unexpected `Host`
  headers, so a page you merely visit cannot drive your local install.
- No user authentication is enforced unless `SEG_API_TOKEN` is configured; the
  application is otherwise assumed to run in a trusted local environment.
- The Serving API does not implement `SEG_API_TOKEN` at all. What reaches it is
  decided by the bind address and your firewall, and by nothing else.

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

One advisory is accepted for now: GHSA-4j2p-28q2-5m79 in accelerate, which has
no fixed release, is ignored by the CI `pip-audit` steps because the two
checkpoint loaders it affects are called neither by Seg-Studio nor by peft, the
only package that brings accelerate in; a local run still lists it, and the
ignore is removed once accelerate ships a fix.

## Security Measures

- **Path traversal protection:** All file operations use `_safe_child()` / `_safe_dir()` to prevent directory escape.
- **Upload size limits:** 200 MB per file for image/asset uploads, enforced on both client and server. The Serving API's `/count` and `/segment` refuse a body over 50 MB with 413. ZIP project import has its own limit (4 GB by default, configurable in Settings up to 64 GB).
- **Input sanitization:** Filenames are sanitized to prevent injection attacks.
- **Security headers:** `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY` (generated report HTML is served `SAMEORIGIN` for the in-app preview iframe), `Referrer-Policy: strict-origin-when-cross-origin`.
- **No secrets in error responses:** The global exception handler returns sanitized 500 errors without stack traces.
- **Structured logging:** Errors are logged server-side with timestamps; no sensitive data is sent to the client.
