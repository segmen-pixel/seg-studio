# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Shared startup progress state and the boot loading page.

Extracted from main.py during the pre-OSS refactor so middleware,
endpoints and startup tasks can share the state without importing
app.main. main re-exports the names for backward compatibility.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from starlette.responses import HTMLResponse, JSONResponse

if TYPE_CHECKING:
    from fastapi import FastAPI
    from starlette.requests import Request

startup_state: dict[str, Any] = {"ready": False, "steps": [], "current": "", "warnings": []}

LOADING_HTML = """\
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Seg-Studio --Starting...</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a2e;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  display:flex;align-items:center;justify-content:center;min-height:100vh}
.card{background:#16213e;border-radius:12px;padding:40px 48px;max-width:420px;width:90%;
  box-shadow:0 8px 32px rgba(0,0,0,.4);text-align:center}
h1{font-size:20px;font-weight:600;margin-bottom:4px;color:#fff}
.sub{font-size:12px;color:#8899aa;margin-bottom:24px}
.spinner{width:36px;height:36px;border:3px solid #2a3a5e;border-top-color:#4fc3f7;
  border-radius:50%;animation:spin .8s linear infinite;margin:0 auto 20px}
@keyframes spin{to{transform:rotate(360deg)}}
.current{font-size:13px;color:#4fc3f7;min-height:20px;margin-bottom:16px}
.steps{text-align:left;font-size:12px;color:#8899aa;line-height:1.8}
.steps .done{color:#66bb6a}
.steps .done::before{content:"\\2714 ";color:#66bb6a}
.error{color:#ef5350;font-size:13px;margin-top:12px}
</style>
</head>
<body>
<div class="card">
  <h1>Seg-Studio</h1>
  <div class="sub">See Every Pixel, From Pixel</div>
  <div class="spinner" id="sp"></div>
  <div class="current" id="cur">Connecting...</div>
  <div class="steps" id="steps"></div>
  <div class="error" id="err"></div>
</div>
<script>
(function(){
  var iv=setInterval(function(){
    fetch("/startup-status").then(function(r){return r.json()}).then(function(d){
      var el=document.getElementById("steps");
      el.innerHTML="";d.steps.forEach(function(s){var div=document.createElement("div");div.className="done";div.textContent=s;el.appendChild(div)});
      document.getElementById("cur").textContent=d.current||"";
      if(d.error) document.getElementById("err").textContent=d.error;
      if(d.ready){
        clearInterval(iv);
        document.getElementById("sp").style.borderTopColor="#66bb6a";
        document.getElementById("cur").textContent="Startup complete";
        setTimeout(function(){location.href="/ui/"},600);
      }
    }).catch(function(){
      document.getElementById("cur").textContent="Connecting to server...";
    });
  },800);
})();
</script>
</body>
</html>
"""


def discard_openapi_cache(app: FastAPI) -> None:
    """Forget the generated OpenAPI document so the next reader rebuilds it.

    FastAPI builds the schema once and keeps it on ``app.openapi_schema`` for
    the rest of the process. Most of this app's routers are registered from
    the background startup, so a document built before that finishes describes
    a half-built server -- and would be served as the truth from then on.
    """
    app.openapi_schema = None


def install_startup_gate(app: FastAPI, state: dict[str, Any] | None = None) -> None:
    """Hold back every request that would observe a half-registered app.

    `/startup-status` always answers, because the loading page polls it. API
    calls get a 503 the UI retries. The schema URL gets that same 503: served
    during startup it is generated from the routers registered so far and then
    cached (see `discard_openapi_cache`), and counting the paths in
    `/openapi.json` is how we tell which build a server is running. Everything
    else HTML-ish gets the loading page.
    """
    st = startup_state if state is None else state
    openapi_url = app.openapi_url  # None when the schema is turned off

    @app.middleware("http")
    async def startup_loading_guard(request: Request, call_next):
        if not st["ready"]:
            path = request.url.path
            # Always let the polling endpoint through
            if path == "/startup-status":
                return await call_next(request)
            # API requests and the schema: 503 so the caller retries instead of
            # being handed -- and remembering -- a partial answer.
            if path.startswith(("/api/", "/v2/", "/ws/")) or path == openapi_url:
                return JSONResponse(
                    {"detail": "Server is starting up", "status": "loading"},
                    status_code=503,
                )
            # Serve loading page for browser requests (HTML-accepting)
            accept = request.headers.get("accept", "")
            if "text/html" in accept or path.startswith("/ui"):
                return HTMLResponse(LOADING_HTML)
        return await call_next(request)
