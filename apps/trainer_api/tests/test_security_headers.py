# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Security-header middleware.

Report HTML is embedded in an in-app iframe (same-origin preview tab), so it
must be served with ``X-Frame-Options: SAMEORIGIN`` instead of the global
``DENY``. Everything else must stay ``DENY``. These guard the regression that
broke the report preview iframe.
"""
from __future__ import annotations


def test_report_html_allows_same_origin_framing(client):
    # A non-existent report still exercises the path-based middleware: security
    # headers are applied to the 404 response too.
    resp = client.get("/api/v1/projects/_none/reports/_none/report.html")
    assert resp.headers.get("x-frame-options") == "SAMEORIGIN"


def test_non_html_report_asset_stays_denied(client):
    # Only HTML report files are framable; other assets keep the strict default.
    resp = client.get("/api/v1/projects/_none/reports/_none/chart.png")
    assert resp.headers.get("x-frame-options") == "DENY"


def test_regular_api_response_denies_framing(client):
    resp = client.get("/version")
    assert resp.status_code == 200
    assert resp.headers.get("x-frame-options") == "DENY"
