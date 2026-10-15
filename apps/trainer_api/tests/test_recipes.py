# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Tests for /projects/{id}/recipes endpoints."""
from __future__ import annotations

import json

VALID_RECIPE = {
    "version": 1,
    "name": "test-recipe",
    "rules": [
        {
            "class_id": 1,
            "steps": [
                {
                    "type": "hsv_range",
                    "params": {
                        "h_min": 0, "h_max": 10,
                        "s_min": 100, "s_max": 255,
                        "v_min": 100, "v_max": 255,
                    },
                },
            ],
        }
    ],
}


def _import_recipe(client, pid, recipe=None):
    data = json.dumps(recipe or VALID_RECIPE).encode()
    return client.post(
        f"/api/v1/projects/{pid}/recipes/import",
        files={"file": ("recipe.json", data, "application/json")},
    )


def test_import_recipe(client, project_id):
    resp = _import_recipe(client, project_id)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "recipe_id" in body


def test_list_recipes(client, project_id):
    _import_recipe(client, project_id)
    resp = client.get(f"/api/v1/projects/{project_id}/recipes")
    assert resp.status_code == 200
    recipes = resp.json()["recipes"]
    assert len(recipes) >= 1


def test_active_recipe(client, project_id):
    _import_recipe(client, project_id)
    resp = client.get(f"/api/v1/projects/{project_id}/recipes/active")
    assert resp.status_code == 200
    assert resp.json()["recipe"] is not None


def test_delete_recipe(client, project_id):
    imp = _import_recipe(client, project_id)
    rid = imp.json()["recipe_id"]
    resp = client.delete(f"/api/v1/projects/{project_id}/recipes/{rid}")
    assert resp.status_code == 200
    # Active should be cleared
    active = client.get(f"/api/v1/projects/{project_id}/recipes/active")
    assert active.json()["recipe"] is None


def test_apply_recipe(client, project_with_image):
    """Apply recipe to an image that has no mask yet → hasMask becomes True."""
    pid, item_id = project_with_image
    _import_recipe(client, pid)
    resp = client.post(
        f"/api/v1/projects/{pid}/recipes/apply",
        json={"item_ids": [item_id]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["applied"] >= 1
    # Verify hasMask
    idx = client.get(f"/api/v1/projects/{pid}/datasets/annotate").json()
    item = next(it for it in idx["items"] if it["id"] == item_id)
    assert item["annotation"]["hasMask"] is True


AGENT = {"X-Seg-Agent": "mcp/write", "X-Seg-Agent-Tool": "recipe_apply"}


def _annotation(client, pid, item_id):
    idx = client.get(f"/api/v1/projects/{pid}/datasets/annotate").json()
    return next(it for it in idx["items"] if it["id"] == item_id)["annotation"]


def test_an_agent_s_recipe_does_not_overwrite_a_person_s_mask(client, project_with_image):
    """The mask route's rule holds for a recipe too: named images are written
    over whatever they hold, and an agent may not do that to a person's work
    unless it says so."""
    import io

    import numpy as np
    from PIL import Image

    pid, item_id = project_with_image
    _import_recipe(client, pid)
    painted = np.zeros((16, 16), np.uint8)
    painted[2:6, 2:6] = 2
    buf = io.BytesIO()
    Image.fromarray(painted, mode="L").save(buf, format="PNG")
    r = client.put(f"/api/v1/projects/{pid}/datasets/annotate/masks/{item_id}.png",
                   files={"file": ("m.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200, r.text

    r = client.post(f"/api/v1/projects/{pid}/recipes/apply", json={"item_ids": [item_id]},
                    headers=AGENT)
    assert r.status_code == 409, r.text
    assert _annotation(client, pid, item_id)["classIds"] == [2], "the person's mask stands"

    r = client.post(f"/api/v1/projects/{pid}/recipes/apply?overwrite=1",
                    json={"item_ids": [item_id]}, headers=AGENT)
    assert r.status_code == 200 and r.json()["applied"] == 1, r.text
    assert _annotation(client, pid, item_id).get("by") == "mcp/write", "the agent's write says so"


def test_item_ids_must_be_a_list(client, project_with_image):
    """A bare string matched the ids by substring and skipped the check on a
    person's mask that a list goes through, so an agent wrote over it."""
    import io

    import numpy as np
    from PIL import Image

    pid, item_id = project_with_image
    _import_recipe(client, pid)
    painted = np.zeros((16, 16), np.uint8)
    painted[2:6, 2:6] = 2
    buf = io.BytesIO()
    Image.fromarray(painted, mode="L").save(buf, format="PNG")
    r = client.put(f"/api/v1/projects/{pid}/datasets/annotate/masks/{item_id}.png",
                   files={"file": ("m.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200, r.text

    r = client.post(f"/api/v1/projects/{pid}/recipes/apply", json={"item_ids": item_id},
                    headers=AGENT)
    assert r.status_code == 400, r.text
    assert _annotation(client, pid, item_id)["classIds"] == [2], "the person's mask stands"
    r = client.post(f"/api/v1/projects/{pid}/recipes/apply", json=[item_id], headers=AGENT)
    assert r.status_code == 400, r.text
