# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Every change written into runtime_settings.json leaves a file it cannot read alone.

The file holds the LAN token, the vlm block and whatever a person wrote there
by hand, and the app tells people to edit it. The device picker, the LAN
switch and the import default read it leniently -- a file with a typo read as
{} -- and wrote the merge back: the file became that one key. Only the
model-server choice read it strictly. Now every writer does, and refuses.

And the lenient read, which startup uses, takes a file in another encoding as
nothing set: a UnicodeDecodeError is not a JSONDecodeError, and it stopped
the server starting.

Each fails before this change.
"""
from __future__ import annotations

import json

import pytest

from app.core import state, torch_device
from app.core.import_settings import save_default_image_format

BROKEN = {
    "a typo": b'{"api_token": "s3cret", "vlm": {"backend": "ollama",}}',
    "not an object": b"[1, 2]",
    "a byte-order mark": b'\xef\xbb\xbf{"api_token": "s3cret"}',
    "a Windows code page": '{"api_token": "s3cret", "note": "検査"}'.encode("cp932"),
    "UTF-16": '{"api_token": "s3cret"}'.encode("utf-16"),
}


@pytest.fixture
def settings_file(monkeypatch, tmp_path):
    path = tmp_path / "runtime_settings.json"
    monkeypatch.setattr(torch_device, "RUNTIME_SETTINGS_PATH", path)
    return path


@pytest.mark.parametrize("why", sorted(BROKEN))
def test_the_lenient_read_takes_a_file_it_cannot_read_as_nothing_set(settings_file, why):
    settings_file.write_bytes(BROKEN[why])
    assert torch_device.read_runtime_settings() == {}
    assert torch_device.read_lan_access_setting() is False
    with pytest.raises(torch_device.RuntimeSettingsUnreadable):
        torch_device.read_runtime_settings_strict()


@pytest.mark.parametrize("why", sorted(BROKEN))
@pytest.mark.parametrize("write", [
    lambda: torch_device.merge_runtime_settings({"torch_device": "cpu"}),
    lambda: torch_device.save_lan_access_setting(True),
    lambda: save_default_image_format("jpg"),
], ids=["merge", "lan", "import-format"])
def test_a_write_into_a_file_it_cannot_read_is_refused_and_the_file_kept(settings_file, why, write):
    settings_file.write_bytes(BROKEN[why])
    with pytest.raises(Exception) as refused:
        write()
    assert getattr(refused.value, "status_code", None) == 409, refused.value
    assert "runtime_settings.json" in refused.value.detail and "Nothing was changed" in refused.value.detail
    assert settings_file.read_bytes() == BROKEN[why], "the token and the rest are still there"


def test_the_device_in_use_is_not_changed_by_a_refused_choice(settings_file, monkeypatch):
    settings_file.write_bytes(BROKEN["a typo"])
    monkeypatch.setattr(state, "SELECTED_TORCH_DEVICE", "auto")
    with pytest.raises(Exception) as refused:
        torch_device.set_configured_torch_device("cpu")
    assert getattr(refused.value, "status_code", None) == 409
    assert torch_device.current_configured_torch_device() == "auto", "the refusal says nothing was changed"


def test_over_http_the_refusal_is_a_409_saying_where(client, settings_file):
    settings_file.write_bytes(BROKEN["a typo"])
    for method, url, body in (("put", "/api/v1/system/network", {"lan_access": True}),
                              ("put", "/api/v1/system/import-defaults", {"format": "jpg"}),
                              ("put", "/api/v1/hardware/torch/device", {"device": "cpu"})):
        got = getattr(client, method)(url, json=body)
        assert got.status_code == 409, (url, got.text)
        assert "runtime_settings.json" in got.json()["detail"], got.text
    assert settings_file.read_bytes() == BROKEN["a typo"]


def test_a_file_it_can_read_keeps_its_other_keys(settings_file, monkeypatch):
    settings_file.write_text(json.dumps({"api_token": "s3cret", "vlm": {"model": "m"}}), encoding="utf-8")
    monkeypatch.setattr(state, "SELECTED_TORCH_DEVICE", "auto")
    torch_device.set_configured_torch_device("cpu")
    torch_device.save_lan_access_setting(True)
    save_default_image_format("jpg")
    got = json.loads(settings_file.read_text(encoding="utf-8"))
    assert got["api_token"] == "s3cret" and got["vlm"] == {"model": "m"}
    assert got["torch_device"] == "cpu" and got["lan_access"] is True
    assert torch_device.current_configured_torch_device() == "cpu"


@pytest.mark.parametrize("text", [None, "", "\n"])
def test_no_file_or_an_empty_one_is_written_as_before(settings_file, text):
    if text is not None:
        settings_file.write_text(text, encoding="utf-8")
    torch_device.save_lan_access_setting(True)
    assert json.loads(settings_file.read_text(encoding="utf-8")) == {"lan_access": True}
