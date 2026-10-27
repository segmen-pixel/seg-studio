# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The camera opens a device by index, never a string a request chose.

cv2.VideoCapture takes a string as well and opens it as a URL (http://,
rtsp://) or a file path through FFmpeg; the frames come back as JPEGs on
/ws/v2/camera. Taken from a request, that is the server fetching internal
addresses, or reading video and image files off its disk, for whoever asked.
"""
from __future__ import annotations

import pytest

from app.core.camera_manager import CameraConfig, OpenCVSource, get_camera_manager


@pytest.fixture
def camera_config_restored():
    mgr = get_camera_manager()
    before = mgr._config
    yield mgr
    mgr.configure(before)


@pytest.mark.parametrize("source", [
    "rtsp://10.0.0.5/stream",
    "http://169.254.169.254/latest/meta-data/",
    "/etc/hosts",
    "C:\\Users\\someone\\Videos\\clip.mp4",
])
def test_the_route_refuses_anything_but_an_index(client, camera_config_restored, source):
    res = client.post("/v2/camera/config", json={"device_id": source})
    assert res.status_code == 422, res.text
    assert camera_config_restored._config.device_id != source


@pytest.mark.parametrize("given,stored", [(0, 0), (2, 2), ("1", 1)])
def test_the_route_takes_an_index(client, camera_config_restored, given, stored):
    res = client.post("/v2/camera/config", json={"device_id": given})
    assert res.status_code == 200, res.text
    assert camera_config_restored._config.device_id == stored
    assert isinstance(camera_config_restored._config.device_id, int)


class _Capture:
    calls: list = []

    def __init__(self, source):
        _Capture.calls.append(source)

    def isOpened(self):
        return False


@pytest.fixture
def fake_capture(monkeypatch):
    import cv2

    _Capture.calls = []
    monkeypatch.setattr(cv2, "VideoCapture", _Capture)
    return _Capture


@pytest.mark.parametrize("source", ["rtsp://10.0.0.5/stream", "/etc/hosts", "file:///etc/hosts"])
def test_opencv_is_never_handed_a_string(fake_capture, source):
    # The route refuses these already; this is the same rule one step later,
    # for anything that builds a CameraConfig some other way.
    assert OpenCVSource().open(source, 640, 480, 30) is False
    assert fake_capture.calls == []


def test_opencv_gets_the_index_as_a_number(fake_capture):
    OpenCVSource().open(1, 640, 480, 30)
    OpenCVSource().open("2", 640, 480, 30)
    assert fake_capture.calls == [1, 2]
    assert all(type(c) is int for c in fake_capture.calls)


def test_the_default_config_is_device_zero():
    assert CameraConfig().device_id == 0
