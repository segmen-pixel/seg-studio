# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Camera capture manager with Tri-Loop pattern.

Architecture:
  Thread A (Capture)  — hardware read only, no processing
  Thread B (Inference) — waits for new frame, runs predict_frame(np.ndarray)
  Thread C (Preview)   — downsample + JPEG encode at ≤15fps for browser

Design decisions:
  - FrameSource ABC for future industrial camera SDK support (Basler/Spinnaker)
  - Condition variable + frame_id for latest-wins without race conditions
  - Preview always CPU (GPU reserved for inference)
  - workers=1 enforced (single camera handle)
"""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FrameSource interface
# ---------------------------------------------------------------------------
class FrameSource(ABC):
    """Abstract camera source. Subclass for OpenCV, Basler Pylon, etc."""

    @abstractmethod
    def open(self, device_id: int | str, width: int, height: int, fps: int) -> bool:
        """Open the camera device. Returns True on success."""
        ...

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read one frame. Returns (success, BGR ndarray)."""
        ...

    @abstractmethod
    def release(self) -> None:
        """Release the camera device."""
        ...

    @abstractmethod
    def is_opened(self) -> bool:
        ...


class OpenCVSource(FrameSource):
    """OpenCV VideoCapture implementation."""

    def __init__(self) -> None:
        self._cap = None

    def open(self, device_id: int | str, width: int, height: int, fps: int) -> bool:
        import cv2
        cap = cv2.VideoCapture(int(device_id) if str(device_id).isdigit() else device_id)
        if not cap.isOpened():
            return False
        if width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0:
            cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap = cap
        _logger.info("OpenCV camera opened: device=%s, resolution=%dx%d, fps=%d",
                      device_id, width, height, fps)
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()


# ---------------------------------------------------------------------------
# Camera config
# ---------------------------------------------------------------------------
@dataclass
class CameraConfig:
    device_id: int | str = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    preview_max_width: int = 640
    preview_fps: int = 15


# ---------------------------------------------------------------------------
# CameraManager — Tri-Loop pattern
# ---------------------------------------------------------------------------
class CameraManager:
    """Manages camera capture, preview streaming, and inference binding.

    States:
      IDLE     — camera not connected
      PREVIEW  — capture + preview threads running (no model needed)
      INSPECT  — capture + preview + inference threads running
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)

        # Frame buffer (protected by _lock)
        self._latest_frame: np.ndarray | None = None
        self._frame_id: int = 0

        # Threads
        self._capture_thread: threading.Thread | None = None
        self._preview_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Camera source
        self._source: FrameSource | None = None
        self._config = CameraConfig()

        # Preview consumers (WebSocket send callbacks)
        self._preview_consumers: list = []  # list of async callbacks
        self._preview_lock = threading.Lock()

        # Inference consumer callback
        self._inference_callback = None

        # Inference session info (set by attach_session)
        self._inference_info: dict | None = None

        # Latest preview JPEG (for polling)
        self._latest_preview: bytes | None = None

    @property
    def state(self) -> str:
        if self._source is None or not self._source.is_opened():
            return "IDLE"
        if self._inference_thread is not None and self._inference_thread.is_alive():
            return "INSPECT"
        return "PREVIEW"

    @property
    def frame_id(self) -> int:
        return self._frame_id

    def configure(self, config: CameraConfig) -> None:
        self._config = config

    def start(self, source: FrameSource | None = None) -> bool:
        """Start camera capture and preview threads."""
        if self._source is not None and self._source.is_opened():
            _logger.warning("Camera already running")
            return True

        self._stop_event.clear()
        src = source or OpenCVSource()
        ok = src.open(
            self._config.device_id,
            self._config.width,
            self._config.height,
            self._config.fps,
        )
        if not ok:
            _logger.error("Failed to open camera device %s", self._config.device_id)
            return False

        self._source = src

        # Thread A: Capture
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="cam-capture"
        )
        self._capture_thread.start()

        # Thread C: Preview
        self._preview_thread = threading.Thread(
            target=self._preview_loop, daemon=True, name="cam-preview"
        )
        self._preview_thread.start()

        _logger.info("Camera started (state=PREVIEW)")
        return True

    def stop(self) -> None:
        """Stop all camera threads and release hardware."""
        self._stop_event.set()

        # Wake up any waiting threads
        with self._new_frame:
            self._new_frame.notify_all()

        if self._capture_thread:
            self._capture_thread.join(timeout=3)
            self._capture_thread = None
        if self._preview_thread:
            self._preview_thread.join(timeout=3)
            self._preview_thread = None
        if self._inference_thread:
            self._inference_thread.join(timeout=3)
            self._inference_thread = None

        if self._source:
            self._source.release()
            self._source = None

        with self._lock:
            self._latest_frame = None
            self._frame_id = 0
        self._latest_preview = None
        self._inference_info = None
        self._inference_callback = None
        _logger.info("Camera stopped")

    def attach_inference(self, info: dict, callback) -> None:
        """Bind inference session to camera. Starts inference thread.

        Args:
            info: model info dict from _resolve_model_info
            callback: callable(result_dict) to deliver results
        """
        self._inference_info = info
        if callback is not None:
            self._inference_callback = callback

        if self._inference_thread is not None and self._inference_thread.is_alive():
            _logger.warning("Inference thread already running, skipping")
            return

        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="cam-inference"
        )
        self._inference_thread.start()
        _logger.info("Inference thread started (state=INSPECT)")

    def detach_inference(self) -> None:
        """Stop inference thread, keep camera running for preview."""
        self._inference_info = None
        self._inference_callback = None

        # Wake inference thread so it exits
        with self._new_frame:
            self._new_frame.notify_all()

        if self._inference_thread:
            self._inference_thread.join(timeout=3)
            self._inference_thread = None
        _logger.info("Inference detached (state=PREVIEW)")

    def add_preview_consumer(self, callback) -> None:
        with self._preview_lock:
            self._preview_consumers.append(callback)

    def remove_preview_consumer(self, callback) -> None:
        with self._preview_lock:
            self._preview_consumers = [c for c in self._preview_consumers if c is not callback]

    # -------------------------------------------------------------------
    # Thread A: Capture loop (high priority, minimal work)
    # -------------------------------------------------------------------
    def _capture_loop(self) -> None:
        _logger.debug("Capture loop started")
        while not self._stop_event.is_set():
            if self._source is None:
                break
            ok, frame = self._source.read()
            if not ok or frame is None:
                time.sleep(0.001)
                continue

            with self._new_frame:
                self._latest_frame = frame  # BGR numpy array
                self._frame_id += 1
                self._new_frame.notify_all()

        _logger.debug("Capture loop stopped")

    # -------------------------------------------------------------------
    # Thread C: Preview loop (low priority, ≤15fps)
    # -------------------------------------------------------------------
    def _preview_loop(self) -> None:
        import cv2

        _logger.debug("Preview loop started")
        interval = 1.0 / max(1, self._config.preview_fps)
        last_sent_id = 0

        while not self._stop_event.is_set():
            with self._new_frame:
                # Wait for new frame or timeout
                self._new_frame.wait(timeout=interval)
                if self._stop_event.is_set():
                    break
                if self._latest_frame is None or self._frame_id <= last_sent_id:
                    continue
                frame = self._latest_frame.copy()
                fid = self._frame_id
                last_sent_id = fid

            # Downscale for browser (CPU only — protect GPU)
            h, w = frame.shape[:2]
            max_w = self._config.preview_max_width
            if w > max_w:
                scale = max_w / w
                frame = cv2.resize(frame, (max_w, int(h * scale)), interpolation=cv2.INTER_NEAREST)

            # JPEG encode (CPU)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                continue

            jpeg_bytes = buf.tobytes()
            self._latest_preview = jpeg_bytes

            # Deliver to consumers
            with self._preview_lock:
                consumers = list(self._preview_consumers)
            for cb in consumers:
                try:
                    cb(jpeg_bytes, fid)
                except Exception:
                    _logger.warning("Preview consumer callback failed", exc_info=True)

            # Rate limit
            time.sleep(interval)

        _logger.debug("Preview loop stopped")

    # -------------------------------------------------------------------
    # Thread B: Inference loop (waits on condition, latest-wins)
    # -------------------------------------------------------------------
    def _inference_loop(self) -> None:
        _logger.debug("Inference loop started")
        last_processed_id = 0

        while not self._stop_event.is_set():
            # Check if still attached
            if self._inference_info is None:
                break

            with self._new_frame:
                self._new_frame.wait(timeout=0.1)
                if self._stop_event.is_set() or self._inference_info is None:
                    break
                if self._latest_frame is None or self._frame_id <= last_processed_id:
                    continue
                # Copy frame under lock
                frame = self._latest_frame.copy()
                fid = self._frame_id
                last_processed_id = fid

            # Run inference (outside lock)
            info = self._inference_info
            if info is None:
                break

            try:
                from .inference_runtime import get_inference_runtime
                runtime = get_inference_runtime()
                result = runtime.predict_frame(
                    frame_bgr=frame,
                    onnx_path=info["onnx_path"],
                    device_id=info["device_id"],
                    num_classes=info["num_classes"],
                    normalize=info["normalize"],
                    patch_size=info["patch_size"],
                    frame_id=f"cam-{fid}",
                    classes=info.get("classes"),
                )

                cb = self._inference_callback
                if cb is not None:
                    cb(result.to_dict(), fid)
                else:
                    _logger.warning("Camera inference result for frame %d dropped: no callback", fid)
            except Exception:
                _logger.exception("Camera inference error for frame %d", fid)

        _logger.debug("Inference loop stopped")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_camera: CameraManager | None = None
_camera_lock = threading.Lock()


def get_camera_manager() -> CameraManager:
    global _camera
    if _camera is not None:
        return _camera
    with _camera_lock:
        if _camera is None:
            _camera = CameraManager()
        return _camera
