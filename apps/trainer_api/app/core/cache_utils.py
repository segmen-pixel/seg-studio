# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Thread-safe LRU cache with optional TTL for concurrent FastAPI use."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Hashable
from typing import Any


class ThreadSafeLRUCache:
    """Thread-safe LRU cache with optional TTL and maxsize.

    Parameters:
        maxsize: Maximum number of entries. 0 = unlimited.
        ttl: Time-to-live in seconds. 0 = no expiry.
    """

    __slots__ = ("_maxsize", "_ttl", "_lock", "_data")

    def __init__(self, maxsize: int = 0, ttl: float = 0.0) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = threading.Lock()
        self._data: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()

    # -- read --

    def get(self, key: Hashable, default: Any = None) -> Any:
        with self._lock:
            if key not in self._data:
                return default
            ts, value = self._data[key]
            if self._ttl > 0 and (time.monotonic() - ts) >= self._ttl:
                del self._data[key]
                return default
            self._data.move_to_end(key)
            return value

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            if key not in self._data:
                return False
            if self._ttl > 0:
                ts, _ = self._data[key]
                if (time.monotonic() - ts) >= self._ttl:
                    del self._data[key]
                    return False
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    # -- write --

    def put(self, key: Hashable, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._data[key] = (time.monotonic(), value)
            else:
                self._data[key] = (time.monotonic(), value)
                if self._maxsize > 0 and len(self._data) > self._maxsize:
                    self._data.popitem(last=False)

    def pop(self, key: Hashable, default: Any = None) -> Any:
        with self._lock:
            entry = self._data.pop(key, None)
            return entry[1] if entry is not None else default

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    # -- iteration (snapshot) --

    def keys(self) -> list:
        with self._lock:
            return list(self._data.keys())
