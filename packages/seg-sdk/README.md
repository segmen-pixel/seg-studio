# Seg-Studio Inference SDK

Python client library for the Seg-Studio inference server.

---

## At a glance

| Item | Value |
|------|-------|
| **pip install name** | `seg-inference-sdk` |
| **import name** | `seg_sdk` |
| **Image input** | `bytes` (JPEG recommended) |
| **Result type** | `InferenceResult` |

```python
from seg_sdk import SegClient, InferenceResult
```

---

## Installation

```bash
# Basic (REST only)
pip install ./packages/seg-sdk              # macOS / Linux
pip install .\packages\seg-sdk              # Windows

# With WebSocket streaming support
pip install "./packages/seg-sdk[ws]"        # macOS / Linux
pip install ".\packages\seg-sdk[ws]"        # Windows

# With async client support
pip install "./packages/seg-sdk[async]"     # macOS / Linux
pip install ".\packages\seg-sdk[async]"     # Windows

# Everything
pip install "./packages/seg-sdk[all]"       # macOS / Linux
pip install ".\packages\seg-sdk[all]"       # Windows
```

---

## Quick start

```python
from seg_sdk import SegClient

client = SegClient("http://localhost:8002")
client.start_session(project_id="your-project-id", run_id="your-run-id")
result = client.predict(open("frame.jpg", "rb").read())
print(result.judgement, result.latency_ms)
```

---

## Single-image inference

```python
from seg_sdk import SegClient

client = SegClient("http://localhost:8002", timeout=30)

# Start a session (loads the model and warms it up)
client.start_session(
    project_id="your-project-id",
    run_id="your-run-id",
    backend="onnx",        # "onnx" | "coreml"
)

# Run inference
image_bytes = open("test.jpg", "rb").read()
result = client.predict(image_bytes)

print(f"judgement: {result.judgement}")       # "OK" or "NG"
print(f"defect_found: {result.defect_found}")
print(f"latency: {result.latency_ms}")

for region in result.regions:
    cx, cy = region.centroid
    print(
        f"  {region.class_name}: {region.area_px}px, "
        f"bbox={region.bbox}, centroid=({cx},{cy}), "
        f"conf={region.confidence:.3f}"
    )

# End the session
client.stop_session()
```

---

## Batch inference over a folder

```python
import csv
from pathlib import Path
from seg_sdk import SegClient

client = SegClient("http://localhost:8002")
client.start_session(project_id="your-project-id", run_id="your-run-id")

image_dir = Path("./images")
results = []

for img_path in sorted(image_dir.glob("*.jpg")):
    result = client.predict(img_path.read_bytes(), frame_id=img_path.name)
    results.append({
        "file": img_path.name,
        "judgement": result.judgement,
        "defect_found": result.defect_found,
        "num_regions": len(result.regions),
    })
    print(f"{img_path.name}: {result.judgement}")

# Write CSV (utf-8-sig so Excel opens it cleanly)
with open("results.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "judgement", "defect_found", "num_regions"])
    writer.writeheader()
    writer.writerows(results)

client.stop_session()
```

---

## WebSocket streaming

```python
import time
from pathlib import Path
from seg_sdk import SegClient

client = SegClient("http://localhost:8002")
client.start_session(project_id="your-project-id", run_id="your-run-id")

# Open a WebSocket stream (requires websocket-client)
stream = client.open_stream(source_id="cam-01")

for img_path in sorted(Path("./frames").glob("*.jpg")):
    frame_id = stream.send_frame(img_path.read_bytes(), frame_id=img_path.name)

    result = stream.recv_result(timeout=2.0)
    if result:
        print(f"[{result.frame_id}] {result.judgement} ({len(result.regions)} regions)")
    else:
        print(f"[{frame_id}] timeout")

    time.sleep(0.033)  # ~30 fps

stream.close()
client.stop_session()
```

---

## Async client

```python
import asyncio
from seg_sdk import AsyncSegClient

async def main():
    async with AsyncSegClient("http://localhost:8002") as client:
        await client.start_session(project_id="your-project-id", run_id="your-run-id")

        image_bytes = open("frame.jpg", "rb").read()
        result = await client.predict(image_bytes)
        print(f"{result.judgement} - regions: {len(result.regions)}")

        await client.stop_session()

asyncio.run(main())
```

---

## Data models

### InferenceResult

| Field | Type | Description |
|-------|------|-------------|
| `frame_id` | `str` | Frame identifier |
| `judgement` | `str` | `"OK"` or `"NG"` |
| `defect_found` | `bool` | Whether a defect was detected |
| `regions` | `list[Region]` | Detected defect regions, sorted by area (descending) |
| `summary` | `dict` | Per-image statistics (see below) |
| `latency_ms` | `dict` | Server-side latency breakdown in ms (see below) |
| `result_id` | `str` | Unique result identifier |

#### `summary` fields

Per-image aggregate statistics.

| Key | Type | Meaning | Example |
|-----|------|---------|---------|
| `fg_ratio` | `float` | Fraction of defect pixels over the whole image (0.0–1.0) | `0.0138` = 1.38% |
| `max_confidence` | `float` | Highest softmax confidence in the image (0.0–1.0) | `0.982` |
| `num_defects` | `int` | Number of connected defect regions (= `len(result.regions)`) | `3` |

#### `latency_ms` fields

Server-side processing-time breakdown in milliseconds. Useful for profiling bottlenecks.

| Key | Meaning |
|-----|---------|
| `decode` | JPEG/PNG decoding of the uploaded bytes |
| `inference` | Model inference (sliding-window) |
| `postprocess` | Connected-component analysis, region stats, overlay encoding |
| `total` | End-to-end time on the server for this frame |

### Region

Each entry is one connected defect region extracted by
`cv2.connectedComponentsWithStats`. All coordinates are expressed in the
**original input image's pixel space** — no client-side scaling needed.

| Field | Type | Description |
|-------|------|-------------|
| `class_name` | `str` | Defect class name |
| `class_id` | `int` | Defect class ID |
| `area_px` | `int` | Region area in pixels |
| `bbox` | `tuple[int,int,int,int]` | Bounding box `(x, y, w, h)` |
| `centroid` | `tuple[int,int]` | Centroid `(cx, cy)` — convenient for sending pick coordinates to a robot |
| `confidence` | `float` | Mean confidence inside the region |

---

## Error handling

```python
import requests
from seg_sdk import SegClient

client = SegClient("http://localhost:8002")

try:
    client.start_session(project_id="xxx", run_id="yyy")
    result = client.predict(open("test.jpg", "rb").read())
except requests.exceptions.ConnectionError:
    print("Cannot connect to server. Make sure the inference server is running.")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e.response.status_code} - {e.response.text}")
except FileNotFoundError:
    print("Image file not found.")
```

---

## FAQ

### Q: Why does `import seg_sdk` differ from the pip name `seg-inference-sdk`?
A: The pip distribution name is `seg-inference-sdk`; the Python import name is `seg_sdk`. The shorter import is reserved on PyPI under a different package.

### Q: Do I need to call `start_session` for every request?
A: No. Call it once; the session is reused for subsequent `predict` calls against the same model.

### Q: `recv_result` returned `None`
A: That's a timeout. Increase the `timeout` argument. `None` is also returned when the server is overloaded or a frame is dropped.

### Q: `ModuleNotFoundError: No module named 'websocket'`
A: WebSocket streaming is an optional extra. Install it with:
```bash
pip install "seg-inference-sdk[ws]"
```

### Q: `ModuleNotFoundError: No module named 'httpx'`
A: The async client is an optional extra. Install it with:
```bash
pip install "seg-inference-sdk[async]"
```

### Q: What image format should I use?
A: **JPEG is recommended.** PNG works as well, but the larger payload increases network transfer time.

### Q: The server has `SEG_API_TOKEN` set and requests fail (HTTP 401 / WebSocket closes with code 4401)
A: When `SEG_API_TOKEN` is set on the server, the `/v2/*` and `/ws/v2/*` endpoints require an `X-API-Token` header (or an `?api_token=` query parameter for WebSockets). The SDK does **not** yet send this token, so use it against servers where `SEG_API_TOKEN` is unset (the localhost default).

---

## Examples

| File | Description |
|------|-------------|
| [`examples/quick_start.py`](examples/quick_start.py) | Minimal single-image inference |
| [`examples/batch_inspect.py`](examples/batch_inspect.py) | Folder batch inference with CSV output |
| [`examples/ws_stream.py`](examples/ws_stream.py) | WebSocket streaming |
| [`examples/async_example.py`](examples/async_example.py) | Async client |

---

## Minimal working snippet

```python
from seg_sdk import SegClient

client = SegClient("http://localhost:8002")
client.start_session(project_id="your-project-id", run_id="your-run-id")
result = client.predict(open("image.jpg", "rb").read())
print(result.judgement)  # "OK" or "NG"
```

---

A Japanese version of this README is available at [README.ja.md](README.ja.md).
