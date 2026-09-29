# Seg-Studio Inference SDK

Seg-Studio 推論サーバーと通信するための Python クライアントライブラリです。

---

## まず最初に 3 つだけ覚えてください

| 項目 | 値 |
|------|----|
| **pip install 名** | `seg-inference-sdk` |
| **import 名** | `seg_sdk` |
| **画像の渡し方** | `bytes` (JPEG 推奨) |
| **結果の型** | `InferenceResult` |

```python
from seg_sdk import SegClient, InferenceResult
```

---

## インストール

```bash
# 基本 (REST のみ)
pip install ./packages/seg-sdk              # macOS / Linux
pip install .\packages\seg-sdk              # Windows

# WebSocket ストリーミングも使う場合
pip install "./packages/seg-sdk[ws]"        # macOS / Linux
pip install ".\packages\seg-sdk[ws]"        # Windows

# 非同期クライアントも使う場合
pip install "./packages/seg-sdk[async]"     # macOS / Linux
pip install ".\packages\seg-sdk[async]"     # Windows

# 全部入り
pip install "./packages/seg-sdk[all]"       # macOS / Linux
pip install ".\packages\seg-sdk[all]"       # Windows
```

---

## クイックスタート (5 行)

```python
from seg_sdk import SegClient

client = SegClient("http://localhost:8002")
client.start_session(project_id="your-project-id", run_id="your-run-id")
result = client.predict(open("frame.jpg", "rb").read())
print(result.judgement, result.latency_ms)
```

---

## 単発推論の例

```python
from seg_sdk import SegClient

client = SegClient("http://localhost:8002", timeout=30)

# セッション開始 (モデルのロード + ウォームアップ)
client.start_session(
    project_id="your-project-id",
    run_id="your-run-id",
    backend="onnx",        # "onnx" | "coreml"
)

# 推論
image_bytes = open("test.jpg", "rb").read()
result = client.predict(image_bytes)

print(f"判定: {result.judgement}")       # "OK" or "NG"
print(f"欠陥あり: {result.defect_found}")
print(f"レイテンシ: {result.latency_ms}")

for region in result.regions:
    cx, cy = region.centroid
    print(
        f"  {region.class_name}: {region.area_px}px, "
        f"bbox={region.bbox}, centroid=({cx},{cy}), "
        f"conf={region.confidence:.3f}"
    )

# セッション終了
client.stop_session()
```

---

## フォルダ一括バッチ推論の例

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

# CSV 出力 (Excel で開けるように utf-8-sig)
with open("results.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "judgement", "defect_found", "num_regions"])
    writer.writeheader()
    writer.writerows(results)

client.stop_session()
```

---

## WebSocket ストリーミングの例

```python
import time
from pathlib import Path
from seg_sdk import SegClient

client = SegClient("http://localhost:8002")
client.start_session(project_id="your-project-id", run_id="your-run-id")

# WebSocket ストリームを開く (websocket-client が必要)
stream = client.open_stream(source_id="cam-01")

for img_path in sorted(Path("./frames").glob("*.jpg")):
    frame_id = stream.send_frame(img_path.read_bytes(), frame_id=img_path.name)

    result = stream.recv_result(timeout=2.0)
    if result:
        print(f"[{result.frame_id}] {result.judgement} ({len(result.regions)} regions)")
    else:
        print(f"[{frame_id}] タイムアウト")

    time.sleep(0.033)  # ~30fps

stream.close()
client.stop_session()
```

---

## 非同期版の例

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

## データモデル

### InferenceResult

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `frame_id` | `str` | フレーム ID |
| `judgement` | `str` | `"OK"` または `"NG"` |
| `defect_found` | `bool` | 欠陥が検出されたか |
| `regions` | `list[Region]` | 検出された欠陥領域のリスト（面積の降順） |
| `summary` | `dict` | 画像全体の統計（下表参照） |
| `latency_ms` | `dict` | レイテンシ [ms]（下表参照） |
| `result_id` | `str` | 結果の一意 ID |

#### `summary` の中身

画像 1 枚あたりの全体統計です。

| キー | 型 | 意味 | 例 |
|------|-----|------|----|
| `fg_ratio` | `float` | 欠陥ピクセルが画像全体に占める比率（0.0〜1.0） | `0.0138` = 1.38% |
| `max_confidence` | `float` | 画像内で最も高い softmax 信頼度（0.0〜1.0） | `0.982` |
| `num_defects` | `int` | 検出された欠陥領域の個数（= `len(result.regions)`） | `3` |

#### `latency_ms` の中身

処理時間の内訳（ミリ秒）。ボトルネック分析に使えます。

| キー | 意味 |
|------|------|
| `decode` | 画像バイト列の JPEG/PNG デコード |
| `inference` | モデル推論本体（sliding-window） |
| `postprocess` | 連結成分抽出・領域統計・オーバーレイ生成 |
| `total` | 上記すべてを含むサーバ側の総処理時間 |

### Region

各要素は `cv2.connectedComponentsWithStats` で抽出された 1 つの連結欠陥領域です。
座標値はすべて **入力画像の元解像度** に揃っているので、クライアント側でスケーリングする必要はありません。

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `class_name` | `str` | 欠陥クラス名 |
| `class_id` | `int` | 欠陥クラス ID |
| `area_px` | `int` | 欠陥領域の面積（ピクセル） |
| `bbox` | `tuple[int,int,int,int]` | バウンディングボックス `(x, y, w, h)` |
| `centroid` | `tuple[int,int]` | 重心座標 `(cx, cy)`。ロボットへのピック座標送信等に便利 |
| `confidence` | `float` | 領域内の平均信頼度 |

---

## エラーハンドリング

```python
import requests
from seg_sdk import SegClient

client = SegClient("http://localhost:8002")

try:
    client.start_session(project_id="xxx", run_id="yyy")
    result = client.predict(open("test.jpg", "rb").read())
except requests.exceptions.ConnectionError:
    print("サーバーに接続できません。推論サーバーが起動しているか確認してください。")
except requests.exceptions.HTTPError as e:
    print(f"HTTP エラー: {e.response.status_code} - {e.response.text}")
except FileNotFoundError:
    print("画像ファイルが見つかりません。")
```

---

## FAQ

### Q: `import seg_sdk` なのに pip install は `seg-inference-sdk` ?
A: はい。pip パッケージ名は `seg-inference-sdk`、Python の import 名は `seg_sdk` です。

### Q: `start_session` は毎回呼ぶ必要がありますか?
A: いいえ。一度呼べばセッションが維持されます。同じモデルで連続推論する場合は最初の 1 回だけで OK です。

### Q: `recv_result` が `None` を返します
A: タイムアウトです。`timeout` を大きくしてみてください。サーバー側の処理が重い場合や、フレームが drop された場合に `None` が返ります。

### Q: `ModuleNotFoundError: No module named 'websocket'`
A: WebSocket を使うには追加インストールが必要です:
```bash
pip install "seg-inference-sdk[ws]"
```

### Q: `ModuleNotFoundError: No module named 'httpx'`
A: 非同期クライアントを使うには追加インストールが必要です:
```bash
pip install "seg-inference-sdk[async]"
```

### Q: 画像形式は何がいいですか?
A: **JPEG 推奨**です。PNG も使えますが、ファイルサイズが大きくなるためネットワーク転送に時間がかかります。

### Q: サーバーで `SEG_API_TOKEN` を設定するとリクエストが失敗します (HTTP 401 / WebSocket が code 4401 で切断)
A: サーバーで `SEG_API_TOKEN` が設定されている場合、`/v2/*` と `/ws/v2/*` は `X-API-Token` ヘッダ（WebSocket は `?api_token=` クエリでも可）が必須になります。SDK は現時点でこのトークン送信に**未対応**のため、`SEG_API_TOKEN` 未設定のサーバー（localhost デフォルト）に対して使用してください。

---

## サンプルスクリプト一覧

| ファイル | 内容 |
|---------|------|
| [`examples/quick_start.py`](examples/quick_start.py) | 最小構成の単発推論 |
| [`examples/batch_inspect.py`](examples/batch_inspect.py) | フォルダ一括推論 + CSV 出力 |
| [`examples/ws_stream.py`](examples/ws_stream.py) | WebSocket ストリーミング |
| [`examples/async_example.py`](examples/async_example.py) | 非同期クライアント |

---

## 迷ったらこれだけで大丈夫です

```python
from seg_sdk import SegClient

client = SegClient("http://localhost:8002")
client.start_session(project_id="your-project-id", run_id="your-run-id")
result = client.predict(open("image.jpg", "rb").read())
print(result.judgement)  # "OK" or "NG"
```
