# FA向けストリーミング推論API + SDK 設計書

> **歴史的設計メモ**: 本書は設計時点の計画です。`/v2/session/*`・`POST /v2/infer`・`WS /ws/v2/infer`（latest_wins）・同期/非同期 SDK は実装済みですが、以下は乖離しています: `/v2/results/{result_id}/*` エンドポイントと `exactly_once` モードは未実装。SDK はストリームクラスが `client.py`/`async_client.py` 内にあり（`stream.py` は無し）、配布は `setup.py` ではなく `pyproject.toml`。`InferenceResult` に `confidence` フィールドはありません（`regions[].confidence` を参照）。また `SEG_API_TOKEN` 設定時は `/v2/*`・`/ws/v2/*` もトークン必須ですが、SDK はまだ `X-API-Token` を送信しません。

## 設計協議結果

---

## 1. ベンチマーク結果

| 処理 | 所要時間 |
|------|---------|
| ORT GPU推論 (batch=1, 256x256) | **3ms** median, 11ms P99 |
| 理論最大FPS (推論のみ) | **285 fps** |
| 画像デコード (JPEG→numpy) | ~2-5ms |
| 前処理 (normalize) | ~1ms |
| 後処理 (argmax+統計) | ~2ms |
| WebSocket往復 (LAN) | ~1ms |
| **E2E推定** | **~10-15ms (65-100fps)** |

→ 最速要件は十分達成可能。Python GCのP99ジッター(+10-20ms)を含めても30fps以上。

---

## 2. アーキテクチャ

### プロトコル選択: WebSocket + REST ハイブリッド

| 用途 | プロトコル | 理由 |
|------|-----------|------|
| 連続フレーム推論 | **WebSocket** | 双方向、低オーバーヘッド、バックプレッシャー制御 |
| 単発推論 / 外部連携 | **REST** | シンプル、FAエンジニア向け、ファイアウォール通過容易 |
| モデル管理 / セッション | **REST** | ステートレス操作 |
| GUIモニタリング | **WebSocket** (結果購読) + REST (画像取得) |

**gRPC推奨について**: Phase 1 では WebSocket。gRPC は C++/C# クライアント生成が強みだが、依存が重くFAエンジニアにはハードルが高い。需要が出たら Phase 3 で検討。

### エンドポイント設計 (v2 API)

```
# ストリーミング推論
WS   /ws/v2/infer                    → フレーム投入 & 結果受信 (双方向バイナリ)

# 単発推論 (外部ソフト向け)
POST /v2/infer                       → 1枚投入 + 結果即返し (同期)

# セッション管理
POST /v2/session/start               → モデルロード + GPU ウォームアップ
POST /v2/session/stop                → セッション解放
GET  /v2/session/status              → ランタイム状態 (GPU, モデル, FPS)

# 結果取得 (マスク画像など重いデータ)
GET  /v2/results/{result_id}/mask.png
GET  /v2/results/{result_id}/detail
```

---

## 3. WebSocket プロトコル

### メッセージフロー

```
Client                          Server
  |                                |
  |--- WS connect --->             |
  |<-- hello.ok (session, credits) |
  |                                |
  |--- frame.meta (JSON) -------->|
  |--- frame.data (binary) ------>|
  |<-- frame.accept / frame.drop  |
  |<-- result (JSON) -------------|
  |                                |
  |--- frame.meta ------->        |  (次フレーム)
  ...
```

### メッセージ定義

```json
// hello.ok (server → client)
{
  "type": "hello.ok",
  "session_id": "s-abc123",
  "credits": 1,
  "policy": "latest_wins",
  "model_id": "screw-v7",
  "capabilities": ["judgement", "mask", "regions"]
}

// frame.meta (client → server)
{
  "type": "frame.meta",
  "frame_id": "f-1001",
  "ts_ns": 1741824000123000000,
  "content_type": "image/jpeg",
  "source_id": "cam-01"
}
// 直後に binary frame data

// frame.accept (server → client)
{ "type": "frame.accept", "frame_id": "f-1001" }

// frame.drop (server → client) — バックプレッシャー
{ "type": "frame.drop", "frame_id": "f-1001", "reason": "backpressure" }

// result (server → client)
{
  "type": "result",
  "frame_id": "f-1001",
  "judgement": "NG",
  "defect_found": true,
  "regions": [
    { "class": "scratch", "class_id": 12, "area_px": 1834, "bbox": [100,200,150,80], "confidence": 0.992 }
  ],
  "summary": { "fg_ratio": 0.012, "max_confidence": 0.992, "num_defects": 1 },
  "latency_ms": { "decode": 3.1, "preprocess": 0.8, "inference": 3.2, "postprocess": 1.5, "total": 8.6 },
  "result_id": "r-abc123"
}
```

---

## 4. InferenceRuntime 改修

### 新メソッド

```python
class InferenceRuntime:
    # 既存: predict_batch_stream() — バッチ推論用、変更なし

    def predict_one(self, image_bytes: bytes, model_spec: ModelSpec,
                    save_artifacts: bool = False, timeout_s: float = 1.0) -> InferenceResult:
        """単発即実行。GPUキューをバイパスし直接 session.run()。"""
        # 1. JPEG decode → numpy
        # 2. normalize
        # 3. session.run() (既にロード済みのセッション再利用)
        # 4. argmax + 統計計算
        # 5. InferenceResult 返却 (メモリのみ、ファイル保存しない)

    def offer_stream_frame(self, source_id: str, frame_id: str,
                           image_bytes: bytes, model_spec: ModelSpec) -> bool:
        """ストリーム用。latest-frame-wins キュー (depth=1) に投入。"""
        slot = self._stream_slots[source_id]  # queue.Queue(maxsize=1)
        if slot.full():
            slot.get_nowait()  # 古いフレーム破棄
        slot.put_nowait((frame_id, image_bytes, model_spec))
        return True
```

### バックプレッシャー戦略

| モード | キュー深さ | 動作 | 用途 |
|--------|-----------|------|------|
| `latest_wins` | 1 | 古いフレーム即破棄 | 連続検査 (デフォルト) |
| `exactly_once` | N | キュー溢れたら frame.drop 返却 | PLC トリガ検査 |

---

## 5. 結果フォーマット

### PLC向け (最小・数値中心)
```json
{
  "station_id": "LINE1-ST10",
  "frame_id": "f-1001",
  "judgement": 1,
  "defect_code": 12,
  "confidence_permille": 992,
  "area_px": 1834,
  "cycle_ms": 39
}
```

### MES向け (追跡情報付き)
```json
{
  "event_type": "inspection.result",
  "station_id": "LINE1-ST10",
  "serial_no": "SN00012345",
  "lot_no": "LOT20260313A",
  "recipe_id": "screw-v7",
  "frame_id": "f-1001",
  "captured_at": "2026-04-01T12:00:00.123Z",
  "judgement": "NG",
  "image_ref": "/v2/results/r-abc123/mask.png",
  "defects": [
    { "class_id": 12, "class_name": "scratch", "confidence": 0.992, "area_px": 1834 }
  ]
}
```

**ピクセルマスクは結果JSONに含めない**。必要時のみ result_id 経由で REST 取得。

---

## 6. Python SDK

### パッケージ構成
```
packages/seg-sdk/
  seg_sdk/
    __init__.py
    client.py          # SegClient (sync)
    async_client.py    # AsyncSegClient
    models.py          # InferenceResult, Region, etc.
    stream.py          # SegStream / AsyncSegStream
  setup.py             # wheel 配布対応 (オフライン環境用)
```

### インターフェース

```python
# 同期 (FAエンジニア向け — sync必須)
from seg_sdk import SegClient

client = SegClient("http://localhost:8002")
client.start_session(project_id="xxx", run_id="yyy")

# 単発推論
result = client.predict(open("frame.jpg", "rb").read())
print(result.judgement, result.confidence, result.latency_ms)

# ストリーム推論
stream = client.open_stream(source_id="cam-01")
stream.send_frame(frame_bytes, frame_id="f-1001")
result = stream.recv_result(timeout=0.2)
stream.close()

# 非同期
from seg_sdk import AsyncSegClient

async with AsyncSegClient("http://localhost:8002") as client:
    result = await client.predict(frame_bytes)

    async with client.open_stream("cam-01") as stream:
        await stream.send_frame(frame_bytes, frame_id="f-1001")
        result = await stream.recv_result()
```

### 配布
- `pip install seg-inference-sdk` (オンライン)
- `seg_sdk-x.y.z-py3-none-any.whl` (オフライン — 工場PCはインターネットなし)

---

## 7. GUI連携

- 既存の NDJSON バッチUI (`fetchRunPredictBatch`) は**そのまま維持**
- 新規: **ライブ検査画面** を追加
  - WebSocket で結果イベント購読
  - フレーム画像・マスクは REST で lazy load
  - リアルタイム統計 (FPS, OK/NG率, confidence分布)

---

## 8. レビュー指摘への対応表

| 指摘 | 対応 | Phase |
|------|------|-------|
| gRPC推奨 | WebSocket で開始、gRPC は需要次第 | Phase 3 |
| 別プロセス化 | 統合維持 + ウォッチドッグ (推論タイムアウト検知) | Phase 1 |
| PLC直結 | SDK レベルで対応。PLC プロトコル変換はユーザー責務 | Phase 1 |
| JSON重い | 統計値モード標準、ピクセルマスクは REST 別取得 | Phase 1 |
| オフライン SDK | wheel 配布 | Phase 1 |
| sync API必須 | SegClient (sync) + AsyncSegClient (async) 両方提供 | Phase 1 |
| P99レイテンシ計測 | ベンチマーク実施済み (3ms median, 11ms P99) | Done |
| C++ critical path | 不要 (3ms/frame で余裕。120fps GigE が来たら再検討) | Phase 3 |

---

## 9. 実装優先順位

1. `InferenceRuntime.predict_one()` — 単発即実行パス
2. `POST /v2/infer` — REST 単発推論エンドポイント
3. `WS /ws/v2/infer` — WebSocket ストリーミング
4. `seg_sdk.SegClient` — Python SDK (sync)
5. `seg_sdk.AsyncSegClient` — async版
6. GUI ライブ検査画面
