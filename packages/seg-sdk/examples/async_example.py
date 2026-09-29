#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""非同期クライアントの最小サンプル。

使い方:
    pip install "seg-inference-sdk[async]"
    python async_example.py

事前に httpx パッケージが必要です。
"""
from __future__ import annotations

import asyncio
import sys

import httpx
from seg_sdk import AsyncSegClient

# ---- 設定 ----
BASE_URL = "http://localhost:8002"
PROJECT_ID = "your-project-id"
RUN_ID = "your-run-id"
IMAGE_PATH = "test.jpg"


async def main() -> None:
    # 画像を読み込み
    try:
        image_bytes = open(IMAGE_PATH, "rb").read()
    except FileNotFoundError:
        print(f"エラー: 画像ファイルが見つかりません: {IMAGE_PATH}", file=sys.stderr)
        sys.exit(1)

    try:
        async with AsyncSegClient(BASE_URL, timeout=30) as client:
            # セッション開始
            print("セッション開始中...")
            await client.start_session(
                project_id=PROJECT_ID,
                run_id=RUN_ID,
                backend="onnx",
            )
            print("セッション開始完了")

            # 推論
            result = await client.predict(image_bytes)

            # 結果を表示
            print(f"判定           : {result.judgement}")
            print(f"欠陥あり       : {result.defect_found}")
            print(f"検出領域数     : {len(result.regions)}")

            # サマリ (画像全体の統計)
            s = result.summary
            print("サマリ情報:")
            print(f"  fg_ratio       : {s.get('fg_ratio', 0.0):.4%}  (欠陥ピクセル比率)")
            print(f"  max_confidence : {s.get('max_confidence', 0.0):.3f}")
            print(f"  num_defects    : {s.get('num_defects', 0)}")

            # レイテンシ [ms]
            lat = result.latency_ms
            print(
                f"レイテンシ [ms]  : decode={lat.get('decode', 0)} "
                f"inference={lat.get('inference', 0)} "
                f"postprocess={lat.get('postprocess', 0)} "
                f"total={lat.get('total', 0)}"
            )

            # 各NG領域 (面積の大きい順)
            for i, region in enumerate(result.regions, 1):
                x, y, w, h = region.bbox
                cx, cy = region.centroid
                print(
                    f"  [{i}] {region.class_name} (id={region.class_id}) "
                    f"area={region.area_px}px "
                    f"bbox=(x={x}, y={y}, w={w}, h={h}) "
                    f"centroid=({cx}, {cy}) "
                    f"conf={region.confidence:.3f}"
                )

            # セッション終了
            await client.stop_session()

    except httpx.ConnectError:
        print(
            f"エラー: サーバーに接続できません: {BASE_URL}\n"
            "推論サーバーが起動しているか確認してください。",
            file=sys.stderr,
        )
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"エラー: HTTP {e.response.status_code} - {e.response.text}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
