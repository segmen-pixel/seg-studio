#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""WebSocket ストリーミング推論サンプル。

画像フォルダを疑似カメラとして使い、WebSocket でフレームを送信します。

使い方:
    pip install "seg-inference-sdk[ws]"
    python ws_stream.py

事前に websocket-client パッケージが必要です。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests
from seg_sdk import SegClient

# ---- 設定 ----
BASE_URL = "http://localhost:8002"
PROJECT_ID = "your-project-id"
RUN_ID = "your-run-id"
FRAMES_DIR = "./frames"
FPS = 30  # 疑似カメラの fps

# 対応する画像拡張子
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    frames_dir = Path(FRAMES_DIR)
    if not frames_dir.is_dir():
        print(f"エラー: フレームフォルダが見つかりません: {frames_dir}", file=sys.stderr)
        sys.exit(1)

    # フレーム画像一覧
    frame_files = sorted(
        p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not frame_files:
        print(f"エラー: 画像ファイルが見つかりません: {frames_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"{len(frame_files)} フレームを検出しました")

    # クライアント作成・セッション開始
    client = SegClient(BASE_URL, timeout=30)
    try:
        client.start_session(project_id=PROJECT_ID, run_id=RUN_ID, backend="onnx")
    except requests.exceptions.ConnectionError:
        print(f"エラー: サーバーに接続できません: {BASE_URL}", file=sys.stderr)
        sys.exit(1)

    # WebSocket ストリームを開く
    print("WebSocket ストリーム接続中...")
    stream = client.open_stream(source_id="cam-01")
    print("接続完了")

    interval = 1.0 / FPS
    ok_count = 0
    ng_count = 0
    timeout_count = 0

    try:
        for i, img_path in enumerate(frame_files, 1):
            image_bytes = img_path.read_bytes()

            # フレーム送信
            frame_id = stream.send_frame(image_bytes, frame_id=img_path.name)

            # 結果受信
            result = stream.recv_result(timeout=2.0)
            if result is not None:
                status = result.judgement
                n_regions = len(result.regions)
                fg = result.summary.get("fg_ratio", 0.0)
                lat = result.latency_ms.get("total", 0)
                # 先頭 (最大領域) の重心だけ1行ログに出す。全部出すと
                # 30fps では流れすぎるので代表だけ。
                if result.regions:
                    top = result.regions[0]
                    cx, cy = top.centroid
                    print(
                        f"[{i}/{len(frame_files)}] {result.frame_id}: "
                        f"{status} regions={n_regions} fg={fg:.2%} "
                        f"top={top.class_name}@({cx},{cy}) "
                        f"lat={lat}ms"
                    )
                else:
                    print(
                        f"[{i}/{len(frame_files)}] {result.frame_id}: "
                        f"{status} regions=0 fg={fg:.2%} lat={lat}ms"
                    )
                if status == "OK":
                    ok_count += 1
                else:
                    ng_count += 1
            else:
                print(f"[{i}/{len(frame_files)}] {frame_id}: タイムアウト")
                timeout_count += 1

            # fps 制御
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n中断しました")
    finally:
        # ストリーム・セッション終了
        stream.close()
        client.stop_session()

    # サマリ表示
    total = ok_count + ng_count + timeout_count
    print("\n--- 完了 ---")
    print(f"合計: {total} フレーム")
    print(f"OK: {ok_count} / NG: {ng_count} / タイムアウト: {timeout_count}")


if __name__ == "__main__":
    main()
