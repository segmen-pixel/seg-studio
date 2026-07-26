#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""最小構成の単発推論サンプル。

使い方:
    python quick_start.py
"""
from __future__ import annotations

import sys

import requests
from seg_sdk import SegClient

# ---- 設定 ----
BASE_URL = "http://localhost:8002"
PROJECT_ID = "your-project-id"
RUN_ID = "your-run-id"
IMAGE_PATH = "test.jpg"


def main() -> None:
    # クライアント作成
    client = SegClient(BASE_URL, timeout=30)

    try:
        # セッション開始 (モデルロード + ウォームアップ)
        print("セッション開始中...")
        client.start_session(project_id=PROJECT_ID, run_id=RUN_ID, backend="onnx")
        print("セッション開始完了")

        # 画像を読み込んで推論
        image_bytes = open(IMAGE_PATH, "rb").read()
        result = client.predict(image_bytes)

        # 結果を表示
        print(f"判定           : {result.judgement}")
        print(f"欠陥あり       : {result.defect_found}")
        print(f"検出領域数     : {len(result.regions)}")

        # サマリ情報 (辞書) — 画像全体の統計
        #   fg_ratio       : 欠陥ピクセルの全体比率 (0.0〜1.0)
        #   max_confidence : 画像内の最大信頼度 (0.0〜1.0)
        #   num_defects    : 検出された欠陥領域数 (= len(result.regions))
        s = result.summary
        print("サマリ情報:")
        print(f"  fg_ratio       : {s.get('fg_ratio', 0.0):.4%}  (欠陥ピクセル比率)")
        print(f"  max_confidence : {s.get('max_confidence', 0.0):.3f}  (画像内の最大信頼度)")
        print(f"  num_defects    : {s.get('num_defects', 0)}")

        # レイテンシ (辞書) — 処理時間の内訳 [ms]
        #   decode       : 画像デコード
        #   inference    : モデル推論 (sliding-window)
        #   postprocess  : CCA + 領域抽出
        #   total        : 全体
        lat = result.latency_ms
        print("レイテンシ [ms]:")
        print(f"  decode={lat.get('decode', 0)}  "
              f"inference={lat.get('inference', 0)}  "
              f"postprocess={lat.get('postprocess', 0)}  "
              f"total={lat.get('total', 0)}")

        # 各NG領域 — 面積の大きい順に並んでいる
        if result.regions:
            print("検出された欠陥領域:")
            for i, region in enumerate(result.regions, 1):
                x, y, w, h = region.bbox
                cx, cy = region.centroid
                print(
                    f"  [{i}] {region.class_name} (id={region.class_id}) "
                    f"area={region.area_px}px  "
                    f"bbox=(x={x}, y={y}, w={w}, h={h})  "
                    f"centroid=({cx}, {cy})  "
                    f"conf={region.confidence:.3f}"
                )

    except FileNotFoundError:
        print(f"エラー: 画像ファイルが見つかりません: {IMAGE_PATH}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(
            f"エラー: サーバーに接続できません: {BASE_URL}\n"
            "推論サーバーが起動しているか確認してください。",
            file=sys.stderr,
        )
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"エラー: HTTP {e.response.status_code} - {e.response.text}", file=sys.stderr)
        sys.exit(1)
    finally:
        # セッション終了
        try:
            client.stop_session()
        except Exception:
            pass


if __name__ == "__main__":
    main()
