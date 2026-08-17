---
marp: true
theme: default
paginate: true
size: 16:9
header: "Seg-Studio 機能カタログ"
footer: "Seg-Studio v0.9.8 — 2026-07"
style: |
  section {
    font-family: "Noto Sans JP", "Hiragino Sans", system-ui, sans-serif;
    background: #fafafa;
  }
  section.title {
    background: linear-gradient(135deg, #1565c0, #4fc3f7);
    color: white;
    justify-content: center;
    text-align: center;
  }
  section.title h1 { font-size: 64px; margin: 0; }
  section.title h2 { font-size: 24px; margin-top: 16px; opacity: 0.9; font-weight: normal; }
  h1 { color: #1565c0; border-bottom: 3px solid #4fc3f7; padding-bottom: 8px; }
  h2 { color: #1976d2; }
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 14px;
    font-weight: bold;
  }
  .star { background: #ffca28; color: #424242; }
  .good { background: #66bb6a; color: white; }
  .std { background: #90a4ae; color: white; }
  table { font-size: 18px; }
  table th { background: #e3f2fd; }
---

<!-- _class: title -->

# Seg-Studio

## 画像セグメンテーションをオールインワンで
### アノテ → 学習 → 評価 → 配布、1 つのツールで完結

---

# なぜ Seg-Studio？

| 特徴 | Seg-Studio | 他ツール |
|------|:---:|:---:|
| ブラウザで完結 | ✅ | — |
| SAM ワンクリック抽出 | ✅ | 一部 |
| Perlin CutPaste 合成 | ✅ | ほぼなし |
| 屋外用 Lighting 拡張 | ✅ | なし |
| CoreML Updatable 出力 | ✅ | なし |
| 日本語 UI | ✅ | 多くはない |

<br>

**つまり** — 国内製造業 / 現場検査に特化した、即戦力のフルスタック GUI。

---

# 🖊 アノテーション

| 機能 | できること | バッジ |
|---|---|:---:|
| ブラシ / ワンド / 消しゴム | 基本の描画ツール一式 | <span class="badge std">標準</span> |
| 🆒 **SAM マーキング** | クリック 1 発で対象抽出 (Meta SAM) | <span class="badge star">目玉</span> |
| クラック追跡 | クリックで候補を自動トレース → 選択 | <span class="badge good">便利</span> |
| スポット検出 | DoG で点状欠陥を検出 | <span class="badge good">便利</span> |
| Superpixel 選択 | 似た領域をクリック 1 発 | <span class="badge good">便利</span> |
| **Mark Clean** | OK 画像をワンクリックで学習データ化 | <span class="badge star">目玉</span> |

<!-- スクショ (Marp bg): assets/handbook/catalog_annotate.png -->

---

# 🧪 データ拡張

| 機能 | できること | バッジ |
|---|---|:---:|
| 🆒 **Perlin CutPaste** | 既存欠陥を歪ませランダム位置に貼付 | <span class="badge star">目玉</span> |
| 🆒 **Lighting バリエーション** | 日中/夕方/夜 の色温度変換（屋外） | <span class="badge star">目玉</span> |
| Auto-config | データ特性から最適設定を推薦 | <span class="badge good">便利</span> |
| Annotation Patches | 重心ベースの効率パッチサンプリング | <span class="badge std">標準</span> |

<!-- スクショ (Marp bg): assets/handbook/catalog_augment.png -->

---

# 🎓 学習

| 機能 | できること | バッジ |
|---|---|:---:|
| 学習モード | 通常 / クイック / 転移 / 個数カウント | <span class="badge std">標準</span> |
| ローカル GPU 学習 | そのままの PC で | <span class="badge std">標準</span> |
| 転移学習 | 過去 run のチェックポイントを指定して追加学習 | <span class="badge good">便利</span> |
| 🆒 **DINOv2 特徴蒸留** | 142M 画像の教師から蒸留 | <span class="badge star">目玉</span> |
| Lovász-Softmax Loss | 境界に強いロス選択可 | <span class="badge good">便利</span> |
| 多様な arch | SimpleUNet / STDC | <span class="badge std">標準</span> |
| Deep Supervision | 補助ロスで中間層も学習 | <span class="badge std">標準</span> |
| 🆒 **個数カウント** | 既存マスクから学習して 1 枚あたりの部品数を数える | <span class="badge star">目玉</span> |
| タイル学習・推論 | 小さな物体を撮影時の解像度のまま扱う | <span class="badge good">便利</span> |

<!-- スクショ (Marp bg): assets/handbook/catalog_training.png -->

---

# 📊 評価・可視化

| 機能 | できること | バッジ |
|---|---|:---:|
| メトリクス一覧 | F1 / mIoU / Precision / Recall | <span class="badge std">標準</span> |
| ヒートマップ | 信頼度を色で可視化 | <span class="badge good">便利</span> |
| **CCA 分析** | 検出領域の数・サイズ分布 | <span class="badge good">便利</span> |
| **Pattern Overlay** | 欠陥だけを背景に重畳 | <span class="badge good">便利</span> |
| 🆒 **ライブ検査** | カメラ映像にリアルタイム推論 | <span class="badge star">目玉</span> |
| しきい値スライダ | Confidence 閾値を即座に変更 | <span class="badge good">便利</span> |
| Batch Export | 全画像の推論結果を一括書き出し | <span class="badge good">便利</span> |

<!-- スクショ (Marp bg): assets/handbook/catalog_results.png -->

---

# 📦 配布・エクスポート

| 機能 | できること | バッジ |
|---|---|:---:|
| ONNX | Python/C++ サーバー推論 | <span class="badge std">標準</span> |
| CoreML | iOS / macOS アプリ組み込み | <span class="badge std">標準</span> |
| 🆒 **CoreML Updatable** | iOS 上でオンデバイス再学習 | <span class="badge star">目玉</span> |
| OpenVINO | FP32/FP16/INT8、Intel エッジ向け | <span class="badge good">便利</span> |
| Python SDK | `pip install` ですぐ使える | <span class="badge std">標準</span> |
| REST API | `/v2/infer` で単発推論 | <span class="badge std">標準</span> |
| カウント API | `/count` でクラス別個数と個体ごとの矩形 | <span class="badge good">便利</span> |
| 🆒 **WebSocket 推論** | 連続フレームで低遅延ストリーミング | <span class="badge star">目玉</span> |

<!-- スクショ (Marp bg): assets/handbook/catalog_export.png -->

---

# 代表ワークフロー（5 ステップ）

1. **プロジェクト作成** → クラス設定（例: キズ / シミ）
2. **画像 D&D** → 動画フレーム抽出も可
3. **アノテート** （SAM / ブラシ / クラック追跡）→ Mark Clean
4. **モード選択 → 学習** （データ分割は自動）
5. **Results 確認** → Export（ONNX / CoreML / Updatable）

<!-- スクショ (Marp bg): assets/handbook/01_overview.png -->

**詳細は** 📘 `docs/ja/handbook.md` の 16 章通しで。

---

# おすすめユースケース

| ユースケース | 特に刺さる機能 |
|---|---|
| 🏭 製造検査 (キズ/シミ/欠陥) | SAM / Perlin CutPaste / ライブ検査 |
| 🌳 屋外検査 (ハンドヘルド) | Lighting 拡張 / Mark Clean / 大量 BG 投入 |
| 📱 モバイルアプリ | CoreML Updatable / 軽量 STDC |
| 🔬 研究用プロトタイプ | 多 arch / Lovász / DINOv2 蒸留 |
| 🏢 オンプレ運用サーバ | REST / WebSocket SDK |

---

<!-- _class: title -->

# Start Here

## 📘 [はじめてのハンドブック](handbook.md)
## 📗 [機能カタログ](catalog.md)
## 🏠 [README](../../README.ja.md)

### Apache 2.0 / Copyright 2026 Contributors
