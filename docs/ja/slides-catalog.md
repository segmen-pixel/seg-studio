---
marp: true
theme: default
paginate: true
size: 16:9
header: "Seg-Studio 機能カタログ"
footer: "Seg-Studio"
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
### アノテーション → 学習 → 評価 → 配布まで、1つのツールで完結

---

# なぜ Seg-Studio？

- ブラウザだけで完結
- SAM によるワンクリック抽出
- Perlin CutPaste 合成
- 屋外向けの Lighting 拡張
- CoreML Updatable 形式での出力
- 日本語と英語の UI

<br>

**つまり**、現場の検査業務に合わせて作った、フルスタックの GUI ツールです。

---

# 🖊 アノテーション

| 機能 | できること | バッジ |
|---|---|:---:|
| ブラシ / 消しゴム / バケツ | 基本の描画ツール一式 | <span class="badge std">標準</span> |
| 🆒 **SAM マーキング** | クリック1回で対象を切り出す（Meta SAM） | <span class="badge star">目玉</span> |
| ひび割れ検出 | クリックで候補を自動トレースし、選んで確定 | <span class="badge good">便利</span> |
| スポット検出 | 塗った見本に似た点状の欠陥を検出 | <span class="badge good">便利</span> |
| AI サポートパネル | 視覚モデルが MCP ブリッジ経由でラベル付け | <span class="badge good">便利</span> |
| **OK（欠陥なし）** | OK 画像をワンクリックで学習データにする | <span class="badge star">目玉</span> |

<!-- スクショ (Marp bg): assets/handbook/catalog_annotate.png -->

---

# 🧪 データ拡張

| 機能 | できること | バッジ |
|---|---|:---:|
| 🆒 **Perlin CutPaste** | 既存の欠陥を変形させ、ランダムな位置に貼り付け | <span class="badge star">目玉</span> |
| 🆒 **Lighting バリエーション** | 日中 / 夕方 / 夜の色調に変換（屋外向け） | <span class="badge star">目玉</span> |
| Auto-config | データの特性から最適な設定を推薦 | <span class="badge good">便利</span> |
| Annotation Patches | 重心を基準にした効率的なパッチサンプリング | <span class="badge std">標準</span> |

<!-- スクショ (Marp bg): assets/handbook/catalog_augment.png -->

---

# 🎓 学習

| 機能 | できること | バッジ |
|---|---|:---:|
| 学習モード | 欠陥検知 / 個数検査 | <span class="badge std">標準</span> |
| ローカル GPU 学習 | 手元の PC でそのまま学習 | <span class="badge std">標準</span> |
| 🆒 **DINOv2 特徴蒸留** | 142M 枚の画像で学習した教師から蒸留 | <span class="badge star">目玉</span> |
| Lovász-Softmax Loss | 境界に強い損失関数を選択可能 | <span class="badge good">便利</span> |
| 多様な arch | SimpleUNet / STDC | <span class="badge std">標準</span> |
| Deep Supervision | 補助損失で中間層も学習 | <span class="badge std">標準</span> |
| 🆒 **個数カウント** | 既存のマスクから学習し、画像1枚あたりの部品数を数える | <span class="badge star">目玉</span> |
| タイル学習・推論 | 小さな物体を撮影時の解像度のまま扱う | <span class="badge good">便利</span> |

<!-- スクショ (Marp bg): assets/handbook/catalog_training.png -->

---

# 📊 評価・可視化

| 機能 | できること | バッジ |
|---|---|:---:|
| メトリクス一覧 | F1 / mIoU / Precision / Recall | <span class="badge std">標準</span> |
| ヒートマップ | 確信度を色で表示 | <span class="badge good">便利</span> |
| **CCA 分析** | 検出領域の数とサイズの分布 | <span class="badge good">便利</span> |
| **Pattern Overlay** | 欠陥だけを元の画像に重ねる | <span class="badge good">便利</span> |
| 🆒 **ライブ検査** | カメラ映像をリアルタイムで推論 | <span class="badge star">目玉</span> |
| しきい値スライダー | 確信度のしきい値をその場で変更 | <span class="badge good">便利</span> |
| Batch Export | 全画像の推論結果を一括で書き出し | <span class="badge good">便利</span> |

<!-- スクショ (Marp bg): assets/handbook/catalog_results.png -->

---

# 📦 配布・エクスポート

| 機能 | できること | バッジ |
|---|---|:---:|
| ONNX | Python / C++ のサーバーでの推論向け | <span class="badge std">標準</span> |
| CoreML | iOS / macOS アプリへの組み込み向け | <span class="badge std">標準</span> |
| 🆒 **CoreML Updatable** | iOS 上でオンデバイス再学習 | <span class="badge star">目玉</span> |
| OpenVINO | FP32 / FP16 / INT8。Intel のエッジデバイス向け | <span class="badge good">便利</span> |
| Python SDK | `pip install` ですぐに使える | <span class="badge std">標準</span> |
| REST API | `/v2/infer` で1枚ずつ推論 | <span class="badge std">標準</span> |
| カウント API | `/count` がクラス別の個数と、個体ごとの矩形を返す | <span class="badge good">便利</span> |
| 🆒 **WebSocket 推論** | 連続するフレームを低遅延でストリーミング推論 | <span class="badge star">目玉</span> |

<!-- スクショ (Marp bg): assets/handbook/catalog_export.png -->

---

# 代表的なワークフロー（5ステップ）

1. **プロジェクトの作成** → クラスの設定（例：キズ / シミ）
2. **画像のドラッグ＆ドロップ** → 動画からのフレーム抽出も可能
3. **アノテーション**（SAM / ブラシ / ひび割れ検出）→ 欠陥の無い画像に **OK**
4. **モードの選択 → 学習**（データの分割は自動）
5. **結果の確認** → エクスポート（ONNX / CoreML / Updatable）

<!-- スクショ (Marp bg): assets/handbook/01_overview.png -->

**全体を通して学ぶには** 📘 `docs/ja/handbook.md`（全16章を順に進める構成）をご覧ください。

---

# おすすめのユースケース

| ユースケース | 特に役立つ機能 |
|---|---|
| 🏭 製造検査（キズ / シミ / その他） | SAM / Perlin CutPaste / ライブ検査 |
| 🌳 屋外検査（ハンドヘルド） | Lighting 拡張 / OK 画像 / 背景画像の大量投入 |
| 📱 モバイルアプリ | CoreML Updatable / 軽量な STDC |
| 🔬 研究用のプロトタイプ | 多様な arch / Lovász / DINOv2 蒸留 |
| 🏢 オンプレミスの運用サーバー | REST / WebSocket SDK |

---

<!-- _class: title -->

# Start Here

## 📘 [はじめてのハンドブック](handbook.md)
## 📗 [機能カタログ](catalog.md)
## 🏠 [README](../../README.ja.md)

### Apache 2.0 / Copyright 2026 Contributors
