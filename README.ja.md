<div align="center">

# Seg-Studio

**セマンティックセグメンテーションモデルの学習、アノテーション、デプロイを一つのデスクトップアプリで。**

![License](https://img.shields.io/badge/license-Apache_2.0-blue)
![Version](https://img.shields.io/badge/version-0.9.7-orange)
![Platform](https://img.shields.io/badge/platform-Windows%20|%20macOS-lightgrey)
![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/status-beta-yellow)

Seg-Studio は、ローカルマシン上で完結するオープンソースのセマンティックセグメンテーション統合環境です。SAM を活用したスマートツールで画像をアノテーションし、PyTorch モデルをリアルタイム監視付きで学習し、CoreML や ONNX にエクスポートしてエッジデバイスにデプロイできます。クラウドアカウントは不要です。

[クイックスタート](#クイックスタート) | [機能一覧](#主な機能) | [English](README.md)

</div>

<p align="center">
  <img src="docs/images/hero.gif" alt="SAMアノテーション、学習、評価、エクスポートをワンストップで" width="900" />
</p>

<p align="center"><sub>44 秒の一連の流れ: SAM クリックでアノテーション → auto-tune で学習 → ヒートマップと評価レポートを確認 → ONNX / CoreML へエクスポート。</sub></p>

---

## スクリーンショット

<table>
  <tr>
    <td align="center">
      <img src="docs/images/screenshot_projects.png" width="400" /><br />
      <b>Projects</b> -- データセットと学習結果の管理
    </td>
    <td align="center">
      <img src="docs/images/screenshot_annotate.png" width="400" /><br />
      <b>Annotate</b> -- ブラシ、ワンド、SAM クリックなど
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="docs/images/screenshot_training.png" width="400" /><br />
      <b>Training</b> -- リアルタイムの損失値と F1 曲線
    </td>
    <td align="center">
      <img src="docs/images/screenshot_results.png" width="400" /><br />
      <b>Results</b> -- 推論結果、ヒートマップ、エクスポート
    </td>
  </tr>
</table>

---

## なぜ Seg-Studio なのか

| | **Seg-Studio** | LabelMe | CVAT | Label Studio |
|---|:---:|:---:|:---:|:---:|
| アノテーションツール | あり | あり | あり | あり |
| SAM クリックセグメンテーション | 5モデル内蔵 | なし | 組み込み | ML Backend 経由 |
| モデル学習（内蔵） | あり | なし | なし | ML Backend 経由 |
| リアルタイム学習モニター | あり | N/A | N/A | N/A |
| CoreML / ONNX エクスポート | あり | N/A | N/A | N/A |
| 単一 GPU、クラウド不要 | あり | あり | Docker | Docker or Cloud |
| 自動チューニング（損失、LR、重み） | あり | N/A | N/A | N/A |

---

## 主な機能

**アノテーション**
- ブラシ、ポリゴン、ワンド（フラッドフィル）、スポット検出、クラックトレース、スーパーピクセル、Perlin ベースのリッジ検出
- Move ツール：マスク領域のドラッグによる位置調整
- Mark Clean：欠陥なし画像のフラグ付け
- SAM クリックセグメンテーション -- 5 モデル対応（MobileSAM、SAM2 Tiny/Small、TinySAM、EfficientSAM）
- MLP Assist による半自動ラベリング
- レシピベースの自動ラベリングパイプライン

**学習**
- PyTorch 学習 + WebSocket によるリアルタイム監視（損失値、F1、mIoU）
- 学習モード選択：通常 / クイック / 転移学習
- Auto-config v2：プロジェクト統計から arch / patch_size / base_channels を自動推薦
- 重心ベースのアノテーションパッチサンプリング
- Lovász-Softmax Loss
- 損失関数、学習率、クラス重みの自動チューニング
- 高解像度画像向けスライディングウィンドウ検証
- 知識蒸留サポート（DINOv2 特徴量蒸留）
- マルチプロジェクト対応のグローバル学習キュー（予約 run の自動起動）

**Results とデプロイ**
- GT / 推論マスクの分離表示、パターンオーバーレイ（斜線 / ドット / 格子）
- ピクセルレベルの Confidence ヒートマップ、ヒストグラム、画像ごとのスコア
- 後処理 CCA（連結成分、最小面積フィルタ）
- Live inspection モード、複数プロジェクトのバッチエクスポート
- 評価レポート生成（HTML / PDF / Excel）
- iPad / iPhone デプロイ向け CoreML エクスポート
- クロスプラットフォーム推論向け ONNX エクスポート
- OpenVINO IR エクスポート（INT8 量子化対応、`--with-openvino` でのオプトインインストール）
- Results ビューでのカウント・面積計測

**インターフェース**
- 日英バイリンガル UI（ヘッダーからいつでも切り替え可能）

**インタラクティブなオンボーディング**
- UI 内ハンズオンチュートリアル：初級 / 中級 / エキスパートの 3 モード、
  スポットライトオーバーレイ、SVG アニメ、キーボードショートカットに対応。
  ヘッダーの ▶ ボタンからいつでも再生可能。
- 「次に見るべきタブ」のガイド点滅と、未確認結果の青色パルスで
  初見ユーザーを各ステップへ誘導。

---

## クイックスタート

### 1. コードを入手する（git 不要）

[Releases ページ](https://github.com/segmen-pixel/seg-studio/releases) から
最新の **Source code (zip)** をダウンロードし、好きな場所に展開してください
（リポジトリページ上部の緑の **Code → Download ZIP** ボタンでも同じです）。
git を使う場合: `git clone https://github.com/segmen-pixel/seg-studio.git`

### 2. インストールして起動する

**Windows（NVIDIA GPU）:**
```bash
install-windows.bat
start-windows.bat
```

ターミナルは不要です。展開したフォルダ直下の `install-windows.bat` →
`start-windows.bat` をダブルクリックするだけでも動きます。インストーラーは GPU を自動判別し
（`cpu` / `cuda124` 指定で上書き可）、Python 3.10+ が見つからない場合は
インストール手順を案内します。起動スクリプトはサーバー準備完了後に
ブラウザで UI を自動的に開きます。

**macOS（Apple Silicon / Intel）:**
```bash
bash install-macos.sh
bash start-macos.sh
```

ブラウザで **http://localhost:8002/ui/** を開いてください。
停止は `stop-windows.bat` / `bash stop-macos.sh` でできます。

> git は任意です。インストーラーが git を使うのは SAM アシスト用ライブラリの
> 取得だけで、git が無くても SAM クリックセグメンテーション以外は全て動作します。
> macOS: Apple Silicon では推論に MPS（Metal）が自動的に使われます。学習には NVIDIA CUDA GPU（Windows）が必要です。

**Docker（docker compose）:**
```bash
docker compose up --build
```

ブラウザで **http://localhost:5173/** を開いてください。UI コンテナの nginx が
`/api`、`/v2`、`/ws` を trainer API にプロキシします。すべてのポートは
`127.0.0.1` のみに公開されます。

---

## ワークフロー

```
Projects  -->  Annotate  -->  Train  -->  Results  -->  Deploy
   |              |             |            |            |
 プロジェクト   SAM、ブラシ、   パラメータ    推論結果の    CoreML
 の作成または   ワンド、MLP     設定と学習    評価と比較    または ONNX
 インポート     でラベル付け    の実行                     にエクスポート
```

---

## アーキテクチャ選択

| アーキテクチャ | パラメータ数 | モデルサイズ | 推論 (RTX 3090) | 特徴 |
|---|---:|---:|---:|---|
| **SimpleUNet** (bc=64) | 1.9 M | 7.3 MB | 2.9 ms · 339 img/s | 安定、高 F1、GroupNorm + SE attention |
| **STDC** (bc=32) | 2.9 M | 11.2 MB | 1.3 ms · 758 img/s | 軽量、最速推論 |
| **DeepLabV3+** (bc=32) | 4.8 M | 18.5 MB | 5.1 ms · 198 img/s | ASPP + MobileNetV3 エンコーダ、広い受容野 |

全モデルが GroupNorm、JIT トレーシング、設定可能な output stride に対応しています。
推論レイテンシは 256×256・バッチサイズ1 の単一画像での値です。GPU + CPU の
完全なベンチマークと再現方法は [BENCHMARKS.md](BENCHMARKS.md) を参照してください。

学習のデフォルトは **DeepLabV3+** です — 37 プロジェクトの工場検査
ライブラリで最多 (17/37、STDC 15、SimpleUNet 5) のプロジェクトで
per-project best でした。推論速度優先なら STDC、メモリ最小なら
SimpleUNet を選んでください。

---

## MCP ブリッジ

Seg-Studio を MCP 対応ツールに接続して、プログラムからプロジェクトを検査できます:

```bash
pip install fastmcp httpx
python scripts/mcp_server.py --api http://localhost:8002 --policy read
```

データセット、アノテーション、学習、推論、エクスポート、システムの各カテゴリにわたる 37 ツールを提供。ポリシーレベル: `read`、`write`、`full`。

---

## 動作環境

- **OS:** Windows 10 / 11（64-bit）または macOS 12 以上（Apple Silicon 推奨）
- **GPU（学習）:** 学習には CUDA 対応 NVIDIA GPU（Windows）が必要（VRAM 4 GB 以上推奨）。Apple Silicon（MPS）/ CPU はアノテーションと推論のみ。
  - Windows インストーラーの既定は CUDA 12.8 PyTorch wheel（Turing /
    RTX 20xx 以降、Blackwell RTX 5090 を含む）。旧世代 GPU
    （Maxwell / Pascal / Volta）では `install_windows.bat cuda124` で
    CUDA 12.4 wheel を導入
  - Blackwell（sm_120）環境向けの pin 済み並列 lockfile
    `requirements-cu128.txt` も提供。手順は
    [`docs/BLACKWELL_MIGRATION.md`](docs/BLACKWELL_MIGRATION.md) を参照
- **Python:** 3.10 以上
- **Node.js:** 18 以上（UI ビルド用）

---

## プロジェクト構成

```
seg-studio/
  apps/
    trainer_api/     # FastAPI バックエンド
    serving_api/     # ONNX 推論 API
    trainer_ui/      # React フロントエンド
  packages/
    segcore/         # 学習コア（モデル定義、データセット、学習ループ）
    seg-sdk/         # 推論 API 向け Python クライアント SDK
  models/
    sam_checkpoints/ # SAM モデルチェックポイント
  scripts/
    windows/         # Windows セットアップ・起動スクリプト
    macos/           # macOS セットアップ・起動スクリプト
```

---

## コミュニティ

- **コントリビュート** -- プルリクエストを歓迎します。大きな変更の場合は、まず Issue を作成してください。
- **ディスカッション** -- 質問やアイデアは [GitHub Discussions](https://github.com/segmen-pixel/seg-studio/discussions) をご利用ください。
- **セキュリティ** -- 脆弱性の報告は [GitHub Security Advisories](https://github.com/segmen-pixel/seg-studio/security/advisories) から非公開で送信してください。

---

## ドキュメント

### はじめての方へ

- 📘 **[はじめてのハンドブック](docs/ja/handbook.md)** — 14 章の通しワークフロー（画像→モデル→推論まで）
- 📗 **[機能カタログ](docs/ja/catalog.md)** — 機能一覧（1 枚もの、俯瞰用）

### リファレンス

- [ユーザーガイド](docs/ja/user-guide.md)
- [開発者クイックスタート](docs/ja/dev-quickstart.md)
- [デプロイ手順](docs/ja/deployment.md)
- [トラブルシューティング](docs/ja/troubleshooting.md)
- [インポート / エクスポート](docs/ja/import_export.md)
- [ロードマップ](docs/ja/ROADMAP.md)
- [API リファレンス](http://localhost:8002/docs)（サーバー起動中に利用可能）

英語版ドキュメントは [README.md](README.md) を参照してください。

---

<div align="center">

Copyright 2026 Segmen-Pixel and Seg-Studio contributors.
[Apache License 2.0](LICENSE) に基づきライセンスされます。

サードパーティライセンス: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) /
上流アトリビューション: [NOTICE](NOTICE)

</div>

---

## 免責事項

本ソフトウェアおよび同梱・参照される学習済みモデル（SAM 系、DINOv2 等）は、
[Apache License 2.0](LICENSE) 第 7 条に基づき「現状有姿（AS IS）」で提供されます。
学習済みモデルの利用結果（推論結果の正確性、第三者の権利との関係を含む）について、
製作者および貢献者は一切の責任を負いません。
産業用途や安全性が要求される文脈で利用される場合は、利用者ご自身の責任で
適切な検証を行った上でご使用ください。

各社商標（Apple, CoreML, PyTorch, NVIDIA, CUDA, ONNX, SAM, DINOv2 等）は
それぞれの権利者に帰属します。詳細は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
をご参照ください。
