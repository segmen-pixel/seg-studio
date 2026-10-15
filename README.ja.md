<div align="center">

# Seg-Studio

**画像の「どこに何があるか」を塗り分ける AI を、自分の PC で作れるアプリ。**

![License](https://img.shields.io/badge/license-Apache_2.0-blue)
![Version](https://img.shields.io/badge/version-0.9.12-orange)
![Platform](https://img.shields.io/badge/platform-Windows%20|%20macOS%20|%20Linux-lightgrey)
![Python](https://img.shields.io/badge/python-3.11%2B-brightgreen)
![Status](https://img.shields.io/badge/status-beta-yellow)

[試してみる](#試してみる) | [できること](#できること) | [English](README.md)

</div>

<p align="center">
  <img src="docs/images/hero.gif" alt="SAMアノテーション、学習、評価、エクスポートをワンストップで" width="900" />
</p>

<p align="center"><sub>44秒で一通り：クリックでラベル付け → 学習 → 結果の確認 → 書き出し</sub></p>

Seg-Studio は、画像に写った傷・汚れ・部品・細胞などを**ピクセル単位で見つける AI モデル**を、プログラミングなしで作れるアプリです。画像に印を付け、学習させ、結果を確かめるまでを一つの画面で行えます。

- **データは外に出ません。** すべて手元の PC の中で動き、クラウドのアカウントも要りません。
- **無料で使えます。** オープンソース（Apache-2.0）で、業務にも利用できます。
- **ブラウザで操作します。** 起動するとブラウザに画面が開きます。日本語と英語に対応しています。

---

## できること

| | |
|---|---|
| 🖌️ **ラベル付け** | 画像の上で、見つけたいところを塗ります。クリック1つで形を切り出す AI（SAM）も使えます。 |
| 🧠 **学習** | 塗った画像から AI モデルを作ります。細かい設定は自動で決まるので、ボタンを押して待つだけです。進み具合と学習ログは、学習中に随時表示されます。 |
| 🔍 **確認** | 学習したモデルで新しい画像を判定し、何を見つけて何を見逃したかを画像で確かめます。レポートは PDF や Excel で保存できます。 |
| 🔢 **数える** | 接していたり重なっていたりする物も、1個ずつ数えます。すでに塗ったマスクをそのまま使うので、追加の作業は要りません。 |
| 📦 **書き出し** | 作ったモデルを ONNX や CoreML などの形式で書き出し、ほかのソフトウェアや iPhone / iPad で使えます。 |

全機能の一覧は[機能カタログ](docs/ja/catalog.md)にあります。

## スクリーンショット

<table>
  <tr>
    <td align="center">
      <img src="docs/images/screenshot_projects.png" width="400" /><br />
      <b>Projects</b>：データセットと学習結果の管理
    </td>
    <td align="center">
      <img src="docs/images/screenshot_annotate.png" width="400" /><br />
      <b>Annotate</b>：ブラシ、SAM のクリック指定など
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="docs/images/screenshot_training.png" width="400" /><br />
      <b>Training</b>：進み具合、スコア、学習ログをリアルタイムに表示
    </td>
    <td align="center">
      <img src="docs/images/screenshot_results.png" width="400" /><br />
      <b>Results</b>：推論結果、ヒートマップ、エクスポート
    </td>
  </tr>
</table>


---

## 試してみる

**必要なもの：** Windows 10 / 11、macOS、Linux のいずれか。約 5 GB の空きディスク。インストール中のインターネット接続。NVIDIA の GPU があると学習がずっと速くなります（無くても動きますが、学習に時間がかかります）。

### Windows

1. [Releases ページ](https://github.com/segmen-pixel/seg-studio/releases)から最新版の **Source code (zip)** をダウンロードし、`C:\seg-studio` のような短いパスの場所に展開します。
2. 展開したフォルダにある `install-windows.bat` をダブルクリックします。Python など足りないものも自動でインストールされます（ダウンロードが多いので、しばらく時間がかかります）。
3. 終わったら `start-windows.bat` をダブルクリックします。しばらくするとブラウザに画面が開きます。

### macOS

先に Python 3.11、Node.js、git をインストールしておきます（`brew install python@3.11 node git`）。`python3 --version` で 3.11以上と表示されることを確かめたら、zip を展開したフォルダでターミナルを開き、次のコマンドを実行します。

```bash
bash install-macos.sh
bash start-macos.sh
```

### 画面が開いたら

ブラウザで **http://localhost:8002/ui/** が開きます（開かないときは、このアドレスを入力してください）。画面は日本語で始まります。英語にするには、ヘッダーの右端にある **EN** をクリックします。初めての方は[はじめてのファーストラン手順](docs/ja/first-run-manual.md)をご覧ください。最初の結果が出るまで約10分です。

Linux、Docker、オフライン環境、旧世代の GPU でのインストール方法と、OS ごとに使える機能は、[インストールの詳細](docs/ja/install.md)にあります。

---

## 困ったとき

| こんなとき | 見るところ |
|---|---|
| インストールや起動がうまくいかない | [トラブルシューティング](docs/ja/troubleshooting.md) |
| ボタンや設定の意味を知りたい | [ユーザーガイド](docs/ja/user-guide.md) |
| サンプルデータで一通り練習したい | [はじめてのハンドブック](docs/ja/handbook.md)（全16章） |
| 質問したい、要望を伝えたい | [GitHub Issues](https://github.com/segmen-pixel/seg-studio/issues) |

---

## もっと詳しく（エンジニア向け）

| 目的 | 読むもの |
|---|---|
| AI モデル（ローカル LLM や、普段使っているアシスタント）に作業させる | [AI サポートと MCP ブリッジ](docs/ja/ai-support.md) |
| Ollama で AI サポートを使えるようにするまでを、順を追って確認する（Windows と macOS） | [Ollama で AI サポートを使う（手順例）](docs/ja/ai-support-ollama.md) |
| 共有マシンや LAN で動かす | [デプロイ手順](docs/ja/deployment.md) |
| データの取り込み・書き出しの形式を知る | [インポート / エクスポート](docs/ja/import_export.md) |
| Intel の CPU / NPU で推論する | [OpenVINO エクスポート](docs/openvino_export.md)（英語） |
| モデルの速度とサイズを比べる | [BENCHMARKS.md](BENCHMARKS.md)（英語） |
| 推論 API を呼び出す | サーバーの起動中に開ける [API リファレンス](http://localhost:8002/docs) |
| 開発に参加する | [CONTRIBUTING.md](CONTRIBUTING.md)（英語）と[開発者クイックスタート](docs/ja/dev-quickstart.md) |
| 今後の予定を知る | [ロードマップ](docs/ja/ROADMAP.md) |

### 主な機能

| 機能 | 内容 |
|---|---|
| アノテーションツール | ブラシ、矩形、消しゴム、バケツ塗り、スポット検出、ひび割れ検出 |
| SAM のクリックセグメンテーション | 5モデルを内蔵（macOS では4モデル） |
| モデル学習 | アプリに内蔵（同じ PC でそのまま学習） |
| リアルタイム学習モニター | 学習中にエポック、最高 F1、学習ログを随時表示 |
| モデルのエクスポート | ONNX と CoreML（OpenVINO は追加インストールで対応） |
| 動作環境 | 手元の PC（NVIDIA の GPU、Apple Silicon、または CPU のみ）。クラウドのサービスも Docker も不要 |
| 自動チューニング | 損失関数、学習率、クラス重み |
| データの取り込み・書き出し | 画像と PNG マスクの ZIP で、プロジェクトをインポート / エクスポート（[インポート / エクスポート](docs/ja/import_export.md)） |


### 機能一覧（全件）

<details>
<summary><b>機能一覧（全件）</b>（クリックで展開）</summary>

**アノテーション**
- ブラシ、矩形、消しゴム、バケツ塗り、スポット検出、ひび割れ検出
- Move ツール：マスク領域をドラッグして位置を調整
- OK：欠陥のない画像に OK の印を付ける
- SAM のクリックセグメンテーション：MobileSAM、SAM2 Tiny/Small、TinySAM、EfficientSAM の5モデルに対応（macOS では TinySAM が入らないため4モデル）
- AI サポートパネル：ツール呼び出しに対応した視覚モデルが、開いているプロジェクトを MCP ブリッジ経由でラベル付けし、その様子を画面で確認できる（ご自身で動かすモデルサーバーが必要。既定は Ollama。[MCP ブリッジ](docs/ja/ai-support.md)を参照）
- レシピベースの自動ラベル付けパイプライン

**学習**
- PyTorch による学習と、WebSocket によるリアルタイム監視（損失、F1、mIoU）
- 学習モードの選択：欠陥検知 / 個数検査
- 個数カウント（インスタンスセグメンテーション）：接触した物体を1個ずつ分離。既存のマスクから合成したデータで学習するため、追加のアノテーションは不要
- カウント用のタイル学習・推論：小さな物体を撮影時の解像度のまま扱う（学習と推論は常に同じパッチサイズ）
- Auto-config v2：プロジェクトの統計から arch / patch_size / base_channels を自動で推薦
- 重心ベースのアノテーションパッチサンプリング
- Lovász-Softmax Loss
- 損失関数、学習率、クラス重みの自動チューニング
- 高解像度画像向けのスライディングウィンドウ検証
- 知識蒸留に対応（DINOv2 の特徴量蒸留。Apache-2.0 のモデル定義は `segcore` に同梱しています。`scripts/build_installer.py` で作ったパッケージには Apache-2.0 の重みが入り、それ以外では初回の使用時にダウンロードします）
- 複数プロジェクトにまたがるグローバル学習キュー（予約した run を自動で開始）

**結果とデプロイ**
- GT / 推論マスクの分離表示、パターンオーバーレイ（斜線 / ドット / 格子）
- ピクセル単位の確信度ヒートマップ、ヒストグラム、画像ごとのスコア
- 後処理の CCA（連結成分、最小面積フィルタ）
- ライブ検査モード、複数プロジェクトの一括エクスポート
- 評価レポートの生成（HTML / PDF / Excel）
- iPad / iPhone への組み込み向けの CoreML エクスポート
- クロスプラットフォーム推論向けの ONNX エクスポート
- OpenVINO IR エクスポート（INT8 量子化に対応。`--with-openvino` を指定したときだけインストール）
- 結果ビューでのカウントと面積の計測（連結成分）。カウントモデルを学習すれば、個体単位のカウントも可能
- `POST /count` のサービングエンドポイント：クラス別の個数と、個体ごとの矩形を返す

**インターフェース**
- 日本語と英語の UI（ヘッダーからいつでも切り替え可能）

**インタラクティブなオンボーディング**
- UI 内のハンズオンチュートリアル。初級 / 中級 / エキスパートの3モードがあり、スポットライト表示、SVG アニメーション、キーボード操作に対応しています。ヘッダーの ▶ ボタンからいつでも再生できます。
- 「次に見るべきタブ」の点滅と、まだ見ていない結果を知らせる青いパルスで、初めての方を各ステップへ案内します。

</details>

---

## コミュニティ

- **コントリビュート**：プルリクエストを歓迎します。大きな変更をする場合は、先に Issue を作成してください。
- **質問・アイデア**：[GitHub Issues](https://github.com/segmen-pixel/seg-studio/issues/new/choose) から Issue を作成してください。
- **セキュリティ**：脆弱性は [GitHub Security Advisories](https://github.com/segmen-pixel/seg-studio/security/advisories) から非公開で報告してください。

---

<div align="center">

Copyright 2026 Segmen-Pixel and Seg-Studio contributors. [Apache License 2.0](LICENSE) に基づいてライセンスされています。

サードパーティのライセンス：[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) / 上流のアトリビューション：[NOTICE](NOTICE) / リリースごとの SBOM（CycloneDX と SPDX）：各 [GitHub Release](https://github.com/segmen-pixel/seg-studio/releases) に添付

</div>

---

## 免責事項

本ソフトウェアと、同梱または参照される学習済みモデル（SAM 系、DINOv2 など）は、[Apache License 2.0](LICENSE) 第7条に基づき「現状有姿（AS IS）」で提供されます。製作者および貢献者は、推論結果の正確性についても、その結果と第三者の権利との関係についても、何ら保証しません。産業用途や安全性が重要な業務に使う場合は、利用者ご自身の責任で検証してください。

各社の商標（Apple、CoreML、PyTorch、NVIDIA、CUDA、ONNX、SAM、DINOv2 など）は、それぞれの権利者に帰属します。詳しくは [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) をご覧ください。
