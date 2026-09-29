<div align="center">

# Seg-Studio

**画像セグメンテーションモデルの学習、アノテーション、デプロイを一つのデスクトップアプリで。**

![License](https://img.shields.io/badge/license-Apache_2.0-blue)
![Version](https://img.shields.io/badge/version-0.9.10-orange)
![Platform](https://img.shields.io/badge/platform-Windows%20|%20macOS%20|%20Linux-lightgrey)
![Python](https://img.shields.io/badge/python-3.11%2B-brightgreen)
![Status](https://img.shields.io/badge/status-beta-yellow)

Seg-Studio は、ローカルマシン上で完結するオープンソースの画像セグメンテーション統合環境です。SAM を活用したスマートツールで画像をアノテーションし、PyTorch モデルをリアルタイム監視付きで学習し、CoreML や ONNX にエクスポートしてエッジデバイスにデプロイできます。クラウドアカウントは不要です。セマンティックセグメンテーションに加えて、既に描いたマスクをそのまま流用して物体を 1 個ずつ数えることもできます (追加のアノテーションは不要です)。

[クイックスタート](#クイックスタート) | [機能一覧](#主な機能) | [English](README.md)

</div>

<p align="center">
  <img src="docs/images/hero.gif" alt="SAMアノテーション、学習、評価、エクスポートをワンストップで" width="900" />
</p>

<p align="center"><sub>44 秒の一連の流れ: SAM クリックでアノテーション → auto-tune で学習 → ヒートマップと評価レポートを確認 → ONNX / CoreML へエクスポート。</sub></p>

---

## どこから読むか

試すまでは 3 ステップです。**ZIP をダウンロード → `install` と `start` を実行 →
`http://localhost:8002/ui/` を開く。** 実際のコマンドは下の
[クイックスタート](#クイックスタート) にあります。

| 目的 | 読むもの |
|---|---|
| **はじめて使う** | まず [クイックスタート](#クイックスタート) でインストールし、続けて [はじめてのファーストラン手順](docs/ja/first-run-manual.md) へ。アプリを開いてから最初の推論結果までを一本道でたどる最短手順です（約 10 分） |
| **インストールや起動でつまずいた** | [トラブルシューティング](docs/ja/troubleshooting.md) |
| **特定の機能を調べたい** | [ユーザーガイド](docs/ja/user-guide.md) — 各タブ・ツール・設定のリファレンス |
| **一通り詳しく学びたい** | [はじめてのハンドブック](docs/ja/handbook.md) — サンプルデータで同じ流れを最初から最後まで通す全 16 章のチュートリアル |
| **共有マシンや LAN で動かしたい** | [デプロイ手順](docs/ja/deployment.md) — トークンでのサインイン、リバースプロキシ、バックアップ |
| **開発に参加したい** | [CONTRIBUTING.md](CONTRIBUTING.md)（英語）と [開発者クイックスタート](docs/ja/dev-quickstart.md) |

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
      <b>Annotate</b> -- ブラシ、SAM クリックなど
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

要点だけ挙げると、ブラシ / SAM クリックで画像にラベルを付け、PyTorch の
セグメンテーションモデルを手元の PC で学習し（損失と F1 はリアルタイムに確認できます）、
推論結果を確認して ONNX または CoreML に書き出す、という流れです。既に描いたマスクを
流用した個数カウントにも対応しています。

DINOv2 特徴蒸留、Lovász-Softmax、後処理 CCA、OpenVINO INT8、Perlin CutPaste 合成、
グローバル学習キューといった応用機能は、いずれも任意です。インストール
手順をページ上部に残すため、以下の折りたたみに全件をそのまま収めています。1 枚もので
俯瞰したい場合は [機能カタログ](docs/ja/catalog.md) をご覧ください。

<details>
<summary><b>機能一覧（全件）</b> — クリックで展開</summary>

**アノテーション**
- ブラシ、矩形、消しゴム、バケツ塗り、スポット検出、クラックトレース
- Move ツール：マスク領域のドラッグによる位置調整
- Mark Clean：欠陥なし画像のフラグ付け
- SAM クリックセグメンテーション -- 5 モデル対応（MobileSAM、SAM2 Tiny/Small、TinySAM、EfficientSAM）
- AI サポート パネル：ツール呼び出しに対応した視覚モデルが、開いているプロジェクトを MCP ブリッジ経由でラベル付けし、その様子を画面で見られます（モデルサーバーの設定が必要。[MCP ブリッジ](#mcp-ブリッジ) を参照）
- レシピベースの自動ラベリングパイプライン

**学習**
- PyTorch 学習 + WebSocket によるリアルタイム監視（損失値、F1、mIoU）
- 学習モード選択：通常/ 個数カウント
- 個数カウント (インスタンスセグメンテーション)：接触した物体を 1 個ずつ分離。既存マスクからの合成データで学習するため追加アノテーション不要
- カウントのタイル学習・推論：小さな物体を撮影時の解像度のまま扱う (学習と推論は常に同一パッチサイズ)
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
- Results ビューでのカウント・面積計測 (連結成分)。カウントモデル学習時は個体単位のカウントも可能
- `POST /count` サービングエンドポイント：クラス別個数と個体ごとの矩形を返す

**インターフェース**
- 日英バイリンガル UI（ヘッダーからいつでも切り替え可能）

**インタラクティブなオンボーディング**
- UI 内ハンズオンチュートリアル：初級 / 中級 / エキスパートの 3 モード、
  スポットライトオーバーレイ、SVG アニメ、キーボードショートカットに対応。
  ヘッダーの ▶ ボタンからいつでも再生可能。
- 「次に見るべきタブ」のガイド点滅と、未確認結果の青色パルスで
  初見ユーザーを各ステップへ誘導。

</details>

---

## クイックスタート

### 0. インストール前チェック

- **ディスク容量**: CUDA 版 PyTorch の導入について、インストーラー自身が
  **約 5 GB の空き** を目安として案内します（ダウンロードだけで約 2.5 GB）。
  OpenVINO を追加する場合はさらに約 300 MB、加えて 5 つの SAM チェックポイントも
  保存されます。画像・学習ラン・エクスポートしたモデルは同じフォルダ内の
  `projects/` に置かれるため、余裕のある場所に展開してください。
- **管理者権限は通常不要です**: 書き込み先は展開したフォルダの中だけです
  （仮想環境 `.venv-windows` / `.venv-macos`、`models/`、`logs/`、`projects/`）。
  管理者権限が要るのは、Windows インストーラーに `winget` 経由で
  Python・Node.js・git を入れさせる場合と、`npm install` が権限エラーで
  失敗した場合だけです。
- **Windows ではパスを短くしてください**: パスが 260 文字を超えると仮想環境の
  作成に失敗します。インストーラーは `C:\seg-studio` のような短いパスへの移動を
  案内します。ウイルス対策ソフトが仮想環境の作成をブロックする場合は、その
  フォルダを除外設定に追加してください。
- **先に Python 3.11 以上を入れてください**: 見つからない場合、どちらの
  インストーラーも停止します。Windows インストーラーは 3.11 を最優先で探し、
  次に 3.12 → 3.13 の順に探します。依存関係の lockfile が 3.11 向けに
  コンパイルされており、pin された全パッケージにビルド済み wheel があると
  保証できるのが 3.11 だけだからです。
  Windows では最後の手段として `winget install Python.Python.3.11` を試み、
  導入後に PATH が通っていない場合はターミナルを閉じて再実行するよう案内します。
  macOS では事前にご用意ください（`brew install python@3.11`）。
- **ブラウザ UI をビルドするのは Node.js 18 以上です**: ビルド済みの UI は
  リポジトリに含まれていない（`dist/` は未コミット）ため、`npm` が無いと API は
  起動しても `http://localhost:8002/ui/` に表示するものがありません。`npm` が無い
  場合、Windows インストーラーは `winget` で Node.js 22 LTS の導入を試み、macOS
  では警告して UI ビルドをスキップします。API だけで良い場合は `--skip-ui` を
  指定してください。
- **git: macOS では必須、Windows では自動導入。** `install-macos.sh` は git が
  無いと致命的な前提不足として停止します（`brew install git`）。
  `install-windows.bat` は Python や Node.js と同様に `winget` で git を導入し、
  それが失敗した場合のみ SAM アシスト用ライブラリをスキップして続行します。
  Windows は SAM チェックポイントの取得に `curl`（Windows 10 1803 以降に標準搭載）
  も使います。
- **ダブルクリックだけで完結します。** `install-windows.bat` と
  `start-windows.bat` は展開したフォルダの直下にあり、ダブルクリックした場合は
  結果を読めるようコンソールを開いたままにします。スクリプトから呼び出して
  すぐ戻したい場合は `SEG_NO_PAUSE=1` を設定してください（CI 環境でも同様）。
- **NVIDIA ドライバ**: 本リポジトリは最低ドライババージョンを固定していないため、
  ここでも具体的な数値は示しません。インストーラーが実際に見ているのは
  `nvidia-smi` を実行できるかどうかです。実行できれば CUDA 12.8 wheel（Turing /
  RTX 20xx 以降、Blackwell を含む）、できなければ CPU 版が入ります。Maxwell /
  Pascal / Volta では `install-windows.bat cuda124` をお使いください。導入後は
  `python -c "import torch; print(torch.cuda.is_available())"` が `True` を返すことを
  ご確認ください。
- **SAM の重みが無くても、SAM クリックアシスト以外はすべて動作します**:
  ブラシ、矩形、消しゴム、スポット検出、クラック追跡、
  学習、評価、各種エクスポートはいずれも SAM の重みに依存しません。
  5 つのチェックポイントは Windows インストール時に、それ以外では初回使用時に
  ダウンロードされ、ソースに記録された SHA-256 で検証されます。
- **インターネットはインストール時に必要で、動作時には不要です**: インストール時に
  PyPI から PyTorch と Python 依存関係、GitHub から SAM アシスト用ライブラリ、
  SAM チェックポイント、UI の npm パッケージを取得します。それ以降、
  アノテーション・学習・評価・エクスポートはすべてローカルで動作します。例外は
  3 つで、未取得の SAM チェックポイント、DINOv2 の重み（学習で初めて必要に
  なったときにダウンロード。既定でオンの学習設定 Auto が DINOv2 の特徴を読み、
  DINOv2 蒸留と「DINOv2 特徴で分割」も使います）、そして任意の [AI サポート パネル](#mcp-ブリッジ)
  です。このパネルは、実行中にモデルが見る画像、指示、プロジェクト名、ツールの結果を
  設定したモデルサーバーに送ります（別のアドレスを指定するか `openai` バックエンドを
  選ばない限り、このマシン上のサーバーです）。ブリッジとこのパネルに必要な fastmcp は
  別にダウンロードして入れます。オフライン環境向けには、OS と Python のバージョンが
  同じ接続済みマシンで `python scripts/install.py --offline-pack <dir>` を実行して
  バンドルを作成してください。バンドルに入るのは、trainer API のロックファイル
  （`apps/trainer_api/requirements.txt`）が固定しているパッケージと SAM チェックポイントです。
  ONNX Runtime（serving API のロックファイルにあり、バンドル作成では読みません）、
  EfficientSAM と TinySAM のライブラリ、UI の npm パッケージ、fastmcp、DINOv2 の重み
  （`dinov2_vitb14_pretrain.pth`。torch の `hub/checkpoints` フォルダへ）は入らないので、そのマシンに必要なものは別に持ち込んで
  ください。また、ロックファイルが MobileSAM と SAM 2 のライブラリを git の URL で指定しているため、
  バンドルの `install_offline.py` は `--no-index` でもこの 2 つを GitHub から取得しに行きます。

### 1. コードを入手する（git 不要）

[Releases ページ](https://github.com/segmen-pixel/seg-studio/releases) から
最新の **Source code (zip)** をダウンロードし、好きな場所に展開してください
（リポジトリページ上部の緑の **Code → Download ZIP** ボタンでも同じものが取れます）。
git を使う場合: `git clone https://github.com/segmen-pixel/seg-studio.git`

Windows・macOS とも、ビルド済みパッケージは公開していません。Releases ページの
リリースはソースのアーカイブ（と SBOM）なので、下の「2.」は全ての環境で必要です。
パッケージ自体は
`python scripts/build_installer.py` でチェックアウトから作れます。
公開する際には Releases ページに `SHA256SUMS.txt` を添えて掲載します。

### 2. インストールして起動する

**Windows（NVIDIA GPU）:**
```bash
install-windows.bat
start-windows.bat
```

ターミナルは不要です。展開したフォルダ直下の `install-windows.bat` →
`start-windows.bat` をダブルクリックするだけでも動きます。インストーラーは GPU を自動判別し
（`cpu` / `cuda124` 指定で上書き可）、Python 3.11+ が見つからない場合は
インストール手順を案内します。起動スクリプトはサーバー準備完了後に
ブラウザで UI を自動的に開きます。

**macOS（Apple Silicon / Intel）:**
```bash
bash install-macos.sh
bash start-macos.sh
```

**Linux（NVIDIA GPU または CPU）:**
```bash
python3.11 -m venv .venv
.venv/bin/python scripts/install.py
bash scripts/start_local.sh
```

Python 3.11・Node.js 22・git は事前に入れておいてください（Ubuntu なら
`apt install python3.11 python3.11-venv git`、Node.js は nodejs.org か
ディストリビューションのパッケージから）。スクリプトは仮想環境の外では
動かず、この 3 つが揃っているかを何もダウンロードする前に確認します。
Windows インストーラーと同じ環境（CUDA 12.8 の PyTorch wheel、CUDA 版
ONNX Runtime、SAM 5 種すべて）を構築し、Ubuntu 22.04 + RTX 3080 Ti で
アノテーション・GPU 学習・CUDA 推論まで確認済みです。

停止は `stop-windows.bat` / `bash stop-macos.sh` /
`bash scripts/stop_local.sh` でできます。

Windows では `restart-windows.bat` がポート 8002 の trainer API だけを再起動します。
「次回サーバー起動時に反映されます」と表示される設定変更は、これで反映されます。
LAN トークンを再利用するのでブラウザのセッションはそのまま使え、新しいサーバーは
切り離して起動されるため、SSH 越しに再起動してもセッションの終了で止まりません。
学習の実行中・予約中は `--force` を付けない限り再起動を拒否し、`--dry-run` では
何をするかを表示するだけです。serving API と UI の開発サーバーには触れないので、
そちらは停止してから起動し直してください。

> Windows のインストーラーは必要なものを自分で用意します。Python・Node.js・git が
> 無ければ winget 経由で自動インストールするので、3 つとも入っていない PC でも
> ダブルクリック 1 回で動く状態になります。winget が無い環境（Windows 10 で
> App Installer が入っていない場合など）では Python の入手先を案内して停止します。
> Node.js と git は無くても停止はせず、Node.js が無ければ UI が再ビルドされず、
> git が無ければ SAM クリックセグメンテーションが使えません。
> macOS では git が必須で、`install-macos.sh` は git が無いと
> 停止します。
> macOS の Apple Silicon では MPS（Metal）が自動的に使われます。推論だけでなく
> 学習にも使えますが、個数カウントには NVIDIA GPU が必要です。
> どの環境で何が動くかは [プラットフォーム対応](#プラットフォーム対応) を
> 参照してください。

### 3. UI を開く

ブラウザで **http://localhost:8002/ui/** を開いてください。インストールはこれで完了です。

セグメンテーションが初めての方は、続けて
[はじめてのファーストラン手順](docs/ja/first-run-manual.md) へお進みください。
アプリを開いてから最初の推論結果まで、約 10 分でたどれます。

### 別の方法: Docker（docker compose）

コンテナで動かす場合の導線です。CPU のみで、セマンティック学習は動きますが大幅に遅くなります。
個数カウント（インスタンス）の学習には、ネイティブの NVIDIA 環境が必要です。

```bash
# Windows の場合は python3 ではなく python を使ってください
python3 -c "import secrets; print('SEG_API_TOKEN=' + secrets.token_urlsafe(24))" >> .env
docker compose up --build
```

ブラウザで **http://localhost:5173/** を開いてください。UI コンテナの nginx が
`/api`、`/v2`、`/ws` を trainer API にプロキシします。すべてのポートは
`127.0.0.1` のみに公開されます。`.env` の作成は任意ではなく必須です。各コンテナは
自身のネットワーク名前空間の中で `0.0.0.0` にバインドするため、トークンなしでは
trainer が起動を拒否します。また GPU を要求していないので、アノテーション・推論・UI は
動き、セマンティック学習は CPU で動きますが大幅に遅くなり、個数カウント（インスタンス）の
学習は使えません。詳細は [デプロイメント](docs/ja/deployment.md) を参照してください。

---

## ワークフロー

```
Projects  -->  Annotate  -->  Train  -->  Results  -->  Deploy
   |              |             |            |            |
 プロジェクト   SAM、ブラシ、   パラメータ    推論結果の    CoreML
 の作成または   スポット検出    設定と学習    評価と比較    または ONNX
 インポート     でラベル付け    の実行                     にエクスポート
```

---

## アーキテクチャ選択

| アーキテクチャ | パラメータ数 | モデルサイズ | 推論 (RTX 3090) | 特徴 |
|---|---:|---:|---:|---|
| **SimpleUNet** (bc=64) | 1.9 M | 7.3 MB | 2.9 ms · 339 img/s | 安定、高 F1、GroupNorm + SE attention |
| **STDC** (bc=32) | 2.9 M | 11.2 MB | 1.3 ms · 758 img/s | 軽量、最速推論 |

全モデルが GroupNorm、JIT トレーシング、設定可能な output stride に対応しています。
推論レイテンシは 256×256・バッチサイズ1 の単一画像での値です。GPU + CPU の
完全なベンチマークと再現方法は [BENCHMARKS.md](BENCHMARKS.md) を参照してください。

学習のデフォルトは **SimpleUNet** です — 小さく安定していて、最初の
一本に向いています。アーキテクチャはデータセットのプロファイルから
auto-config が提案するため、手で選ぶ必要はほとんどありません。精度や
推論速度を優先する場合は STDC を試し、手元のデータで両者を比べてください。

---

## MCP ブリッジ

Seg-Studio を MCP 対応ツールに接続して、プログラムからプロジェクトを検査・アノテーションできます。

**モデルは持ち込み。** ブリッジを動かすモデルは Seg-Studio に含まれていません。
どの MCP クライアントからでも、その裏にどのモデルがいても使え、アプリ内の
AI サポート パネル（下記）もそうしたクライアントの 1 つとして、設定したモデル
サーバーと話します。接続時に渡すプレイブックが手順、ツールが算数です。

### アプリの中で: AI サポート パネル

同じラベル付けのループは UI にも組み込まれています。モデルサーバーを設定すると、
**ラベル** タブのツール列の下端に **AI** ボタンが現れ、**AI サポート** パネルが開きます。
何をラベル付けするか書いて **実行** を押すと、ツール呼び出しに対応した視覚モデルが
ブリッジ経由でそのプロジェクトをラベル付けしていき、その様子を画面で見られます。
使い方は [ユーザーガイド](docs/ja/user-guide.md#ai-サポート-パネル) にあります。

モデルサーバーの場所は Seg-Studio を動かしているマシン側で決めるもので、画面からは
変えられません。trainer API の環境変数 `SEG_VLM_*` か、プロジェクトフォルダ
（`SEG_PROJECTS_DIR` の指定が無ければ `projects/`）の `runtime_settings.json` にある
`vlm` ブロックで指定します。両方で同じ項目を指定した場合は環境変数が優先されます。

| 環境変数 | `vlm` のキー | 指定するもの |
|---|---|---|
| `SEG_VLM_BACKEND` | `backend` | `ollama`・`openai`・`mlx`・`vllm`・`lmstudio`・`llamacpp` のいずれか。指定が無ければ `ollama` |
| `SEG_VLM_BASE_URL` | `base_url` | サーバーの `http://` または `https://` のアドレス。省略するとバックエンドごとの既定のアドレスで、`ollama`・`lmstudio`・`vllm`・`mlx`・`llamacpp` はこのマシン（`127.0.0.1` のそれぞれの標準ポート）、`openai` は `https://api.openai.com/v1` です。アドレスだけではバックエンドは決まりません。バックエンドの指定（`SEG_VLM_BACKEND` か `backend`）が無ければ `ollama` のままで、Ollama 独自の API で話しかけるため、OpenAI 互換サーバー（LM Studio・vLLM・MLX・llama.cpp）にはバックエンド名も指定してください |
| `SEG_VLM_MODEL` | `model` | 使うモデル。パネルにはサーバーが持っているモデルの一覧が出ます |
| `SEG_VLM_API_KEY_ENV` | `api_key_env` | キーが要るサーバー向けに、キーを入れた環境変数の**名前**。その変数自体は trainer API の環境から読むので、API を起動する環境に設定してください。値を変えたら API を次に起動したときに反映されます |

```json
{
  "vlm": {
    "backend": "lmstudio",
    "model": "<ツール呼び出しに対応した視覚モデル>"
  }
}
```

ブロックはファイルに既にある内容の横に追加してください。ファイルはリクエストのたびに
読み直されるので、ページを再読み込みすれば反映されます。環境変数は trainer API を
次に起動したときに反映されます。実行には MCP クライアントも必要で、これは Seg-Studio
には含まれていません（下記）。入っていない場合は **実行** を押すと、入れるためのコマンドが
そのまま表示されます。

実行中にモデルが見る画像（長辺を 1280 px に拡大・縮小した画像全体、その切り抜き、マスクを
重ねて描いた画像。人が描いたマスクを重ねたものも含みます）、指示、
プロジェクト名、ツールの結果は、設定したサーバーに送られます。キーの環境変数を指定していればキーも送られます。
信頼できるサーバーを使い、このマシンの外にあるサーバーには `https://` を使ってください。
実行はブリッジを `--policy write` で使い、開始したプロジェクトだけを扱い、人が描いた
マスクや人が Mark Clean した画像を置き換えることはありません。

### ブリッジを動かす

Seg-Studio の環境の Python で実行してください。ブリッジとレシピ系ツールが使う
httpx・NumPy・SciPy・Pillow・OpenCV はその Python にすでに入っているので、追加するのは
fastmcp だけで、このリリースで確認したバージョンに固定して入れます。fastmcp は任意で、
必要なのはブリッジ、下の例、AI サポート パネルからの実行だけです。その Python の
`-m pip` で入れてください。新しく開いたシェルで素の `pip` を使うと、たいてい別の
Python に入ります。fastmcp と mcp のライセンス、fastmcp が一緒に入れるパッケージに
ついての注記は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) の
*Optional, installed by the user (not bundled)* にあります。

```bash
# Seg-Studio フォルダで実行。<python> は Windows なら .venv-windows\Scripts\python.exe、
# macOS なら .venv-macos/bin/python、Linux なら .venv/bin/python
<python> -m pip install "fastmcp==4.0.1"
<python> scripts/mcp_server.py --api http://localhost:8002 --policy read
```

別のマシンの API を使う場合は `--api http://<host>:8002` を指定し、そのサーバーの
共有シークレットを `--token`（または `SEG_API_TOKEN`）で渡してください。LAN に
bind した Seg-Studio は認証なしのリクエストを拒否します。値は初回 LAN 起動時に
起動スクリプトが表示するものです。

プロジェクト / データセット / クラス / アノテーション / 学習 / 推論 / エクスポート /
レシピ / システムにわたる 75 ツールを公開します。ポリシーは `read`（既定）/ `write` /
`full` の 3 段階で、呼び出しは階層付きで stderr に記録され、ポリシー外のツールは
黙って劣化せず拒否します。破壊的なツール（`train_run_delete`、`clear_class`）には
`full` が必要です。

`--policy write` では参照だけでなくアノテーションもできます。`sam_segment` で
マスクを提案し（スコア最大の 1 枚ではなく 3 段階の粒度すべてを返します）、
`mask_put` で書き込み、`mark_clean` で「欠陥なし」を宣言し、`prelabel_run` で
学習済み run から未アノテーション画像に下書きを入れられます。個数カウントの
プレイブックもツールになっています。`teacher_band`・`accept_mask`・`accept_masks`・
`accept_points`・`zoom_plan`・`zoom_score`・`teacher_view`・`calibrate_sam`・
`write_kept` がその算数を受け持つので、視覚モデルは矩形を 1 つずつ出しながら手順を
進められます。`spot_detect` / `spot_write` は、人が塗った点状の欠陥に似たものを
見つけて書き込みます。

`--policy write` では、`mask_put`・`write_kept`・`spot_write`・`mark_clean`・
`recipe_apply` は人が描いたマスクも、人が Mark Clean した画像も置き換えません。
そのまま残し、残したことを返します。そうした画像に `overwrite=true` を指定するには
`--policy full` が必要です。置き換えたものの控えがどこにも残らないためです。
`prelabel_run` は `overwrite` を付けない限り未アノテーションの画像だけに下書きを入れ、
付けた場合はアノテーション済みの画像も置き換えます。その前に元のマスクのファイルを
`masks_replaced` に退避しますが、その控えは同じ画像に次の下書きを入れると置き換わります。
対象の画像（指定した画像、指定が無ければプロジェクトの全画像）に人が描いたマスクや
人が付けた Mark Clean が 1 枚でもあれば、これにも `--policy full` が必要で、`write` では
呼び出し全体が拒否されます。

ブリッジにはプレイブックも同梱されています。`scripts/mcp_playbook/` の 8 つの
Markdown ページを、接続時の instructions（概要ページ）・リソース
（`segstudio://playbook` が一覧）・4 つのプロンプト（`count_objects_from_teacher` /
`mark_single_object` / `find_defect_candidates` / `bootstrap_loop`、プロジェクトを
埋めてページを返す）の 3 経路でクライアントに渡します。どの画像にどのツールを使うかが
書いてあるので、「SAM の既定レベルが背景を返すことがある」といった知見をモデルが
手元のデータで再発見せずに済みます。`--playbook DIR`（または `SEG_MCP_PLAYBOOK`）で
自前のページを同梱分の上にページ単位で重ねられます。

ブラウザでプロジェクトを開いている人には、ブリッジの作業が見えます。ブリッジは
すべてのリクエストで自分の名前（`X-Seg-Agent: mcp/<policy>`）と使っているツール
を名乗り、アノテーション系のルートがそれを記録し、UI は書き込みが届いている間
ヘッダーに「MCP 自動ラベル中」のチップを出して、変更された画像を再読込します。
LIVE ON にすると、ラベル付け中の画像へ表示が自動で移動します。
ブラウザ自身の操作はこのフィード（`GET /api/v1/agent/activity`）に残らないので、
表示されるのはエージェントがしたことだけです。

アノテーションの前後の工程も操作できます。`project_create` と
`dataset_set_split` でプロジェクトの作成と train / test の振り分け、
`run_splits` で学習済み run が使った split の確認、`predict_status` で
推論済み画像の一覧取得（途中から再開できます）、`predict_operating_points`
で recall 優先・precision 優先それぞれの閾値と、その根拠となる検出数を
取得できます。

枚数をこなす用途では、SAM のループより `prelabel_run` を使ってください。SAM が
返すのは物体で、アノテーションは物体の中の欠陥領域であることが多いためです。
完璧なプロンプト（欠陥の中心のクリックや、正確な外接矩形）を与えても、SAM の
マスクは、数枚で学習したモデルほどには手描きの欠陥と一致しません。

### 付属の例

`scripts/examples/` に小さな例を 2 つ同梱しています。fastmcp を入れたうえで、同じ
Python で実行してください。`qwen_mcp_agent.py` はツール呼び出し対応のローカル視覚モデルに
コマンドラインからプロジェクトをラベル付けさせ、
`qwen_mcp_chat.py` は同じループにチャット画面を付けたもので、指示を打ってツール呼び出しを
眺められます。ブリッジにつないだホスト型のアシスタントでも同じことができます。

どのモデルが答えるかはフラグで決まります。`qwen_mcp_agent.py` の既定は `--backend ollama`
です。`qwen_mcp_chat.py` は `--backend` を付けないと画面で最後に選んだサーバーを使い、
Ollama になるのは初回の起動時だけです。`mlx`・`vllm`・`lmstudio`・`llamacpp`・`openai` は
どれも `/v1/chat/completions` を話すので、場所を `--base-url` で、キーが要る場合はキーを
入れた環境変数の名前を `--api-key-env` で渡すだけです。そのキーは `--backend` のサーバー
（`--backend` を付けずに起動したチャット画面では、起動時のサーバー）にだけ送られます。画像・ツール呼び出し・
ツール結果は、アプリと同じモデルサーバー層（`scripts/examples/vlm_backends.py` が
再エクスポート）がサーバーごとの形に整えるので、Mac の MLX サーバーやネットワーク上の
GPU マシンなど、モデルに合ったマシンで動かせます。ループ側はその違いを知りません。

```bash
<python> scripts/examples/qwen_mcp_chat.py --backend mlx \
    --base-url http://vlm-host:8080/v1 --model qwen2.5-vl-7b
```

チャット画面にはログインが無く、その裏のブリッジは書き込み権限で動くため、この
マシンの中だけで待ち受けます。`--host` は `127.0.0.1`・`localhost`・`::1` の
いずれかで、別のマシンからは `ssh -L 8765:127.0.0.1:8765 <host>` のような SSH
トンネル経由で開いてください。他のページからのリクエストは拒否します。モデル
サーバーのアドレス（`--base-url`）とキーを入れた環境変数（`--api-key-env`）は
コマンドラインでしか指定できません。画面からできるのは、既知の別のサーバーへの
切り替え（そのサーバーの既定アドレスへ、キー無しで接続）とモデルの選択だけで、
画像とキーの送り先は変えられません。画面は日本語で、英語にするには `--lang en` を
付けます。

---

## 動作環境

- **OS:** Windows 10 / 11（64-bit）、macOS 12 以上（Apple Silicon 推奨）、または
  Linux（Ubuntu 22.04 で確認済み。`scripts/install.py` を使います）
- **Python:** 3.11 以上（依存関係のロックファイルは 3.11 向けにコンパイル）
- **Node.js:** 18 以上（UI ビルド用）
- **ディスク:** CUDA 版の導入に約 5 GB の空き。詳細はクイックスタートの
  「0. インストール前チェック」を参照してください

### プラットフォーム対応

| | Windows + NVIDIA | Linux + NVIDIA | Apple Silicon（MPS） | CPU のみ |
|---|---|---|---|---|
| アノテーション / SAM アシスト | 対応 | 対応 | 対応（SAM 5 種のうち 4 種。TinySAM は macOS では導入されません） | 対応 |
| セマンティックセグメンテーションの学習 | 対応 | 対応 | 対応 | 対応（大幅に低速） |
| 個数カウント（インスタンス）の学習 | 対応 | 対応 | 非対応 | 非対応 |
| ONNX エクスポート | 対応 | 対応 | 対応 | 対応 |
| ONNX 推論 | 対応（CUDA プロバイダ） | 対応（CUDA プロバイダ） | 対応（CPU プロバイダ） | 対応（CPU プロバイダ） |
| Core ML エクスポート | `coremltools` を import できる場合のみ | `coremltools` を import できる場合のみ | 対応 | `coremltools` を import できる場合のみ |

- 学習のデバイス選択は既定が `auto` で、CUDA → MPS → CPU の順に選ばれます。
  NVIDIA でのセマンティックセグメンテーション学習は VRAM 4 GB 以上を推奨します。
- Windows インストーラーの既定は CUDA 12.8 PyTorch wheel（Turing / RTX 20xx
  以降、Blackwell RTX 5090 を含む）です。旧世代 GPU（Maxwell / Pascal / Volta）
  では `install-windows.bat cuda124` で CUDA 12.4 wheel を導入します。Linux の
  スクリプトも同じ CUDA 12.8 wheel を使い、切り替えオプションはありません
  （GPU の無いマシンでは CPU で動きます）。
- MPS では混合精度が無効になるため、同等の NVIDIA GPU より学習は遅くなります。
  MPS はシステム全体と共有するユニファイドメモリを使うため、メモリ不足になる
  場合は [トラブルシューティング](docs/ja/troubleshooting.md) を参照してください。
- **個数カウント（インスタンスセグメンテーション）の学習には NVIDIA GPU が
  必要です。** VRAM の自動調整は CUDA デバイスでのみ行われ、RTX 3090 での実測値
  では `small` モデルが既定のバッチ 8 で 8 GiB、バッチ 4 で 5.5 GiB、バッチ 2 で
  3.5 GiB を必要とします。3.5 GiB 未満は非対応です。
- Core ML エクスポートには `coremltools`（8.3.0 に固定）が必要で、import
  できない場合は HTTP 501 を返します。macOS インストーラーはこれを明示的に
  導入します。
- OpenVINO IR エクスポートは Windows インストーラーのオプション
  （`install-windows.bat --with-openvino`、約 300 MB）です。macOS
  インストーラーに同等のオプションはありません。
- ビルド済みパッケージはどちらのプラットフォームでも公開していません。リリースは
  ソースのアーカイブで、インストールスクリプトが手元で環境を構築します
  （クイックスタートの 2.）。macOS については意図的です。署名のない `.app` は
  利用者全員が Gatekeeper を手動で解除することになり、`install-macos.sh` を
  実行するより手間が増えるためです。

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

- 🚀 **[はじめてのファーストラン手順](docs/ja/first-run-manual.md)** — アプリを開いてから最初の推論結果までの最短手順（約 10 分）
- 📘 **[はじめてのハンドブック](docs/ja/handbook.md)** — サンプルデータで通す全 16 章のチュートリアル（画像→モデル→推論→SDK）
- 📗 **[機能カタログ](docs/ja/catalog.md)** — 機能一覧（1 枚もの、俯瞰用）

### リファレンス

- [ユーザーガイド](docs/ja/user-guide.md)
- [開発者クイックスタート](docs/ja/dev-quickstart.md)
- [デプロイ手順](docs/ja/deployment.md)
- [トラブルシューティング](docs/ja/troubleshooting.md)
- [インポート / エクスポート](docs/ja/import_export.md)
- [OpenVINO エクスポート](docs/openvino_export.md) — 書き出した IR を Intel CPU / iGPU / NPU で動かす（英語）
- [機能カタログのスライド](docs/ja/slides-catalog.md) — 紹介用デッキの Marp ソース
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
