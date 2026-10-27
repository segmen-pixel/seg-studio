// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";
import type { HealthInfo } from "../api";
import { useI18n } from "../i18n";

// changes: Japanese, changesEn: English. The About dialog picks by the active UI language.
const CHANGELOG = [
  {
    version: "0.9.13",
    date: "2026-10-27",
    changes: [
      "🆕 Seg-Studio をアプリとして開けるようになりました。Windows ではスタートメニューとデスクトップのショートカットから、macOS では ~/Applications の Seg-Studio.app から起動します。起動中はタスクバーとタスクトレイ（macOS では Dock とメニューバー）にアイコンが出て、「Seg-Studio を開く」「ログのフォルダを開く」「停止して終了」を選べます。",
      "🎯 アイコンを、ブラウザのタブと同じ藍色に白い S にそろえました。",
      "🛡 既知の脆弱性に対応するため、Pillow 13.0.0、aiohttp 3.14.5、certifi 2026.10.23、brotli 1.2.1、filelock 4.2.4、setuptools 84.2.1、fastmcp 4.1.2 に更新しました。インストーラーをもう一度実行してください。",
      "🛡 既知の脆弱性に対応するため、UI の開発用パッケージ vite を 6.4.5 に更新しました。",
      "🐛 1ビットや16ビットのマスクも、「縮小して軽量化」したエクスポートと縮小コピーで正しく書き出すようにしました。",
    ],
    changesEn: [
      "🆕 Seg-Studio opens as an app: from the Start menu and desktop shortcut on Windows, from Seg-Studio.app in ~/Applications on macOS. While it runs it sits in the taskbar and the tray (the Dock and the menu bar on macOS), with Open Seg-Studio, Open the log folder, and Stop and quit",
      "🎯 One icon everywhere: the browser tab's white S on indigo",
      "🛡 Dependencies: Pillow 13.0.0, aiohttp 3.14.5, certifi 2026.10.23, brotli 1.2.1, filelock 4.2.4, setuptools 84.2.1, fastmcp 4.1.2 (known vulnerabilities addressed). Run the installer again",
      "🛡 Dependencies (UI development packages): vite 6.4.5 (known vulnerabilities addressed)",
      "🐛 A 1-bit or 16-bit mask is written correctly in a shrunk export and a shrunk copy",
    ],
  },
  {
    version: "0.9.12",
    date: "2026-10-15",
    changes: [
      "🆕 AI サポート：追加のインストールや設定ファイルなしで始められます。MCP ライブラリはインストールに含まれ、何も設定していなければ、パネルが同じ PC の Ollama を探します。モデルサーバーの種類とモデルは、パネルで選べます。",
      "🎯 AI サポート：ラベルの無い画像に、1枚ずつ順にラベルを付けます。始める前に、残りの枚数と所要時間の見込みを示して、進めてよいかを確認します。人がラベルを付けた画像があれば、先にその画像で倍率を試します。1つずつ分かれた物体は SAM で、散らばった小さな点はスポット検出で扱います。",
      "🛡 モデルには最初に、データをほかへ渡さないこと、個人情報を読み上げないことを指示します（指示であって、通信を止める仕組みではありません）。MCP ブリッジは、起動時に更新の確認をしません。同じ PC のモデルサーバーと API には、プロキシを通さずに接続します。",
      "🆕 MCP ブリッジ：ツールが81個になりました。画像を部分ごとに見るツールを追加しました。--policy で許可したツールだけを一覧に出し、画像は MCP の画像として渡します。",
      "🐛 縦向きに撮った写真（EXIF の向き）が、横倒しで取り込まれる不具合を修正しました。以前に取り込んだ写真は変わりません。",
      "🐛 クラスを削除したあとに保存すると、ラベルの無い画素が背景として記録される不具合を修正しました。削除の確認では、そのクラスが塗られている画像の枚数を表示します。",
      "🎯 ひび割れ検出を作り直しました。明るい線や太い線も選べます。線の上以外をクリックしたときは何も選ばず、その理由を表示します。",
      "🐛 macOS のインストーラーが途中で止まる不具合を修正しました。チュートリアルのサンプルは、欠陥の無い16枚に OK（欠陥なし）マークが付いた状態で取り込まれます。",
      "🎯 AI サポート：物体を扱うプロジェクトで、人がラベルを付けた画像がまだ無いときは、最初に1枚を手でラベル付けするよう頼みます。SAM を実行できないときは、理由を示して止まります。",
      "🐛 サーバーのエラーを、画面と同じ言語で表示するようにしました。要求が断られたときに、理由にかかわらず「必須パラメータが不足しています」と表示される不具合を修正しました。",
      "🐛 NVIDIA GPU のある Windows にインストールすると、numpy や setuptools などがロックファイルと異なるバージョンになる不具合を修正しました。",
      "🛡 UI の開発用パッケージ @typescript-eslint を 8.70.1 に更新しました。npm audit が braces について出していた指摘は出なくなります。",
      "🎯 画面、AI サポートのメッセージ、レポート、文書の日本語を、自然な日本語に書き直しました。同じものは同じ名前で呼びます（しきい値、確信度など）。",
      "🛡 LAN モードで、開発用の画面サーバー（5173）をネットワークに公開しないようにしました。Docker 版の画面から保存できない不具合や、設定ファイルが壊れていると上書きしてトークンが消える不具合なども直しました。",
    ],
    changesEn: [
      "🆕 AI support starts without an extra install step or a settings file: the MCP library is part of the install, and with nothing configured the panel looks for Ollama on the same PC. The kind of model server and the model are chosen in the panel",
      "🎯 AI support labels the unlabelled images one by one. Before it starts it asks whether to go ahead, with the number of images left and a time estimate, and where a person has labelled images it first tries the zoom on them. Distinct objects go to SAM, scatterings of small marks to spot detection",
      "🛡 Before anything else the model is told not to pass the data on and not to read out personal information (instructions, not something that blocks traffic). The MCP bridge makes no update check when it starts, and a model server and API on the same PC are reached without a proxy",
      "🆕 MCP bridge: 81 tools. Tools for looking at an image part by part are added; only the tools --policy allows are listed, and pictures are handed over as MCP images",
      "🐛 Fixed photos taken upright (EXIF orientation) being imported lying on their side. Photos imported earlier are not changed",
      "🐛 Fixed a save after deleting a class recording unlabelled pixels as background. The delete confirmation shows how many images the class is painted in",
      "🎯 Crack trace rebuilt: light and wide lines can be selected, and a click that is not on a line is not selected, with the reason shown",
      "🐛 Fixed the macOS installer stopping part-way. The tutorial sample's 16 defect-free images arrive with their OK (no defects) mark",
      "🎯 AI support: on a project of objects nobody has labelled yet, the run first asks for one image labelled by hand. When SAM cannot run, it stops and says why",
      "🐛 Server errors are shown in the screen's language. Fixed a refusal being shown as \"A required parameter is missing\" whatever its reason",
      "🐛 Fixed an install on Windows with an NVIDIA GPU leaving numpy, setuptools and others at versions other than the lockfile's",
      "🛡 Dependencies (UI development packages): @typescript-eslint 8.70.1. The advisory npm audit reported for braces no longer appears",
      "🎯 The Japanese of the screen, the AI support messages, the reports and the documentation is rewritten to read as Japanese, with one name for one thing",
      "🛡 LAN mode no longer exposes the development screen server (5173) to the network. Also fixed: the Docker screen could not save, and an unreadable settings file was written over, losing the token",
    ],
  },
  {
    version: "0.9.11",
    date: "2026-10-13",
    changes: [
      "🛡 既知の脆弱性に対応するため、urllib3 2.8.0、multidict 6.9.1、fsspec 2026.9.0 に更新しました。",
      "🛡 既知の脆弱性に対応するため、UI の開発用パッケージを source-map-js 1.2.2、brace-expansion 5.0.12 に更新しました。",
      "🧹 アプリから選べなかった蒸留のオプションを削除しました（distill_mode は off / feature のみ）。アプリの使い方と DINOv2 蒸留の動作は変わりません。",
    ],
    changesEn: [
      "🛡 Dependencies: urllib3 2.8.0, multidict 6.9.1, fsspec 2026.9.0 (known vulnerabilities addressed)",
      "🛡 Dependencies (UI development packages): source-map-js 1.2.2, brace-expansion 5.0.12 (known vulnerabilities addressed)",
      "🧹 Distillation options that could not be selected in the app are removed (distill_mode takes off / feature). Nothing changes in how the app is used, and DINOv2 distillation behaves as before",
    ],
  },
  {
    version: "0.9.10",
    date: "2026-09-29",
    changes: [
      "🛡 ライセンス：DINOv2 のモデル定義（Apache-2.0）を同梱し、そこからモデルを組み立てるようにしました。重みと計算結果は従来と同じです。",
      "🛡 ライセンス：どの方法でインストールしても、torchmetrics の非商用モジュールを置き換え、GPL のみ、またはライセンスが明確でない pyphen の辞書を削除します。起動時にも、学習 API が改めて確認します。",
      "📦 cuda-bindings を 13.4.3 に更新しました（Linux のみ、Apache-2.0）。ライセンスの検査では、独自ライセンスは確認済みのものだけを通します。git 依存の検査は MobileSAM、SAM 2、EfficientSAM、TinySAM を対象にします。",
    ],
    changesEn: [
      "🛡 Licensing: DINOv2 is built from its Apache-2.0 model definition, which ships with Seg-Studio; same weights, same results",
      "🛡 Licensing: every install route replaces torchmetrics' non-commercial module and deletes pyphen dictionaries that are GPL-only or not clearly licensed; the trainer API checks again at startup",
      "📦 cuda-bindings 13.4.3 (Linux only, Apache-2.0). Licence gates: a proprietary licence needs a reviewed entry; the git-dependency gate covers MobileSAM, SAM 2, EfficientSAM and TinySAM",
    ],
  },
  {
    version: "0.9.9",
    date: "2026-09-29",
    changes: [
      "🆕 AI サポート（Assist）パネルを追加しました。開いているプロジェクトに、ビジョンモデルがラベルを付けます。モデルサーバーを設定すると、ラベルタブのツール列に AI ボタンが表示されます。",
      "🆕 MCP ブリッジに、書き込み用のツールとプレイブックを追加しました。人が描いたマスクや、人が OK（欠陥なし）マークを付けた画像を置き換えるには、--policy full が必要です。",
      "🎯 スポット検出をサーバー側で実行するようにし、しきい値を1〜60の範囲の1本のスケールにまとめました。SAM のクリックは対象を指定するものだけになりました（除外クリックは廃止）。",
      "🧹 マジックワンドとスーパーピクセルをパレットから削除しました。自動ラベルのボタンは「学習済みモデルから下書き」に置き換えました。",
      "🐛 クラスを削除・統合すると、大きな画像（タイルで保存）の OK マークと要確認マークが消える不具合を修正しました。",
      "🎯 個数検査：学習時に対象の並び方（バラバラ／整列）を指定できるようにしました。大きな写真を縮小せずに学習し、タイルの境界をまたぐ対象も1個として数えます。",
      "🛡 セキュリティの問題を修正しました（サンプルのチャットページはループバックだけで待ち受ける、MCP ブリッジが ID を検査する、エラー応答にサーバーのファイルパスを含めない、など）。",
    ],
    changesEn: [
      "🆕 AI support (Assist) panel — a vision model labels the project you have open. Once a model server is configured, an AI button appears in the Annotate tool column",
      "🆕 MCP bridge: write tools and a playbook. Replacing a mask a person drew, or an image they marked OK (no defects), needs --policy full",
      "🎯 Spot detection runs on the server with one Threshold scale (1–60); SAM takes positive clicks only (exclude-click removed)",
      "🧹 Magic wand and superpixel removed from the palette; the auto-label button is replaced by Draft from a trained model",
      "🐛 Deleting or merging a class no longer drops the OK marks and Review flags of large (tiled) images",
      "🎯 Count inspection: training takes how objects lie (Scattered / Aligned) and no longer squashes large photos; an object across tile seams is counted once",
      "🛡 Security fixes (the example chat page listens on loopback only, the MCP bridge checks ids, error responses no longer carry server file paths, and more)",
    ],
  },
  {
    version: "0.9.8",
    date: "2026-07-27",
    changes: [
      "🆕 インスタンス（個数カウント）の学習モードを追加しました。接したり重なったりしている対象の個数を数えます。既存のセマンティックマスクから学習データを合成するので、アノテーションを追加する必要はありません。",
      "🎯 個数は、検証データで較正したしきい値と、マスク IoU 0.7 での重複除去で数えます。1個の対象を二重に検出する DETR 系の癖を抑えます。",
      "🎯 1つのモデルで複数のクラスを同時に数えます。クラスの対応は、エクスポートした推論契約にも引き継がれます。",
      "🖥 モデルサイズは Small（既定）、Medium、Large から選べます。検出した VRAM に合わせてバッチサイズを自動で調整します（実効バッチサイズは変わりません）。",
      "📤 ONNX エクスポートと、serving_api の /count エンドポイントに対応しました。",
      "🎨 結果ビューに、番号付きのインスタンス表示（Okabe-Ito パレット）と、クラス別の個数のチップを追加しました。",
      "🛡 rfdetr は、Apache-2.0 の本体と Seg の重みだけを使います。PML-1.0 の「plus」拡張と、検出用の XL/2XL モデルは使いません。",
      "🧹 バージョン表記のずれを解消しました（バックエンドの一部が、1つ前のリリースの値を返していました）。",
    ],
    changesEn: [
      "🆕 Instance (count) training mode added — counts touching / overlapping objects. No extra annotation work: training data is synthesized from the existing semantic masks",
      "🎯 Counting uses a threshold calibrated on validation data plus mask-IoU 0.7 dedup, suppressing the DETR-family habit of detecting one object twice",
      "🎯 One model counts several classes at once; the class mapping is carried into the exported inference contract",
      "🖥 Model sizes Small (default) / Medium / Large. Batch size auto-adjusts to the detected VRAM (effective batch is preserved)",
      "📤 ONNX export and the serving_api /count endpoint are supported",
      "🎨 Results view: numbered instance overlay (colorblind-safe palette) and per-class count chips",
      "🛡 rfdetr: only the Apache-2.0 core and Seg weights are used. The PML-1.0 \"plus\" extensions and the detection XL/2XL models are excluded",
      "🧹 Version string drift fixed (part of the backend reported a value one release old)",
    ],
  },
  {
    version: "0.9.7",
    date: "2026-07-17",
    changes: [
      "🧹 異常検知モードを完全に削除しました（統計的な z-score だけの仕組みで、名ばかりの機能になっていたため）。異常検知は Cls-Studio に移ります。",
      "☁ この配布物からクラウド学習のオプションを削除しました（ローカルで完結する範囲に絞りました）。",
      "🧹 自動ドナーによる warm-start を削除しました（明示的に指定する転移学習とモデル検索は変わりません）。",
      "🐛 auto_mode=\"off\" を指定したときに、arch が知らないうちに書き換わる不具合を修正しました（リクエストの値をそのまま使います）。",
    ],
    changesEn: [
      "🧹 Anomaly detection mode fully removed (the single statistical z-score path was nominal only; anomaly detection moves to Cls-Studio)",
      "☁ Cloud training option removed from this distribution (scope consolidated to local-only)",
      "🧹 Automatic donor warm-start removed (explicit transfer learning / model search unchanged)",
      "🐛 Fixed arch being silently rewritten when auto_mode=\"off\" was given (the request value is now honored verbatim)",
    ],
  },
  {
    version: "0.9.6",
    date: "2026-07-08",
    changes: [
      "🎯 wave1-6 の EDA にもとづいて、学習の既定値を見直しました（base_channels=128、fg_patch_prob=0.7 など）。",
      "📊 auto-config：ML 予測器が使えずに z-score に切り替えたとき、その理由を学習ログに書くようにしました。",
      "🎯 best_model_v6 のバンドルを、26個の特徴量をすべて埋めて再学習しました（LOPO top-5 hit 5.4%→10.8%）。",
    ],
    changesEn: [
      "🎯 Training defaults refreshed from the wave1-6 EDA (base_channels=128 / fg_patch_prob=0.7 and others)",
      "📊 auto-config: the reason for falling back to z-score when the ML predictor is unavailable is now stated in the training log",
      "🎯 best_model_v6 bundle retrained with all 26 features filled (LOPO top-5 hit 5.4%→10.8%)",
    ],
  },
  {
    version: "0.9.5",
    date: "2026-05-08",
    changes: [
      "🛡 ライセンスの整理：蒸留の teacher は DINOv2 と SAM2（Apache-2.0）です。",
      "🛡 DINOv2 の重み（Apache-2.0）を同梱しました。",
      "📜 LGPL の全文（libvips、Cairo、Pango、FFmpeg、pyphen）を licenses/third_party/lgpl/ に同梱しました。",
      "📜 PyTorch の CUDA wheel が取り込む NVIDIA のライブラリ（cuDNN、cuBLAS など）のライセンス文を、インストーラーにも含めました。",
      "🧹 蒸留の API を簡素化しました。precompute と ensemble のエンドポイントを削除し、online teacher だけに対応します。",
      "🧹 NOTICE、THIRD_PARTY_NOTICES.md、MODEL_WEIGHTS.md を整備し、商標の免責事項を追加しました。",
    ],
    changesEn: [
      "🛡 License cleanup: the distillation teachers are DINOv2 / SAM2 (Apache-2.0)",
      "🛡 DINOv2 is bundled as its Apache-2.0 weights",
      "📜 Full LGPL texts (libvips / Cairo / Pango / FFmpeg / pyphen) included under licenses/third_party/lgpl/",
      "📜 License texts of the NVIDIA libs (cuDNN / cuBLAS etc.) pulled in by the PyTorch CUDA wheel propagated to the installer",
      "🧹 Distillation API slimmed down: precompute / ensemble endpoints removed, online teacher only",
      "🧹 NOTICE / THIRD_PARTY_NOTICES.md / MODEL_WEIGHTS.md updated, trademark disclaimer added",
    ],
  },
  {
    version: "0.9.4",
    date: "2026-04-17",
    changes: [
      "🎓 ハンズオンチュートリアルを追加しました。初級、中級、エキスパートの3つのモードがあり、スポットライトでの案内と12種類の SVG アニメーションを備えています。",
      "🎯 学習が終わったランの結果ボタンが青く点滅します（説明モードがオンのとき、まだ見ていないランだけ）。",
      "🎯 説明モードでは、次に見るタブ（プロジェクト、ラベル、学習）が点滅して案内します。",
      "⌨ チュートリアルをキーボードで操作できるようにしました。Enter、→、←、Esc が使え、モードは 1、2、3 キーで選べます。",
      "🎨 ヘッダーに、チュートリアルを再生する ▶︎ ボタンを追加しました。",
    ],
    changesEn: [
      "🎓 Hands-on tutorial: Beginner / Intermediate / Expert modes, spotlight guide + 12 SVG animations",
      "🎯 The results button of a finished run blinks blue (description mode on, unviewed runs only)",
      "🎯 In description mode, the tab to look at next (Projects/Annotate/Training) blinks as a guide",
      "⌨ Keyboard control inside the tutorial: Enter/→/←/Esc, mode selection 1/2/3",
      "🎨 ▶︎ tutorial replay button added to the header",
    ],
  },
  {
    version: "0.9.3",
    date: "2026-04-10",
    changes: [
      "🔍 異常検知：auto-config、パッチ単位の学習、ステップごとの進捗表示に対応しました。",
      "🎨 結果タブ：GT と推論のマスクを分けて表示し、パターンのメニュー（斜線、ドット、格子）を追加しました。",
      "🎨 説明モード：ツールチップを JS で作り直しました（レイアウトが崩れる問題を解消）。",
      "🖱 ブラシサイズを改善し、スライダーを修正しました（両端まで動かせます）。",
      "📁 フォルダのインポートに対応し、入れ子のフォルダ名を見出しとして表示します。",
      "📁 複数のプロジェクトをまとめてエクスポートできるようにしました。",
      "⚡ スライディングウィンドウ推論で max による集約を使うようにし、キャッシュの無効化を修正しました。",
      "⚡ 学習キュー：予約したランが自動で始まり、起動時にキューを復元します。",
      "🖥 ライブ検査：オーバーレイのラベルとフィルタを追加し、上部のバーを刷新しました。",
      "🧹 OK マーク（欠陥の無い画像の印）を追加しました。",
      "🌐 結果タブの UI をすべて日本語にしました。",
      "🐛 GPU が2枚の環境での判定を修正しました（すべてのデバイスが使用中のときだけ busy と判定します）。",
      "🐛 領域のラベルとクラス一覧に、確信度のしきい値が反映されるようにしました。",
      "🐛 予約したランが続けて始まるときのクラッシュを修正しました（GPU のクールダウンと再試行）。",
    ],
    changesEn: [
      "🔍 Anomaly detection: auto-config, patch training, step progress display",
      "🎨 Results tab: separate GT / prediction mask display, pattern menu (hatch / dots / grid)",
      "🎨 Description mode: tooltips reimplemented in JS (fixes layout breakage)",
      "🖱 Brush size improvements, slider UI fix (reaches both ends)",
      "📁 Folder import, nested folder name shown as a header",
      "📁 Batch project export",
      "⚡ Sliding-window inference uses max aggregation; cache invalidation fix",
      "⚡ Training queue: scheduled runs start automatically, queue restored at startup",
      "🖥 Live inspect: overlay labels, filters, topbar refresh",
      "🧹 Mark Clean (flag images with no defects)",
      "🌐 Results tab UI fully translated to Japanese",
      "🐛 Dual-GPU fix (busy only when all devices are in use)",
      "🐛 Confidence threshold applied to region labels / class list",
      "🐛 Scheduled-run chain crash fixed (GPU cooldown + retry)",
    ],
  },
  {
    version: "0.9.2",
    date: "2026-04-04",
    changes: [
      "🐛 ファイル名に # を含む画像が表示されない不具合を修正しました（URL を encodeURIComponent で処理）。",
      "🐛 起動直後に API が 404 を連発する問題を修正しました（ルーターの登録が終わるまでは 503 を返します）。",
      "🐛 skimage の .pyi スタブが欠けていた問題を修正しました（スーパーピクセル機能が復旧）。",
      "🐛 ORT の CUDA EP が使えない問題を修正しました（onnxruntime の CPU 版で上書きされなくなりました）。",
      "🐛 combo_library.json のパスが誤っていた問題を修正しました（パッケージ内に同梱）。",
      "🐛 DINOv2 のチェックポイントが壊れる問題に対処しました（インストーラーに同梱）。",
      "🖱 画像リスト：Shift/Ctrl+クリックでアクティブな画像が切り替わらないようにしました。",
      "🖱 画像リスト：選択中の画像を、枠線と背景色でよりはっきり示すようにしました。",
      "🖱 画像リスト：Shift+矢印キーでの範囲選択を修正しました（戻る方向に動かすと選択を解除します）。",
      "🖱 画像リスト：Delete キーで選択中の画像を削除できるようにしました。",
      "📁 名前を変えてインポートする機能を追加しました（prefix/suffix の付与と、ドラッグ＆ドロップでの並べ替え）。",
      "🧹 保存ボタンを削除しました（自動保存があるため不要）。",
      "🎨 favicon.ico を追加しました。",
    ],
    changesEn: [
      "🐛 Images whose file name contains # now display (URL encodeURIComponent)",
      "🐛 API 404 burst at startup fixed (returns 503 until routers are registered)",
      "🐛 Missing skimage .pyi stub fixed (superpixel feature restored)",
      "🐛 ORT CUDA EP unavailable fixed (onnxruntime CPU build no longer overwrites it)",
      "🐛 Wrong combo_library.json path fixed (bundled inside the package)",
      "🐛 Corrupt DINOv2 checkpoint workaround (bundled with the installer)",
      "🖱 Image list: Shift/Ctrl+click no longer switches the active image",
      "🖱 Image list: stronger selection highlight (border + background)",
      "🖱 Image list: Shift+arrow range selection fix (deselects when moving back)",
      "🖱 Image list: Delete key removes the selected images",
      "📁 Rename import (prefix/suffix + drag-and-drop reordering)",
      "🧹 Save button removed (autosave makes it unnecessary)",
      "🎨 favicon.ico added",
    ],
  },
  {
    version: "0.9.1",
    date: "2026-03-23",
    changes: [
      "☁ GPU の天気アイコン、残高、本日の利用額を表示するようにしました。",
      "☁ 推論結果を自動でダウンロードするようにしました（predictions.zip）。",
      "☁ クラウドのランに、停止ボタンと切断ボタンを追加しました。",
      "🔍 異常検知モジュールを追加しました（OK 画像だけで異常な箇所を検出します）。",
      "🎯 学習モードを選ぶ画面を追加しました（通常、異常検知、クイック、転移学習）。",
      "📊 ピクセル単位の前景（FG）の確信度ヒストグラムを追加しました。",
      "⚡ 画像を並列でインポートするようにしました（10枚ずつのバッチと、マルチスレッドでの変換）。",
      "📁 同じ画像をインポートしたときに、スキップ、上書き、両方残すから選べるダイアログを追加しました。",
      "🌐 日本語と英語の翻訳キーを追加しました（クラウド関連の文字列40件以上）。",
      "🧹 使われていないコードを整理しました（print を logger に置き換え、removed コメントを削除）。",
      "🐛 エクスポートで日本語のファイル名が正しく扱われない問題を修正しました。",
      "🐛 sample_weights の不具合を修正しました（蒸留のフォールバック時）。",
      "🐛 onnx の遅延インポートでルーターの登録に失敗する問題を修正しました。",
    ],
    changesEn: [
      "☁ GPU weather icon + balance + today's spend display",
      "☁ Automatic download of inference results (predictions.zip)",
      "☁ Stop and disconnect buttons for cloud runs",
      "🔍 Anomaly detection module (finds anomalous regions from OK images only)",
      "🎯 Training mode selection UI (normal / anomaly / quick / transfer learning)",
      "📊 Pixel-level FG confidence histogram",
      "⚡ Parallel image import (batches of 10 + multithreaded conversion)",
      "📁 Dialog for duplicate image imports (skip / overwrite / keep both)",
      "🌐 Japanese/English translation keys added (40+ cloud-related strings)",
      "🧹 Dead code cleanup (print→logger, removed comments deleted)",
      "🐛 Japanese file names in export fixed",
      "🐛 sample_weights bug fixed (distill fallback)",
      "🐛 onnx lazy import fixed (router registration failure)",
    ],
  },
  {
    version: "0.9.0-beta",
    date: "2026-03-20",
    changes: [
      "マスクの保存先を Zarr のストアに移行しました。",
      "GPU リソースのロックを追加しました（学習中と推論中は SAM を無効にします）。",
      "学習のフェーズを表示するようにしました。",
      "Windows での DataLoader の問題を修正しました（num_workers=0 に固定）。",
    ],
    changesEn: [
      "Zarr mask store migration",
      "GPU resource lock (SAM disabled during training/inference)",
      "Training phase status display",
      "Windows DataLoader fix (num_workers=0 fixed)",
    ],
  },
];

const LICENSE_ENTRIES = [
  "FastAPI — MIT",
  "Uvicorn — BSD-3-Clause",
  "SQLModel — MIT",
  "Pydantic — MIT",
  "python-multipart — Apache-2.0",
  "ONNX — Apache-2.0",
  "PyTorch — BSD-style",
  "torchvision — BSD-3-Clause",
  "NumPy — Modified BSD",
  "Pillow — MIT-CMU",
  "httpx — BSD-3-Clause",
  "OpenCV — Apache-2.0",
  "opencv-python-headless (wheel) — MIT; FFmpeg LGPLv2.1",
  "scikit-learn — BSD-3-Clause",
  "scikit-image — BSD-3-Clause",
  "onnxruntime — MIT",
  "transformers — Apache-2.0",
  "MobileSAM — Apache-2.0",
  "SAM 2 — Apache-2.0",
  "TinySAM — Apache-2.0",
  "EfficientSAM — Apache-2.0",
  "React / React DOM — MIT",
  "Zustand — MIT",
  "Immer — MIT",
  "Vite — MIT",
];

interface AboutDialogProps {
  open: boolean;
  onClose: () => void;
  healthInfo: HealthInfo | null;
}

const AboutDialog: React.FC<AboutDialogProps> = ({ open, onClose, healthInfo }) => {
  const { lang, t } = useI18n();
  if (!open) return null;
  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-panel about-panel" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <h2>{t("about.title")}</h2>
          <button className="ghost" onClick={onClose} data-desc={t("common.close")} data-desc-pos="bottom">×</button>
        </div>
        <section className="about-info">
          <div className="about-title">Seg-Studio</div>
          <div className="about-meta">
            <span>v{__APP_VERSION__}{__APP_CHANNEL__ ? " " + __APP_CHANNEL__ : ""}</span>
            <span className="muted">Build: {__BUILD_DATE__}</span>
          </div>
          <div className="muted" style={{ fontSize: 12 }}>Copyright 2026 Segmen-Pixel and Seg-Studio contributors. Apache License 2.0.</div>
        </section>
        {healthInfo && (
          <section>
            <h3>{t("about.system")}</h3>
            <div className="about-system-grid">
              {healthInfo.disk && (
                <div className="about-system-item">
                  <span className="about-system-label">Disk</span>
                  <span>{healthInfo.disk.free_gb} GB free / {healthInfo.disk.total_gb} GB ({healthInfo.disk.used_pct}% used)</span>
                </div>
              )}
              {healthInfo.ram && (
                <div className="about-system-item">
                  <span className="about-system-label">RAM</span>
                  <span>{healthInfo.ram.available_gb} GB free / {healthInfo.ram.total_gb} GB ({healthInfo.ram.used_pct}% used)</span>
                </div>
              )}
              {healthInfo.gpu && (
                <div className="about-system-item">
                  <span className="about-system-label">GPU</span>
                  <span>{healthInfo.gpu.name} — {healthInfo.gpu.vram_total_mb} MB VRAM</span>
                </div>
              )}
            </div>
          </section>
        )}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          <section>
            <h3>{t("about.licenses")}</h3>
            <div className="about-license-scroll">
              <ul className="license-list">
                {LICENSE_ENTRIES.map((e) => <li key={e}>{e}</li>)}
              </ul>
            </div>
          </section>
          <section>
            <h3>{t("about.changelog")}</h3>
            <div className="about-license-scroll">
              {CHANGELOG.map((entry) => (
                <div key={entry.version} style={{ marginBottom: 12 }}>
                  <div style={{ fontWeight: 600, fontSize: 13 }}>
                    v{entry.version} <span className="muted" style={{ fontWeight: 400 }}>({entry.date})</span>
                  </div>
                  <ul style={{ margin: "4px 0 0", paddingLeft: 20, fontSize: 12, lineHeight: 1.6 }}>
                    {(lang === "ja" ? entry.changes : entry.changesEn).map((c, i) => <li key={i}>{c}</li>)}
                  </ul>
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};

export default React.memo(AboutDialog);
