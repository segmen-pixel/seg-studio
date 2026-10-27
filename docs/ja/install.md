# インストールの詳細

[README](../../README.ja.md) の「試してみる」の手順だけではうまくいかなかったときにお読みください。インストール前に確認すること、OS ごとの手順、Docker、オフライン環境、どの環境で何が動くかをまとめています。

## 0. インストール前チェック

- **ディスク容量**：CUDA 版 PyTorch を入れる段階について、インストーラーは**約 5 GB の空き**を目安として案内します（ダウンロードだけで約 2.5 GB あります）。OpenVINO を追加する場合はさらに約 300 MB 必要で、ほかに SAM のチェックポイント5つも保存されます。画像・学習ラン・書き出したモデルは同じフォルダ内の `projects/` にたまっていくため、空き容量に余裕のある場所に展開してください。
- **通常、管理者権限は要りません。** 書き込み先は、展開したフォルダの中だけです（仮想環境の `.venv-windows` / `.venv-macos`、`models/`、`logs/`、`projects/`）。管理者権限が必要になるのは、足りない Python・Node.js・git を Windows インストーラーに `winget` で入れさせる場合と、`npm install` が権限エラーで止まった場合だけです。
- **Windows ではパスを短くしてください。** パスが260文字の上限を超えると、仮想環境の作成に失敗します。インストーラーは、`C:\seg-studio` のような短いパスへフォルダを移すよう案内します。ウイルス対策ソフトが仮想環境の作成をブロックする場合は、そのフォルダを除外設定に追加してください。
- **Python 3.11以上が必要です。** Windows インストーラーは、Python が見つからなければ自動で導入し、導入できない場合や、見つかった Python が 3.11 より古い場合は停止します。macOS インストーラーは、Python が見つからなければ停止します。Windows インストーラーは 3.11 を最優先で探し、次に 3.12、3.13 の順に探します。依存関係のロックファイルは 3.11 向けにコンパイルされており、固定したすべてのパッケージにビルド済みの wheel があると保証できるのは 3.11 だけだからです。Windows では最後の手段として `winget install Python.Python.3.11` を試します。導入した Python にまだ PATH が通っていない場合は、ターミナルを閉じて再実行するよう案内します。macOS では事前に Python を入れておき（`brew install python@3.11`）、`python3 --version` で 3.11以上と表示されることを確認してください。macOS インストーラーは `python3` コマンドを実行し、そのバージョンが 3.11 より古いと停止します。
- **ブラウザ UI のビルドには Node.js 18以上が必要です。** ビルド済みの UI はリポジトリに含まれていません（`dist/` はコミットしていません）。そのため `npm` が無いと、API は起動しても `http://localhost:8002/ui/` に表示するものがありません。`npm` が無い場合、Windows インストーラーは `winget` で Node.js 22 LTS の導入を試みます。macOS では警告を出して UI のビルドを飛ばします。API だけを使う場合は `--skip-ui` を指定してください。
- **git は macOS では必須で、Windows では自動で導入されます。** `install-macos.sh` は git が無いと、必須の前提条件が欠けているとして停止します（`brew install git` で入れてください）。`install-windows.bat` は Python や Node.js と同じく `winget` で git を導入し、それに失敗した場合は SAM アシスト用ライブラリだけを飛ばしてインストールを続けます。Windows では、SAM チェックポイントのダウンロードに `curl`（Windows 10 1803以降に標準搭載）も使います。
- **ダブルクリックだけで使えます。** `install-windows.bat` と `start-windows.bat` は展開したフォルダの直下にあります。ダブルクリックで実行した場合は、結果を読めるようにコンソールを開いたままにします。スクリプトから呼び出していて、すぐに処理を戻したい場合は `SEG_NO_PAUSE=1` を設定してください（CI 上で実行した場合も同じ動作になります）。
- **NVIDIA ドライバ**：このリポジトリでは最低限必要なドライバのバージョンを定めていないため、ここでも具体的な数値は挙げません。インストーラーが実際に確認しているのは、`nvidia-smi` を実行できるかどうかです。実行できれば CUDA 12.8 版の wheel（Turing / RTX 20xx 以降。Blackwell を含む）が、実行できなければ CPU 版が入ります。Maxwell / Pascal / Volta では `install-windows.bat cuda124` を実行してください。導入後は、`python -c "import torch; print(torch.cuda.is_available())"` を実行して `True` と表示されることを確認してください。
- **SAM の重みが無くても、SAM クリックアシスト以外はすべて動きます。** ブラシ、矩形、消しゴム、スポット検出、ひび割れ検出、学習、評価、各種エクスポートは、いずれも SAM の重みを使いません。5つのチェックポイントは、Windows ではインストール時に、それ以外の環境では初めて使うときにダウンロードされ、ソースに記録された SHA-256 で検証されます。
- **インターネット接続はインストール時には必要ですが、使うときには不要です。** インストール時には、PyPI から PyTorch と Python の依存パッケージを、GitHub から SAM アシスト用ライブラリを取得し、さらに SAM チェックポイントと UI の npm パッケージをダウンロードします。それ以降、アノテーション・学習・評価・エクスポートはすべて手元のマシンで動きます。ただし、次の3つは必要になった時点で通信します。1つ目は、まだダウンロードしていない SAM チェックポイントです。2つ目は DINOv2 の重みで、学習で初めて必要になったときにダウンロードします（既定でオンになっている学習設定 Auto が DINOv2 の特徴を使います。DINOv2 蒸留と「DINOv2 特徴で分割」も同様です）。3つ目は、任意で使う [AI サポートパネル](ai-support.md)です。このパネルは、実行中にモデルが見る画像、指示、プロジェクト名、ツールの結果を、使用するモデルサーバーに送ります（別のアドレスを指定するか `openai` バックエンドを選ばない限り、送り先はこのマシン上のサーバーです）。オフライン環境で使う場合は、OS と Python のバージョンが同じで、インターネットにつながっているマシンで `python scripts/install.py --offline-pack <dir>` を実行し、先にバンドルを作成してください。バンドルに入るのは、trainer API のロックファイル（`apps/trainer_api/requirements.txt`）で固定しているパッケージ（torch、torchvision と git の行を除く）、torch と torchvision 本体（Windows と Linux は PyTorch の CUDA 12.8 版のインデックスから、macOS は PyPI から取得）、SAM チェックポイントです。バンドルの `install_offline.py` は、依存関係を解き直さずに、固定した版のまま入れます。ONNX Runtime（serving API のロックファイルで固定しており、バンドルの作成時には読み込みません）、EfficientSAM と TinySAM のライブラリ、UI の npm パッケージ、DINOv2 の重み（`dinov2_vitb14_pretrain.pth`。torch の `hub/checkpoints` フォルダに置きます）は入らないので、そのマシンで必要なものは別途持ち込んでください。ロックファイルが git の URL で指定している MobileSAM と SAM 2 のライブラリは、これまでどおり GitHub から取得します。GitHub に接続できないマシンでは、`install_offline.py` がその旨を表示し、それ以外をすべて入れます。

## 1. コードを入手する（git 不要）

[Releases ページ](https://github.com/segmen-pixel/seg-studio/releases)から最新の **Source code (zip)** をダウンロードし、好きな場所に展開してください（リポジトリページ上部にある緑色の **Code → Download ZIP** ボタンからも同じものを入手できます）。git を使う場合は `git clone https://github.com/segmen-pixel/seg-studio.git` を実行します。

Windows 版・macOS 版とも、ビルド済みのパッケージは公開していません。Releases ページのリリースはソースのアーカイブ（と SBOM）なので、次の「2.」の手順はどの環境でも必要です。パッケージは、チェックアウトしたソースから `python scripts/build_installer.py` で作成できます。パッケージを公開する場合は、ダウンロードしたファイルを照合できるよう、Releases ページに `SHA256SUMS.txt` を添えて掲載します。

## 2. インストールして起動する

**Windows（NVIDIA GPU）：**
```bash
install-windows.bat
start-windows.bat
```

ターミナルを使う必要はありません。展開したフォルダの直下にある `install-windows.bat` をダブルクリックし、続けて `start-windows.bat` をダブルクリックするだけで動きます。インストーラーは GPU を自動で判別し（`cpu` または `cuda124` を指定すると上書きできます）、Python が見つからない場合は `winget` で導入します。導入できない場合は、インストールの手順を案内します。起動スクリプトは、サーバーの準備ができるとブラウザで UI を自動的に開きます。インストール後は、**Seg-Studio** のショートカットからも、コンソールを出さずに同じように起動できます（下の「Seg-Studio のアプリ」を参照）。

**macOS（Apple Silicon / Intel）：**
```bash
bash install-macos.sh
bash start-macos.sh
```

**Linux（NVIDIA GPU または CPU）：**
```bash
python3.11 -m venv .venv
.venv/bin/python scripts/install.py
bash scripts/start_local.sh
```

Python 3.11、Node.js 18以上（22 LTS を推奨）、git をあらかじめインストールしておいてください（Ubuntu の場合は `apt install python3.11 python3.11-venv git`。Node.js は nodejs.org かディストリビューションのパッケージから入れます）。このスクリプトは仮想環境の外では動かず、何かをダウンロードする前に、この3つがそろっているかを確認します。構築される環境は Windows インストーラーと同じです（CUDA 12.8 版 PyTorch の wheel、CUDA 版 ONNX Runtime、SAM 5種すべて）。Ubuntu 22.04 と RTX 3080 Ti の組み合わせで、アノテーション、GPU での学習、CUDA での推論まで動作を確認しています。

停止するときは `stop-windows.bat` / `bash stop-macos.sh` / `bash scripts/stop_local.sh` を使います。

**Seg-Studio のアプリ。** インストーラーは、そのチェックアウト用のアプリも作ります。Windows ではスタートメニューとデスクトップの **Seg-Studio** のショートカット、macOS では `~/Applications` の **Seg-Studio** です。開くと、コンソールのウィンドウを出さずに trainer API と serving API を起動し、API の準備ができたら画面を開きます。そのあとは、タスクバーとタスクトレイ（Windows）、または Dock とメニューバー（macOS）にアイコンが常駐し、**Seg-Studio を開く**、**ログのフォルダを開く**、**停止して終了** を選べます。**停止して終了** を選ぶと、停止スクリプトが実行されます。学習、推論、AI サポートのどれかが実行中のときは、どのデバイスで動いていても、先に確認します。Windows ではタスクバーのボタンの **ウィンドウを閉じる**、macOS では Dock や Cmd+Q でアプリを終了しても、同じように停止します。動作中にもう一度アプリを開くか、タスクバーのボタンや Dock のアイコンをクリックすると、画面が開きます。別のフォルダの Seg-Studio や古い版の Seg-Studio がすでに動いている場合は、それを停止してこのフォルダの Seg-Studio を起動するかを確認します。**停止して終了** で止めるのは、このフォルダの Seg-Studio か、アプリ自身が起動したものだけです。ポートを使っていた別のプログラムや、そのまま使うことにした別のフォルダの Seg-Studio は止めません。停止スクリプトは、`stop-windows.bat` と同じく、ポート 8002、8001、5173、8081 で待ち受けているものを止めます。

アプリが使うのは `.bat` / `.sh` と同じ起動・停止スクリプトで、UI の開発サーバーは起動しません。アプリのログは `logs/windows/app.log`（macOS では `logs/macos/app.log`）で、起動スクリプトの出力は同じフォルダの `app_start.log` に残ります。macOS では、アプリの起動部分の出力も同じフォルダの `app_console.log` に残ります。LAN からのアクセスを許可しているときは、起動スクリプトは LAN トークンそのものではなく、その保存場所（projects フォルダの `runtime_settings.json` の `api_token`）を書き出します。

どちらもインストーラーがお使いの PC 上で作るもので、ダウンロードしたものではないため、Windows や macOS が署名を確かめることはありません。作らない場合は、`install-windows.bat` に `--no-shortcuts`、`install-macos.sh` に `--no-app` を付けます。macOS の Dock のアイコンとメニューバーの項目には pyobjc（MIT）を使います。インストーラーが `apps/trainer_api/requirements-macos-app.txt` から入れます。pyobjc が無くてもアプリは起動して画面を開きますが、メニューの代わりにダイアログで確認します。Dock には、Python がフレームワーク版（python.org 版または Homebrew 版）なら Seg-Studio と、それ以外の Python（pyenv、conda、uv など）なら Python と表示されます。チェックアウトがデスクトップ、書類、ダウンロードのフォルダにある場合は、そのフォルダを使ってよいかを macOS が一度だけ確認します。許可しなかった場合は、アプリを開いたときにその旨を表示します（**システム設定 → プライバシーとセキュリティ → ファイルとフォルダ** で許可するか、フォルダを移してください）。アプリは、最後にインストーラーを実行したフォルダの Seg-Studio を開きます。消すときは、ショートカットまたは `Seg-Studio.app` を削除します。タスクバーにピン留めした Seg-Studio は、ピン留めしたときのフォルダを開き続けます。新しい版を新しいフォルダにインストールしたら、ピン留めを外して、付け直してください。画面がビルドされていない場合（`--skip-ui` を付けたときや、UI のビルドに失敗したとき）は、アプリはビルド済みの画面を表示するものなので、インストーラーはショートカットもアプリも作りません。

Windows では、`restart-windows.bat` でポート 8002 の trainer API だけを再起動できます。「次回サーバー起動時に反映されます」と表示される設定変更は、この再起動で反映されます。LAN トークンを再利用するので、ブラウザのセッションはそのまま使えます。また、新しいサーバーは切り離して起動されるため、SSH 経由で再起動しても、SSH のセッションを閉じたときに一緒に止まることはありません。学習の実行中や予約中は、`--force` を付けない限り再起動しません。`--dry-run` を付けると、何をするかを表示するだけで、実際には再起動しません。serving API と UI の開発サーバーは再起動の対象外なので、そちらは停止してから起動し直してください。

> Windows のインストーラーは、必要なものを自分で用意します。Python・Node.js・git が無ければ winget で自動的にインストールするので、3つとも入っていない PC でも、ダブルクリック1回で使える状態になります。winget が無い環境（App Installer が入っていない Windows 10 など）では、Python の入手先を案内して停止します。Node.js と git は無くても停止しませんが、Node.js が無いと UI が再ビルドされず、git が無いと SAM クリックセグメンテーションが使えません。macOS では git が必須で、`install-macos.sh` は git が無いと停止します。Apple Silicon の macOS では MPS（Metal）が自動的に使われ、推論だけでなく学習にも使えます。ただし、個数カウントには NVIDIA GPU が必要です。どの環境で何が動くかは[プラットフォーム対応](#プラットフォーム対応)をご覧ください。

## 3. UI を開く

ブラウザで **http://localhost:8002/ui/** を開いてください。インストールはこれで完了です。

セグメンテーションが初めての方は、続けて[はじめてのファーストラン手順](first-run-manual.md)に進んでください。アプリを開いてから最初の推論結果が出るまでを、約10分でたどれます。

## 別の方法：Docker（docker compose）

コンテナで動かす場合の手順です。CPU のみで動作し、セマンティック学習は動きますが大幅に遅くなります。個数カウント（インスタンス）の学習には、コンテナを使わずに直接インストールした NVIDIA 環境が必要です。

```bash
# Windows の場合は python3 ではなく python を使ってください
python3 -c "import secrets; print('SEG_API_TOKEN=' + secrets.token_urlsafe(24))" >> .env
docker compose up --build
```

ブラウザで **http://localhost:5173/** を開いてください。UI コンテナの nginx が、`/api`、`/v2`、`/ws` へのリクエストを trainer API に中継します。どのポートも `127.0.0.1` にだけ公開されます。`.env` の作成は省略できません。各コンテナは自身のネットワーク名前空間の中で `0.0.0.0` にバインドするため、トークンが無いと trainer が起動しないからです。また、GPU を要求していないため、アノテーション・推論・UI は使えるものの、セマンティック学習は CPU で実行されるので大幅に遅くなります。個数カウント（インスタンス）の学習は使えません。詳しくは[デプロイメントガイド](deployment.md#docker-によるデプロイ任意)をご覧ください。


## 動作環境

- **OS**：Windows 10 / 11（64-bit）、macOS 12以上（Apple Silicon を推奨）、または Linux（Ubuntu 22.04 で確認済み。`scripts/install.py` を使用）
- **Python**：3.11以上（依存関係のロックファイルは 3.11 向けにコンパイルされており、3.10 では解決できません）
- **Node.js**：18以上（UI のビルドにのみ使用）
- **ディスク**：CUDA 版の導入に約 5 GB の空き（詳しくは上の「0. インストール前チェック」を参照）

## プラットフォーム対応

| | Windows + NVIDIA | Linux + NVIDIA | Apple Silicon（MPS） | CPU のみ |
|---|---|---|---|---|
| アノテーション / SAM アシスト | 対応 | 対応 | 対応（SAM 5種のうち4種。TinySAM は macOS には導入されません） | 対応 |
| セマンティックセグメンテーションの学習 | 対応 | 対応 | 対応 | 対応（大幅に低速） |
| 個数カウント（インスタンス）の学習 | 対応 | 対応 | 非対応 | 非対応 |
| ONNX エクスポート | 対応 | 対応 | 対応 | 対応 |
| ONNX 推論 | 対応（CUDA プロバイダ） | 対応（CUDA プロバイダ） | 対応（CPU プロバイダ） | 対応（CPU プロバイダ） |
| Core ML エクスポート | `coremltools` を import できる場合のみ | `coremltools` を import できる場合のみ | 対応 | `coremltools` を import できる場合のみ |

- 学習に使うデバイスは既定で `auto` になっており、CUDA → MPS → CPU の順に選ばれます。NVIDIA でセマンティックセグメンテーションを学習する場合は、VRAM 4 GB 以上を推奨します。
- Windows インストーラーは、既定で CUDA 12.8 版 PyTorch の wheel（Turing / RTX 20xx 以降。Blackwell の RTX 5090 を含む）を導入します。旧世代の GPU（Maxwell / Pascal / Volta）では、`install-windows.bat cuda124` で CUDA 12.4 版の wheel を導入してください。Linux のスクリプトも同じ CUDA 12.8 版の wheel を使い、切り替えのオプションはありません（GPU の無いマシンでは CPU で動きます）。
- MPS では混合精度が無効になるため、同程度の NVIDIA GPU より学習が遅くなります。また、MPS はシステム全体と共有するユニファイドメモリを使います。メモリが不足する場合は[トラブルシューティング](troubleshooting.md)をご覧ください。
- **個数カウント（インスタンスセグメンテーション）の学習には NVIDIA GPU が必要です。** VRAM に合わせた自動調整は CUDA デバイスでのみ行われます。RTX 3090 での実測では、`small` モデルに必要なメモリは既定のバッチ 8 で 8 GiB、バッチ 4 で 5.5 GiB、バッチ 2 で 3.5 GiB です。3.5 GiB 未満には対応していません。
- Core ML のエクスポートには `coremltools`（8.3.0 に固定）が必要で、import できない場合は HTTP 501 を返します。macOS インストーラーは `coremltools` を明示的に導入します。
- OpenVINO IR のエクスポートは、Windows インストーラーのオプションです（`install-windows.bat --with-openvino`、約 300 MB）。macOS インストーラーには同等のオプションはありません。
- ビルド済みのパッケージは、どちらのプラットフォームでも公開していません。リリースはソースのアーカイブで、インストールスクリプトが手元のマシンで環境を構築します（上の「2.」）。macOS については意図的にそうしています。署名の無い `.app` を配布すると、利用者全員が Gatekeeper で手動で許可しなければならず、`install-macos.sh` を実行するより手間がかかるためです。
