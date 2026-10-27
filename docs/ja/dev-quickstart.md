# 開発者向けクイックスタート

Seg-Studio の開発環境を、Windows、macOS、Linux / WSL のいずれかで構築する手順です。

## 前提条件

| ツール | バージョン | 備考 |
|--------|-----------|------|
| Python | 3.11+ | ロックファイルは 3.11 向けにコンパイルされており、3.10 では解決できません |
| Node.js | 18+ | UI のビルドと開発サーバーに必要です |
| npm | 9+ | Node.js に同梱されています |
| NVIDIA GPU (CUDA) | 任意 | CUDA 12.8（cu128）。旧世代の GPU では cu124。セマンティック学習は MPS / CPU でも動きます（低速）。個数カウント（インスタンス）の学習には CUDA が必要です。アノテーションと推論は GPU が無くても使えます。 |
| Apple Silicon | 任意 | macOS での MPS アクセラレーション（M1/M2/M3/M4） |

次のコマンドでバージョンを確認してください。

```bash
python --version      # 3.11+
node --version        # v18+
npm --version         # 9+
nvidia-smi            # 任意 — CUDA ドライバの確認
```

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

## セットアップ手順

### 1. 仮想環境の作成

**Windows：**
```bat
cd seg-studio
python -m venv .venv-windows
.venv-windows\Scripts\activate
```

**macOS：**
```bash
cd seg-studio
python3 -m venv .venv-macos
source .venv-macos/bin/activate
```

**Linux / WSL：**
```bash
cd seg-studio
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Python パッケージのインストール

```bash
pip install --no-deps -r apps/trainer_api/requirements.txt
pip install --no-deps -e packages/segcore
```

（`--no-deps` は、ロックファイルを書かれたとおりにインストールするための指定です。ロックファイルは依存関係をすべて解決済みなので、ここで解決し直すと、固定した OpenCV とは別のビルドがもう1つ入ってしまいます。）

（`segcore` はローカルの学習コアパッケージで、editable インストールが正式な手順です。インストールしていなくても、API 側の `sys.path` による代替手段で動作します。）

Windows で NVIDIA GPU を使う場合は、CUDA 対応の PyTorch に入れ替えます。ロックファイルは torch 2.13.0 を固定していて、PyPI の Windows 向けの 2.13.0 は CPU 専用の wheel です。CUDA のインデックスには 2.13 のビルドが無いため、そのインデックスから普通に `pip install` しても CPU 版の wheel が残ります（「already satisfied」と表示されるだけです）。`install-windows.bat` と同じように、次のコマンドで最新の CUDA ビルドに入れ替えてください。旧世代の GPU（Maxwell / Pascal / Volta）では、`cu128` の代わりに `cu124` を指定します。

```bat
pip install --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

（`--force-reinstall` を付けると、インストール済みの torch が入れ替わります。`--no-deps` を付けると、ロックファイルのほかの固定はそのまま残ります。torch が必要とするパッケージは、もともとロックファイルに含まれています。ロックファイル全体をもう一度インストールすると CPU 版の wheel に戻るので、そのあとでこのコマンドを実行し直してください。）

Linux では、ロックファイルで入る PyPI の torch がもともと CUDA 13 のビルドなので、入れ替える必要はありません。インストーラーと同じ CUDA 12.8 のビルドにしたい場合は、新しい仮想環境で、上のロックファイルのコマンドの代わりに `python scripts/install.py` を実行します。このスクリプトは、ロックファイルから torch と CUDA ランタイムの固定を除いてインストールし、torch を cu128 のインデックスから入れます。

macOS（Apple Silicon の MPS）と CPU だけで使う場合は、追加で入れるものはありません。ロックファイルで入る PyPI の torch で動きます。

### 3. UI パッケージのインストール

```bash
cd apps/trainer_ui
npm install
cd ../..
```

## API サーバーの起動

初回起動時に `dist/` が無いか古い場合は、API が UI を自動でビルドします。

**Windows（簡易起動）：**
```bat
scripts\windows\start_api_only.bat
```

**Windows（フルスタック：serving API と UI 開発サーバーを含む）：**
```bat
scripts\windows\start_local_windows.bat
```

**Windows（serving API のみ。ONNX 推論用で、学習用の依存パッケージは不要）：**
```bat
scripts\windows\start_serving_api.bat
```

**macOS（フルスタック）：**
```bash
bash scripts/macos/start_local_macos.sh
```

**macOS（API のみ）：**
```bash
bash scripts/macos/start_api.sh
```

**Linux / WSL：**
```bash
# 仮想環境を有効化した状態で、リポジトリのルートから実行してください
export SEG_PROJECTS_DIR="$(pwd)/projects"
export SEG_DB_PATH="$(pwd)/projects/app.db"
python -m uvicorn apps.trainer_api.app.main:app --port 8002
# LAN アクセスを許可する場合: SEG_API_TOKEN を設定したうえで、--host 0.0.0.0 と
# SEG_HOST=0.0.0.0 の両方を指定してください。--host はポートを開け、SEG_HOST は
# トークンが必要かどうかを起動時チェックが判断するのに使われます。トークン未設定の
# サーバーはこのマシン上のクライアントにしか応答しないため、--host だけではポートが
# 開くだけで、ほかの端末には 401 が返ります。
```

API はポート **8002** にバインドされます。初回起動時は、PyTorch、OpenCV、scikit-learn などの重いモジュールをバックグラウンドで読み込みます。起動が完了するまで、ブラウザにはローディング画面が表示されます（ハードウェアによって約5〜15秒）。

## UI 開発サーバーの起動

フロントエンドを開発するときは、ホットリロード付きの開発サーバーを使います。

```bash
cd apps/trainer_ui
npm run dev
```

Vite の開発サーバーが **http://localhost:5173** で起動し、API へのリクエストは自動的に `localhost:8002` にプロキシされます。React / TypeScript のソースファイルを編集すると、変更がすぐに反映されます。

API から直接配信する本番用のビルドを作る場合は、次を実行してください。

```bash
cd apps/trainer_ui
npm run build
```

ビルドが終わると、**http://localhost:8002/ui/** から UI にアクセスできます。

## 接続確認

```bash
curl http://localhost:8002/api/v1/health
```

正常な場合は、次のような JSON（バージョン、ディスク、RAM、GPU の情報を含みます）が返ります。

```json
{
  "status": "ok",
  "version": "0.9.x",
  ...
}
```

そのほかの便利なエンドポイントは次のとおりです。

| URL | 説明 |
|-----|------|
| http://localhost:8002/ui/ | ビルド済み UI（本番用） |
| http://localhost:8002/docs | Swagger API ドキュメント |
| http://localhost:8002/startup-status | 起動の進行状況 |
| http://localhost:5173 | Vite 開発サーバー（起動中の場合） |

## 環境変数

| 変数名 | デフォルト値 | 説明 |
|--------|------------|------|
| `SEG_PROJECTS_DIR` | `<repo>/projects` | すべてのプロジェクトデータのルートディレクトリ |
| `SEG_DB_PATH` | `<repo>/projects/app.db` | SQLite データベースのパス |
| `SEG_MODELS_DIR` | `<repo>/models` | モデルレジストリ（エクスポート / 有効化したモデル）の保存先。SAM チェックポイントは常に `<repo>/models/sam_checkpoints/` にあり、この変数では移動しません |
| `LOG_LEVEL` | `INFO` | ログレベル（`DEBUG`、`INFO`、`WARNING`、`ERROR`） |
| `LOG_FORMAT` | `text` | ログ形式（`text` または `json`） |
| `SEG_HOST` | `127.0.0.1` | バインドアドレス。LAN からアクセスするには `0.0.0.0` を設定してください（この場合は `SEG_API_TOKEN` が必須になります） |
| `SEG_API_TOKEN` | （空） | LAN にバインドするときの共有シークレット。`SEG_HOST` がループバック以外のときは必須で、未設定だと起動しません。起動スクリプトが、LAN 向けの初回起動時に自動で生成します |
| `SEG_TORCH_DEVICE` | `auto` | 使用するデバイスの指定（`cuda:0`、`cpu`、`auto`） |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | CUDA のメモリアロケータの設定（起動スクリプトが自動で設定します） |
| `SEG_STARTUP_NO_INSTALL` | （未設定） | `1` にすると、起動時に pip や npm で何もインストールせず、UI もビルドせず、ライセンス除外（`scripts/_nc_stubs.py`）も再適用しません（除外がまだ必要なファイルをログに出すだけです）。trainer API のテストスイートがこの変数を設定します（`apps/trainer_api/tests/conftest.py`） |
| `SEG_VLM_BACKEND`・`SEG_VLM_BASE_URL`・`SEG_VLM_MODEL`・`SEG_VLM_API_KEY_ENV` | （未設定） | **ラベル**タブの AI サポートパネルが使うモデルサーバー（[ユーザーガイド](user-guide.md#ai-サポートパネル)を参照） |

## よくある問題と対処法

### ポート 8002 で ECONNREFUSED エラーが発生する

API サーバーが起動していないか、起動処理がまだ終わっていません。

1. API のプロセスが動いているか確認してください（プロセスの一覧で `uvicorn` を探します）。
2. バックグラウンドでの起動処理が終わるまで待ってください。`/startup-status` エンドポイントは、すべてのルーターが登録されるまで `{"ready": false}` を返します。
3. ログにエラーが出ていないか確認してください。
   - Windows：`logs\windows\trainer_<timestamp>.log`（起動するたびに1つ作られます。`scripts\windows\status_windows.bat` で最新のログの末尾を表示できます）
   - macOS：`logs/macos/trainer.log`
   - Linux：コンソール出力または `/tmp/seg_trainer.log`

### CUDA が認識されない

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

`False` と表示される場合は、次を確認してください。

1. `nvidia-smi` で、NVIDIA ドライバがインストールされていることを確認します。
2. Windows では、CUDA 対応の PyTorch に入れ替えます（[Python パッケージのインストール](#2-python-パッケージのインストール)を参照）。
   ```bat
   pip install --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu128 torch torchvision
   ```
   Linux では、ロックファイルで入る CUDA 13 のビルドを使うには、CUDA 13 に対応したドライバが必要です。ドライバが古い場合は、上の手順で CUDA 12.8 のビルドにしてください。
3. CUDA を標準以外の場所にインストールしている場合は、`CUDA_HOME` / `CUDA_PATH` が正しく設定されているか確認します。

### Vite プロキシエラー（開発サーバーから API 呼び出し時に 404）

Vite の開発サーバーは、API へのリクエストを `localhost:8002` にプロキシします。API が別のホストやポートで動いている場合は、`apps/trainer_ui/vite.config.mjs` を編集してください。

```js
proxy: {
  "/api": { target: "http://YOUR_API_HOST:8002", ws: true },
  // ... その他のルート (/v2, /ws, /health, /version, /startup-status)
}
```

### TypeScript ビルドエラー

```bash
cd apps/trainer_ui
npx tsc --noEmit
```

報告された型エラーは、コミットする前に修正してください。本番ビルド（`npm run build`）は最初に `tsc -b` を実行するため、型エラーがあるとビルドに失敗します。

### ポートが使用中

```bash
# Windows
netstat -ano | findstr :8002

# macOS / Linux
lsof -i :8002
```

競合しているプロセスを終了するか、`--port` オプションで別のポートを指定してください。

### /ui/ にアクセスすると白い画面が表示される

UI がビルドされていません。次を実行してください。

```bash
cd apps/trainer_ui
npm install
npm run build
```

または、Vite の開発サーバー（`npm run dev`）を起動して、`http://localhost:5173` を開いてください。

## サービスの停止

**Windows：**
```bat
scripts\windows\stop_local_windows.bat
```

**macOS：**
```bash
bash scripts/macos/stop_local_macos.sh
```

**Linux：**
```bash
# フォアグラウンドで起動した場合は Ctrl+C で uvicorn を停止できます
# バックグラウンドで起動した場合:
pkill -f "uvicorn apps.trainer_api"
```

## テストの実行

TypeScript のチェック、リント、Python の import の検証に加え、必要に応じて E2E テストまでをまとめて実行するテストランナーが、`scripts/test.sh`（Linux / WSL）と `scripts/test.bat`（Windows）にあります。

```bash
# Linux / WSL
bash scripts/test.sh

# Windows
scripts\test.bat
```
