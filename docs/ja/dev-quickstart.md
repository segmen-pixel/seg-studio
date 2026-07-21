# 開発者向けクイックスタート

Seg-Studio の開発環境を Windows、macOS、または Linux/WSL 上で構築する手順です。

## 前提条件

| ツール | バージョン | 備考 |
|--------|-----------|------|
| Python | 3.11+ | 3.10 でも動作しますが 3.11 以上を推奨します |
| Node.js | 18+ | UI のビルドおよび開発サーバーに必要です |
| npm | 9+ | Node.js に同梱されています |
| NVIDIA GPU (CUDA) | 学習に必須 | CUDA 12.8 (cu128)（旧世代 GPU は cu124）。アノテーション/推論は GPU なしでも可。 |
| Apple Silicon | 任意 | macOS での MPS アクセラレーション（M1/M2/M3/M4） |

以下のコマンドでバージョンを確認してください。

```bash
python --version      # 3.11+
node --version        # v18+
npm --version         # 9+
nvidia-smi            # 任意 — CUDA ドライバの確認
```

## セットアップ手順

### 1. 仮想環境の作成

**Windows:**
```bat
cd seg-studio
python -m venv .venv-windows
.venv-windows\Scripts\activate
```

**macOS:**
```bash
cd seg-studio
python3 -m venv .venv-macos
source .venv-macos/bin/activate
```

**Linux / WSL:**
```bash
cd seg-studio
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Python パッケージのインストール

```bash
pip install -r apps/trainer_api/requirements.txt
pip install -e packages/segcore
```

（`segcore` はローカルの学習コアパッケージで、editable インストールが正式な手順です。未インストールでも API 側の `sys.path` フォールバックで動作します。）

NVIDIA GPU をお持ちの場合は、CUDA 対応の PyTorch をインストールしてください（推奨）。旧世代 GPU（Maxwell/Pascal/Volta）の場合は `cu128` の代わりに `cu124` を指定してください。

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

macOS の場合（デフォルトの PyPI ホイールに Apple Silicon の MPS サポートが含まれています）:

```bash
pip install torch torchvision
```

CPU のみで使用する場合は以下を実行してください。

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. UI パッケージのインストール

```bash
cd apps/trainer_ui
npm install
cd ../..
```

## API サーバーの起動

初回起動時に `dist/` が存在しないか古い場合、API が自動的に UI をビルドします。

**Windows（簡易起動）:**
```bat
scripts\windows\start_api_only.bat
```

**Windows（フルスタック: Serving API + UI 開発サーバー含む）:**
```bat
scripts\windows\start_local_windows.bat
```

**macOS（フルスタック）:**
```bash
bash scripts/macos/start_local_macos.sh
```

**macOS（API のみ）:**
```bash
bash scripts/macos/start_api.sh
```

**Linux / WSL:**
```bash
# 仮想環境を有効化した状態で、リポジトリのルートから実行してください
export SEG_PROJECTS_DIR="$(pwd)/projects"
export SEG_DB_PATH="$(pwd)/projects/app.db"
python -m uvicorn apps.trainer_api.app.main:app --port 8002
# LAN アクセスを許可する場合: --host 0.0.0.0 を追加（または SEG_HOST=0.0.0.0 を設定）
```

API はポート **8002** にバインドされます。初回起動時は PyTorch、OpenCV、scikit-learn などの重いモジュールがバックグラウンドで読み込まれます。起動が完了するまでブラウザにはローディング画面が表示されます（ハードウェアにより約 5〜15 秒）。

## UI 開発サーバーの起動

フロントエンド開発時はホットリロード付きの開発サーバーを使用します。

```bash
cd apps/trainer_ui
npm run dev
```

Vite 開発サーバーが **http://localhost:5173** で起動し、API へのリクエストは自動的に `localhost:8002` にプロキシされます。React/TypeScript のソースファイルを編集すると、変更が即座に反映されます。

本番用ビルドを API から直接配信する場合は以下を実行してください。

```bash
cd apps/trainer_ui
npm run build
```

ビルド後は **http://localhost:8002/ui/** から UI にアクセスできます。

## 接続確認

```bash
curl http://localhost:8002/api/v1/health
```

正常な場合、以下のような JSON レスポンスが返ります。

```json
{
  "status": "ok",
  "version": "1.0.0",
  ...
}
```

その他の便利なエンドポイントは以下のとおりです。

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
| `SEG_MODELS_DIR` | `<repo>/models` | モデルレジストリと SAM チェックポイントの保存先 |
| `LOG_LEVEL` | `INFO` | ログレベル: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_FORMAT` | `text` | ログ形式: `text` または `json` |
| `SEG_HOST` | `127.0.0.1` | バインドアドレス。LAN アクセスには `0.0.0.0` を設定してください |
| `SEG_TORCH_DEVICE` | `auto` | デバイス指定: `cuda:0`, `cpu`, `auto` |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | CUDA メモリアロケータ設定（起動スクリプトで自動設定されます） |

## よくある問題と対処法

### ポート 8002 で ECONNREFUSED エラーが発生する

API サーバーが起動していないか、起動処理が完了していません。

1. API プロセスが動作しているか確認してください（プロセスリストで `uvicorn` を検索）。
2. バックグラウンド起動の完了を待ってください。`/startup-status` エンドポイントはすべてのルーターが登録されるまで `{"ready": false}` を返します。
3. ログにエラーが出ていないか確認してください。
   - Windows: `logs\windows\trainer.log`
   - macOS: `logs/macos/trainer.log`
   - Linux: コンソール出力または `/tmp/seg_trainer.log`

### CUDA が認識されない

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

`False` と表示される場合は以下を確認してください。

1. NVIDIA ドライバがインストールされていることを確認します: `nvidia-smi`
2. CUDA 対応の PyTorch を再インストールします:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
3. 非標準のインストールパスを使用している場合は `CUDA_HOME` / `CUDA_PATH` が正しく設定されているか確認してください。

### Vite プロキシエラー（開発サーバーから API 呼び出し時に 404）

Vite 開発サーバーは API リクエストを `localhost:8002` にプロキシします。API が別のホストやポートで動作している場合は `apps/trainer_ui/vite.config.mjs` を編集してください。

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

報告された型エラーをコミット前に修正してください。本番ビルド（`npm run build`）は最初に `tsc -b` を実行するため、型エラーがあるとビルドが失敗します。

### ポートが使用中

```bash
# Windows
netstat -ano | findstr :8002

# macOS / Linux
lsof -i :8002
```

競合するプロセスを終了するか、`--port` オプションで別のポートを指定してください。

### /ui/ にアクセスすると白い画面が表示される

UI がビルドされていません。以下を実行してください。

```bash
cd apps/trainer_ui
npm install
npm run build
```

または Vite 開発サーバー（`npm run dev`）を起動して `http://localhost:5173` にアクセスしてください。

## サービスの停止

**Windows:**
```bat
scripts\windows\stop_local_windows.bat
```

**macOS:**
```bash
bash scripts/macos/stop_local_macos.sh
```

**Linux:**
```bash
# フォアグラウンドで起動した場合は Ctrl+C で uvicorn を停止できます
# バックグラウンドで起動した場合:
pkill -f "uvicorn apps.trainer_api"
```

## テストの実行

TypeScript チェック、リンティング、Python インポート検証、およびオプションの E2E テストを含む統合テストランナーについては、`scripts/test.sh`（Linux/WSL）または `scripts/test.bat`（Windows）を参照してください。

```bash
# Linux / WSL
bash scripts/test.sh

# Windows
scripts\test.bat
```
