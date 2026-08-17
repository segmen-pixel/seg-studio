# デプロイメントガイド

## セキュリティに関する重要事項

Seg-Studio は **ローカル環境または信頼できるネットワーク** での使用を前提として設計されており、初期状態でもそうなっています。`scripts/` 配下の起動スクリプトはいずれも既定で `127.0.0.1` にバインドするため、インストール直後のサーバーは動かしているマシンからしか到達できません。LAN 公開はオプトインで、設定画面の **LAN からのアクセスを許可**、または `SEG_HOST=0.0.0.0` で有効になり、次回起動から反映されます。

このループバック既定では、trainer API に **トークンは不要** です（Host 許可リストと同一オリジンチェックは常に有効なので、たまたま開いた Web ページがローカルの API を操作することはできません）。LAN / リバースプロキシ運用では、共有シークレット `SEG_API_TOKEN` を設定すると、`/api/v1/*`、`/v2/*`、`/ws/v2/*` へのすべてのリクエストに `X-API-Token` ヘッダが必須になります（WebSocket も対象。ブラウザからはヘッダの代わりに `?api_token=<値>` を付与できます）。Web UI は最初に一度サインインし、以降はセッション Cookie を使います。この 2 つの設定は連動しています。`SEG_HOST`（または保存された LAN トグル）がループバック以外のアドレスを指しているのに `SEG_API_TOKEN` が空の場合、trainer API は **起動を拒否** します。この判定が見るのは `SEG_HOST` であって uvicorn の `--host` ではないため、`--host 0.0.0.0` を手で渡すときは `SEG_HOST=0.0.0.0` も必ず設定してください。設定を忘れても外部に公開されることはありません。トークン未設定のサーバーは動かしているマシン上のクライアントにしか応答しないため、ポートは開くものの外部からのリクエストはすべて拒否されます。ただし失敗の通知が、起動時のメッセージではなくリクエストごとの 401 になります。

トークンが守るのは trainer API だけです。ポート 8001 の serving API には認証が一切ありません。そのポートを開ける前に、後述の **serving API（ポート 8001）には認証がありません** を必ずお読みください。どちらの API も、TLS と本格的な認証を備えたリバースプロキシなしにパブリックインターネットへ公開しないでください。

### ブラウザからのサインイン

ブラウザは `<img src>`・スタイルシート・ダウンロードリンクにカスタムヘッダを付けられないため、ヘッダだけを認証手段にすると同梱の Web UI はオーバーレイ 1 枚すら表示できません。そのため UI は最初に一度サインインします。

1. `SEG_HOST=0.0.0.0` で起動します。設定画面の LAN アクセスを有効にして、各プラットフォームの起動スクリプト（`scripts/windows/start_local_windows.bat` / `scripts/macos/start_api.sh` / `scripts/start_local.sh`）から起動しても同じです。LAN で最初に起動したとき、起動スクリプトが `SEG_API_TOKEN` を生成して表示し、`projects/runtime_settings.json` の `api_token` に保存します。次回以降は同じ値を再利用します。自分のシークレットを使いたい場合は環境変数で設定してください。
2. 別の端末から `http://<サーバー>:8002/ui/` を開くと、UI 読み込み前にトークンの入力を求められます。
3. 認証に成功するとサーバーがセッション Cookie を発行します。`HttpOnly`（ページのスクリプトからシークレットを読めない）、`SameSite=Strict`（他サイトから便乗できない）で、値はトークンそのものではなくハッシュです。以降はブラウザが同一オリジンの全リクエスト（画像・ダウンロード含む）に自動で付与します。

Cookie はトークンから決定的に導出されるため、サーバーを再起動してもセッションは維持され、`SEG_API_TOKEN` を変更すれば既存セッションはすべて無効になります。Cookie は「自動で送られてしまう」資格情報なので、Cookie 認証のリクエストには同一オリジン（CSRF）チェックが引き続き適用されます（スクリプトや `curl` が使う `X-API-Token` ヘッダは対象外）。なお同梱の `seg-sdk` クライアントはまだトークンを送らないため、サーバーと同じマシン上で実行するか（次の段落のとおり、ローカルからのリクエストはトークンを免除されます）、`SEG_API_TOKEN` 未設定のサーバーに接続してください。**この用途にリバースプロキシは不要**です。必要になるのは TLS や二段目の認証を追加したい場合だけです。

サーバーを動かしているマシン自身からのリクエストはトークンを免除されます。LAN に公開したからといって、自分のデスクトップアプリに自分でサインインさせられるのはおかしいためです。判定には TCP の接続元アドレスを使います。Host ヘッダと違い接続元アドレスはリモートから偽装できません。免除されるのはトークンだけで、同一オリジンチェックと Host 許可リスト（デフォルトのローカルインストールを守っているものと同じ）は引き続き適用されます。`X-Forwarded-For` / `X-Forwarded-Host` / `X-Real-IP` / `Forwarded` を持つリクエストはローカル扱いしません。同一ホスト上のリバースプロキシもループバックから接続しますが、その背後のリクエストはローカルではないからです。

サインイン用の 3 経路（`/api/v1/auth/status`、`/auth/session`、`/auth/logout`）は `/api/v1/*` の中で唯一トークンガードの対象外です。資格情報を持たないクライアントが資格情報を得るための入口だからです。`/auth/status` はトークンが必要かどうかしか返さず、`/auth/session` はトークンを知っている相手にしか何も渡しません。

## serving API（ポート 8001）には認証がありません

`apps/serving_api` は、意図的に最小構成にしてある 2 つめの FastAPI アプリです。trainer が `models/registry/` で有効化した ONNX モデルを読み込み、`POST /segment` と `POST /count` に応答します。**`SEG_API_TOKEN` を読みません。トークン・Cookie・Host・オリジンのいずれのチェックも持っていません。** ポート 8001 に到達できるものは、推論の実行、登録済み model id の一覧取得（`GET /models`）、レジストリポインタの再読み込み（`POST /reload`）をすべて実行できます。

これが安全なのはループバック既定のときだけです。起動スクリプトは serving API を trainer と同じ `SEG_HOST` で起動するため（`scripts/windows/start_local_windows.bat`、`scripts/macos/start_local_macos.sh`、`scripts/start_local.sh`、`scripts/windows/start_serving_api.bat`）、設定画面の **LAN からのアクセスを許可** を ON にすると、ポート 8001 も LAN に公開されます。trainer と違い、トークンによる保護はありません。ON にする前に、どちらかを選んでください。

- serving API はループバックのままにし、呼び出し側も同じマシンに置く。これが既定で、設定は不要です。
- TLS 終端と認証を行うリバースプロキシの背後に置き（後述の nginx 設定例で `proxy_pass http://127.0.0.1:8001;`）、**かつ** ホストのファイアウォールでポート 8001 を塞いで、プロキシ経由以外の経路をなくす。

`docker compose` では `127.0.0.1:8001` のみに公開されます。

## 推奨アーキテクチャ

```
[インターネット] → [リバースプロキシ (nginx)] → [Seg-Studio API :8002]
                        |
                  TLS 終端処理
                  Basic / OAuth 認証
                  レート制限
```

外部アクセスが必要な場合は、必ずリバースプロキシを経由し、TLS（HTTPS）と認証を設定してください。

## リバースプロキシの設定（nginx）

以下は nginx を使用したリバースプロキシの構成例です。TLS 証明書と Basic 認証を組み合わせています。

```nginx
server {
    listen 443 ssl;
    server_name seg-studio.example.com;

    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;

    # Basic 認証
    auth_basic "Seg-Studio";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        proxy_pass http://127.0.0.1:8002;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        # WebSocket 対応（学習ログ /api/v1/ws/...、ストリーミング /ws/v2/...）
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        client_max_body_size 2G;  # ZIP アップロード用
    }
}
```

この設定が転送する `Host` ヘッダはプロキシの `server_name` であり、`localhost` ではありません。この名前は既定で DNS リバインディング対策の許可リストに入っていないため、`SEG_API_TOKEN` を設定していないサーバーでは、プロキシ経由のリクエストがすべて 403 `Host header not allowed` で拒否されます。次のように許可リストへ追加してください。

```bash
SEG_ALLOWED_HOSTS=seg-studio.example.com
```

（有効な `X-API-Token`、または Web UI がサインイン後に受け取るセッション Cookie を伴うリクエストは、Host チェックより前に許可されます。そのためトークンで保護した構成ではこの設定は不要です。）

### htpasswd ファイルの作成

```bash
# apache2-utils がインストールされていない場合
sudo apt install apache2-utils

# ユーザーを作成
sudo htpasswd -c /etc/nginx/.htpasswd username
```

## 環境変数

| 変数名 | デフォルト値 | 説明 |
|--------|------------|------|
| `SEG_PROJECTS_DIR` | `./projects` | プロジェクトデータの保存ディレクトリ |
| `SEG_DB_PATH` | `./projects/app.db` | SQLite データベースのパス |
| `SEG_MODELS_DIR` | `./models` | エクスポート/登録済みモデルのディレクトリ |
| `SEG_TORCH_DEVICE` | `auto` | PyTorch デバイス（`cuda:0`、`mps`、`cpu`、または `auto`） |
| `SEG_API_TOKEN` | （空） | 任意の共有シークレット。設定すると `/api/v1/*`、`/v2/*`、`/ws/v2/*` に `X-API-Token`（ブラウザ WebSocket は `?api_token=`）が必須になります。Web UI は最初に一度サインインし、以降はセッション Cookie を使います（「ブラウザからのサインイン」参照） |
| `SEG_PREPARED_IMAGE_FORMAT` | `lossless` | 学習用コピーの形式。既定は PNG（可逆）で書き出します。ディスク容量とデコード速度を優先する場合のみ `jpeg` を指定します（PNG 比で約 3.6 分の 1 のサイズ。計測値はハンドブック参照） |
| `LOG_LEVEL` | `INFO` | ログレベル（`DEBUG`, `INFO`, `WARNING`, `ERROR`） |
| `LOG_FORMAT` | `text` | ログ形式（`text` または `json`） |
| `SEG_HOST` | `127.0.0.1` | バインドアドレス。LAN アクセスには `0.0.0.0` を設定。`SEG_API_TOKEN` が必須で、未設定なら起動時に拒否されます |
| `SEG_ALLOWED_HOSTS` | （空） | `Host` ヘッダで追加で許可するホスト名のカンマ区切りリスト（DNS リバインディング対策の許可リスト。ループバック名は常に許可されます）。リバースプロキシが自身のホスト名を転送し、`SEG_API_TOKEN` を設定していない構成で必要です |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | VRAM 管理を改善する CUDA アロケータ設定。環境変数が未設定の場合、API が起動時に自動設定します |

## Docker によるデプロイ（任意）

リポジトリに `docker-compose.yml` が同梱されています（trainer API は `127.0.0.1:8002`、serving API は `127.0.0.1:8001`、UI は `127.0.0.1:5173`。ポートはループバックのみに公開されます）。

**先に `SEG_API_TOKEN` を設定してください。** 各コンテナは自身のネットワーク名前空間の中で `0.0.0.0` にバインドします（公開ポートが届くようにするためです）。これは起動時チェックから見ると非ループバック bind なので、シークレットが空のままだと trainer コンテナは即座に終了します。`docker-compose.yml` と同じ場所に `.env` を作って設定してください（リポジトリルートがコンテナにマウントされ、アプリが起動時にこのファイルを読み込みます）。

```bash
# Windows の場合は python3 ではなく python を使ってください
python3 -c "import secrets; print('SEG_API_TOKEN=' + secrets.token_urlsafe(24))" >> .env
docker compose up -d
```

なお compose スタックは GPU を要求していないため（`deploy.resources.reservations.devices` の記述がありません）、trainer コンテナは CPU で動きます。アノテーション・推論・UI は使えますが、学習はできません。学習には、NVIDIA CUDA GPU を搭載したマシンへのネイティブインストールが必要です（「GPU 要件」を参照）。

Docker を使用しない場合は、直接実行してください。

```bash
pip install -r apps/trainer_api/requirements.txt

# localhost のみ（デフォルト）:
python -m uvicorn apps.trainer_api.app.main:app --port 8002

# LAN アクセス（リバースプロキシ + 認証と併用してください）:
SEG_HOST=0.0.0.0 python -m uvicorn apps.trainer_api.app.main:app --host 0.0.0.0 --port 8002
```

## データバックアップ

プロジェクトデータは `$SEG_PROJECTS_DIR`（デフォルト: `projects/`）に保存されます。

| パス | 内容 |
|------|------|
| `projects/app.db` | プロジェクトメタデータ（SQLite データベース） |
| `projects/{project_id}/` | 画像、マスク、学習ラン、モデルファイル |

**バックアップ手順:**

`projects/` ディレクトリ全体を定期的にバックアップしてください。

```bash
# 例: タイムスタンプ付きバックアップ
cp -r projects/ "backup/projects_$(date +%Y%m%d_%H%M%S)/"
```

データベースファイル `app.db` はプロジェクトの一覧情報を管理しています。画像やモデルのデータは各プロジェクトディレクトリ内に格納されるため、`projects/` フォルダ全体をコピーすれば完全なバックアップになります。

## GPU 要件

**NVIDIA（Windows / Linux）:**

| レベル | GPU | VRAM | 備考 |
|--------|-----|------|------|
| 最小要件 | GTX 1650 | 4GB | batch_size=4〜8 で学習可能 |
| 推奨 | RTX 3060 以上 | 8GB 以上 | 大きなバッチサイズ、SAM2 モデルの使用が可能 |

**Apple Silicon（macOS）:**

| レベル | チップ | 備考 |
|--------|--------|------|
| 推奨 | M1/M2/M3/M4 | 学習・推論とも MPS アクセラレーションを利用可能。CUDA が無い環境では `auto` が MPS を選びます |
| Intel Mac | - | CPU のみ。自動テストの対象外のため、未検証として扱ってください |

MPS では混合精度が無効になるため、同等の NVIDIA GPU より学習は遅くなります。
また MPS はシステム全体と共有するユニファイドメモリを使うため、メモリ不足に
なる場合は [トラブルシューティング](troubleshooting.md) を参照してください。

**CPU のみ:** アノテーション・推論・学習のいずれも全プラットフォームで動作します。
CPU 学習はテストスイートでも実行されていますが、GPU に比べて大幅に低速です。
`SEG_TORCH_DEVICE=cpu` を設定してください。

**個数カウント（インスタンスセグメンテーション）の学習には NVIDIA GPU が必要です。**
VRAM の自動調整は CUDA デバイスでのみ行われ、`small` モデルの実測値は既定の
バッチ 8 で 8 GiB、バッチ 4 で 5.5 GiB、バッチ 2 で 3.5 GiB です。3.5 GiB 未満は
非対応です。

## 本番環境でのチェックリスト

デプロイ前に以下の項目を確認してください。

- [ ] リバースプロキシ経由で TLS（HTTPS）が有効になっている
- [ ] 認証（Basic 認証または OAuth）が設定されている
- [ ] `SEG_PROJECTS_DIR` に十分なディスク容量がある
- [ ] 定期的なバックアップスケジュールが設定されている
- [ ] ログレベルが本番環境に適切に設定されている（通常は `WARNING` 以上）
- [ ] ファイアウォールでポート 8002 への直接アクセスがブロックされている（リバースプロキシのみ許可）
