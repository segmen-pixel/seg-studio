# デプロイメントガイド

## セキュリティに関する重要事項

Seg-Studio は**ローカル環境または信頼できるネットワーク**で使うことを前提に設計されており、初期設定もそのようになっています。`scripts/` 配下の起動スクリプトはいずれも既定で `127.0.0.1` にバインドするため、インストールした直後のサーバーには、そのサーバーを動かしているマシンからしかアクセスできません。LAN への公開は、利用者が明示的に有効にした場合だけです。設定画面の **LAN からのアクセスを許可**をオンにするか、`SEG_HOST=0.0.0.0` を設定すると有効になり、次回の起動から反映されます。

このループバックの既定設定では、trainer API に**トークンは不要**です（Host の許可リストと同一オリジンのチェックは常に有効なので、たまたま開いた Web ページがローカルの API を操作することはできません）。LAN やリバースプロキシで運用する場合は、共有シークレット `SEG_API_TOKEN` を設定すると、`/api/v1/*`、`/v2/*`、`/ws/v2/*` へのすべてのリクエストで `X-API-Token` ヘッダーが必須になります（WebSocket も対象です。ブラウザからは、ヘッダーの代わりに `?api_token=<値>` を付けることもできます）。Web UI は最初に一度だけサインインし、以降はセッション Cookie を使います。この2つの設定は連動しています。`SEG_HOST`（または保存された LAN アクセスの設定）がループバック以外のアドレスを指しているのに `SEG_API_TOKEN` が空の場合、trainer API は**起動しません**。この判定に使われるのは `SEG_HOST` で、uvicorn の `--host` ではありません。そのため、`--host 0.0.0.0` を手動で指定するときは、`SEG_HOST=0.0.0.0` も必ず設定してください。設定し忘れても、外部に公開されてしまうことはありません。トークンが未設定のサーバーは、そのサーバーを動かしているマシン上のクライアントにしか応答しないため、ポートは開いても、外部からのリクエストはすべて拒否されます。ただし、起動時にメッセージが出るのではなく、リクエストのたびに 401 が返る形で失敗することになります。

トークンで保護されるのは trainer API だけです。ポート 8001 の serving API には、認証の仕組みがまったくありません。このポートを開ける前に、後述の [serving API（ポート 8001）には認証がありません](#serving-apiポート-8001には認証がありません)を必ずお読みください。どちらの API も、TLS と本格的な認証を備えたリバースプロキシを置かずにインターネットへ公開しないでください。

### ブラウザからのサインイン

ブラウザは、`<img src>`、スタイルシート、ダウンロードリンクにカスタムヘッダーを付けられません。そのため、ヘッダーだけを認証手段にすると、同梱の Web UI はオーバーレイを1枚も表示できなくなります。そこで UI は、最初に一度だけサインインします。

1. `SEG_HOST=0.0.0.0` を設定して起動します。設定画面で LAN からのアクセスを有効にし、各プラットフォームの起動スクリプト（`scripts/windows/start_local_windows.bat` / `scripts/macos/start_api.sh` / `scripts/start_local.sh`）で起動し直しても同じです。LAN 向けに初めて起動したとき、起動スクリプトが `SEG_API_TOKEN` を生成して表示し、`projects/runtime_settings.json` の `api_token` に保存します。次回以降の起動では同じ値を使います。自分で決めたシークレットを使いたい場合は、環境変数で設定してください。
2. 別の端末から `http://<サーバー>:8002/ui/` を開くと、UI が読み込まれる前にトークンの入力を求められます。
3. 認証に成功すると、サーバーがセッション Cookie を発行します。この Cookie には `HttpOnly`（ページのスクリプトからシークレットを読めない）と `SameSite=Strict`（ほかのサイトから便乗して使えない）が設定されており、値はトークンそのものではなく、トークンのハッシュです。以降はブラウザが、同一オリジンへのすべてのリクエスト（画像やダウンロードを含む）にこの Cookie を自動で付けます。

Cookie はトークンから決定的に導出されるため、サーバーを再起動してもセッションは維持されます。また、`SEG_API_TOKEN` を変更すると、既存のセッションはすべて無効になります。Cookie はブラウザが自動的に送ってしまう種類の資格情報なので、Cookie で認証されたリクエストにも同一オリジン（CSRF）のチェックが引き続き適用されます（スクリプトや `curl` が使う `X-API-Token` ヘッダーは対象外です）。なお、同梱の `seg-sdk` クライアントはまだトークンを送らないため、サーバーと同じマシン上で実行するか（次の段落のとおり、そのマシンからのリクエストにはトークンが要りません）、`SEG_API_TOKEN` を設定していないサーバーに接続してください。**この用途にリバースプロキシは不要です**。リバースプロキシが必要になるのは、TLS や2段目の認証を追加したい場合だけです。

サーバーを動かしているマシン自身からのリクエストには、トークンは要りません。LAN に公開したからといって、自分のデスクトップアプリに自分のブラウザからサインインしなければならないのはおかしいためです。この判定には TCP の接続元アドレスを使います。Host ヘッダーと違い、接続元アドレスはリモートから偽装できません。免除されるのはトークンだけで、同一オリジンのチェックと Host の許可リスト（既定のループバック構成を守っているものと同じ）は引き続き適用されます。`X-Forwarded-For` / `X-Forwarded-Host` / `X-Real-IP` / `Forwarded` のいずれかを持つリクエストは、ローカルとして扱いません。同じホスト上のリバースプロキシもループバックから接続してきますが、その背後にあるリクエストはローカルから来たものではないからです。

サインイン用の3つのルート（`/api/v1/auth/status`、`/auth/session`、`/auth/logout`）は、`/api/v1/*` の中で唯一、トークンによる保護の対象外です。資格情報を持たないクライアントが資格情報を得るための入口だからです。`/auth/status` はトークンが必要かどうかしか返さず、`/auth/session` はトークンを知っている相手にしか何も発行しません。

## serving API（ポート 8001）には認証がありません

`apps/serving_api` は、意図的に最小限の構成にしてある2つ目の FastAPI アプリです。trainer が `models/registry/` で有効化した ONNX モデルを読み込み、`POST /segment` と `POST /count` に応答します。**`SEG_API_TOKEN` は読み込まず、トークン・Cookie・Host・オリジンのチェックも一切ありません。** ポート 8001 に接続できれば、推論の実行、登録済みの model id の一覧取得（`GET /models`）、レジストリのポインタの再読み込み（`POST /reload`）をすべて実行できます。

これで問題ないのは、ループバックの既定設定のときだけです。起動スクリプトは、serving API を trainer と同じ `SEG_HOST` で起動します（`scripts/windows/start_local_windows.bat`、`scripts/macos/start_local_macos.sh`、`scripts/start_local.sh`、`scripts/windows/start_serving_api.bat`）。そのため、設定画面の **LAN からのアクセスを許可**をオンにすると、ポート 8001 も LAN に公開されます。しかも trainer と違い、トークンによる保護はありません。オンにする前に、次のどちらかを選んでください。

- serving API はループバックのままにし、呼び出す側も同じマシンに置きます。これが既定の構成で、設定は不要です。
- TLS の終端と認証を行うリバースプロキシの背後に置き（後述の nginx の設定例で `proxy_pass http://127.0.0.1:8001;` とします）、**さらに**ホストのファイアウォールでポート 8001 を塞いで、プロキシを通らない経路をなくします。

`docker compose` では、serving API は `127.0.0.1:8001` にだけ公開されます。

## 推奨アーキテクチャ

```
[インターネット] → [リバースプロキシ (nginx)] → [Seg-Studio API :8002]
                        |
                  TLS 終端処理
                  Basic / OAuth 認証
                  レート制限
```

## リバースプロキシの設定（nginx）

以下は、nginx を使ったリバースプロキシの構成例です。TLS 証明書と Basic 認証を組み合わせています。

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

この構成では `SEG_API_TOKEN` を設定してください。nginx はこのマシンから接続しますが、nginx が付ける `X-Real-IP` ヘッダー（`X-Forwarded-For` や `Forwarded` も同じ）によって、サーバーはリクエストが別の場所から来たものだと判断します。トークンを設定していないサーバーは、そうしたリクエストをすべて 401 `This server has no API token configured` で拒否します。トークンを設定すれば、Web UI は一度サインインしたあとセッション Cookie を使い、ほかのクライアントは `X-API-Token` を送ります。どちらの場合も、`Host` ヘッダーは許可リストと照合されません。`SEG_ALLOWED_HOSTS` を設定しても、この動きは変わりません。この設定が効くのは、サーバーがこのマシンからのリクエストだと判断したものだけで、プロキシを経由したリクエストはそれに当たらないためです。

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
| `SEG_MODELS_DIR` | `./models` | エクスポート / 登録済みモデルのディレクトリ |
| `SEG_TORCH_DEVICE` | `auto` | PyTorch のデバイス（`cuda:0`、`mps`、`cpu`、`auto` のいずれか） |
| `SEG_API_TOKEN` | （空） | 任意の共有シークレット。設定すると、`/api/v1/*`、`/v2/*`、`/ws/v2/*` で `X-API-Token`（ブラウザの WebSocket では `?api_token=`）が必須になります。Web UI は最初に一度だけサインインし、以降はセッション Cookie を使います（[ブラウザからのサインイン](#ブラウザからのサインイン)を参照） |
| `SEG_PREPARED_IMAGE_FORMAT` | `lossless` | **非推奨。** `prepared/images/` に置く学習用コピーの形式。既定では PNG（RGB は可逆。アルファは失われます）で書き出します。`jpeg` を指定すると、画質と引き換えに、学習用コピーがかなり小さくなり、デコードも速くなります（トレードオフはハンドブックを参照）。今後はプロジェクト単位の設定（`PUT /api/v1/projects/{id}/image-store`）を使ってください。`images/` に何を保存するかは、そちらの設定で決まります |
| `LOG_LEVEL` | `INFO` | ログレベル（`DEBUG`、`INFO`、`WARNING`、`ERROR`） |
| `LOG_FORMAT` | `text` | ログ形式（`text` または `json`） |
| `SEG_HOST` | `127.0.0.1` | バインドアドレス。LAN からアクセスするには `0.0.0.0` を設定します。その場合は `SEG_API_TOKEN` が必須で、未設定だと起動時に拒否されます |
| `SEG_ALLOWED_HOSTS` | （空） | `Host` ヘッダーで追加で許可するホスト名（カンマ区切り）。DNS リバインディング対策の許可リストで、ループバックの名前は常に許可されます。効くのは、転送ヘッダーの無い、このマシンからのリクエストだけです。`X-Real-IP` や `X-Forwarded-For` を付けるリバースプロキシでは、代わりに `SEG_API_TOKEN` が必要です |
| `SEG_ANNOTATION_URL`、`SEG_CVAT_URL` | （未設定） | trainer API が `/annotate/*`（`/cvat/*`）を転送する先のアドレス。未設定ならそのルートはありません。起動スクリプトは、Label Studio を起動するとき（`SEG_START_LABEL_STUDIO=1`）だけ `SEG_ANNOTATION_URL` を `http://localhost:8081` に設定します。このルートは `/api/v1/*` と同じように保護され、トークンとセッション Cookie は転送先に渡りません |
| `SEG_VLM_BACKEND` | （未設定） | AI サポートパネルが使うモデルサーバーの種類。`ollama`・`openai`・`mlx`・`vllm`・`lmstudio`・`llamacpp` のいずれか（指定が無ければ `ollama`）。これか、`SEG_VLM_BASE_URL`・`SEG_VLM_API_KEY_ENV` のどれかを設定するとサーバーが固定され、パネルには種類の一覧の代わりにそのサーバーが表示されます（[ユーザーガイド](user-guide.md#ai-サポートパネル)を参照） |
| `SEG_VLM_BASE_URL` | （種類ごとの既定のアドレス） | モデルサーバーのアドレス（`http://` または `https://`）。省略すると、`ollama`・`lmstudio`・`vllm`・`mlx`・`llamacpp` では `127.0.0.1` の各サーバーの標準ポート、`openai` では `https://api.openai.com/v1` になります。そのため、手元の OpenAI 互換サーバーを使う場合は、これを指定するか、そのサーバー用の種類を選んでください。実行中にモデルが見る画像、指示、プロジェクト名、ツールの結果は、このアドレスに送られます |
| `SEG_VLM_MODEL` | （未設定） | パネルで別のモデルを選ばない場合に使うモデル |
| `SEG_VLM_API_KEY_ENV` | （未設定） | モデルサーバーの API キーを入れた環境変数の名前。その値は Bearer トークンとしてそのサーバーに送られます。その環境変数も trainer API の環境に設定してください |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | VRAM の管理を改善する CUDA アロケータの設定。この環境変数が未設定の場合は、API が起動時に自動で設定します |

## Docker によるデプロイ（任意）

リポジトリには `docker-compose.yml` が含まれています（trainer API は `127.0.0.1:8002`、serving API は `127.0.0.1:8001`、UI は `127.0.0.1:5173`。どのポートもループバックにだけ公開されます）。

**先に `SEG_API_TOKEN` を設定してください。** 各コンテナは、自身のネットワーク名前空間の中で `0.0.0.0` にバインドします（公開したポートからコンテナに届くようにするためです）。起動時のチェックはこれをループバック以外へのバインドとみなすため、シークレットが空のままだと trainer コンテナはすぐに終了します。`docker-compose.yml` と同じ場所に `.env` を作り、そこで設定してください（リポジトリのルートがコンテナにマウントされ、アプリは起動時にこのファイルを読み込みます）。

```bash
# Windows の場合は python3 ではなく python を使ってください
python3 -c "import secrets; print('SEG_API_TOKEN=' + secrets.token_urlsafe(24))" >> .env
docker compose up -d
```

なお、compose スタックは GPU を要求していないため（`deploy.resources.reservations.devices` の記述がありません）、trainer コンテナは CPU で動きます。アノテーション・推論・UI は使え、セマンティック学習も CPU で動きますが、大幅に遅くなります。個数カウント（インスタンス）の学習には、コンテナを使わずに直接インストールした NVIDIA 環境が必要です（[GPU 要件](#gpu-要件)を参照）。

AI サポートパネルは、モデルサーバーを `127.0.0.1` で探します。compose スタックの中では、このアドレスは trainer コンテナ自身を指すため、パネルの一覧にあるどの種類のサーバーも見つかりません。ホストや別のマシンで動いているモデルサーバーを使う場合は、trainer サービスの `environment` に `SEG_VLM_BACKEND` と `SEG_VLM_BASE_URL` を設定してください。アドレスには、コンテナから接続できるものを指定します。

Docker を使わない場合は、次のように直接実行してください。

```bash
pip install --no-deps -r apps/trainer_api/requirements.txt

# localhost のみ（デフォルト）:
python -m uvicorn apps.trainer_api.app.main:app --port 8002

# LAN アクセス（リバースプロキシ + 認証と併用してください）:
SEG_HOST=0.0.0.0 python -m uvicorn apps.trainer_api.app.main:app --host 0.0.0.0 --port 8002
```

## データバックアップ

プロジェクトデータは `$SEG_PROJECTS_DIR`（既定は `projects/`）に保存されます。

| パス | 内容 |
|------|------|
| `projects/app.db` | プロジェクトのメタデータ（SQLite データベース） |
| `projects/{project_id}/` | 画像、マスク、学習ラン、モデルファイル |

`projects/` ディレクトリ全体を定期的にバックアップしてください。

## GPU 要件

**NVIDIA（Windows / Linux）**

| レベル | GPU | VRAM | 備考 |
|--------|-----|------|------|
| 最小要件 | GTX 1650 | 4GB | batch_size=4〜8 で学習可能 |
| 推奨 | RTX 3060以上 | 8GB 以上 | 大きなバッチサイズや SAM2 モデルを使用可能 |

**Apple Silicon（macOS）**

| レベル | チップ | 備考 |
|--------|--------|------|
| 推奨 | M1/M2/M3/M4 | 学習・推論とも MPS アクセラレーションを利用可能。CUDA が無い環境では `auto` が MPS を選びます |
| Intel Mac | - | CPU のみ。インストーラーがアーキテクチャを判別し、その旨を表示します。自動テストの対象外のため、未検証として扱ってください |

MPS では混合精度が無効になるため、同程度の NVIDIA GPU より学習が遅くなります。また、MPS はシステム全体と共有するユニファイドメモリを使います。メモリが不足する場合は[トラブルシューティング](troubleshooting.md)をご覧ください。

**CPU のみ**：アノテーション・推論・学習のいずれも、すべてのプラットフォームで動作します。CPU での学習はテストスイートでも実行していますが、GPU に比べて大幅に遅くなります。`SEG_TORCH_DEVICE=cpu` を設定してください。

**個数カウント（インスタンスセグメンテーション）の学習には NVIDIA GPU が必要です。** VRAM に合わせた自動調整は CUDA デバイスでのみ行われます。`small` モデルの実測値は、既定のバッチ 8 で 8 GiB、バッチ 4 で 5.5 GiB、バッチ 2 で 3.5 GiB です。3.5 GiB 未満には対応していません。

## 本番環境でのチェックリスト

デプロイの前に、次の項目を確認してください。

- [ ] リバースプロキシ経由で TLS（HTTPS）が有効になっている。
- [ ] 認証（Basic 認証または OAuth）が設定されている。
- [ ] `SEG_PROJECTS_DIR` に十分なディスク容量がある。
- [ ] 定期的なバックアップのスケジュールが設定されている。
- [ ] ログレベルが本番環境に適した値になっている（通常は `WARNING` 以上）。
- [ ] ファイアウォールで、ポート 8002 への直接アクセスがブロックされている（リバースプロキシからのアクセスのみ許可）。
- [ ] ファイアウォールで、ポート 8001（認証の無い serving API）への直接アクセスもブロックされている。
