# デプロイメントガイド

## セキュリティに関する重要事項

Seg-Studio は **ローカル環境または信頼できるネットワーク** での使用を前提として設計されています。デフォルトでは API に **認証はかかっていません**。LAN / リバースプロキシ運用では、任意の共有シークレット `SEG_API_TOKEN` を設定すると、`/api/v1/*`、`/v2/*`、`/ws/v2/*` へのすべてのリクエストに `X-API-Token` ヘッダが必須になります（WebSocket も対象。ブラウザからはヘッダの代わりに `?api_token=<値>` を付与できます）。TLS と本格的な認証を備えたリバースプロキシなしにパブリックインターネットへ公開しないでください。

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
| `SEG_API_TOKEN` | （空） | 任意の共有シークレット。設定すると `/api/v1/*`、`/v2/*`、`/ws/v2/*` に `X-API-Token`（ブラウザ WebSocket は `?api_token=`）が必須になります |
| `LOG_LEVEL` | `INFO` | ログレベル（`DEBUG`, `INFO`, `WARNING`, `ERROR`） |
| `LOG_FORMAT` | `text` | ログ形式（`text` または `json`） |
| `SEG_HOST` | `127.0.0.1` | バインドアドレス。LAN アクセスには `0.0.0.0` を設定（リバースプロキシ併用推奨） |
| `PYTORCH_CUDA_ALLOC_CONF` | - | `expandable_segments:True` で VRAM 管理が改善されます（API が自動設定） |

## Docker によるデプロイ（任意）

リポジトリに `docker-compose.yml` が同梱されています（trainer API は `127.0.0.1:8002`、serving API は `127.0.0.1:8001`、UI は `127.0.0.1:5173`。ポートはループバックのみに公開されます）。

```bash
docker compose up -d
```

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
| 推奨 | M1/M2/M3/M4 | 推論は MPS アクセラレーション。学習には NVIDIA CUDA GPU が必要 |
| Intel Mac | - | CPU のみ。アノテーションと推論のみ |

**CPU のみ:** アノテーションと推論は全プラットフォームで動作します。`SEG_TORCH_DEVICE=cpu` を設定してください。学習は CPU では行えません（NVIDIA CUDA GPU が必要）。

GPU を搭載していない場合でも、推論やアノテーション補助機能（SAM など）は CPU で動作します。ただし、学習には NVIDIA CUDA GPU が必要です（CPU では学習できません）。

## 本番環境でのチェックリスト

デプロイ前に以下の項目を確認してください。

- [ ] リバースプロキシ経由で TLS（HTTPS）が有効になっている
- [ ] 認証（Basic 認証または OAuth）が設定されている
- [ ] `SEG_PROJECTS_DIR` に十分なディスク容量がある
- [ ] 定期的なバックアップスケジュールが設定されている
- [ ] ログレベルが本番環境に適切に設定されている（通常は `WARNING` 以上）
- [ ] ファイアウォールでポート 8002 への直接アクセスがブロックされている（リバースプロキシのみ許可）
