# Troubleshooting

## 起動しない / API 接続不可

**症状:** ブラウザに「Connecting to API server...」が表示され続ける

1. サービス状態を確認:
   - Windows: `scripts\windows\status_windows.bat`
   - macOS: `lsof -iTCP:8002 -sTCP:LISTEN`
2. ログファイルを確認:
   - Windows: `logs\windows\trainer_<タイムスタンプ>.log`（起動ごとに 1 本。`scripts\windows\status_windows.bat` が最新を表示します）
   - macOS: `logs/macos/trainer.log`
3. ポート 8002 が別プロセスで使用されていないか確認:
   ```bash
   # Windows
   netstat -ano | findstr :8002
   # macOS
   lsof -i :8002
   ```
4. Python 仮想環境が正しくアクティベートされているか確認

## ブラウザが真っ白

**症状:** `http://localhost:8002/ui/` にアクセスしても白い画面

1. UI がビルドされているか確認: `apps\trainer_ui\dist\` フォルダが存在するか
2. ビルドされていない場合:
   ```bat
   cd apps\trainer_ui
   npm install
   npm run build
   ```
3. 開発サーバー経由の場合は `http://localhost:5173` にアクセス

## OOM (Out of Memory) エラー

**症状:** 学習中に CUDA out of memory エラー

1. **パッチサイズを下げる:** 学習タブの設定で小さくする (例: 256 → 192 → 128)
2. **バッチサイズを下げる**（デフォルト 8。4 → 2 → 1 と試す）
3. **他のGPU使用アプリを終了:** ブラウザの GPU アクセラレーションを無効化
4. **4GB VRAM GPU の場合:** 自動的に low-VRAM モードが適用されますが、
   それでもOOMになる場合はパッチサイズを 128 にしてください
5. Windows/Linux: 環境変数 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` が設定されていることを確認
   （デフォルトで設定済み）
6. **macOS (MPS):** ユニファイドメモリを使用するため、他のアプリのメモリ使用量も影響します。
   メモリ不足の場合は入力サイズを下げるか、不要なアプリを閉じてください

## CUDA が認識されない（Windows / Linux）

**症状:** GPU が利用可能なのに CPU しか選択肢に出ない

1. NVIDIA ドライバが最新か確認: `nvidia-smi`
2. CUDA 対応 PyTorch がインストールされているか確認:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
3. `False` の場合、CUDA 版 PyTorch を再インストール:
   ```bash
   # 通常は cu128 (Turing/RTX 20xx 以降、Blackwell 含む)。
   # Maxwell/Pascal/Volta の古い GPU のみ cu124
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
   Windows では `install-windows.bat cuda` (古い GPU は
   `install-windows.bat cuda124`) を実行し直しても構いません。
4. 学習タブの **デバイス** セレクタで手動選択

## MPS が認識されない（macOS）

**症状:** Apple Silicon Mac なのに MPS デバイスが選択肢に出ない

1. macOS 12 (Monterey) 以上か確認
2. MPS 対応 PyTorch がインストールされているか確認:
   ```bash
   python3 -c "import torch; print(torch.backends.mps.is_available())"
   ```
3. `False` の場合、PyTorch を再インストール:
   ```bash
   pip install --upgrade torch torchvision
   ```
4. Intel Mac の場合、MPS は利用不可（CPU のみ）

## SAM アシストが動かない

**症状:** SAM クリック / ボックスのセグメンテーションでエラー

1. チェックポイントファイルが存在するか確認: `models\sam_checkpoints\`
2. 必要なファイル: `mobile_sam.pt`, `sam2.1_hiera_tiny.pt` など
3. `/api/v1/sam/models` エンドポイントで `checkpoint_exists: true` を確認
4. ログで具体的なエラーを確認: `logs\trainer_errors.log`

## エクスポートエラー

**症状:** CoreML / ONNX エクスポートが失敗する

1. **CoreML:** `coremltools` がインストールされているか確認（macOS 推奨）
   ```bash
   pip install coremltools
   ```
2. **ONNX:** `onnx` と `onnxruntime` がインストールされているか確認
3. 学習が正常に完了したランを選択しているか確認（status: completed）

## カウント: 学習が始まらない

**症状:** 個数カウント学習が即座に「instance mode needs at least 4 annotated
images containing one of classes [...]」で失敗する

1. **マスクの枚数ではなく、クラスが実際に入っている枚数**を数えてください。
   値 255 で塗られたマスクは「未塗り」であって前景ではなく、1 枚にも数えません。
   ラベルタブでいくつか開き、クラスの色が本当に載っているか確認してください。
2. **Mark Clean 画像は数に入りません** — 定義上すべて背景だからです。
3. このチェックは GPU を使う前に走るので、数秒で失敗します。メッセージには
   探したクラスが出ます。想定と違う場合は、クラスパネルでどれが
   **アクティブ** になっているか確認してください。

## カウント: 個数が合わない

**症状:** 報告される個数が常に少ない、または多い

1. **まずオーバーレイを見てください。** 結果タブの **検出ハイライト** を ON に
   すると背景が暗くなり、個体ごとに色が変わります。接触した 2 個が同じ色なら
   統合されており、1 個が 2 色なら分割されています。
2. **常に約 2 倍多い:** タイルごとに重複カウントされています。書き出した契約に
   `patch_size` があるか確認してください。タイル学習したモデルをタイルなしで
   推論した (またはその逆) 場合、**エラーなく**個数だけ狂います。
3. **少なく、かつキリの良い数で頭打ち:** モデルサイズごとの 1 枚あたりの検出
   上限 (既定の Small は 100、Medium/Large は 200) に達している可能性が
   あります。`truncation_warning` はタイルオフ (`instance_patch_size = 0`) で
   書き出したモデルにのみ付き、既定の 768 タイルでは出ません。個数そのものから
   飽和を判断し、切り出し範囲を狭めるか、フレームを分割してください。
4. **数個だけ少ない:** タイルのオーバーラップより大きい物体が継ぎ目にまたがって
   落ちている可能性があります。既定の 768 パッチではオーバーラップは 192px です。
   部品がそれより大きい場合は `instance_patch_size` を上げてください。
5. **閾値を確認してください。** `/count` は使用した `threshold` を返します。
   学習時に hold-out 画像で校正した値なので、かなり違うシーンでは別の値が
   適切なことがあります。

## カウント: 合成サンプルが実際と違う

**症状:** プレビューに出る合成画像が、実際のシーンに似ていない

1. **物体が大きすぎる / 小さすぎる:** 単体面積帯を `0` (自動) ではなく明示的に
   設定してください。自動は blob から推定するため、小さなゴミや極端に大きい
   blob が混じると歪みます。
2. **切り抜きが少なすぎる:** 合成ログに `n_cutouts` と、面積帯で除外された
   blob 数が出ます。ほとんど除外されているなら、面積帯がデータに合っていません。
3. **背景が単調:** 背景プレートはアノテーション済み画像からしか作られません。
   シーンの別の場所を数枚アノテーションしてください。

## カウント: `POST /count` が 409 を返す

**症状:** `the active model is a semantic-segmentation export; use /segment`

serving レジストリで有効化されているモデルが、カウント用ではなくセマンティック用
です。カウント学習を **エクスポート → ONNX (Serving)** で書き出し、その model id
を有効化してください。カウント用の書き出しには `model.onnx` と一緒に
`instance_inference.json` が入ります。セマンティック用には入りません。

## ログの確認方法

API 自身がローテーション付きのログファイルを書き出し、それとは別に起動スクリプトがコンソール出力を保存します。どちらも役に立ちますが、同じファイルではありません。

| 種類 | 場所 |
|---|---|
| API のログすべて | リポジトリルートの `logs\app.log`（20MB × 10 世代） |
| 警告・エラーのみ | リポジトリルートの `logs\trainer_errors.log`（10MB × 5 世代） |
| インストーラ版 Windows での上記 2 つ | `%LOCALAPPDATA%\Seg-Studio\logs\` |
| コンソール出力（Windows 起動スクリプト） | `logs\windows\trainer_<タイムスタンプ>.log` と `serving_<タイムスタンプ>.log`（起動ごとに 1 組） |
| コンソール出力（macOS 起動スクリプト） | `logs/macos/trainer.log`、`logs/macos/serving.log` |
| コンソール出力（`scripts/start_local.sh`） | `/tmp/seg_trainer.log`、`/tmp/seg_serving.log` |
| 学習ラン 1 回分 | `projects/<project_id>/training/runs/<run_id>/train.log` |

- **JSON ログ:** 環境変数 `LOG_FORMAT=json` で JSON 形式出力
- **ログレベル変更:** 環境変数 `LOG_LEVEL=DEBUG` でデバッグ出力

## 依存関係の監査

セキュリティ脆弱性のチェック:
```bash
# Windows
scripts\audit.bat
# macOS / Linux
bash scripts/audit.sh
```

## 問題を報告する

ここまでで解決しない場合は、[バグ報告テンプレート](../../.github/ISSUE_TEMPLATE/bug_report.md) を使って <https://github.com/segmen-pixel/seg-studio/issues/new/choose> から issue を作成してください。

**含めていただきたい情報:**

- **Seg-Studio のバージョン** — アプリ上部のヘッダーに表示されます。API のレスポンスヘッダ `X-API-Version` でも確認できます
- **OS とバージョン** — 例: Windows 11 23H2、macOS 14.5
- **GPU とドライバ** — `nvidia-smi` の出力、または「Apple Silicon M2」「CPU のみ」など
- **起動方法** — `start-windows.bat` / `bash start-macos.sh` / `scripts\windows\start_local_windows.bat` / `docker compose up` など
- **インストール方法** — リリース ZIP、git clone、Windows インストーラのいずれか。あわせて `python --version`
- **エラーメッセージの全文** — 画面写真ではなくテキストでコピーしてください
- **ログの末尾 50 行程度** — まず `trainer_errors.log`、次に `app.log`（場所は「ログの確認方法」を参照）:

```bash
# Windows (PowerShell)
Get-Content logs\trainer_errors.log -Tail 50
# macOS / Linux
tail -n 50 logs/trainer_errors.log
```

**含めないでください:**

- `.env`、`SEG_API_TOKEN`、`projects/runtime_settings.json` の `api_token` の値 — 認証情報です。issue は公開されます
- 顧客データや本番の画像・マスク・書き出したモデル — トリミングやマスキングをするか、公開できる画像で再現してから貼ってください
- 顧客や現場が特定できるプロジェクト名・ファイルパス・ホスト名・ユーザー名 — 貼る前に置き換えてください
- 個人情報全般

ログにはファイルパスやプロジェクト名が含まれます。貼り付ける前に必ず目を通してください。**セキュリティ上の脆弱性** は公開 issue ではなく、[SECURITY.md](../../SECURITY.md) の手順で報告してください。
