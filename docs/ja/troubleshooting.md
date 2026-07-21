# Troubleshooting

## 起動しない / API 接続不可

**症状:** ブラウザに「Connecting to API server...」が表示され続ける

1. サービス状態を確認:
   - Windows: `scripts\windows\status_windows.bat`
   - macOS: `lsof -iTCP:8002 -sTCP:LISTEN`
2. ログファイルを確認:
   - Windows: `logs\windows\trainer.log`
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
   # 通常は cu124、RTX 50 系 (Blackwell) は cu128 を使用
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
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

## ログの確認方法

- **API サーバーログ:** コンソール出力（タイムスタンプ付き）
- **エラーログ:**
  - Windows: `logs\trainer_errors.log`
  - macOS: `logs/macos/trainer.log`
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
