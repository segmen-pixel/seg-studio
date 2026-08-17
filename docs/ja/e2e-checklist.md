# E2E テストチェックリスト

リリース前・大きな変更後に確認するチェックリスト。

---

## 0. 自動 E2E テスト (Playwright)

手動チェックの前に、まず自動テストスイート（69 テスト）を実行する:

```bash
cd apps/trainer_ui
npx playwright test
```

- **`--reporter` フラグは付けないこと** — `playwright.config.ts` の reporter リストに
  カスタムの skip-budget レポーター（`e2e/skip-budget-reporter.ts`）が含まれており、
  CLI で `--reporter` を渡すと上書きされて skip 予算ゲートが無効になる。
- スイートは自己シード式: 初回実行時に fixture プロジェクト（`zz-e2e-seed-1/2`）と
  学習済み fixture run を自動作成する（初回のみ学習に約 2 分かかる）。
- 事前条件: API サーバーが `:8002` で起動していること。
- **⚠️ UI を変更したら先に `npm run build` すること。** Playwright の baseURL は
  `http://localhost:8002/ui`、つまり **API が配信するビルド済みバンドル**であって
  Vite dev server (`:5173`) ではない。global-setup はビルドしないので、
  ビルドを忘れると**古い UI に対して緑になる**。2026-07-25 に実害: CSS 変更が
  反映されず、効いていない値を見て「修正が足りない」と誤診した。

---

## 1. Windows 起動スクリプト (`scripts/windows/`)

### 1.1 start_local_windows.bat

```bat
REM 事前: 既存サービスを停止
scripts\windows\stop_local_windows.bat

REM ヘルプ表示
scripts\windows\start_local_windows.bat --help
```

- [ ] `--help` でヘルプが表示され、エラーなく終了する
- [ ] 正常起動: 実行後に以下を確認
  - [ ] `netstat -ano | findstr :8002 | findstr LISTENING` → PID が表示される
  - [ ] `curl -s http://127.0.0.1:8002/startup-status` → `"ready":true` が返る
  - [ ] ブラウザが自動で `http://127.0.0.1:8002/ui/` を開く
  - [ ] コンソールに `Trainer UI : http://127.0.0.1:8002/ui/` が表示される
- [ ] コンソールに「の使い方が誤っています」等のエラーが出ていないこと
- [ ] Vite dev server: `netstat -ano | findstr :5173 | findstr LISTENING` → PID が表示される
- [ ] ポート競合時: 8002 が既に使用中でも警告を出して起動する
- [ ] venv 未作成: `.venv-windows` が無い場合にエラーメッセージが出る
- [ ] npm 無し環境: npm がパスに無くても API は起動する（UI dev server はスキップ）

### 1.2 stop_local_windows.bat

- [ ] 全サービスが停止する
- [ ] 停止後に `netstat -ano | findstr :8002 | findstr LISTENING` → 結果なし

### 1.3 install_windows.bat

- [ ] クリーンな環境（.venv-windows なし）で実行してエラーなく完了する

---

## 1b. macOS 起動スクリプト (`scripts/macos/`)

### 1b.1 start_local_macos.sh

```bash
# 事前: 既存サービスを停止
bash scripts/macos/stop_local_macos.sh

# ヘルプ表示
bash scripts/macos/start_local_macos.sh --help
```

- [ ] `--help` でヘルプが表示され、エラーなく終了する
- [ ] 正常起動: 実行後に以下を確認
  - [ ] `lsof -iTCP:8002 -sTCP:LISTEN` → PID が表示される
  - [ ] `curl -s http://127.0.0.1:8002/startup-status` → `"ready":true` が返る
  - [ ] ブラウザが自動で開く
  - [ ] コンソールに `Trainer UI : http://127.0.0.1:8002/ui/` が表示される
- [ ] ポート競合時: 8002 が既に使用中の場合に警告が出る
- [ ] venv 未作成: `.venv-macos` が無い場合にエラーメッセージが出る
- [ ] npm 無し環境: npm がパスに無くても API は起動する（UI dev server はスキップ）

### 1b.2 stop_local_macos.sh

- [ ] 全サービスが停止する
- [ ] 停止後に `lsof -iTCP:8002 -sTCP:LISTEN` → 結果なし

### 1b.3 install_macos.sh

- [ ] クリーンな環境（.venv-macos なし）で実行してエラーなく完了する
- [ ] Apple Silicon: MPS available と表示される
- [ ] coremltools がインストールされる

---

## 2. API 起動

- [ ] `http://localhost:8002/docs` → Swagger UI が表示される
- [ ] `http://localhost:8002/api/v1/health` → JSON レスポンス（disk/RAM/GPU 情報）
- [ ] `http://localhost:8002/ui/` → Trainer UI が表示される

---

## 3. Annotate タブ

### 3.1 画像リスト操作
- [ ] クリック → その画像を選択+表示（他の選択解除）
- [ ] Ctrl+クリック → 個別トグル（画像は表示）
- [ ] Shift+クリック → 範囲選択（アンカーから）
- [ ] ArrowDown/Up → 上下の画像に移動+表示（単一選択）、スクロール追従
- [ ] Shift+ArrowDown/Up → 上下に移動+選択範囲拡張
- [ ] 選択状態が背景色で表示される（チェックボックスではない）
- [ ] 画像ドラッグ&ドロップでアップロード
- [ ] ZIP ドラッグ&ドロップでインポート

### 3.2 描画ツール
- [ ] Brush (B): 描画、[ ] でサイズ縮小、] で拡大
- [ ] Eraser (E): 消去
- [ ] Bucket (G): 塗りつぶし
- [ ] Wand (W): マジックワンド選択
- [ ] SAM (S): クリックセグメンテーション、Enter 確定、Esc キャンセル
- [ ] SAM Box (X): ボックスセグメンテーション
- [ ] Spot Detect (D): スポット検出、Enter 確定、Esc キャンセル
- [ ] Superpixel (P): スーパーピクセル
- [ ] Crack Trace (C): クラック追跡
- [ ] Measure (M): 計測
- [ ] Move (V): 画像の移動

### 3.3 編集操作
- [ ] Ctrl+Z → Undo
- [ ] Ctrl+Y → Redo
- [ ] Ctrl+S → 保存
- [ ] Delete/Backspace → クリア
- [ ] 1-9 → クラス切替

---

## 4. Training タブ

- [ ] アーキテクチャ選択（SimpleUNet / STDC）
- [ ] 学習開始 → ログがリアルタイム表示
- [ ] 学習中の run に青パルスドット表示
- [ ] Score タブ → 推論結果サマリ + クラス別棒グラフ

---

## 5. CLI 学習

```bash
python scripts/cli_train.py --project <ID> --device cuda:0 --no-crop --epochs 5
```

- [ ] デフォルトパラメータが推奨値と一致
- [ ] 学習が正常に開始・終了する
- [ ] スライディングウィンドウ評価が自動有効化される

---

## 6. Result タブ

- [ ] Training の chart アイコンから Result タブが開く
- [ ] Infer All → 全画像推論
- [ ] ヒートマップ表示（confidence / class overlay）
- [ ] プロジェクト切替で Result タブが復元される

---

## 7. 個数カウント (インスタンス)

セマンティック学習が済んだプロジェクトで行う。カウント学習は、そこで描いた
マスクを素材に合成データを作るため、専用のアノテーションは要らない。

- [ ] Training タブでカウント学習を開始できる (対象クラスが 4 枚以上必要)
- [ ] クラスが 4 枚未満のとき、GPU を掴む前にエラーで止まる
- [ ] 合成サンプルのプレビューが表示され、実シーンに似ている
- [ ] ログに `n_cutouts` と面積帯による除外数が出る
- [ ] 学習完了後、`instance_inference.json` に `patch_size` が入っている
      (**学習と推論のパッチサイズ一致の要。欠けていると推論が黙って
      全画面リサイズに落ちる**)
- [ ] Result タブで検出ハイライトを ON にすると個体ごとに色が変わる
- [ ] クラス別の個数が表示される
- [ ] 画像より大きいパッチサイズのとき、タイル分割されず 1 パスになる

### カウント API

```bash
curl -F image=@sample.jpg http://localhost:8002/count
```

- [ ] `count` / `counts_by_class` / `instances` が返る
- [ ] 各個体に `bbox` / `centroid` / `area` / `rle` が入っている
- [ ] `threshold` / `dedup_iou` がレスポンスに含まれる
- [ ] セマンティックモデルを有効化した状態では 409 が返る
- [ ] 200 個を超える画像で `truncation_warning` が出る

---

## バッチファイル変更時の注意

Windows バッチファイル (`*.bat`) を編集する際の既知の問題:

1. **`goto` は `()` ブロック内で使えない** — `if/else` の代わりに `goto :label` を使う
2. **`echo` 内の `)` はブロック終端と解釈される** — `^)` でエスケープするか、`goto` パターンでブロック外に出す
   - 例: `echo port 8081)...` → `)` がブロック終端、`...` がコマンド扱い → 「の使い方が誤っています」エラー
   - 修正: `echo port 8081^)...` or `if not "..."=="..." goto :skip` パターン
3. **`start` コマンドの引用符** — 最初の引用符はウィンドウタイトル扱い、空文字列 `""` を入れる
4. **`cd /d X && command >> log 2>&1` は `cmd /c` 内で構文エラーになる場合がある** — `npm --prefix` 等で `cd` を避ける
5. **変更後は必ず `--help` と実起動の両方をテストする**
6. **コンソール出力に「の使い方が誤っています」が出たら**: `()` ブロック内の `)` エスケープ漏れを疑う

### bat 変更時のテスト手順

```bat
REM 1. ヘルプが通るか
scripts\windows\start_local_windows.bat --help

REM 2. 既存サービスを停止してクリーンに起動
scripts\windows\stop_local_windows.bat
scripts\windows\start_local_windows.bat

REM 3. 起動確認（全ポート + エラーなし）
netstat -ano | findstr :8002 | findstr LISTENING
netstat -ano | findstr :5173 | findstr LISTENING
curl -s http://127.0.0.1:8002/startup-status

REM 4. コンソール出力にエラーがないか確認
REM    NG: 「の使い方が誤っています」→ ()ブロック内の ) エスケープ漏れ
REM    NG: 「プロセスはファイルにアクセスできません」→ ログファイルロック

REM 5. ログにエラーがないか
type logs\windows\trainer.log
type logs\windows\ui_dev.log
```
