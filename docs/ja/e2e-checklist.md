# E2E テストチェックリスト

リリースの前と、大きな変更を加えた後に確認する項目の一覧です。

---

## 0. 自動 E2E テスト（Playwright）

手動で確認する前に、まず自動テストスイートを実行します。

```bash
cd apps/trainer_ui
npx playwright test
```

- **`--reporter` フラグは付けないでください**。`playwright.config.ts` の reporter リストには、独自の skip-budget レポーター（`e2e/skip-budget-reporter.ts`）が入っています。CLI で `--reporter` を渡すとこのリストが上書きされ、skip 予算のゲートが効かなくなります。
- テスト用のデータは自動で用意されます。初回の実行時に、fixture のプロジェクト（`zz-e2e-seed-1/2`）と学習済みの fixture run が自動で作成されます（学習に約2分かかるのは初回だけです）。
- 事前に API サーバーを `:8002` で起動しておく必要があります。
- **⚠️ UI を変更したら、先に `npm run build` を実行してください**。Playwright の baseURL は `http://localhost:8002/ui` です。つまりテストの対象は **API が配信するビルド済みのバンドル**で、Vite の dev server（`:5173`）ではありません。global-setup はビルドを行わないため、ビルドを忘れると**古い UI に対してテストが通ってしまいます**（反映されていない CSS を見て、修正が足りないと誤って判断しがちです）。

---

## 1. Windows の起動スクリプト（`scripts/windows/`）

### 1.1 start_local_windows.bat

```bat
REM 事前: 既存サービスを停止
scripts\windows\stop_local_windows.bat

REM ヘルプ表示
scripts\windows\start_local_windows.bat --help
```

- [ ] `--help` でヘルプが表示され、エラーなく終了すること
- [ ] 正常に起動すること（実行後に次の点を確認）
  - [ ] `netstat -ano | findstr :8002 | findstr LISTENING` → PID が表示されること
  - [ ] `curl -s http://127.0.0.1:8002/startup-status` → `"ready":true` が返ること
  - [ ] ブラウザで `http://127.0.0.1:8002/ui/` が自動で開くこと
  - [ ] コンソールに `Trainer UI : http://127.0.0.1:8002/ui/` が表示されること
- [ ] コンソールに「の使い方が誤っています」などのエラーが出ていないこと
- [ ] Vite の dev server：`netstat -ano | findstr :5173 | findstr LISTENING` → PID が表示されること
- [ ] ポートの競合：8002 がすでに使われていても、警告を出して起動すること
- [ ] venv が未作成：`.venv-windows` が無いときにエラーメッセージが出ること
- [ ] npm が無い環境：npm がパスに無くても API が起動すること（UI の dev server は起動しない）

### 1.2 stop_local_windows.bat

- [ ] すべてのサービスが停止すること
- [ ] 停止後に `netstat -ano | findstr :8002 | findstr LISTENING` → 何も表示されないこと

### 1.3 install_windows.bat

- [ ] クリーンな環境（.venv-windows なし）で実行し、エラーなく完了すること

---

## 1b. macOS の起動スクリプト（`scripts/macos/`）

### 1b.1 start_local_macos.sh

```bash
# 事前: 既存サービスを停止
bash scripts/macos/stop_local_macos.sh

# ヘルプ表示
bash scripts/macos/start_local_macos.sh --help
```

- [ ] `--help` でヘルプが表示され、エラーなく終了すること
- [ ] 正常に起動すること（実行後に次の点を確認）
  - [ ] `lsof -iTCP:8002 -sTCP:LISTEN` → PID が表示されること
  - [ ] `curl -s http://127.0.0.1:8002/startup-status` → `"ready":true` が返ること
  - [ ] ブラウザが自動で開くこと
  - [ ] コンソールに `Trainer UI : http://127.0.0.1:8002/ui/` が表示されること
- [ ] ポートの競合：8002 がすでに使われているときに警告が出ること
- [ ] venv が未作成：`.venv-macos` が無いときにエラーメッセージが出ること
- [ ] npm が無い環境：npm がパスに無くても API が起動すること（UI の dev server は起動しない）

### 1b.2 stop_local_macos.sh

- [ ] すべてのサービスが停止すること
- [ ] 停止後に `lsof -iTCP:8002 -sTCP:LISTEN` → 何も表示されないこと

### 1b.3 install_macos.sh

- [ ] クリーンな環境（.venv-macos なし）で実行し、エラーなく完了すること
- [ ] Apple Silicon：MPS available と表示されること
- [ ] coremltools がインストールされること

---

## 2. API の起動

- [ ] `http://localhost:8002/docs` → Swagger UI が表示されること
- [ ] `http://localhost:8002/api/v1/health` → JSON が返ること（disk / RAM / GPU の情報）
- [ ] `http://localhost:8002/ui/` → Trainer UI が表示されること

---

## 3. ラベルタブ

### 3.1 画像リストの操作
- [ ] クリック → その画像を選択して表示（ほかの選択は解除）
- [ ] Ctrl+クリック → 1枚ずつ選択を切り替え（画像は表示）
- [ ] Shift+クリック → アンカーからの範囲選択
- [ ] ArrowDown/Up → 下 / 上の画像に移動して表示（単一選択）、スクロールも追従
- [ ] Shift+ArrowDown/Up → 下 / 上に移動しながら選択範囲を拡張
- [ ] 選択状態は背景色で表示（チェックボックスではない）
- [ ] 画像のドラッグ&ドロップ → アップロード
- [ ] ZIP のドラッグ&ドロップ → インポート

### 3.2 描画ツール
- [ ] ブラシ（B）：描画。`[` でサイズを縮小、`]` で拡大
- [ ] 矩形（R）：ドラッグした矩形を選択中のクラスで塗りつぶし
- [ ] 消しゴム（E）：消去
- [ ] 塗りつぶし（G）：領域の塗りつぶし
- [ ] SAM（S）：クリックはすべて対象の指定（右クリックは使わない）。粒度ボタン（subpart / part / whole）。プレビュー内のクリックまたは Enter で確定、I で反転、Esc でキャンセル
- [ ] SAM ボックス（X）：ボックスによるセグメンテーション
- [ ] スポット検出（D）：見本を塗って検出。しきい値スライダー（1〜60、大きいほど検出が少ない）。Enter で確定、Esc でキャンセル
- [ ] ひび割れ検出（C）：ひび割れの追跡
- [ ] 計測（M）：計測
- [ ] 移動（V）：マスク領域の移動

### 3.3 編集操作
- [ ] Ctrl+Z → 元に戻す
- [ ] Ctrl+Y → やり直し
- [ ] Ctrl+S → 保存
- [ ] Delete/Backspace → クリア
- [ ] 1〜9 → クラスの切り替え

### 3.4 AI サポートパネル
- [ ] 何も設定していなくても、ツール列に **AI** ボタンが表示され、パネルにモデルサーバーの種類の一覧（既定は Ollama）とモデルの一覧が出ること
- [ ] モデルサーバーが動いていないとき、問い合わせ先のアドレスと理由がパネルに出ること
- [ ] 種類の一覧で別の種類を選ぶと、モデルの一覧がそのサーバーのものに変わること
- [ ] `vlm` ブロックにアドレスかキーを書く（または `SEG_VLM_BACKEND` などを設定する）と、種類の一覧の代わりにそのサーバーが表示されること
- [ ] fastmcp が無い環境で実行すると、導入コマンドを示すエラーが出ること
- [ ] **実行**を押すと会話、ツール呼び出し、画像が順に表示され、書き込まれた画像がキャンバスに反映されること
- [ ] **一時停止** → **再開**、**停止**が、ステップの区切りで効くこと
- [ ] 人が描いたマスクと、人が OK を付けた画像が置き換えられないこと
- [ ] ページを再読み込みしても会話が残り、実行中であれば表示がそのまま追従すること

---

## 4. 学習タブ

- [ ] **アーキテクチャ**（SimpleUNet / STDC）を選べること
- [ ] **学習開始**を押すと、ログがリアルタイムで表示されること
- [ ] 学習中の run に、青く点滅するドットが表示されること
- [ ] **スコア**タブに、推論結果のサマリーとクラス別の棒グラフが表示されること

---

## 5. CLI での学習

```bash
python scripts/cli_train.py --project <ID> --device cuda:0 --epochs 5
```

- [ ] 既定のパラメータが推奨値と一致していること
- [ ] 学習が正常に始まり、終わること
- [ ] スライディングウィンドウ評価が自動で有効になること

---

## 6. 結果タブ

- [ ] 学習タブのグラフのアイコンから結果タブが開くこと
- [ ] **推論実行**で全画像が推論されること
- [ ] ヒートマップ（confidence / class overlay）が表示されること
- [ ] プロジェクトを切り替えたときに結果タブが復元されること

---

## 7. 個数カウント（インスタンス）

セマンティックの学習が済んだプロジェクトで確認します。個数カウントの学習では、そのプロジェクトで描いたマスクを素材にして合成データを作るため、専用のアノテーションは必要ありません。

- [ ] 学習タブで**個数カウント**の学習を開始できること（対象クラスの画像が4枚以上必要）
- [ ] 対象クラスの画像が4枚未満のとき、GPU を確保する前にエラーで止まること
- [ ] **合成プレビュー**に合成サンプルが表示され、実際のシーンに似ていること
- [ ] ログに `n_cutouts` と、面積帯によって除外された数が出ること
- [ ] 学習完了後、`instance_inference.json` に `patch_size` が入っていること（**学習と推論でパッチサイズを一致させるための要の値です。欠けていると、推論は何も知らせずに画像全体をリサイズする方式に切り替わります**）
- [ ] 結果タブで検出ハイライトを ON にすると、個体ごとに色が変わること
- [ ] クラスごとの個数が表示されること
- [ ] パッチサイズが画像より大きいとき、タイルに分割されず1パスで処理されること

### カウント API

```bash
curl -F image=@sample.jpg http://localhost:8001/count
```

- [ ] `count` / `counts_by_class` / `instances` が返ること
- [ ] 各個体に `bbox` / `centroid` / `area` / `rle` が入っていること
- [ ] `threshold` / `dedup_iou` がレスポンスに含まれること
- [ ] セマンティックのモデルを有効にした状態では 409 が返ること
- [ ] タイル分割なし（`instance_patch_size = 0`）のモデルで、個数が検出数の上限の90%以上になる画像に `truncation_warning` が出ること

---

## バッチファイル変更時の注意

Windows のバッチファイル（`*.bat`）を編集するときに起きやすい、既知の問題です。

1. **`goto` は `()` ブロックの中では使えません**。`if/else` の代わりに `goto :label` を使います。
2. **`echo` の中の `)` はブロックの終わりと解釈されます**。`^)` でエスケープするか、`goto` を使うパターンでブロックの外に出します。
   - 例：`echo port 8081)...` と書くと、`)` でブロックが終わり、`...` がコマンドとして扱われるため、「の使い方が誤っています」というエラーになります。
   - 修正：`echo port 8081^)...` と書くか、`if not "..."=="..." goto :skip` のパターンを使います。
3. **`start` コマンドの引用符**：最初の引用符で囲んだ部分はウィンドウタイトルとして扱われるので、空文字列 `""` を入れます。
4. **`cd /d X && command >> log 2>&1` は `cmd /c` の中で構文エラーになることがあります**。`npm --prefix` などを使って `cd` を避けます。
5. **変更後は、必ず `--help` と実際の起動の両方をテストしてください**。
6. **コンソール出力に「の使い方が誤っています」が出たら**、`()` ブロックの中の `)` のエスケープ漏れを疑ってください。

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
