# インスタンスセグメンテーション統合 (v0.9.8) — 設計

Status: **Draft v1** (2026-07-22) — スコープ判断はプロダクトオーナーと確定済み。
English version: [docs/design_instance_segmentation_v098.md](../design_instance_segmentation_v098.md)

エビデンス: ねじカウントの PoC (2026-07-21/22) — セマンティックマスクから合成した
copy-paste データで RF-DETR-Seg Nano を fine-tune し、実写 test セットで
**厳密一致 32/32 (100%)**、**手作業のインスタンスアノテーション 0 枚**
(val segm mAP 0.844)。唯一残った誤り種別 (単一物体への重複検出) は
mask-IoU dedup で完全に抑制できることを確認済み。

## 1. ゴール / 非ゴール

**ゴール (v0.9.8)**

- 厳密カウント用インスタンスセグを学習モードとして製品化:
  セマンティックマスク → 合成インスタンスデータセット → RF-DETR-Seg fine-tune →
  インスタンス推論 + カウント → ONNX export → serving エンドポイント。
- 新たなアノテーション負担ゼロ: インスタンス GT は塗らずに合成する。

**非ゴール (v0.9.8)**

- 手動インスタンスアノテーション UI (合成ファーストで足りない実案件が出た時に再検討)。
- auto_select / モデルサーチのインスタンス対応。
- TensorRT / tflite export、CPU 学習。

## 2. 確定済みの判断

| # | 判断 | 理由 |
|---|------|------|
| D1 | **合成ファースト**: インスタンスアノテ UI は作らない | PoC が手アノテ 0 で 32/32。既存セマンティック塗り (255=未塗装規約) は無変更 |
| D2 | **rfdetr はコア lockfile に同梱** (オプション extra にしない) | オーナー判断。インストールサイズ増・pip-audit / license 対象拡大は許容 (§4.5) |
| D3 | **ONNX export + serving_api エンドポイントまで実装** | オーナー判断。R1 (export 実現性) リスクを負う (§8) |
| D4 | **合成設定は学習フォーム埋め込み** (専用タブなし) | `training_mode="instance"` 選択時のみ表示、UI をフラットに保つ |
| D5 | **RF-DETR *Seg* 系のみ** (Small 既定、Medium / Large 選択可。Nano は当初提供したが廃止 — D5a 参照) | Seg checkpoint は全サイズ Apache-2.0。検出系 XL/2XL と rfdetr の "plus" extra は PML-1.0 → 使用禁止、ban パターン追加 (§4.5) |
| D5a | **Nano を新規学習から廃止 (2026-07-23)** | 参照カウントワークロードで Nano は segm mAP50 0.92、Small 0.94 / Medium 0.99 に対し厳密カウント用途には不足。既存 Nano checkpoint の推論は維持するためクラスマッピングは残す |
| D6 | **カウント = 閾値 (val キャリブレーション) + mask-IoU dedup (0.7)** | DETR 系の重複マスクを抑制。接触しているだけの個体は mask IoU ≈ 0 で誤マージなし (PoC 実証) |

## 3. ユーザーフロー

1. 今までどおりセマンティックマスクを塗る (`PUT …/datasets/annotate/masks/{item}.png`、255=未塗装)。
2. 学習フォームのモード選択に **「インスタンス (計数)」** が追加。選ぶと合成セクション
   (枚数・個体数/枚・スタックペア確率・seed・面積バンド上書き) と、合成サンプル 2〜3 枚を
   その場で描画する **プレビュー** ボタンが現れる。
3. 開始 → 切り出し抽出 → run ディレクトリ内に COCO データセット合成 →
   RF-DETR-Seg fine-tune → `metrics.json` 書き込み。
4. Results タブ → インスタンスオーバーレイ (番号バッジ、Okabe–Ito パレット) + 枚毎カウント。
5. ONNX export → serving_api でカウント提供。

## 4. アーキテクチャ

### 4.1 segcore — 新パッケージ `segcore/instseg/`

- `compose.py` — copy-paste コンポーザ (PoC ジェネレータの移植):
  - セマンティックマスクからの切り出し抽出 (単体サイズ帯の連結成分。帯は
    blob 面積ヒストグラムから**自動推定** + 手動上書き — PoC は固定 3200–8500 px);
  - N 枚毎の inpaint による背景プレート生成;
  - ペインターズアルゴリズム合成 → 各インスタンスの可視マスクを厳密に取得;
  - **同軸スタックペア**: 切り出し毎に PCA で主軸推定、2 個を共通軸へ回転
    (180° 極性ランダム)、軸方向に接触〜僅か重なり (係数 0.80–0.97) +
    垂直ジッタで配置。PoC 唯一のミスを潰したパターン;
  - 実写 full-GT 画像 (全 blob が単体サイズ帯) の train/val 混入 (PoC 同様);
  - seed による決定論; COCO writer (rfdetr が期待する roboflow 形式)。
- `count.py` — 信頼度フィルタ + 貪欲 mask-IoU dedup (高 conf 優先、IoU > 0.7 抑制) +
  カウント。純 numpy、単体テスト対象。
- `train_rfdetr.py` — サブプロセスエントリ (Windows: `if __name__ == "__main__"`
  spawn ガード、`num_workers=0`)。run 設定 → `RFDETRSegNano/Small.train()` へ変換し、
  rfdetr のメトリクスを本製品の `metrics.json` 形式へ翻訳。

### 4.2 trainer_api

- `schemas.py:132` — `training_mode` に `"instance"` 追加。新設 `instance_synthesis`
  設定ブロック (`n_train=500`, `n_val=80`, `objects_min=4`, `objects_max=8`,
  `stack_pair_prob=0.55`, `seed`, `area_band=[auto]`)。バリデーションゲートは
  既存のモードゲートテスト (`tests/test_train_mode_gate.py`) に倣う。
- `training_launcher.py` — インスタンス分岐: セマンティック prepare をスキップし
  compose → rfdetr サブプロセス。進捗は既存 run ログチャネルで stream。
- 推論ルート (run スコープ、新設): バッチ推論で item 毎に
  `instances.json` (`{instances: [{id, conf, bbox, rle, area}], count, threshold, dedup_iou}`)、
  `overlay.png`、既存ビューア互換のセマンティック形式合成マスク PNG を生成。
  RLE は pycocotools (`rfdetr[train]` と同時に入る)。
- 学習フォーム用プレビュールート: オンデマンドで 2〜3 枚合成 (CPU、高速)。
- Export: `export_routes.py` / `training_exports.py` の流儀に沿った
  インスタンス ONNX export ルート (まず fp32/fp16、int8 はスコープ外)。

### 4.3 serving_api

- インスタンス推論 capability を新設 (export 済み ONNX の ORT セッション):
  後処理 = 閾値 + dedup + カウント、レスポンス = インスタンス (RLE) + カウント。
- serving は最小主義を維持: RLE エンコードは numpy 約 30 行で自前実装 —
  **serving に pycocotools は追加しない**。

### 4.4 trainer_ui

- 学習フォーム (`training/`): モード選択 + 条件表示の合成セクション
  (フラットな項目構成、ネスト追加なし) + プレビュー列。
- Results (`results/`): インスタンスオーバーレイ描画 (番号バッジ。Okabe–Ito
  パレット — PoC viz のパレットが既に Okabe–Ito で紫なし・色弱対応)、カウント
  チップ。`MeasurementPanel` にインスタンスモードを追加し、union-find の代わりに
  `instances.json` を読む。
- run 種別対応: セマンティック専用パネル (confidence/error ヒートマップ、
  ピクセルヒストグラム) はインスタンス run では**エラーにせず非表示**にする。
- キャンバス操作は追加しない: オーバーレイは表示のみ。ズーム/パン/キーボードの
  既存操作系は不変。

### 4.5 依存とライセンス

- `apps/trainer_api/requirements.in`: `rfdetr==1.8.*` + `[train]` extra を追加
  (export 系依存は R1 検証後)。互換性確認済み (2026-07-22):
  `torch>=2.2` (コア 2.13.* / cu128 2.11.* とも OK)、`transformers>=5.1,<6`
  (当方 pin 5.5.4)、`pydantic>=2,<3` (当方 2.12.*)。`requirements-cu128.in` も同時追加。
- 新規推移的依存のライセンス (PyPI、2026-07-22 確認): supervision MIT、
  pytorch-lightning Apache-2.0、albumentations MIT、peft Apache-2.0、
  torchmetrics Apache-2.0、pycocotools BSD-2-Clause、roboflow Apache-2.0、
  rf100vl MIT、onnxsim MIT/Apache-2.0/BSD-2、polygraphy Apache-2.0、
  faster-coco-eval は **リポジトリで要再確認 (classifiers 欠落。GitHub 上は Apache-2.0)**。
  lockfile bump コミットには全依存の
  `LICENSE: <pkg> <ver> <SPDX> confirmed at <URL>` 証跡を必須とし、
  uv 0.11.11 + `.in` ヘッダーどおりの再生成、`THIRD_PARTY_NOTICES.md` 更新を行う。
- **ban パターン**: rfdetr の "plus" extra と検出系 XL/2XL 識別子を
  `scripts/nc-vendor-patterns.txt` (dev 専用ファイル) に追加し、pre-commit と
  公開 CI で再混入をブロック。公開 repo の `NC_VENDOR_PATTERNS` 変数は
  v0.9.8 リリース時に更新 (リリースチェックリスト §1)。
- Windows 注意 (2026-07-22 実証): `onnxsim` は `==0.4.36` に pin 必須 —
  cp311 win_amd64 wheel があるのはこの系列で、新しい版は sdist ビルドが
  Windows ロングパス制限で失敗する。
- recon で見つけた掃除項目: `CONTRIBUTING.md` の依存節が torch 2.6.0 のまま
  (`requirements.in` は 2.13.*) — 併せて修正。

### 4.6 Windows 固有

- rfdetr 学習は `num_workers=0` + spawn ガード (多 worker sweep の pagefile
  問題も回避)。
- VRAM (2026-07-22 実測、RTX 3090、Nano、1 epoch、実効 batch 16):

  | batch | grad-accum | peak allocated | peak reserved | 必要 GPU |
  |-------|-----------|----------------|---------------|----------|
  | 8 | 2 | 5.75 GiB | 6.5 GiB | ≥ 8 GiB |
  | 4 | 4 | 3.29 GiB | 3.6 GiB | ≥ 5.5 GiB |
  | 2 | 8 | 2.01 GiB | 2.2 GiB | ≥ 3.5 GiB |

  自動縮退 (`instance_training._fit_batch_to_vram`): 必要 tier が
  `total_memory` に収まるまで batch を半減 (grad-accum を倍増、実効 batch
  維持)。下限は batch 2。3.5 GiB 未満はサポート外。

## 5. データ契約

- run ディレクトリ: `train_config.json`、`instseg_dataset/{train,valid}/…` (COCO)、
  rfdetr checkpoint、`metrics.json`、`instances/{item}.json` + オーバーレイ。
- `metrics.json` (インスタンス run): `segm_mAP_50_95_val`、`segm_AP50_val`、
  `AR_val`、`count_exact_val` (実写 val サブセット基準)、`best_epoch`、
  `epochs_effective`、`dataset_stats {n_synth, n_real, n_cutouts, stack_pair_ratio}`。
  `MetricsSection.tsx` は条件付きでサブセットを描画。

## 6. テスト

- 単体 (CI、CPU): コンポーザの seed 決定論; スタックペアの同軸性
  (角度差 + ギャップのアサーション); dedup (重複は抑制・隣接は保持); RLE 往復。
- API: インスタンスモードゲート、合成バリデーション、プレビュー、推論成果物の形。
- e2e: 学習フォーム spec (高速、描画+バリデーション); インスタンス学習スモークは
  既存 skip-budget 機構の `@heavy` ゲート (GPU 必須、dev 機のみ)。
- rfdetr 学習の golden 固定はしない (環境間の非決定性)。固定 tiny 合成セットに
  対するカウントレベルのアサーションで代替。

## 7. マイルストーン

| M | 内容 | 目安 |
|---|------|------|
| M1 | segcore `instseg/` 移植 (compose/count) + 単体テスト + **ONNX export スパイク (R1)** | 1–2 日 |
| M2 | trainer_api インスタンスモード end-to-end (dev 機) + VRAM 実測 | 2–3 日 |
| M3 | Results UI (オーバーレイ/カウント) + 推論成果物 + プレビュー | 2 日 |
| M4 | ONNX export + serving エンドポイント | 2 日 |
| M5 | lockfile + license 証跡 + docs/handbook + e2e spec | 1–2 日 |

## 8. リスク

- **R1 — rfdetr Seg の ONNX export: 解消済み (2026-07-22 M1 スパイク)。**
  学習済み PoC checkpoint の RF-DETR-Seg Nano を ONNX (opset 17) へ export し、
  onnxruntime CPU で実推論を確認。契約: 入力 `1×3×312×312` (export 既定。
  `shape=` で変更可 — patch-12 バックボーンに合う寸法であること)、出力
  `dets 1×100×4` (正規化 box)、`labels 1×100×2` (logits)、
  `masks 1×100×78×78` (縮小解像度の mask logits)。serving 後処理 =
  sigmoid → 信頼度閾値 → mask リサイズ → dedup → カウントで §4.3 の想定どおり。
  **M4 補遺 (2026-07-22):** 物体信頼度は `sigmoid(labels[:, 0])`
  (COCO category 1 は内部クラス index 0 にマップ)。numpy チェーン全体
  (stretch リサイズ → /255 → ImageNet 正規化 → ORT → conf ≥ thr →
  mask sigmoid → bilinear リサイズ → >0.5 → 面積 ≥ 16 → 貪欲 mask-IoU dedup)
  が SDK `model.predict` のカウントと完全一致 (PoC テストセットで GT 32/32、
  SDK 一致 32/32)。serving `/count` と export される
  `instance_inference.json` 契約はこのチェーンを符号化している。
- **R2 — roboflow SDK の挙動。** import 時のネットワーク/テレメトリ無しと
  オフラインインストール可否を検証。必要なら lazy import で隔離。
- **R3 — コア lockfile の肥大** (D2、許容済み): インストールサイズ + 監査対象増。
- **R4 — 小 VRAM GPU**: M2 の実測 + 自動 grad-accum で緩和。
- **R5 — 分離可能な剛体部品以外への合成ドメインギャップ**: 面積バンド自動推定、
  プレビュー、実写 full-GT 混入で緩和。handbook に既知の限界として明記。
