# インスタンスセグメンテーション統合（v0.9.8）の設計

状態：**Draft v1**（2026-07-22）。

英語版：[docs/design_instance_segmentation_v098.md](../design_instance_segmentation_v098.md)

根拠：ねじの個数カウントで行った PoC（2026-07-21/22）。セマンティックマスクから合成した copy-paste データで RF-DETR-Seg Nano を fine-tune したところ、実写のテストセットで**個数の完全一致 32/32（100%）**、**手作業のインスタンスアノテーションは0枚**だった（val segm mAP 0.844）。唯一残った誤りの種類（1つの物体を重複して検出する）は、mask-IoU dedup で完全に抑えられることを確認済み。

## 1. 目標と対象外

**目標（v0.9.8）**

- 個数を正確に数えるためのインスタンスセグメンテーションを、正式な学習モードとして製品に組み込む。流れは、セマンティックマスク → 合成インスタンスデータセット → RF-DETR-Seg の fine-tune → インスタンス推論とカウント → ONNX export → serving エンドポイント。
- アノテーションの手間を新たに増やさない。インスタンスの正解（GT）は塗らずに合成する。

**対象外（v0.9.8）**

- 手作業でインスタンスをアノテーションする UI。合成ファーストで対応できない実案件が出たときに改めて検討する。
- auto_select とモデルサーチのインスタンス対応。
- TensorRT / tflite への export、CPU での学習。

## 2. 確定済みの判断

| # | 判断 | 理由 |
|---|------|------|
| D1 | **合成ファースト**：インスタンスをアノテーションする UI は作らない | PoC では手作業のアノテーション0枚で 32/32。既存のセマンティックの塗り（255 = 未塗装の規約）は変更しない |
| D2 | **rfdetr はコアの lockfile に含める**（オプションの extra にはしない） | オーナーの判断。インストールサイズの増加と、pip-audit / ライセンス確認の対象拡大は許容する（§4.5） |
| D3 | **ONNX export と serving_api のエンドポイントまで実装する** | オーナーの判断。R1（export が実現できるか）のリスクを負う（§8） |
| D4 | **合成の設定は学習フォームに埋め込む**（専用のタブは作らない） | `training_mode="instance"` を選んだときだけ表示し、UI をフラットに保つ |
| D5 | **RF-DETR の *Seg* 系のみ**（既定は Small、Medium / Large も選択可。Nano は当初提供したが廃止。D5a を参照） | Seg の checkpoint はどのサイズも Apache-2.0。検出系の XL/2XL と rfdetr の "plus" extra は PML-1.0 なので使用禁止とし、ban パターンを追加する（§4.5） |
| D5a | **新規の学習では Nano を廃止（2026-07-23）** | 参照用のカウントのワークロードで、Nano の segm mAP50 は 0.92 にとどまり、Small の 0.94、Medium の 0.99 に比べて厳密なカウントには不十分。既存の Nano checkpoint での推論は続けられるよう、クラスのマッピングは残す |
| D6 | **カウント = しきい値（val でキャリブレーション）+ mask-IoU dedup（0.7）** | DETR 系で起きる重複マスクを抑える。接しているだけの個体どうしは mask IoU ≈ 0 なので、誤って統合されない（PoC で確認済み） |

## 3. ユーザーフロー

1. これまでどおりセマンティックマスクを塗る（`PUT …/datasets/annotate/masks/{item}.png`、255 = 未塗装）。
2. 学習フォームのモード選択に**個数カウント**が加わる。選ぶと、合成のセクション（枚数、1枚あたりの個体数、スタックペアの確率、seed、面積帯の上書き）と、合成サンプルを2〜3枚その場で描画する**合成プレビュー**（**生成**ボタン）が表示される。
3. 開始すると、切り出しの抽出、run ディレクトリ内での COCO データセットの合成、RF-DETR-Seg の fine-tune、`metrics.json` の書き込みが順に行われる。
4. 結果タブで、インスタンスのオーバーレイ（番号バッジ、Okabe–Ito パレット）と画像ごとの個数を表示する。
5. ONNX に export し、serving_api で個数を返す。

## 4. アーキテクチャ

### 4.1 segcore — 新パッケージ `segcore/instseg/`

- `compose.py` — copy-paste のコンポーザ。PoC のジェネレータを移植したもので、次の処理を行う。
  - セマンティックマスクから切り出しを抽出する。対象は単体の面積帯に入る連結成分。面積帯は blob の面積ヒストグラムから**自動推定**し、手動で上書きもできる（PoC では 3200–8500 px に固定していた）。
  - N 枚ごとに inpaint で背景プレートを作る。
  - ペインターズアルゴリズムで合成し、各インスタンスの見えている部分のマスクを正確に得る。
  - **同軸スタックペア**：切り出しごとに PCA で主軸を推定し、2個を共通の軸にそろえて回転させる（180° の向きはランダム）。軸方向には、接する程度からわずかに重なる程度（係数 0.80–0.97）の位置に置き、軸に垂直な方向に小さなジッタを加える。PoC で唯一の誤りをなくしたのがこのパターンである。
  - 実写の full-GT 画像（すべての blob が単体の面積帯に入るもの）を、PoC と同様に train/val に混ぜる。
  - seed を固定すれば決定論的に動く。COCO writer は、rfdetr が想定する roboflow 形式で書き出す。
- `count.py` — 確信度によるフィルタ、貪欲法による mask-IoU dedup（conf の高いほうを残し、IoU > 0.7 を抑制）、カウントを行う。numpy だけで実装し、単体テストの対象とする。
- `train_rfdetr.py` — サブプロセスのエントリポイント（Windows 向けに `if __name__ == "__main__"` の spawn ガードと `num_workers=0`）。run の設定を `RFDETRSegNano/Small.train()` 用に変換し、rfdetr のメトリクスを本製品の `metrics.json` の形式に変換する。

### 4.2 trainer_api

- `schemas.py:132` — `training_mode` に `"instance"` を追加する。任意の設定ブロック `instance_synthesis`（`n_train=500`、`n_val=80`、`objects_min=4`、`objects_max=8`、`stack_pair_prob=0.55`、`seed`、`area_band=[auto]`）を新設する。バリデーションのゲートは、既存のモードゲートのテスト（`tests/test_train_mode_gate.py`）にならう。
- `training_launcher.py` — インスタンス用の分岐を設ける。セマンティックの prepare は行わず、compose → rfdetr のサブプロセスの順に実行する。進捗は既存の run ログのチャネルで配信する。`train_config.json` はこれまでどおり書き出す。
- 推論ルート（新設、run 単位）：バッチ推論で item ごとに `instances.json`（`{instances: [{id, conf, bbox, rle, area}], count, threshold, dedup_iou}`）、`overlay.png`、既存のビューアでそのまま表示できるセマンティック形式の合成マスク PNG を生成する。RLE には pycocotools を使う（`rfdetr[train]` と一緒に入る）。
- 学習フォーム用のプレビュールート：要求に応じて2〜3枚を合成する（CPU で高速に動く）。
- Export：`export_routes.py` / `training_exports.py` の流儀に沿って、インスタンス用の ONNX export ルートを設ける（まず fp32/fp16 に対応し、int8 はスコープ外）。

### 4.3 serving_api

- インスタンス推論の capability を新設する（export 済みの ONNX を ORT セッションで動かす）。後処理はしきい値処理、dedup、カウント。レスポンスはインスタンス（RLE）と個数。
- serving は最小限の構成を保つ。RLE のエンコードは numpy で約30行の自前実装とし、**serving には pycocotools を追加しない**。

### 4.4 trainer_ui

- 学習フォーム（`training/`）：モード選択、条件に応じて表示する合成セクション（項目はフラットに並べ、ネストは増やさない）、プレビューの表示欄。
- 結果タブ（`results/`）：インスタンスのオーバーレイ描画（番号バッジ付き。パレットは Okabe–Ito。PoC の可視化で使ったパレットがすでに Okabe–Ito で、紫を含まず、色覚の多様性に配慮している）と、個数を示すチップ。`MeasurementPanel` にインスタンスモードを追加し、union-find の代わりに `instances.json` を読む。
- run の種類に応じた表示：セマンティック専用のパネル（confidence / error のヒートマップ、ピクセルのヒストグラム）は、インスタンスの run では**エラーにせず非表示**にする。
- キャンバスの操作は追加しない。オーバーレイは表示専用で、ズーム、パン、キーボードの既存の操作は変えない。

### 4.5 依存とライセンス

- `apps/trainer_api/requirements.in`：`rfdetr==1.8.*` を `[train]` extra 付きで追加する（export 系の依存は R1 の検証後）。互換性は 2026-07-22 に確認済み。`torch>=2.2`（コアの 2.13.*、cu128 の 2.11.* とも問題なし）、`transformers>=5.1,<6`（こちらは 5.5.4 に pin）、`pydantic>=2,<3`（こちらは 2.12.*）。`requirements-cu128.in` にも同じものを追加する。
- 新たに入る推移的依存のライセンス（PyPI で 2026-07-22 に確認）：supervision MIT、pytorch-lightning Apache-2.0、albumentations MIT、peft Apache-2.0、torchmetrics Apache-2.0、pycocotools BSD-2-Clause、roboflow Apache-2.0、rf100vl MIT、onnxsim MIT/Apache-2.0/BSD-2、polygraphy Apache-2.0。faster-coco-eval は**リポジトリで要再確認**（classifiers が欠けている。GitHub 上は Apache-2.0）。lockfile を更新するコミットには、全依存について `LICENSE: <pkg> <ver> <SPDX> confirmed at <URL>` の証跡を必ず残す。lockfile は `.in` のヘッダーどおり uv 0.11.11 で再生成し、`THIRD_PARTY_NOTICES.md` も更新する。
- **ban パターン**：rfdetr の "plus" extra と、検出系 XL/2XL の識別子を `scripts/nc-vendor-patterns.txt`（dev 専用のファイル）に追加し、pre-commit と公開 CI で再び混入するのを防ぐ。公開リポジトリの `NC_VENDOR_PATTERNS` 変数は、v0.9.8 のリリース時に更新する（リリースチェックリスト §1）。
- Windows での注意（2026-07-22 に確認）：`onnxsim` は `==0.4.36` に pin する必要がある。cp311 win_amd64 の wheel があるのはこの系列で、新しい版は sdist からのビルドが Windows のロングパス制限で失敗する。
- 事前調査で見つかった整理項目：`CONTRIBUTING.md` の依存関係の節が torch 2.6.0 のままになっている（`requirements.in` は 2.13.*）。あわせて修正する。

### 4.6 Windows 固有

- rfdetr の学習は `num_workers=0` と spawn ガードで動かす（複数 worker の sweep で起きる pagefile の問題も避けられる）。
- VRAM（2026-07-22 に実測。RTX 3090、Nano、1 epoch、実効 batch 16）：

  | batch | grad-accum | peak allocated | peak reserved | 必要 GPU |
  |-------|-----------|----------------|---------------|----------|
  | 8 | 2 | 5.75 GiB | 6.5 GiB | ≥ 8 GiB |
  | 4 | 4 | 3.29 GiB | 3.6 GiB | ≥ 5.5 GiB |
  | 2 | 8 | 2.01 GiB | 2.2 GiB | ≥ 3.5 GiB |

  自動縮退（`instance_training._fit_batch_to_vram`）：必要な tier が `total_memory` に収まるまで batch を半分にする（grad-accum を倍にし、実効 batch は保つ）。下限は batch 2。3.5 GiB 未満はサポート外。

## 5. データ契約

- run ディレクトリ：`train_config.json`、`instseg_dataset/{train,valid}/…`（COCO）、rfdetr の checkpoint、`metrics.json`、`instances/{item}.json` とオーバーレイ。
- `metrics.json`（インスタンスの run）：`segm_mAP_50_95_val`、`segm_AP50_val`、`AR_val`、`count_exact_val`（実写の val サブセットで計算）、`best_epoch`、`epochs_effective`、`dataset_stats {n_synth, n_real, n_cutouts, stack_pair_ratio}`。`MetricsSection.tsx` は、このサブセットを条件に応じて表示する。

## 6. テスト

- 単体テスト（CI、CPU）：seed を固定したときのコンポーザの決定論性、スタックペアの同軸性（角度差とギャップのアサーション）、dedup（重複は抑制し、隣接は残す）、RLE の往復変換。
- API：インスタンスモードのゲート、合成設定のバリデーション、プレビュー、推論成果物の形式。
- e2e：学習フォームの spec（高速。描画とバリデーション）。インスタンス学習のスモークテストは、既存の skip-budget の仕組みで `@heavy` としてゲートする（GPU が必要で、dev 機でのみ実行）。
- rfdetr の学習結果を golden として固定することはしない（環境によって結果が変わるため）。代わりに、固定した小さな合成セットに対して個数レベルのアサーションを行う。

## 7. マイルストーン

| M | 内容 | 目安 |
|---|------|------|
| M1 | segcore `instseg/` の移植（compose/count）+ 単体テスト + **ONNX export のスパイク（R1）** | 1–2日 |
| M2 | trainer_api のインスタンスモードを dev 機で end-to-end に通す + VRAM の実測 | 2–3日 |
| M3 | 結果タブの UI（オーバーレイ / カウント）+ 推論成果物 + プレビュー | 2日 |
| M4 | ONNX export + serving エンドポイント | 2日 |
| M5 | lockfile + ライセンスの証跡 + docs/handbook + e2e spec | 1–2日 |

## 8. リスク

- **R1 — rfdetr Seg の ONNX export：解消済み（2026-07-22 の M1 スパイク）**。学習済みの PoC checkpoint である RF-DETR-Seg Nano を ONNX（opset 17）に export し、onnxruntime の CPU 実行で実際に推論できることを確認した。確認した契約は次のとおり。入力は `1×3×312×312`（export の既定値。`shape=` で変更できるが、patch-12 のバックボーンに合う寸法にする必要がある）。出力は `dets 1×100×4`（正規化した box）、`labels 1×100×2`（logits）、`masks 1×100×78×78`（縮小した解像度の mask logits）。serving の後処理は sigmoid → 確信度のしきい値 → mask のリサイズ → dedup → カウントで、§4.3 の想定どおりである。**M4 補遺（2026-07-22）**：物体の確信度は `sigmoid(labels[:, 0])`（COCO の category 1 は内部のクラス index 0 に対応する）。numpy による処理全体（stretch リサイズ → /255 → ImageNet 正規化 → ORT → conf ≥ thr → mask の sigmoid → bilinear リサイズ → >0.5 → 面積 ≥ 16 → 貪欲法の mask-IoU dedup）で、SDK の `model.predict` とまったく同じ個数が得られた（PoC のテストセットで GT と 32/32 一致、SDK と 32/32 一致）。serving の `/count` と、export される `instance_inference.json` の契約は、この処理の流れをそのまま実装している。
- **R2 — roboflow SDK の挙動**。import 時にネットワーク通信やテレメトリが発生しないこと、オフラインでもインストールできることを検証する。必要なら lazy import で隔離する。
- **R3 — コアの lockfile の肥大化**（D2、許容済み）：インストールサイズと監査の対象が増える。
- **R4 — VRAM の少ない GPU**：M2 での実測と、grad-accum の自動調整で緩和する。
- **R5 — 分離できる剛体部品以外での、合成データのドメインギャップ**：面積帯の自動推定、プレビュー、実写の full-GT 画像の混入で緩和する。handbook に既知の制限として明記する。
