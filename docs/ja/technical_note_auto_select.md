# Technical Note: Auto-Select Transfer Learning

> **⚠️ 2026-07 更新**: 学習開始時の**自動 donor 適用は廃止**された
> (ADR-005 addendum)。本書の profile library と `selector.recommend()` は
> **明示的な転移学習 (Transfer モード) とモデル検索**の裏付けとして存続する。
> 「学習開始時に自動で donor が付く」という記述は歴史的経緯であり、現行の
> 自動適用経路は存在しない。

**実装場所**: `packages/segcore/segcore/auto_select/`
**統合先**: `apps/trainer_api/app/routers/` (モデル検索 API) + Transfer モード

> **Sibling note**: combo recommendation (XGBoost ensemble model picking
> `(arch, bc, patch_size, …)` for a new project) is documented in
> [`auto_select_v6_combo_predictor.md`](../auto_select_v6_combo_predictor.md).
> This document covers the **transfer-learning** side (donor checkpoint
> selection from the profile library).

---

## 1. 概要

新プロジェクトの学習開始時に、過去の完了済みプロジェクトから最も類似したものを自動検索し、
そのmodel.ptを初期重みとして転移学習する仕組み。

**目的**: 少数画像(10〜30枚)での学習収束を加速し、F1を底上げする。

**設計原則**:
- ユーザーは何もしなくていい (デフォルトON)
- 初回学習時はライブラリが空なので何も起きない (graceful degradation)
- 学習完了ごとにプロファイルが蓄積され、次第に精度が上がる (flywheel効果)

---

## 2. アーキテクチャ

```
┌─────────────────────────────────────────────────┐
│  training_runner.py                              │
│                                                  │
│  ┌──────────────┐   ┌────────────────────────┐  │
│  │ 学習開始前    │   │ 学習完了後              │  │
│  │              │   │                        │  │
│  │ auto_select  │   │ save_training_profile  │  │
│  │ _pretrained()│   │ ()                     │  │
│  └──────┬───────┘   └───────────┬────────────┘  │
│         │                       │                │
└─────────┼───────────────────────┼────────────────┘
          │                       │
          ▼                       ▼
┌─────────────────────────────────────────────────┐
│  packages/segcore/segcore/auto_select/           │
│                                                  │
│  ┌─────────┐  ┌────────────┐  ┌──────────────┐  │
│  │schema.py│  │similarity. │  │selector.py   │  │
│  │         │  │py          │  │              │  │
│  │Profile  │  │cosine      │  │recommend()   │  │
│  │Transfer │  │handcrafted │  │arch voting   │  │
│  │Recommend│  │meta        │  │epoch scaling │  │
│  └────┬────┘  └─────┬──────┘  └──────┬───────┘  │
│       │             │                │           │
│  ┌────┴────┐  ┌─────┴──────┐  ┌─────┴────────┐  │
│  │profile_ │  │dino_       │  │features.py   │  │
│  │io.py    │  │features.py │  │(既存)         │  │
│  │.npz     │  │DINOv2      │  │handcrafted   │  │
│  │save/load│  │vitb14      │  │features      │  │
│  └─────────┘  └────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────┘
```

---

## 3. 類似度スコアリング

### ハイブリッドスコア

```
score = 0.70 × DINOv2_cosine + 0.20 × handcrafted_sim + 0.10 × meta_sim
```

DINOv2が利用不可の場合 (初期段階):
```
score = 0.75 × handcrafted_sim + 0.25 × meta_sim
```

### 3.1 DINOv2 Cosine Similarity (重み: 0.70)

- **モデル**: `dinov2_vitb14` (768次元, Facebook Research)
- **抽出方法**: 画像をパッチトークンに分解 → マスクでFG/BG分離 → 平均embedding
- **保存**: 4つのFGセントロイド (k-means) + FG/BG/global平均
- **マッチング**: query × candidateの全セントロイドペアでmax cosine

```python
dino_sim = max(cosine(q_centroid_i, c_centroid_j))
           for i in query.centroids
           for j in candidate.centroids
```

**根拠**: DINOv2は自己教師あり学習で獲得した意味的特徴量を持ち、
「金属表面のキズ」と「樹脂表面のキズ」の類似性を捉えられる。
画素レベルの類似度ではなく、欠陥の「概念的類似性」をスコア化する。

### 3.2 Handcrafted Feature Similarity (重み: 0.20)

8次元の特徴ベクトル:

| # | 特徴 | 意味 |
|---|------|------|
| 0 | color_divergence | FG/BG間のLAB色空間Bhattacharyya距離 |
| 1 | boundary_complexity | 境界長 / √(FG面積) |
| 2 | texture_contrast | ラプラシアン分散の差 (FG vs BG) |
| 3 | fg_scatter | 画像あたり平均連結成分数 |
| 4 | fg_ratio | 前景ピクセル比率 |
| 5 | mean_fg_area_px | 欠陥の平均面積 (px) |
| 6 | class_imbalance_ratio | max/min クラスピクセル比 |
| 7 | num_active_classes | アクティブクラス数 |

**距離**: Standardized Euclidean → exp(-0.5 × d)
- ライブラリ全体のstdで標準化 (スケール不変)
- ライブラリ < 3プロジェクトの場合は生Euclidean

### 3.3 Meta Similarity (重み: 0.10)

```python
meta_sim = 0.7 × scale_sim + 0.3 × class_sim

scale_sim = exp(-|log(1+q_area) - log(1+c_area)|)
class_sim = 1 / (1 + |q_classes - c_classes|)
```

**役割**: 欠陥のスケール (面積) とクラス数の互換性をチェック。
微小欠陥 (10px) と大面積欠陥 (10000px) のミスマッチを防ぐ。

---

## 4. アーキテクチャ投票

Top-K候補 (K=5) から重み付き投票でアーキテクチャを決定:

```python
vote[arch] += similarity × best_f1
target_arch = argmax(vote)
```

**例**: Top-5に SimpleUNet×3 (sim=0.8, 0.7, 0.6) と STDC×2 (sim=0.75, 0.65) がいる場合:
- SimpleUNet: 0.8×0.85 + 0.7×0.80 + 0.6×0.75 = 1.69
- STDC: 0.75×0.82 + 0.65×0.78 = 1.12
→ SimpleUNet が選択される

---

## 5. エポック・LRスケーリング (歴史的記録 — 自動適用は廃止)

自動 donor 適用時代は、donorとの類似度 + 同一アーキテクチャかどうかで
エポック数を決定していた。現行では from-scratch 予算 (wave6 min_width
ルール、60/80/100) のみが `auto_mode="recipe_only"` で適用される:

| 類似度 | 同arch | エポック | LR倍率 | 意図 |
|--------|--------|---------|--------|------|
| ≥ 0.80 | Yes | 12-13 (25%) | 0.25× | 高類似: fine-tuneのみ |
| ≥ 0.60 | Yes | 20 (40%) | 0.35× | 中類似: 適度に調整 |
| ≥ 0.40 | Yes | 30 (60%) | 0.50× | 低類似: 大幅調整 |
| < 0.40 | Yes | 40 (80%) | 0.75× | 微低類似: ほぼscratch |
| any | No | 40 (80%) | 1.00× | 異arch: 重み転送なし |

**注意**: early_stoppingは常に有効。推奨エポック数は上限であり、
収束が早ければ早期終了する。

---

## 6. プロファイル保存形式

`feature_profile.npz` (NumPy compressed archive):

```
├── dino_global_mean    (768,)   float32  — 全画像平均embedding
├── dino_fg_mean        (768,)   float32  — 前景パッチ平均
├── dino_bg_mean        (768,)   float32  — 背景パッチ平均
├── dino_fg_centroids   (4,768)  float32  — 前景k-meansセントロイド
├── handcrafted         (8,)     float32  — 手作り特徴量ベクトル
├── scalars_json        string   — JSON: project_id, run_id, arch, etc.
├── meta_json           string   — JSON: num_train, num_classes, etc.
└── handcrafted_names   (8,)     string   — 特徴量名リスト
```

**ファイルサイズ**: 約25KB/プロファイル (圧縮後)
**保存場所**: `projects/{id}/runs/{run_id}/feature_profile.npz`

---

## 7. データフロー

### モデル検索 / 明示転移時 (現行)

```
1. モデル検索 API: build_query_profile() でクエリ特徴量を計算
2. → load_library(PROJECTS_DIR) でライブラリスキャン
3. → selector.recommend(): 類似度ランキング → 候補提示
4. → ユーザーが候補から checkpoint を明示選択 (Transfer モード)
5. → 通常の学習パイプラインへ (train.py:_load_pretrained で重み読込)
```

(旧・学習開始時の自動 donor 適用フローは 2026-07 に廃止)

### 学習完了時

```
1. training_runner.py: 学習正常完了
2. → metrics.json から best_F1, best_mIoU 読込
3. → features.py: handcrafted特徴量を計算
4. → profile_io.save_profile() で .npz 保存
5. → 次回の load_library() でライブラリに含まれる
```

---

## 8. GUI統合

### HyperparameterForm.tsx (転移学習セクション)

| 設定項目 | デフォルト | 説明 |
|---------|-----------|------|
| 自動モデル選択 | ON | 類似プロジェクトのmodel.ptを初期値に |
| DINO特徴量抽出 | OFF | ONでDINOv2 embeddingも使用 (GPU必要) |
| DINOv2蒸留 | ON | 学習中のfeature蒸留 (既存機能) |

### API Schema (TrainRequest)

```python
auto_select: bool = Field(default=True)          # legacy, 受理するが無視
auto_config: bool = Field(default=True)          # legacy toggle
auto_mode: str = Field(default="recipe_only")    # ADR-005 Phase D + addendum
```

`auto_mode` は Auto 系機能を 1 個のノブに集約した後継フィールド。値は
`"recipe_only" | "off"` の 2 択:

- `"recipe_only"` (デフォルト): Auto-config ON (arch/bc/patch/distill 推奨 +
  from-scratch エポック予算)
- `"off"`: OFF (完全手動)

旧 `"full"` (自動 donor 適用) は廃止 — 送られてきた場合は `"recipe_only"`
に coerce される。

**互換ルール** (2026-07 時点):
- リクエスト body に `auto_config` が明示的に含まれていればそれが勝つ
  (pre-Phase-D API 互換)。`auto_select` は受理するが何も制御しない
- 明示指定が無ければ `auto_mode` から派生
- `auto_mode` に未知の値が入ると silent に `"recipe_only"` に落ちる (誤タイポで
  学習が壊れるより run を通した方が実害少ない、という判断)

Frontend UI (`HyperparameterForm.tsx` の master switch) は単一の
Auto On/Off トグル (2026-07-07、cb98adb)。API payload は
`auto_mode: "recipe_only" | "off"` のみを送信する。

Backend は他 API caller (legacy) 用に `auto_config` の互換解決を維持して
いるので、外部システムからの旧 API はそのまま動く (`auto_select` は無視)。

---

## 9. 制限事項と今後の改善

### 現在の制限

1. **DINOv2 embeddingは学習完了時に保存されない**: 現在はhandcraftedのみ。
   DINOv2抽出にはGPUと~30秒が必要で、学習完了後の自動実行はリスクあり。
2. **checkpoint_pathの移動に弱い**: プロファイルに絶対パスを保存。
   プロジェクトを別マシンに移動するとパスが壊れる。
3. **ライブラリサイズ**: O(N²)のペアワイズ比較。100プロジェクト超では
   インデックスやANN (Approximate Nearest Neighbor) が必要。

### 改善案

1. 学習完了後にバックグラウンドでDINOv2 embeddingを抽出・保存
2. checkpoint_pathを相対パスに変更
3. FAISS/Annoyによる高速近似探索 (将来)
4. Ensemble KLとDINOv2のプロジェクト特性に基づく自動切替

---

## 10. テスト

`tests/test_auto_select.py` — 転移学習まわりは 14 テスト
（ファイル全体では combo / time / VRAM predictor 分を含め 31 テスト）

| カテゴリ | テスト数 | 内容 |
|---------|---------|------|
| Schema | 3 | has_dino判定, features_to_handcrafted変換 |
| Similarity | 5 | cosine, self-similarity, no-dino fallback |
| Selector | 3 | empty library, donor発見, arch投票 |
| ProfileIO | 3 | save/load roundtrip, library scan |

```bash
pytest tests/test_auto_select.py -v
```
