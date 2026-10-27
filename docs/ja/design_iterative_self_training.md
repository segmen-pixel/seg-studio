# 設計ドキュメント：Iterative Self-Training Loop + Hard Negative Mining

> **状態**：APPROVED
>
> **対象**：seg-studio（セマンティックセグメンテーションの学習・推論ツール）
>
> **過去の設計メモ**：本書は設計時点の計画である。現在の実装状況は次のとおり。`dataset.py` の pseudo-label の重み付け（`pseudo_ids` / `pseudo_weight`）と、HNM / iterative mining（`hard_mining.py` / `iterative_mining.py`）は実装済み。一方、`pseudo_label.py`、`self_training.py` のコントローラ、`/train/self-train` API、UI（Phase 2〜4）は未実装である。segcore のパスは現在、ネストした構成（`packages/segcore/segcore/...`）になっている。

## 1. 目的

少数のアノテーション済み画像（5〜10枚）から始め、**アノテーションを自動で増やしながら**精度を上げていくセルフトレーニングのループを実装する。

### 現在の課題
- 手動のアノテーションには時間がかかる（1枚10分以上）。
- Quick Learning（40 epoch の短期学習）は、手動でアノテーションした画像だけで学習する。
- 推論結果は表示されるが、それをアノテーションに自動で反映する仕組みがない。
- 少数の画像で学習すると、FP（偽陽性）が多くなりやすい。

### 目標
```
手動5枚 → 学習 → 全画像推論 → 高確信度を自動マーキング → 再学習 → ...
```
各イテレーションで、Recall を維持しながら Precision を上げる。

---

## 2. 設計仕様

| 項目 | 仕様 | 根拠 |
|---|------|------|
| FG/BG のしきい値 | **共通のしきい値を使う** | ignore は使わない。confidence >= th なら予測したクラス、< th なら BG（0）。BG も学習の対象とする。 |
| イテレーションごとのしきい値 | **最初の2回は高いしきい値で固定し、以降は class-wise coverage を見て緩める** | 単純に緩めると誤検出が増えるおそれがある。 |
| 小領域の除去 | **`max(絶対下限 px, 手動ラベル面積分布の下位 5-10%)`** | 微小な欠陥に対して min-area を強くかけすぎると recall が大きく落ちる。 |
| HNM の頻度 | **3 epoch**（Quick Learning）/ **5 epoch**（Full） | 毎 epoch 行うと、VRAM の少ない環境では重い。 |
| pseudo-label の loss weight | **V1 は 0.5 で固定** → 将来は confidence に比例させる（0.3–0.7 で clip） | V1 は単純に固定値とする。 |
| HNM のサンプリング | **30% または最大20枚。precision が下がったら増やす** | precision が下がったときはサンプルを増やす。 |
| pseudo-label の管理 | **別ディレクトリ `pseudo_masks/` + manifest JSON** | manual を canonical（正本）として保持する。 |
| 手動で修正した後 | **自動で `manual_corrected` として扱い、provenance を保持する** | provenance の追跡が重要。 |
| β の値 | **1.5** | 2.0 では FP を許容しすぎる。 |
| β の推移 | **iteration 0 は β=1.5、以降は β=1.0** | 初回は Recall を重視し、その後はバランス重視に移る。 |

### 簡略化のための重要な決定
- **ignore_index（255）は使わない**：confidence の低いピクセルは BG（0）にする。既存のパイプラインを変える必要がない。
- **BG も学習する**：FP を減らすには BG の学習が欠かせない。ignore で逃げない。

---

## 3. アーキテクチャ

### 3.1 全体フロー
```
┌─────────────────────────────────────────────────────────────┐
│              Iterative Self-Training Controller               │
│                                                               │
│  Iteration 0:                                                 │
│    Input: 手動マーキング N枚                                   │
│    → 通常学習 (full training, 80 epoch)                        │
│    → 全画像推論 (SW)                                           │
│    → Pseudo-label生成 (高threshold, FG+BG)                     │
│    → pseudo_masks/ に保存 + manifest.json                      │
│                                                               │
│  Iteration 1..K:                                              │
│    Input: manual masks/ + pseudo_masks/ (weight=0.5)           │
│    → Quick Learning (40ep, HNM 3ep毎, OHEM=0.3, β=1.0)        │
│    → 全画像推論                                                │
│    → Pseudo-label更新                                          │
│    → 収束判定 (F1改善 < 0.5% or Precision低下 > 2%)            │
│                                                               │
│  Output: 最終モデル + 全画像のマスク                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 pseudo-label の生成（シンプル版）

```python
def generate_pseudo_labels(
    pred_mask: np.ndarray,        # uint8 (H, W) — argmax予測
    confidence_map: np.ndarray,   # float32 (H, W) — [0, 1]
    threshold: float,             # 例: 0.85
    min_area_px: int,             # 小領域除去しきい値
) -> np.ndarray:
    """
    予測→Pseudo-mask変換。ignoreは使わない。
    高confidence FG → クラスID、それ以外 → 0 (BG)
    """
    pseudo_mask = np.zeros_like(pred_mask)  # デフォルト: BG

    # 高確信度の前景ピクセルのみ採用
    fg_confident = (pred_mask > 0) & (confidence_map >= threshold)
    pseudo_mask[fg_confident] = pred_mask[fg_confident]

    # 小領域除去 (ノイズ)
    pseudo_mask = remove_small_components(pseudo_mask, min_area=min_area_px)

    return pseudo_mask
```

### 3.3 ディレクトリ構造

```
projects/{pid}/datasets/
├── annotate/
│   ├── images/           ← 全画像
│   ├── masks/            ← 手動マスクのみ (canonical)
│   └── pseudo_masks/     ← 自動生成マスク (新規)
├── pseudo_manifest.json  ← Pseudo-label メタデータ (新規)
└── prepared/             ← prepare時にmerge
```

**pseudo_manifest.json:**
```json
{
  "version": 1,
  "threshold": 0.85,
  "source_run_id": "quick0",
  "iteration": 1,
  "created_at": "2026-04-01T12:00:00",
  "items": {
    "img003": {"mean_confidence": 0.91, "fg_pixels": 1234, "fg_ratio": 0.001},
    "img004": {"mean_confidence": 0.88, "fg_pixels": 890, "fg_ratio": 0.0007}
  }
}
```

### 3.4 Dataset への統合（prepare 時）

```python
def prepare_with_pseudo(project_id, pseudo_weight=0.5):
    """
    手動masks/ + pseudo_masks/ をmergeしてprepared/に配置。
    Val/Testは手動ラベルのみ。Pseudo-labelはtrain setのみ。
    """
    # 1. 手動マスク画像 → train/val/test split (既存ロジック)
    manual_ids = get_manual_mask_ids()
    train_ids, val_ids, test_ids = split(manual_ids)

    # 2. Pseudo-mask画像 → train setに追加
    pseudo_ids = get_pseudo_mask_ids()
    pseudo_ids = [p for p in pseudo_ids if p not in manual_ids]  # 手動優先
    train_ids += pseudo_ids

    # 3. prepared/masks/ にコピー (手動はそのまま、pseudoもそのまま)
    # 4. weight情報をDatasetに渡す (pseudo画像はloss weight=0.5)
```

### 3.5 HNM の強化

```
既存 (5 epoch毎):
  _mine_hard_negatives() → FP中心座標 → _hn_centers → パッチサンプリング

変更:
  - Quick Learning: 3 epoch毎に短縮
  - OHEM ratio: 0.0 → 0.3 (top 30% hardest pixels)
  - HNM sampling: 30% or 最大20枚
```

### 3.6 モデル選択基準

```python
# Iteration 0: Recall重視 (β=1.5)
f_beta = (1 + 1.5**2) * P * R / (1.5**2 * P + R)

# Iteration 1+: バランス (β=1.0 = F1)
f_beta = 2 * P * R / (P + R)
```

### 3.7 収束判定

```python
def should_stop(metrics_history, max_iterations=5):
    if len(metrics_history) < 2:
        return False
    prev, curr = metrics_history[-2], metrics_history[-1]

    # F1改善 < 0.5%
    if curr["val_f1"] - prev["val_f1"] < 0.005:
        return True
    # Precision低下 > 2%
    if curr["val_precision"] < prev["val_precision"] - 0.02:
        return True
    # 最大iteration
    return len(metrics_history) >= max_iterations
```

---

## 4. 実装計画

### Phase 1：pseudo-label の生成と保存（MVP）
- `packages/segcore/segcore/training/pseudo_label.py` — 新規
  - `generate_pseudo_labels()`：予測を mask に変換（しきい値処理と小領域の除去）
  - `save_pseudo_manifest()`：manifest JSON の書き出し
- `apps/trainer_api/app/core/dataset_prep.py` — 修正
  - `prepare_with_pseudo()`：pseudo_masks/ の merge に対応
  - Val/Test は manual のみ

### Phase 2：Self-Training Controller
- `apps/trainer_api/app/core/self_training.py` — 新規
  - 学習 → 推論 → pseudo-label → 再学習のループ
  - 収束判定、メトリクスの履歴
  - イテレーションの管理

### Phase 3：HNM と OHEM の強化
- `packages/segcore/segcore/training/train.py` — 修正
  - HNM の頻度：Quick Learning では 3 epoch ごと
  - OHEM ratio：self-training では 0.3
  - pseudo-label の loss weight=0.5 に対応
- `packages/segcore/segcore/training/dataset.py` — 修正
  - pseudo-label を考慮（weight_map に対応）

### Phase 4：API と UI
- API：`/projects/{pid}/train/self-train`、`/status`
- UI：Self-Training ボタン、イテレーションの進捗

### 将来の拡張
- Mean Teacher（EMA）— 全員が「最も有力な強化策」として推奨
- FixMatch/ClassMix — 次点
- Boundary ignore band — 境界の劣化への対策
- Class-wise threshold — クラスの不均衡への対策
- Confidence calibration — しきい値の意味のずれへの対策

---

## 5. レビューの要約

### 主な指摘（設計レビュー）
1. confidence の定義：`sum(probs[1:])` は FG らしさを表す。BG の確信度には `p(bg)` を使うべき → **V1 は FG だけを採用することで回避**
2. ignore_index の経路の問題：6箇所で 255 → 0 に変換されている → **ignore を使わない方針で回避**
3. Focal + OHEM は calibration を悪化させる → **将来の課題として認識**
4. boundary ignore band → **将来の拡張**
5. class-wise の pseudo coverage の追跡 → **manifest.json で対応**

### 追加の指摘（精度レビュー）
1. FixMatch 的なアプローチ（弱い aug でラベルを作り、強い aug で学習）→ **将来の拡張**
2. Boundary Drift（境界が丸まる、太くなる）→ **将来 boundary ignore band で対応**
3. UI での提案型（AI が提案し、ユーザーが承認）→ **Phase 4 で検討**

### 合意事項
- 段階的な実装（MVP → 拡張）
- Val set は manual のみ（絶対条件）
- 将来の最有力候補は Mean Teacher
- MixMatch は低優先（batch=1 の制約のため）
