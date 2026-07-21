# 設計ドキュメント: Iterative Self-Training Loop + Hard Negative Mining

> **Status**: APPROVED
> **Context**: seg-studio — セマンティックセグメンテーション学習/推論ツール
>
> **歴史的設計メモ**: 本書は設計時点の計画です。現在の実装状況: `dataset.py` の pseudo-label 重み付け（`pseudo_ids` / `pseudo_weight`）と HNM / iterative mining（`hard_mining.py` / `iterative_mining.py`）は実装済みですが、`pseudo_label.py`・`self_training.py` コントローラ・`/train/self-train` API・UI（Phase 2〜4）は未実装です。segcore のパスは現在ネスト構成（`packages/segcore/segcore/...`）です。

## 1. 目的

少数のアノテーション画像（5〜10枚）から始めて、**自動的にアノテーションを拡張**しながら精度を向上させるセルフトレーニングループを実装する。

### 現在の課題
- 手動アノテーションは時間がかかる（1枚10分〜）
- Quick Learning（40epoch短期学習）は手動アノテーション画像のみで学習
- 推論結果は表示されるが、アノテーションへの自動フィードバックがない
- 少数画像学習ではFP（偽陽性）が多くなりがち

### 目標
```
手動5枚 → 学習 → 全画像推論 → 高確信度を自動マーキング → 再学習 → ...
```
各イテレーションでRecallを維持しつつPrecisionを上げる。

---

## 2. 設計仕様

| 項目 | 仕様 | 根拠 |
|---|------|------|
| FG/BG threshold | **統一 threshold** | ignore 不使用。confidence >= th → 予測 class、< th → BG(0)。BG も学習対象。 |
| Iteration 毎の threshold | **初回 2 回は高閾値固定、以降 class-wise coverage を見て緩和** | 単純緩和は誤検出増加リスク。 |
| 小領域除去 | **`max(絶対下限 px, 手動ラベル面積分布の下位 5-10%)`** | 微小欠陥への aggressive min-area は recall 破壊。 |
| HNM 頻度 | **3 epoch** (Quick Learning) / **5 epoch** (Full) | 毎 epoch は小 VRAM 環境に重い。 |
| Pseudo-label loss weight | **V1: 固定 0.5** → 将来: confidence 比例 (0.3–0.7 clip) | V1 はシンプルに固定。 |
| HNM sampling | **30% or 最大 20 枚、precision 低下時に拡大** | precision 低下時にサンプル拡大。 |
| Pseudo-label 管理 | **別ディレクトリ `pseudo_masks/` + manifest JSON** | manual を canonical として保持。 |
| 手動修正後 | **自動で `manual_corrected` 扱い、provenance 保持** | provenance 追跡が重要。 |
| β 値 | **1.5** | 2.0 は FP 許容過多。 |
| β 変化 | **iteration 0: β=1.5、以降: β=1.0** | 初回 Recall 重視 → バランスへ移行。 |

### 重要な簡略化決定
- **ignore_index (255) は使わない**: 低confidence → BG(0)。既存パイプライン変更不要
- **BGも学習する**: FP削減にはBGの学習が不可欠。ignoreで逃げない

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

### 3.2 Pseudo-Label生成 (シンプル版)

```python
def generate_pseudo_labels(
    pred_mask: np.ndarray,        # uint8 (H, W) — argmax予測
    confidence_map: np.ndarray,   # float32 (H, W) — [0, 1]
    threshold: float,             # 例: 0.85
    min_area_px: int,             # 小領域除去閾値
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

### 3.4 Dataset統合 (prepare時)

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

### 3.5 HNM強化

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

### Phase 1: Pseudo-Label生成 + 保存 (MVP)
- `packages/segcore/segcore/training/pseudo_label.py` — 新規
  - `generate_pseudo_labels()`: 予測→mask変換 (threshold + 小領域除去)
  - `save_pseudo_manifest()`: manifest JSON書き出し
- `apps/trainer_api/app/core/dataset_prep.py` — 修正
  - `prepare_with_pseudo()`: pseudo_masks/ のmerge対応
  - Val/Testはmanual-only

### Phase 2: Self-Training Controller
- `apps/trainer_api/app/core/self_training.py` — 新規
  - 学習→推論→Pseudo-label→再学習ループ
  - 収束判定、メトリクス履歴
  - イテレーション管理

### Phase 3: HNM + OHEM強化
- `packages/segcore/segcore/training/train.py` — 修正
  - HNM頻度: Quick Learning時は3 epoch
  - OHEM ratio: self-training時は0.3
  - Pseudo-label loss weight=0.5 対応
- `packages/segcore/segcore/training/dataset.py` — 修正
  - Pseudo-label aware (weight_map対応)

### Phase 4: API + UI
- API: `/projects/{pid}/train/self-train`, `/status`
- UI: Self-Trainingボタン、イテレーション進捗

### 将来拡張
- Mean Teacher (EMA) — 全員「最有力強化」と推奨
- FixMatch/ClassMix — 次点
- Boundary ignore band — 境界劣化対策
- Class-wise threshold — 不均衡クラス対策
- Confidence calibration — threshold意味のズレ対策

---

## 5. レビュー要約

### 主要指摘 (設計レビュー)
1. confidence定義: `sum(probs[1:])` はFGらしさ。BG確信度には `p(bg)` を使うべき → **V1はFGのみ採用で回避**
2. ignore_index経路問題: 6箇所で255→0変換 → **ignoreを使わない方針で回避**
3. Focal+OHEMはcalibrationを悪化 → **将来課題として認識**
4. boundary ignore band → **将来拡張**
5. class-wise pseudo coverage追跡 → **manifest.jsonで対応**

### 追加指摘 (精度レビュー)
1. FixMatch的アプローチ (弱aug→ラベル、強aug→学習) → **将来拡張**
2. Boundary Drift (境界が丸まる/太くなる) → **boundary ignore bandで将来対応**
3. UI提案型 (AI提案→ユーザー承認) → **Phase 4で検討**

### 合意事項
- 段階的実装: MVP→拡張
- Val setはmanual-only (絶対条件)
- Mean Teacherが将来最有力
- MixMatch優先度低 (batch=1制約)
