# Seg-Studio: Recipe Prompt Template

このドキュメントは、画像からセグメンテーション用レシピJSONを生成するためのプロンプトテンプレートです。

---

## 使い方

1. 下の「LLMに渡すプロンプト」セクションをコピー
2. 対象画像と一緒にLLMに送信
3. 出力されたJSONファイルを保存（`.json`）
4. Seg-Studio の Annotator → **Import Recipe** でアップロード

---

## LLMに渡すプロンプト

以下をコピーしてLLMに送ってください。`{{CLASS_NAME}}` と `{{DESCRIPTION}}` を実際の値に置き換えてください。

```
添付画像を分析し、「{{CLASS_NAME}}」（{{DESCRIPTION}}）を検出するためのレシピJSONを生成してください。

## 出力フォーマット

以下のJSON仕様に厳密に従ってください。コードブロック内にJSONのみを出力してください。

{
  "version": 1,
  "name": "レシピ名",
  "description": "何を検出するか、どのような特徴に基づくかの説明",
  "rules": [
    {
      "class_id": <正の整数 (1〜4)>,
      "class_name": "クラス名",
      "steps": [
        <ステップオブジェクトの配列>
      ]
    }
  ]
}

## ステップタイプ

### 1. hsv_range — HSV色空間でのしきい値処理
画像をHSV色空間に変換し、指定範囲内のピクセルを抽出します。

{
  "type": "hsv_range",
  "params": {
    "h_min": 0,    // 色相 最小値 (0–179)
    "h_max": 179,  // 色相 最大値 (0–179)  ※h_min > h_maxの場合は赤色系の折り返し処理
    "s_min": 0,    // 彩度 最小値 (0–255)
    "s_max": 255,  // 彩度 最大値 (0–255)
    "v_min": 0,    // 明度 最小値 (0–255)
    "v_max": 255   // 明度 最大値 (0–255)
  },
  "combine": "or"  // 省略可。"or"(デフォルト), "and", "subtract"
}

### 2. lab_range — Lab色空間でのしきい値処理
画像をCIE Lab色空間に変換し、指定範囲内のピクセルを抽出します。
照明変化に強い検出に向いています。

{
  "type": "lab_range",
  "params": {
    "l_min": 0,    // 輝度 最小値 (0–255)
    "l_max": 255,  // 輝度 最大値 (0–255)
    "a_min": 0,    // 緑–赤 軸 最小値 (0–255, 128が中心)
    "a_max": 255,  // 緑–赤 軸 最大値 (0–255)
    "b_min": 0,    // 青–黄 軸 最小値 (0–255, 128が中心)
    "b_max": 255   // 青–黄 軸 最大値 (0–255)
  },
  "combine": "or"
}

### 3. morphology — モルフォロジー演算
マスクのノイズ除去や穴埋めに使います。色ステップの後に配置してください。

{
  "type": "morphology",
  "params": {
    "operation": "close",  // "open"=ノイズ除去, "close"=穴埋め, "dilate"=膨張, "erode"=収縮
    "kernel_size": 5,      // カーネルサイズ（奇数: 3, 5, 7, ...）
    "iterations": 1        // 省略可、デフォルト1
  }
}

### 4. area_filter — 面積フィルタ
小さな検出領域を除去します。ステップの最後に配置することを推奨します。

{
  "type": "area_filter",
  "params": {
    "min_area_px": 100       // 最小面積（ピクセル数）
    // または "min_area_ratio": 0.001  // 画像全体に対する最小面積比率
  }
}

## combine パラメータ（色ステップ用）

色ステップ（hsv_range, lab_range）には "combine" を指定できます:
- "or"（デフォルト）: 既存マスクとOR結合（領域を追加）
- "and": 既存マスクとAND結合（共通部分のみ残す）
- "subtract": 既存マスクからこのステップの結果を除外

## ルール設計のガイドライン

1. **まず色で検出**: hsv_range または lab_range で対象の色範囲を指定
2. **ノイズ除去**: morphology の open でゴマ状ノイズを除去
3. **穴埋め**: morphology の close で内部の穴を埋める
4. **小領域除去**: area_filter で誤検出の小さな領域を除去
5. **複数の色範囲**: 対象が複数色を持つ場合は hsv_range を複数並べる（combine: "or"）
6. **背景除外**: 背景色が対象色に近い場合は combine: "subtract" で除外

## 複数クラスの場合

rulesに複数のルールを記述できます。class_id が小さい順に先勝ちで処理されます。
class_id は 1〜4 の正の整数を使ってください（0は背景として予約済み）。

## 画像分析時の注意事項

- HSVの色相(H)はOpenCV形式で 0–179 です（360度の半分）
- 赤色系は色相が0付近と179付近に分かれるため、h_min > h_max で折り返し指定が必要です
- Lab空間の a, b チャンネルは 128 が中性です
- カーネルサイズは検出対象のスケールに合わせて選んでください
  - 小さい対象: 3–5
  - 大きい対象: 7–15
- area_filterのmin_area_pxは、検出したい最小の対象物のサイズを目安にしてください
```

---

## 出力例

### 例1: 革表面の検出

```json
{
  "version": 1,
  "name": "革検出",
  "description": "茶色〜赤茶色の革表面を検出",
  "rules": [
    {
      "class_id": 1,
      "class_name": "革",
      "steps": [
        {
          "type": "hsv_range",
          "params": {
            "h_min": 8,
            "h_max": 25,
            "s_min": 60,
            "s_max": 255,
            "v_min": 50,
            "v_max": 220
          }
        },
        {
          "type": "morphology",
          "params": { "operation": "close", "kernel_size": 7 }
        },
        {
          "type": "morphology",
          "params": { "operation": "open", "kernel_size": 3 }
        },
        {
          "type": "area_filter",
          "params": { "min_area_px": 200 }
        }
      ]
    }
  ]
}
```

### 例2: 植物（緑）と花（赤・ピンク）の2クラス検出

```json
{
  "version": 1,
  "name": "植物・花検出",
  "description": "緑の葉と赤〜ピンクの花を分離検出",
  "rules": [
    {
      "class_id": 1,
      "class_name": "葉",
      "steps": [
        {
          "type": "hsv_range",
          "params": {
            "h_min": 30,
            "h_max": 85,
            "s_min": 40,
            "s_max": 255,
            "v_min": 30,
            "v_max": 255
          }
        },
        {
          "type": "morphology",
          "params": { "operation": "close", "kernel_size": 5 }
        },
        {
          "type": "area_filter",
          "params": { "min_area_px": 100 }
        }
      ]
    },
    {
      "class_id": 2,
      "class_name": "花",
      "steps": [
        {
          "type": "hsv_range",
          "params": {
            "h_min": 160,
            "h_max": 10,
            "s_min": 50,
            "s_max": 255,
            "v_min": 80,
            "v_max": 255
          }
        },
        {
          "type": "morphology",
          "params": { "operation": "open", "kernel_size": 3 }
        },
        {
          "type": "area_filter",
          "params": { "min_area_px": 50 }
        }
      ]
    }
  ]
}
```

### 例3: Lab色空間 + 背景除外を使った金属検出

```json
{
  "version": 1,
  "name": "金属部品検出",
  "description": "Lab色空間で銀色の金属部品を検出し、白い背景を除外",
  "rules": [
    {
      "class_id": 1,
      "class_name": "金属",
      "steps": [
        {
          "type": "lab_range",
          "params": {
            "l_min": 120,
            "l_max": 220,
            "a_min": 120,
            "a_max": 136,
            "b_min": 118,
            "b_max": 135
          }
        },
        {
          "type": "lab_range",
          "params": {
            "l_min": 230,
            "l_max": 255,
            "a_min": 125,
            "a_max": 131,
            "b_min": 125,
            "b_max": 131
          },
          "combine": "subtract"
        },
        {
          "type": "morphology",
          "params": { "operation": "close", "kernel_size": 9 }
        },
        {
          "type": "morphology",
          "params": { "operation": "open", "kernel_size": 5 }
        },
        {
          "type": "area_filter",
          "params": { "min_area_px": 500 }
        }
      ]
    }
  ]
}
```

---

## ステップタイプ早見表

| type | 用途 | 必須パラメータ | 備考 |
|------|------|--------------|------|
| `hsv_range` | HSV色閾値 | `h_min/max`, `s_min/max`, `v_min/max` | H: 0–179, S/V: 0–255 |
| `lab_range` | Lab色閾値 | `l_min/max`, `a_min/max`, `b_min/max` | 全チャンネル 0–255 |
| `morphology` | 形態変換 | `operation`, `kernel_size` | open/close/dilate/erode |
| `area_filter` | 小領域除去 | `min_area_px` or `min_area_ratio` | 最後に配置推奨 |

## class_id 対応表

| class_id | 用途 |
|----------|------|
| 0 | 背景（予約済・使用不可） |
| 1 | クラス1（デフォルト赤） |
| 2 | クラス2（デフォルト青） |
| 3 | クラス3（デフォルト緑） |
| 4 | クラス4 |
