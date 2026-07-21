# Import / Export フォーマット仕様

Seg-Studio のプロジェクトデータを外部ツールと相互運用するための仕様書。

---

## Export（エクスポート）

### エンドポイント

```
GET /api/v1/projects/{project_id}/datasets/export
```

任意のクエリパラメータ: `resize_scale`（0.1〜1.0）— エクスポート時に画像（Lanczos）とマスク（最近傍）を縮小します。

プロジェクトタイル内の **エクスポート** ボタンで実行。エクスポートダイアログでは元データのサイズ、前景（欠陥領域）分析、「縮小して軽量化」オプションを確認してから ZIP をダウンロードできます。

### 出力形式

ZIP アーカイブ。ファイル名: `{プロジェクト名}_{YYYYMMDD_HHMM}.zip`（縮小時は `{プロジェクト名}_s{scale}_{YYYYMMDD_HHMM}.zip`）

```
{prefix}/
├── images/          # 元画像（オリジナルファイル名を保持）
│   ├── sample_001.png
│   ├── sample_002.jpg
│   └── ...
├── masks/           # アノテーションマスク（グレースケール PNG）
│   ├── {item_id}.png  # 未アノテーション画像には全ゼロ（背景のみ）マスクが入る
│   └── ...
├── train.txt        # 学習用 item ID リスト（1行1ID）
├── val.txt          # 検証用 item ID リスト（1行1ID）
├── training/        # 学習ラン（チェックポイント・設定・メトリクス）
│   └── runs/{run_id}/...
└── metadata.json    # プロジェクトメタデータ
```

### masks/ のフォーマット

| 項目 | 値 |
|------|----|
| ファイル名 | `{item_id}.png` (item ID) |
| チャンネル | シングルチャンネル（グレースケール、PIL mode `"L"`） |
| ピクセル値 | クラス ID（0 = background, 1+ = ユーザー定義クラス） |
| 無視インデックス | 255（未アノテーション領域） |

### metadata.json

```json
{
  "project_id": "uuid-string",
  "project_name": "Bolt",
  "exported_at": "2026-04-01T12:00:00",
  "num_images": 97,
  "num_train": 78,
  "num_val": 19,
  "classes": [
    { "id": 0, "name": "background", "color": [0, 0, 0] },
    { "id": 1, "name": "キズ", "color": [242, 36, 36] }
  ],
  "ignore_index": 255,
  "items": [
    { "id": "item-uuid", "filename": "sample_001.png" }
  ]
}
```

### train.txt / val.txt

- split が `prepared/splits/` に存在すればそれを使用
- なければエクスポート対象の全 item を 80/20 でランダム分割
- 1行に1つの item ID

---

## Import（インポート）

### エンドポイント

```
POST /api/v1/projects/{project_id}/datasets/annotate/import_zip
```

プロジェクト タブの **インポート** ボタンで **ZIP ファイル** を選択します。ZIP のファイル名で新規プロジェクトが作成され、その中にアーカイブの内容がインポートされます。

### 対応 ZIP 構造

#### パターン A: フラット構造

```
MyProject.zip
├── images/
│   ├── img_001.png
│   └── ...
├── masks/
│   ├── img_001.png
│   └── ...
└── classes.json
```

#### パターン B: ネスト構造

```
MyProject.zip
├── datasets/
│   └── prepared/
│       ├── images/
│       │   ├── img_001.png
│       │   └── ...
│       └── masks/
│           ├── img_001.png
│           └── ...
└── classes.json
```

> `images/` と `masks/` はアーカイブ内の **任意の深さ** で検出されます（直上のフォルダ名で判定）。`images/` フォルダなしで ZIP 直下に置いた画像も取り込まれます。

### 画像ファイル（images/）

| 項目 | 値 |
|------|----|
| 対応拡張子 | `.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.webp` |
| 大文字小文字 | 不問 |
| 補足 | PNG 以外はインポート時に PNG へ変換されます |

### マスクファイル（masks/）

| 項目 | 値 |
|------|----|
| 対応拡張子 | `.png` のみ |
| チャンネル | シングルチャンネル（グレースケール）推奨。RGB の場合 R チャンネルがクラス ID として扱われる |
| ピクセル値 | クラス ID（0 = background, 1+ = ユーザー定義クラス） |

### マスクと画像の対応ルール

ZIP 内に `metadata.json` がある場合（= Seg-Studio からのエクスポート）は、その `items` 配列（元ファイル名 → item ID）で対応付けます。それ以外は **ファイル名のステム（拡張子を除いた部分）が一致** するものを対応付けます。

```
images/bolt_001.png  ←→  masks/bolt_001.png   (ステム: bolt_001)
images/sample.jpg    ←→  masks/sample.png     (ステム: sample)
```

対応するマスクが見つからない画像は、マスクなし（未アノテーション）としてインポートされる。

### クラス定義ファイル

`classes.json` をアーカイブ内の任意の深さで自動検出します。`classes.json` がなくても、`metadata.json` に `classes` 配列があればそこから取り込みます。

```json
{
  "version": 1,
  "ignore_index": 255,
  "classes": [
    { "id": 0, "name": "background", "color": [0, 0, 0], "active": true },
    { "id": 1, "name": "キズ",       "color": [242, 36, 36], "active": true }
  ]
}
```

- `color`: RGB 配列 `[R, G, B]`
- `version`, `ignore_index`, `active` フィールドは任意

### インポート処理フロー

```
1. ZIP 選択 → ZIP ファイル名でプロジェクト作成
2. アーカイブをスキャン: images/, masks/, classes.json, metadata.json
3. 画像を PNG に変換（並列処理）して登録
4. metadata.json の items またはステム名でマスクを照合
5. クラス定義を登録（classes.json または metadata.json）
6. マスク内に未定義クラス ID があれば自動補完（reconcile）
7. プロジェクト一覧を更新
```

### 制約事項

- ZIP 内に画像が1枚もない場合はエラー
- マスクのみ（画像なし）のインポートは不可
- クラス定義ファイルは任意（なくてもインポート可能）
- マスクは `.png` のみ（それ以外の形式は無視されます）

---

## 外部ツール連携ガイド

### 他ツールから Seg-Studio へインポートする場合

以下の構造で ZIP アーカイブを準備:

```
project_name.zip
├── images/    # 元画像
├── masks/     # クラスID のグレースケール PNG（ステム名を画像と一致させる）
└── classes.json  # クラス定義（任意）
```

### Seg-Studio から他ツールへエクスポートする場合

Export ZIP を展開すると:

- `images/`: 元画像（オリジナルファイル名）
- `masks/`: グレースケールマスク（ファイル名は item ID）
- `metadata.json` の `items` 配列で item ID → オリジナルファイル名のマッピングが参照可能
- `train.txt` / `val.txt` で学習/検証の分割情報を取得可能
- `training/`: 学習ラン（チェックポイント・設定・メトリクス）— データセットだけ必要な場合は無視して構いません
