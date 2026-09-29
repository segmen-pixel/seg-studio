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
│   ├── runs/{run_id}/...   # インポートでは読み込まれない
│   └── pretrained/  # インポート済み事前学習チェックポイント（ある場合）
└── metadata.json    # プロジェクトメタデータと画像ごとのマーク
```

### masks/ のフォーマット

| 項目 | 値 |
|------|----|
| ファイル名 | `{item_id}.png` (item ID) |
| チャンネル | シングルチャンネル（グレースケール、PIL mode `"L"`） |
| ピクセル値 | クラス ID（0 = background, 1+ = ユーザー定義クラス） |
| 無視インデックス | 255（旧バージョンの未塗り領域） |

> **255 は学習から除外されません。** フィールド名に反して、マスク値 255 は旧バージョンで
> 塗られなかった画素を表し、データセット準備の時点で背景（クラス 0）に変換されます。
> loss にも metric にも 255 のままでは渡らないため、モデルはその画素を背景として学習し、
> スコアにも背景として計上されます。`ignore_index` フィールドは 255 固定の契約値で、
> 範囲外のクラス ID に対する内部の安全クランプ用に予約されています。本当に無視したい
> 領域がある場合、255 に頼らないでください。

PNG ではなくタイルで保存されている大きな画像のマスクも、タイルから行の帯ごとに読み出して他のマスクと同じ PNG として書き出します。`resize_scale` による縮小も同じ最近傍です。0.9.9 より前は、そうした画像には全ゼロの穴埋めか、タイルの横に残っていた古い PNG が書き出されていました。


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
    { "id": "sample_001", "filename": "sample_001.png", "name": "sample_001.jpg",
      "hasMask": true, "markedClean": false, "set": "train" },
    { "id": "sample_002", "filename": "sample_002.png", "name": "sample_002.png",
      "hasMask": true, "markedClean": true },
    { "id": "sample_003", "filename": "sample_003.png", "name": "sample_003.png",
      "hasMask": false, "markedClean": false,
      "draft": true, "draftRun": "review", "draftReason": "境界が不明瞭" }
  ]
}
```

`items` の各要素が 1 枚の画像を表します。`id` と `filename` は常にあり、それ以外は `masks/` の画素だけでは表せない「その画像について決めたこと」の記録で、インポート時に読み戻されます（[画像ごとのマーク](#画像ごとのマーク) を参照）。

| フィールド | 出力条件 | 意味 |
|-----------|---------|------|
| `id` | 常に | item ID。マスクは `masks/{id}.png` |
| `filename` | 常に | `images/` 内のファイル名 |
| `name` | 設定されているとき | 画像一覧に表示される名前。ないときはインポート時にファイル名を表示 |
| `hasMask` | 常に | `false`: マスクのない画像。`masks/` 内のファイルはエクスポートが書いた全ゼロの穴埋め |
| `markedClean` | 常に | `true`: **OK** 指定済み（確認済みで、ラベル付けするものがない画像）。全ゼロのマスクは未ラベルではなく負例 |
| `set` | `train` / `test` のときのみ | 学習 / テストの手動割り当て |
| `draft` | 付いているとき | `true`: **要確認** フラグ、または学習ランが下書きしてまだ誰も確認していないマスク |
| `draftRun` | `draft` と一緒に | **要確認** フラグなら `"review"`、それ以外は下書きしたラン |
| `draftReason` | `draft` と一緒に（理由があるとき） | フラグを付けた理由 |
| `by` | エージェントがマスクを保存したとき | 書いたエージェント。人が保存したマスクには付かない |
| `synthetic` | 合成サンプル | `true`: 合成機能で生成された画像 |

以前のバージョンのエクスポートには `id` と `filename` しかありません。

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

インポートが読むのはアーカイブ内の 1 か所だけです。ZIP の直下、または直下に画像がなくフォルダが 1 つだけの場合（エクスポートや、フォルダを圧縮した ZIP）はそのフォルダです。そこで:

- `images/` フォルダに画像があれば、それが取り込む画像で、マスクは隣の `masks/` フォルダ内の `.png` です
- なければ、その場所に直接置かれた画像を取り込みます。隣に `masks/` フォルダがあればそこからマスクを読みます

数えるのは `images/` と `masks/` の直下にあるファイルだけで、**それより深いフォルダの画像は取り込みません**。エクスポートの `training/` 配下の学習ラン（ランの信頼度グラフは PNG）、`prepared/` のコピー、画像のサブフォルダなどが該当します。`__MACOSX/`、`._*`、`Thumbs.db`、`.DS_Store`、`desktop.ini` は無視します。この場所に画像が 1 枚もなければインポートはエラーになり、エラーには読み飛ばした画像の 1 つが示されます。

トップフォルダの中にも画像がなくフォルダが 1 つだけなら、さらにその中へ進みます（合わせて最大 3 階層）。エクスポートを Windows の **すべて展開** で展開すると同じ名前のフォルダの中に入り、そのフォルダをもう一度圧縮すると 2 階層になるためです（パターン D）。この方法で `images/`・`masks/` に入ることはなく、トップフォルダより下では `training/`・`runs/`・`prepared/` にも入りません。

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

#### パターン B: トップフォルダ 1 つ（Seg-Studio のエクスポート）

```
sample_20260401_1200.zip
└── sample_20260401_1200/
    ├── images/
    ├── masks/
    ├── metadata.json
    ├── train.txt / val.txt
    └── training/        # 取り込まれない
```

#### パターン C: 画像を入れたフォルダ

```
photos.zip
└── photos/              # または ZIP 直下に画像を直接
    ├── img_001.jpg
    ├── img_002.jpg
    └── masks/           # 任意
        └── img_001.png
```

#### パターン D: 展開して圧縮し直したエクスポート

```
sample_20260401_1200.zip
└── sample_20260401_1200/          # すべて展開が作ったフォルダ
    └── sample_20260401_1200/      # エクスポート本来のフォルダ
        ├── images/
        ├── masks/
        └── metadata.json
```

> 以前のバージョンは `images/` と `masks/` を **任意の深さ** で探し、`masks/` 以外にある画像をすべて取り込んでいたため、エクスポートを取り込むと学習ランのグラフが余分な画像として入っていました。深いフォルダ（例: `datasets/prepared/images/`）を前提に作った ZIP は、`images/` と `masks/` を最上位に移してください。

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
| チャンネル | シングルチャンネル（グレースケール）推奨。RGB の場合、インポート時は OpenCV の第 1 チャンネル（BGR 順のため青）がクラス ID として読まれるため、曖昧さを避けるには 1ch グレースケールを使用 |
| ピクセル値 | クラス ID（0 = background, 1+ = ユーザー定義クラス） |

### マスクと画像の対応ルール

ZIP 内に `metadata.json` がある場合（= Seg-Studio からのエクスポート）は、その `items` 配列（元ファイル名 → item ID）で対応付けます。それ以外は **ファイル名のステム（拡張子を除いた部分）が一致** するものを対応付けます。

```
images/bolt_001.png  ←→  masks/bolt_001.png   (ステム: bolt_001)
images/sample.jpg    ←→  masks/sample.png     (ステム: sample)
```

対応するマスクが見つからない画像は、マスクなし（未アノテーション）としてインポートされる。

### 画像ごとのマーク

`metadata.json` の `items` に [metadata.json](#metadatajson) の表のフィールドがあれば、インポート時に元に戻します:

- `markedClean: true` — 画像は再び **OK** になります。ただしマスクが取り込まれ、何も塗られていない場合に限ります（塗りのあるマスクへの OK は無視）
- `hasMask: false` — `masks/` 内の全ゼロの穴埋めは捨て、画像は未ラベルのままにします
- `set`、`name`、`draft` / `draftRun` / `draftReason`（**要確認** フラグ）、`synthetic` — そのまま戻します。`by` はマスクと一緒のときだけ戻します

これらのフィールドがない場合（以前のバージョンのエクスポートや他ツールの ZIP）は従来どおり、渡されたマスクはすべて保存済みとして扱い、全ゼロのマスクは **OK** ではなく未ラベルの画像になります。

### クラス定義ファイル

`images/` の隣（または ZIP 直下）の `classes.json` を最優先で使います。なければ `metadata.json` の `classes` 配列（エクスポートなら、エクスポート時点のプロジェクトのクラス）を使い、どちらもない場合に限ってアーカイブのより深い場所の `classes.json`（学習ランが持つコピーなど）を使います。

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
2. アーカイブをスキャン: images/, masks/, classes.json, metadata.json（深いフォルダは読み飛ばす）
3. 画像を PNG に変換（並列処理）して登録
4. metadata.json の items またはステム名でマスクを照合し、画像ごとのマークを復元
5. クラス定義を登録（classes.json または metadata.json）
6. マスク内に未定義クラス ID があれば自動補完（reconcile）
7. プロジェクト一覧を更新
```

### レスポンス

```json
{ "status": "ok", "image_count": 5, "mask_count": 4, "clean_count": 2,
  "classes_imported": true, "reconciled_classes": 0,
  "passed_over": 2, "passed_over_sample": ["images/old/img_006.png", "img_007.png"] }
```

`passed_over` は、画像を読む場所の外にあって取り込まなかった画像の数、`passed_over_sample` はそのうち最大 5 件のパスです。エクスポート自身の `training/` フォルダは数えません（学習ランのグラフは意図して取り込まないため）。プロジェクト タブのインポート完了メッセージ（画像一覧に ZIP をドロップした場合はそのメッセージ）にもこの数が表示されます。

### 制約事項

- 画像を読む場所に画像が1枚もない場合はエラー
- インポートで戻るのはデータセット（画像・マスク・クラス定義・画像ごとのマーク）です。ZIP 内の学習ラン・モデル・学習/検証の分割ファイルは取り込まれません
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
- `metadata.json` の `items` 配列で item ID → オリジナルファイル名のマッピングと、画像ごとのマーク（OK・要確認・学習/テスト）が参照可能
- `train.txt` / `val.txt` で学習/検証の分割情報を取得可能
- `training/`: 学習ラン（チェックポイント・設定・メトリクス）— データセットだけ必要な場合は無視して構いません（インポートでは読み込まれません）
