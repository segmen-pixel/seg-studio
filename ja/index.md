---
title: "Seg-Studio — ローカルで完結するセグメンテーション学習 GUI"
description: "オープンソースのセマンティックセグメンテーション統合 GUI。SAM 支援アノテーション、PyTorch 学習、個数カウント、ONNX / CoreML エクスポートをローカル環境だけで実行できます。Windows / macOS 対応。"
lang: ja
---

# Seg-Studio

**アノテーション・学習・エクスポートを、すべて手元のマシンで。**

Seg-Studio は、セマンティックセグメンテーションのためのオールインワンの GUI ツールです。Windows と macOS でローカルのアプリとして動き、SAM を使ったアノテーション、PyTorch によるモデルの学習、ONNX と CoreML へのエクスポートまでを1つの画面で行えます。クラウドのアカウントも画像のアップロードも要らず、データと学習済みモデルは実行したマシンから出ません。

セグメンテーションに加えて、**アノテーション済みのマスクを使った個数カウント**にも対応しています。カウントのためにアノテーションをやり直す必要はありません。

現在のリリース：**v0.9.13（beta）** · Apache-2.0 · [ダウンロード](https://github.com/segmen-pixel/seg-studio/releases/latest) · [ソース](https://github.com/segmen-pixel/seg-studio) · [English](../)

![SAM でアノテーションし、学習・評価してエクスポートするまでの流れ](../docs/images/hero.gif)

## どこから読むか

| 目的 | ページ |
|---|---|
| はじめて使う | [はじめての手順](../docs/ja/first-run-manual.md) — アプリを開いてから最初の推論結果まで約10分 |
| 機能を1つ調べたい | [ユーザーガイド](../docs/ja/user-guide.md) — 各タブ・ツール・設定の説明 |
| ひと通り理解したい | [ハンドブック](../docs/ja/handbook.md) — サンプルデータで最初から最後まで、全16章 |
| Linux、Docker、オフライン環境に入れたい | [インストールの詳細](../docs/ja/install.md) |
| インストールでつまずいた | [トラブルシューティング](../docs/ja/troubleshooting.md) |
| AI モデルにラベル付けを任せたい | [AI サポート](../docs/ja/ai-support.md) — [Ollama での手順例](../docs/ja/ai-support-ollama.md)もあります |
| 共有マシンや LAN で使う | [デプロイガイド](../docs/ja/deployment.md) |
| 開発に参加したい | [開発者向けクイックスタート](../docs/ja/dev-quickstart.md) |

## 主な機能

- **SAM を使ったアノテーション** — クリックでの領域の切り出し、ブラシ、矩形、クラスの管理、自動保存。マスクはインデックス付きの PNG なので、ほかのツールにも持ち出せます。
- **コードを書かない学習** — タスクを選ぶと、バックボーン、エポック数、データ拡張、学習率がレシピとして自動で決まります。どの項目も、あとから手で変えられます。
- **学習中の確認と評価** — 学習中はエポックとそこまでの最高 F1 を表示し、終わったあとは画像ごとの F1・適合率・再現率・IoU、確信度の分布、ヒートマップで、モデルが何を学んだかを確かめられます。
- **個数カウント** — 既存のマスクから物体の個数を数えます。
- **エクスポート** — ONNX、OpenVINO IR、CoreML。エッジデバイス、iPhone、Seg-Studio を入れていない PC でも推論できます。
- **オフラインで動作** — クラウドのアカウント不要、テレメトリなし、画像を外部に送りません。

機能の一覧は[機能カタログ](../docs/ja/catalog.md)、実測値は[ベンチマーク](../BENCHMARKS.md)にあります。

## インストール

ZIP をダウンロードして `install` を実行し、スタートメニュー（macOS では `~/Applications`）の **Seg-Studio** を開きます。ブラウザに `http://localhost:8002/ui/` の画面が開きます。Windows と macOS それぞれのコマンドは [README（日本語）](../README.ja.md)に、そこから学習までの流れは[はじめての手順](../docs/ja/first-run-manual.md)にあります。

Python 3.11 以降が必要です。CUDA の GPU があれば学習はかなり速くなりますが、必須ではありません。CPU でも学習でき、Apple Silicon は MPS で動きます。

## リファレンス

- [インポート / エクスポートの形式](../docs/ja/import_export.md)
- [ロードマップ](../docs/ja/ROADMAP.md) · [変更履歴](../CHANGELOG.md) · [セキュリティポリシー](../SECURITY.md)

## ライセンス

Apache-2.0 です。同梱しているサードパーティのコンポーネントとそのライセンスは、[THIRD_PARTY_NOTICES](https://github.com/segmen-pixel/seg-studio/blob/main/THIRD_PARTY_NOTICES.md) に記載しています。
