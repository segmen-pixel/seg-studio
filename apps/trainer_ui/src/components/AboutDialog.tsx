// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import React from "react";
import type { HealthInfo } from "../api";
import { useI18n } from "../i18n";

const CHANGELOG = [
  {
    version: "0.9.8",
    date: "2026-07-27",
    changes: [
      "🆕 インスタンス（個数カウント）学習モードを追加 — 触れ合う/重なる対象の個数を数える。追加のアノテーション作業は不要で、既存のセマンティックマスクから学習データを合成する",
      "🎯 数え方は検証データで較正したしきい値 + マスク IoU 0.7 の重複除去。1 個の対象が二重検出される DETR 系の癖を抑制",
      "🎯 1 モデルで複数クラスを同時にカウント。クラス対応はエクスポートした推論契約にも引き継がれる",
      "🖥 モデルサイズ Small（既定）/ Medium / Large。検出した VRAM に合わせてバッチを自動調整（実効バッチは維持）",
      "📤 ONNX エクスポートと serving_api の /count エンドポイントに対応",
      "🎨 結果ビューに番号付きインスタンス表示（Okabe-Ito パレット）とクラス別カウントチップを追加",
      "🛡 rfdetr は Apache-2.0 の本体と Seg 重みのみ採用。PML-1.0 の \"plus\" 拡張と検出 XL/2XL は対象外",
      "🧹 バージョン表記のずれを解消（バックエンドの一部が 1 リリース古い値を返していた）",
    ],
  },
  {
    version: "0.9.7",
    date: "2026-07-17",
    changes: [
      "🧹 異常検知モードを完全撤去（統計 z-score 1 本の有名無実化を解消、異常検知は AnomaLens へ）",
      "☁ クラウド学習オプションを OSS 配布から除外（ローカル完結スコープに集約、実装は内部保管）",
      "🧹 自動ドナー warm-start を撤去（明示的な転移学習 / モデル検索は不変）",
      "🐛 auto_mode=\"off\" 指定時に arch が黙って書き換わる不具合を修正（リクエスト値を verbatim 尊重）",
    ],
  },
  {
    version: "0.9.6",
    date: "2026-07-08",
    changes: [
      "🎯 学習デフォルトを wave1-6 EDA から刷新（base_channels=128 / fg_patch_prob=0.7 ほか）",
      "📊 auto-config: ML 予測器が使えない際の z-score フォールバック理由を学習ログに明示",
      "🎯 best_model_v6 バンドルを 26 特徴フル充填で再学習（LOPO top-5 hit 5.4%→10.8%）",
    ],
  },
  {
    version: "0.9.5",
    date: "2026-05-08",
    changes: [
      "🛡 ライセンス整備: 蒸留 teacher を DINOv2 / SAM2 (Apache-2.0) のみに統一",
      "🛡 DINOv2 hub source tree の同梱停止 (NC ライセンス混入を回避)、weight だけ Apache-2.0 で bundle",
      "📜 LGPL 全文 (libvips / Cairo / Pango / FFmpeg / pyphen) を licenses/third_party/lgpl/ に同梱",
      "📜 PyTorch CUDA wheel 経由の NVIDIA libs (cuDNN / cuBLAS 等) ライセンス文を installer に伝搬",
      "🧹 distillation API スリム化: precompute / ensemble エンドポイント削除、online teacher のみサポート",
      "🧹 NOTICE / THIRD_PARTY_NOTICES.md / MODEL_WEIGHTS.md 整備、商標 disclaimer 追加",
    ],
  },
  {
    version: "0.9.4",
    date: "2026-04-17",
    changes: [
      "🎓 ハンズオンチュートリアル: 初級/中級/エキスパートの3モード、スポットライトガイド + SVGアニメ12種",
      "🎯 学習完了ランの結果ボタンが青く点滅（説明モードON時、未確認だけ）",
      "🎯 説明モード中、次に見るべきタブ（Projects/Annotate/Training）をガイド点滅",
      "⌨ チュートリアル内でキーボード操作: Enter/→/←/Esc、モード選択 1/2/3",
      "🎨 ヘッダーに ▶︎ チュートリアル再生ボタンを追加",
    ],
  },
  {
    version: "0.9.3",
    date: "2026-04-10",
    changes: [
      "🔍 異常検知: auto-config、パッチ学習、ステップ進捗表示",
      "🎨 結果タブ: GT/推論マスク分離表示、パターンメニュー（斜線/ドット/格子）",
      "🎨 説明モード: ツールチップをJS方式に全面刷新（レイアウト崩壊解消）",
      "🖱 ブラシサイズ改善、スライダーUI修正（端まで到達）",
      "📁 フォルダインポート対応、ネストフォルダ名ヘッダー表示",
      "📁 バッチプロジェクトエクスポート",
      "⚡ SW推論 max集約、キャッシュ無効化修正",
      "⚡ 学習キュー: 予約run自動起動、起動時キュー復帰",
      "🖥 ライブ検査: オーバーレイラベル、フィルタ、topbar刷新",
      "🧹 Mark Clean機能（欠陥なし画像マーク）",
      "🌐 結果タブUI全日本語化",
      "🐛 デュアルGPU対応修正（全デバイス使用中のみbusy判定）",
      "🐛 リージョンラベル/クラスリストの確信度閾値反映",
      "🐛 予約run連鎖クラッシュ修正（GPUクールダウン+リトライ）",
    ],
  },
  {
    version: "0.9.2",
    date: "2026-04-04",
    changes: [
      "🐛 ファイル名に # を含む画像の表示修正（URL encodeURIComponent対応）",
      "🐛 起動時API 404ラッシュ修正（ルーター登録完了まで503返却）",
      "🐛 skimage .pyi stub欠落修正（superpixel機能復旧）",
      "🐛 ORT CUDA EP不可修正（onnxruntime CPU版上書き問題解消）",
      "🐛 combo_library.json パス不正修正（パッケージ内同梱）",
      "🐛 DINOv2チェックポイント破損対策（インストーラーにバンドル）",
      "🖱 画像リスト: Shift/Ctrl+クリックでアクティブ画像が切り替わらないよう修正",
      "🖱 画像リスト: 選択ハイライト強化（枠線+背景色）",
      "🖱 画像リスト: Shift+矢印の範囲選択修正（戻り方向で選択解除）",
      "🖱 画像リスト: Deleteキーで選択画像削除",
      "📁 リネームインポート機能（prefix/suffix + D&D並び替え）",
      "🧹 セーブボタン削除（自動保存で不要）",
      "🎨 favicon.ico追加",
    ],
  },
  {
    version: "0.9.1",
    date: "2026-03-23",
    changes: [
      "☁ GPU天気アイコン+残高+本日消費額表示",
      "☁ 推論結果自動ダウンロード（predictions.zip方式）",
      "☁ クラウドrun停止ボタン+切断ボタン",
      "🔍 異常検知モジュール（OK画像のみで異常箇所検出）",
      "🎯 学習モード選択UI（通常/異常検知/クイック/転移学習）",
      "📊 ピクセルレベルFG Confidenceヒストグラム",
      "⚡ 並列画像インポート（10枚バッチ+マルチスレッド変換）",
      "📁 重複画像インポート時のダイアログ（スキップ/上書き/両方残す）",
      "🌐 日英翻訳キー追加（40+クラウド関連文字列）",
      "🧹 デッドコード掃除（print→logger、removedコメント削除）",
      "🐛 エクスポート時の日本語ファイル名修正",
      "🐛 sample_weightsバグ修正（distill fallback時）",
      "🐛 onnx lazy import修正（ルーター登録失敗）",
    ],
  },
  {
    version: "0.9.0-beta",
    date: "2026-03-20",
    changes: [
      "Zarrマスクストア移行",
      "GPUリソースロック（学習/推論中のSAM無効化）",
      "学習フェーズステータス表示",
      "Windows DataLoader修正（num_workers=0固定）",
    ],
  },
];

const LICENSE_ENTRIES = [
  "FastAPI — MIT",
  "Uvicorn — BSD-3-Clause",
  "SQLModel — MIT",
  "Pydantic — MIT",
  "python-multipart — Apache-2.0",
  "ONNX — Apache-2.0",
  "PyTorch — BSD-style",
  "torchvision — BSD-3-Clause",
  "NumPy — Modified BSD",
  "Pillow — MIT-CMU",
  "httpx — BSD-3-Clause",
  "OpenCV — Apache-2.0",
  "opencv-python (wheel) — MIT; FFmpeg LGPLv2.1",
  "scikit-learn — BSD-3-Clause",
  "scikit-image — BSD-3-Clause",
  "onnxruntime — MIT",
  "transformers — Apache-2.0",
  "MobileSAM — Apache-2.0",
  "SAM 2 — Apache-2.0",
  "TinySAM — Apache-2.0",
  "EfficientSAM — Apache-2.0",
  "React / React DOM — MIT",
  "Zustand — MIT",
  "Immer — MIT",
  "Vite — MIT",
];

interface AboutDialogProps {
  open: boolean;
  onClose: () => void;
  healthInfo: HealthInfo | null;
}

const AboutDialog: React.FC<AboutDialogProps> = ({ open, onClose, healthInfo }) => {
  const { t } = useI18n();
  if (!open) return null;
  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-panel about-panel" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <h2>{t("about.title")}</h2>
          <button className="ghost" onClick={onClose} data-desc={t("common.close")} data-desc-pos="bottom">×</button>
        </div>
        <section className="about-info">
          <div className="about-title">Seg-Studio</div>
          <div className="about-meta">
            <span>v{__APP_VERSION__}</span>
            <span className="muted">Build: {__BUILD_DATE__}</span>
          </div>
          <div className="muted" style={{ fontSize: 12 }}>Copyright 2026 Segmen-Pixel and Seg-Studio contributors. Apache License 2.0.</div>
        </section>
        {healthInfo && (
          <section>
            <h3>{t("about.system")}</h3>
            <div className="about-system-grid">
              {healthInfo.disk && (
                <div className="about-system-item">
                  <span className="about-system-label">Disk</span>
                  <span>{healthInfo.disk.free_gb} GB free / {healthInfo.disk.total_gb} GB ({healthInfo.disk.used_pct}% used)</span>
                </div>
              )}
              {healthInfo.ram && (
                <div className="about-system-item">
                  <span className="about-system-label">RAM</span>
                  <span>{healthInfo.ram.available_gb} GB free / {healthInfo.ram.total_gb} GB ({healthInfo.ram.used_pct}% used)</span>
                </div>
              )}
              {healthInfo.gpu && (
                <div className="about-system-item">
                  <span className="about-system-label">GPU</span>
                  <span>{healthInfo.gpu.name} — {healthInfo.gpu.vram_total_mb} MB VRAM</span>
                </div>
              )}
            </div>
          </section>
        )}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          <section>
            <h3>{t("about.licenses")}</h3>
            <div className="about-license-scroll">
              <ul className="license-list">
                {LICENSE_ENTRIES.map((e) => <li key={e}>{e}</li>)}
              </ul>
            </div>
          </section>
          <section>
            <h3>変更履歴</h3>
            <div className="about-license-scroll">
              {CHANGELOG.map((entry) => (
                <div key={entry.version} style={{ marginBottom: 12 }}>
                  <div style={{ fontWeight: 600, fontSize: 13 }}>
                    v{entry.version} <span className="muted" style={{ fontWeight: 400 }}>({entry.date})</span>
                  </div>
                  <ul style={{ margin: "4px 0 0", paddingLeft: 20, fontSize: 12, lineHeight: 1.6 }}>
                    {entry.changes.map((c, i) => <li key={i}>{c}</li>)}
                  </ul>
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};

export default React.memo(AboutDialog);
