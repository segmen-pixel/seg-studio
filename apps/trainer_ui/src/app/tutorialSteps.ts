// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Segmen-Pixel and Seg-Studio contributors
import type { TabId } from "./types";

export type TutorialMode = "beginner" | "intermediate" | "expert";

export type TutorialAdvance =
  | { type: "click"; selector: string }
  | { type: "tabEnter"; tabId: TabId }
  | { type: "manual" };

export type TutorialStep = {
  id: string;
  /** CSS selector for the target element. Null = centered modal (no spotlight). */
  targetSelector: string | null;
  /** Tab the user should be on for this step; if different, we auto-switch. Undefined = any tab. */
  requireTab?: TabId;
  titleJa: string;
  titleEn: string;
  bodyJa: string;
  bodyEn: string;
  /** Condition to auto-advance to the next step. Manual = user must press Next. */
  advanceOn: TutorialAdvance;
  /** Tooltip placement relative to the target. */
  placement?: "top" | "bottom" | "left" | "right" | "center";
  /** Which modes include this step. Undefined = all modes. */
  modes?: TutorialMode[];
  /** Shown on the welcome step only; renders mode-select buttons instead of Next. */
  isModeSelect?: boolean;
  /**
   * When set, the tutorial programmatically clicks the matching element
   * once the step is shown. Use to auto-open a dialog whose contents the
   * next steps will spotlight (e.g. AugmentDialog). One-shot per step
   * activation — re-entering the same step does NOT re-click.
   */
  onEnterClickSelector?: string;
};

/**
 * All tutorial steps in a single ordered list. Filter by mode at runtime via `modes`.
 * Steps with `modes` undefined appear in every mode.
 */
export const TUTORIAL_STEPS: TutorialStep[] = [
  {
    id: "welcome",
    targetSelector: null,
    titleJa: "seg-studio へようこそ！",
    titleEn: "Welcome to seg-studio!",
    bodyJa: "ハンズオンチュートリアルを始めます。自分に合ったモードを選んでください。途中でスキップもできます。",
    bodyEn: "Let's take a hands-on tour. Pick the mode that matches your experience. You can skip any time.",
    advanceOn: { type: "manual" },
    placement: "center",
    isModeSelect: true,
  },
  {
    id: "projects-tab",
    targetSelector: '[data-tutorial-step="projects-tab"]',
    titleJa: "プロジェクトタブ",
    titleEn: "Projects tab",
    bodyJa: "まずは「プロジェクト」タブを開きます。プロジェクトは画像・アノテーション・学習ラン・設定をひとまとめにする作業単位で、案件や検査対象ごとに分けて管理します。各プロジェクトは独立したフォルダに保存され、エクスポート/インポートで他PCへ持ち運べます。",
    bodyEn: "Start by opening the Projects tab. A project is a workspace bundling images, annotations, training runs, and settings together — keep one per job or inspection target. Each project lives in its own folder and can be exported/imported to move between machines.",
    advanceOn: { type: "tabEnter", tabId: "projects" },
    placement: "bottom",
  },
  {
    id: "create-project",
    targetSelector: '[data-tutorial-step="create-project-btn"]',
    requireTab: "projects",
    titleJa: "新規プロジェクトを作成",
    titleEn: "Create a new project",
    bodyJa: "「新規プロジェクト」ボタンで作成します。名前とメモを付けておくと後で探しやすくなります。既存のプロジェクトを開きたい場合はこのステップを飛ばし、下のグリッドから選択してください。ZIP からのインポートで既存データを取り込むこともできます。",
    bodyEn: "Click New Project to create one — a name and a memo make it easy to find later. If you already have a project, skip this and pick it from the grid below, or import existing data from a ZIP.",
    advanceOn: { type: "manual" },
    placement: "bottom",
  },
  {
    id: "annotate-tab",
    targetSelector: '[data-tutorial-step="annotate-tab"]',
    titleJa: "アノテーションタブへ",
    titleEn: "Switch to Annotate tab",
    bodyJa: "次は「アノテーション」タブです。学習用の画像を追加し、検出したい対象に正解マスク（ラベル）を塗ります。ここで作る正解データの質と量がモデル精度を直接左右する、最も重要な工程です。",
    bodyEn: "Next up: the Annotate tab. Add your training images and paint ground-truth masks (labels) on the targets you want to detect. This is the most important stage — the quality and quantity of labels here directly determine model accuracy.",
    advanceOn: { type: "tabEnter", tabId: "annotate" },
    placement: "bottom",
  },
  {
    id: "add-images",
    targetSelector: '[data-tutorial-step="add-images"]',
    requireTab: "annotate",
    titleJa: "画像を追加",
    titleEn: "Add images",
    bodyJa: "「+」ボタンで学習用画像を追加します。個別ファイルのほか、フォルダのドラッグ＆ドロップ、ZIP の一括取り込み、動画からのフレーム抽出にも対応。検出対象のバリエーション（明るさ・角度・個体差）が偏らないように集めるのがコツです。",
    bodyEn: "Use the + button to add training images. Besides individual files, you can drag & drop folders, bulk-import a ZIP, or extract frames from a video. Aim for a varied set (lighting, angle, individual differences) so the model generalizes well.",
    advanceOn: { type: "manual" },
    placement: "bottom",
  },
  {
    id: "annotate-tools",
    targetSelector: '[data-tutorial-step="annotate-tools"]',
    requireTab: "annotate",
    titleJa: "ツールパレット",
    titleEn: "Tool palette",
    bodyJa: "ブラシ・消しゴム・バケツ（塗りつぶし）などの基本ツールがあります。右パネルで選択し、ブラシサイズを調整しながら対象を塗ります。ホイールでズーム、スペースドラッグで移動。次に紹介する AI ツールと組み合わせると作業が大幅に速くなります。",
    bodyEn: "Basic tools: brush, eraser, and bucket (fill). Pick one in the right panel and paint, adjusting brush size as you go. Scroll to zoom, space-drag to pan. Combine these with the AI tools shown next to speed things up dramatically.",
    advanceOn: { type: "manual" },
    placement: "left",
    modes: ["beginner", "intermediate", "expert"],
  },
  // --- Intermediate: AI-assist tools ---
  {
    id: "tool-sam",
    targetSelector: '[data-tutorial-step="annotate-tools"]',
    requireTab: "annotate",
    titleJa: "AIツール: SAM",
    titleEn: "AI tool: SAM",
    bodyJa: "SAM（Segment Anything Model）は、画像をクリックするだけで AI がその対象の領域を推定し、マスク候補を自動生成する汎用セグメンテーションです。複雑な輪郭や曲線も一瞬で囲めるので、手塗りの手間を大幅に削減できます。生成された候補は微調整も可能で、学習不要でどんな対象にも使えます。",
    bodyEn: "SAM (Segment Anything Model) is a general-purpose segmenter: click a target and the AI estimates its region and proposes a mask. It captures complex outlines and curves instantly, cutting manual painting time sharply. The proposal can be fine-tuned afterward, and it needs no training to work on any object.",
    advanceOn: { type: "manual" },
    placement: "left",
    modes: ["intermediate", "expert"],
  },
  {
    id: "tool-spotdetect",
    targetSelector: '[data-tutorial-step="annotate-tools"]',
    requireTab: "annotate",
    titleJa: "AIツール: スポット検出",
    titleEn: "AI tool: Spot Detector",
    bodyJa: "クリックした点を基準に、色・輝度が近い小さな欠陥（点状）を周辺から自動検出してまとめてマスク化します。微小NGやパーティクル、ブツのように数が多く一つずつ塗ると大変な対象に最適。しきい値を調整すれば、拾う範囲の厳しさを変えられます。",
    bodyEn: "From the point you click, it auto-detects nearby small defects (dots) of similar color/brightness and masks them all at once. Ideal for tiny defects, particles, and specks that are tedious to paint one by one. Tune the threshold to control how aggressively it picks them up.",
    advanceOn: { type: "manual" },
    placement: "left",
    modes: ["intermediate", "expert"],
  },
  {
    id: "tool-superpixel",
    targetSelector: '[data-tutorial-step="annotate-tools"]',
    requireTab: "annotate",
    titleJa: "AIツール: スーパーピクセル",
    titleEn: "AI tool: Superpixel",
    bodyJa: "画像を色やテクスチャが均質な小領域（スーパーピクセル）に自動分割し、境界を物体の輪郭に沿わせます。あとは対象を構成するピースをクリックしていくだけでマスクが完成。ピクセル単位で塗るより速く、輪郭も綺麗に出ます。分割の粒度は調整可能です。",
    bodyEn: "Auto-segments the image into small color/texture-uniform regions (superpixels) whose borders hug object edges. Just click the pieces that make up your target to build the mask — faster than pixel painting and with cleaner edges. The granularity of the split is adjustable.",
    advanceOn: { type: "manual" },
    placement: "left",
    modes: ["intermediate", "expert"],
  },
  {
    id: "tool-crack",
    targetSelector: '[data-tutorial-step="annotate-tools"]',
    requireTab: "annotate",
    titleJa: "AIツール: クラック追跡",
    titleEn: "AI tool: Crack Tracer",
    bodyJa: "クラックやキズのような線状の欠陥を、始点と終点をクリックするだけで経路を自動追跡してマスク化します。曲がった線も画像の濃淡をたどって繋いでくれるので、細い線を1ピクセルずつ塗る手間から解放されます。線の太さは後から調整できます。",
    bodyEn: "For line-shaped defects like cracks and scratches, click a start and end point and it auto-traces the path into a mask. It follows the image's intensity to connect even curved lines, freeing you from painting thin lines pixel by pixel. Line thickness can be adjusted afterward.",
    advanceOn: { type: "manual" },
    placement: "left",
    modes: ["intermediate", "expert"],
  },
  {
    id: "annotate-classes",
    targetSelector: '[data-tutorial-step="annotate-classes"]',
    requireTab: "annotate",
    titleJa: "クラス一覧",
    titleEn: "Class list",
    bodyJa: "検出したい対象の種類（クラス）を管理します。「キズ」「異物」「欠け」のように欠陥タイプごとにクラスを分けると、推論時に種類別の検出ができます。各クラスに色を割り当て、塗る前に対象のクラスを選択します。1クラスだけ（欠陥 or 正常の2値）でも学習できます。",
    bodyEn: "Manage the categories (classes) you want to detect. Splitting them by defect type — e.g. scratch, foreign object, chip — lets the model report detections per class at inference. Assign each class a color and select the right class before painting. A single class (defect vs. normal) is enough to train, too.",
    advanceOn: { type: "manual" },
    placement: "left",
  },
  {
    id: "annotate-save",
    targetSelector: null,
    requireTab: "annotate",
    titleJa: "自動保存されます",
    titleEn: "Autosave",
    bodyJa: "塗った内容は操作ごとに自動保存されるので、保存忘れの心配はありません（Ctrl+S で手動保存も可）。学習にはまず 10 枚以上のアノテーション済み画像が目安ですが、対象のバリエーションが多いほど精度は安定します。全部塗り切らなくても、代表的な数枚から試し始められます。",
    bodyEn: "Every edit autosaves, so there's no risk of losing work (Ctrl+S also saves manually). Aim for at least ~10 annotated images to start training — the more variety, the more stable the accuracy. You don't have to finish everything; start experimenting from a few representative images.",
    advanceOn: { type: "manual" },
    placement: "center",
  },
  // Perlin CutPaste — keep the prose minimal and let the user click the
  // 拡張 button to see the actual dialog. The button's onClick opens the
  // AugmentDialog, and advanceOn={type:"click"} pushes the tutorial
  // forward at the same time so the next step (annotate-augment-dialog)
  // lands while the dialog is open.
  // Perlin synthesis — tutorial programmatically opens AugmentDialog so the
  // user can see the actual UI; the next step spotlights the preview inside
  // the open dialog.
  {
    id: "annotate-augment",
    targetSelector: '[data-tutorial-step="annotate-augment"]',
    requireTab: "annotate",
    onEnterClickSelector: '[data-tutorial-step="annotate-augment"]',
    titleJa: "「拡張」で欠陥水増し",
    titleEn: "Synthesise defects with Augment",
    bodyJa: "ダイアログを開きました。Perlin 変形で既存欠陥をコピペ合成して dataset に追加できます。",
    bodyEn: "Dialog opened. Perlin-warp and paste your existing defects to synthesise new images into the dataset.",
    advanceOn: { type: "manual" },
    placement: "right",
    modes: ["intermediate", "expert"],
  },
  {
    id: "annotate-augment-dialog",
    targetSelector: ".augment-preview-canvas",
    requireTab: "annotate",
    titleJa: "プレビューで効果を確認",
    titleEn: "Preview the effect",
    bodyJa: "プレビューはスライダーを動かすたびに即更新。生成枚数・Perlin 強度・色ジッターを決めて「生成」で dataset に追加されます。",
    bodyEn: "The preview updates live. Pick generation count, Perlin strength, and colour jitter, then press Generate to add to the dataset.",
    advanceOn: { type: "manual" },
    placement: "right",
    modes: ["intermediate", "expert"],
  },
  {
    id: "training-tab",
    targetSelector: '[data-tutorial-step="training-tab"]',
    titleJa: "学習タブへ",
    titleEn: "Switch to Training tab",
    bodyJa: "「学習」タブでアノテーション済みデータからモデルを学習します。学習モードの選択、ハイパーパラメータの設定、学習の実行と進捗監視をここで行います。GPU があれば自動で使われ、無い場合は CPU で実行されます。",
    bodyEn: "The Training tab turns your annotated data into a model: pick a training mode, set hyperparameters, launch training, and watch its progress here. A local GPU is used automatically when present; otherwise training falls back to CPU.",
    advanceOn: { type: "tabEnter", tabId: "training" },
    placement: "bottom",
  },
  // --- Intermediate: training modes overview — spotlight the actual GUI
  //     buttons (.training-mode-select wraps all 4 cards) instead of
  //     duplicating them as an SVG illustration in the popup. ---
  {
    id: "training-modes",
    targetSelector: ".training-mode-select",
    requireTab: "training",
    titleJa: "4つの学習モード",
    titleEn: "Four training modes",
    bodyJa: "Standard / Quick / Transfer の 3 つから選びます。各カードの「?」で詳しい説明、クリックでそのモードに切替。",
    bodyEn: "Pick one of Standard / Quick / Transfer. Each card has a ? for details; click the card to select that mode.",
    advanceOn: { type: "manual" },
    placement: "bottom",
    modes: ["intermediate", "expert"],
  },
  // --- Open the detailed (hyperparameter) panel, then point at real controls (form order) ---
  {
    id: "open-hyperparams",
    targetSelector: ".training-hyper-toggle",
    requireTab: "training",
    titleJa: "詳細設定を開く",
    titleEn: "Open detailed settings",
    bodyJa: "「詳細設定」を開くと、アーキテクチャ・データ拡張・重み/損失などのハイパーパラメータが4カテゴリで表示されます。まずここを開いてから、主要な設定を順に見ていきましょう。",
    bodyEn: "Open Detailed Settings to reveal the hyperparameters — architecture, augmentation, weights and loss — grouped into four categories. Open it first, then we will walk through the key ones.",
    advanceOn: { type: "click", selector: ".training-hyper-toggle" },
    placement: "bottom",
    modes: ["intermediate", "expert"],
  },
  // --- Expert: Auto Config (master switch above the 4 category columns) ---
  {
    id: "expert-auto-select",
    targetSelector: '[data-tutorial-step="auto-config-master"]',
    requireTab: "training",
    titleJa: "Auto Config (自動推薦)",
    titleEn: "Auto Config (recommendations)",
    bodyJa: "学習開始時、サーバーがデータセットの背景特徴量を解析して arch (アーキテクチャ) / モデルサイズ (base_channels) / patch_size / DINOv2 蒸留 ON/OFF を自動で上書きします。ON 中は対象 4 項目が dim 表示 + (auto) バッジ付きになり、手動で値を変更するとこのスイッチが自動 OFF に切り替わります。",
    bodyEn: "At training start the server analyses background features in your dataset and auto-overrides arch / model size (base_channels) / patch_size / DINOv2 distill on the form. While ON, those four fields render dimmed with an (auto) badge; editing any of them manually flips this master switch OFF automatically.",
    advanceOn: { type: "manual" },
    placement: "right",
    modes: ["expert"],
  },
  {
    id: "training-start",
    targetSelector: '[data-tutorial-step="training-start"]',
    requireTab: "training",
    titleJa: "学習を開始",
    titleEn: "Start training",
    bodyJa: "「学習開始」ボタンで、選択したハイパーパラメータによる学習が始まります。学習中は進捗バー・損失・検証 F1 スコアがリアルタイムに表示され、エポックごとに精度が上がっていく様子を確認できます。途中で停止しても、その時点までのベストモデルが保存されます。",
    bodyEn: "Press Start to begin training with the selected hyperparameters. Progress, loss, and validation F1 update live, so you can watch accuracy climb epoch by epoch. If you stop early, the best model up to that point is kept.",
    advanceOn: { type: "manual" },
    placement: "right",
  },
  {
    id: "training-results",
    targetSelector: '[data-tutorial-step="training-runs"]',
    requireTab: "training",
    titleJa: "学習完了→結果を見る",
    titleEn: "View results after training",
    bodyJa: "学習が完了するとランリストに結果（F1 などの指標）が表示されます。未確認のランは「結果を見る」ボタン（棒グラフアイコン）が青く点滅して知らせてくれます。複数ランを並べて比較でき、良かった設定は転移学習の元モデルとして再利用できます。",
    bodyEn: "When a run finishes, its results (F1 and other metrics) appear in the run list, and any unseen run pulses its blue result button (bar-chart icon) to flag it. You can compare runs side by side and reuse a strong one as the base model for transfer learning.",
    advanceOn: { type: "manual" },
    placement: "right",
  },
  // --- Intermediate+: Results-tab overview (SVG preview) ---
  {
    id: "results-overview",
    targetSelector: null,
    titleJa: "結果タブの見方",
    titleEn: "Reading the Results tab",
    bodyJa: "結果タブでは学習成果を多角的に確認できます。①ヒートマップ＝予測スコアの分布（どこを怪しいと見たか）、②F1 学習曲線＝epoch ごとの精度推移（頭打ち/過学習の判断）、③混同行列＝クラス間の取り違え、④マスク予測＝入力画像と推論結果の重ね合わせ。これらを見て、追加アノテやパラメータ調整の方針を立てます。",
    bodyEn: "The Results tab gives several lenses on your model: ① a heatmap of prediction scores (where it suspected a defect), ② the F1-over-epoch curve (to spot plateaus or overfitting), ③ a confusion matrix (class mix-ups), and ④ mask predictions (input overlaid with the result). Use them to decide where to add annotations or tune parameters.",
    advanceOn: { type: "manual" },
    placement: "center",
    modes: ["intermediate", "expert"],
  },
  // --- Expert: post-processing & export & report ---
  {
    id: "expert-postprocess",
    targetSelector: null,
    titleJa: "CCA後処理 (min_area)",
    titleEn: "CCA post-processing (min_area)",
    bodyJa: "推論結果に連結成分分析（CCA＝隣接する前景ピクセルを1つの塊にまとめる処理）をかけ、min_area 未満の小さな塊を除去します。孤立したノイズ検出や点状の誤検出を消すのに効果的で、閾値は『実在する最小欠陥のサイズ』を基準に決めます（大きすぎると本物の小欠陥も消えます）。max_area を併用すれば大きすぎる誤検出も除けます。",
    bodyEn: "Runs connected-component analysis (CCA — grouping touching foreground pixels into blobs) on the prediction and drops any blob below min_area. Great for clearing isolated noise and dot-like false positives; set the threshold from your smallest real defect (too high removes genuine small defects too). Pair it with max_area to also remove implausibly large false positives.",
    advanceOn: { type: "manual" },
    placement: "center",
    modes: ["expert"],
  },
  {
    id: "expert-export",
    targetSelector: null,
    titleJa: "モデルのエクスポート",
    titleEn: "Export your model",
    bodyJa: "学習完了したランは CoreML / ONNX / OpenVINO にエクスポートできます。CoreML は iOS/macOS のオンデバイス、ONNX は Web や各種ランタイム、OpenVINO は Intel CPU/GPU 向けです。OpenVINO は FP32 / FP16 / INT8 の精度を選べ、INT8 ほど軽量・高速になります（精度は微減）。組み込み先に合わせて形式と精度を選びます。",
    bodyEn: "Completed runs export to CoreML, ONNX, or OpenVINO. CoreML targets on-device iOS/macOS, ONNX is portable for the Web and many runtimes, and OpenVINO targets Intel CPUs/GPUs. OpenVINO offers FP32 / FP16 / INT8 precision — lower precision is smaller and faster with a slight accuracy drop. Choose the format and precision to match your deployment target.",
    advanceOn: { type: "manual" },
    placement: "center",
    modes: ["expert"],
  },
  {
    id: "expert-inspect",
    targetSelector: null,
    titleJa: "Live Inspection",
    titleEn: "Live Inspection",
    bodyJa: "設定で Inspect タブ（ライブ検査）を有効にすると、学習済みモデルで新しい画像をその場で推論し、予測マスクを重ねて確認できます。入力はカメラ映像またはファイル/フォルダに対応。デプロイ前に本番に近い画像でモデルを素早く確認でき、閾値や後処理（min_area 等）を変えながらリアルタイムに効果を見られます。",
    bodyEn: "Enable the Inspect tab (Live Inspection) in Settings to run your trained model on new images on the spot, overlaying the predicted masks. It accepts a camera feed or files/folders. Sanity-check the model on production-like images before deploying — and tweak the threshold and post-processing (min_area, etc.) with real-time feedback.",
    advanceOn: { type: "manual" },
    placement: "center",
    modes: ["expert"],
  },
  {
    id: "expert-report",
    targetSelector: null,
    titleJa: "評価レポート生成",
    titleEn: "Evaluation report",
    bodyJa: "結果ページのヘッダーのレポートアイコンから、全画像の推論結果と統計をまとめた HTML レポートを生成できます。精度・再現率・F1、ハードケース画像、スコア分布などをセクション単位で含められ、ブラウザ印刷で A4 PDF 化も可能。チーム共有・トレーサビリティ・顧客提出に使え、出力するセクションはチェックボックスで選べます。",
    bodyEn: "From the report icon in the results header, generate an HTML report of predictions and statistics across all images. Include precision/recall/F1, hard-case images, and the score distribution as toggleable sections — and print to an A4 PDF from the browser. Useful for team sharing, traceability, and customer deliverables; choose which sections to include via checkboxes.",
    advanceOn: { type: "manual" },
    placement: "center",
    modes: ["expert"],
  },
  {
    id: "done",
    targetSelector: null,
    titleJa: "チュートリアル完了！",
    titleEn: "Tutorial complete!",
    bodyJa: "お疲れさまでした。ヘッダーの▶︎ボタンでいつでもこのチュートリアルを再生できます。説明モードを ON にすると、ボタンにカーソルを合わせるだけで機能説明が表示されます。",
    bodyEn: "Nicely done. Replay this tutorial any time via the ▶︎ button in the header. Enable description mode for hover-to-read explanations on every button.",
    advanceOn: { type: "manual" },
    placement: "center",
  },
];

export function getStepsForMode(mode: TutorialMode): TutorialStep[] {
  return TUTORIAL_STEPS.filter((s) => !s.modes || s.modes.includes(mode));
}
