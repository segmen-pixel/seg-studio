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
    bodyJa: "実際に操作しながら、使い方を順に紹介します。経験に合わせてモードを選んでください。スキップはいつでもできます。",
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
    bodyJa: "まず「プロジェクト」タブを開きます。プロジェクトは、画像、アノテーション、学習ラン、設定をまとめた作業の単位で、案件や検査対象ごとに1つずつ作ります。プロジェクトはそれぞれ専用のフォルダに保存され、エクスポートとインポートでほかの PC に移せます。",
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
    bodyJa: "「プロジェクト作成」ボタンで作成します。名前と説明を入れておくと、あとで探しやすくなります。すでにプロジェクトがある場合は、この手順を飛ばして下の一覧から選んでください。既存のデータを ZIP からインポートすることもできます。",
    bodyEn: "Click New Project to create one — a name and a memo make it easy to find later. If you already have a project, skip this and pick it from the grid below, or import existing data from a ZIP.",
    advanceOn: { type: "manual" },
    placement: "bottom",
  },
  {
    id: "annotate-tab",
    targetSelector: '[data-tutorial-step="annotate-tab"]',
    titleJa: "ラベルタブへ",
    titleEn: "Switch to Annotate tab",
    bodyJa: "次は「ラベル」タブです。学習用の画像を追加し、検出したい対象に正解のマスク（ラベル）を塗ります。ここで作るラベルの質と量がモデルの精度を直接決めるので、もっとも重要な工程です。",
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
    bodyJa: "「+」ボタンで学習用の画像を追加します。ファイルを1つずつ選ぶほか、フォルダのドラッグ＆ドロップ、ZIP の一括取り込み、動画からのフレームの切り出しもできます。モデルがさまざまな画像に対応できるよう、明るさ、角度、個体差が偏らないように集めるのがコツです。",
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
    bodyJa: "ブラシ、消しゴム、塗りつぶしなどの基本のツールがあります。右のパネルでツールを選び、ブラシのサイズを調整しながら対象を塗ります。ホイールでズームし、Space キーを押しながらドラッグすると表示を移動できます。次に紹介する AI ツールと組み合わせると、作業がずっと速くなります。",
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
    titleJa: "AI ツール：SAM",
    titleEn: "AI tool: SAM",
    bodyJa: "SAM（Segment Anything Model）は汎用のセグメンテーションモデルです。対象をクリックすると、AI がその領域を推定してマスクの候補を出します。複雑な輪郭や曲線もすぐに囲めるので、手で塗る手間が大きく減ります。候補はあとから細かく調整できます。学習しなくても、どんな対象にも使えます。",
    bodyEn: "SAM (Segment Anything Model) is a general-purpose segmenter: click a target and the AI estimates its region and proposes a mask. It captures complex outlines and curves instantly, cutting manual painting time sharply. The proposal can be fine-tuned afterward, and it needs no training to work on any object.",
    advanceOn: { type: "manual" },
    placement: "left",
    modes: ["intermediate", "expert"],
  },
  {
    id: "tool-spotdetect",
    targetSelector: '[data-tutorial-step="annotate-tools"]',
    requireTab: "annotate",
    titleJa: "AI ツール：スポット検出",
    titleEn: "AI tool: Spot Detector",
    bodyJa: "見本のスポットを1つブラシで塗って「検出」を押すと、画像全体から似たスポットを探して、まとめてマスクにします。Enter で確定します。微小な欠陥、パーティクル、ブツのように、数が多くて1つずつ塗るのが大変な対象に向いています。しきい値を調整すると、どこまで拾うかを変えられます。",
    bodyEn: "Paint one example spot with the brush, press Detect, and it finds similar spots across the whole image; Enter to confirm. Ideal for tiny defects, particles, and specks that are tedious to paint one by one. Tune the threshold to control how aggressively it picks them up.",
    advanceOn: { type: "manual" },
    placement: "left",
    modes: ["intermediate", "expert"],
  },
  {
    id: "tool-crack",
    targetSelector: '[data-tutorial-step="annotate-tools"]',
    requireTab: "annotate",
    titleJa: "AI ツール：ひび割れ検出",
    titleEn: "AI tool: Crack Tracer",
    bodyJa: "ひび割れやキズのような線状の欠陥に使います。ひび割れの上をクリックすると、つながった線を自動でたどってマスクにします。Enter で確定します。画像の濃淡をたどるので曲がった線もつながり、細い線を1ピクセルずつ塗る必要はありません。線の太さはあとから調整できます。",
    bodyEn: "For line-shaped defects like cracks and scratches, click on the crack and it auto-traces the connected path into a mask; Enter to confirm. It follows the image's intensity to connect even curved lines, freeing you from painting thin lines pixel by pixel. Line thickness can be adjusted afterward.",
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
    bodyJa: "検出したい対象の種類（クラス）を管理します。「キズ」「異物」「欠け」のように欠陥の種類ごとにクラスを分けると、推論のときに種類ごとに検出できます。クラスごとに色を割り当て、塗る前に正しいクラスを選んでください。クラスが1つだけ（欠陥か正常かの2値）でも学習できます。",
    bodyEn: "Manage the categories (classes) you want to detect. Splitting them by defect type — e.g. scratch, foreign object, chip — lets the model report detections per class at inference. Assign each class a color and select the right class before painting. A single class (defect vs. normal) is enough to train, too.",
    advanceOn: { type: "manual" },
    placement: "left",
  },
  {
    id: "annotate-save",
    targetSelector: null,
    requireTab: "annotate",
    titleJa: "自動保存",
    titleEn: "Autosave",
    bodyJa: "塗った内容は操作のたびに自動で保存されるので、保存し忘れる心配はありません（Ctrl+S で手動でも保存できます）。学習を始めるには、アノテーション済みの画像が10枚以上あるのが目安です。画像のバリエーションが多いほど、精度は安定します。すべて塗り終えていなくても、代表的な数枚から試し始められます。",
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
    titleJa: "「拡張」で欠陥を合成",
    titleEn: "Synthesise defects with Augment",
    bodyJa: "ダイアログを開きました。既存の欠陥を Perlin 変形して貼り付け、合成した画像をデータセットに追加できます。",
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
    bodyJa: "プレビューはスライダーを動かすとすぐに更新されます。生成枚数、Perlin 変形強度、色ジッターを決めて「生成」を押すと、データセットに追加されます。",
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
    bodyJa: "「学習」タブでは、アノテーション済みのデータからモデルを学習します。学習モードの選択、ハイパーパラメータの設定、学習の開始と進み具合の確認をここで行います。GPU があれば自動で使い、無ければ CPU で学習します。",
    bodyEn: "The Training tab turns your annotated data into a model: pick a training mode, set hyperparameters, launch training, and watch its progress here. A local GPU is used automatically when present; otherwise training falls back to CPU.",
    advanceOn: { type: "tabEnter", tabId: "training" },
    placement: "bottom",
  },
  // --- Intermediate: training modes overview — spotlight the actual GUI
  //     buttons (.training-mode-select wraps both task cards) instead of
  //     duplicating them as an SVG illustration in the popup. ---
  {
    id: "training-modes",
    targetSelector: ".training-mode-select",
    requireTab: "training",
    titleJa: "2つのタスク",
    titleEn: "Two tasks",
    bodyJa: "「欠陥検知」か「個数検査」のどちらかを選びます。各カードの「?」で詳しい説明を表示し、カードをクリックするとそのタスクに切り替わります。",
    bodyEn: "Pick Defect detection or Count inspection. Each card has a ? for details; click the card to select it.",
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
    bodyJa: "「詳細設定」を開くと、アーキテクチャ、データ拡張、重みと損失などのハイパーパラメータが4つのカテゴリに分かれて表示されます。まずここを開いてから、主な設定を順に見ていきましょう。",
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
    titleJa: "Auto Config（自動推薦）",
    titleEn: "Auto Config (recommendations)",
    bodyJa: "学習を始めるときに、サーバーがデータセットの背景の特徴量を解析し、フォームの arch（アーキテクチャ）、モデルサイズ（base_channels）、patch_size、DINOv2 蒸留のオン／オフを自動で上書きします。オンの間は、この4項目が薄く表示され、「自動」バッジが付きます。どれかを手で変更すると、このスイッチは自動でオフになります。",
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
    bodyJa: "「学習開始」ボタンを押すと、選んだハイパーパラメータで学習が始まります。学習中は進捗バー、損失、検証の F1 スコアがリアルタイムで表示され、エポックごとに精度が上がっていく様子を確認できます。途中で止めても、その時点までのベストモデルが保存されます。",
    bodyEn: "Press Start to begin training with the selected hyperparameters. Progress, loss, and validation F1 update live, so you can watch accuracy climb epoch by epoch. If you stop early, the best model up to that point is kept.",
    advanceOn: { type: "manual" },
    placement: "right",
  },
  {
    id: "training-results",
    targetSelector: '[data-tutorial-step="training-runs"]',
    requireTab: "training",
    titleJa: "学習後に結果を見る",
    titleEn: "View results after training",
    bodyJa: "学習が終わると、ランの一覧に結果（F1 などの指標）が表示されます。まだ見ていないランは、「結果を見る」ボタン（棒グラフのアイコン）が青く点滅して知らせます。複数のランを並べて比べられ、結果の良いランは転移学習の元のモデルとして使えます。",
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
    bodyJa: "結果タブでは、学習したモデルをいくつかの面から確認できます。①予測スコアのヒートマップ（どこを欠陥と疑ったか）、②エポックごとの F1 の学習曲線（頭打ちや過学習がわかります）、③混同行列（クラスどうしの取り違え）、④マスクの予測（入力画像に推論結果を重ねたもの）です。これらを見て、どこにアノテーションを足すか、どのパラメータを調整するかを決めます。",
    bodyEn: "The Results tab gives several lenses on your model: ① a heatmap of prediction scores (where it suspected a defect), ② the F1-over-epoch curve (to spot plateaus or overfitting), ③ a confusion matrix (class mix-ups), and ④ mask predictions (input overlaid with the result). Use them to decide where to add annotations or tune parameters.",
    advanceOn: { type: "manual" },
    placement: "center",
    modes: ["intermediate", "expert"],
  },
  // --- Expert: post-processing & export & report ---
  {
    id: "expert-postprocess",
    targetSelector: null,
    titleJa: "CCA 後処理（min_area）",
    titleEn: "CCA post-processing (min_area)",
    bodyJa: "推論結果に連結成分分析（CCA：隣り合う前景のピクセルを1つの塊にまとめる処理）をかけ、min_area より小さい塊を取り除きます。孤立したノイズや点状の誤検出を消すのに効果的です。しきい値は、実際にある最小の欠陥の大きさを基準に決めてください（大きすぎると、本物の小さな欠陥まで消えます）。max_area も使えば、大きすぎる誤検出も取り除けます。",
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
    bodyJa: "学習が終わったランは、CoreML、ONNX、OpenVINO の形式でエクスポートできます。CoreML は iOS や macOS のデバイス上での推論、ONNX は Web やさまざまなランタイム、OpenVINO は Intel の CPU や GPU に向いています。OpenVINO では FP32、FP16、INT8 から数値の精度を選べます。ビット数が少ないほど軽く速くなりますが、推論の精度はわずかに下がります。組み込み先に合わせて、形式と数値の精度を選んでください。",
    bodyEn: "Completed runs export to CoreML, ONNX, or OpenVINO. CoreML targets on-device iOS/macOS, ONNX is portable for the Web and many runtimes, and OpenVINO targets Intel CPUs/GPUs. OpenVINO offers FP32 / FP16 / INT8 precision — lower precision is smaller and faster with a slight accuracy drop. Choose the format and precision to match your deployment target.",
    advanceOn: { type: "manual" },
    placement: "center",
    modes: ["expert"],
  },
  {
    id: "expert-inspect",
    targetSelector: null,
    titleJa: "ライブ検査",
    titleEn: "Live Inspection",
    bodyJa: "設定で「ライブ検査」タブを有効にすると、学習済みのモデルで新しい画像をその場で推論し、予測したマスクを重ねて確認できます。入力にはカメラの映像か、ファイルやフォルダを使えます。導入する前に本番に近い画像でモデルをすばやく確かめられ、しきい値や後処理（min_area など）を変えながら、その効果をリアルタイムで見られます。",
    bodyEn: "Enable the Inspect tab (Live Inspection) in Settings to run your trained model on new images on the spot, overlaying the predicted masks. It accepts a camera feed or files/folders. Sanity-check the model on production-like images before deploying — and tweak the threshold and post-processing (min_area, etc.) with real-time feedback.",
    advanceOn: { type: "manual" },
    placement: "center",
    modes: ["expert"],
  },
  {
    id: "expert-report",
    targetSelector: null,
    titleJa: "評価レポート",
    titleEn: "Evaluation report",
    bodyJa: "結果ページのヘッダーにあるレポートのアイコンから、全画像の推論結果と統計をまとめた HTML レポートを作れます。適合率、再現率、F1、Hard Case の画像、スコアの分布などをセクションごとに含められ、ブラウザから印刷すれば A4 の PDF にもなります。チームでの共有、トレーサビリティ、お客様への提出に使えます。含めるセクションはチェックボックスで選びます。",
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
    bodyJa: "お疲れさまでした。このチュートリアルは、ヘッダーの ▶︎ ボタンからいつでも再生できます。説明モードをオンにすると、ボタンにマウスを合わせるだけで説明が表示されます。",
    bodyEn: "Nicely done. Replay this tutorial any time via the ▶︎ button in the header. Enable description mode for hover-to-read explanations on every button.",
    advanceOn: { type: "manual" },
    placement: "center",
  },
];

export function getStepsForMode(mode: TutorialMode): TutorialStep[] {
  return TUTORIAL_STEPS.filter((s) => !s.modes || s.modes.includes(mode));
}
