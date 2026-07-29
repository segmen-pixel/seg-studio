# Handbook screenshot shot list

Capture instructions for the figures the handbook still needs. These lived inline
in the handbook as HTML comments; a reader looking at the published Markdown
source saw an unfinished document, so they were moved here.

Once captured, put the file in `docs/images/` and reference it from the handbook
as a normal Markdown image. Use the filename given in the `[...]` of each entry.

## Section: 1. Welcome
- screenshot: Projects tab open; the full tab bar (Projects / Annotate / Training / Live Inspect) visible with at least the SIM project in the list.

## Section: 2. Launch & UI Tour
- 図（スクリーンショット未収録）: Landing screen [02_launch.png]
- screenshot: Fresh launch state — browser just opened localhost:8002/ui/, no project selected, ⚙️ button visible top-right.

## Section: 3. Create a Project
- screenshot: The project-name field filled with 'SIM', about to click Create Project.

## Section: 4. Add Images
- 図（スクリーンショット未収録）: Drag & drop onto the list [04a_drag_drop.png]
- screenshot: Annotate tab with the image list empty or partially populated; ideally a folder drop cursor mid-drag, or the progress bar mid-upload.
- 図（スクリーンショット未収録）: Video frame extraction [04b_video_import.png]
- screenshot: Video drop dialog showing the frame-interval input (extract every N frames).

## Section: 5. Define Classes
- screenshot: Class panel on the right of Annotate with SIM's three classes (background / scratch / stain) and their color swatches.

## Section: 6. Annotate
- screenshot: Brush mid-stroke on a scratch defect — circular brush cursor on the defect, partial red fill in progress.
- 図（スクリーンショット未収録）: SAM result [06b_sam.png]
- screenshot: SAM tool active after a left-click — positive-point marker (green +) and the proposed mask highlighted, pre-Enter preview.
- 図（スクリーンショット未収録）: Crack & spot tools [06c_crack_spot.png]
- screenshot: Crack trace result after two endpoint clicks, or spot detect highlighting point defects.

## Section: 7. Mark Clean for OK images
- screenshot: Image list with several clean images multi-selected (blue highlight); the OK (Mark Clean) button is highlighted.

## Section: 8. Augment Your Data
- 図（スクリーンショット未収録）: Perlin CutPaste dialog [08a_perlin.png]
- screenshot: Augment dialog with Perlin CutPaste enabled; preview canvas shows a warped defect and the class dropdown is open.
- 図（スクリーンショット未収録）: Lighting variants [08b_lighting.png]
- screenshot: Thumbnail gallery after generating Lighting variants — same scene in daytime / evening / night side by side.

## Section: 9. Prepare the Dataset
- 図（スクリーンショット未収録）: Prepare report [09_prepare.png]
- screenshot: Run list showing the "Preparing dataset..." placeholder and the resulting train / val counts.

## Section: 10. Choose Training Settings
- 図（スクリーンショット未収録）: Auto-config panel [10a_auto_config.png]
- screenshot: Auto-config recommendation panel with suggested arch / patch_size / base_channels badges.
- screenshot: Full hyperparameter form in Training tab — epochs, loss_type, transfer learning toggle all visible.

## Section: 11. Run Training
- 図（スクリーンショット未収録）: Local training live [11a_local_training.png]
- screenshot: Local training in progress — loss / F1 / mIoU line chart updating live with a bottom progress bar.

## Section: 12. Inspect the Results
- 図（スクリーンショット未収録）: Metrics on Results [12a_metrics.png]
- screenshot: Results tab metric header — F1 / mIoU / Precision / Recall rendered as large badges.
- 図（スクリーンショット未収録）: Heatmap [12b_heatmap.png]
- screenshot: One image open in Results with heatmap overlay on; high-confidence regions glow red/yellow.
- 図（スクリーンショット未収録）: Live Inspect [12c_live_inspection.png]
- screenshot: Live Inspect tab with webcam feed + real-time mask overlay; fps counter visible.

## Section: 13. Export the Model
- screenshot: Export menu for a trained run showing ONNX / CoreML / CoreML Updatable buttons.

## Section: 14. Call it from the SDK
- 図（スクリーンショット未収録）: SDK output [14_sdk_output.png]
- screenshot: Terminal after running quick_start.py — judgement, region count, centroid and latency printed.
