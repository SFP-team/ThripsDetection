# Visible Thrips-Damage Severity System

## Complete scientific, model, server, and UI handoff

**Prepared:** 2026-08-26  
**GPU project root:** `/home/fpt/ThripsDetection`  
**Selected backbone:** `facebook/dinov3-vits16-pretrain-lvd1689m`  
**Protocol:** `visible_new_growth_damage_v2`

This is the authoritative handoff for the next agent implementing the prediction UI. It describes the target, annotations, dataset, training, evaluation, model files, tested inference commands, expected output, limitations, and required UI behavior.

## 1. Product goal

The system does **not** detect thrips insects. The rover RGB photographs do not resolve the insects reliably, and insect detection was never the target.

The target is:

> Estimate visible new-growth damage for every photographed plant, distinguish mild from severe damage, rank plants from least to most damaged, and identify which plants deserve attention first.

The model may support scouting and treatment prioritization. It must not invent a pesticide product, concentration, or per-plant chemical dose from an image. Visible damage can persist after a pest has left or been controlled. Any chemical rate must come from the applicable product label and the field's agronomic protocol.

## 2. Desired user workflow

The next UI should implement this flow:

```text
Upload one raw rover image or a folder of images
    -> send photographs to the GPU
    -> segment each plant with center-seeded BiRefNet
    -> generate overlapping 512 px tiles at stride 384
    -> retain foliage tiles
    -> run the five-model DINOv3 ensemble
    -> estimate new-growth probability and damage probabilities per tile
    -> aggregate overlapping tiles into plant features
    -> predict continuous whole-plant severity and a rounded 1-5 score
    -> rank all uploaded plants from most to least visibly damaged
    -> display confidence, review flags, and a tile heatmap
    -> allow CSV download
```

For one photograph, the system can return one severity result. For a folder or field block, it should also return rank and percentile. Batch ranking is the preferred operational use because model ranking is stronger than exact integer calibration.

## 3. Annotation target and dataset

### 3.1 Tile protocol

Images were segmented and tiled before annotation:

- Plant segmentation: center-seeded BiRefNet
- Tile size: 512 x 512 pixels
- Tile stride: 384 pixels
- A foliage filter removed low-foliage and obvious tube/background tiles
- Every retained tile was assigned a tissue type
- Visible damage was annotated on new growth
- The model never tries to see or localize insects

The current training target uses:

- Tissue: new growth (`flush`) versus other tissue (`mature` or `tube`)
- Ordered visible damage on new growth: `healthy < mild < severe`
- `uncertain` new-growth tiles contribute to tissue learning but are excluded from the damage loss

Do not introduce a 1-5 label on individual tiles. The 1-5 rating exists only at whole-plant level.

### 3.2 Dataset totals

The assembled training dataset is at:

```text
/home/fpt/ThripsDetection
```

Validated totals:

| Item | Count |
|---|---:|
| Source plant photographs | 127 |
| Retained tiles | 2,593 |
| Plants with whole-plant scores | 127 |
| New-growth tiles | 1,514 |
| Old-leaf tiles | 1,066 |
| Non-leaf/tube tiles | 13 |
| Healthy new-growth tiles | 773 |
| Mild new-growth tiles | 364 |
| Severe new-growth tiles | 358 |
| Uncertain new-growth tiles | 19 |

Plant-score distribution:

| Human score | Plants |
|---:|---:|
| 1 | 57 |
| 2 | 18 |
| 3 | 23 |
| 4 | 24 |
| 5 | 5 |

Collections:

| Stored collection identifier | Plants | Meaning |
|---|---:|---|
| `row1_chili` | 36 | First affected-row collection. The word `chili` is a legacy folder identifier; the photographed plants are blueberries. |
| `row2_chili` | 39 | Second affected-row collection; same legacy naming note. |
| `citra_healthy` | 52 | Deliberately collected healthy controls from the same field at different positions. |

All 52 Citra healthy photographs were assigned whole-plant score 1 by explicit user decision. The Citra images were intentionally retained in the combined dataset to provide genuine healthy examples. Collection is used for splitting and auditing only and is never supplied to the learned plant model as a feature.

The 39 genotype scores for the second row were mapped to photographs in verified capture order and stored in the metadata. `IMG_0391.JPG` was absent from the available photographs and annotations and is not silently represented in the dataset.

Manually reviewed dropped-tile adjudications included healthy, mild, and severe examples. They were considered during completion of the labeled dataset rather than assuming every dropped tile was unusable.

### 3.3 Important data files

```text
/home/fpt/ThripsDetection/training_tiles.csv
/home/fpt/ThripsDetection/plants.csv
/home/fpt/ThripsDetection/metadata/tiles_foliage.csv
/home/fpt/ThripsDetection/metadata/source_map.csv
/home/fpt/ThripsDetection/metadata/genotype_map.csv
/home/fpt/ThripsDetection/training/artifacts/splits.csv
/home/fpt/ThripsDetection/training/artifacts/dataset_freeze.json
```

The dataset freeze record protects the exact data/split state used for the reported results.

## 4. Leakage-safe experimental design

The fixed split contains:

- Development: 102 plants
- Untouched final test: 25 plants
- Five development folds with 20, 20, 21, 20, and 21 plants

Final-test composition:

- 10 Citra healthy plants
- 7 first-row plants
- 8 second-row plants
- Scores: eleven 1s, four 2s, four 3s, five 4s, and one 5

Leakage controls:

1. Splits are by source plant image, never by tile.
2. All tiles from one plant stay together.
3. Model and aggregation choices were made using only development folds.
4. Development plant features use out-of-fold tile predictions. A plant is not summarized using a tile model trained on that plant.
5. The 25 final-test plants were opened only after the backbone, fine-tuning method, features, and plant model were fixed.
6. Collection is never a learned input feature.

The split file must remain fixed. Do not regenerate it for the UI.

## 5. Model architecture

### 5.1 Tile model

The selected tile backbone is:

```text
facebook/dinov3-vits16-pretrain-lvd1689m
```

The tile network has one shared DINOv3 image backbone and two heads:

1. Binary tissue head: probability that the tile contains new growth
2. Ordinal damage head: cumulative thresholds representing `healthy < mild < severe`

The ordinal formulation prevents damage from being treated as three unrelated categories. It produces:

```text
p_healthy
p_mild
p_severe
expected_damage = p_mild + 2 * p_severe
```

The network also produces `p_flush`, the new-growth probability. During plant aggregation, a tile's contribution is weighted by `p_flush`; mature leaves do not dominate the severity estimate.

### 5.2 Plant aggregation

Overlapping tiles are coverage-corrected so the same leaf area is not counted repeatedly. The plant-level features are:

- Mean expected new-growth damage
- Severe probability mass
- Any-damage probability mass
- 75th percentile expected damage
- 90th percentile expected damage
- Mean of three spatially distinct highest-damage tiles
- Upper-canopy mean expected damage
- Upper-canopy severe mass
- Effective visible new-growth area
- Spatial spread of expected damage
- Mean prediction entropy

The aggregation implementation is:

```text
/home/fpt/ThripsDetection/training/aggregate_plants.py
```

### 5.3 Plant score model

The plant model is a deliberately small ordinal threshold model. It fits four regularized logistic models for:

```text
P(score > 1)
P(score > 2)
P(score > 3)
P(score > 4)
```

The cumulative probabilities are constrained to be monotonic and converted into probabilities for scores 1 through 5. The primary numeric output is the probability-weighted expected score. A rounded 1-5 score is secondary.

The plant model never receives `collection` or filename as an input feature.

## 6. Training procedure

### 6.1 Backbone bakeoff

Three self-supervised backbones were compared using the same five locked development folds and frozen-backbone linear probing:

| Backbone | Mean damage macro-F1 | Mean QWK | Mean severe recall | Mean tissue F1 |
|---|---:|---:|---:|---:|
| DINOv3 ConvNeXt-Tiny | 0.735 | 0.773 | 0.771 | 0.833 |
| **DINOv3 ViT-S/16** | **0.745** | **0.781** | 0.752 | **0.857** |
| DINOv2 registers small | 0.722 | 0.765 | 0.705 | 0.850 |

DINOv3 ViT-S/16 was selected for the best overall and cross-collection balance. ConvNeXt had slightly higher severe recall in the frozen probe, but ViT-S/16 had stronger overall damage and tissue performance and was more stable across the affected collections.

### 6.2 Fine-tuning

Each fine-tuning fold was initialized from its own best frozen-probe checkpoint. The final two ViT transformer blocks and normalization layers were unfrozen; the rest of the backbone remained frozen. The classification heads were also trained.

Training configuration:

| Setting | Value |
|---|---:|
| Input size | 512 |
| Folds | 5 |
| Batch size | 16 |
| Gradient accumulation | 2 |
| Head learning rate | 0.0003 |
| Backbone learning rate | 0.00001 |
| Weight decay | 0.05 |
| Dropout | 0.2 |
| Maximum probe epochs | 40 |
| Maximum fine-tune epochs | 60 |
| Early-stopping patience | 10 |
| Seed | 20260826 |

Training used plant-aware and class-aware sampling. `uncertain` damage labels were masked from damage loss. Standard restrained augmentations included horizontal flip, small affine perturbation, modest color jitter, and occasional light Gaussian blur.

Fine-tuned validation macro-F1 by fold:

| Fold | Best damage macro-F1 |
|---:|---:|
| 0 | 0.797 |
| 1 | 0.762 |
| 2 | 0.741 |
| 3 | 0.783 |
| 4 | 0.791 |
| Mean | **0.775** |

Fine-tuning improved mean macro-F1 from approximately 0.745 to 0.775.

### 6.3 Deployment ensemble

For inference, load all five fine-tuned fold checkpoints. Run all five models on each retained tile and average their `p_flush`, `p_healthy`, `p_mild`, and `p_severe` probabilities before plant aggregation.

Do not pick only the single best fold. The five-fold average is the tested deployment model.

## 7. Results

### 7.1 Development out-of-fold tile results

These predictions cover all 102 development plants with no plant-level leakage:

| Metric | Result |
|---|---:|
| Damage tiles | 1,168 |
| Damage macro-F1 | 0.775 |
| Balanced accuracy | 0.783 |
| Quadratic weighted kappa | 0.823 |
| Healthy recall | 83.1% |
| Mild recall | 71.3% |
| Severe recall | 80.5% |
| Healthy directly predicted severe | 1.16% |
| Severe directly predicted healthy | 0.77% |
| New-growth tissue F1 | 89.1% |

### 7.2 Development out-of-fold plant results

| Metric | Result |
|---|---:|
| Plants | 102 |
| Mean absolute error | 0.565 |
| Median absolute error | 0.294 |
| Spearman rank correlation | 0.850 |
| Quadratic weighted kappa | 0.809 |
| Exact rounded score | 57.8% |
| Within one score point | 91.2% |

### 7.3 Untouched final-test tile results

The final test was evaluated once after the method was locked:

| Metric | Result |
|---|---:|
| Test tiles | 513 |
| Evaluated new-growth damage tiles | 327 |
| Damage macro-F1 | **0.787** |
| Balanced accuracy | **0.807** |
| Quadratic weighted kappa | **0.882** |
| Healthy recall | **82.8%** |
| Mild recall | **70.5%** |
| Severe recall | **88.7%** |
| Healthy directly predicted severe | **0%** |
| Severe directly predicted healthy | **0%** |
| New-growth tissue F1 | **92.4%** |

Final-test damage confusion matrix, rows are truth and columns are predicted `healthy, mild, severe`:

```text
healthy  140  29   0
mild       2  43  16
severe     0  11  86
```

The primary tile errors are between adjacent categories, not healthy-to-severe extremes.

### 7.4 Untouched final-test plant results

| Metric | Result |
|---|---:|
| Plants | 25 |
| Mean absolute error | **0.706** |
| Median absolute error | **0.482** |
| Spearman rank correlation | **0.862** |
| Quadratic weighted kappa | **0.780** |
| Exact rounded score | **52.0%** |
| Within one score point | **80.0%** |

Plant confusion matrix, rows are human scores 1-5 and columns are rounded model scores 1-5:

```text
score 1: 10  1  0  0  0
score 2:  0  0  1  3  0
score 3:  0  0  2  0  2
score 4:  0  0  1  0  4
score 5:  0  0  0  0  1
```

### 7.5 Collection stress test

| Collection | Test plants | Plant MAE | Exact score | Within one |
|---|---:|---:|---:|---:|
| Citra healthy | 10 | 0.114 | 100.0% | 100.0% |
| First affected row | 7 | 1.158 | 14.3% | 57.1% |
| Second affected row | 8 | 1.050 | 25.0% | 75.0% |

The Citra controls are visually easier than healthy or mildly damaged tissue within affected blocks. Overall metrics therefore overstate exact-score reliability for affected-row plants. The model's strongest supported use is relative ranking and broad visible-damage prioritization. Exact 1-5 calibration should be improved with more affected-block score-1 to score-3 plants.

## 8. Correct interpretation

Supported statements:

- The model detects and ranks **visible new-growth damage patterns**.
- It distinguishes mild and severe-looking injury with useful accuracy.
- It is good at identifying obviously healthy Citra controls.
- It is strong enough for an initial field-ranking UI and supervised field pilot.
- Its continuous score and within-batch rank are more reliable than treating every rounded integer as exact.

Unsupported statements:

- It does not detect insects.
- It does not prove that thrips are currently present.
- It does not determine when damage occurred.
- It does not distinguish every possible non-thrips stress without additional data.
- It does not independently prescribe pesticide concentration or dose.
- It is not yet validated for autonomous use in a new farm, camera, season, or lighting regime.

## 9. Model weights and required artifacts

### 9.1 Five DINOv3 tile checkpoints

Use all five:

```text
/home/fpt/ThripsDetection/training/artifacts/facebook__dinov3-vits16-pretrain-lvd1689m/fold_0/finetune/best.pt
/home/fpt/ThripsDetection/training/artifacts/facebook__dinov3-vits16-pretrain-lvd1689m/fold_1/finetune/best.pt
/home/fpt/ThripsDetection/training/artifacts/facebook__dinov3-vits16-pretrain-lvd1689m/fold_2/finetune/best.pt
/home/fpt/ThripsDetection/training/artifacts/facebook__dinov3-vits16-pretrain-lvd1689m/fold_3/finetune/best.pt
/home/fpt/ThripsDetection/training/artifacts/facebook__dinov3-vits16-pretrain-lvd1689m/fold_4/finetune/best.pt
```

Each checkpoint is approximately 86.5 MB and includes the complete fine-tuned state dictionary, model ID, fold, stage, training configuration, and validation metrics.

Checkpoint root passed to inference:

```text
/home/fpt/ThripsDetection/training/artifacts/facebook__dinov3-vits16-pretrain-lvd1689m
```

### 9.2 Plant severity model

```text
/home/fpt/ThripsDetection/training/artifacts/plant_model/model.joblib
```

This file includes the four fitted ordinal threshold models and the exact ordered plant-feature list.

### 9.3 Do not deploy only one file

A complete deployment needs:

1. All five `best.pt` tile checkpoints
2. `plant_model/model.joblib`
3. The Python package under `/home/fpt/ThripsDetection/training/`
4. Access to the Hugging Face architecture/config for `facebook/dinov3-vits16-pretrain-lvd1689m`
5. The same image normalization and 512 px transform
6. The same segmentation, tiling, foliage filtering, and aggregation logic

The current GPU server already has the required DINOv3 model cached and can run with `HF_HUB_OFFLINE=1`. On another server, accept the model's Hugging Face/Meta terms and download the base model once. Do not place an access token in source code, a Markdown file, or a Git commit.

## 10. Server runtime

Tested environment:

```text
Python       3.10.12
PyTorch      2.6.0+cu124
Transformers 5.14.1
scikit-learn 1.7.2
pandas       2.3.3
NumPy        2.2.6
GPU          2 x NVIDIA GeForce RTX 4090, 24 GB each
```

Python executable used:

```text
/home/fpt/RaghavWork/Segmentation_ResearchPaper/.venv/bin/python
```

BiRefNet project used for segmentation:

```text
/home/fpt/RaghavWork/Segmentation_ResearchPaper
```

## 11. Direct inference from an already tiled run

Assume a run folder contains:

```text
RUN/
  tiles_foliage.csv
  foliage_tiles/
  images.csv
  plant_crops/
```

The native `tiles_foliage.csv` from `segment_and_tile.py` followed by `filter_tube_tiles.py` is accepted directly. Inference automatically:

- keeps only rows with `decision=keep`
- resolves tile files as `foliage_tiles/<tile>`
- supplies `field_upload` as the collection audit value when collection is absent

Run:

```bash
cd /home/fpt/ThripsDetection

PYTHON=/home/fpt/RaghavWork/Segmentation_ResearchPaper/.venv/bin/python
RUN=/absolute/path/to/the/tiled/run
MODEL_ROOT=/home/fpt/ThripsDetection/training/artifacts/facebook__dinov3-vits16-pretrain-lvd1689m
PLANT_MODEL=/home/fpt/ThripsDetection/training/artifacts/plant_model/model.joblib

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 "$PYTHON" -m training.predict_ensemble \
  --dataset-root "$RUN" \
  --input-csv "$RUN/tiles_foliage.csv" \
  --checkpoint-root "$MODEL_ROOT" \
  --stage finetune \
  --output "$RUN/tile_predictions.csv"

"$PYTHON" -m training.aggregate_plants \
  --dataset-root "$RUN" \
  --metadata "$RUN/tiles_foliage.csv" \
  --predictions "$RUN/tile_predictions.csv" \
  --output "$RUN/plant_features.csv"

"$PYTHON" -m training.predict_plants \
  --features "$RUN/plant_features.csv" \
  --model "$PLANT_MODEL" \
  --output "$RUN/plant_severity.csv"
```

The final file is:

```text
$RUN/plant_severity.csv
```

### 11.1 Plant severity output schema

```text
image
collection
p_score_1
p_score_2
p_score_3
p_score_4
p_score_5
predicted_expected
predicted_score
confidence
needs_review
```

Definitions:

- `predicted_expected`: probability-weighted continuous score from 1.0 to 5.0; preferred numeric result
- `predicted_score`: rounded score from 1 to 5; secondary display value
- `confidence`: maximum of the five plant-score probabilities
- `needs_review`: currently true when confidence is below 0.60

`needs_review` is a low-confidence flag, not a complete out-of-distribution detector.

### 11.2 Tile output schema

```text
image
tile
fold
p_flush
p_healthy
p_mild
p_severe
expected_damage
```

The `fold` value is `-1` for ensemble inference and is not a biological output.

## 12. Raw photograph ingestion

Raw JPG/PNG photographs must first run through the existing segmentation scripts:

```text
segment_and_tile.py
filter_tube_tiles.py
```

Canonical copies for server-side prediction jobs are placed at:

```text
/home/fpt/ThripsDetection/tools/segment_and_tile.py
/home/fpt/ThripsDetection/tools/filter_tube_tiles.py
```

The existing annotation app already demonstrates the correct short GPU-job flow in `annotator/gpu_jobs.py`:

1. Create a unique job directory
2. Upload and flatten raw photographs into `images/`
3. Run `segment_and_tile.py`
4. Run `filter_tube_tiles.py`
5. Run tile ensemble inference
6. Aggregate plants
7. Run plant prediction
8. Return results to the local UI
9. Disconnect from the GPU

Never reuse an old tiled run merely because raw filenames match. Each prediction upload should create a new unique job directory.

Recommended remote layout:

```text
/home/fpt/Chili thrips detection pictures/prediction_jobs/
  <YYYYMMDD-HHMMSS>-<job_id>/
    images/
    run/
      tiles_foliage.csv
      images.csv
      foliage_tiles/
      plant_crops/
      overlays/
      tile_predictions.csv
      plant_features.csv
      plant_severity.csv
```

The legacy parent-folder name containing `Chili` may remain for compatibility, but UI copy should say blueberry visible-damage severity.

## 13. Required UI page

Add a separate prediction page. Do not alter annotation storage keys or mix prediction outputs into annotation labels.

Suggested navigation item:

```text
Predict damage
```

Suggested title:

```text
Predict visible plant damage
```

Suggested explanatory copy:

```text
Upload rover photographs to estimate visible new-growth damage and rank plants
from least to most damaged. This model measures visible injury; it does not
detect insects or prescribe pesticide dosage.
```

### 13.1 Inputs

- One raw JPG/PNG
- Multiple raw JPGs/PNGs
- A local folder path when supported by the desktop app
- Optional field/block/run name for auditing only

Reject a tiled folder on the raw-photo control and direct it to an optional advanced "Existing tiled run" control. Preserve original image filenames in results.

### 13.2 Progress states

Recommended job states:

```text
sending
segmenting
filtering
scoring_tiles
aggregating_plants
ready
error
```

Friendly text:

| State | UI text |
|---|---|
| `sending` | Sending photographs to the GPU |
| `segmenting` | Finding and cutting out each plant |
| `filtering` | Keeping leaf tiles |
| `scoring_tiles` | Measuring visible damage |
| `aggregating_plants` | Calculating whole-plant severity |
| `ready` | Predictions are ready |

### 13.3 Result table

Sort by `predicted_expected` descending by default.

Required columns:

| Column | Source |
|---|---|
| Rank | Computed within uploaded batch |
| Plant image | `image` |
| Damage index | `predicted_expected`, show two decimals |
| Rounded score | `predicted_score` |
| Broad severity | UI mapping described below |
| Confidence | `confidence`, show percentage |
| Review | `needs_review` |
| Details | Opens plant evidence view |

Recommended operational display mapping. This is UI presentation, not a new training label:

| Continuous score | Broad display |
|---:|---|
| 1.00 to 1.49 | Minimal visible damage |
| 1.50 to 2.49 | Mild visible damage |
| 2.50 to 3.49 | Moderate visible damage |
| 3.50 to 4.49 | Severe visible damage |
| 4.50 to 5.00 | Very severe visible damage |

Also show batch rank and percentile. Recommended focus grouping for the initial field pilot:

- High priority: top 20% by continuous damage index
- Medium priority: next 30%
- Low priority: bottom 50%
- Manual review: any low-confidence result regardless of rank

These percentages should be configurable and labeled as prioritization, not universal biological or treatment thresholds.

### 13.4 Plant evidence view

Show:

- Original photograph
- Segmented plant crop
- Number of retained tiles
- Continuous damage index
- Rounded score
- Five score probabilities
- Confidence and review warning
- A heatmap or tile overlay on the original/crop
- Highest-damage distinct tiles

Suggested tile visualization:

- Tile opacity or border strength weighted by `p_flush`
- Healthy color from `p_healthy`
- Mild color from `p_mild`
- Severe color from `p_severe`
- Clicking a tile shows all four probabilities

Do not present a mature-leaf tile as strong severity evidence just because it has unusual color. `p_flush` must control its contribution.

### 13.5 CSV export

Export at least:

```text
rank,image,predicted_expected,predicted_score,severity_group,
p_score_1,p_score_2,p_score_3,p_score_4,p_score_5,
confidence,needs_review,priority_group
```

Optionally provide a second tile-level CSV using the tested `tile_predictions.csv` schema.

### 13.6 Suggested APIs

The current app is a FastAPI-served SPA. Suggested additions:

```text
POST /api/prediction-jobs/gpu
POST /api/prediction-jobs/gpu-upload
GET  /api/prediction-jobs/{job_id}
GET  /api/prediction-runs
GET  /api/prediction-runs/{run_id}
GET  /api/prediction-runs/{run_id}/plants
GET  /api/prediction-runs/{run_id}/plants/{image}
GET  /api/prediction-runs/{run_id}/export
```

Return paths or media endpoints for crops, overlays, and tiles. Do not return raw server filesystem paths to a browser when a media route is appropriate.

### 13.7 Server-side job command sequence

After segmentation/filtering creates `RUN`, execute the three commands in Section 11. Capture stdout/stderr in a per-job log, check each exit code, and only mark `ready` after `plant_severity.csv` exists and contains one row per uploaded source image.

Use GPU inference server-side. Do not send model weights to the browser.

## 14. Quality and failure handling

The UI must explicitly handle:

- No plant detected
- No foliage tiles retained
- Uploaded file is not an image
- Duplicate basenames in one upload
- GPU unavailable
- Segmentation command failure
- Missing model checkpoint
- Missing plant model
- Output row count does not match source-image count
- Low confidence
- Very little predicted new-growth area

Recommended warning copy:

```text
The model could not see enough new growth for a reliable severity estimate.
Review this plant manually or capture another photograph.
```

Do not silently assign score 1 when segmentation or tile generation fails.

`confidence < 0.60` is already available. Add an additional UI quality warning for very low `effective_flush_area` after inspecting its distribution on the development data; do not invent a threshold without calculating it.

## 15. Pesticide-related UI boundary

The user ultimately wants to know which plants to focus on and how to prioritize treatment. The supported model output is:

```text
Visible damage priority: high / medium / low
```

The UI must not derive chemical concentration or a per-plant pesticide quantity from the model score. Visible damage is historical and may persist without current active infestation. Product rates, application volume, maximum frequency, re-entry interval, and pre-harvest interval are governed by the applicable product label and field protocol.

If a later agronomist-approved treatment module is added, keep it separate from model inference. It may consume the model's priority group, but all product/rate rules must be explicitly configured from the approved label and reviewed by the responsible agronomist/applicator.

Useful authoritative references:

- EPA pesticide labeling requirements: <https://www.epa.gov/pesticide-registration/labeling-requirements>
- EPA directions for use and application-rate requirements: <https://www.epa.gov/pesticide-labels/label-review-training-module-3-special-issues-page-11>
- UF/IFAS chilli thrips on blueberries: <https://edis.ifas.ufl.edu/publication/IN1298/pdf>

## 16. What remains to do

### 16.1 Engineering work now

1. Add the prediction page and navigation entry.
2. Add new GPU prediction-job APIs without changing annotation APIs.
3. Reuse the existing raw-photo upload and short SSH/GPU job pattern.
4. Chain segmentation, filtering, ensemble inference, plant aggregation, and plant prediction.
5. Persist prediction run metadata and CSV outputs separately from annotation labels.
6. Build ranked batch results.
7. Build plant evidence/heatmap view.
8. Add downloads and error handling.
9. Test with one image, multiple images, and a complete folder.
10. Verify a known final-test run reproduces the existing output before field deployment.

No additional annotation or retraining is required before building and testing this first prediction UI.

### 16.2 Scientific work after the first UI pilot

Collect approximately 30 to 50 additional plants concentrated on:

- Healthy/score-1 plants inside or adjacent to affected blocks
- Score-2 affected-block plants
- Borderline score-2 versus score-3 examples
- Different genotypes under the same affected-block backgrounds
- Non-thrips stresses that can resemble curling, bronzing, or deformation
- Repeated images under different rover position and lighting

The largest current gap is affected-block healthy versus mild damage. Additional obvious Citra controls are lower priority because the model already recognizes that collection very well.

After collecting new data:

1. Freeze a new versioned dataset.
2. Preserve the current final test as a historical benchmark.
3. Create a new field/season holdout.
4. Retrain the five tile folds and plant model.
5. Recalibrate confidence/review thresholds.
6. Compare performance by collection, genotype, field block, and lighting.

## 17. Reproduction and validation artifacts

Training code:

```text
/home/fpt/ThripsDetection/training/model.py
/home/fpt/ThripsDetection/training/data.py
/home/fpt/ThripsDetection/training/train_tiles.py
/home/fpt/ThripsDetection/training/predict_tiles.py
/home/fpt/ThripsDetection/training/predict_ensemble.py
/home/fpt/ThripsDetection/training/aggregate_plants.py
/home/fpt/ThripsDetection/training/train_plant_model.py
/home/fpt/ThripsDetection/training/predict_plants.py
/home/fpt/ThripsDetection/training/evaluate_oof.py
/home/fpt/ThripsDetection/training/evaluate_final.py
```

Evaluation outputs:

```text
/home/fpt/ThripsDetection/training/artifacts/facebook__dinov3-vits16-pretrain-lvd1689m/oof_finetune_tiles.csv
/home/fpt/ThripsDetection/training/artifacts/facebook__dinov3-vits16-pretrain-lvd1689m/oof_finetune_tiles_evaluation.json
/home/fpt/ThripsDetection/training/artifacts/facebook__dinov3-vits16-pretrain-lvd1689m/oof_finetune_plants.csv
/home/fpt/ThripsDetection/training/artifacts/plant_model/metrics.json
/home/fpt/ThripsDetection/training/artifacts/plant_model/oof_predictions.csv
/home/fpt/ThripsDetection/training/artifacts/final_test/tiles.csv
/home/fpt/ThripsDetection/training/artifacts/final_test/plants.csv
/home/fpt/ThripsDetection/training/artifacts/final_test/plant_predictions.csv
/home/fpt/ThripsDetection/training/artifacts/final_test/deployment_predictions.csv
/home/fpt/ThripsDetection/training/artifacts/final_test/metrics.json
```

All five fine-tuning folds completed without failure. All eight training-package unit tests passed. The assembled dataset validation passed with all 127 scores present.

## 18. Known implementation notes

- `predicted_expected` is the primary display number.
- Use all five fine-tuned checkpoints for production inference.
- Keep `HF_HUB_OFFLINE=1` on the current server after confirming the model cache.
- Do not use ConvNeXt or DINOv2 checkpoints for the shipped UI; they are research comparisons.
- Do not regenerate splits.
- Do not use the final-test set for more tuning.
- Do not make `collection` a model feature.
- Do not score individual tiles 1-5.
- Do not call the system an insect detector.
- Do not claim low confidence is equivalent to out-of-distribution detection.
- Do not silently turn processing failures into healthy predictions.
- Keep prediction sessions separate from annotation sessions and labels.

## 19. Acceptance checklist for the next agent

The prediction UI is ready for handoff when all of these are true:

- [ ] One raw rover JPG can be uploaded and produces exactly one plant result.
- [ ] A folder can be uploaded and produces one result per photograph.
- [ ] Each job has a unique remote directory.
- [ ] Native `tiles_foliage.csv` is passed directly to tested inference code.
- [ ] All five ViT-S/16 checkpoints are loaded and averaged.
- [ ] Plant aggregation uses coverage correction and `p_flush` weighting.
- [ ] The plant model at `plant_model/model.joblib` is used.
- [ ] Results are sorted by `predicted_expected` descending.
- [ ] Continuous score, rounded score, five probabilities, confidence, and review flag are visible.
- [ ] The original image, crop, and tile evidence can be inspected.
- [ ] Results can be downloaded as CSV.
- [ ] Failed segmentation never becomes score 1.
- [ ] Low-confidence results are visibly flagged.
- [ ] The UI says visible damage, not insect detection.
- [ ] The UI does not prescribe pesticide concentration or quantity.
- [ ] A reproduction test matches the existing final-test deployment CSV.

## 20. Bottom line

The trained backend is complete. It can accept a correctly segmented/tiled rover photograph, predict tile-level visible damage, aggregate it into a whole-plant continuous severity score, round it to the familiar 1-5 scale, and rank a batch of plants.

The next task is product integration: connect the existing raw-photo GPU preparation flow to the three tested inference stages and expose ranked results, confidence, evidence tiles, and downloads in a new prediction page.

For current field use, position the system as a **visible-damage ranking and prioritization tool with human review**, not an autonomous pesticide prescription system.
