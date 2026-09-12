# Thrips severity training

This directory is the reproducible training pipeline for the completed
`visible_new_growth_damage_v2` tile annotations. It does not write into the
annotation CSVs or image folders.

## Scientific target

The tile network has one shared self-supervised image backbone and two heads:

1. tissue: new growth (`flush`) versus other (`mature` or `tube`);
2. ordered visible damage on new growth: `healthy < mild < severe`.

`uncertain` tiles contribute to the tissue head but not to the damage loss.
The collection/block name is used for splitting and auditing only. It is never
an input feature.

The plant model is deliberately small. It aggregates out-of-fold tile
probabilities into coverage-corrected severity features, then learns the human
whole-plant score. A plant is never evaluated with tile predictions from a
model that trained on tiles from that plant.

## Server layout

Commands below assume the assembled dataset is `/home/fpt/ThripsDetection`.
Outputs go to `training/artifacts/`, which is ignored by Git.

```bash
cd /home/fpt/ThripsDetection
bash training/bootstrap_server.sh
source .training-venv/bin/activate

# Safe now. Reports missing whole-plant scores without changing data.
python -m training.validate_data --dataset-root .
python -m training.prepare_score_template --dataset-root .
python -m training.smoke_test --dataset-root . --model-id facebook/dinov2-with-registers-small

# After filling human_score with integers 1 through 5:
python -m training.import_scores training/artifacts/plant_scores_to_complete.csv --dataset-root .

# Run after all 127 plant scores have been joined.
python -m training.make_splits --dataset-root . --output training/artifacts/splits.csv

# One fold, frozen probe first.
python -m training.train_tiles --dataset-root . --splits training/artifacts/splits.csv \
  --fold 0 --stage probe

# Then only if the probe is healthy, unfreeze the final backbone stage.
python -m training.train_tiles --dataset-root . --splits training/artifacts/splits.csv \
  --fold 0 --stage finetune
```

Repeat folds 0 through 4. Produce out-of-fold tile predictions, aggregate
plants, and fit the ordinal plant model:

```bash
for fold in 0 1 2 3 4; do
  python -m training.predict_tiles --dataset-root . \
    --splits training/artifacts/splits.csv --fold "$fold"
done

python -m training.aggregate_plants --dataset-root . \
  --predictions training/artifacts/oof_tiles.csv \
  --output training/artifacts/oof_plants.csv

python -m training.train_plant_model \
  --features training/artifacts/oof_plants.csv \
  --splits training/artifacts/splits.csv
```

After the backbone, aggregation, and plant model have been selected without
looking at the final test plants, run the five-fold ensemble exactly once:

```bash
python -m training.predict_ensemble --dataset-root . \
  --splits training/artifacts/splits.csv \
  --checkpoint-root training/artifacts/facebook__dinov3-vits16-pretrain-lvd1689m \
  --stage finetune --output training/artifacts/final_test/tiles.csv

python -m training.aggregate_plants --dataset-root . \
  --predictions training/artifacts/final_test/tiles.csv \
  --output training/artifacts/final_test/plants.csv

python -m training.evaluate_final --dataset-root . \
  --tile-predictions training/artifacts/final_test/tiles.csv \
  --plant-features training/artifacts/final_test/plants.csv \
  --plant-model training/artifacts/plant_model/model.joblib \
  --output-root training/artifacts/final_test
```

## Backbones

Selected after the locked five-fold bakeoff:
`facebook/dinov3-vits16-pretrain-lvd1689m`.

Also evaluated: `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`, native
512 px.
The Meta/Hugging Face license must be accepted and a read token supplied once:

```bash
huggingface-cli login
python -m training.download_model --model-id facebook/dinov3-convnext-tiny-pretrain-lvd1689m
```

Public fallback/baseline:
`facebook/dinov2-with-registers-small`. The model ID is always recorded with
each checkpoint; changing it does not silently reuse incompatible weights.

## Score a newly tiled field run

The native `tiles_foliage.csv` produced by `segment_and_tile.py` followed by
`filter_tube_tiles.py` can be used directly. The inference command keeps only
`decision=keep`, resolves retained images as `foliage_tiles/<tile>`, and uses
`field_upload` as the collection audit value. It does not need human labels.
Segmentation and tiling must use the same 512 px / 384 stride process as the
training data.

```bash
python -m training.predict_ensemble --dataset-root /path/to/new/run \
  --splits training/artifacts/splits.csv \
  --input-csv /path/to/new/run/tiles_foliage.csv \
  --checkpoint-root training/artifacts/facebook__dinov3-vits16-pretrain-lvd1689m \
  --stage finetune --output /path/to/new/run/tile_predictions.csv

python -m training.aggregate_plants --dataset-root /path/to/new/run \
  --metadata /path/to/new/run/tiles_foliage.csv \
  --predictions /path/to/new/run/tile_predictions.csv \
  --output /path/to/new/run/plant_features.csv

python -m training.predict_plants \
  --features /path/to/new/run/plant_features.csv \
  --model training/artifacts/plant_model/model.joblib \
  --output /path/to/new/run/plant_severity.csv
```

`predicted_expected` is the preferred continuous severity output. The rounded
1–5 value is provided for compatibility with human ratings. `needs_review`
flags low-confidence predictions; it is not an out-of-distribution detector.

## Non-negotiable leakage rules

- Splits are by source plant image, never by tile.
- The final test plants are selected once after all human scores arrive.
- Final test plants are excluded from all model selection and calibration.
- Plant-score training uses out-of-fold tile probabilities, not predictions
  made by a tile model that saw the same plant.
- `collection` is for stratification and stress tests, never a learned feature.
