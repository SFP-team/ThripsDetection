# Locked plan — chilli thrips plant scoring

Last updated 18 August 2026. This is the method we train and label against. It is not a list of extra models to try.

## Problem

Rover photos of young blueberry plants (Canon 18 MP, white grow tubes). We need a number for how badly chilli thrips have hit each plant.

The insects themselves are not visible. What we can see is injury on **new flush at the top**: cup, crinkle, bronze or silver, and in bad cases dark or dead tips.

The official field scale is **UF / IFAS 0–4 on new flush only**. A rover JPEG cannot reliably tell 0 from 1 (early vein bronze is a couple of pixels). So we publish 0–4 for scouts, and we train **three practical bins**: looks clean (0–1), clearly damaged (2–3), badly hit (4).

We already have 36 plants with human whole-plant scores 1–5 (`human_thrips_scores.csv`). We still need about 80–100 more photos, especially more 1s and 5s.

## What failed

- Classical color / texture maps: Spearman about 0.21
- Frozen ImageNet EfficientNet on the whole crop: about 0
- Scoring every tile 1–5: people will not agree; the scale is for the whole plant

## What worked

Frozen DINOv3 tile-mean embeddings + ridge, leave-one-plant-out, hit **Spearman 0.68** vs the 36 human scores (`dino_probe.py`). Signal lives in the leaf squares, not in a whole-plant embedding. That first probe likely shrank tiles to 256 px; we remeasure at **native 512** before we change the backbone.

## Pipeline (locked)

1. Take the rover photo.
2. Cut the plant out of the background (Center BiRefNet, fold 0).
3. Cut that plant into **512 px squares**, stride **384**.
4. Keep leaf squares. Drop pot, mulch, sky. Tube only counts in the lower part of the box. **Do not drop pale silvered upper flush** — the old “is it green enough?” filter can throw away the exact damaged tips.
5. A person labels each kept square with the protocol below. Not 1–5.
6. A small model learns flush healthy vs injured.
7. For a new plant we score the flush squares, weight the **top** of the plant more, and map “how much of the young growth looks hurt” to the official 0–4 score.
8. That grade is checked against a human whole-plant score, with **session-blocked** evaluation (tomorrow’s photos are a new day; the first 36 were taken in about 11 minutes).

## Tile labels (what the UI asks)

Do **not** put a 1–5 on a square.

| Field | Values | When |
|---|---|---|
| Tissue | `flush` / `mature` / `tube` | Every square |
| Injury | `healthy` / `injured` / `skip` | Flush only. Mature and tube are skip. |
| Curl | `yes` / `no` | Flush healthy or injured only |

Rules the annotator sees:

- Mixed flush = injured
- Lime crinkle with no bronze or cup = healthy
- Curl is separate from color: cupped, twisted, or crinkled = yes
- Mature wood, pot, tag, sky = skip, not “injured”

Export columns: `image, tile, rel_y, tissue, injury, curl, label, annotator, labeled_at`. The join key is the tile filename, e.g. `IMG_0327_x737_y784.jpg`.

## Plant score

Flush-weighted injured fraction + DINOv3 tile-mean → a small ordinal head (`mord.LogisticAT`). Do **not** train attention-MIL / TransMIL at n ≈ 136. Mean and max beat attention in similar work; with this few plants an attention bag will memorize the two worst bushes.

Scout / publish **UF/IFAS 0–4**. Train **3 classes** (0–1 / 2–3 / 4).

## Models to bake off (only these)

Tile training: LoRA r=8, hue jitter **off**, plant-grouped BCE. Fallback: frozen MLP.

| Model | Why |
|---|---|
| DINOv3-B @ 512 (`facebook/dinov3-vitb16-pretrain-lvd1689m`) | Current floor |
| DINOv3-L | Larger same family |
| C-RADIOv4-SO400M | 2026 bake-off only |

If the larger two do not beat DINOv3-B at native 512, drop them.

**Do not** use unpaired leaf pretraining, PlantCLEF / BioCLIP / SigLIP2 / PE-Core as the main model, or insect detectors.

## What will raise the score more than a new net

1. Keep silvered upper flush (fix the foliage filter).
2. Second rater on the same 36 plants, IFAS wording.
3. Treat the next field day as a new session.
4. Score the plant in the field from the young shoots; a closer top-canopy shot helps if we can get it.

## Honest target

After ~100 more plants and clean flush labels: the computer’s ranking mostly matches the scout, usually within one score. It will not look like 99% “one leaf on a table” papers. Those photos are a different job.

## One sentence for a supervisor

We still cut the plant into leaf squares and mark each square damaged or healthy. We only count the young top leaves, we do not rate each square 1 to 5, and we add those squares up to match the whole-plant score.
