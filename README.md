# ThripsDetection

Lab tools for scoring chilli thrips injury on young blueberry plants from rover RGB photos (UF / IFAS, Gainesville).

The shipped job is a **plant score**, not an insect count. Insects are too small to see on these frames. Damage shows on the **young flush**: bronze or silver patches, cup, crinkle, curl.

## What is in this repo

- `PLAN.md` — locked method: how we tile, what we label, which models we will bake off, how a plant score is formed
- `annotator/` — local labeling UI (tissue → injury → curl)
- `segment_and_tile.py` / `filter_tube_tiles.py` — Center BiRefNet plant crop, then 512 px tiles
- `dino_probe.py`, `probe_plant_scores.py`, `compare_classical_scores.py`, `calibrate_thrips_scores.py` — early probes
- `human_thrips_scores.csv` — whole-plant scores for the first 36 rover photos

Photos, foliage tiles, the SQLite label database, and GPU passwords stay **off git**. They live in `annotator/data/` on the machine that runs the UI.

## Label the tiles

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r annotator/requirements.txt
python -m annotator --port 8765
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765).

1. Enter your name and load the existing 627 foliage tiles (or a local tile folder).
2. For each square: **flush / mature / tube**. Mature and tube save as skip.
3. If flush: **healthy / injured / skip**, then **curl yes / no**.
4. Mixed flush = injured. Lime crinkle with no bronze or cup = healthy. No 1–5 on a square.
5. Keys: `1` `2` `3` tissue then injury, `Y` `N` curl, `Z` undo, `Esc` back.
6. Review the four columns, then export CSV.

GPU host, user, and password are saved only in local `annotator/data/settings.json` (gitignored). Do not commit that file.

## Do not

- Score tiles 1–5
- Train attention-MIL or insect detectors
- Put GPU passwords, API keys, or raw rover photos in this repository
