# Agent handoff — ThripsDetection annotator

Read this before changing the app, the UI, or anything about where files live. Short human setup is in `ANNOTATOR.md`. Locked science is in `PLAN.md`.

Repo: https://github.com/SFP-team/ThripsDetection (private). Branch: `main`.

Rohit (git: `raghav-rathi`) is the engineer. He is **not** the annotator. Annotators clone this repo on their own computer, label there, and send work back. Do not assume the person at the keyboard knows the GPU password or wants a live GPU session.

## What this product is

Rover RGB photos of young blueberries. We score **thrips damage on tissue**, not insects. Insects are too small.

The shipped UI labels **tiles**, not whole plants 1–5.

Pipeline:

1. Center BiRefNet cuts the plant out.
2. 512 px tiles, stride 384.
3. Foliage filter keeps leaf squares.
4. A person marks tissue, then injury/curl on new growth only.
5. Later: flush-weighted injured + curl + DINOv3 + ordinal plant score. Need about 100 labeled plants.

Do **not** score tiles 1–5. Do **not** train attention-MIL or insect detectors.

## Product model (locked)

The app is **local**. The GPU is used in two short jobs only:

1. **GPU photos:** send raw JPGs → segment on the GPU → pull tiles back → new local session.
2. **Export to GPU:** write local CSV first, then merge onto the GPU.

While someone labels, there is **no live GPU connection**. No GPU job queue. One person on the GPU at a time. Host-key policy is `AutoAddPolicy` (lab network). Do not add extra SSH hardening unless asked.

GPU slot for now is **`annotator_1` only**. Do not add `annotator_2` unless asked.

Never reuse an old tiled run by matching JPG names on the GPU path. `_reuse_existing` in `annotator/pipeline.py` is for the old prepare flow. The GPU photos path must not call it.

## Setup on an annotator computer

```bash
git clone https://github.com/SFP-team/ThripsDetection.git
cd ThripsDetection
python3 -m venv .venv
source .venv/bin/activate
pip install -r annotator/requirements.txt
python -m annotator --port 8765
```

Windows: `.venv\Scripts\activate`.

Open http://127.0.0.1:8765

After a UI or code change on GitHub:

```bash
git pull
source .venv/bin/activate
python -m annotator --port 8765
```

`git pull` does not touch `annotator/data/`. Marks stay.

The server is uvicorn with **`reload=False`**. After Python changes, restart `python -m annotator --port 8765`. Static files (`annotator/static/*`) are read from disk per request; a browser hard refresh is enough for HTML/CSS/JS.

`--reset-kit` wipes local data down to one unlabeled 627-tile batch from `annotator/data/cache/existing_run/`. It also **deletes `settings.json`**. Do not run it on an annotator machine unless Rohit asked.

## Secrets (hand-given, never git)

The GPU card stays off until this computer has host, user, and password.

Put **one** of these on the machine (handed on USB / AirDrop / private message, never GitHub):

- `annotator/data/settings.json`
- or `.env` at the repo root
- or `annotator/data/.env`

Required fields / env vars:

| settings.json | .env |
|---|---|
| `ssh_host` | `ANNOTATOR_SSH_HOST` |
| `ssh_user` | `ANNOTATOR_SSH_USER` |
| `ssh_password` | `ANNOTATOR_SSH_PASSWORD` |
| `remote_project` | `ANNOTATOR_REMOTE_PROJECT` |
| `remote_python` | `ANNOTATOR_REMOTE_PYTHON` |
| `remote_work` | `ANNOTATOR_REMOTE_WORK` |
| `annotator_slot` | `ANNOTATOR_SLOT` (keep `annotator_1`) |
| `device` | `ANNOTATOR_DEVICE` (usually `cuda:0`) |

Default GPU host is `10.248.22.167`, user `fpt`. Remote project is `/home/fpt/RaghavWork/Segmentation_ResearchPaper`. Remote work root is `/home/fpt/Chili thrips detection pictures/annotator_jobs`.

**Never** put the password in this repo, in a commit, or in a markdown file. `GET /api/settings` must never return `ssh_password`. It returns `ssh_password_set` and `gpu_ready` only. `POST /api/settings` does **not** save; do not build a settings form that writes the password into git.

`.gitignore` already blocks `annotator/data/`, `.env`, `.env.*`, `*.db`, `settings.json`.

## Home and navigation

Home (`/`) is the **default**. Opening the app does not jump into the last label screen. Label/Review open only from Continue, a session in the list, or a `/batches/<id>/label` or `/review` URL (refresh mid-work still works).

Name once at the top, then three cards:

1. **Continue** — latest local session (`/api/sessions` first row).
2. **Local tiles** — already tiled folder (`tiles_foliage.csv` + `foliage_tiles/` or `tiles/`).
3. **GPU photos** — raw JPG/PNG. Disabled when `gpu_ready` is false.

A tiled folder on the GPU card is rejected: “That folder is already tiled. Use Local tiles, not GPU photos.” Raw photos on the Local tiles card fail with the `tiles_foliage.csv` missing toast.

Website copy: no em dashes. Title: “Tiles annotator for thrips detection.”

## Label protocol (do not change without Rohit)

Storage keys (not display words):

| UI | key |
|---|---|
| New growth | `flush` |
| Old leaves | `mature` |
| Not a leaf | `tube` |
| Healthy / Injured / Skip | `healthy` / `injured` / `skip` |
| Curl yes / no | `yes` / `no` |

Rules:

- Every tile gets tissue.
- Mature and tube save as `injury=skip`, no curl.
- Flush skip: no curl.
- Flush healthy or injured **requires** curl.
- Mixed flush = injured.
- Lime crinkle with no bronze or cup = healthy.
- Curl is separate from color: cupped, twisted, or crinkled = yes.
- No 1–5 on a square.

Keys: `1` `2` `3` tissue then injury, `Y` `N` curl, `Z` undo, `Esc` clears this tile’s draft. Arrow keys move along the plant filmstrip.

Review columns: new growth healthy, new growth injured, old leaves, not a leaf. Click a review tile, fix it, save, return to Review (`fromReview`).

Export columns (join key is tile filename, e.g. `IMG_0327_x737_y784.jpg`):

`image, tile, rel_y, tissue, injury, curl, label, annotator, labeled_at`

`label` is a copy of `injury` for older tools.

## How to change the UI

Frontend is a FastAPI-served SPA, not React.

| File | What |
|---|---|
| `annotator/static/index.html` | Home cards, label steps, review columns, nav |
| `annotator/static/app.js` | All client behavior, keys, resume, GPU job poll |
| `annotator/static/styles.css` | Light theme, three-card home, label layout |
| `annotator/static/logo.png` | Blueberry Breeding Program mark |
| `annotator/app.py` | Routes and APIs |
| `annotator/sessions.py` | Local session folders |
| `annotator/gpu_jobs.py` | SSH send / segment / pull / export merge |
| `annotator/export_merge.py` | Create / append / new-folder rules |
| `annotator/db.py` | SQLite labels |
| `annotator/config.py` | Settings load, public settings, remote root |
| `annotator/pipeline.py` | Tile ingest, old SSH prepare, SSH helpers |

Keep the protocol and storage keys stable. You can change button wording, layout, and help text. If you change keys (`flush` / `mature` / `tube`), you break exports and training.

After HTML/CSS/JS: hard refresh. After Python: restart the process.

## Where things are saved — this computer

Root: `annotator/data/` (gitignored).

```
annotator/data/
  settings.json          # optional, hand-given
  .env                   # optional, hand-given
  labels.db              # SQLite: batches, images, tiles, labels, undo_stack
  last_session.json      # last session_key + batch_id
  jobs/<job_id>.json     # GPU job progress (also kept in memory)
  cache/existing_run/    # optional local copy of the 627-tile birefnet run
  sessions/<session_key>/
    session.json         # name, annotator, created_at, tiles, source
    incoming/            # copied or downloaded tile run
      tiles_foliage.csv
      images.csv
      foliage_tiles/
      plant_crops/
    photos/              # GPU upload originals (GPU path only)
    exports/
      labels.csv         # always written on Export CSV or Export to GPU
      progress.json
  media/<batch_id>/      # tile/crop copies when the run is not already under data/
```

Session key: `{YYYYMMDD-HHMMSS}_{slug}` from the session name. Unique. Never `batch_{id}` on the GPU.

Labels live in SQLite. Each save **inserts** a new `labels` row. Display and export use the latest `id` per tile. Undo deletes the last `undo_stack` row and that label. Name on the label row is the Home name field (`thrips.annotator.name` in localStorage).

Browser only (not the source of truth):

- `localStorage["thrips.annotator.name"]`
- `localStorage["thrips.annotator.session"]` — `{batchId, tileId, screen, plantFilter, fromReview}`
- `sessionStorage["thrips.draft.<tileId>"]` — in-progress tissue/injury/curl

Home does not auto-restore Label from localStorage. Continue does, for the latest session, and keeps the tile if that session matches.

`git pull` never deletes this folder. Do not `git add` it.

## Where things are saved — GPU

Remote root:

`/home/fpt/Chili thrips detection pictures/annotator_jobs/annotator_1/`

```
annotator_1/
  jobs/<YYYYMMDD-HHMMSS>-<job_id>/
    images/                 # raw JPGs uploaded for that job
    segment_and_tile.py     # copied from this repo for that job
    filter_tube_tiles.py
    run/                    # BiRefNet output
      tiles_foliage.csv
      images.csv
      foliage_tiles/
      plant_crops/
      tiles/
      masks/
      overlays/
  exports/                  # written ONLY on Export to GPU
    labels.csv              # first export, or append target when images overlap
    <session_key>/labels.csv  # when local images do not overlap any existing CSV
```

A GPU photos job always creates a **new** `jobs/<stamp>-<id>/`. It never writes `batch_<local_id>`. It never reads `analysis_2026-08-17/birefnet_tiles` as a cache.

Rover originals on the server (not owned by this app):

`/home/fpt/Chili thrips detection pictures/IMG_*.JPG`

Older tiled run (local-tiles source, not GPU-photos cache):

`/home/fpt/Chili thrips detection pictures/analysis_2026-08-17/birefnet_tiles`

## Session flows

**Local tiles**

1. User drops a folder or pastes a path.
2. `find_tile_run` requires `tiles_foliage.csv`.
3. Folder is copied to `sessions/<key>/incoming/`.
4. `ingest_run` fills SQLite and `media/` if needed.
5. Label only on this computer.

**GPU photos**

1. Need `gpu_ready`.
2. Reject if `tiles_foliage.csv` is in the folder.
3. SSH: mkdir job + exports, SFTP photos (flattened to `path.name`), copy the two scripts, run `segment_and_tile.py` then `filter_tube_tiles.py` with remote venv Python, tar-download `tiles_foliage.csv`, `images.csv`, `foliage_tiles`, `plant_crops` into `sessions/<key>/incoming/`.
4. `finish_tile_session(..., source="gpu")`.
5. SSH disconnects. Label locally.

Progress steps: `sending` → `cutting` → `leaves` → `copying` → `ready`. Poll `GET /api/jobs/<id>`.

## Export merge (locked)

Export CSV always writes local `sessions/<key>/exports/labels.csv` first.

Export to GPU then:

| Situation | Action |
|---|---|
| No local labels | `empty` — do not write remote |
| No remote CSV | `create` `exports/labels.csv` |
| Remote CSV exists and **image names overlap** | `append` local `(image, tile)` pairs not already there |
| Same tile, different mark | **keep the GPU row** |
| Remote CSV exists, **no image overlap** | `new_folder` `exports/<session_key>/labels.csv` |
| Retry | same compare/append (resume) |

Implementation: `annotator/export_merge.py`. Tests: `annotator/tests/test_export_merge.py`.

## APIs

| Method | Path | Role |
|---|---|---|
| GET | `/` `/batches/{id}/label` `/batches/{id}/review` | SPA |
| GET | `/api/settings` | Public settings. No password. |
| POST | `/api/settings` | Returns public settings. Does not save. |
| GET | `/api/sessions` | Ready sessions, newest first |
| POST | `/api/sessions/from-path` | Local tiled folder path |
| POST | `/api/sessions/upload` | Local tiled folder upload |
| POST | `/api/sessions/gpu` | GPU photos from a path |
| POST | `/api/sessions/gpu-upload` | GPU photos upload |
| GET | `/api/jobs/{id}` | GPU job status |
| GET | `/api/batches/{id}/next` | Tile + plant filmstrip |
| POST | `/api/label` | Save one tile |
| POST | `/api/undo` | Undo last save in that batch |
| GET | `/api/batches/{id}/review` | Review columns |
| GET | `/api/batches/{id}/export` | Download CSV + write local export |
| POST | `/api/batches/{id}/export-gpu` | Local write + GPU merge |
| GET | `/media/tile/{id}` `/media/crop/{id}` `/media/context/{id}` | Images |

`POST /api/prepare` and `/api/prepare/upload` return 400 (“This copy only labels tiles already in the folder.”). Do not re-enable old “tiles already here” clone-every-time import.

## Git rules

- Never commit `annotator/data/`, `.env`, `settings.json`, `.venv`, photos, or labels.
- Never put the GPU password in the repo.
- Never add `Co-authored-by: Cursor` or Cursor as author. Use the existing git user. Do not change git config.
- Commit only when the user asks. Do not force-push `main` unless they ask.
- Leftover `BlueberryBreedingprogram.png` is not part of the app. Do not commit it unless asked.

## Do not

- Score tiles 1–5
- Change storage keys without a migration plan
- Call `_reuse_existing` from the GPU photos path
- Write GPU exports except from Export to GPU
- Build a shared live GPU URL as the main workflow
- Add a GPU queue or `annotator_2` unless asked
- Expect raw photos to become tiles on a machine with no settings file
- Copy this laptop’s whole `annotator/data/` onto an annotator PC (that copies test sessions). Send only `settings.json` or `.env`.
