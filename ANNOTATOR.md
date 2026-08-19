# Annotator kit

Agents working on this repo: read `AGENTS.md` first. It has setup, where every file lives (this computer and the GPU), how labels are saved, and how to change the UI.

This repo is the **app**. Pictures and labels stay off git.

- Clone or pull this repo to get UI and code updates.
- Get the tiled image folder from the GPU server separately.
- Sessions, marks, and exports live only in `annotator/data/` on this computer.

The app does not need a GPU password to label. If someone later adds a local `.env` or `settings.json` by hand, keep it on that machine. Never commit it.

## First time on the annotator computer

```bash
git clone https://github.com/SFP-team/ThripsDetection.git
cd ThripsDetection
python3 -m venv .venv
source .venv/bin/activate
pip install -r annotator/requirements.txt
python -m annotator --port 8765
```

Windows: `python -m venv .venv` then `.venv\Scripts\activate`.

Open http://127.0.0.1:8765

On home: type your name once, then pick a card.

- **Continue** opens the last session on this computer.
- **Local tiles** needs a folder that is already tiled (`tiles_foliage.csv` plus the leaf squares).
- **GPU photos** sends raw rover JPGs to the GPU, waits for leaf tiles, then starts a local session. That card stays off until this computer has a local settings file.

Raw rover photos alone will not start a Local tiles session.

## After a UI or code change

```bash
cd ThripsDetection
git pull
source .venv/bin/activate
python -m annotator --port 8765
```

`git pull` updates the app only. It does not touch `annotator/data/`. The last session and all marks stay.

## Start a session

1. Type your name.
2. Upload the tiled folder from the server.
3. The app copies it into `annotator/data/sessions/<session>/`.
4. You label. Marks stay in that session.

A new folder upload creates a **new** session. The old session is unchanged.

## Come back later

Close the browser whenever you want. Open the same address. You land on Home. Continue (or pick a session) returns you to the same work.

Home lists every session on this computer.

## GPU photos (optional)

If this computer has a local `annotator/data/settings.json` or `.env` (handed to you, never committed):

1. Type your name.
2. Use the GPU photos card. Drop raw rover JPGs, or paste their folder path.
3. The app sends them to the GPU, waits for leaf tiles, then starts a session here.
4. Label only on this computer. The GPU is not used while you label.
5. Export CSV always writes locally. **Export to GPU** is a second button. It adds new `(image, tile)` rows and does not overwrite marks already on the GPU. If the images do not overlap, it writes a new folder.

If the GPU card is greyed out, this computer has no password. Use Local tiles instead.

GPU work lands in `annotator_jobs/annotator_1/` on the server: `jobs/` for one segment run, `exports/` only when someone presses Export to GPU.

## Export / send back

Export CSV writes into that session folder:

`annotator/data/sessions/<session>/exports/labels.csv`

Copy that CSV back by hand, or use Export to GPU if this computer has settings. Do not `git add` or `git push` labels, photos, or settings.

## Do not

- Do not commit `annotator/data/`, `settings.json`, `.env`, or `.venv`
- Do not put a GPU password in this repository
- Do not expect new raw photos to become tiles on this computer
