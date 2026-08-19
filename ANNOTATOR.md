# Annotator kit

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

Copy the tiled folder from the server onto this computer. On home: type your name, then drop that folder, pick it, or paste its path.

The folder must already be tiled (`tiles_foliage.csv` plus the leaf squares). Raw rover photos alone will not start a session.

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

Close the browser whenever you want. Open the same address. You return to the same session and tile.

Home lists every session on this computer.

## Export / send back

Export CSV writes into that session folder:

`annotator/data/sessions/<session>/exports/labels.csv`

Copy that CSV (or the whole session folder) back onto the GPU. Do not `git add` or `git push` labels, photos, or settings.

## Do not

- Do not commit `annotator/data/`, `settings.json`, `.env`, or `.venv`
- Do not put a GPU password in this repository
- Do not expect new raw photos to become tiles on this computer
