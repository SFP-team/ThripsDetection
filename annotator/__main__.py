from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab tile annotator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--reset-kit",
        action="store_true",
        help="Wipe local data down to one unlabeled 627-tile batch",
    )
    args = parser.parse_args()
    if args.reset_kit:
        from annotator.kit import reset_clean_kit

        batch_id, count = reset_clean_kit()
        print(f"Clean kit ready: batch {batch_id}, {count} tiles")
        return
    uvicorn.run("annotator.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
