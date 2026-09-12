from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .train_plant_model import predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    payload = joblib.load(args.model)
    feature_columns = list(payload["features"])
    missing = set(feature_columns) - set(features.columns)
    if missing:
        raise ValueError(f"plant features are missing {sorted(missing)}")
    probabilities, expected = predict(payload["models"], features[feature_columns])
    identity_columns = [column for column in ["image", "collection"] if column in features.columns]
    output = features[identity_columns].copy()
    for index, score in enumerate(range(1, 6)):
        output[f"p_score_{score}"] = probabilities[:, index]
    output["predicted_expected"] = expected
    output["predicted_score"] = np.clip(np.rint(expected), 1, 5).astype(int)
    output["confidence"] = probabilities.max(axis=1)
    output["needs_review"] = output.confidence.lt(0.60)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote severity predictions for {len(output)} plants to {args.output}")


if __name__ == "__main__":
    main()
