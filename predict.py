import sys
import time

import joblib

from features import extract_features

MODEL_PATH = "model.pkl"


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]

    

    bundle = joblib.load(MODEL_PATH)
    t0 = time.perf_counter()
    model = bundle["model"]
    scaler = bundle["scaler"]

    feats = extract_features(image_path).reshape(1, -1)
    feats_scaled = scaler.transform(feats)
    score = model.predict_proba(feats_scaled)[0][1]

    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000

    print(f"{score:.4f}")
    print(f"[latency: {elapsed_ms:.2f} ms, incl. model load]", file=sys.stderr)


if __name__ == "__main__":
    main()
