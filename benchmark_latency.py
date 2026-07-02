import glob
import os
import time

import joblib

from features import extract_features

MODEL_PATH = "model.pkl"
DATA_DIR = "data"


def list_all_images():
    files = []
    for sub in ("real", "screen"):
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            files.extend(glob.glob(os.path.join(DATA_DIR, sub, ext)))
    return files


def main():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    scaler = bundle["scaler"]

    files = list_all_images()
    if not files:
        print("No images found in data/real or data/screen.")
        return

    for f in files[:3]:
        feats = extract_features(f).reshape(1, -1)
        model.predict_proba(scaler.transform(feats))

    times = []
    for f in files:
        t0 = time.perf_counter()
        feats = extract_features(f).reshape(1, -1)
        feats_scaled = scaler.transform(feats)
        _ = model.predict_proba(feats_scaled)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    n = len(times)
    print(f"Images benchmarked: {n}")
    print(f"Mean latency:   {sum(times) / n:.2f} ms")
    print(f"Median latency: {times[n // 2]:.2f} ms")
    print(f"p95 latency:    {times[int(n * 0.95)]:.2f} ms")
    print(f"Min / Max:      {times[0]:.2f} ms / {times[-1]:.2f} ms")


if __name__ == "__main__":
    main()
