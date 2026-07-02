import glob
import os
import time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from features import extract_features, FEATURE_NAMES

DATA_DIR = "data"
REAL_DIR = os.path.join(DATA_DIR, "real")
SCREEN_DIR = os.path.join(DATA_DIR, "screen")
MODEL_PATH = "model.pkl"

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def list_images(folder):
    files = set()
    for ext in IMG_EXTS:
        files.update(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def build_dataset():
    real_files = list_images(REAL_DIR)
    screen_files = list_images(SCREEN_DIR)

    print(f"Found {len(real_files)} real images, {len(screen_files)} screen images")
    if len(real_files) < 10 or len(screen_files) < 10:
        raise SystemExit("Need at least ~10 images per class (aim for 50). Fill data/real and data/screen first.")

    X, y, paths = [], [], []
    for f in real_files:
        try:
            X.append(extract_features(f))
            y.append(0)
            paths.append(f)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    for f in screen_files:
        try:
            X.append(extract_features(f))
            y.append(1)
            paths.append(f)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    return np.array(X), np.array(y), paths


def main():
    X, y, paths = build_dataset()

    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X, y, paths, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model_name = "extra_trees"
    model = ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    acc = accuracy_score(y_test, preds)

    print(f"\n=== {model_name} ===")
    print(f"Holdout accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["real", "screen"]))

    best_preds = model.predict(X_test_s)
    wrong = [(p, yt, yp) for p, yt, yp in zip(paths_test, y_test, best_preds) if yt != yp]
    if wrong:
        print("\nMisclassified holdout images:")
        for p, yt, yp in wrong:
            print(f"  {p}  true={yt} pred={yp}")

    print("\nFeature importance:")
    for name, imp in sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:24s} {imp:.4f}")

    scaler_final = StandardScaler()
    X_all_s = scaler_final.fit_transform(X)
    model.fit(X_all_s, y)

    joblib.dump({"model": model, "scaler": scaler_final, "model_name": model_name}, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    sample = X[0].reshape(1, -1)
    n_runs = 50
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = model.predict_proba(scaler_final.transform(sample))
    t1 = time.perf_counter()
    print(f"Approx classifier inference time: {(t1 - t0) / n_runs * 1000:.3f} ms (feature extraction is extra, see predict.py timing)")


if __name__ == "__main__":
    main()
