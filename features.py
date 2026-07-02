import cv2
import os
import numpy as np


def _load_gray_rgb(path, size=192):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    original_h, original_w = img.shape[:2]
    file_size = os.path.getsize(path)
    pixel_count = original_w * original_h
    metadata = np.array([
        float(original_w),
        float(original_h),
        float(original_w / max(original_h, 1)),
        float(file_size),
        float(file_size / max(pixel_count, 1)),
        float(np.log1p(file_size)),
        float(np.log1p(pixel_count)),
    ], dtype=np.float64)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray, metadata


def _fft_features(gray):
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    low_mask = dist <= (min(h, w) * 0.06)
    high_mask = dist > (min(h, w) * 0.25)

    low_energy = mag[low_mask].mean()
    high_energy = mag[high_mask].mean()
    hf_ratio = high_energy / (low_energy + 1e-6)

    mid_mask = (dist > min(h, w) * 0.08) & (dist < min(h, w) * 0.25)
    mid_vals = mag[mid_mask]
    peak_score = 0.0
    if mid_vals.size > 0:
        mu, sigma = mid_vals.mean(), mid_vals.std() + 1e-6
        peak_score = float(((mid_vals - mu) / sigma > 3.5).mean())

    return hf_ratio, peak_score


def _sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _texture_uniformity(gray):
    h, w = gray.shape
    center = gray[1:-1, 1:-1].astype(np.int32)
    codes = np.zeros_like(center, dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for i, (dy, dx) in enumerate(offsets):
        neighbor = gray[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx].astype(np.int32)
        codes |= ((neighbor >= center) << i).astype(np.uint8)
    hist, _ = np.histogram(codes, bins=256, range=(0, 256), density=True)
    uniformity = float(np.sum(hist ** 2))
    entropy = float(-np.sum(hist * np.log2(hist + 1e-9)))
    return uniformity, entropy


def _color_stats(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    sat_mean = float(s.mean())
    sat_std = float(s.std())

    b, g, r = cv2.split(img_bgr.astype(np.float32))
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    blue_bias = float(b.mean() - r.mean())

    clip_ratio = float((v > 0.97).mean())

    return sat_mean, sat_std, colorfulness, blue_bias, clip_ratio, float(v.mean()), float(v.std())


def _edge_density(gray):
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    thresh = mag.mean() + mag.std()
    density = float((mag > thresh).mean())
    return density


def _glare_stats(gray):
    _, thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    if num_labels <= 1:
        return 0.0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    small_blobs = areas[(areas > 3) & (areas < 400)]
    blob_count = float(len(small_blobs))
    blob_area_frac = float(small_blobs.sum() / (gray.shape[0] * gray.shape[1])) if len(small_blobs) else 0.0
    return blob_count, blob_area_frac


def extract_features(path):
    img, gray, metadata = _load_gray_rgb(path)

    hf_ratio, peak_score = _fft_features(gray)
    sharpness = _sharpness(gray)
    uniformity, entropy = _texture_uniformity(gray)
    sat_mean, sat_std, colorfulness, blue_bias, clip_ratio, v_mean, v_std = _color_stats(img)
    edge_density = _edge_density(gray)
    blob_count, blob_area_frac = _glare_stats(gray)

    image_feats = np.array([
        hf_ratio,
        peak_score,
        sharpness,
        uniformity,
        entropy,
        sat_mean,
        sat_std,
        colorfulness,
        blue_bias,
        clip_ratio,
        v_mean,
        v_std,
        edge_density,
        blob_count,
        blob_area_frac,
    ], dtype=np.float64)

    return np.concatenate([image_feats, metadata])


FEATURE_NAMES = [
    "hf_ratio", "moire_peak_score", "sharpness", "texture_uniformity", "texture_entropy",
    "sat_mean", "sat_std", "colorfulness", "blue_bias", "clip_ratio",
    "v_mean", "v_std", "edge_density", "glare_blob_count", "glare_blob_area_frac",
    "original_width", "original_height", "aspect_ratio", "file_size_bytes",
    "bytes_per_pixel", "log_file_size", "log_pixel_count",
]
