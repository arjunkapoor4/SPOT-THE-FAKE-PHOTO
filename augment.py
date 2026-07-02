import cv2


def _rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def _center_crop_zoom(img, zoom):
    h, w = img.shape[:2]
    nh, nw = int(h / zoom), int(w / zoom)
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    cropped = img[y0:y0 + nh, x0:x0 + nw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)


def _brightness_contrast(img, alpha, beta):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def _jpeg_recompress(img, quality):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def generate_variants(img):
    variants = [img]
    variants.append(cv2.flip(img, 1))
    variants.append(_rotate(img, 4))
    variants.append(_rotate(img, -4))
    variants.append(_center_crop_zoom(img, 1.12))
    variants.append(_brightness_contrast(img, alpha=1.15, beta=10))
    variants.append(_brightness_contrast(img, alpha=0.85, beta=-10))
    variants.append(_jpeg_recompress(img, quality=60))
    return variants
