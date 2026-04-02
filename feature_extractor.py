import cv2
import numpy as np
import os
import csv


# ------------------ HELPERS ------------------

def color_entropy(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [36], [0, 180]).flatten()
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    return -(p * np.log2(p)).sum()


def lbp_texture(gray, radius=1, n_points=8):
    """Simple LBP texture measure."""
    rows, cols = gray.shape
    lbp = np.zeros_like(gray, dtype=np.float32)
    for i in range(radius, rows - radius):
        for j in range(radius, cols - radius):
            center = gray[i, j]
            code = 0
            for k in range(n_points):
                angle = 2 * np.pi * k / n_points
                ni = int(round(i - radius * np.sin(angle)))
                nj = int(round(j + radius * np.cos(angle)))
                if 0 <= ni < rows and 0 <= nj < cols:
                    code |= (1 << k) if gray[ni, nj] >= center else 0
            lbp[i, j] = code
    return lbp.var() / 10000


def saturation_mean(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1].mean() / 255.0


# ------------------ MAIN EXTRACTOR ------------------

def extract_features_image(img):
    if img is None:
        return None

    h, w = img.shape[:2]

    y1, y2 = int(h * 0.15), int(h * 0.92)
    x1, x2 = int(w * 0.08), int(w * 0.92)

    if y2 <= y1 or x2 <= x1:
        return None

    crop = img[y1:y2, x1:x2]
    img = cv2.resize(crop, (300, 300))

    # CLAHE for better feature extraction under varying lighting
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img = cv2.GaussianBlur(img, (5, 5), 0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # RED DETECTION (inflammation)
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 60, 50]), np.array([12, 255, 255])),
        cv2.inRange(hsv, np.array([168, 60, 50]), np.array([180, 255, 255]))
    )

    # YELLOW DETECTION (pus)
    mask_yellow = cv2.inRange(hsv, np.array([15, 70, 80]), np.array([38, 255, 255]))

    # DARK DETECTION (necrosis)
    mask_dark = cv2.inRange(gray, 0, 60)

    # PINK/PALE DETECTION (healing tissue)
    mask_pink = cv2.inRange(hsv, np.array([140, 20, 150]), np.array([170, 120, 255]))

    kernel = np.ones((5, 5), np.uint8)
    for mask in [mask_red, mask_yellow, mask_dark, mask_pink]:
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, dst=mask)

    total = img.shape[0] * img.shape[1] + 1e-9

    red_area = np.sum(mask_red > 0) / total
    yellow_area = np.sum(mask_yellow > 0) / total
    dark_area = np.sum(mask_dark > 0) / total
    pink_area = np.sum(mask_pink > 0) / total

    # RED INTENSITY
    red_channel = img[:, :, 2].astype(float)
    if np.sum(mask_red > 0) > 0:
        red_pixels = red_channel[mask_red > 0]
        red_intensity = red_pixels.mean() / 255
        red_std = red_pixels.std() / 255
    else:
        red_intensity = 0.0
        red_std = 0.0

    # EDGE
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = min((np.sum(edges > 0) / total) * 5, 1.0)

    # TEXTURE (variance-based)
    texture = gray.var() / 1000

    # ENTROPY
    entropy = color_entropy(img) / 6

    # SATURATION (helps detect inflamed tissue)
    saturation = saturation_mean(img)

    # DERIVED
    ry_ratio = red_area / (yellow_area + 1e-6)
    pus_necrosis_combined = yellow_area + dark_area

    return [
        red_area, yellow_area, dark_area, pink_area,
        red_intensity, red_std,
        edge_ratio, texture, entropy,
        saturation, ry_ratio, pus_necrosis_combined
    ]


FEATURE_NAMES = [
    "red_area", "yellow_area", "dark_area", "pink_area",
    "red_intensity", "red_std",
    "edge_ratio", "texture", "entropy",
    "saturation", "ry_ratio", "pus_necrosis_combined"
]


# ------------------ DATASET CSV ------------------

if __name__ == "__main__":
    dataset_path = "../dataset"
    output_path = "../data/data.csv"
    os.makedirs("../data", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_NAMES + ["label"])

        total = 0
        for label in ["healthy", "inflamed", "infected"]:
            folder = os.path.join(dataset_path, label)
            if not os.path.exists(folder):
                print(f"⚠️  Folder missing: {folder}")
                continue

            count = 0
            for file in os.listdir(folder):
                path = os.path.join(folder, file)
                img = cv2.imread(path)
                features = extract_features_image(img)
                if features is None:
                    continue
                writer.writerow(features + [label])
                count += 1
                total += 1

            print(f"  {label}: {count} samples")

    print(f"\n✅ data.csv created — {total} total samples")