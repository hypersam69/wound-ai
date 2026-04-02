import os, shutil, csv
import cv2
import numpy as np

SOURCE = "../Wound_dataset"
TARGET = "../dataset"

MIN_FILE_SIZE_KB = 20

CSV_PATH = "../data/labels.csv"


def estimate_features(img):
    """Rough automatic feature estimation (bootstrap labels)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # redness
    red_mask = ((hsv[:,:,0] < 10) | (hsv[:,:,0] > 170))
    redness = np.mean(red_mask)

    # yellow (pus)
    yellow_mask = ((hsv[:,:,0] > 20) & (hsv[:,:,0] < 35))
    pus = np.mean(yellow_mask)

    # dark (necrosis)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    necrosis = np.mean(gray < 50)

    # severity (rough)
    severity = int((0.4*pus + 0.3*necrosis + 0.3*redness) * 100)

    return round(redness,3), round(pus,3), round(necrosis,3), severity


def safe_organize():

    src_abs = os.path.abspath(SOURCE)
    tgt_abs = os.path.abspath(TARGET)

    if src_abs == tgt_abs:
        raise ValueError("❌ SOURCE and TARGET cannot be same")

    if not os.path.exists(src_abs):
        raise FileNotFoundError(f"❌ SOURCE not found: {src_abs}")

    # create folders
    for split in ["healthy", "inflamed", "infected"]:
        split_path = os.path.join(TARGET, split)
        if os.path.exists(split_path):
            shutil.rmtree(split_path)
        os.makedirs(split_path)

    # prepare CSV
    os.makedirs("../data", exist_ok=True)

    with open(CSV_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "image", "label", "severity",
            "redness", "pus", "necrosis"
        ])

        counts = {"healthy":0, "inflamed":0, "infected":0}

        def copy_folder(src_folder, label):
            src = os.path.join(SOURCE, src_folder)
            dst = os.path.join(TARGET, label)

            if not os.path.exists(src):
                print(f"⚠️ Skipping missing: {src_folder}")
                return

            for f in os.listdir(src):
                if not f.lower().endswith(('.jpg','.png','.jpeg')):
                    continue

                fpath = os.path.join(src, f)

                if os.path.getsize(fpath)/1024 < MIN_FILE_SIZE_KB:
                    continue

                try:
                    img = cv2.imread(fpath)
                    if img is None:
                        continue

                    # copy image
                    new_name = f"{label}_{counts[label]}_{f}"
                    dst_path = os.path.join(dst, new_name)
                    shutil.copy2(fpath, dst_path)

                    # extract features
                    redness, pus, necrosis, severity = estimate_features(img)

                    # write CSV
                    writer.writerow([
                        new_name,
                        label,
                        severity,
                        redness,
                        pus,
                        necrosis
                    ])

                    counts[label] += 1

                except Exception as e:
                    continue

        # ---------- MAPPING ----------
        copy_folder("Abrasions", "healthy")

        copy_folder("Cut", "inflamed")
        copy_folder("Laceration", "inflamed")
        copy_folder("Burns", "inflamed")
        copy_folder("Ingrown_nails", "inflamed")

        copy_folder("Stab_wound", "infected")
        copy_folder("wound", "infected")
        copy_folder("wound 2", "infected")

    # ---------- SUMMARY ----------
    print("\n── Dataset Summary ──")
    total = sum(counts.values())

    for k,v in counts.items():
        print(f"{k:<10}: {v}")

    print(f"\nTotal: {total}")
    print(f"\nCSV saved at: {CSV_PATH}")

    print("\n✅ HYBRID DATASET READY")
    print("\nNext:")
    print("→ Train CNN (cnn_efficientnet.py)")
    print("→ Train ML (train_model.py with CSV)")
    print("→ Use hybrid fusion")


if __name__ == "__main__":
    safe_organize()