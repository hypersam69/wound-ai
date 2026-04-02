"""
feedback_db.py — Confidence-Weighted Feedback Storage
=====================================================
Strategy 3: Corrections are weighted by how wrong the model was.

Weight formula:
  base_weight     = confidence_error (how far off the model was)
  severity_bonus  = extra weight if model said healthy but user says infected
  final_weight    = base_weight + severity_bonus  (capped at 1.0)

Higher weight = this sample teaches the model MORE during retraining.
"""

import sqlite3
import os
import json
import datetime
import shutil

# ── PATHS ────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DB_PATH        = os.path.join(BASE_DIR, "..", "data", "feedback.db")
FEEDBACK_IMGS  = os.path.join(BASE_DIR, "..", "data", "feedback_images")
RETRAIN_THRESH = 30    # retrain RF after this many APPROVED samples
MIN_PER_CLASS  = 5     # minimum approved samples per class before retrain

os.makedirs(FEEDBACK_IMGS, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Severity of correction (how bad was the model's mistake)
# Used to calculate weight
CORRECTION_SEVERITY = {
    ("healthy",  "inflamed"): 0.3,   # mild mistake
    ("healthy",  "infected"): 0.8,   # serious mistake
    ("inflamed", "healthy"):  0.3,
    ("inflamed", "infected"): 0.4,
    ("infected", "healthy"):  0.8,   # serious mistake
    ("infected", "inflamed"): 0.4,
    ("healthy",  "healthy"):  0.0,   # correct
    ("inflamed", "inflamed"): 0.0,
    ("infected", "infected"): 0.0,
}


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS feedback (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            image_path      TEXT    NOT NULL,
            ai_prediction   TEXT    NOT NULL,
            ai_confidence   REAL    NOT NULL,
            user_label      TEXT,
            was_correct     INTEGER,
            weight          REAL    DEFAULT 0.5,
            features        TEXT,
            auto_flagged    INTEGER DEFAULT 0,
            admin_approved  INTEGER DEFAULT 0,
            admin_rejected  INTEGER DEFAULT 0,
            used_in_train   INTEGER DEFAULT 0,
            notes           TEXT
        );

        CREATE TABLE IF NOT EXISTS retrain_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT    NOT NULL,
            samples_used INTEGER NOT NULL,
            cv_f1_before REAL,
            cv_f1_after  REAL,
            improved     INTEGER,
            notes        TEXT
        );
    """)
    conn.commit()
    conn.close()


def calculate_weight(ai_prediction, ai_confidence, user_label):
    """
    Calculate how much this feedback should influence retraining.

    Parameters
    ----------
    ai_prediction : str   what model said
    ai_confidence : float model confidence (0-100)
    user_label    : str   what user says it actually is

    Returns
    -------
    float   weight between 0.1 and 1.0
    """
    was_correct = (ai_prediction == user_label)

    if was_correct:
        # Correct prediction — lower weight (model already knows this)
        # But if confidence was LOW and still correct → medium weight
        # (model got lucky, should reinforce)
        if ai_confidence < 60:
            return round(0.4, 2)
        return round(0.2, 2)

    # Wrong prediction — weight based on:
    # 1. How confident the model was (high confidence + wrong = learn more)
    confidence_error = ai_confidence / 100.0

    # 2. Severity of the mistake
    severity = CORRECTION_SEVERITY.get(
        (ai_prediction, user_label), 0.4
    )

    weight = (confidence_error * 0.6) + (severity * 0.4)
    return round(min(weight, 1.0), 2)


def save_feedback(image_path, ai_prediction, ai_confidence,
                  user_label=None, features=None, auto_flagged=False):
    """
    Save prediction feedback to database.

    Parameters
    ----------
    image_path    : str   path to the image file
    ai_prediction : str   model's prediction
    ai_confidence : float model's confidence (0-100)
    user_label    : str   user's correction (None if not provided yet)
    features      : list  12 extracted features (optional, speeds up retraining)
    auto_flagged  : bool  True if flagged automatically due to low confidence

    Returns
    -------
    int   record ID
    """
    init_db()

    timestamp  = datetime.datetime.now().isoformat()
    was_correct = None
    weight      = 0.5   # default until user label known

    if user_label:
        was_correct = int(ai_prediction == user_label)
        weight = calculate_weight(ai_prediction, ai_confidence, user_label)

    # Save image copy to feedback_images/
    label_dir = os.path.join(FEEDBACK_IMGS,
                             user_label if user_label else "unlabeled")
    os.makedirs(label_dir, exist_ok=True)

    ext   = os.path.splitext(image_path)[1] or ".jpg"
    fname = f"fb_{timestamp[:19].replace(':','-')}_{ai_prediction}{ext}"
    dest  = os.path.join(label_dir, fname)

    try:
        shutil.copy2(image_path, dest)
    except Exception as e:
        dest = image_path   # fall back to original path

    features_json = json.dumps(features) if features else None

    conn = get_db()
    cur  = conn.execute("""
        INSERT INTO feedback
        (timestamp, image_path, ai_prediction, ai_confidence,
         user_label, was_correct, weight, features,
         auto_flagged, admin_approved, used_in_train)
        VALUES (?,?,?,?,?,?,?,?,?,0,0)
    """, (timestamp, dest, ai_prediction, ai_confidence,
          user_label, was_correct, weight,
          features_json, int(auto_flagged)))

    record_id = cur.lastrowid
    conn.commit()
    conn.close()

    print(f"[Feedback] id={record_id} | AI:{ai_prediction}({ai_confidence:.0f}%)"
          f" → User:{user_label} | weight={weight} | flagged={auto_flagged}")

    return record_id


def update_user_label(record_id, user_label):
    """
    Update feedback record when user provides their label later.
    Recalculates weight based on actual correction.
    """
    init_db()
    conn = get_db()
    row  = conn.execute(
        "SELECT ai_prediction, ai_confidence FROM feedback WHERE id=?",
        (record_id,)
    ).fetchone()

    if not row:
        conn.close()
        return False

    was_correct = int(row["ai_prediction"] == user_label)
    weight = calculate_weight(
        row["ai_prediction"], row["ai_confidence"], user_label
    )

    conn.execute("""
        UPDATE feedback
        SET user_label=?, was_correct=?, weight=?
        WHERE id=?
    """, (user_label, was_correct, weight, record_id))
    conn.commit()
    conn.close()

    return True


def admin_approve(record_id, notes=None):
    """Admin approves this feedback sample for training."""
    init_db()
    conn = get_db()
    conn.execute("""
        UPDATE feedback
        SET admin_approved=1, admin_rejected=0, notes=?
        WHERE id=?
    """, (notes, record_id))
    conn.commit()
    conn.close()
    _check_retrain_trigger()


def admin_reject(record_id, notes=None):
    """Admin rejects this feedback sample (won't be used in training)."""
    init_db()
    conn = get_db()
    conn.execute("""
        UPDATE feedback
        SET admin_rejected=1, admin_approved=0, notes=?
        WHERE id=?
    """, (notes, record_id))
    conn.commit()
    conn.close()


def get_pending_review():
    """Get all samples waiting for admin review."""
    init_db()
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM feedback
        WHERE admin_approved=0
          AND admin_rejected=0
          AND user_label IS NOT NULL
        ORDER BY weight DESC, timestamp DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    """Return full feedback statistics."""
    init_db()
    conn = get_db()

    total    = conn.execute("SELECT COUNT(*) as c FROM feedback").fetchone()["c"]
    correct  = conn.execute("SELECT COUNT(*) as c FROM feedback WHERE was_correct=1").fetchone()["c"]
    approved = conn.execute("SELECT COUNT(*) as c FROM feedback WHERE admin_approved=1 AND used_in_train=0").fetchone()["c"]
    rejected = conn.execute("SELECT COUNT(*) as c FROM feedback WHERE admin_rejected=1").fetchone()["c"]
    trained  = conn.execute("SELECT COUNT(*) as c FROM feedback WHERE used_in_train=1").fetchone()["c"]
    flagged  = conn.execute("SELECT COUNT(*) as c FROM feedback WHERE auto_flagged=1").fetchone()["c"]

    by_class = conn.execute("""
        SELECT user_label, COUNT(*) as cnt, AVG(weight) as avg_w
        FROM feedback WHERE user_label IS NOT NULL
        GROUP BY user_label
    """).fetchall()

    retrains = conn.execute("SELECT COUNT(*) as c FROM retrain_log").fetchone()["c"]

    conn.close()

    accuracy = round(correct / total * 100, 1) if total > 0 else 0

    return {
        "total"         : total,
        "ai_accuracy"   : accuracy,
        "approved"      : approved,
        "rejected"      : rejected,
        "trained_on"    : trained,
        "auto_flagged"  : flagged,
        "retrains_done" : retrains,
        "by_class"      : {r["user_label"]: {
                              "count": r["cnt"],
                              "avg_weight": round(r["avg_w"], 2)
                           } for r in by_class}
    }


def _check_retrain_trigger():
    """
    Check if enough approved samples exist to trigger RF retrain.
    Called automatically after each admin approval.
    """
    init_db()
    conn = get_db()

    rows = conn.execute("""
        SELECT user_label, COUNT(*) as cnt
        FROM feedback
        WHERE admin_approved=1 AND used_in_train=0
          AND user_label IS NOT NULL
        GROUP BY user_label
    """).fetchall()
    conn.close()

    counts = {r["user_label"]: r["cnt"] for r in rows}
    total  = sum(counts.values())

    print(f"[Retrain check] approved unused: {counts} total={total}")

    if total >= RETRAIN_THRESH:
        has_enough_per_class = all(
            counts.get(c, 0) >= MIN_PER_CLASS
            for c in ["healthy", "inflamed", "infected"]
        )
        if has_enough_per_class:
            print("[Retrain] Threshold reached — starting RF retrain...")
            retrain_rf()
        else:
            missing = {c: MIN_PER_CLASS - counts.get(c, 0)
                      for c in ["healthy","inflamed","infected"]
                      if counts.get(c, 0) < MIN_PER_CLASS}
            print(f"[Retrain] Threshold reached but need more per class: {missing}")


def retrain_rf():
    """
    Retrain Random Forest using original data + approved feedback.
    Samples are weighted by their feedback weight score.
    Only replaces model if new CV F1 >= old CV F1.
    """
    import pandas as pd
    import numpy as np
    import joblib
    import cv2
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    print("\n[Retrain] Starting weighted RF retrain...")

    base_csv = os.path.join(BASE_DIR, "..", "data", "data.csv")
    model_path  = os.path.join(BASE_DIR, "..", "models", "model.pkl")
    scaler_path = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")

    if not os.path.exists(base_csv):
        print("[Retrain] data.csv not found — run feature_extractor.py first")
        return

    # Load base data
    df_base = pd.read_csv(base_csv)
    base_weights = [1.0] * len(df_base)   # base data has weight 1.0

    # Load approved feedback
    init_db()
    conn = get_db()
    fb_rows = conn.execute("""
        SELECT * FROM feedback
        WHERE admin_approved=1 AND used_in_train=0
          AND user_label IS NOT NULL
    """).fetchall()
    conn.close()

    print(f"[Retrain] Base samples: {len(df_base)} | Feedback samples: {len(fb_rows)}")

    feature_cols = [
        "red_area","yellow_area","dark_area","pink_area",
        "red_intensity","red_std","edge_ratio","texture",
        "entropy","saturation","ry_ratio","pus_necrosis_combined"
    ]

    feedback_rows   = []
    feedback_weights = []

    for row in fb_rows:
        feats = None

        # Use stored features if available
        if row["features"]:
            try:
                feats = json.loads(row["features"])
            except:
                pass

        # Re-extract from image if needed
        if feats is None:
            try:
                import sys
                sys.path.insert(0, BASE_DIR)
                from feature_extractor import extract_features_image
                img = cv2.imread(row["image_path"])
                if img is not None:
                    feats = extract_features_image(img)
            except Exception as e:
                print(f"[Retrain] Feature extract failed for {row['image_path']}: {e}")

        if feats and len(feats) == 12:
            feedback_rows.append(feats + [row["user_label"]])
            # Weight multiplied by 3 to amplify feedback influence
            feedback_weights.append(min(row["weight"] * 3.0, 3.0))

    if feedback_rows:
        df_fb = pd.DataFrame(feedback_rows, columns=feature_cols + ["label"])
        df_all = pd.concat([df_base, df_fb], ignore_index=True)
        sample_weights = np.array(base_weights + feedback_weights)
    else:
        df_all = df_base
        sample_weights = np.array(base_weights)

    X = df_all[feature_cols].values
    y = df_all["label"].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # Evaluate CURRENT model first (before replacing)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    old_f1 = None
    if os.path.exists(model_path):
        try:
            old_model = joblib.load(model_path)
            old_sc    = joblib.load(scaler_path)
            X_old_sc  = old_sc.transform(df_base[feature_cols].values)
            old_scores = cross_val_score(
                old_model, X_old_sc,
                df_base["label"].values,
                cv=cv, scoring="f1_macro"
            )
            old_f1 = round(old_scores.mean(), 4)
            print(f"[Retrain] Old model CV F1: {old_f1}")
        except Exception as e:
            print(f"[Retrain] Could not evaluate old model: {e}")

    # Train new model with sample weights
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=10,
        min_samples_split=3, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_sc, y, sample_weight=sample_weights)

    # Evaluate new model
    new_scores = cross_val_score(rf, X_sc, y, cv=cv, scoring="f1_macro")
    new_f1 = round(new_scores.mean(), 4)
    print(f"[Retrain] New model CV F1: {new_f1}")

    # Only deploy if improved (or no old model exists)
    improved = old_f1 is None or new_f1 >= old_f1

    if improved:
        calibrated = CalibratedClassifierCV(rf, method="isotonic", cv=3)
        calibrated.fit(X_sc, y)
        joblib.dump(calibrated, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"[Retrain] ✅ New model deployed! F1: {old_f1} → {new_f1}")
    else:
        print(f"[Retrain] ⚠️  New model NOT deployed (F1 worse: {old_f1} → {new_f1})")

    # Mark feedback as used regardless
    conn = get_db()
    conn.execute("""
        UPDATE feedback SET used_in_train=1
        WHERE admin_approved=1 AND used_in_train=0
    """)
    conn.execute("""
        INSERT INTO retrain_log
        (timestamp, samples_used, cv_f1_before, cv_f1_after, improved, notes)
        VALUES (?,?,?,?,?,?)
    """, (datetime.datetime.now().isoformat(),
          len(feedback_rows), old_f1, new_f1,
          int(improved),
          f"Weighted retrain with {len(feedback_rows)} feedback samples"))
    conn.commit()
    conn.close()

    return {"old_f1": old_f1, "new_f1": new_f1, "improved": improved}


if __name__ == "__main__":
    init_db()
    stats = get_stats()
    print("\n── Feedback DB Stats ──")
    for k, v in stats.items():
        print(f"  {k}: {v}")