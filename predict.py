"""
predict.py — Hybrid inference engine.
CNN (60%) + ML (30%) + Medical Signal (10%)
"""

import cv2
import numpy as np
import joblib
import os
import logging

log = logging.getLogger(__name__)

# ── PATHS ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

LABELS = ["healthy", "inflamed", "infected"]

# Fusion weights
W_CNN     = 0.60
W_ML      = 0.30
W_MEDICAL = 0.10

# ── LAZY GLOBALS ─────────────────────────────────────────────
_cnn_model = None
_rf_model  = None
_scaler    = None
_classes   = None


def _load_all():
    """Load all models once and cache globally."""
    global _cnn_model, _rf_model, _scaler, _classes

    if _cnn_model is not None:
        return  # already loaded

    from tensorflow.keras.models import load_model as tf_load

    cnn_path = os.path.join(MODELS_DIR, "cnn_model.h5")
    rf_path  = os.path.join(MODELS_DIR, "model.pkl")
    sc_path  = os.path.join(MODELS_DIR, "scaler.pkl")
    cl_path  = os.path.join(MODELS_DIR, "classes.pkl")

    if not os.path.exists(cnn_path):
        raise FileNotFoundError(f"CNN model not found: {cnn_path}")
    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"RF model not found: {rf_path}")

    log.info("Loading CNN model...")
    _cnn_model = tf_load(cnn_path)

    log.info("Loading RF model...")
    _rf_model = joblib.load(rf_path)
    _scaler   = joblib.load(sc_path)

    try:
        _classes = joblib.load(cl_path)
    except FileNotFoundError:
        _classes = LABELS

    log.info(f"All models loaded. Classes: {_classes}")


# ── HELPERS ──────────────────────────────────────────────────

def _align_probs(raw_probs, source_classes, target_labels):
    aligned = np.zeros(len(target_labels))
    for i, lbl in enumerate(target_labels):
        if lbl in source_classes:
            aligned[i] = raw_probs[list(source_classes).index(lbl)]
    s = aligned.sum()
    return aligned / s if s > 0 else aligned


def _medical_to_probs(score):
    healthy_sig  = max(0.0, 1.0 - score / 30.0)
    inflamed_sig = max(0.0, 1.0 - abs(score - 25) / 25.0)
    infected_sig = min(1.0, score / 65.0)
    vec = np.array([healthy_sig, inflamed_sig, infected_sig], dtype=float)
    s = vec.sum()
    return vec / s if s > 0 else np.array([0.33, 0.33, 0.34])


def _agreement_confidence(cnn_p, ml_p, fused_p):
    if np.argmax(cnn_p) == np.argmax(ml_p):
        avg_top = (np.max(cnn_p) + np.max(ml_p)) / 2
        return 0.85 + 0.15 * avg_top
    else:
        margin = np.max(fused_p) - np.sort(fused_p)[-2]
        return 0.60 + 0.25 * margin


# ── MAIN PREDICTION ──────────────────────────────────────────

def predict_image(img_path: str) -> dict:
    """
    Full hybrid prediction pipeline.

    Parameters
    ----------
    img_path : str
        Absolute or relative path to the image file.

    Returns
    -------
    dict
        All prediction outputs, or {"error": str} on failure.
    """
    # Load models on first call
    try:
        _load_all()
    except Exception as e:
        return {"error": f"Model loading failed: {e}"}

    # Read image
    img = cv2.imread(img_path)
    if img is None:
        return {"error": "Cannot read image — file may be corrupt or unsupported format"}

    # Image quality check
    gray        = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    low_quality = blur_score < 30

    # ── CNN ──────────────────────────────────────────────────
    img_cnn = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    img_cnn = np.expand_dims(img_cnn, axis=0)

    cnn_raw   = _cnn_model.predict(img_cnn, verbose=0)[0]
    cnn_p     = _align_probs(cnn_raw, _classes, LABELS)
    cnn_class = LABELS[np.argmax(cnn_p)]
    cnn_conf  = float(np.max(cnn_p)) * 100

    # ── FEATURE EXTRACTION + ML ──────────────────────────────
    try:
        from feature_extractor import extract_features_image
        features = extract_features_image(img)
    except Exception as e:
        log.warning(f"Feature extraction failed: {e} — using CNN only")
        features = None

    if features is not None:
        try:
            fs      = _scaler.transform([features])
            ml_raw  = _rf_model.predict_proba(fs)[0]
            ml_p    = _align_probs(ml_raw, _classes, LABELS)
            ml_class = LABELS[np.argmax(ml_p)]
            ml_conf  = float(np.max(ml_p)) * 100
            ml_ok    = True
        except Exception as e:
            log.warning(f"RF inference failed: {e}")
            ml_ok = False
    else:
        ml_ok = False

    if not ml_ok:
        # Fall back to CNN only
        ml_p     = cnn_p.copy()
        ml_class = cnn_class
        ml_conf  = cnn_conf
        features = [0.0] * 12

    # ── MEDICAL LOGIC ────────────────────────────────────────
    try:
        from medical_logic import analyze_medical
        score, stage, findings, risk_flags = analyze_medical(features)
    except Exception as e:
        log.warning(f"Medical logic failed: {e}")
        score, stage, findings, risk_flags = 0, "Unknown", [], []

    med_p = _medical_to_probs(score)

    # ── WEIGHTED FUSION ──────────────────────────────────────
    fused_p = W_CNN * cnn_p + W_ML * ml_p + W_MEDICAL * med_p
    fused_p = fused_p / fused_p.sum()

    # ── DECISION CONSISTENCY RULES ───────────────────────────
    # Rule 1: CNN very confident → boost its weight
    if cnn_conf > 90:
        fused_p = 0.80 * cnn_p + 0.15 * ml_p + 0.05 * med_p
        fused_p /= fused_p.sum()

    # Rule 2: Pus detected but predicted healthy → bump up
    pus_area  = features[1] if features else 0
    dark_area = features[2] if features else 0

    if LABELS[np.argmax(fused_p)] == "healthy" and pus_area > 0.08:
        fused_p[0] *= 0.3
        fused_p[1] += fused_p[0] * 0.7
        fused_p /= fused_p.sum()

    # Rule 3: Necrosis + high score → infected
    if LABELS[np.argmax(fused_p)] != "infected" and dark_area > 0.12 and score > 55:
        fused_p[2] += 0.25
        fused_p /= fused_p.sum()

    fused_class = LABELS[np.argmax(fused_p)]

    # Agreement check
    if cnn_class == ml_class == fused_class:
        agreement = "high"
    elif cnn_class == fused_class or ml_class == fused_class:
        agreement = "medium"
    else:
        agreement = "low"

    # ── CONFIDENCE ───────────────────────────────────────────
    agree_mult = _agreement_confidence(cnn_p, ml_p, fused_p)
    final_conf = min(float(np.max(fused_p)) * 100 * agree_mult, 99.0)
    if low_quality:
        final_conf *= 0.85
    final_conf = round(final_conf, 1)

    return {
        "final":       fused_class,
        "confidence":  final_conf,
        "agreement":   agreement,
        "score":       score,
        "stage":       stage,
        "findings":    findings,
        "risk_flags":  risk_flags,
        "low_quality": low_quality,
        "blur_score":  round(blur_score, 1),
        "cnn": {
            "class":      cnn_class,
            "confidence": round(cnn_conf, 1),
            "probs": {l: round(float(p) * 100, 1)
                      for l, p in zip(LABELS, cnn_p)}
        },
        "ml": {
            "class":      ml_class,
            "confidence": round(ml_conf, 1),
            "probs": {l: round(float(p) * 100, 1)
                      for l, p in zip(LABELS, ml_p)}
        },
        "fused_probs": {l: round(float(p) * 100, 1)
                        for l, p in zip(LABELS, fused_p)},
        "medical": {
            "score": score,
            "stage": stage
        }
    }