"""
predict.py — Memory-efficient hybrid inference for Render deployment.
Uses TFLite instead of full TensorFlow to stay within 512MB RAM limit.

Local:  uses cnn_model.h5  (full TensorFlow, accurate)
Render: uses cnn_model.tflite (TFLite, 15MB RAM vs 400MB)
"""

import cv2
import numpy as np
import joblib
import os
import logging

log = logging.getLogger(__name__)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")

LABELS   = ["healthy", "inflamed", "infected"]
W_CNN    = 0.60
W_ML     = 0.30
W_MEDICAL= 0.10

# ── GLOBALS ──────────────────────────────────────────────────
_cnn_interpreter = None   # TFLite interpreter
_cnn_model       = None   # Full TF model (local only)
_rf_model        = None
_scaler          = None
_classes         = None
_use_tflite      = None   # auto-detected


def _detect_mode():
    """Detect whether to use TFLite or full TF."""
    global _use_tflite
    if _use_tflite is not None:
        return _use_tflite

    tflite_path = os.path.join(MODELS_DIR, "cnn_model.tflite")
    h5_path     = os.path.join(MODELS_DIR, "cnn_model.h5")

    # Prefer TFLite if available (saves memory on Render)
    if os.path.exists(tflite_path):
        _use_tflite = True
        log.info("Using TFLite model (memory efficient)")
    elif os.path.exists(h5_path):
        _use_tflite = False
        log.info("Using full TensorFlow model")
    else:
        raise FileNotFoundError(
            f"No CNN model found. Expected:\n"
            f"  {tflite_path}\n  {h5_path}"
        )
    return _use_tflite


def _load_all():
    """Load all models once."""
    global _cnn_interpreter, _cnn_model, _rf_model, _scaler, _classes

    if _rf_model is not None:
        return  # already loaded

    use_tflite = _detect_mode()

    # ── CNN ──────────────────────────────────────────────────
    if use_tflite:
        try:
            # Try tflite_runtime first (lighter)
            import tflite_runtime.interpreter as tflite
            _cnn_interpreter = tflite.Interpreter(
                model_path=os.path.join(MODELS_DIR, "cnn_model.tflite")
            )
        except ImportError:
            # Fall back to TFLite from TensorFlow
            import tensorflow as tf
            _cnn_interpreter = tf.lite.Interpreter(
                model_path=os.path.join(MODELS_DIR, "cnn_model.tflite")
            )
        _cnn_interpreter.allocate_tensors()
        log.info("TFLite interpreter ready")
    else:
        import tensorflow as tf
        _cnn_model = tf.keras.models.load_model(
            os.path.join(MODELS_DIR, "cnn_model.h5")
        )
        log.info("Full TF model loaded")

    # ── RF + SCALER ──────────────────────────────────────────
    rf_path = os.path.join(MODELS_DIR, "model.pkl")
    sc_path = os.path.join(MODELS_DIR, "scaler.pkl")
    cl_path = os.path.join(MODELS_DIR, "classes.pkl")

    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"RF model not found: {rf_path}")

    _rf_model = joblib.load(rf_path)
    _scaler   = joblib.load(sc_path)

    try:
        _classes = joblib.load(cl_path)
    except FileNotFoundError:
        _classes = LABELS

    log.info(f"RF model loaded. Classes: {_classes}")


def _cnn_predict(img_array):
    """
    Run CNN inference.
    img_array: float32 array (1, 224, 224, 3) values in [0, 1]
    Returns: probability array of shape (3,)
    """
    if _use_tflite:
        inp  = _cnn_interpreter.get_input_details()
        out  = _cnn_interpreter.get_output_details()

        # TFLite expects float32
        _cnn_interpreter.set_tensor(inp[0]['index'], img_array.astype(np.float32))
        _cnn_interpreter.invoke()
        return _cnn_interpreter.get_tensor(out[0]['index'])[0]
    else:
        return _cnn_model.predict(img_array, verbose=0)[0]


def _align_probs(raw_probs, source_classes, target_labels):
    aligned = np.zeros(len(target_labels))
    for i, lbl in enumerate(target_labels):
        if lbl in source_classes:
            aligned[i] = raw_probs[list(source_classes).index(lbl)]
    s = aligned.sum()
    return aligned / s if s > 0 else aligned


def _medical_to_probs(score):
    v = np.array([
        max(0.0, 1.0 - score / 30.0),
        max(0.0, 1.0 - abs(score - 25) / 25.0),
        min(1.0, score / 65.0)
    ], dtype=float)
    s = v.sum()
    return v / s if s > 0 else np.array([0.33, 0.33, 0.34])


def _agreement_confidence(cnn_p, ml_p, fused_p):
    if np.argmax(cnn_p) == np.argmax(ml_p):
        return 0.85 + 0.15 * (np.max(cnn_p) + np.max(ml_p)) / 2
    margin = np.max(fused_p) - np.sort(fused_p)[-2]
    return 0.60 + 0.25 * margin


# ── MAIN ─────────────────────────────────────────────────────

def predict_image(img_path: str) -> dict:
    try:
        _load_all()
    except Exception as e:
        return {"error": f"Model loading failed: {e}"}

    img = cv2.imread(img_path)
    if img is None:
        return {"error": "Cannot read image — may be corrupt or unsupported format"}

    # Quality check
    gray        = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    low_quality = blur_score < 30

    # ── CNN ──────────────────────────────────────────────────
    img_cnn = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    img_cnn = np.expand_dims(img_cnn, axis=0)

    cnn_raw   = _cnn_predict(img_cnn)
    cnn_p     = _align_probs(cnn_raw, _classes, LABELS)
    cnn_class = LABELS[np.argmax(cnn_p)]
    cnn_conf  = float(np.max(cnn_p)) * 100

    # ── FEATURES + ML ────────────────────────────────────────
    ml_ok = False
    features = None

    try:
        from feature_extractor import extract_features_image
        features = extract_features_image(img)
        if features and len(features) == 12:
            fs       = _scaler.transform([features])
            ml_raw   = _rf_model.predict_proba(fs)[0]
            ml_p     = _align_probs(ml_raw, _classes, LABELS)
            ml_class = LABELS[np.argmax(ml_p)]
            ml_conf  = float(np.max(ml_p)) * 100
            ml_ok    = True
    except Exception as e:
        log.warning(f"ML inference failed: {e}")

    if not ml_ok:
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

    # ── FUSION ───────────────────────────────────────────────
    fused_p = W_CNN * cnn_p + W_ML * ml_p + W_MEDICAL * med_p
    fused_p /= fused_p.sum()

    # Rule 1: CNN very confident
    if cnn_conf > 90:
        fused_p = 0.80*cnn_p + 0.15*ml_p + 0.05*med_p
        fused_p /= fused_p.sum()

    # Rule 2: Pus present but predicted healthy
    pus_area  = features[1] if features else 0
    dark_area = features[2] if features else 0
    if LABELS[np.argmax(fused_p)] == "healthy" and pus_area > 0.08:
        fused_p[0] *= 0.3
        fused_p /= fused_p.sum()

    # Rule 3: Necrosis + high score
    if LABELS[np.argmax(fused_p)] != "infected" and dark_area > 0.12 and score > 55:
        fused_p[2] += 0.25
        fused_p /= fused_p.sum()

    fused_class = LABELS[np.argmax(fused_p)]

    if cnn_class == ml_class == fused_class:
        agreement = "high"
    elif cnn_class == fused_class or ml_class == fused_class:
        agreement = "medium"
    else:
        agreement = "low"

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
            "probs":      {l: round(float(p)*100, 1) for l, p in zip(LABELS, cnn_p)}
        },
        "ml": {
            "class":      ml_class,
            "confidence": round(ml_conf, 1),
            "probs":      {l: round(float(p)*100, 1) for l, p in zip(LABELS, ml_p)}
        },
        "fused_probs": {l: round(float(p)*100, 1) for l, p in zip(LABELS, fused_p)},
        "medical":     {"score": score, "stage": stage}
    }
