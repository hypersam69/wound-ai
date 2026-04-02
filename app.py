"""
app.py — Wound AI Analyzer
Production Flask backend with feedback + admin dashboard.
"""

import os
import uuid
import logging
from flask import (Flask, render_template, request,
                   jsonify, url_for, send_file)
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config.update(
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024,
    UPLOAD_FOLDER      = os.path.join("static", "uploads"),
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff"},
    SECRET_KEY         = os.environ.get("SECRET_KEY", "wound-ai-secret-key"),
)
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ── REGISTER ADMIN BLUEPRINT ─────────────────────────────────
from admin import admin_bp
app.register_blueprint(admin_bp)

# ── CONFIDENCE THRESHOLD FOR AUTO-FLAGGING ───────────────────
AUTO_FLAG_THRESHOLD = 65   # predictions below this % are auto-flagged

# ── LAZY MODEL LOADING ───────────────────────────────────────
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        log.info("Loading models...")
        from predict import predict_image
        _predictor = predict_image
        log.info("Models loaded")
    return _predictor


def allowed_file(filename):
    return ("." in filename and
            filename.rsplit(".", 1)[1].lower()
            in app.config["ALLOWED_EXTENSIONS"])


def cleanup_uploads(max_files=100):
    folder = app.config["UPLOAD_FOLDER"]
    files  = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)],
        key=os.path.getmtime
    )
    for f in files[:-max_files]:
        try: os.remove(f)
        except: pass


# ── ROUTES ───────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("index.html", result=None,
                               error="No file uploaded.")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html", result=None,
                               error="No file selected.")

    if not allowed_file(file.filename):
        return render_template("index.html", result=None,
                               error="Invalid file type. Use JPG, PNG, or BMP.")

    # Save upload
    ext      = secure_filename(file.filename).rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    cleanup_uploads()

    try:
        predict_fn = get_predictor()
        result     = predict_fn(filepath)

        if "error" in result:
            return render_template("index.html", result=None,
                                   error=result["error"])

        # ── AUTO-FLAG LOW CONFIDENCE ──────────────────────────
        auto_flagged = result["confidence"] < AUTO_FLAG_THRESHOLD

        # Save every prediction to feedback DB (no user label yet)
        try:
            from feedback_db import save_feedback, init_db
            init_db()
            record_id = save_feedback(
                image_path    = filepath,
                ai_prediction = result["final"],
                ai_confidence = result["confidence"],
                user_label    = None,
                features      = None,
                auto_flagged  = auto_flagged
            )
            result["_record_id"] = record_id
            result["_auto_flagged"] = auto_flagged
        except Exception as e:
            log.warning(f"Feedback save failed: {e}")
            result["_record_id"] = None
            result["_auto_flagged"] = False

        image_url = url_for("static", filename=f"uploads/{filename}")

        log.info(f"Prediction: {result['final']} "
                 f"conf={result['confidence']}% "
                 f"flagged={auto_flagged}")

        return render_template("index.html",
                               result=result,
                               error=None,
                               image_url=image_url)

    except Exception as e:
        log.exception("Prediction failed")
        return render_template("index.html", result=None,
                               error=f"Analysis failed: {str(e)}")


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Receive user feedback (correct/wrong + optional label).
    Called via AJAX from the result page.
    """
    data      = request.get_json()
    record_id = data.get("record_id")
    user_label = data.get("user_label")

    if not record_id or not user_label:
        return jsonify({"error": "Missing record_id or user_label"}), 400

    if user_label not in ["healthy", "inflamed", "infected"]:
        return jsonify({"error": "Invalid label"}), 400

    try:
        from feedback_db import update_user_label
        success = update_user_label(record_id, user_label)
        if success:
            return jsonify({"status": "saved", "message": "Thank you for your feedback!"})
        return jsonify({"error": "Record not found"}), 404
    except Exception as e:
        log.exception("Feedback update failed")
        return jsonify({"error": str(e)}), 500


@app.route("/feedback-image/<int:record_id>")
def feedback_image(record_id):
    """Serve feedback images for admin dashboard."""
    if not session_admin_check():
        return "Unauthorized", 401

    try:
        from feedback_db import get_db, init_db
        init_db()
        conn = get_db()
        row  = conn.execute(
            "SELECT image_path FROM feedback WHERE id=?",
            (record_id,)
        ).fetchone()
        conn.close()

        if row and os.path.exists(row["image_path"]):
            return send_file(row["image_path"])
    except Exception as e:
        log.warning(f"Feedback image not found: {e}")

    return "Image not found", 404


def session_admin_check():
    from flask import session
    return session.get("admin_logged_in", False)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint."""
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    file = request.files["image"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    ext      = secure_filename(file.filename).rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        predict_fn = get_predictor()
        result     = predict_fn(filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.errorhandler(413)
def too_large(e):
    return render_template("index.html", result=None,
                           error="File too large. Max 16MB."), 413

@app.errorhandler(500)
def server_error(e):
    return render_template("index.html", result=None,
                           error="Server error. Please try again."), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)