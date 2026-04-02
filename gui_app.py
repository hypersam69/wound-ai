"""
gui_app.py — Wound AI Analyzer v2
Dark-theme, production-grade Tkinter GUI.
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import time
import os
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFilter

# Lazy model import (deferred to after window creation)
_models_loaded = False
model = scaler = cnn_model = classes = None


def load_models():
    global model, scaler, cnn_model, classes, _models_loaded
    import joblib
    from tensorflow.keras.models import load_model as tf_load

    model = joblib.load("../models/model.pkl")
    scaler = joblib.load("../models/scaler.pkl")
    cnn_model = tf_load("../models/cnn_model.h5")
    try:
        classes = joblib.load("../models/classes.pkl")
    except FileNotFoundError:
        classes = ["healthy", "inflamed", "infected"]
    _models_loaded = True


# ─────────────────────────── COLORS ───────────────────────────
BG0 = "#0d0f14"       # deepest background
BG1 = "#13161e"       # panel/card background
BG2 = "#1a1e28"       # secondary panel
BORDER = "#252a38"    # subtle border
TEXT0 = "#e8eaf0"     # primary text
TEXT1 = "#8e95aa"     # secondary text
TEXT2 = "#545b70"     # muted text
ACCENT = "#4f8ef7"    # primary accent (blue)
GREEN = "#3cba6f"
YELLOW = "#e8a628"
RED = "#e8494a"
HEALTHY_C = "#3cba6f"
INFLAMED_C = "#e8a628"
INFECTED_C = "#e8494a"

LABELS = ["healthy", "inflamed", "infected"]

# ─────────────────────────── HELPERS ───────────────────────────

def stage_color(score):
    if score < 35:
        return GREEN
    elif score < 65:
        return YELLOW
    return RED


def result_color(label):
    return {"healthy": HEALTHY_C, "inflamed": INFLAMED_C, "infected": INFECTED_C}.get(label, ACCENT)


def make_rounded_image(pil_img, radius=12):
    """Clip PIL image to rounded corners."""
    w, h = pil_img.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(pil_img.convert("RGBA"), mask=mask)
    return out


# ─────────────────────────── MAIN APP ───────────────────────────

class WoundApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wound AI Analyzer")
        self.geometry("860x780")
        self.configure(bg=BG0)
        self.resizable(True, True)
        self.minsize(720, 640)

        self._photo = None          # keep reference
        self._analyzing = False
        self._dot_frame = 0
        self._last_result = None

        self._setup_styles()
        self._build_ui()
        self._start_model_load()

    # ─── STYLES ───
    def _setup_styles(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")   # clam supports color overrides on Windows
        except Exception:
            style.theme_use("default")

        # Single style — update background color at runtime via style.configure
        style.configure("Wound.Horizontal.TProgressbar",
                        background=GREEN,
                        troughcolor=BG2,
                        borderwidth=0,
                        thickness=10)
        self._bar_style = style

    # ─── BUILD UI ───
    def _build_ui(self):
        # ── TOP BAR ──
        topbar = tk.Frame(self, bg=BG0, height=56)
        topbar.pack(fill="x", padx=0, pady=0)
        topbar.pack_propagate(False)

        tk.Label(topbar, text="⬡ Wound AI", bg=BG0, fg=ACCENT,
                 font=("Courier", 16, "bold")).pack(side="left", padx=20, pady=12)

        self.status_dot = tk.Label(topbar, text="●", bg=BG0, fg=RED, font=("Courier", 10))
        self.status_dot.pack(side="right", padx=6)
        self.status_lbl = tk.Label(topbar, text="Loading models…", bg=BG0, fg=TEXT1,
                                   font=("Courier", 10))
        self.status_lbl.pack(side="right", padx=0)

        # Separator line
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # ── MAIN BODY ──
        body = tk.Frame(self, bg=BG0)
        body.pack(fill="both", expand=True, padx=20, pady=16)

        # Left column: image
        left = tk.Frame(body, bg=BG0)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self._image_card(left)
        self._action_buttons(left)

        # Right column: results
        right = tk.Frame(body, bg=BG0)
        right.pack(side="right", fill="both", expand=True)

        self._result_card(right)
        self._findings_card(right)

        # ── BOTTOM BAR ──
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        bottom = tk.Frame(self, bg=BG0, height=36)
        bottom.pack(fill="x")
        bottom.pack_propagate(False)
        tk.Label(bottom, text="Disclaimer: Not a substitute for clinical diagnosis.",
                 bg=BG0, fg=TEXT2, font=("Courier", 9)).pack(side="left", padx=20, pady=8)

    def _image_card(self, parent):
        card = tk.Frame(parent, bg=BG1, bd=0, highlightthickness=1,
                        highlightbackground=BORDER)
        card.pack(fill="both", expand=True, pady=(0, 10))

        # Canvas for image display
        self.img_canvas = tk.Canvas(card, bg=BG1, bd=0, highlightthickness=0,
                                    width=320, height=320)
        self.img_canvas.pack(padx=16, pady=16, fill="both", expand=True)

        # Placeholder
        self._draw_placeholder()

    def _draw_placeholder(self):
        self.img_canvas.delete("all")
        w = self.img_canvas.winfo_reqwidth()
        h = self.img_canvas.winfo_reqheight()
        cx, cy = w // 2, h // 2
        self.img_canvas.create_rectangle(2, 2, w - 2, h - 2,
                                         outline=BORDER, fill=BG1, width=1)
        self.img_canvas.create_text(cx, cy - 12, text="[ ]",
                                    fill=TEXT2, font=("Courier", 28))
        self.img_canvas.create_text(cx, cy + 20, text="Select an image to analyze",
                                    fill=TEXT2, font=("Courier", 11))

    def _action_buttons(self, parent):
        row = tk.Frame(parent, bg=BG0)
        row.pack(fill="x", pady=(0, 4))

        self.btn_select = tk.Button(
            row, text="Select Image",
            bg=ACCENT, fg="#ffffff",
            activebackground="#3a7ae0", activeforeground="#ffffff",
            relief="flat", bd=0, padx=16, pady=8,
            font=("Courier", 11, "bold"),
            command=self._select_image
        )
        self.btn_select.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.btn_export = tk.Button(
            row, text="Export PDF",
            bg=BG2, fg=TEXT1,
            activebackground=BORDER, activeforeground=TEXT0,
            relief="flat", bd=0, padx=16, pady=8,
            font=("Courier", 11),
            command=self._export_pdf,
            state="disabled"
        )
        self.btn_export.pack(side="left", fill="x", expand=True)

    def _result_card(self, parent):
        card = tk.Frame(parent, bg=BG1, bd=0, highlightthickness=1,
                        highlightbackground=BORDER)
        card.pack(fill="x", pady=(0, 10))

        inner = tk.Frame(card, bg=BG1)
        inner.pack(fill="x", padx=16, pady=14)

        # Diagnosis label
        diag_row = tk.Frame(inner, bg=BG1)
        diag_row.pack(fill="x")

        tk.Label(diag_row, text="DIAGNOSIS", bg=BG1, fg=TEXT2,
                 font=("Courier", 9)).pack(side="left")

        self.conf_lbl = tk.Label(diag_row, text="", bg=BG1, fg=TEXT1,
                                 font=("Courier", 9))
        self.conf_lbl.pack(side="right")

        self.diag_lbl = tk.Label(inner, text="—", bg=BG1, fg=TEXT0,
                                 font=("Courier", 22, "bold"), anchor="w")
        self.diag_lbl.pack(fill="x", pady=(4, 0))

        self.stage_lbl = tk.Label(inner, text="", bg=BG1, fg=TEXT1,
                                  font=("Courier", 11), anchor="w")
        self.stage_lbl.pack(fill="x")

        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", pady=10)

        # Severity bar
        bar_row = tk.Frame(inner, bg=BG1)
        bar_row.pack(fill="x")

        tk.Label(bar_row, text="Severity", bg=BG1, fg=TEXT1,
                 font=("Courier", 10)).pack(side="left")
        self.score_lbl = tk.Label(bar_row, text="—", bg=BG1, fg=TEXT0,
                                  font=("Courier", 10, "bold"))
        self.score_lbl.pack(side="right")

        self.progress = ttk.Progressbar(inner, length=400, mode="determinate",
                                        style="Wound.Horizontal.TProgressbar")
        self.progress.pack(fill="x", pady=(6, 10))

        # CNN / ML row
        model_row = tk.Frame(inner, bg=BG1)
        model_row.pack(fill="x")

        self.cnn_lbl = tk.Label(model_row, text="CNN  —", bg=BG1, fg=TEXT1,
                                font=("Courier", 10))
        self.cnn_lbl.pack(side="left")

        self.ml_lbl = tk.Label(model_row, text="ML   —", bg=BG1, fg=TEXT1,
                               font=("Courier", 10))
        self.ml_lbl.pack(side="right")

        # Probability bars
        self.prob_frame = tk.Frame(inner, bg=BG1)
        self.prob_frame.pack(fill="x", pady=(10, 0))

        self._prob_bars = {}
        for lbl in LABELS:
            row = tk.Frame(self.prob_frame, bg=BG1)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=lbl.capitalize(), bg=BG1, fg=TEXT1,
                     font=("Courier", 10), width=9, anchor="w").pack(side="left")
            bar = ttk.Progressbar(row, length=200, mode="determinate",
                                  style="Wound.Horizontal.TProgressbar")
            bar.pack(side="left", fill="x", expand=True, padx=(4, 8))
            pct = tk.Label(row, text="—", bg=BG1, fg=TEXT1,
                           font=("Courier", 10), width=5, anchor="e")
            pct.pack(side="left")
            self._prob_bars[lbl] = (bar, pct)

    def _findings_card(self, parent):
        card = tk.Frame(parent, bg=BG1, bd=0, highlightthickness=1,
                        highlightbackground=BORDER)
        card.pack(fill="both", expand=True)

        tk.Label(card, text="FINDINGS", bg=BG1, fg=TEXT2,
                 font=("Courier", 9)).pack(anchor="w", padx=16, pady=(12, 0))

        self.findings_frame = tk.Frame(card, bg=BG1)
        self.findings_frame.pack(fill="both", expand=True, padx=16, pady=(6, 12))

        self._placeholder_findings()

    def _placeholder_findings(self):
        for w in self.findings_frame.winfo_children():
            w.destroy()
        tk.Label(self.findings_frame, text="No analysis yet.",
                 bg=BG1, fg=TEXT2, font=("Courier", 10),
                 anchor="w").pack(anchor="w")

    # ─── IMAGE DISPLAY ───
    def _show_image(self, path):
        img = Image.open(path)
        img = img.convert("RGB")

        cw = max(self.img_canvas.winfo_width(), 320)
        ch = max(self.img_canvas.winfo_height(), 320)
        img.thumbnail((cw - 8, ch - 8), Image.LANCZOS)

        img_rounded = make_rounded_image(img, radius=10)
        self._photo = ImageTk.PhotoImage(img_rounded)

        self.img_canvas.delete("all")
        self.img_canvas.create_image(cw // 2, ch // 2,
                                     anchor="center", image=self._photo)

    # ─── ANALYSIS ───
    def _select_image(self):
        if not _models_loaded:
            messagebox.showwarning("Models loading", "Please wait — models are still loading.")
            return

        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not path:
            return

        self._show_image(path)
        self._start_analysis(path)

    def _start_analysis(self, path):
        self._analyzing = True
        self.btn_select.config(state="disabled")
        self.btn_export.config(state="disabled")
        self._set_status("Analyzing…", YELLOW)
        self._animate_loading()
        threading.Thread(target=self._run_analysis, args=(path,), daemon=True).start()

    def _animate_loading(self):
        if not self._analyzing:
            return
        dots = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.diag_lbl.config(text=dots[self._dot_frame % len(dots)], fg=ACCENT)
        self._dot_frame += 1
        self.after(80, self._animate_loading)

    def _run_analysis(self, path):
        try:
            from predict import predict_image
            result = predict_image(path)
            self.after(0, self._show_result, result)
        except Exception as e:
            self.after(0, self._show_error, str(e))

    def _show_result(self, result):
        self._analyzing = False
        self.btn_select.config(state="normal")

        if "error" in result:
            self._show_error(result["error"])
            return

        self._last_result = result
        final = result["final"]
        score = result["score"]
        stage = result["stage"]
        conf = result["confidence"]

        c = result_color(final)

        # Diagnosis
        self.diag_lbl.config(text=final.upper(), fg=c)
        self.stage_lbl.config(text=stage)
        self.conf_lbl.config(text=f"Confidence: {conf}%")

        # Severity bar
        self.score_lbl.config(text=f"{score}%", fg=stage_color(score))
        self.progress["value"] = score

        bar_color = GREEN if score < 35 else (YELLOW if score < 65 else RED)
        self._bar_style.configure("Wound.Horizontal.TProgressbar", background=bar_color)

        # Model labels
        cnn = result["cnn"]
        ml = result["ml"]
        self.cnn_lbl.config(text=f"CNN  {cnn['class']} ({cnn['probs'][cnn['class']]}%)")
        self.ml_lbl.config(text=f"ML   {ml['class']} ({ml['probs'][ml['class']]}%)")

        # Probability bars
        fused = result["fused_probs"]
        for lbl in LABELS:
            bar, pct = self._prob_bars[lbl]
            v = fused.get(lbl, 0)
            bar["value"] = v
            pct.config(text=f"{v:.0f}%")

        # Findings
        for w in self.findings_frame.winfo_children():
            w.destroy()

        for finding in result["findings"]:
            row = tk.Frame(self.findings_frame, bg=BG1)
            row.pack(fill="x", pady=1)
            tk.Label(row, text="›", bg=BG1, fg=ACCENT,
                     font=("Courier", 11)).pack(side="left")
            tk.Label(row, text=finding, bg=BG1, fg=TEXT0,
                     font=("Courier", 10), anchor="w",
                     wraplength=340, justify="left").pack(side="left", padx=(4, 0))

        if result.get("risk_flags"):
            tk.Frame(self.findings_frame, bg=BORDER, height=1).pack(
                fill="x", pady=(6, 4))
            for flag in result["risk_flags"]:
                row = tk.Frame(self.findings_frame, bg=BG1)
                row.pack(fill="x", pady=1)
                tk.Label(row, text="⚠", bg=BG1, fg=RED,
                         font=("Courier", 11)).pack(side="left")
                tk.Label(row, text=flag, bg=BG1, fg=RED,
                         font=("Courier", 10), anchor="w").pack(side="left", padx=4)

        self._set_status("Analysis complete", GREEN)
        self.btn_export.config(state="normal")

    def _show_error(self, msg):
        self._analyzing = False
        self.btn_select.config(state="normal")
        self.diag_lbl.config(text="ERROR", fg=RED)
        self.stage_lbl.config(text=msg[:60])
        self._set_status("Error", RED)
        messagebox.showerror("Analysis Error", msg)

    # ─── PDF EXPORT ───
    def _export_pdf(self):
        if not self._last_result:
            return
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas as rl_canvas
            from reportlab.lib import colors
        except ImportError:
            messagebox.showerror("Missing library",
                                 "Install reportlab: pip install reportlab --break-system-packages")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")]
        )
        if not path:
            return

        r = self._last_result
        c = rl_canvas.Canvas(path, pagesize=A4)
        w, h = A4

        c.setFillColorRGB(0.08, 0.09, 0.12)
        c.rect(0, 0, w, h, fill=1, stroke=0)

        c.setFillColorRGB(0.31, 0.56, 0.97)
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, h - 60, "Wound AI Analyzer — Medical Report")

        c.setFillColorRGB(0.54, 0.58, 0.67)
        c.setFont("Helvetica", 10)
        import datetime
        c.drawString(50, h - 80, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

        c.setStrokeColorRGB(0.15, 0.16, 0.22)
        c.line(50, h - 92, w - 50, h - 92)

        y = h - 120

        def row(label, value, color=None):
            nonlocal y
            c.setFillColorRGB(0.54, 0.58, 0.67)
            c.setFont("Helvetica", 10)
            c.drawString(50, y, label)
            if color:
                c.setFillColor(color)
            else:
                c.setFillColorRGB(0.91, 0.92, 0.94)
            c.setFont("Helvetica-Bold", 11)
            c.drawString(180, y, str(value))
            y -= 22

        rc = {"healthy": colors.HexColor("#3cba6f"),
              "inflamed": colors.HexColor("#e8a628"),
              "infected": colors.HexColor("#e8494a")}

        row("Diagnosis", r["final"].upper(), rc.get(r["final"]))
        row("Stage", r["stage"])
        row("Severity Score", f"{r['score']}%")
        row("Confidence", f"{r['confidence']}%")
        row("CNN", f"{r['cnn']['class']} ({r['cnn']['probs'][r['cnn']['class']]}%)")
        row("ML", f"{r['ml']['class']} ({r['ml']['probs'][r['ml']['class']]}%)")

        y -= 10
        c.setStrokeColorRGB(0.15, 0.16, 0.22)
        c.line(50, y, w - 50, y)
        y -= 20

        c.setFillColorRGB(0.54, 0.58, 0.67)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Findings:")
        y -= 18

        for finding in r["findings"]:
            c.setFillColorRGB(0.91, 0.92, 0.94)
            c.setFont("Helvetica", 10)
            c.drawString(60, y, f"› {finding}")
            y -= 16

        if r.get("risk_flags"):
            y -= 8
            c.setFillColorRGB(0.91, 0.29, 0.29)
            c.setFont("Helvetica-Bold", 11)
            c.drawString(50, y, "Risk Flags:")
            y -= 18
            for flag in r["risk_flags"]:
                c.setFillColorRGB(0.91, 0.29, 0.29)
                c.setFont("Helvetica", 10)
                c.drawString(60, y, f"⚠ {flag}")
                y -= 16

        y -= 20
        c.setFillColorRGB(0.33, 0.36, 0.44)
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(50, y, "This report is generated by an AI system and is not a substitute for clinical evaluation.")

        c.save()
        messagebox.showinfo("Exported", f"Report saved to:\n{path}")

    # ─── STATUS ───
    def _set_status(self, text, color=TEXT1):
        self.status_lbl.config(text=text, fg=color)
        self.status_dot.config(fg=color)

    # ─── MODEL LOADING ───
    def _start_model_load(self):
        threading.Thread(target=self._bg_load_models, daemon=True).start()

    def _bg_load_models(self):
        try:
            load_models()
            self.after(0, self._set_status, "Ready", GREEN)
        except Exception as e:
            self.after(0, self._set_status, f"Model load failed: {e}", RED)


# ─────────────────────────── RUN ───────────────────────────

if __name__ == "__main__":
    app = WoundApp()
    app.mainloop()