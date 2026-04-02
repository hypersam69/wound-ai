"""
admin.py — Password-protected admin dashboard.
Mount this alongside app.py using Flask blueprints.

Access at: http://localhost:5000/admin
Default password: set ADMIN_PASSWORD in environment
"""

import os
import functools
from flask import (Blueprint, render_template_string,
                   request, redirect, url_for, session, jsonify)

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "woundai2024")

# ── AUTH ─────────────────────────────────────────────────────

def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin.login"))
        return f(*args, **kwargs)
    return decorated


@admin_bp.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return redirect(url_for("admin.dashboard"))
        error = "Wrong password"

    return render_template_string(LOGIN_HTML, error=error)


@admin_bp.route("/logout")
def logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin.login"))


# ── DASHBOARD ────────────────────────────────────────────────

@admin_bp.route("/")
@login_required
def dashboard():
    from feedback_db import get_stats, get_pending_review, init_db
    init_db()
    stats   = get_stats()
    pending = get_pending_review()
    return render_template_string(
        DASHBOARD_HTML, stats=stats, pending=pending
    )


@admin_bp.route("/approve/<int:record_id>", methods=["POST"])
@login_required
def approve(record_id):
    from feedback_db import admin_approve
    notes = request.form.get("notes", "")
    admin_approve(record_id, notes)
    return redirect(url_for("admin.dashboard"))


@admin_bp.route("/reject/<int:record_id>", methods=["POST"])
@login_required
def reject(record_id):
    from feedback_db import admin_reject
    notes = request.form.get("notes", "")
    admin_reject(record_id, notes)
    return redirect(url_for("admin.dashboard"))


@admin_bp.route("/approve-all", methods=["POST"])
@login_required
def approve_all():
    from feedback_db import get_pending_review, admin_approve
    pending = get_pending_review()
    for row in pending:
        if row["weight"] >= 0.4:   # only auto-approve high-weight items
            admin_approve(row["id"])
    return redirect(url_for("admin.dashboard"))


@admin_bp.route("/retrain", methods=["POST"])
@login_required
def manual_retrain():
    from feedback_db import retrain_rf
    result = retrain_rf()
    return jsonify(result or {"error": "Retrain failed"})


@admin_bp.route("/stats-json")
@login_required
def stats_json():
    from feedback_db import get_stats
    return jsonify(get_stats())


# ── HTML TEMPLATES ───────────────────────────────────────────

LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Admin Login — Wound AI</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0a0c10;color:#e2e6f0;font-family:'IBM Plex Mono',monospace;display:flex;align-items:center;justify-content:center;min-height:100vh}
  .card{background:#0f1218;border:1px solid #252b3a;padding:40px;width:340px}
  .card::before{content:'';display:block;height:2px;background:#4f8ef7;margin-bottom:32px}
  h2{font-size:14px;letter-spacing:0.15em;text-transform:uppercase;color:#8892a8;margin-bottom:24px}
  input{width:100%;padding:12px;background:#151820;border:1px solid #252b3a;color:#e2e6f0;font-family:inherit;font-size:13px;margin-bottom:16px}
  input:focus{outline:none;border-color:#4f8ef7}
  button{width:100%;padding:12px;background:#4f8ef7;color:#fff;border:none;font-family:inherit;font-size:12px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;cursor:pointer}
  button:hover{background:#2563eb}
  .error{color:#e84a4a;font-size:12px;margin-bottom:12px}
</style>
</head>
<body>
<div class="card">
  <h2>// Admin Access</h2>
  {% if error %}<div class="error">✕ {{ error }}</div>{% endif %}
  <form method="POST">
    <input type="password" name="password" placeholder="Enter admin password" autofocus>
    <button type="submit">Login</button>
  </form>
</div>
</body>
</html>
"""

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Admin Dashboard — Wound AI</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400&display=swap" rel="stylesheet">
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0a0c10;color:#e2e6f0;font-family:'IBM Plex Mono',monospace;min-height:100vh}
  body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(79,142,247,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(79,142,247,0.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0}
  .wrap{position:relative;z-index:1;max-width:1100px;margin:0 auto;padding:0 24px 60px}
  header{display:flex;align-items:center;justify-content:space-between;padding:24px 0 20px;border-bottom:1px solid #252b3a;margin-bottom:36px}
  .logo{font-size:14px;font-weight:600;letter-spacing:0.08em}
  .logo span{color:#4f8ef7}
  .header-actions{display:flex;gap:10px;align-items:center}
  .tag{font-size:9px;color:#4a5268;letter-spacing:0.15em;text-transform:uppercase;border:1px solid #252b3a;padding:3px 8px}
  .btn{padding:8px 16px;border:none;font-family:inherit;font-size:11px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;cursor:pointer}
  .btn-primary{background:#4f8ef7;color:#fff}
  .btn-primary:hover{background:#2563eb}
  .btn-danger{background:rgba(232,74,74,0.15);color:#e84a4a;border:1px solid rgba(232,74,74,0.3)}
  .btn-sm{padding:5px 10px;font-size:10px}
  .btn-approve{background:rgba(52,201,122,0.15);color:#34c97a;border:1px solid rgba(52,201,122,0.3)}
  .btn-approve:hover{background:rgba(52,201,122,0.25)}
  .btn-reject{background:rgba(232,74,74,0.1);color:#e84a4a;border:1px solid rgba(232,74,74,0.2)}
  .btn-reject:hover{background:rgba(232,74,74,0.2)}
  .btn-logout{background:transparent;color:#4a5268;border:1px solid #252b3a;font-size:10px;padding:6px 12px}
  .stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:36px}
  .stat-card{background:#0f1218;border:1px solid #252b3a;padding:20px 16px}
  .stat-card::before{content:'';display:block;height:1px;background:linear-gradient(90deg,transparent,#4f8ef7,transparent);opacity:0.4;margin-bottom:16px}
  .stat-label{font-size:9px;letter-spacing:0.15em;text-transform:uppercase;color:#4a5268;margin-bottom:8px}
  .stat-value{font-size:24px;font-weight:600}
  .stat-value.green{color:#34c97a}
  .stat-value.yellow{color:#f0a528}
  .stat-value.red{color:#e84a4a}
  .stat-value.blue{color:#4f8ef7}
  .section-title{font-size:9px;letter-spacing:0.2em;text-transform:uppercase;color:#4a5268;margin-bottom:16px}
  .card{background:#0f1218;border:1px solid #252b3a;padding:24px;margin-bottom:20px}
  .pending-table{width:100%;border-collapse:collapse}
  .pending-table th{font-size:9px;letter-spacing:0.15em;text-transform:uppercase;color:#4a5268;padding:0 12px 12px;text-align:left;border-bottom:1px solid #252b3a}
  .pending-table td{padding:12px;border-bottom:1px solid #1c2030;font-size:12px;vertical-align:middle}
  .pending-table tr:last-child td{border-bottom:none}
  .weight-bar{height:4px;background:#1c2030;margin-top:4px}
  .weight-fill{height:100%;background:#4f8ef7;transition:width 0.3s}
  .weight-fill.high{background:#e84a4a}
  .weight-fill.med{background:#f0a528}
  .weight-fill.low{background:#34c97a}
  .badge{display:inline-block;font-size:9px;padding:2px 6px;letter-spacing:0.05em}
  .badge-healthy{background:rgba(52,201,122,0.12);color:#34c97a}
  .badge-inflamed{background:rgba(240,165,40,0.12);color:#f0a528}
  .badge-infected{background:rgba(232,74,74,0.12);color:#e84a4a}
  .badge-flagged{background:rgba(79,142,247,0.12);color:#4f8ef7}
  .action-form{display:inline}
  .empty-state{text-align:center;padding:48px;color:#4a5268;font-size:13px}
  .retrain-card{background:#0f1218;border:1px solid #252b3a;padding:24px;display:flex;align-items:center;justify-content:space-between;margin-bottom:20px}
  .retrain-info p{font-family:'IBM Plex Sans',sans-serif;font-size:13px;color:#8892a8;margin-top:4px}
  #retrain-result{font-size:12px;color:#34c97a;margin-top:8px}
  .by-class{display:flex;gap:16px;margin-top:12px}
  .class-pill{padding:6px 14px;background:#151820;border:1px solid #252b3a;font-size:11px}
  .class-pill span{color:#4a5268;font-size:10px;display:block}
  .img-thumb{width:60px;height:60px;object-fit:cover;filter:saturate(0.7)}
  @media(max-width:700px){.stats-grid{grid-template-columns:1fr 1fr}.pending-table .hide-mobile{display:none}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="logo">WOUND<span>AI</span> <span style="color:#4a5268;font-weight:300">/ admin</span></div>
    <div class="header-actions">
      <span class="tag">Strategy 3 · Weighted Learning</span>
      <a href="{{ url_for('admin.logout') }}"><button class="btn btn-logout">Logout</button></a>
    </div>
  </header>

  <!-- STATS -->
  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-label">Total Feedback</div>
      <div class="stat-value blue">{{ stats.total }}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">AI Accuracy</div>
      <div class="stat-value {% if stats.ai_accuracy > 75 %}green{% elif stats.ai_accuracy > 55 %}yellow{% else %}red{% endif %}">
        {{ stats.ai_accuracy }}%
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Pending Review</div>
      <div class="stat-value yellow">{{ pending|length }}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Approved</div>
      <div class="stat-value green">{{ stats.approved }}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Rejected</div>
      <div class="stat-value red">{{ stats.rejected }}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Auto-Flagged</div>
      <div class="stat-value blue">{{ stats.auto_flagged }}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Retrains Done</div>
      <div class="stat-value">{{ stats.retrains_done }}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Used in Train</div>
      <div class="stat-value green">{{ stats.trained_on }}</div>
    </div>
  </div>

  <!-- BY CLASS -->
  {% if stats.by_class %}
  <div class="card" style="margin-bottom:20px">
    <div class="section-title">// feedback by class</div>
    <div class="by-class">
      {% for label, data in stats.by_class.items() %}
      <div class="class-pill badge-{{ label }}">
        {{ label | upper }}
        <span>{{ data.count }} samples · avg weight {{ data.avg_weight }}</span>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endif %}

  <!-- RETRAIN CONTROLS -->
  <div class="retrain-card">
    <div class="retrain-info">
      <div class="section-title">// model retraining</div>
      <p>RF retrains automatically at {{ stats.approved }} approved samples. CNN requires manual run.</p>
      <div id="retrain-result"></div>
    </div>
    <div style="display:flex;gap:10px">
      <button class="btn btn-primary" onclick="manualRetrain()">
        ↻ Retrain RF Now
      </button>
      {% if pending %}
      <form action="{{ url_for('admin.approve_all') }}" method="POST" style="display:inline">
        <button class="btn" style="background:rgba(79,142,247,0.15);color:#4f8ef7;border:1px solid rgba(79,142,247,0.3)" type="submit">
          ✓ Approve High-Weight
        </button>
      </form>
      {% endif %}
    </div>
  </div>

  <!-- PENDING REVIEW TABLE -->
  <div class="section-title">// pending review ({{ pending|length }})</div>

  {% if pending %}
  <div class="card" style="padding:0;overflow:hidden">
    <table class="pending-table">
      <thead>
        <tr>
          <th>Image</th>
          <th>AI Said</th>
          <th>User Says</th>
          <th class="hide-mobile">Confidence</th>
          <th>Weight</th>
          <th class="hide-mobile">Flagged</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        {% for row in pending %}
        <tr>
          <td>
            {% if row.image_path %}
            <img src="/feedback-image/{{ row.id }}" class="img-thumb"
                 onerror="this.style.display='none'">
            {% else %}—{% endif %}
          </td>
          <td>
            <span class="badge badge-{{ row.ai_prediction }}">
              {{ row.ai_prediction | upper }}
            </span>
          </td>
          <td>
            {% if row.user_label %}
            <span class="badge badge-{{ row.user_label }}">
              {{ row.user_label | upper }}
            </span>
            {% if row.user_label != row.ai_prediction %}
            <span style="color:#e84a4a;font-size:10px;display:block;margin-top:2px">corrected</span>
            {% else %}
            <span style="color:#34c97a;font-size:10px;display:block;margin-top:2px">confirmed</span>
            {% endif %}
            {% else %}—{% endif %}
          </td>
          <td class="hide-mobile">
            <span style="color:{% if row.ai_confidence < 60 %}#e84a4a{% elif row.ai_confidence < 75 %}#f0a528{% else %}#34c97a{% endif %}">
              {{ "%.0f"|format(row.ai_confidence) }}%
            </span>
          </td>
          <td>
            <span style="font-size:11px">{{ row.weight }}</span>
            <div class="weight-bar">
              <div class="weight-fill {% if row.weight >= 0.7 %}high{% elif row.weight >= 0.4 %}med{% else %}low{% endif %}"
                   style="width:{{ (row.weight * 100)|int }}%"></div>
            </div>
          </td>
          <td class="hide-mobile">
            {% if row.auto_flagged %}
            <span class="badge badge-flagged">auto</span>
            {% else %}—{% endif %}
          </td>
          <td>
            <form action="{{ url_for('admin.approve', record_id=row.id) }}" method="POST" class="action-form">
              <button class="btn btn-sm btn-approve" type="submit">✓ Approve</button>
            </form>
            <form action="{{ url_for('admin.reject', record_id=row.id) }}" method="POST" class="action-form" style="margin-left:4px">
              <button class="btn btn-sm btn-reject" type="submit">✕ Reject</button>
            </form>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  {% else %}
  <div class="card">
    <div class="empty-state">
      No pending feedback to review.<br>
      <span style="font-size:11px;color:#252b3a;margin-top:8px;display:block">
        Feedback appears here when users submit corrections.
      </span>
    </div>
  </div>
  {% endif %}

</div>
<script>
async function manualRetrain() {
  const btn = event.target;
  btn.disabled = true;
  btn.textContent = '↻ Retraining…';
  document.getElementById('retrain-result').textContent = '';

  try {
    const r = await fetch("{{ url_for('admin.manual_retrain') }}", {method:'POST'});
    const data = await r.json();

    if (data.error) {
      document.getElementById('retrain-result').textContent = '✕ ' + data.error;
      document.getElementById('retrain-result').style.color = '#e84a4a';
    } else if (data.improved) {
      document.getElementById('retrain-result').textContent =
        `✓ Model improved! F1: ${data.old_f1} → ${data.new_f1}`;
    } else {
      document.getElementById('retrain-result').textContent =
        `⚠ Model not improved (F1: ${data.old_f1} → ${data.new_f1}). Old model kept.`;
      document.getElementById('retrain-result').style.color = '#f0a528';
    }
  } catch(e) {
    document.getElementById('retrain-result').textContent = '✕ Request failed';
    document.getElementById('retrain-result').style.color = '#e84a4a';
  }

  btn.disabled = false;
  btn.textContent = '↻ Retrain RF Now';
}
</script>
</body>
</html>
"""