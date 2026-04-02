"""
medical_logic.py — Clinically-grounded multi-layer medical reasoning.
Mimics real diagnostic decision-making with risk-based language.
"""


def analyze_medical(features):
    """
    Parameters
    ----------
    features : list of 12 floats
        [red_area, yellow_area, dark_area, pink_area,
         red_intensity, red_std, edge_ratio, texture, entropy,
         saturation, ry_ratio, pus_necrosis_combined]

    Returns
    -------
    score      : float  0-100
    stage      : str
    findings   : list[str]   clinical observations
    risk_flags : list[str]   high-priority alerts
    """

    (red_area, yellow_area, dark_area, pink_area,
     red_intensity, red_std, edge_ratio, texture, entropy,
     saturation, ry_ratio, pus_necrosis_combined) = features

    findings   = []
    risk_flags = []

    # ── LAYER 1: PRIMARY INFECTION INDICATORS ────────────────
    # These are the strongest clinical signals

    if yellow_area > 0.15:
        findings.append("Substantial yellowish exudate — high risk of purulent infection")
        risk_flags.append("Significant pus/exudate detected")
    elif yellow_area > 0.08:
        findings.append("Yellowish discoloration present — possible early exudate formation")
    elif yellow_area > 0.03:
        findings.append("Trace yellowish areas — monitor for pus development")

    if dark_area > 0.15:
        findings.append("Extensive dark tissue regions — possible necrosis or eschar")
        risk_flags.append("Possible necrotic tissue")
    elif dark_area > 0.08:
        findings.append("Dark tissue areas present — reduced tissue perfusion suspected")
    elif dark_area > 0.03:
        findings.append("Mild dark discoloration — tissue health monitoring advised")

    # ── LAYER 2: INFLAMMATION INDICATORS ────────────────────
    if red_area > 0.30:
        findings.append("Extensive redness — significant inflammatory response")
        risk_flags.append("High redness — active inflammation")
    elif red_area > 0.15:
        findings.append("Moderate redness — active inflammatory response present")
    elif red_area > 0.06:
        findings.append("Mild redness — early or resolving inflammation")

    if red_std > 0.28:
        findings.append("Non-uniform redness distribution — heterogeneous tissue response")

    # ── LAYER 3: STRUCTURAL INDICATORS ──────────────────────
    if edge_ratio > 0.38:
        findings.append("Highly irregular wound margins — complex wound geometry")
    elif edge_ratio > 0.22:
        findings.append("Moderately irregular wound edges — tissue disruption present")

    if texture > 0.75:
        findings.append("High surface texture variance — disrupted wound bed")
    elif texture > 0.45:
        findings.append("Moderate tissue irregularity observed")

    # ── LAYER 4: HEALING INDICATORS ─────────────────────────
    if pink_area > 0.20 and red_area < 0.08 and yellow_area < 0.04:
        findings.append("Pink granulation tissue visible — possible healing response")

    if not findings:
        findings.append("No significant pathological visual indicators detected")

    # ── SEVERITY SCORE — CALIBRATED SUB-COMPONENTS ──────────
    # Each component has a capped maximum contribution

    # Pus (0–38): strongest infection predictor
    pus_score = min(yellow_area / 0.20, 1.0) * 38

    # Necrosis (0–27)
    necrosis_score = min(dark_area / 0.18, 1.0) * 27

    # Redness (0–18)
    red_score = min(red_area / 0.28, 1.0) * 18

    # Surface irregularity (0–10)
    surface_score = min(
        (texture / 1.0) * 0.55 + (edge_ratio / 0.5) * 0.45, 1.0
    ) * 10

    # Red variability (0–7)
    variability_score = min(red_std / 0.30, 1.0) * 7

    score = pus_score + necrosis_score + red_score + surface_score + variability_score

    # ── INTERACTION BOOSTS (clinically motivated) ────────────
    # Pus + redness together = strong infection signal
    if red_area > 0.18 and yellow_area > 0.06:
        score = min(score + 12, 100)

    # Necrosis + irregular edges = severe tissue damage
    if dark_area > 0.10 and edge_ratio > 0.28:
        score = min(score + 9, 100)

    # All three primary indicators present
    if red_area > 0.10 and yellow_area > 0.05 and dark_area > 0.05:
        score = min(score + 8, 100)

    # ── HEALING PENALTY ──────────────────────────────────────
    if pink_area > 0.15 and pus_score < 8:
        score = max(score - 10, 0)

    score = round(min(score, 100), 1)

    # ── STAGE ────────────────────────────────────────────────
    if score < 15:
        stage = "Healthy"
    elif score < 35:
        stage = "Mild Inflammation"
    elif score < 65:
        stage = "Moderate Infection"
    else:
        stage = "Severe Infection"

    return score, stage, findings, risk_flags