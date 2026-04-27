"""
Advanced Fake Job Detection System - Web Server
================================================
Flask server that loads the pre-trained model and serves predictions.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import warnings
import pickle
import os
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# ─── global state ────────────────────────────────────────────────────────────
model_components = None
model_ready = False

# ─── text helpers (copied from training for consistency) ─────────────────────

def clean_text(text: str, stop_words) -> str:
    if not text:
        return ""
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    return " ".join(w for w in text.split() if w and w not in stop_words)


def rule_score(text: str, high_risk_kw, medium_risk_kw):
    """Returns (flag, score 0-1, matched_keywords)."""
    if not text:
        return False, 0.0, []
    t = text.lower()
    score = 0.0
    matched = []
    for kw in high_risk_kw:
        if kw in t:
            score += 0.4
            matched.append(kw)
    for kw in medium_risk_kw:
        if kw in t:
            score += 0.15
            matched.append(kw)
    if t.count("!") >= 2:
        score += 0.1
    if sum(1 for c in text if c.isupper()) > max(20, len(text) * 0.1):
        score += 0.1
    score = min(score, 1.0)
    return score >= 0.6, score, matched


def build_features(text_raw: str, row_dict: dict = None, high_risk_kw=None, medium_risk_kw=None) -> np.ndarray:
    """
    Returns a 1-D numpy array of hand-crafted features.
    """
    if high_risk_kw is None:
        high_risk_kw = []
    if medium_risk_kw is None:
        medium_risk_kw = []
        
    rd = row_dict or {}
    t = str(text_raw or "")

    # structural
    salary = str(rd.get("salary_range", ""))
    has_salary = float(bool(salary and salary != "nan"))
    salary_wide = 0.0
    if has_salary:
        nums = re.findall(r"\d+", salary.replace(",", ""))
        if len(nums) >= 2:
            lo, hi = int(nums[0]), int(nums[-1])
            salary_wide = float(abs(hi - lo) > 100_000)

    telecommute = float(str(rd.get("telecommuting", "0")) in {"1", "True", "true"})
    has_logo = float(str(rd.get("has_company_logo", "0")) in {"1", "True", "true"})
    has_q = float(str(rd.get("has_questions", "0")) in {"1", "True", "true"})

    emp_map = {"full-time": 1, "part-time": 2, "contract": 3, "temporary": 3,
               "other": 0, "": 0}
    emp = emp_map.get(str(rd.get("employment_type", "")).lower(), 0)

    exp_map = {"not applicable": 0, "internship": 1, "entry level": 1,
               "associate": 2, "mid-senior level": 3, "director": 4,
               "executive": 5, "": 0}
    exp = exp_map.get(str(rd.get("required_experience", "")).lower(), 0)

    edu_map = {"unspecified": 0, "high school or equivalent": 1,
               "some college coursework completed": 2, "associate degree": 3,
               "bachelor's degree": 4, "certification": 3,
               "master's degree": 5, "doctorate": 5, "professional": 5, "": 0}
    edu = edu_map.get(str(rd.get("required_education", "")).lower(), 0)

    # textual
    title = str(rd.get("title", ""))
    title_caps = (sum(1 for c in title if c.isupper()) / max(len(title), 1))
    desc_excl = float(t.count("!"))
    n_high = sum(1 for kw in high_risk_kw if kw in t.lower())
    n_med = sum(1 for kw in medium_risk_kw if kw in t.lower())
    tlen = float(np.log1p(len(t)))
    words = t.split()
    avg_wlen = float(np.mean([len(w) for w in words])) if words else 0.0
    uniq_ratio = float(len(set(words)) / max(len(words), 1))
    digit_ratio = float(sum(c.isdigit() for c in t) / max(len(t), 1))

    return np.array([has_salary, salary_wide, telecommute, has_logo, has_q,
                     emp, exp, edu, title_caps, desc_excl, n_high, n_med,
                     tlen, avg_wlen, uniq_ratio, digit_ratio], dtype=float)


def load_model(model_path="models/fake_job_model.pkl"):
    """Load the pre-trained model from disk."""
    global model_components, model_ready
    
    try:
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            print("Please run model.py first to train and save the model.")
            return False
        
        with open(model_path, 'rb') as f:
            model_components = pickle.load(f)
        
        model_ready = True
        print("✅ Model loaded successfully!")
        print(f"   Threshold: {model_components['threshold']:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model_ready = False
        return False


def predict_text(raw_text: str, row_dict: dict = None):
    """Make prediction using loaded model."""
    global model_components, model_ready
    
    if not model_ready:
        raise RuntimeError("Model not loaded")
    
    # Extract components
    tfidf = model_components['tfidf']
    lr, rf, gb = model_components['base_models']
    meta_clf = model_components['meta_clf']
    threshold = model_components['threshold']
    high_risk_kw = model_components['high_risk_kw']
    medium_risk_kw = model_components['medium_risk_kw']
    stop_words = model_components['stop_words']
    
    # Rule-based scoring
    flag, rscore, matched_kw = rule_score(raw_text, high_risk_kw, medium_risk_kw)
    row_dict = row_dict or {}
    
    # Feature extraction
    cleaned = clean_text(raw_text, stop_words)
    X_tfidf = tfidf.transform([cleaned])
    X_hand = csr_matrix(build_features(raw_text, row_dict, high_risk_kw, medium_risk_kw).reshape(1, -1))
    X_full = hstack([X_tfidf, X_hand])
    
    # Base model predictions
    p_lr = float(lr.predict_proba(X_full)[0, 1])
    p_rf = float(rf.predict_proba(X_full)[0, 1])
    Xd = X_full.toarray()
    p_gb = float(gb.predict_proba(Xd)[0, 1])
    
    # Meta-learner prediction
    meta_input = np.array([[p_lr, p_rf, p_gb]])
    meta_prob = float(meta_clf.predict_proba(meta_input)[0, 1])
    
    # Rule amplification
    if flag:
        meta_prob = max(meta_prob, min(0.97, meta_prob + rscore * 0.35))
    
    meta_prob = float(np.clip(meta_prob, 0.0, 1.0))
    pct = round(meta_prob * 100.0, 1)
    decision = meta_prob >= threshold
    
    # Label assignment
    if pct >= 70:
        label = "High Risk Fake Job"
    elif pct >= 40:
        label = "Potential Scam"
    elif pct >= 20:
        label = "Suspicious"
    else:
        label = "Likely Legitimate"
    
    # Explanations
    explanations = []
    if matched_kw:
        explanations.append(f"Matched keywords: {', '.join(matched_kw[:5])}")
    if p_gb > 0.6:
        explanations.append("Gradient Boosting: high suspicion pattern detected")
    if row_dict.get("telecommuting") in (1, "1", True, "True"):
        explanations.append("Remote-only role (elevated risk)")
    if not row_dict.get("has_company_logo"):
        explanations.append("No company logo")
    
    return {
        "prediction": label,
        "probability": pct,
        "decision_threshold": round(threshold * 100, 1),
        "flagged": decision,
        "details": {
            "meta_prob": pct,
            "prob_lr": round(p_lr * 100, 1),
            "prob_rf": round(p_rf * 100, 1),
            "prob_gb": round(p_gb * 100, 1),
            "rule_score": round(rscore * 100, 1),
        },
        "explanations": explanations,
        "model_ready": True,
    }


# ─── routes ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok", 
        "model_ready": model_ready,
        "threshold": round(model_components['threshold'] * 100, 1) if model_ready else None
    })


@app.route("/predict", methods=["POST"])
def predict():
    global model_ready
    
    if not model_ready:
        return jsonify({"error": "Model not loaded. Please ensure model file exists."}), 500

    data = request.get_json(force=True)
    if not data or "description" not in data:
        return jsonify({"error": "Missing 'description' field"}), 400

    text = data.pop("description", "")
    try:
        result = predict_text(text, row_dict=data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reload", methods=["POST"])
def reload_model():
    """Endpoint to reload the model without restarting the server."""
    global model_ready
    success = load_model()
    if success:
        return jsonify({"status": "success", "message": "Model reloaded successfully"})
    else:
        return jsonify({"status": "error", "message": "Failed to reload model"}), 500


if __name__ == "__main__":
    print("Starting Fake Job Detection Server...")
    print("Loading pre-trained model...")
    
    # Try to load model, if not found, attempt to train
    if not load_model():
        print("No pre-trained model found. Please run model.py first to train the model.")
        print("Example: python model.py")
        sys.exit(1)
    
    print(f"Starting server on http://localhost:5000")
    print("Available endpoints:")
    print("  GET  /health      - Check server health")
    print("  POST /predict     - Make predictions")
    print("  POST /reload      - Reload the model")
    app.run(host="0.0.0.0", port=5000, debug=False)