"""
Advanced Fake Job Detection System - Web Server (FraudX Engine)
===============================================================
Flask server optimized for Vercel Serverless Functions. Features
real-time generative AI forensic explanations powered by Groq.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import warnings
import pickle
import os
import sys 
from scipy.sparse import hstack, csr_matrix
from groq import Groq

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# ─── GLOBAL STATE & ENV CONFIGURATION ────────────────────────────────────────
model_components = None
model_ready = False

# Read key dynamically from Vercel Environment Variables (Do not hardcode!)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ─── TEXT HELPERS ────────────────────────────────────────────────────────────

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
    """Returns a 1-D numpy array of hand-crafted features."""
    if high_risk_kw is None:
        high_risk_kw = []
    if medium_risk_kw is None:
        medium_risk_kw = []
        
    rd = row_dict or {}
    t = str(text_raw or "")

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

    emp_map = {"full-time": 1, "part-time": 2, "contract": 3, "temporary": 3, "other": 0, "": 0}
    emp = emp_map.get(str(rd.get("employment_type", "")).lower(), 0)

    exp_map = {"not applicable": 0, "internship": 1, "entry level": 1, "associate": 2, "mid-senior level": 3, "director": 4, "executive": 5, "": 0}
    exp = exp_map.get(str(rd.get("required_experience", "")).lower(), 0)

    edu_map = {"unspecified": 0, "high school or equivalent": 1, "some college coursework completed": 2, "associate degree": 3, "bachelor's degree": 4, "certification": 3, "master's degree": 5, "doctorate": 5, "professional": 5, "": 0}
    edu = edu_map.get(str(rd.get("required_education", "")).lower(), 0)

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

    return np.array([float(has_salary), float(salary_wide), float(telecommute), 
                     float(has_logo), float(has_q), float(emp), float(exp), float(edu), 
                     float(title_caps), float(desc_excl), float(n_high), float(n_med),
                     float(tlen), float(avg_wlen), float(uniq_ratio), float(digit_ratio)], dtype=float)


def load_model(model_path="models/fake_job_model.pkl"):
    """Load the pre-trained model from disk."""
    global model_components, model_ready
    try:
        # Vercel paths are relative to root context
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            return False
        
        with open(model_path, 'rb') as f:
            model_components = pickle.load(f)
        
        model_ready = True
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model_ready = False
        return False


# ─── GENAI EXPLANATION PIPELINE ──────────────────────────────────────────────

def get_llm_explanation(prediction_label, probability, details, explanations, raw_text):
    """Calls Groq Cloud API to dynamically explain engine threat data."""
    global groq_client
    
    if not groq_client:
        return "AI Analysis Engine Standby: Set GROQ_API_KEY environment variable to generate narrative summaries."

    system_instruction = "You are FraudX AI, an elite enterprise cyber-forensics and threat intelligence analyst."
    
    prompt = f"""
    Review this metadata and textual sample from an analyzed job posting. Write a highly professional, 
    concise 3-sentence risk summary for an executive security report. Explain exactly *why* the underlying models 
    generated this threat level based on the data points.

    [DIAGNOSTICS METRICS]
    - Consolidated Threat Verdict: {prediction_label}
    - Final Fraud Vector Probability: {probability}%
    - Sub-Model Metrics:
      * Logistic Regression Analysis: {details['prob_lr']}% risk pattern
      * Random Forest Vector: {details['prob_rf']}% anomaly density
      * Gradient Boosting Classifier: {details['prob_gb']}% structural mismatch
    - Heuristic Flags: {', '.join(explanations) if explanations else 'No direct heuristic triggers recorded.'}

    [TEXT BODY SNIPPET]
    \"\"\"{raw_text[:700]}...\"\"\"

    [OUTPUT CONSTRAINTS]
    Focus strictly on objective threat analysis (e.g., semantic formatting inconsistencies, high-risk compensation models, or missing structural corporate tokens). Do not use lists, bullet points, introductory remarks, or greetings. Keep the summary under 90 words.
    """

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=150
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Groq API failure: {e}")
        return "Forensic narrative engine unavailable. Review structural machine learning metrics matrix below."


# ─── INFERENCE LOGIC ─────────────────────────────────────────────────────────

def predict_text(raw_text: str, row_dict: dict = None):
    global model_components, model_ready
    
    if not model_ready:
        # Fallback load attempt inside cold serverless context blocks
        if not load_model():
            raise RuntimeError("Model architecture uninitialized on cloud cluster.")
    
    tfidf = model_components['tfidf']
    lr, rf, gb = model_components['base_models']
    meta_clf = model_components['meta_clf']
    threshold = model_components['threshold']
    high_risk_kw = model_components['high_risk_kw']
    medium_risk_kw = model_components['medium_risk_kw']
    stop_words = model_components['stop_words']
    
    flag, rscore, matched_kw = rule_score(raw_text, high_risk_kw, medium_risk_kw)
    row_dict = row_dict or {}
    
    cleaned = clean_text(raw_text, stop_words)
    X_tfidf = tfidf.transform([cleaned])
    X_hand = csr_matrix(build_features(raw_text, row_dict, high_risk_kw, medium_risk_kw).reshape(1, -1))
    X_full = hstack([X_tfidf, X_hand])
    
    p_lr = float(lr.predict_proba(X_full)[0, 1])
    p_rf = float(rf.predict_proba(X_full)[0, 1])
    Xd = X_full.toarray()
    p_gb = float(gb.predict_proba(Xd)[0, 1])
    
    meta_input = np.array([[p_lr, p_rf, p_gb]])
    meta_prob = float(meta_clf.predict_proba(meta_input)[0, 1])
    
    if flag:
        meta_prob = max(meta_prob, min(0.97, meta_prob + rscore * 0.35))
    
    meta_prob = float(np.clip(meta_prob, 0.0, 1.0))
    pct = round(meta_prob * 100.0, 1)
    decision = bool(meta_prob >= threshold)
    
    if pct >= 70:
        label = "⚠️ High Risk - Potential Scam"
    elif pct >= 40:
        label = "⚠️ Potential Scam"
    elif pct >= 20:
        label = "🔍 Suspicious"
    else:
        label = "✅ Likely Legitimate"
    
    explanations = []
    if matched_kw:
        explanations.append(f"Matched high-risk strings: {', '.join(matched_kw[:3])}")
    if p_gb > 0.6:
        explanations.append("Gradient Boosting pattern mismatch")
    if row_dict.get("telecommuting") in (1, "1", True, "True"):
        explanations.append("Unverified remote operational scope")
    if not row_dict.get("has_company_logo") or str(row_dict.get("has_company_logo")) in ("0", "False", "false"):
        explanations.append("Omission of verified corporate brand identifier")

    metrics_details = {
        "meta_prob": float(pct),
        "prob_lr": float(round(p_lr * 100, 1)),
        "prob_rf": float(round(p_rf * 100, 1)),
        "prob_gb": float(round(p_gb * 100, 1)),
        "rule_score": float(round(rscore * 100, 1)),
    }

    llm_analysis_report = get_llm_explanation(label, pct, metrics_details, explanations, raw_text)
    
    return {
        "prediction": str(label),
        "probability": float(pct),
        "decision_threshold": float(round(threshold * 100, 1)),
        "flagged": bool(decision),
        "details": metrics_details,
        "explanations": list(explanations),
        "llm_analysis": llm_analysis_report,
        "model_ready": bool(True),
    }

# ─── ROUTE ENDPOINTS ──────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    global model_ready
    if not model_ready:
        load_model() # Pre-load model context on verification checks
        
    threshold_val = None
    if model_ready and model_components:
        threshold_val = float(round(model_components['threshold'] * 100, 1))
    
    return jsonify({
        "status": "ok", 
        "model_ready": bool(model_ready),
        "threshold": threshold_val,
        "groq_pipeline_active": bool(groq_client is not None)
    })


@app.route("/predict", methods=["POST"])
def predict():
    global model_ready
    if not model_ready:
        if not load_model():
            return jsonify({"error": "Model architecture uninitialized."}), 500

    try:
        data = request.get_json(force=True)
        if not data or "description" not in data:
            return jsonify({"error": "Missing mandatory field 'description'"}), 400

        text = data.pop("description", "")
        result = predict_text(text, row_dict=data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reload", methods=["POST"])
def reload_model():
    if load_model():
        return jsonify({"status": "success", "message": "Ensemble models loaded successfully."})
    return jsonify({"status": "error", "message": "Failed system re-initialization."}), 500


# ─── COLD-START OPTIMIZATION FOR CLOUD RUNTIME ──────────────────────────────
# Triggering an internal baseline load during Vercel's initial module parsing step
load_model()

# CRITICAL: DO NOT call app.run() inside production frameworks on Vercel.
# The server instance is called explicitly using WSGI handlers behind the scenes.
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)