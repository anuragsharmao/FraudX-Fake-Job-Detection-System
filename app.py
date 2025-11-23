from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

model_lr = None
model_rf = None
tfidf = None
model_ready = False

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    words = text.split()
    return ' '.join([w for w in words if w and w not in stop_words])


def rule_based_score(text):
    """Return (flag, score) where flag is True if high-confidence scam detected.
    score in 0..1 indicating suspiciousness from rules.
    """
    if not text:
        return False, 0.0
    t = text.lower()
    score = 0.0
    # high risk keywords
    high_kw = [
    'send money', 'wire transfer', 'western union', 'money transfer', 'pay money',
    'pay for equipment', 'pay for training', 'apply now and pay', 'registration fee',
    'processing fee', 'transfer fee', 'advance payment', 'upfront fee', 'security deposit',
    'membership fee', 'transaction charge', 'application fee', 'training fee',
    'bank account', 'paypal', 'upi', 'crypto payment', 'bitcoin payment',
    'westernunion', 'moneygram', 'cash app', 'provide card details',
    'earn $', 'earn money', 'quick money', 'daily income', 'unlimited income',
    'make money fast', 'get rich', 'high income with no work', 'instant earnings',
    'no interview required', 'work 1 hour and earn',
    'click here', 'visit the link', 'fill this form', 'login to verify',
    'submit your id proof', 'upload aadhar', 'upload pan',
    'urgent hiring', 'instant hire', 'selected without interview',
    'no background check', 'job guarantee', '100% placement guarantee',
    'work from home and earn daily', 'sms sending job', 'form filling job',
    'referral bonus', 'binary plan', 'multi level marketing', 'network marketing',
    'commission-based no base salary', 'recruit others to earn', 'agent recruitment',
    'share otp', 'provide otp', 'atm card details', 'cvv', 'upload scanned documents',
    'provide passport copy', 'kyc verification fee'
]

    # medium risk keywords
    med_kw = [
    'free', 'guaranteed', 'no experience', 'limited positions', 'act now',
    'hiring fast', 'immediate openings', 'quick selection', 'flexible work',
    'no skills required', 'limited seats', 'great opportunity', 'weekly payout',
    'no degree required', 'easy work', 'hurry up', 'start immediately',
    'home-based job', 'remote opportunity', 'part-time flexible', 'simple online work',
    'bonus payout', 'commission only', 'work anytime', 'no interview',
    'bulk hiring', 'apply immediately', 'walk-in with resume'
]


    for kw in high_kw:
        if kw in t:
            score += 0.4
    for kw in med_kw:
        if kw in t:
            score += 0.15

    # suspicious patterns
    if t.count('!') >= 2:
        score += 0.1
    if sum(1 for c in text if c.isupper()) > max(20, len(text) * 0.1):
        score += 0.1

    # clamp
    score = min(score, 1.0)
    flag = score >= 0.6
    return flag, score


def initialize_models():
    """Load data, train TF-IDF, Logistic Regression and Random Forest models."""
    global model_lr, model_rf, tfidf, model_ready
    try:
        print('Loading dataset...')
        df = pd.read_csv('fake_job_postings.csv', encoding='utf-8', low_memory=False)
        print('Rows:', len(df))

        # basic cleaning
        for col in ['job_id', 'Unnamed: 0']:
            if col in df.columns:
                df = df.drop(col, axis=1)

        text_cols = ['title', 'location', 'department', 'company_profile',
                     'description', 'requirements', 'benefits', 'industry', 'function']
        for c in text_cols:
            if c not in df.columns:
                df[c] = ''
        df['all_text'] = df[text_cols].fillna('').agg(' '.join, axis=1).apply(clean_text)

        # Prepare X and y
        X_text = df['all_text']
        y = df['fraudulent'].astype(int)

        # TF-IDF
        tfidf = TfidfVectorizer(max_features=2000)
        X_t = tfidf.fit_transform(X_text)

        # Train Logistic Regression with class weights
        model_lr = LogisticRegression(class_weight='balanced', max_iter=500)
        model_lr.fit(X_t, y)

        # Train Random Forest with class weights
        model_rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        model_rf.fit(X_t, y)

        model_ready = True
        print('Models trained.')
        return True
    except Exception as e:
        print('Model init failed:', e)
        model_ready = False
        return False


def ensemble_predict(text):
    """Return (label_str, prob_percent, details)
    Ensemble of rule-based + LR + RF. Higher rule score forces high probability.
    """
    global model_lr, model_rf, tfidf
    if not model_ready:
        raise RuntimeError('Model not ready')

    flag, rule_score = rule_based_score(text)

    cleaned = clean_text(text)
    x = tfidf.transform([cleaned])

    # get probabilities (ensure numeric stability)
    prob_lr = 0.0
    prob_rf = 0.0
    try:
        pr_lr = model_lr.predict_proba(x)[0]
        pr_lr = np.array(pr_lr, dtype=float)
        pr_lr = pr_lr / pr_lr.sum() if pr_lr.sum() > 0 else pr_lr
        prob_lr = float(pr_lr[1])
    except Exception:
        prob_lr = 0.0
    try:
        pr_rf = model_rf.predict_proba(x)[0]
        pr_rf = np.array(pr_rf, dtype=float)
        pr_rf = pr_rf / pr_rf.sum() if pr_rf.sum() > 0 else pr_rf
        prob_rf = float(pr_rf[1])
    except Exception:
        prob_rf = 0.0

    # combine: weighted average (give RF slight more weight)
    combined = (0.45 * prob_lr) + (0.55 * prob_rf)

    # if rule-based strong, amplify
    if flag:
        combined = max(combined, min(0.95, combined + rule_score * 0.5))

    # final percent
    pct = round(min(max(combined, 0.0), 1.0) * 100.0, 1)

    if pct >= 70:
        label = 'High Risk Fake Job'
    elif pct >= 40:
        label = 'Potential Scam'
    elif pct >= 15:
        label = 'Suspicious'
    else:
        label = 'Likely Legitimate'

    details = {
        'prob_lr': round(prob_lr * 100.0, 1),
        'prob_rf': round(prob_rf * 100.0, 1),
        'rule_score': round(rule_score * 100.0, 1)
    }

    return label, pct, details


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_ready': model_ready})


@app.route('/predict', methods=['POST'])
def predict():
    global model_ready
    if not model_ready:
        ok = initialize_models()
        if not ok:
            return jsonify({'error': 'Model initialization failed'}), 500

    data = request.get_json(force=True)
    if not data or 'description' not in data:
        return jsonify({'error': 'Missing description field'}), 400

    text = data.get('description', '')
    try:
        label, pct, details = ensemble_predict(text)
        return jsonify({'prediction': label, 'probability': pct, 'details': details, 'model_ready': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print('Initializing models...')
    initialize_models()
    print('Starting server on http://localhost:5000')
    app.run(host='0.0.0.0', port=5000, debug=True)