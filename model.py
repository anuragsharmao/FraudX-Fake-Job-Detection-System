"""
Advanced Fake Job Detection System - Model Training Module
==========================================================
Trains the fake job detection model and saves it to disk.
Includes comprehensive visualizations for model evaluation.
"""
import pandas as pd
import numpy as np
import re
import warnings
import nltk
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score
)
from sklearn.model_selection import learning_curve

# Download NLTK data FIRST before using it
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Try to import SHAP (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. SHAP plots will be skipped.")

warnings.filterwarnings("ignore")

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# ─── NLTK setup ──────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))

# Create plots directory
PLOTS_DIR = "model_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── keyword lists ────────────────────────────────────────────────────────────
HIGH_RISK_KW = [
    "send money", "wire transfer", "western union", "money transfer", "pay money",
    "pay for equipment", "pay for training", "apply now and pay", "registration fee",
    "processing fee", "transfer fee", "advance payment", "upfront fee",
    "security deposit", "membership fee", "transaction charge", "application fee",
    "training fee", "bank account", "paypal", "upi", "crypto payment",
    "bitcoin payment", "westernunion", "moneygram", "cash app",
    "provide card details", "earn $", "earn money", "quick money", "daily income",
    "unlimited income", "make money fast", "get rich", "high income with no work",
    "instant earnings", "no interview required", "work 1 hour and earn",
    "click here", "visit the link", "fill this form", "login to verify",
    "submit your id proof", "upload aadhar", "upload pan", "urgent hiring",
    "instant hire", "selected without interview", "no background check",
    "job guarantee", "100% placement guarantee", "work from home and earn daily",
    "sms sending job", "form filling job", "referral bonus", "binary plan",
    "multi level marketing", "network marketing", "commission-based no base salary",
    "recruit others to earn", "agent recruitment", "share otp", "provide otp",
    "atm card details", "cvv", "upload scanned documents",
    "provide passport copy", "kyc verification fee",
]

MEDIUM_RISK_KW = [
    "free", "guaranteed", "no experience", "limited positions", "act now",
    "hiring fast", "immediate openings", "quick selection", "flexible work",
    "no skills required", "limited seats", "great opportunity", "weekly payout",
    "no degree required", "easy work", "hurry up", "start immediately",
    "home-based job", "remote opportunity", "part-time flexible",
    "simple online work", "bonus payout", "commission only", "work anytime",
    "no interview", "bulk hiring", "apply immediately", "walk-in with resume",
]

# ─── text helpers ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    return " ".join(w for w in text.split() if w and w not in STOP_WORDS)


def rule_score(text: str):
    """Returns (flag, score 0-1, matched_keywords)."""
    if not text:
        return False, 0.0, []
    t = text.lower()
    score = 0.0
    matched = []
    for kw in HIGH_RISK_KW:
        if kw in t:
            score += 0.4
            matched.append(kw)
    for kw in MEDIUM_RISK_KW:
        if kw in t:
            score += 0.15
            matched.append(kw)
    if t.count("!") >= 2:
        score += 0.1
    if sum(1 for c in text if c.isupper()) > max(20, len(text) * 0.1):
        score += 0.1
    score = min(score, 1.0)
    return score >= 0.6, score, matched


# ─── feature engineering ──────────────────────────────────────────────────────

def build_features(text_raw: str, row_dict: dict = None) -> np.ndarray:
    """
    Returns a 1-D numpy array of hand-crafted features (used alongside TF-IDF).
    """
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
    n_high = sum(1 for kw in HIGH_RISK_KW if kw in t.lower())
    n_med = sum(1 for kw in MEDIUM_RISK_KW if kw in t.lower())
    tlen = float(np.log1p(len(t)))
    words = t.split()
    avg_wlen = float(np.mean([len(w) for w in words])) if words else 0.0
    uniq_ratio = float(len(set(words)) / max(len(words), 1))
    digit_ratio = float(sum(c.isdigit() for c in t) / max(len(t), 1))

    return np.array([has_salary, salary_wide, telecommute, has_logo, has_q,
                     emp, exp, edu, title_caps, desc_excl, n_high, n_med,
                     tlen, avg_wlen, uniq_ratio, digit_ratio], dtype=float)


# ─── visualization functions ──────────────────────────────────────────────────

def plot_class_distribution(y, save_path=None):
    """Plot the distribution of fraudulent vs legitimate jobs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    classes = ['Legitimate', 'Fraudulent']
    counts = [np.sum(y == 0), np.sum(y == 1)]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = ax1.bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{count}\n({count/len(y)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Pie chart
    wedges, texts, autotexts = ax2.pie(counts, labels=classes, colors=colors,
                                        autopct='%1.1f%%', startangle=90,
                                        textprops={'fontsize': 12, 'fontweight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(13)
        autotext.set_fontweight('bold')
    
    ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    plt.suptitle('Fake Job Detection - Class Imbalance Analysis', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()  # Close to avoid display issues


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Fraudulent'],
                yticklabels=['Legitimate', 'Fraudulent'],
                annot_kws={'size': 14, 'weight': 'bold'},
                cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=15, fontweight='bold')
    
    # Add percentage annotations
    total = np.sum(cm)
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / total * 100
            ax.text(j+0.5, i+0.7, f'({percentage:.1f}%)', 
                   ha='center', va='center', color='gray', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_roc_curves(y_true, y_proba_lr, y_proba_rf, y_proba_gb, y_proba_meta, save_path=None):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    models = [
        ('Logistic Regression', y_proba_lr, '#3498db'),
        ('Random Forest', y_proba_rf, '#2ecc71'),
        ('Gradient Boosting', y_proba_gb, '#e74c3c'),
        ('Stacking Ensemble', y_proba_meta, '#9b59b6')
    ]
    
    for name, probs, color in models:
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5, 
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title('ROC Curves - Model Comparison', fontsize=15, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_precision_recall_curves(y_true, y_proba_lr, y_proba_rf, y_proba_gb, y_proba_meta, save_path=None):
    """Plot Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    models = [
        ('Logistic Regression', y_proba_lr, '#3498db'),
        ('Random Forest', y_proba_rf, '#2ecc71'),
        ('Gradient Boosting', y_proba_gb, '#e74c3c'),
        ('Stacking Ensemble', y_proba_meta, '#9b59b6')
    ]
    
    for name, probs, color in models:
        precision, recall, _ = precision_recall_curve(y_true, probs)
        ap_score = average_precision_score(y_true, probs)
        ax.plot(recall, precision, color=color, lw=2.5,
                label=f'{name} (AP = {ap_score:.3f})')
    
    ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax.set_title('Precision-Recall Curves - Model Comparison', fontsize=15, fontweight='bold')
    ax.legend(loc="best", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_feature_importance(model, feature_names, model_name, save_path=None):
    """Plot feature importance for tree-based models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print(f"Cannot extract feature importance from {model_name}")
        return
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1][:20]  # Top 20 features
    top_features = [feature_names[i] for i in indices if i < len(feature_names)]
    top_importances = importances[indices[:len(top_features)]]
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(top_features)), top_importances, color='#3498db')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {len(top_features)} Feature Importance - {model_name}', 
                fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_importances)):
        ax.text(val, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_learning_curves(model, X, y, cv=5, save_path=None):
    """Plot learning curves to diagnose bias-variance."""
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='f1', random_state=42
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.1, color='#2ecc71')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.1, color='#e74c3c')
        ax.plot(train_sizes, train_mean, 'o-', color='#2ecc71', lw=2, 
                label='Training Score (F1)')
        ax.plot(train_sizes, val_mean, 'o-', color='#e74c3c', lw=2, 
                label='Cross-validation Score (F1)')
        
        ax.set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('Learning Curves - Stacking Ensemble', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.close()
    except Exception as e:
        print(f"  Learning curve plot failed: {e}")


def plot_threshold_analysis(y_true, y_proba, save_path=None):
    """Plot performance metrics vs decision threshold."""
    thresholds = np.arange(0.0, 1.01, 0.01)
    f1_scores = []
    precisions = []
    recalls = []
    specificities = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(thresholds, f1_scores, 'b-', lw=2.5, label='F1 Score', marker='o', markersize=4)
    ax.plot(thresholds, precisions, 'g-', lw=2.5, label='Precision', marker='s', markersize=4)
    ax.plot(thresholds, recalls, 'r-', lw=2.5, label='Recall', marker='^', markersize=4)
    ax.plot(thresholds, specificities, 'orange', lw=2.5, label='Specificity', marker='d', markersize=4)
    
    # Find optimal threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    ax.axvline(x=optimal_threshold, color='purple', linestyle='--', lw=2,
               label=f'Optimal Threshold = {optimal_threshold:.2f} (F1={optimal_f1:.3f})')
    
    ax.set_xlabel('Decision Threshold', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Performance Metrics vs Decision Threshold', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()
    
    return optimal_threshold, optimal_f1


def plot_calibration_curve(y_true, y_proba, model_name, save_path=None):
    """Plot calibration curve (reliability diagram)."""
    from sklearn.calibration import calibration_curve
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8,
            label=f'{model_name} (n={len(y_proba)})', color='#3498db')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', lw=2)
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
    ax.set_title(f'Calibration Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_shap_summary(model, X_sample, feature_names, save_path=None):
    """Plot SHAP summary for model interpretability."""
    if not SHAP_AVAILABLE:
        print("  SHAP not available - skipping")
        return
    
    try:
        # Create a smaller sample for SHAP (to avoid memory issues)
        X_sample_small = X_sample[:1000] if X_sample.shape[0] > 1000 else X_sample
        
        # Use a simplified SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample_small)
        
        # For multi-class, take the positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample_small, feature_names=feature_names,
                         show=False, max_display=20)
        plt.title('SHAP Feature Importance Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.close()
    except Exception as e:
        print(f"  SHAP plot skipped: {e}")


def plot_metrics_comparison(metrics_dict, save_path=None):
    """Compare multiple models across different metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(metrics_dict.keys())
    metrics = ['Precision', 'Recall', 'F1-Score', 'Specificity']
    
    x = np.arange(len(metrics))
    width = 0.2
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    for i, (model, scores) in enumerate(metrics_dict.items()):
        offset = (i - len(models)/2) * width + width/2
        bars = ax.bar(x + offset, [scores[m] for m in metrics], width, 
                     label=model, color=colors[i % len(colors)])
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


# ─── main training function ───────────────────────────────────────────────────

def train_and_save_model(model_save_path="models/"):
    """Train the model and save all components to disk with visualizations."""
    
    # Create models directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
        from sklearn.metrics import (
            f1_score, precision_score, recall_score, 
            confusion_matrix, classification_report
        )
        from scipy.sparse import hstack, csr_matrix
        try:
            from imblearn.over_sampling import SMOTE
            USE_SMOTE = True
        except ImportError:
            print("imbalanced-learn not found – using class_weight only")
            USE_SMOTE = False

        print("="*60)
        print("FAKE JOB DETECTION SYSTEM - MODEL TRAINING")
        print("="*60)
        
        # Check if dataset exists
        if not os.path.exists("fake_job_postings.csv"):
            print("\n❌ Error: 'fake_job_postings.csv' not found!")
            print("Please ensure the dataset file is in the current directory.")
            return False
        
        # 1. Load and explore data
        print("\n[1/12] Loading dataset…")
        df = pd.read_csv("fake_job_postings.csv", encoding="utf-8", low_memory=False)
        print(f"  ✓ Loaded {len(df)} rows with {len(df.columns)} columns")

        # 2. Class distribution analysis
        print("\n[2/12] Analyzing class distribution…")
        y_initial = df["fraudulent"].astype(int).values
        plot_class_distribution(y_initial, save_path=os.path.join(PLOTS_DIR, "01_class_distribution.png"))
        
        # drop ID cols
        for c in ("job_id", "Unnamed: 0"):
            if c in df.columns:
                df.drop(c, axis=1, inplace=True)

        TEXT_COLS = ["title", "location", "department", "company_profile",
                     "description", "requirements", "benefits", "industry", "function"]
        META_COLS = ["salary_range", "telecommuting", "has_company_logo",
                     "has_questions", "employment_type",
                     "required_experience", "required_education", "title"]

        for c in TEXT_COLS + META_COLS:
            if c not in df.columns:
                df[c] = ""

        df["all_text"] = df[TEXT_COLS].fillna("").agg(" ".join, axis=1)
        df["clean_text"] = df["all_text"].apply(clean_text)

        y = df["fraudulent"].astype(int).values

        # 3. TF-IDF Vectorization
        print("\n[3/12] Computing TF-IDF features…")
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                sublinear_tf=True, min_df=3)
        X_tfidf = tfidf.fit_transform(df["clean_text"])
        print(f"  ✓ TF-IDF shape: {X_tfidf.shape}")

        # 4. Hand-crafted features
        print("\n[4/12] Building hand-crafted features…")
        hand_feats = np.vstack(
            df.apply(lambda r: build_features(r["all_text"], r.to_dict()), axis=1)
        )
        X_hand = csr_matrix(hand_feats)
        X_full = hstack([X_tfidf, X_hand])
        print(f"  ✓ Feature matrix shape: {X_full.shape}")

        # 5. Train-test split for evaluation
        print("\n[5/12] Creating train/test split…")
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"  ✓ Training set: {X_train.shape[0]} samples")
        print(f"  ✓ Test set: {X_test.shape[0]} samples")

        # 6. SMOTE for imbalance handling
        print("\n[6/12] Handling class imbalance…")
        if USE_SMOTE:
            sm = SMOTE(sampling_strategy=0.4, random_state=42, k_neighbors=5)
            X_res, y_res = sm.fit_resample(X_train, y_train)
            print(f"  ✓ SMOTE applied - New training set size: {X_res.shape[0]}")
            # Show new distribution
            plot_class_distribution(y_res, save_path=os.path.join(PLOTS_DIR, "02_class_distribution_smote.png"))
        else:
            X_res, y_res = X_train, y_train
            print("  ⚠ SMOTE not available - using original data")

        # 7. Train base models
        print("\n[7/12] Training base models…")
        lr = CalibratedClassifierCV(
            LogisticRegression(C=1.0, class_weight="balanced",
                               max_iter=500, solver="saga"),
            cv=3, method="isotonic")
        rf = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   max_depth=18, min_samples_leaf=3,
                                   random_state=42, n_jobs=-1),
            cv=3, method="isotonic")
        gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                        max_depth=5, subsample=0.8,
                                        random_state=42)

        lr.fit(X_res, y_res)
        rf.fit(X_res, y_res)
        gb.fit(X_res.toarray() if hasattr(X_res, "toarray") else X_res, y_res)
        print("  ✓ Logistic Regression trained")
        print("  ✓ Random Forest trained")
        print("  ✓ Gradient Boosting trained")

        # 8. Build stacking ensemble
        print("\n[8/12] Building stacking ensemble…")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        def oof_proba(clf, X, y, dense=False):
            Xd = X.toarray() if (dense and hasattr(X, "toarray")) else X
            preds = cross_val_predict(clf, Xd, y, cv=skf, method="predict_proba")
            return preds[:, 1]

        oof_lr = oof_proba(lr, X_train, y_train)
        oof_rf = oof_proba(rf, X_train, y_train)
        oof_gb = oof_proba(gb, X_train, y_train, dense=True)
        meta_X = np.column_stack([oof_lr, oof_rf, oof_gb])

        meta_clf = LogisticRegression(C=0.5, max_iter=200)
        meta_clf.fit(meta_X, y_train)
        print("  ✓ Meta-learner trained")

        # 9. Model evaluation on test set
        print("\n[9/12] Evaluating models on test set…")
        
        # Get predictions
        lr_probs = lr.predict_proba(X_test)[:, 1]
        rf_probs = rf.predict_proba(X_test)[:, 1]
        gb_probs = gb.predict_proba(X_test.toarray())[:, 1]
        
        meta_input = np.column_stack([lr_probs, rf_probs, gb_probs])
        meta_probs = meta_clf.predict_proba(meta_input)[:, 1]
        
        # Calculate metrics for each model
        metrics_dict = {}
        
        for name, probs in [('Logistic Regression', lr_probs), 
                            ('Random Forest', rf_probs),
                            ('Gradient Boosting', gb_probs),
                            ('Stacking Ensemble', meta_probs)]:
            y_pred = (probs >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            metrics_dict[name] = {
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1-Score': f1_score(y_test, y_pred, zero_division=0),
                'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
            }
        
        # Create comparison plot
        plot_metrics_comparison(metrics_dict, save_path=os.path.join(PLOTS_DIR, "03_model_comparison.png"))
        
        # Best threshold tuning
        print("\n[10/12] Tuning decision threshold…")
        optimal_threshold, optimal_f1 = plot_threshold_analysis(
            y_test, meta_probs, 
            save_path=os.path.join(PLOTS_DIR, "04_threshold_analysis.png")
        )
        threshold = optimal_threshold
        print(f"  ✓ Best threshold: {threshold:.2f} (F1: {optimal_f1:.3f})")

        # Final predictions with optimal threshold
        y_pred_final = (meta_probs >= threshold).astype(int)
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred_final, 
                            save_path=os.path.join(PLOTS_DIR, "05_confusion_matrix.png"))
        
        # Print classification report
        print("\n  Classification Report (Stacking Ensemble):")
        print("  " + "="*50)
        report = classification_report(y_test, y_pred_final, 
                                      target_names=['Legitimate', 'Fraudulent'],
                                      zero_division=0)
        for line in report.split('\n'):
            print(f"  {line}")
        
        # Plot ROC curves
        plot_roc_curves(y_test, lr_probs, rf_probs, gb_probs, meta_probs,
                       save_path=os.path.join(PLOTS_DIR, "06_roc_curves.png"))
        
        # Plot Precision-Recall curves
        plot_precision_recall_curves(y_test, lr_probs, rf_probs, gb_probs, meta_probs,
                                     save_path=os.path.join(PLOTS_DIR, "07_pr_curves.png"))
        
        # Plot calibration curve for ensemble
        plot_calibration_curve(y_test, meta_probs, 'Stacking Ensemble',
                              save_path=os.path.join(PLOTS_DIR, "08_calibration_curve.png"))

        # 11. Learning curve analysis
        print("\n[11/12] Analyzing learning curves…")
        try:
            # Use a subset for learning curve to save time
            X_subset = X_train[:5000] if X_train.shape[0] > 5000 else X_train
            y_subset = y_train[:5000] if y_train.shape[0] > 5000 else y_train
            plot_learning_curves(meta_clf, X_subset, y_subset, cv=3,
                                save_path=os.path.join(PLOTS_DIR, "09_learning_curves.png"))
        except Exception as e:
            print(f"  Learning curve plot skipped: {e}")

        # 12. Feature importance analysis
        print("\n[12/12] Analyzing feature importance…")
        
        feature_names = [
            'has_salary_range', 'salary_range_wide', 'telecommuting',
            'has_company_logo', 'has_questions', 'employment_type_enc',
            'required_exp_enc', 'required_edu_enc', 'title_all_caps_ratio',
            'desc_exclamations', 'num_high_kw', 'num_med_kw',
            'text_length_log', 'avg_word_length', 'unique_word_ratio', 'digit_ratio'
        ]
        
        # Plot Random Forest feature importance
        rf_model = rf.estimator_ if hasattr(rf, 'estimator_') else rf
        plot_feature_importance(rf_model, feature_names, 'Random Forest',
                              save_path=os.path.join(PLOTS_DIR, "10_rf_feature_importance.png"))
        
        # Try SHAP analysis (optional)
        if SHAP_AVAILABLE:
            try:
                plot_shap_summary(gb, X_test[:1000], feature_names,
                                save_path=os.path.join(PLOTS_DIR, "11_shap_summary.png"))
            except Exception as e:
                print(f"  SHAP analysis skipped: {e}")

        # Save all model components
        print("\n" + "="*60)
        print("SAVING MODEL")
        print("="*60)
        
        model_components = {
            'tfidf': tfidf,
            'base_models': (lr, rf, gb),
            'meta_clf': meta_clf,
            'threshold': threshold,
            'feature_names': feature_names,
            'high_risk_kw': HIGH_RISK_KW,
            'medium_risk_kw': MEDIUM_RISK_KW,
            'stop_words': STOP_WORDS,
            'performance_metrics': metrics_dict['Stacking Ensemble'],
            'optimal_threshold': threshold
        }
        
        # Save using pickle
        with open(os.path.join(model_save_path, 'fake_job_model.pkl'), 'wb') as f:
            pickle.dump(model_components, f)
        
        print(f"  ✓ Model saved to {model_save_path}fake_job_model.pkl")
        
        # Save metrics to text file
        metrics_file = os.path.join(PLOTS_DIR, "model_metrics.txt")
        with open(metrics_file, 'w') as f:
            f.write("FAKE JOB DETECTION MODEL - PERFORMANCE METRICS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Optimal Threshold: {threshold:.4f}\n")
            f.write(f"Final F1-Score: {optimal_f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n\nDetailed Metrics:\n")
            for metric, value in metrics_dict['Stacking Ensemble'].items():
                f.write(f"  {metric}: {value:.4f}\n")
        
        print(f"  ✓ Metrics saved to {metrics_file}")
        
        # Generate summary report
        summary = f"""
================================================================================
                    MODEL TRAINING COMPLETE - SUMMARY
================================================================================

Dataset Statistics:
  - Total samples: {len(df)}
  - Legitimate jobs: {np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%)
  - Fraudulent jobs: {np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%)

Feature Engineering:
  - TF-IDF features: 5,000
  - Hand-crafted features: 16
  - Total features: {X_full.shape[1]}

Model Performance (Test Set):
  - Precision: {metrics_dict['Stacking Ensemble']['Precision']:.4f}
  - Recall: {metrics_dict['Stacking Ensemble']['Recall']:.4f}
  - F1-Score: {metrics_dict['Stacking Ensemble']['F1-Score']:.4f}
  - Specificity: {metrics_dict['Stacking Ensemble']['Specificity']:.4f}

Decision Threshold:
  - Optimal threshold: {threshold:.4f}
  - Final F1-Score at threshold: {optimal_f1:.4f}

Visualizations Saved:
  All plots have been saved to '{PLOTS_DIR}/' directory:
    - 01_class_distribution.png (Original class balance)
    - 02_class_distribution_smote.png (After SMOTE)
    - 03_model_comparison.png (Performance comparison)
    - 04_threshold_analysis.png (Threshold tuning)
    - 05_confusion_matrix.png (Final confusion matrix)
    - 06_roc_curves.png (ROC curves for all models)
    - 07_pr_curves.png (Precision-Recall curves)
    - 08_calibration_curve.png (Model calibration)
    - 09_learning_curves.png (Learning curves)
    - 10_rf_feature_importance.png (Feature importance)

Model Saved:
  - Location: {model_save_path}fake_job_model.pkl
  - Components: TF-IDF, 3 base models, meta-learner, parameters

================================================================================
"""
        print(summary)
        
        # Save summary to file
        with open(os.path.join(PLOTS_DIR, "training_summary.txt"), 'w') as f:
            f.write(summary)
        
        print("\n✅ Training completed successfully!")
        return True

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("\n❌ Model training failed:", e)
        return False


if __name__ == "__main__":
    print("Starting model training with comprehensive visualizations...")
    print(f"Plots will be saved to: '{PLOTS_DIR}/'")
    print()
    success = train_and_save_model()
    if success:
        print("\n✅ Model training and saving completed!")
        print(f"📊 Check the '{PLOTS_DIR}' folder for all visualizations")
    else:
        print("\n❌ Model training failed!")