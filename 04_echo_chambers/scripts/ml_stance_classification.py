"""
ML Stance Classification — RQ3
Ensemble voting classifier (LR+SVM+RF+GB) for stance prediction.
Within-platform and cross-platform experiments with 5-fold cross-validation.
Column schema: Reddit uses self_text + Label; YouTube uses text + Label.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SENTIMENT_OUTPUT_DIR = ROOT_DIR / '02_emotional_tone_analysis' / 'outputs'
OUTPUT_DIR = ROOT_DIR / '04_echo_chambers' / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REDDIT_TEXT_COL   = 'self_text'
YOUTUBE_TEXT_COL  = 'text'
REDDIT_LABEL_COL  = 'Label'
YOUTUBE_LABEL_COL = 'Label'

label_map    = {'P': 'Pro-Palestine', 'I': 'Pro-Israel', 'N': 'Neutral'}
valid_labels = ['P', 'I', 'N']

print("=" * 80)
print("ML STANCE CLASSIFICATION — ENSEMBLE + CROSS-VALIDATION")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📊 Loading data...")
reddit_df  = pd.read_csv(SENTIMENT_OUTPUT_DIR / 'reddit_with_sentiment.csv')
youtube_df = pd.read_csv(SENTIMENT_OUTPUT_DIR / 'youtube_with_sentiment.csv')

# Use the correct text columns for each platform
reddit_df['text']  = reddit_df[REDDIT_TEXT_COL].fillna('')
youtube_df['text'] = youtube_df[YOUTUBE_TEXT_COL].fillna('')

reddit_df  = reddit_df[reddit_df[REDDIT_LABEL_COL].isin(valid_labels)].copy()
youtube_df = youtube_df[youtube_df[YOUTUBE_LABEL_COL].isin(valid_labels)].copy()

reddit_df['label_full']  = reddit_df[REDDIT_LABEL_COL].map(label_map)
youtube_df['label_full'] = youtube_df[YOUTUBE_LABEL_COL].map(label_map)

print(f"✔ Reddit  (filtered): {len(reddit_df):,}")
print(f"✔ YouTube (filtered): {len(youtube_df):,}")

# ============================================================================
# MODEL FACTORY
# ============================================================================
def get_ensemble():
    clf1 = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    clf2 = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
    clf3 = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                   random_state=42, n_jobs=-1)
    clf4 = GradientBoostingClassifier(n_estimators=100, random_state=42)
    return VotingClassifier(
        estimators=[('lr', clf1), ('svm', clf2), ('rf', clf3), ('gb', clf4)],
        voting='soft'
    )

def get_pipeline():
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    model = get_ensemble()
    return Pipeline([('tfidf', tfidf), ('clf', model)])

# ============================================================================
# TRAINING + EVALUATION WITH CROSS-VALIDATION
# ============================================================================
cv_results = {}

def train_evaluate(train_df, test_df, train_name, test_name, run_cv=True):
    print(f"\n🤖 {train_name} → {test_name}")

    le = LabelEncoder()
    y_train = le.fit_transform(train_df['label_full'])
    y_test  = le.transform(test_df['label_full'])

    pipe = get_pipeline()

    # 5-fold cross-validation on training set
    if run_cv and train_name == test_name:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_acc  = cross_val_score(pipe, train_df['text'], y_train,
                                  cv=skf, scoring='accuracy', n_jobs=-1)
        cv_f1   = cross_val_score(pipe, train_df['text'], y_train,
                                  cv=skf, scoring='f1_macro', n_jobs=-1)
        print(f"   5-fold CV Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
        print(f"   5-fold CV F1-macro: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
        cv_results[f'{train_name}'] = {'acc_mean': cv_acc.mean(), 'acc_std': cv_acc.std(),
                                       'f1_mean': cv_f1.mean(), 'f1_std': cv_f1.std()}

    # Train on full train_df, test on test_df
    pipe.fit(train_df['text'], y_train)
    y_pred_enc = pipe.predict(test_df['text'])
    y_pred     = le.inverse_transform(y_pred_enc)
    y_true     = test_df['label_full']

    acc    = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred)

    print(f"   Test Accuracy: {acc:.4f} | F1-macro: {f1_mac:.4f}")
    print(f"   Classification Report:\n{report}")

    # Save report
    report_path = OUTPUT_DIR / f'report_{train_name}_to_{test_name}.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Train: {train_name}  |  Test: {test_name}\n")
        f.write(f"Ensemble Voting Classifier (LR+SVM+RF+GB)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy: {acc:.4f}\nF1-macro: {f1_mac:.4f}\n\n")
        f.write(report)

    # Confusion matrix
    cm    = confusion_matrix(y_true, y_pred,
                             labels=['Pro-Palestine', 'Pro-Israel', 'Neutral'])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pro-P', 'Pro-I', 'Neutral'],
                yticklabels=['Pro-P', 'Pro-I', 'Neutral'])
    ax.set_title(f'Confusion Matrix: {train_name} → {test_name}\n'
                 f'Acc={acc:.2%}, F1={f1_mac:.3f}', fontweight='bold')
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'cm_{train_name}_to_{test_name}.png', dpi=300, bbox_inches='tight')
    print(f"   ✔ Saved: cm_{train_name}_to_{test_name}.png")
    plt.close()

    return pipe, le, acc, f1_mac


# ============================================================================
# EXPERIMENT 1: WITHIN-PLATFORM (with CV)
# ============================================================================
print("\n" + "-" * 60)
print("EXPERIMENT 1: WITHIN-PLATFORM PREDICTION (5-fold CV)")
print("-" * 60)

r_train, r_test = train_test_split(reddit_df,  test_size=0.2, random_state=42,
                                   stratify=reddit_df['label_full'])
y_train, y_test = train_test_split(youtube_df, test_size=0.2, random_state=42,
                                   stratify=youtube_df['label_full'])

r_pipe, r_le, r_acc, r_f1 = train_evaluate(r_train, r_test, 'Reddit',  'Reddit')
y_pipe, y_le, y_acc, y_f1 = train_evaluate(y_train, y_test, 'YouTube', 'YouTube')

# ============================================================================
# EXPERIMENT 2: CROSS-PLATFORM
# ============================================================================
print("\n" + "-" * 60)
print("EXPERIMENT 2: CROSS-PLATFORM PREDICTION")
print("-" * 60)

_, _, ry_acc, ry_f1 = train_evaluate(reddit_df,  youtube_df, 'Reddit',  'YouTube', run_cv=False)
_, _, yr_acc, yr_f1 = train_evaluate(youtube_df, reddit_df,  'YouTube', 'Reddit',  run_cv=False)

# ============================================================================
# EXPERIMENT SUMMARY TABLE
# ============================================================================
print("\n" + "-" * 60)
print("EXPERIMENT SUMMARY")
print("-" * 60)

summary = pd.DataFrame({
    'Experiment':  ['Reddit→Reddit', 'YouTube→YouTube', 'Reddit→YouTube', 'YouTube→Reddit'],
    'Accuracy':    [f'{r_acc:.4f}', f'{y_acc:.4f}', f'{ry_acc:.4f}', f'{yr_acc:.4f}'],
    'F1-macro':    [f'{r_f1:.4f}',  f'{y_f1:.4f}',  f'{ry_f1:.4f}',  f'{yr_f1:.4f}'],
})

if cv_results:
    cv_rows = []
    for exp in ['Reddit', 'YouTube']:
        if exp in cv_results:
            r = cv_results[exp]
            cv_rows.append(f"{r['acc_mean']:.3f}±{r['acc_std']:.3f}")
        else:
            cv_rows.append('N/A')
    cv_rows += ['N/A', 'N/A']
    summary['CV_Acc(5-fold)'] = cv_rows

print(summary.to_string(index=False))
summary.to_csv(OUTPUT_DIR / 'experiment_summary.csv', index=False)
print("\n✔ Saved: experiment_summary.csv")

# ============================================================================
# FEATURE IMPORTANCE (Logistic Regression coefficients)
# ============================================================================
print("\n" + "-" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("-" * 60)

def plot_feature_importance(pipe, le, title_prefix):
    lr_model    = pipe.named_steps['clf'].estimators_[0]
    tfidf_model = pipe.named_steps['tfidf']
    fn          = tfidf_model.get_feature_names_out()
    classes     = le.classes_

    top_features = {}
    for i, cls in enumerate(classes):
        if len(classes) == 2:
            coefs = lr_model.coef_[0] if i == 1 else -lr_model.coef_[0]
        else:
            coefs = lr_model.coef_[i]
        top_idx  = np.argsort(coefs)[-15:]
        top_features[cls] = [(fn[j], coefs[j]) for j in top_idx]

    display_classes = ['Pro-Palestine', 'Pro-Israel', 'Neutral']
    n_display = sum(1 for c in display_classes if c in top_features)
    if n_display == 0:
        return

    fig, axes = plt.subplots(1, n_display, figsize=(7 * n_display, 8))
    if n_display == 1:
        axes = [axes]
    fig.suptitle(f'{title_prefix}: Top Predictive Keywords (LR Coefficients)',
                 fontsize=15, fontweight='bold')

    colors_map = {'Pro-Palestine': '#2ecc71', 'Pro-Israel': '#3498db', 'Neutral': '#95a5a6'}
    ax_idx = 0
    for cls in display_classes:
        if cls not in top_features:
            continue
        words, scores = zip(*top_features[cls])
        axes[ax_idx].barh(list(words), list(scores),
                          color=colors_map.get(cls, 'gray'), alpha=0.85, edgecolor='black')
        axes[ax_idx].set_title(cls, fontsize=13, fontweight='bold')
        axes[ax_idx].set_xlabel('Coefficient Magnitude')
        axes[ax_idx].grid(axis='x', alpha=0.3)
        ax_idx += 1

    plt.tight_layout()
    fname = f'features_{title_prefix.lower().replace(" ", "_")}.png'
    plt.savefig(OUTPUT_DIR / fname, dpi=300, bbox_inches='tight')
    print(f"✔ Saved: {fname}")
    plt.close()

try:
    plot_feature_importance(r_pipe, r_le, 'Reddit')
except Exception as e:
    print(f"⚠️  Reddit feature plot skipped: {e}")

try:
    plot_feature_importance(y_pipe, y_le, 'YouTube')
except Exception as e:
    print(f"⚠️  YouTube feature plot skipped: {e}")

# ============================================================================
# PERFORMANCE COMPARISON BAR CHART
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
exps = ['Reddit→Reddit', 'YouTube→YouTube', 'Reddit→YouTube', 'YouTube→Reddit']
accs = [r_acc, y_acc, ry_acc, yr_acc]
f1s  = [r_f1,  y_f1,  ry_f1,  yr_f1]
x = np.arange(len(exps))
width = 0.35

bars1 = ax.bar(x - width/2, accs, width, label='Accuracy',  color='#3498db', alpha=0.85, edgecolor='black')
bars2 = ax.bar(x + width/2, f1s,  width, label='F1-macro',  color='#e74c3c', alpha=0.85, edgecolor='black')
ax.set_ylim(0, 1.05)
ax.set_xlabel('Experiment'); ax.set_ylabel('Score')
ax.set_title('ML Stance Classification: All Experiments', fontsize=14, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(exps, rotation=15, ha='right')
ax.legend(); ax.grid(axis='y', alpha=0.3)
ax.bar_label(bars1, fmt='%.3f', padding=3, fontsize=9)
ax.bar_label(bars2, fmt='%.3f', padding=3, fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'experiment_comparison.png', dpi=300, bbox_inches='tight')
print("✔ Saved: experiment_comparison.png")
plt.close()

print("\n" + "=" * 80)
print("✅ ML ANALYSIS COMPLETE")
print("=" * 80)
