"""
Phase 5: Machine Learning Models
1. Classification models for stance prediction (Ensemble Voting Classifier)
2. Feature importance analysis (Top keywords per stance)
3. Cross-platform prediction (Train on Reddit -> Predict YouTube and vice versa)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import os
import warnings
warnings.filterwarnings('ignore')

# Create output directory
os.makedirs('ml_output', exist_ok=True)

print("="*80)
print("PHASE 5: MACHINE LEARNING - STANCE PREDICTION (ADVANCED ENSEMBLE)")
print("="*80)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n📊 Loading data...")
reddit_df = pd.read_csv('sentiment_output/reddit_with_sentiment.csv')
youtube_df = pd.read_csv('sentiment_output/youtube_with_sentiment.csv')

# Standardize columns
reddit_df['text'] = reddit_df['clean_text_comments'].fillna('')
youtube_df['text'] = youtube_df['text'].fillna('')

# Filter valid labels (P, I, N)
valid_labels = ['P', 'I', 'N']
reddit_df = reddit_df[reddit_df['Label'].isin(valid_labels)]
youtube_df = youtube_df[youtube_df['label'].isin(valid_labels)]

# Map labels to full names for better plots
label_map = {'P': 'Pro-Palestine', 'I': 'Pro-Israel', 'N': 'Neutral'}
reddit_df['label_full'] = reddit_df['Label'].map(label_map)
youtube_df['label_full'] = youtube_df['label'].map(label_map)

print(f"✓ Reddit (Filtered): {len(reddit_df)} rows")
print(f"✓ YouTube (Filtered): {len(youtube_df)} rows")

# ============================================================================
# 2. MODEL DEFINITION
# ============================================================================
def get_ensemble_model():
    """
    Creates a Voting Classifier composed of:
    1. Logistic Regression (Baseline)
    2. Linear SVM (Good for high-dimensional text)
    3. Random Forest (Captures non-linearities)
    4. Gradient Boosting (Iterative correction)
    """
    clf1 = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    clf2 = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
    clf3 = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    clf4 = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', clf1), 
            ('svm', clf2), 
            ('rf', clf3), 
            ('gb', clf4)
        ],
        voting='soft'
    )
    return voting_clf

# ============================================================================
# 3. TRAINING & EVALUATION FUNCTION
# ============================================================================
def train_evaluate_model(train_df, test_df, train_name, test_name):
    print(f"\n🤖 Training on {train_name}, Testing on {test_name}...")
    
    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(train_df['label_full'])
    y_test_enc = le.transform(test_df['label_full'])
    
    # Vectorize
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
    X_train_vec = tfidf.fit_transform(train_df['text'])
    X_test_vec = tfidf.transform(test_df['text'])
    
    # Train Ensemble
    model = get_ensemble_model()
    model.fit(X_train_vec, y_train_enc)
    
    # Predict
    y_pred_enc = model.predict(X_test_vec)
    y_pred = le.inverse_transform(y_pred_enc)
    y_test = test_df['label_full']
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {acc:.4f}")
    
    # Save Classification Report
    report = classification_report(y_test, y_pred)
    with open(f'ml_output/report_{train_name}_to_{test_name}.txt', 'w') as f:
        f.write(f"Training Data: {train_name}\nTesting Data: {test_name}\n")
        f.write(f"Model: Ensemble Voting Classifier (LR+SVM+RF+GB)\n")
        f.write("="*50 + "\n")
        f.write(report)
    
    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred, labels=['Pro-Palestine', 'Pro-Israel', 'Neutral'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pro-P', 'Pro-I', 'Neutral'],
                yticklabels=['Pro-P', 'Pro-I', 'Neutral'])
    plt.title(f'Ensemble Model Confusion Matrix: {train_name} -> {test_name}\nAccuracy: {acc:.2%}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    try:
        plt.savefig(f'ml_output/cm_{train_name}_to_{test_name}.png', dpi=300)
        print(f"   ✓ Saved confusion matrix: cm_{train_name}_to_{test_name}.png")
    except Exception as e:
        print(f"⚠️ Could not save plot: {e}")
    finally:
        plt.close()
    
    return model, tfidf, le

# ============================================================================
# 4. EXPERIMENT 1: WITHIN-PLATFORM SPLIT
# ============================================================================
print("\n" + "-"*60)
print("EXPERIMENT 1: WITHIN-PLATFORM PREDICTION")
print("-"*(60))

# Reddit Split
r_train, r_test = train_test_split(reddit_df, test_size=0.2, random_state=42, stratify=reddit_df['label_full'])
reddit_model, reddit_tfidf, reddit_le = train_evaluate_model(r_train, r_test, "Reddit", "Reddit")

# YouTube Split
y_train, y_test = train_test_split(youtube_df, test_size=0.2, random_state=42, stratify=youtube_df['label_full'])
youtube_model, youtube_tfidf, youtube_le = train_evaluate_model(y_train, y_test, "YouTube", "YouTube")

# ============================================================================
# 5. EXPERIMENT 2: CROSS-PLATFORM PREDICTION
# ============================================================================
print("\n" + "-"*60)
print("EXPERIMENT 2: CROSS-PLATFORM GENERALIZATION")
print("-"*(60))

# Train Reddit -> Test YouTube
train_evaluate_model(reddit_df, youtube_df, "Reddit", "YouTube")

# Train YouTube -> Test Reddit
train_evaluate_model(youtube_df, reddit_df, "YouTube", "Reddit")

# ============================================================================
# 6. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "-"*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("-"*(60))

def plot_feature_importance(model, tfidf, le, title_prefix):
    # Extract the Logistic Regression estimator from the VotingClassifier
    # It is the first estimator ('lr')
    lr_model = model.estimators_[0]
    
    feature_names = tfidf.get_feature_names_out()
    classes = le.classes_
    
    # Get top 15 features for each class
    top_features = {}
    
    for i, class_label in enumerate(classes):
        # For binary, coef_ is (1, n_features), for multi-class it is (n_classes, n_features)
        if len(classes) == 2:
            coefs = lr_model.coef_[0] if i == 1 else -lr_model.coef_[0]
        else:
            coefs = lr_model.coef_[i]
            
        top_indices = np.argsort(coefs)[-15:] # Top 15
        top_words = [(feature_names[j], coefs[j]) for j in top_indices]
        top_features[class_label] = top_words
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle(f'{title_prefix}: Top Predictive Keywords (Ensemble Model)', fontsize=16, fontweight='bold')
    
    colors = {'Pro-Palestine': '#2ecc71', 'Pro-Israel': '#3498db', 'Neutral': '#95a5a6'}
    
    for i, class_label in enumerate(['Pro-Palestine', 'Pro-Israel', 'Neutral']):
        if class_label in top_features:
            words, scores = zip(*top_features[class_label])
            axes[i].barh(words, scores, color=colors.get(class_label, 'gray'))
            axes[i].set_title(class_label, fontsize=14)
            axes[i].set_xlabel('Coefficient Magnitude')
    
    plt.tight_layout()
    try:
        plt.savefig(f'ml_output/features_{title_prefix.lower()}.png', dpi=300)
        print(f"✓ Saved feature importance: features_{title_prefix.lower()}.png")
    except Exception as e:
        print(f"⚠️ Could not save feature plot: {e}")
    finally:
        plt.close()

# Analyze features for the Reddit model
plot_feature_importance(reddit_model, reddit_tfidf, reddit_le, "Reddit")

# Analyze features for the YouTube model
plot_feature_importance(youtube_model, youtube_tfidf, youtube_le, "YouTube")

print("\n" + "="*80)
print("✅ ML ANALYSIS COMPLETE")
print("="*80)
