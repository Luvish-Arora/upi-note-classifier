import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib
import warnings
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Load Dataset ===
excel_path = 'enhanced_transaction_notes_dataset.xlsx'
df = pd.read_excel(excel_path)
logging.info(f"Dataset shape: {df.shape}")
logging.info(f"Category distribution:\n{df['category'].value_counts()}")

# === Consolidate Similar Categories ===
# Map similar categories to reduce class variance
category_mapping = {
    'Gifts': 'Gifts & Donations',
    'Bills & Expenses': 'Utilities'
}

df['category'] = df['category'].replace(category_mapping)
logging.info(f"Updated category distribution:\n{df['category'].value_counts()}")

# === Enhanced Preprocessing Function ===
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Enhanced preprocessing of transaction notes with lemmatization and key phrase extraction."""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Replace specific transaction patterns
    text = re.sub(r'paid for', 'payment', text)
    text = re.sub(r'bought', 'purchase', text)
    
    # Remove punctuation but preserve specific patterns
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Extract transaction amount if present (could be useful feature)
    amount_match = re.search(r'\d+', text)
    has_amount = '1' if amount_match else '0'
    
    # Remove numbers but keep track if they existed
    text = re.sub(r'\d+', '', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and remove stopwords
    tokens = [word for word in text.split() if word not in stop_words]
    
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Enhanced token mapping with domain knowledge
    token_map = {
        'mcd': 'mcdonalds food restaurant',
        'elec': 'electricity utility bill',
        'bill': 'payment utility',
        'auto': 'transport travel',
        'grocery': 'groceries supermarket',
        'uber': 'travel taxi transport',
        'ola': 'travel taxi transport',
        'amazon': 'shopping online',
        'netflix': 'entertainment subscription',
        'zomato': 'food delivery restaurant',
        'swiggy': 'food delivery restaurant',
        'rent': 'housing home living',
        'gym': 'health fitness',
        'salon': 'personal care beauty',
        'medicine': 'health pharmacy',
        'books': 'education learning',
        'movie': 'entertainment cinema',
        'ticket': 'travel entertainment',
        'doctor': 'health medical',
        'gift': 'gifts donation present',
        'college': 'education tuition',
        'repair': 'maintenance home living',
        'recharge': 'utility phone',
        'dining': 'food restaurant',
        'office': 'work business',
        'coffee': 'food cafe',
        'loan': 'bills expenses finance',
        'emi': 'bills expenses finance',
        'wifi': 'utility internet',
        'haircut': 'personal care salon'
    }
    
    # Apply token mapping with expansion
    normalized_tokens = []
    for token in tokens:
        if token in token_map:
            normalized_tokens.extend(token_map[token].split())
        else:
            normalized_tokens.append(token)
    
    # Add presence of amount as a feature
    processed_text = ' '.join(normalized_tokens)
    if has_amount == '1':
        processed_text += ' has_amount'
        
    return processed_text

# Apply preprocessing
df['processed_note'] = df['note'].apply(preprocess_text)

# === Feature Engineering - Add Length Features ===
df['note_length'] = df['note'].apply(lambda x: len(str(x)) if not pd.isna(x) else 0)
df['word_count'] = df['note'].apply(lambda x: len(str(x).split()) if not pd.isna(x) else 0)

# === Display Samples ===
print("\nPreprocessed Examples:")
for i in range(min(5, len(df))):
    print(f"Original: '{df['note'].iloc[i]}' → Processed: '{df['processed_note'].iloc[i]}'")

# === Split Data ===
X_text = df['processed_note']
y = df['category']

# Use stratified split to maintain class distribution
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

logging.info(f"Train size: {X_train_text.shape[0]}, Test size: {X_test_text.shape[0]}")
logging.info(f"Training class distribution: \n{Counter(y_train)}")
logging.info(f"Testing class distribution: \n{Counter(y_test)}")

# === Text Feature Pipeline with SMOTE ===
text_pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        min_df=2,
        use_idf=True,
        sublinear_tf=True
    )),
    ('smote', SMOTE(random_state=42)),  # Handle class imbalance
    ('classifier', OneVsRestClassifier(LogisticRegression(
        C=1.0,
        solver='liblinear',
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )))
])

# === Model Evaluation Function ===
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    logging.info(f"{model_name} - Test Accuracy: {accuracy:.4f}")
    logging.info(f"{model_name} - F1-Score (Macro): {f1:.4f}")
    
    print(f"\n{model_name} - Test Accuracy: {accuracy:.4f}")
    print(f"{model_name} - F1-Score (Macro): {f1:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # Confusion Matrix
    plt.figure(figsize=(14, 12))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(set(y_test)), yticklabels=sorted(set(y_test)))
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    plt.close()
    
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': y_pred
    }
    return results

# === Try Different Models ===
models = {
    'Logistic Regression': text_pipeline,
    'Random Forest': ImbPipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 3), min_df=2)),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(
            n_estimators=200, 
            max_depth=None, 
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42
        ))
    ]),
    'Gradient Boosting': ImbPipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 3), min_df=2)),
        ('smote', SMOTE(random_state=42)),
        ('classifier', GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ))
    ])
}

best_model = None
best_score = 0
best_model_name = ""

# Train and evaluate each model
for model_name, model in models.items():
    logging.info(f"Training {model_name}...")
    model.fit(X_train_text, y_train)
    results = evaluate_model(model, X_test_text, y_test, model_name)
    
    if results['f1_score'] > best_score:
        best_score = results['f1_score']
        best_model = model
        best_model_name = model_name

logging.info(f"Best model: {best_model_name} with F1-Score: {best_score:.4f}")

# === Hyperparameter Tuning for Best Model ===
logging.info(f"Performing hyperparameter tuning for {best_model_name}...")

if best_model_name == "Logistic Regression":
    param_grid = {
        'classifier__estimator__C': [0.1, 1.0, 10.0],
        'classifier__estimator__solver': ['liblinear', 'saga'],
        'tfidf__max_features': [5000, 10000],
        'tfidf__ngram_range': [(1, 2), (1, 3)]
    }
elif best_model_name == "Random Forest":
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
        'tfidf__max_features': [5000, 10000],
        'tfidf__ngram_range': [(1, 2), (1, 3)]
    }
else:  # Gradient Boosting
    param_grid = {
        'classifier__n_estimators': [100, 150, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.05, 0.1, 0.2],
        'tfidf__max_features': [5000, 10000],
        'tfidf__ngram_range': [(1, 2), (1, 3)]
    }

# Use stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    best_model,
    param_grid,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_text, y_train)

logging.info(f"Best Parameters: {grid_search.best_params_}")
logging.info(f"Best Cross-Validation F1-Score: {grid_search.best_score_:.4f}")

# === Final Model ===
best_tuned_model = grid_search.best_estimator_
final_results = evaluate_model(best_tuned_model, X_test_text, y_test, f"Tuned {best_model_name}")

# === Analyze Feature Importance ===
def analyze_features(model, feature_names, top_n=15):
    if best_model_name == "Logistic Regression":
        # For logistic regression, we can get coefficients
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            classifier = model.named_steps['classifier']
            if hasattr(classifier, 'estimator'):  # OneVsRestClassifier
                if hasattr(classifier.estimator, 'coef_'):
                    classes = sorted(set(y_test))
                    for i, class_name in enumerate(classes):
                        if i < len(classifier.estimators_):
                            estimator = classifier.estimators_[i]
                            if hasattr(estimator, 'coef_'):
                                top_indices = np.argsort(estimator.coef_[0])[-top_n:]
                                top_features = [feature_names[j] for j in top_indices]
                                print(f"\nTop {top_n} features for category '{class_name}':")
                                for idx, feature in enumerate(reversed(top_features)):
                                    print(f"- {feature}")
    
    elif best_model_name == "Random Forest" or best_model_name == "Gradient Boosting":
        # For tree-based models, we can get feature importances
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            classifier = model.named_steps['classifier']
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                indices = np.argsort(importances)[-top_n:]
                print(f"\nTop {top_n} important features overall:")
                for i in reversed(indices):
                    if i < len(feature_names):
                        print(f"- {feature_names[i]}: {importances[i]:.4f}")

# Get feature names
if hasattr(best_tuned_model, 'named_steps') and 'tfidf' in best_tuned_model.named_steps:
    tfidf_vectorizer = best_tuned_model.named_steps['tfidf']
    feature_names = tfidf_vectorizer.get_feature_names_out()
    analyze_features(best_tuned_model, feature_names)

# === Ensemble (Majority Voting) ===
logging.info("Creating ensemble model (majority voting)...")

# Get predictions from all models
all_predictions = {}
for model_name, model in models.items():
    y_pred = model.predict(X_test_text)
    all_predictions[model_name] = y_pred

# Ensemble prediction (majority voting)
ensemble_predictions = []
for i in range(len(y_test)):
    votes = [all_predictions[model_name][i] for model_name in models]
    # Get the most common prediction
    most_common = Counter(votes).most_common(1)[0][0]
    ensemble_predictions.append(most_common)

# Evaluate ensemble
ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
ensemble_f1 = f1_score(y_test, ensemble_predictions, average='macro')

print("\nEnsemble Model (Majority Voting)")
print(f"Accuracy: {ensemble_accuracy:.4f}")
print(f"F1-Score (Macro): {ensemble_f1:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, ensemble_predictions)}")

# === Save Best Model Package ===
model_package = {
    'model': best_tuned_model,
    'preprocess_function': preprocess_text,
    'category_mapping': category_mapping,
    'model_type': best_model_name
}
joblib.dump(model_package, 'improved_transaction_classifier.pkl')
print(f"\nImproved model package saved as 'improved_transaction_classifier.pkl'")

# === Prediction Function ===
def predict_category(transaction_note, model_package):
    preprocess_func = model_package['preprocess_function']
    model = model_package['model']
    
    processed_note = preprocess_func(transaction_note)
    predicted_category = model.predict([processed_note])[0]
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba([processed_note])[0]
        max_prob = max(probabilities)
    else:
        # For models that don't provide probabilities
        max_prob = 1.0
        
    return predicted_category, max_prob

# === Sample Predictions with Best Model ===
print("\nSample Predictions with Best Model:")
sample_notes = [
    "auto Nikhil", "pizza Riya", "rent 4212", "mcd Suraj",
    "haircut Vaibhav", "elec bill", "grocery shopping", "unknown purchase", 
    "flight to Mumbai", "Netflix subscription", "medicine from pharmacy",
    "Amazon order", "college fees", "dinner with friends", "paid water bill"
]

for note in sample_notes:
    cat, prob = predict_category(note, model_package)
    print(f"Note: '{note}' → Category: '{cat}' (Confidence: {prob:.2f})")

print("\nModel training and evaluation complete!")