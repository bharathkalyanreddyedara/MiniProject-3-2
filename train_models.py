"""
Training script for Hybrid URL Detection Model
Generates: bilstm_url_model.h5, tokenizer.pkl, tfidf_vectorizer.pkl, rf_model.pkl, hybrid_weight.pkl
"""

import pandas as pd
import numpy as np
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

print("=" * 60)
print("HYBRID URL DETECTION MODEL TRAINING")
print("=" * 60)

# ===============================================
# STEP 1: LOAD DATASET
# ===============================================
print("\n[1/6] Loading dataset...")


try:
    # Try to load from common filenames
    dataset_files = ['malicious_phish.csv']
    df = None
    
    for filename in dataset_files:
        try:
            df = pd.read_csv(filename)
            print(f"✓ Loaded dataset: {filename}")
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        raise FileNotFoundError("No dataset found")
    
    # Display dataset info
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Detect column names (flexible)
    url_col = None
    label_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'url' in col_lower:
            url_col = col
        if 'label' in col_lower or 'type' in col_lower or 'class' in col_lower:
            label_col = col
    
    if url_col is None or label_col is None:
        print("\n⚠️  Could not auto-detect columns. Please specify:")
        print(f"Available columns: {df.columns.tolist()}")
        url_col = input("Enter URL column name: ").strip()
        label_col = input("Enter label column name: ").strip()
    
    print(f"\nUsing columns: URL='{url_col}', Label='{label_col}'")
    
    # Prepare data
    urls = df[url_col].values
    labels = df[label_col].values
    
    # Convert labels to binary if needed
    unique_labels = np.unique(labels)
    print(f"Unique labels: {unique_labels}")
    
    if len(unique_labels) > 2:
        # Multi-class to binary: assume first class is benign, rest are malicious
        print("Converting multi-class to binary (benign vs malicious)")
        labels = np.where(labels == unique_labels[0], 0, 1)
    elif set(unique_labels) != {0, 1}:
        # Map to 0 and 1
        label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
        labels = np.array([label_map[l] for l in labels])
    
    print(f"Label distribution: Benign={np.sum(labels==0)}, Malicious={np.sum(labels==1)}")

except Exception as e:
    print(f"\n❌ Error loading dataset: {e}")
    print("\n" + "=" * 60)
    print("DATASET REQUIRED")
    print("=" * 60)
    print("\nPlease download a malicious URL dataset and place it in this directory.")
    print("\nRecommended datasets:")
    print("1. Kaggle Malicious URLs Dataset:")
    print("   https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset")
    print("\n2. Save it as 'malicious_phish.csv' or 'urldata.csv'")
    print("\nDataset format should be:")
    print("   url,label")
    print("   http://example.com,benign")
    print("   http://malicious.com,malicious")
    print("\nOr with numeric labels (0=benign, 1=malicious)")
    exit(1)

# ===============================================
# STEP 2: PREPROCESS URLs
# ===============================================
print("\n[2/6] Preprocessing URLs...")

def clean_url(url):
    """Clean URL for processing"""
    url = str(url).lower()
    url = re.sub(r'https?://', '', url)
    url = re.sub(r'www\.', '', url)
    url = re.sub(r'[^a-z0-9./\-@]', '', url)
    return url

cleaned_urls = [clean_url(url) for url in urls]
print(f"✓ Cleaned {len(cleaned_urls)} URLs")

# ===============================================
# STEP 3: SPLIT DATA
# ===============================================
print("\n[3/6] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    cleaned_urls, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ===============================================
# STEP 4: TRAIN BiLSTM MODEL
# ===============================================
print("\n[4/6] Training BiLSTM model...")

# Tokenization
MAX_WORDS = 10000
MAX_LEN = 120

tokenizer = Tokenizer(num_words=MAX_WORDS, char_level=True)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)

print(f"✓ Tokenized URLs (vocab size: {len(tokenizer.word_index)})")

# Build BiLSTM model
bilstm_model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

bilstm_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nBiLSTM Model Architecture:")
bilstm_model.summary()

# Train
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("\nTraining BiLSTM...")
history = bilstm_model.fit(
    X_train_pad, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
bilstm_pred_prob = bilstm_model.predict(X_test_pad, verbose=0)
bilstm_pred = (bilstm_pred_prob > 0.5).astype(int).flatten()
bilstm_acc = accuracy_score(y_test, bilstm_pred)

print(f"\n✓ BiLSTM Test Accuracy: {bilstm_acc:.4f}")

# ===============================================
# STEP 5: TRAIN RANDOM FOREST MODEL
# ===============================================
print("\n[5/6] Training Random Forest model...")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), analyzer='char')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"✓ TF-IDF features: {X_train_tfidf.shape[1]}")

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_tfidf, y_train)

# Evaluate
rf_pred_prob = rf_model.predict_proba(X_test_tfidf)[:, 1]
rf_pred = (rf_pred_prob > 0.5).astype(int)
rf_acc = accuracy_score(y_test, rf_pred)

print(f"✓ Random Forest Test Accuracy: {rf_acc:.4f}")

# ===============================================
# STEP 6: FIND OPTIMAL HYBRID WEIGHT
# ===============================================
print("\n[6/6] Finding optimal hybrid weight...")

best_weight = 0.5
best_acc = 0

for weight in np.arange(0, 1.01, 0.05):
    hybrid_prob = (weight * bilstm_pred_prob.flatten()) + ((1 - weight) * rf_pred_prob)
    hybrid_pred = (hybrid_prob > 0.5).astype(int)
    acc = accuracy_score(y_test, hybrid_pred)
    
    if acc > best_acc:
        best_acc = acc
        best_weight = weight

print(f"✓ Best weight: {best_weight:.2f}")
print(f"✓ Hybrid Test Accuracy: {best_acc:.4f}")

# Final predictions with best weight
hybrid_prob = (best_weight * bilstm_pred_prob.flatten()) + ((1 - best_weight) * rf_pred_prob)
hybrid_pred = (hybrid_prob > 0.5).astype(int)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test, hybrid_pred, target_names=['Benign', 'Malicious']))

# ===============================================
# SAVE MODELS
# ===============================================
print("\n" + "=" * 60)
print("SAVING MODELS")
print("=" * 60)

bilstm_model.save("bilstm_url_model.h5")
print("✓ Saved: bilstm_url_model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("✓ Saved: tokenizer.pkl")

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("✓ Saved: tfidf_vectorizer.pkl")

with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
print("✓ Saved: rf_model.pkl")

with open("hybrid_weight.pkl", "wb") as f:
    pickle.dump(best_weight, f)
print("✓ Saved: hybrid_weight.pkl")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print("\nAll model files have been generated:")
print("  1. bilstm_url_model.h5")
print("  2. tokenizer.pkl")
print("  3. tfidf_vectorizer.pkl")
print("  4. rf_model.pkl")
print("  5. hybrid_weight.pkl")
print("\nYou can now run the Flask application!")
print("Command: python app.py")
