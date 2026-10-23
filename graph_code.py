import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================================
# STEP 1: LOAD MODELS
# =========================================

print("Loading models...")

bilstm_model = load_model("bilstm_url_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("hybrid_weight.pkl", "rb") as f:
    best_weight = pickle.load(f)

# =========================================
# STEP 2: LOAD DATASET
# =========================================

df = pd.read_csv("malicious_phish.csv")  # Change if needed

urls = df["url"].values
labels = df["type"].values

# =========================================
# STEP 3: PREPROCESS URLs
# =========================================

def clean_url(url):
    url = str(url).lower()
    url = re.sub(r'https?://', '', url)
    url = re.sub(r'www\.', '', url)
    url = re.sub(r'[^a-z0-9./\-@]', '', url)
    return url

cleaned_urls = [clean_url(url) for url in urls]

# =========================================
# STEP 4: PREPARE INPUTS
# =========================================

MAX_LEN = 120

# BiLSTM input
seq = tokenizer.texts_to_sequences(cleaned_urls)
pad_seq = pad_sequences(seq, maxlen=MAX_LEN)

# RF input
tfidf = vectorizer.transform(cleaned_urls)

# =========================================
# STEP 5: PREDICTIONS
# =========================================

# BiLSTM
bilstm_prob = bilstm_model.predict(pad_seq, verbose=0).flatten()
bilstm_pred = (bilstm_prob > 0.5).astype(int)

# Random Forest
rf_prob = rf_model.predict_proba(tfidf)[:, 1]
rf_pred = (rf_prob > 0.5).astype(int)

# Hybrid
hybrid_prob = (best_weight * bilstm_prob) + ((1 - best_weight) * rf_prob)
hybrid_pred = (hybrid_prob > 0.5).astype(int)

# =========================================
# STEP 6: METRICS FUNCTION
# =========================================

def get_metrics(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred)
    )

bilstm_metrics = get_metrics(labels, bilstm_pred)
rf_metrics = get_metrics(labels, rf_pred)
hybrid_metrics = get_metrics(labels, hybrid_pred)

# =========================================
# STEP 7: PRINT RESULTS
# =========================================

models = ['BiLSTM', 'Random Forest', 'Hybrid']
metrics = [bilstm_metrics, rf_metrics, hybrid_metrics]

print("\nModel Comparison:")
print("Model\t\tAccuracy\tPrecision\tRecall\t\tF1-Score")

for model, (acc, prec, rec, f1) in zip(models, metrics):
    print(f"{model:15s}{acc:.4f}\t\t{prec:.4f}\t\t{rec:.4f}\t\t{f1:.4f}")

# =========================================
# STEP 8: PLOT GRAPH
# =========================================

accuracy = [m[0]*100 for m in metrics]
precision = [m[1]*100 for m in metrics]
recall = [m[2]*100 for m in metrics]
f1_score_vals = [m[3]*100 for m in metrics]

x = np.arange(len(models))
width = 0.2

plt.figure()

plt.bar(x - 1.5*width, accuracy, width, label='Accuracy')
plt.bar(x - 0.5*width, precision, width, label='Precision')
plt.bar(x + 0.5*width, recall, width, label='Recall')
plt.bar(x + 1.5*width, f1_score_vals, width, label='F1-Score')

plt.xlabel('Models')
plt.ylabel('Score (%)')
plt.title('Performance Comparison of Models')
plt.xticks(x, models)
plt.legend()

plt.tight_layout()

# SAVE IMAGE
plt.savefig("model_comparison.png", dpi=300)

plt.show()