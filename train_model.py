import requests
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from nltk import pos_tag
import nltk
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import roc_curve, auc, accuracy_score
import joblib  # Added for exporting the model
import re
import os

# --- 0. SETUP & DOWNLOADER ---
print(">>> Initializing...")
try:
    nltk.download("averaged_perceptron_tagger_eng")
except:
    pass
nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")
nltk.download("punkt_tab")

DATA_URLS = {
    "train": "https://raw.githubusercontent.com/spyysalo/ncbi-disease/master/conll/train.tsv",
    "dev": "https://raw.githubusercontent.com/spyysalo/ncbi-disease/master/conll/devel.tsv",
    "test": "https://raw.githubusercontent.com/spyysalo/ncbi-disease/master/conll/test.tsv",
}


def load_conll_data(url):
    print(f"Downloading data from {url}...")
    response = requests.get(url)
    response.raise_for_status()

    sentences = []
    current_sent = []
    for line in response.text.strip().split("\n"):
        line = line.strip()
        if not line:
            if current_sent:
                sentences.append(current_sent)
                current_sent = []
        else:
            parts = line.split("\t")
            if len(parts) >= 2:
                current_sent.append((parts[0], parts[-1]))
    if current_sent:
        sentences.append(current_sent)
    return sentences


# --- 1. FEATURE ENGINEERING ---
stemmer = PorterStemmer()


def get_orthographic_features(word):
    return {
        "is_title": word.istitle(),
        "is_all_caps": word.isupper(),
        "is_lower": word.islower(),
        "is_digit": word.isdigit(),
        "is_alnum": word.isalnum(),
        "has_dash": "-" in word,
        "has_slash": "/" in word,
        "has_greek": bool(
            re.search(r"(alpha|beta|gamma|delta|I|II|III|IV)", word, re.I)
        ),
    }


def sent2features(sent):
    words = [token for token, label in sent]
    try:
        pos_tags_list = pos_tag(words)
    except:
        # Fallback if nltk tagger fails
        pos_tags_list = [(w, "NN") for w in words]

    pos_tags = [pos for token, pos in pos_tags_list]

    features_list = []
    for i in range(len(sent)):
        word = sent[i][0]
        postag = pos_tags[i]

        features = {
            "bias": 1.0,
            "word.lower()": word.lower(),
            "word.stem": stemmer.stem(word),
            "postag": postag,
            "prefix-1": word[:1],
            "prefix-2": word[:2],
            "prefix-3": word[:3],
            "suffix-1": word[-1:],
            "suffix-2": word[-2:],
            "suffix-3": word[-3:],
            "suffix-4": word[-4:],
        }
        features.update(get_orthographic_features(word))

        if i > 0:
            word1 = sent[i - 1][0]
            features.update(
                {
                    "-1:word.lower()": word1.lower(),
                    "-1:postag": pos_tags[i - 1],
                    "-1:word.istitle()": word1.istitle(),
                }
            )
        else:
            features["BOS"] = True

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            features.update(
                {
                    "+1:word.lower()": word1.lower(),
                    "+1:postag": pos_tags[i + 1],
                    "+1:word.istitle()": word1.istitle(),
                }
            )
        else:
            features["EOS"] = True

        features_list.append(features)
    return features_list


def sent2labels(sent):
    return [label for token, label in sent]


# --- 2. DATA PROCESSING ---
train_sents = load_conll_data(DATA_URLS["train"])
test_sents = load_conll_data(DATA_URLS["test"])

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]
X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

# --- 3. TRAIN MODEL ---
print(">>> Training CRF Model...")
crf = sklearn_crfsuite.CRF(
    algorithm="lbfgs",
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
    verbose=True,
)
crf.fit(X_train, y_train)

# --- 4. EXPORT MODEL ---
print(">>> Saving model to 'disease_ner_model.pkl'...")
joblib.dump(crf, "disease_ner_model.pkl")

# --- 5. EVALUATION (Accuracy added) ---
print("\n>>> Predicting...")
y_pred = crf.predict(X_test)
labels = list(crf.classes_)
if "O" in labels:
    labels.remove("O")

# Flatten for accuracy calculation
y_test_flat = [item for sublist in y_test for item in sublist]
y_pred_flat = [item for sublist in y_pred for item in sublist]

acc = accuracy_score(y_test_flat, y_pred_flat)
print(f"\n>>> Global Accuracy: {acc:.4f}")

print("\n" + "=" * 40)
print("       CLASSIFICATION REPORT")
print("=" * 40)
sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
print(
    metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3)
)

# --- 6. ROC & AUC ---
print(">>> Calculating ROC/AUC...")
y_probs = crf.predict_marginals(X_test)
y_probs_flat = [token_probs for sent in y_probs for token_probs in sent]

# Binary classification: Disease vs O
y_test_binary = [1 if label != "O" else 0 for label in y_test_flat]
disease_tags = [tag for tag in crf.classes_ if tag != "O"]
y_score_binary = [sum(p.get(t, 0) for t in disease_tags) for p in y_probs_flat]

fpr, tpr, _ = roc_curve(y_test_binary, y_score_binary)
roc_auc = auc(fpr, tpr)
print(f">>> ROC AUC Score: {roc_auc:.4f}")
