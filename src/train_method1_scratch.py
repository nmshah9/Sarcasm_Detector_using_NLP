"""
train_method1_scratch.py
-------------------------
METHOD 1: Build sarcasm detection "from scratch":
  1. Deep text cleaning (stopword removal + lemmatization)
  2. Train our OWN Word2Vec word embeddings on the training corpus
  3. Train a BiLSTM classifier on top of those embeddings

Run:
    python src/train_method1_scratch.py

Outputs (saved to models/):
    - w2v_sarcasm.model         (trained Word2Vec embeddings)
    - tokenizer_scratch.pkl     (Keras tokenizer, word->index mapping)
    - bilstm_sarcasm.weights.h5 (trained BiLSTM weights — weights-only,
                                  for cross-Keras-version compatibility)
    - bilstm_architecture.json  (model architecture, rebuilt at load time)
    - method1_results.json      (train/test accuracy + metrics)
"""

import os
import json
import pickle
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_utils import load_splits, clean_text

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

MAX_VOCAB = 20000
MAX_LEN = 30          # headlines are short; 30 tokens comfortably covers ~99%
EMBED_DIM = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 2


# ---------------------------------------------------------------------------
# 1. Deep text cleaning (heavier than the shared base cleaning)
# ---------------------------------------------------------------------------
def deep_clean(text: str) -> str:
    text = clean_text(text)  # base cleaning from data_utils (lowercase, punctuation, urls)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def tokenize_for_w2v(text: str):
    return text.split()


# ---------------------------------------------------------------------------
# 2. Build Word2Vec embeddings + embedding matrix for Keras
# ---------------------------------------------------------------------------
def build_embedding_matrix(w2v_model, tokenizer, embed_dim):
    vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)
    embedding_matrix = np.zeros((vocab_size, embed_dim))
    hits, misses = 0, 0
    for word, idx in tokenizer.word_index.items():
        if idx >= vocab_size:
            continue
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
            hits += 1
        else:
            misses += 1
    print(f"Embedding coverage: {hits} hits / {misses} misses "
          f"({hits / (hits + misses):.1%} of vocab found in Word2Vec)")
    return embedding_matrix, vocab_size


# ---------------------------------------------------------------------------
# 3. Build the BiLSTM model
# ---------------------------------------------------------------------------
def build_bilstm_model(vocab_size, embed_dim, embedding_matrix, max_len):
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            weights=[embedding_matrix],
            input_length=max_len,
            trainable=True,  # fine-tune embeddings during training
        ),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dropout(0.4),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    print("=" * 70)
    print("METHOD 1: From-scratch (cleaning + Word2Vec + BiLSTM)")
    print("=" * 70)

    train_df, test_df = load_splits()

    print("Applying deep cleaning (stopwords + lemmatization)...")
    train_df["scratch_clean"] = train_df["clean_headline"].apply(deep_clean)
    test_df["scratch_clean"] = test_df["clean_headline"].apply(deep_clean)

    # --- Word2Vec trained ONLY on training data (avoid test leakage) ---
    print("Training Word2Vec embeddings on training corpus...")
    sentences = [tokenize_for_w2v(t) for t in train_df["scratch_clean"]]
    w2v_model = Word2Vec(
        sentences=sentences,
        vector_size=EMBED_DIM,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=4,
        seed=42,
    )
    w2v_model.save(os.path.join(MODEL_DIR, "w2v_sarcasm.model"))

    # --- Keras tokenizer (word -> integer index), fit on training data only ---
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["scratch_clean"])

    with open(os.path.join(MODEL_DIR, "tokenizer_scratch.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    X_train_seq = pad_sequences(
        tokenizer.texts_to_sequences(train_df["scratch_clean"]),
        maxlen=MAX_LEN, padding="post", truncating="post",
    )
    X_test_seq = pad_sequences(
        tokenizer.texts_to_sequences(test_df["scratch_clean"]),
        maxlen=MAX_LEN, padding="post", truncating="post",
    )
    y_train = train_df["is_sarcastic"].values
    y_test = test_df["is_sarcastic"].values

    embedding_matrix, vocab_size = build_embedding_matrix(w2v_model, tokenizer, EMBED_DIM)

    model = build_bilstm_model(vocab_size, EMBED_DIM, embedding_matrix, MAX_LEN)
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    history = model.fit(
        X_train_seq, y_train,
        validation_split=0.1,
        epochs=15,
        batch_size=64,
        callbacks=[early_stop],
        verbose=2,
    )

    # --- Save weights-only + architecture separately (cross-version safe) ---
    model.save_weights(os.path.join(MODEL_DIR, "bilstm_sarcasm.weights.h5"))
    with open(os.path.join(MODEL_DIR, "bilstm_architecture.json"), "w") as f:
        f.write(model.to_json())

    # --- Evaluate on BOTH train and test, as required ---
    train_pred = (model.predict(X_train_seq, verbose=0) > 0.5).astype(int).flatten()
    test_pred = (model.predict(X_test_seq, verbose=0) > 0.5).astype(int).flatten()

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("\n" + "=" * 70)
    print(f"TRAIN accuracy: {train_acc:.4f}")
    print(f"TEST  accuracy: {test_acc:.4f}")
    print("\nTest classification report:")
    print(classification_report(y_test, test_pred, target_names=["not_sarcastic", "sarcastic"]))
    print("Test confusion matrix:")
    print(confusion_matrix(y_test, test_pred))

    results = {
        "method": "Method 1: Scratch (Word2Vec + BiLSTM)",
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "vocab_size": int(vocab_size),
        "max_len": MAX_LEN,
        "embed_dim": EMBED_DIM,
    }
    with open(os.path.join(MODEL_DIR, "method1_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved model artifacts to: {MODEL_DIR}")
    print("Done.")


def load_method1_model():
    """Helper used by app.py / compare_models.py to reload the trained model."""
    with open(os.path.join(MODEL_DIR, "bilstm_architecture.json"), "r") as f:
        model = model_from_json(f.read())
    model.load_weights(os.path.join(MODEL_DIR, "bilstm_sarcasm.weights.h5"))

    with open(os.path.join(MODEL_DIR, "tokenizer_scratch.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


def predict_method1(text: str, model, tokenizer) -> float:
    """Returns sarcasm probability (0-1) for a single headline."""
    cleaned = deep_clean(clean_text(text))
    seq = pad_sequences(tokenizer.texts_to_sequences([cleaned]), maxlen=MAX_LEN,
                         padding="post", truncating="post")
    prob = model.predict(seq, verbose=0)[0][0]
    return float(prob)


if __name__ == "__main__":
    main()
