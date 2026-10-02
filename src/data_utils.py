"""
data_utils.py
--------------
Shared data loading, cleaning, and train/test splitting utilities.

IMPORTANT: All three modeling methods (scratch, HuggingFace, LLM) import
from this file so that every method is evaluated on the EXACT SAME
80/20 train/test split. Without this, comparing accuracy across methods
would not be a fair comparison.

Run this file directly once to generate data/train.csv and data/test.csv:
    python src/data_utils.py
"""

import json
import re
import string
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths (relative to project root — run scripts from the project root folder)
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_FILE_V2 = os.path.join(DATA_DIR, "Sarcasm_Headlines_Dataset_v2.json")
RAW_FILE_V1 = os.path.join(DATA_DIR, "Sarcasm_Headlines_Dataset.json")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_raw_json(path):
    """Loads a JSON-lines (one JSON object per line) sarcasm dataset."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return pd.DataFrame(records)


def load_combined_dataset(use_v2_only=True, dedupe=True):
    """
    Loads the sarcasm dataset(s).

    v2 is a superset/cleaned version of v1 (same source style, more rows,
    fixed key ordering). By default we use v2 only, since combining both
    and deduping is safer than risking leakage from near-duplicate rows
    across the two files ending up split across train and test.
    """
    df_v2 = load_raw_json(RAW_FILE_V2)

    if use_v2_only:
        df = df_v2
    else:
        df_v1 = load_raw_json(RAW_FILE_V1)
        df = pd.concat([df_v1, df_v2], ignore_index=True)

    df = df[["headline", "is_sarcastic", "article_link"]].copy()

    if dedupe:
        before = len(df)
        df = df.drop_duplicates(subset=["headline"]).reset_index(drop=True)
        after = len(df)
        print(f"Deduplicated: {before} -> {after} rows")

    df = df.dropna(subset=["headline"]).reset_index(drop=True)
    return df


def clean_text(text: str) -> str:
    """
    Basic, model-agnostic text cleaning:
    - lowercase
    - remove URLs (shouldn't be in headlines, but safe to strip)
    - remove punctuation
    - collapse extra whitespace

    NOTE: We deliberately do NOT remove stopwords or lemmatize here in the
    shared utility, because:
      - HuggingFace transformer tokenizers work best on natural,
        un-stripped text (they have their own subword tokenization).
      - The "from scratch" method applies its OWN heavier cleaning
        (stopword removal, lemmatization) on top of this base cleaning,
        since classic embeddings benefit more from that.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_splits(save=True):
    """Builds the 80/20 stratified train/test split and optionally saves CSVs."""
    df = load_combined_dataset()
    df["clean_headline"] = df["headline"].apply(clean_text)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["is_sarcastic"],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Total rows: {len(df)}")
    print(f"Train rows: {len(train_df)}  | sarcastic %: {train_df['is_sarcastic'].mean():.3f}")
    print(f"Test rows:  {len(test_df)}  | sarcastic %: {test_df['is_sarcastic'].mean():.3f}")

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        train_df.to_csv(TRAIN_CSV, index=False)
        test_df.to_csv(TEST_CSV, index=False)
        print(f"Saved: {TRAIN_CSV}")
        print(f"Saved: {TEST_CSV}")

    return train_df, test_df


def load_splits():
    """Loads the pre-built train/test CSVs (call build_splits() first)."""
    if not (os.path.exists(TRAIN_CSV) and os.path.exists(TEST_CSV)):
        print("Splits not found — building them now...")
        return build_splits()
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    return train_df, test_df


if __name__ == "__main__":
    build_splits()
