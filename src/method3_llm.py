import os
import json
import time
import argparse
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_utils import load_splits

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a text classification system. You will be given a news headline. "
    "Decide whether the headline is SARCASTIC (satirical, ironic, mocking — "
    "e.g. in the style of The Onion) or NOT SARCASTIC (a straightforward, "
    "literal news headline). "
    "Respond with EXACTLY one word: 'sarcastic' or 'not_sarcastic'. "
    "Do not explain your reasoning. Do not add punctuation."
)

FEW_SHOT_EXAMPLES = [
    ("thirtysomething scientists unveil doomsday clock of hair loss", "sarcastic"),
    ("dem rep. totally nails why congress is falling short on gender, racial equality", "not_sarcastic"),
    ("area man passionate defender of what he imagines constitution to be", "sarcastic"),
    ("scientists discover new species of frog in amazon rainforest", "not_sarcastic"),
]

def build_sample(sample_size=400):
    """Builds a stratified sample of the test set for LLM evaluation."""
    _, test_df = load_splits()
    sample, _ = train_test_split(
        test_df, train_size=sample_size, stratify=test_df["is_sarcastic"], random_state=42
    )
    return sample.reset_index(drop=True)

def build_user_prompt(headline: str) -> str:
    example_lines = "\n".join(
        [f'Headline: "{h}"\nAnswer: {label}' for h, label in FEW_SHOT_EXAMPLES]
    )
    return f"{example_lines}\n\nHeadline: \"{headline}\"\nAnswer:"

def parse_response(text: str) -> int:
    """Maps a free-text LLM response to 0 (not_sarcastic) or 1 (sarcastic)."""
    text = text.strip().lower()
    if "not_sarcastic" in text or "not sarcastic" in text or text.startswith("no"):
        return 0
    if "sarcastic" in text:
        return 1
    return 0

# ---------------------------------------------------------------------------
# Ollama (phi model only)
# ---------------------------------------------------------------------------
def call_ollama(headline: str, model: str = "phi", host: str = "http://localhost:11434") -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(headline)},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    try:
        resp = requests.post(f"{host}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.Timeout:
        return "[timeout]"
    except Exception as e:
        return f"[error: {e}]"

def evaluate_phi(sample_df: pd.DataFrame):
    print(f"\nEvaluating Ollama phi on {len(sample_df)} headlines...")
    preds = []
    for i, row in sample_df.iterrows():
        try:
            raw_response = call_ollama(row["headline"], model="phi")
            pred = parse_response(raw_response)
        except Exception as e:
            print(f"  [warn] row {i} failed ({e}); defaulting to 0")
            pred = 0
        preds.append(pred)
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(sample_df)} done")
        time.sleep(0.05)

    y_true = sample_df["is_sarcastic"].values
    acc = accuracy_score(y_true, preds)

    print(f"\n[ollama - phi] Accuracy on {len(sample_df)}-row sample: {acc:.4f}")
    print(classification_report(y_true, preds, target_names=["not_sarcastic", "sarcastic"]))
    print(confusion_matrix(y_true, preds))

    return {
        "method": "Method 3: LLM (ollama - phi)",
        "provider": "ollama",
        "model": "phi",
        "sample_size": len(sample_df),
        "sample_accuracy": float(acc),
        "note": "Evaluated on a stratified sample of the test set only.",
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=400)
    args = parser.parse_args()

    sample_df = build_sample(args.sample_size)
    results = evaluate_phi(sample_df)

    out_path = os.path.join(MODEL_DIR, "method3_phi_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    main()