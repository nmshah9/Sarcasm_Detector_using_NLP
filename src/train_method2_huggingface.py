"""
train_method2_huggingface.py
------------------------------
METHOD 2: Take a pre-trained Hugging Face transformer (DistilBERT) and
fine-tune it on our sarcasm dataset (transfer learning).

Why DistilBERT: it's 40% smaller / ~60% faster than BERT-base with ~97% of
its language understanding, which matters here because we're fine-tuning
on CPU-friendly hardware and need something that trains in a reasonable
time for a capstone project. Swap MODEL_NAME below for "bert-base-uncased"
or "roberta-base" if you have a GPU and want to push accuracy further.

Run:
    python src/train_method2_huggingface.py

Outputs (saved to models/distilbert_sarcasm/):
    - Fine-tuned model + tokenizer (HF `save_pretrained` format)
    - method2_results.json (train/test accuracy + metrics)
"""

import os

# Force transformers to use ONLY the PyTorch backend, never TensorFlow.
# This MUST be set before `transformers` is imported anywhere in this process.
# Why: this project also installs tensorflow-cpu (for Method 1's BiLSTM), and
# transformers auto-detects installed backends. Recent tensorflow-cpu versions
# ship Keras 3, which transformers' TensorFlow integration does not yet support
# ("Your currently installed version of Keras is Keras 3, but this is not yet
# supported in Transformers"). Since Method 2 only ever uses PyTorch, we tell
# transformers to skip TensorFlow detection entirely rather than requiring the
# tf-keras backwards-compatibility shim.
os.environ.setdefault("USE_TF", "0")

import json
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_utils import load_splits

MODEL_NAME = "distilbert-base-uncased"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "distilbert_sarcasm")
os.makedirs(MODEL_DIR, exist_ok=True)

MAX_LEN = 64
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def main():
    print("=" * 70)
    print(f"METHOD 2: Hugging Face pretrained model — fine-tuning {MODEL_NAME}")
    print("=" * 70)

    train_df, test_df = load_splits()

    # Use un-lemmatized, lightly cleaned text — transformer tokenizers handle
    # raw natural language better than heavily preprocessed text.
    train_df = train_df.rename(columns={"is_sarcastic": "label"})
    test_df = test_df.rename(columns={"is_sarcastic": "label"})

    train_ds = Dataset.from_pandas(train_df[["clean_headline", "label"]])
    test_ds = Dataset.from_pandas(test_df[["clean_headline", "label"]])

    print("Loading tokenizer + pretrained model weights...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    def tokenize_fn(batch):
        return tokenizer(batch["clean_headline"], truncation=True, padding="max_length", max_length=MAX_LEN)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=100,
        report_to="none",
        fp16=torch.cuda.is_available(),  # use mixed precision if a GPU is present
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    print("Fine-tuning... (this is the slow step — expect longer runtime on CPU)")
    trainer.train()

    # --- Evaluate on BOTH train and test, as required ---
    train_pred_out = trainer.predict(train_ds)
    test_pred_out = trainer.predict(test_ds)

    train_preds = np.argmax(train_pred_out.predictions, axis=-1)
    test_preds = np.argmax(test_pred_out.predictions, axis=-1)

    train_acc = accuracy_score(train_df["label"], train_preds)
    test_acc = accuracy_score(test_df["label"], test_preds)

    print("\n" + "=" * 70)
    print(f"TRAIN accuracy: {train_acc:.4f}")
    print(f"TEST  accuracy: {test_acc:.4f}")
    print("\nTest classification report:")
    print(classification_report(test_df["label"], test_preds, target_names=["not_sarcastic", "sarcastic"]))
    print("Test confusion matrix:")
    print(confusion_matrix(test_df["label"], test_preds))

    # --- Save the fine-tuned model + tokenizer for reuse in app.py ---
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    results = {
        "method": f"Method 2: Hugging Face fine-tuned ({MODEL_NAME})",
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "model_name": MODEL_NAME,
        "epochs": NUM_EPOCHS,
    }
    with open(os.path.join(MODEL_DIR, "..", "method2_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved fine-tuned model to: {MODEL_DIR}")
    print("Done.")


def load_method2_model():
    """Helper used by app.py / compare_models.py to reload the fine-tuned model."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return model, tokenizer


def predict_method2(text: str, model, tokenizer) -> float:
    """Returns sarcasm probability (0-1) for a single headline."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    return float(probs[0][1])  # probability of class 1 (sarcastic)


if __name__ == "__main__":
    main()
