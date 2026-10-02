"""
compare_models.py
-------------------
Aggregates the results JSON files produced by each method and prints/plots
a side-by-side comparison, then prints a recommendation.

Run this AFTER you have run at least method 1 and method 2 (method 3 is
optional if you haven't set up Ollama/OpenAI yet):
    python src/compare_models.py
"""

import os
import json
import matplotlib.pyplot as plt

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_json_safe(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def main():
    rows = []

    # --- Method 1 ---
    m1 = load_json_safe(os.path.join(MODEL_DIR, "method1_results.json"))
    if m1:
        rows.append((
            "Method 1: Scratch\n(Word2Vec + BiLSTM)",
            m1.get("train_accuracy"),
            m1.get("test_accuracy"),
        ))

    # --- Method 2 ---
    m2 = load_json_safe(os.path.join(MODEL_DIR, "method2_results.json"))
    if m2:
        rows.append((
            "Method 2: Hugging Face\n(fine-tuned DistilBERT)",
            m2.get("train_accuracy"),
            m2.get("test_accuracy"),
        ))

    # --- Method 3 ---
    m3_data = load_json_safe(os.path.join(MODEL_DIR, "method3_phi_results.json"))
    if m3_data:
        # Normalize: wrap dict in list if needed
        if isinstance(m3_data, dict):
            m3_list = [m3_data]
        else:
            m3_list = m3_data

        for m3 in m3_list:
            label = f"Method 3: LLM\n({m3.get('provider')} - {m3.get('model')})"
            # LLMs are zero/few-shot — no train accuracy, so plot sample accuracy in both slots
            rows.append((label, None, m3.get("sample_accuracy")))

    if not rows:
        print("No results found yet. Run the training scripts for at least one method first.")
        return

    # --- Print table ---
    print("\n" + "=" * 70)
    print(f"{'Method':45s} {'Train Acc':>10s} {'Test/Sample Acc':>18s}")
    print("-" * 70)
    for label, train_acc, test_acc in rows:
        clean_label = label.replace("\n", " ")
        train_str = f"{train_acc:.4f}" if train_acc is not None else "n/a"
        print(f"{clean_label:45s} {train_str:>10s} {test_acc:>18.4f}")
    print("=" * 70)

    # --- Recommendation logic ---
    best = max(rows, key=lambda r: r[2])
    print(f"\nRECOMMENDATION (by test/sample accuracy): {best[0].replace(chr(10), ' ')}"
          f" — {best[2]:.4f}")
    print(
        "\nNote: accuracy alone doesn't capture the full picture — see the MDD "
        "document for a discussion of latency, cost, interpretability, and "
        "deployment considerations that factor into the final recommendation."
    )

    # --- Bar chart ---
    labels = [r[0] for r in rows]
    train_vals = [r[1] if r[1] is not None else 0 for r in rows]
    test_vals = [r[2] for r in rows]
    has_train = [r[1] is not None for r in rows]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    train_bars = ax.bar([i - width / 2 for i in x], train_vals, width, label="Train Accuracy")
    test_bars = ax.bar([i + width / 2 for i in x], test_vals, width, label="Test / Sample Accuracy")

    # Grey out train bars where train accuracy doesn't apply (LLM methods)
    for bar, applicable in zip(train_bars, has_train):
        if not applicable:
            bar.set_alpha(0.15)

    ax.set_ylabel("Accuracy")
    ax.set_title("Sarcasm Detection: Method Comparison")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out_path = os.path.join(MODEL_DIR, "comparison_chart.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved comparison chart to: {out_path}")


if __name__ == "__main__":
    main()
