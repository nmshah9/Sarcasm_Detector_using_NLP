# models/

This folder is intentionally empty in the delivered project — it fills up automatically
once you run the training scripts on your own machine:

| Run this... | ...and this appears here |
|---|---|
| `python src/train_method1_scratch.py` | `w2v_sarcasm.model`, `tokenizer_scratch.pkl`, `bilstm_sarcasm.weights.h5`, `bilstm_architecture.json`, `method1_results.json` |
| `python src/train_method2_huggingface.py` | `distilbert_sarcasm/` (fine-tuned model + tokenizer), `method2_results.json` |
| `python src/method3_llm.py --provider both` | `method3_results.json` |
| `python src/compare_models.py` | `comparison_chart.png` |

These files are not pre-generated because training the BiLSTM and fine-tuning DistilBERT
depend on your local hardware (CPU vs GPU) and can take anywhere from minutes to over an
hour — see the root README.md for expected run times and setup steps.
