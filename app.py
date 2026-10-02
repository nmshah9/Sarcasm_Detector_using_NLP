"""
app.py
-------
Streamlit app for the Sarcasm Detection Capstone.

Lets a user type any headline and see predictions from:
  - Method 1: Word2Vec + BiLSTM (from scratch)
  - Method 2: Fine-tuned DistilBERT (Hugging Face)
  - Method 3: LLM zero/few-shot (Ollama local, or OpenAI if a key is provided)

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    - Push this whole project folder to a GitHub repo
    - On share.streamlit.io, point to app.py
    - IMPORTANT: the models/ folder (BiLSTM weights + fine-tuned DistilBERT)
      must be committed too, OR hosted externally and downloaded at startup
      if they exceed GitHub's 100MB file limit (DistilBERT fine-tuned
      weights are typically ~250MB — see README "Deployment size note").
"""

import os
import sys
import json

# See src/train_method2_huggingface.py for why this is required: it must be set
# before `transformers` is imported anywhere in this process (Streamlit re-runs
# this whole script on every interaction, so setting it here as early as
# possible covers every code path, regardless of which method button is clicked
# first).
os.environ.setdefault("USE_TF", "0")

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

st.set_page_config(page_title="Sarcasm Detector", page_icon="🙃", layout="centered")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

st.title("🙃 Sarcasm Detection — Capstone Demo")
st.caption(
    "Compare three approaches to detecting sarcastic news headlines: "
    "a from-scratch deep learning model, a fine-tuned transformer, and an LLM."
)

headline = st.text_area(
    "Enter a headline to classify:",
    value="area man passionate defender of what he imagines constitution to be",
    height=80,
)

method = st.radio(
    "Choose a method:",
    [
        "Method 1: Word2Vec + BiLSTM (from scratch)",
        "Method 2: Fine-tuned DistilBERT (Hugging Face)",
        "Method 3: LLM (Ollama)",
    ],
)

# ---------------------------------------------------------------------------
# Cached model loaders — each is only loaded once per session, not per click
# ---------------------------------------------------------------------------
@st.cache_resource
def get_method1():
    from train_method1_scratch import load_method1_model
    return load_method1_model()


@st.cache_resource
def get_method2():
    from train_method2_huggingface import load_method2_model
    return load_method2_model()


def show_verdict(prob: float):
    label = "SARCASTIC 🙃" if prob >= 0.5 else "NOT SARCASTIC 🙂"
    st.subheader(label)
    st.progress(min(max(prob, 0.0), 1.0))
    st.write(f"Sarcasm probability: **{prob:.1%}**")


if st.button("Classify", type="primary"):
    if not headline.strip():
        st.warning("Please enter a headline.")
    elif method.startswith("Method 1"):
        try:
            from train_method1_scratch import predict_method1
            model, tokenizer = get_method1()
            prob = predict_method1(headline, model, tokenizer)
            show_verdict(prob)
        except FileNotFoundError:
            st.error(
                "Method 1 model artifacts not found. Run "
                "`python src/train_method1_scratch.py` first to train and save the model."
            )

    elif method.startswith("Method 2"):
        try:
            from train_method2_huggingface import predict_method2
            model, tokenizer = get_method2()
            prob = predict_method2(headline, model, tokenizer)
            show_verdict(prob)
        except OSError:
            st.error(
                "Method 2 fine-tuned model not found. Run "
                "`python src/train_method2_huggingface.py` first to fine-tune and save the model."
            )

    elif method.startswith("Method 3"):
     st.info("Calling the LLM — this may take a few seconds...")
    try:
        from method3_llm import call_ollama, parse_response, build_user_prompt
        # Always use Ollama phi
        raw = call_ollama(headline, model="phi")
        pred = parse_response(raw)

        # Show result
        st.subheader("SARCASTIC 🙃" if pred == 1 else "NOT SARCASTIC 🙂")
        st.caption(f"Raw model response: `{raw.strip()}`")

    except Exception as e:
        st.error(
            f"LLM call failed: {e}\n\n"
            "Make sure Ollama is running locally (`ollama serve`) and that the `phi` model is pulled."
        )
    


# --- LLM provider picker, shown only when relevant ---
if method.startswith("Method 3"):
    # Only Ollama is available now
    st.session_state["llm_provider"] = "ollama"

st.divider()



# ---------------------------------------------------------------------------
# Model comparison section (reads compare_models.py output if available)
# ---------------------------------------------------------------------------
st.subheader("📊 Model Comparison")

chart_path = os.path.join(MODEL_DIR, "comparison_chart.png")
if os.path.exists(chart_path):
    st.image(chart_path, caption="Train/Test accuracy across all three methods")
else:
    st.caption(
        "Run `python src/compare_models.py` after training the models to generate "
        "and display the comparison chart here."
    )

for fname, label in [
    ("method1_results.json", "Method 1 (Scratch)"),
    ("method2_results.json", "Method 2 (Hugging Face)"),
    ("method3_phi_results.json", "Method 3 (LLM)"),
]:
    path = os.path.join(MODEL_DIR, fname)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        with st.expander(f"{label} — raw results"):
            st.json(data)
