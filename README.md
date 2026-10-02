# Sarcasm Detection Capstone Project

Detects whether a news headline is sarcastic using three different NLP approaches,
compares their accuracy, and serves the result through a Streamlit app.

- **Method 1 — From scratch:** text cleaning → self-trained Word2Vec embeddings → BiLSTM
- **Method 2 — Hugging Face:** fine-tuned pretrained `distilbert-base-uncased`
- **Method 3 — LLM:** zero/few-shot prompting via local Ollama and/or the OpenAI API

Dataset: [News Headlines Dataset for Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)
(`Sarcasm_Headlines_Dataset.json` and `Sarcasm_Headlines_Dataset_v2.json`).

---

## 1. Project setup in VS Code

```bash
# 1. Create the project folder and open it in VS Code
mkdir sarcasm-capstone
cd sarcasm-capstone
code .

# 2. Create and activate a virtual environment
python -m venv sarcasmenv

# Windows CMD:
sarcasmenv\Scripts\activate.bat
# macOS/Linux:
source sarcasmenv/bin/activate

# 3. In VS Code: Ctrl+Shift+P -> "Python: Select Interpreter" -> choose sarcasmenv

# 4. Install dependencies
pip install -r requirements.txt
```

Place `Sarcasm_Headlines_Dataset.json` and `Sarcasm_Headlines_Dataset_v2.json` in the
`data/` folder (already done for you in this delivered project).

> **If you already created a venv from an earlier copy of this project:** delete it and
> recreate it fresh before installing (`rmdir /s sarcasmenv` on Windows, then redo step 2).
> An older `requirements.txt` did not pin `scipy`, which on some machines pulls in a
> `scipy` version too new for `gensim` (`ImportError: cannot import name 'triu' from
> 'scipy.linalg'`). Reusing an old venv can leave that broken `scipy` in place even
> after you update `requirements.txt`, since pip won't always downgrade an already
> "satisfied" package. `requirements.txt` now pins `scipy==1.12.0` to fix this.

---

## 2. Step-by-step run order

Run everything from the **project root** (`sarcasm-capstone/`), not from inside `src/`.

### Step 0 — Explore the data (optional but recommended)
```bash
jupyter notebook notebooks/01_EDA.ipynb
```
Class balance, headline length distribution, and lexical differences between the two
classes — this is what motivates the preprocessing and modeling choices below.

### Step 1 — Build the shared 80/20 split
```bash
python src/data_utils.py
```
Creates `data/train.csv` and `data/test.csv`. **All three methods use these same files**
so the accuracy comparison is fair (same rows in train and test everywhere).

### Step 2 — Train Method 1 (Word2Vec + BiLSTM)
```bash
python src/train_method1_scratch.py
```
Trains your own Word2Vec embeddings on the training set, then a BiLSTM classifier
on top. Saves weights + tokenizer to `models/`. Takes a few minutes on CPU.

### Step 3 — Train Method 2 (fine-tune DistilBERT)
```bash
python src/train_method2_huggingface.py
```
Downloads pretrained DistilBERT weights (first run only, ~260MB) and fine-tunes
them on the training set. This is the slowest step on CPU — expect 20-60+ minutes
depending on your machine. If you have an NVIDIA GPU with CUDA installed, install
`torch` per [pytorch.org](https://pytorch.org/get-started/locally/) instead of the
CPU-only pin in requirements.txt, and this will be dramatically faster.

> **Note on TensorFlow/Keras conflict:** this project installs both `tensorflow-cpu`
> (for Method 1) and `transformers` (for Method 2) in the same environment.
> `transformers` auto-detects installed backends, and recent `tensorflow-cpu`
> versions ship Keras 3, which `transformers`' TensorFlow integration doesn't yet
> support — you may otherwise see `Your currently installed version of Keras is
> Keras 3, but this is not yet supported in Transformers`. Since Method 2 only
> uses PyTorch, `train_method2_huggingface.py` and `app.py` both set
> `os.environ["USE_TF"] = "0"` before importing `transformers`, which tells it to
> skip TensorFlow detection entirely. If you still see this error for any reason,
> either confirm you're running the version of these files from this delivery
> (not an older copy), or as a fallback run `pip install tf-keras`.

### Step 4 — Method 3 (LLM prompting) — optional setup first

**Ollama (free, local, no API key):**
```bash
# Install from https://ollama.com/download, then:
ollama pull llama3.1
```
Ollama runs a local server automatically after install (`localhost:11434`).

**OpenAI (paid API):**
```bash
pip install openai
# Windows CMD:
set OPENAI_API_KEY=sk-your-key-here
# macOS/Linux:
export OPENAI_API_KEY=sk-your-key-here
```

Then run:
```bash
python src/method3_llm.py --provider ollama --model llama3.1
python src/method3_llm.py --provider openai --model gpt-4o-mini
# or both in one run:
python src/method3_llm.py --provider both
```
This evaluates on a stratified 400-row sample of the test set (LLM calls are slow/costly,
so a full 5,701-row run isn't practical for a capstone iteration — see the MDD for the
statistical justification).

### Step 5 — Compare all methods
```bash
python src/compare_models.py
```
Reads the results JSON from each method you've run, prints a comparison table,
a recommendation, and saves `models/comparison_chart.png`.

### Step 6 — Launch the Streamlit app
```bash
streamlit run app.py
```
Opens a browser UI where you can type any headline and get predictions from
whichever method(s) you've trained, plus the comparison chart.

---

## 3. Deploying to Streamlit Community Cloud

1. Push this whole folder to a GitHub repo (include `models/` with trained artifacts).
2. **Size note:** the fine-tuned DistilBERT model (Method 2) is typically ~250MB, which
   exceeds GitHub's 100MB single-file limit. Options:
   - Use [Git LFS](https://git-lfs.com/) for the model files, **or**
   - Host the model on Hugging Face Hub and load it with
     `AutoModel.from_pretrained("your-username/sarcasm-distilbert")` in `app.py`
     instead of a local path (recommended — cleanest for deployment).
3. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at
   this repo and `app.py`.
4. Ollama won't be reachable from Streamlit Cloud (it's a local-only server), so on
   the deployed app, use the OpenAI provider for Method 3 and add `OPENAI_API_KEY`
   under the app's "Secrets" settings.

---

## 4. Project structure

```
sarcasm-capstone/
├── data/
│   ├── Sarcasm_Headlines_Dataset.json
│   ├── Sarcasm_Headlines_Dataset_v2.json
│   ├── train.csv                  (generated by data_utils.py)
│   └── test.csv                   (generated by data_utils.py)
├── notebooks/
│   └── 01_EDA.ipynb                (exploratory analysis — run this first)
├── src/
│   ├── data_utils.py               (shared loading/cleaning/splitting)
│   ├── train_method1_scratch.py    (Word2Vec + BiLSTM)
│   ├── train_method2_huggingface.py (fine-tuned DistilBERT)
│   ├── method3_llm.py              (Ollama + OpenAI prompting)
│   └── compare_models.py           (accuracy comparison + chart)
├── models/                         (empty until you train — see models/README.md)
├── app.py                          (Streamlit app)
├── requirements.txt
├── README.md
└── MDD_Sarcasm_Detection.docx      (Model Design Document)
```
