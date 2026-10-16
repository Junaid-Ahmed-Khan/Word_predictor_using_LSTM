# LSTM Next-Word Prediction (TensorFlow/Keras)

A small end-to-end next-word prediction project with an LSTM and a Streamlit UI.

## Requirements

- Python 3.10–3.12 recommended for broad TensorFlow compatibility
- Internet access for installing packages

## Windows setup

Open Command Prompt or the VS Code terminal in this project folder:

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py` is unavailable, use `python` instead. If TensorFlow fails to install on your Python version, install a supported Python version (such as 3.11) and recreate the virtual environment.

## Train the model

```bash
python train.py
```

This creates:
- `artifacts/word_predictor.keras`
- `artifacts/tokenizer.json`
- `artifacts/config.json`

## Launch the web app

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Use your own text

Edit `corpus.txt`. Put one sentence or text line per line. More varied, clean text generally leads to better predictions. Then rerun `python train.py`.

The included corpus is only a tiny demonstration dataset; it is not large enough for reliable general-purpose language prediction. Use a substantially larger, legally usable corpus for meaningful results.

## How it works

1. Tokenizer maps words to integer IDs.
2. Every line is converted into prefix-to-next-word training examples.
3. Padding makes the input sequences the same length.
4. Embedding maps word IDs to vectors.
5. LSTM learns sequence context.
6. Dense + softmax predicts the next token.
7. The model and tokenizer are saved and loaded by the Streamlit app.
