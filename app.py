
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = BASE_DIR / "artifacts"

app = Flask(__name__)

# Load model and tokenizer
MODEL_PATH = ARTIFACTS / "word_predictor.keras"
TOKENIZER_PATH = ARTIFACTS / "tokenizer.json"
CONFIG_PATH = ARTIFACTS / "config.json"

MODEL = tf.keras.models.load_model(MODEL_PATH)

TOKENIZER = tokenizer_from_json(
    TOKENIZER_PATH.read_text(encoding="utf-8")
)

CONFIG = json.loads(
    CONFIG_PATH.read_text(encoding="utf-8")
)

MAX_SEQUENCE_LEN = int(CONFIG["max_sequence_len"])

INDEX_TO_WORD = {
    index: word
    for word, index in TOKENIZER.word_index.items()
}


def predict_next_words(text, top_k=5):
    sequence = TOKENIZER.texts_to_sequences(
        [text.lower().strip()]
    )[0]

    if not sequence:
        return []

    sequence = sequence[-(MAX_SEQUENCE_LEN - 1):]

    padded = pad_sequences(
        [sequence],
        maxlen=MAX_SEQUENCE_LEN - 1,
        padding="pre",
        truncating="pre"
    )

    probabilities = MODEL.predict(padded, verbose=0)[0].copy()

    oov_id = TOKENIZER.word_index.get("<OOV>")

    if oov_id is not None and oov_id < len(probabilities):
        probabilities[oov_id] = 0.0

    results = []

    for token_id in np.argsort(probabilities)[::-1]:
        word = INDEX_TO_WORD.get(int(token_id))

        if word and word != "<OOV>":
            results.append({
                "word": word,
                "score": float(probabilities[token_id])
            })

        if len(results) >= top_k:
            break

    return results


@app.route("/", methods=["GET", "POST"])
def home():
    prompt = "deep learning is"
    predictions = []
    error = None
    top_k = 5

    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()

        try:
            top_k = int(request.form.get("top_k", 5))
            top_k = min(max(top_k, 1), 10)
        except (ValueError, TypeError):
            top_k = 5

        if not prompt:
            error = "Please enter some text."
        else:
            try:
                predictions = predict_next_words(prompt, top_k)

                if not predictions:
                    error = (
                        "No known words found. "
                        "Try a phrase from your training data."
                    )

            except Exception as exc:
                app.logger.exception("Prediction failed")
                error = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        prompt=prompt,
        predictions=predictions,
        error=error,
        top_k=top_k
    )


if __name__ == "__main__":
    print("App location:", Path(__file__).resolve())
    print("Template folder:", app.template_folder)

    template_path = BASE_DIR / "templates" / "index.html"
    print("HTML file exists:", template_path.is_file())

    if template_path.is_file():
        print("HTML file size:", template_path.stat().st_size)

    app.run(debug=True)
