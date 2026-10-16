
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = BASE_DIR / "corpus.txt"

# --------------------------------------------------
# Load text
# --------------------------------------------------

with open(CORPUS_PATH, "r", encoding="utf-8") as file:
    text = file.read()

# --------------------------------------------------
# Tokenization
# --------------------------------------------------

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

vocab_size = len(tokenizer.word_index)

seq_list = []

for line in text.split("\n"):
    if line.strip():
        token_sentences = tokenizer.texts_to_sequences(
            [line.strip()]
        )[0]

        for i in range(1, len(token_sentences)):
            seq_list.append(token_sentences[:i + 1])

if not seq_list:
    raise ValueError(
        "No training sequences were created. "
        "Check that corpus.txt contains valid text."
    )

# --------------------------------------------------
# Padding
# --------------------------------------------------

max_seq_len = max(len(sequence) for sequence in seq_list)

padded_seq = pad_sequences(
    seq_list,
    maxlen=max_seq_len,
    padding="pre"
)

X = padded_seq[:, :-1]
Y = padded_seq[:, -1]

# --------------------------------------------------
# One-hot encoding
# --------------------------------------------------

Y = to_categorical(
    Y,
    num_classes=vocab_size + 1
)

# --------------------------------------------------
# Model
# --------------------------------------------------

model = Sequential()

model.add(
    Embedding(
        vocab_size + 1,
        100,
        input_length=X.shape[1]
    )
)

model.add(LSTM(100))

model.add(
    Dense(
        vocab_size + 1,
        activation="softmax"
    )
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------
# Training
# --------------------------------------------------

model.summary()

history = model.fit(
    X,
    Y,
    epochs=100
)

# --------------------------------------------------
# Save model
# --------------------------------------------------

model.save(ARTIFACTS_DIR / "word_predictor.keras")

# Save tokenizer
tokenizer_json = tokenizer.to_json()

with open(
    ARTIFACTS_DIR / "tokenizer.json",
    "w",
    encoding="utf-8"
) as file:
    file.write(tokenizer_json)

# Save configuration
config = {
    "vocab_size": vocab_size,
    "max_sequence_len": max_seq_len
}

with open(
    ARTIFACTS_DIR / "config.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(config, file, indent=4)

print("\nTraining completed successfully.")
print(f"Vocabulary size: {vocab_size}")
print(f"Maximum sequence length: {max_seq_len}")
print(f"Model saved to: {ARTIFACTS_DIR / 'word_predictor.keras'}")

