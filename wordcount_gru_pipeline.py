# ============================================================
# WORD-COUNT-FROM-AUDIO — GRU MODEL (CLEANED COLAB PIPELINE)
# ============================================================
# This merges your 3 iterations into ONE correct pipeline:
#   - dataset download & extraction
#   - MFCC + delta + delta2 feature extraction (39-dim, seq len 200)
#   - Bidirectional GRU regression model with label normalization
#   - Keras save -> SavedModel export -> TFLite conversion
#   - TFLite inference + evaluation (with correct de-normalization)
#   - Colab download helpers
#
# Fixes vs. your original notebook:
#   - Kept only the IMPROVED model (MFCC+delta+delta2, Bi-GRU,
#     normalized labels) — the earlier plain-MFCC model is redundant.
#   - Re-added `unroll=True` on the GRU (present in v1's "TFLite
#     compatible" comment, missing from the improved v2 model) —
#     helps TFLite conversion avoid TensorList ops.
#   - Fixed the TFLite evaluation step: it now de-normalizes
#     predictions (`pred * y_std + y_mean`) before computing MAE.
#     Your original eval cell was written for the un-normalized
#     v1 model and would've given wrong MAE numbers on v2's output.
#   - One consistent naming scheme throughout instead of 3 different
#     sets of filenames (gru_speech_model / improved_model /
#     final_wordcount_model).
#   - Removed duplicate pip installs / imports across cells.
#
# Paste each "Cell" block into its own Colab cell, or run top-to-bottom
# as a single script.
# ============================================================

# --- Cell 1: install dependencies -----------------------------------------
!pip install -q gdown librosa tensorflow pandas scikit-learn

# --- Cell 2: download & extract dataset ------------------------------------
import gdown
import tarfile
import os

file_id = "1YFnSYLe2kpPmIs8x3NxFWZD01YNEU8gV"
url = f"https://drive.google.com/uc?id={file_id}"
archive_path = "dev-clean.tar.gz"

gdown.download(url, archive_path, quiet=False)

with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall()

print("✅ Dataset extracted")

DATA_DIR = "LibriTTS/dev-clean"
print("📂 Sample speaker folders:", os.listdir(DATA_DIR)[:5])

# --- Cell 3: imports --------------------------------------------------------
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Masking, GRU, Bidirectional, Dense
from tensorflow.keras.callbacks import EarlyStopping

# --- Cell 4: feature extraction ----------------------------------------------
SR = 16000
N_MFCC = 13
MAX_LEN = 200
FEATURES = N_MFCC * 3  # MFCC + delta + delta2

X, y = [], []

print("🔄 Processing dataset (this can take a while)...")

for speaker in os.listdir(DATA_DIR):
    speaker_path = os.path.join(DATA_DIR, speaker)
    if not os.path.isdir(speaker_path):
        continue

    for chapter in os.listdir(speaker_path):
        chapter_path = os.path.join(speaker_path, chapter)
        if not os.path.isdir(chapter_path):
            continue

        trans_file = next(
            (os.path.join(chapter_path, f) for f in os.listdir(chapter_path)
             if f.endswith(".trans.tsv")),
            None
        )
        if trans_file is None:
            continue

        try:
            df = pd.read_csv(trans_file, sep="\t", header=None)
            if df.shape[1] == 3:
                df.columns = ["id", "original_text", "normalized_text"]
            elif df.shape[1] == 2:
                df.columns = ["id", "normalized_text"]
            else:
                continue
        except Exception:
            continue

        for _, row in df.iterrows():
            file_id_ = str(row["id"])
            text = str(row["normalized_text"])
            wav_path = os.path.join(chapter_path, file_id_ + ".wav")

            if not os.path.exists(wav_path):
                continue

            try:
                audio, sr = librosa.load(wav_path, sr=SR)

                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
                delta = librosa.feature.delta(mfcc)
                delta2 = librosa.feature.delta(mfcc, order=2)
                feats = np.vstack([mfcc, delta, delta2]).T  # (time, 39)

                if len(feats) < MAX_LEN:
                    pad = np.zeros((MAX_LEN - len(feats), FEATURES))
                    feats = np.vstack((feats, pad))
                else:
                    feats = feats[:MAX_LEN]

                X.append(feats)
                y.append(len(text.split()))

            except Exception:
                continue

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
print("✅ Dataset ready:", X.shape, y.shape)

# --- Cell 5: train/test split + label normalization --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_mean, y_std = y_train.mean(), y_train.std()
y_train_norm = (y_train - y_mean) / y_std
y_test_norm = (y_test - y_mean) / y_std

# --- Cell 6: build model -------------------------------------------------------
model = Sequential([
    Input(shape=(MAX_LEN, FEATURES)),
    Masking(mask_value=0.0),
    Bidirectional(
        GRU(
            128,
            activation="tanh",
            recurrent_activation="sigmoid",
            reset_after=False,  # TFLite-friendly
            unroll=True,        # TFLite-friendly, avoids TensorList ops
        )
    ),
    Dense(64, activation="relu"),
    Dense(1, activation="linear"),
])

model.compile(optimizer="adam", loss=tf.keras.losses.Huber(), metrics=["mae"])
model.summary()

# --- Cell 7: train --------------------------------------------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train_norm,
    validation_data=(X_test, y_test_norm),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
)

# --- Cell 8: evaluate (Keras) ----------------------------------------------------
pred_norm = model.predict(X_test)
pred = pred_norm.flatten() * y_std + y_mean

mae = np.mean(np.abs(pred - y_test))
print("📊 Keras model MAE (word count):", mae)

print("\n🔎 Sample predictions:")
for i in range(5):
    print(f"True: {y_test[i]:.1f} | Predicted: {pred[i]:.2f}")

# --- Cell 9: save / export / convert ----------------------------------------------
MODEL_NAME = "wordcount_gru_model"

model.save(f"{MODEL_NAME}.keras")
print(f"✅ Keras model saved as {MODEL_NAME}.keras")

model.export(f"{MODEL_NAME}_savedmodel")
print(f"✅ SavedModel exported to {MODEL_NAME}_savedmodel/")

converter = tf.lite.TFLiteConverter.from_saved_model(f"{MODEL_NAME}_savedmodel")
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open(f"{MODEL_NAME}.tflite", "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite model saved as {MODEL_NAME}.tflite")

# --- Cell 10: TFLite inference + evaluation ----------------------------------------
interpreter = tf.lite.Interpreter(model_path=f"{MODEL_NAME}.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ Input shape expected:", input_details[0]["shape"])
print("✅ Output shape:", output_details[0]["shape"])

tflite_preds_norm = []
for i in range(len(X_test)):
    sample = np.expand_dims(X_test[i], axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], sample)
    interpreter.invoke()
    tflite_preds_norm.append(interpreter.get_tensor(output_details[0]["index"])[0][0])

tflite_preds_norm = np.array(tflite_preds_norm)
# IMPORTANT: de-normalize before comparing to raw y_test — this step
# was missing in the original eval cell.
tflite_preds = tflite_preds_norm * y_std + y_mean

tflite_mae = np.mean(np.abs(tflite_preds - y_test))
print("📊 TFLite model MAE (word count):", tflite_mae)

print("\n🔎 Sample TFLite predictions:")
for i in range(5):
    print(f"True: {y_test[i]:.1f} | Predicted: {tflite_preds[i]:.2f}")

# --- Cell 11: download from Colab (optional) -----------------------------------------
from google.colab import files

files.download(f"{MODEL_NAME}.keras")
files.download(f"{MODEL_NAME}.tflite")

# --- Cell 12: push to GitHub (optional template) --------------------------------------
# Uncomment and fill in your details to push the saved model + this script to a repo.
# Use a GitHub Personal Access Token (classic or fine-grained) instead of a password.
#
# !git config --global user.email "you@example.com"
# !git config --global user.name "Your Name"
#
# !git clone https://<TOKEN>@github.com/<username>/<repo>.git
# !cp wordcount_gru_model.keras wordcount_gru_model.tflite <repo>/
# %cd <repo>
# !git add .
# !git commit -m "Add trained word-count GRU model"
# !git push
