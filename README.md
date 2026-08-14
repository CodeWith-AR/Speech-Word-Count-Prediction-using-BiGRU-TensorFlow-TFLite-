

# 🎙️ Speech Word Count Prediction using BiGRU (TensorFlow + TFLite)

A deep learning project that predicts the **number of words spoken in an audio clip** using MFCC-based audio features and a Bidirectional GRU neural network.

This project includes:

- ✅ Audio feature extraction (MFCC + Delta + Delta-Delta)
- ✅ Sequence modeling using Bidirectional GRU
- ✅ Label normalization
- ✅ Early stopping training
- ✅ Model export to `.keras`
- ✅ TensorFlow Lite conversion for mobile/edge deployment

---

## 📌 Project Objective

The goal of this project is:

> 🎯 Predict the number of spoken words directly from raw speech audio.

Instead of performing full speech-to-text transcription, this model estimates word count using acoustic patterns such as speech duration and rhythm.

---

## 📂 Dataset

**Dataset Used:** LibriTTS (dev-clean subset)

- Sample Rate: 16 kHz
- Format: `.wav`
- Transcripts: `.trans.tsv`
- Total Samples Used: ~5716

Dataset structure:

```
LibriTTS/
    speaker_id/
        chapter_id/
            audio.wav
            transcript.trans.tsv
```

---

## 🧠 Feature Engineering

For each audio file:

1. Extract 13 MFCC features
2. Compute:
   - Delta (1st derivative)
   - Delta-Delta (2nd derivative)
3. Stack features:

```
13 MFCC
13 Delta
13 Delta-Delta
----------------
39 Features per time step
```

4. Pad or trim to fixed length:

```
MAX_LEN = 200 time steps
```

Final input shape:

```
(200, 39)
```

---

## 🏗️ Model Architecture

```
Input (200, 39)
↓
Masking Layer
↓
Bidirectional GRU (128 units)
↓
Dense (64, ReLU)
↓
Dense (1) → Word Count (Normalized)
```

### Why Bidirectional GRU?

- Speech depends on temporal patterns.
- Bidirectional GRU processes audio both forward and backward.
- Improves context understanding.

Total Parameters:

```
145,537 (~568 KB)
```

---

## ⚙️ Training Configuration

- Optimizer: Adam
- Loss Function: Huber Loss
- Batch Size: 32
- Epochs: 50 (EarlyStopping applied)
- Validation Split: 20%

### Label Normalization

Targets are standardized before training:

```
y_norm = (y - mean) / std
```

Predictions are de-normalized after inference.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| MAE    | ~5.7 words |
| RMSE   | ~8.0 words |

✅ Significant improvement over baseline (~7.3 MAE)

Example predictions:

```
True: 15 → Pred: 12.86
True: 37 → Pred: 27.52
```

---

## 📦 Model Files

| File | Description |
|------|-------------|
| `improved_model.keras` | Saved Keras model |
| `improved_saved_model/` | TensorFlow SavedModel |
| `improved_model.tflite` | TensorFlow Lite model |

---

# 🚀 Step‑by‑Step Setup Instructions

---

## ✅ 1️⃣ Clone Repository

```bash
git clone https://github.com/CodeWith-AR/Speech-Word-Count-Prediction-using-BiGRU-TensorFlow-TFLite-.git

cd YOUR_REPO
```

---

## ✅ 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install librosa tensorflow pandas scikit-learn
```

---

## ✅ 3️⃣ Download Dataset

Download LibriTTS dev-clean subset from:

https://www.openslr.org/60/

Extract into:

```
LibriTTS/dev-clean
```

---

## ✅ 4️⃣ Train Model

Run:

```bash
python train.py
```

This will:

- Process dataset
- Train model
- Save `.keras` file
- Convert to `.tflite`

---

## ✅ 5️⃣ Evaluate Model

```bash
python evaluate.py
```

This will print:

- MAE
- RMSE
- Sample predictions

---

## ✅ 6️⃣ Convert to TFLite (Already Included)

The training script automatically generates:

```
improved_model.tflite
```

For mobile deployment.

---

# 📱 Using TFLite Model

Example inference:

```python
import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="improved_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

sample = np.expand_dims(X_test[0], axis=0).astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], sample)
interpreter.invoke()

prediction = interpreter.get_tensor(output_details[0]['index'])
```

---

# 📁 Recommended Repository Structure

```
speech-wordcount/
│
├── LibriTTS/
├── train.py
├── evaluate.py
├── improved_model.keras
├── improved_model.tflite
├── requirements.txt
└── README.md
```

---

# 🔬 Limitations

- Model predicts word count, not transcription.
- Performance depends on speaking speed.
- Long sentences may still produce errors.
- Requires SELECT_TF_OPS for TFLite due to GRU.

---

# 🔮 Future Improvements

- ✅ Add speech duration as explicit feature
- ✅ Add silence ratio feature
- ✅ CNN + BiGRU hybrid architecture
- ✅ Pure TFLite model (without Flex ops)
- ✅ Convert to full speech-to-text (CTC model)

---

# 📜 License

This project is licensed under the MIT License.

---

# 🙌 Author

M.Abdur rehman
Developed as part of a deep learning exploration project using TensorFlow and audio processing techniques.

---

# ⭐ If You Found This Useful

Please consider starring the repository.

---

