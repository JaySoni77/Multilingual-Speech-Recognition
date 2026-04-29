# 🎙️ Multilingual Speech Recognition — Whisper Fine-Tuning

Fine-tuning OpenAI's **Whisper Small** model for multilingual speech recognition using the Hugging Face ecosystem. Built for the [Kaggle Multilingual Speech Recognition Competition](https://www.kaggle.com/competitions/multilingual-speech-recognition).

---

## 📌 Project Overview

This project fine-tunes the `openai/whisper-small` model on labeled multilingual audio data and generates transcript predictions for unseen test clips. Model performance is tracked using **Word Error Rate (WER)**.

---

## 🗂️ Repository Structure

```
📦 project-root
 ┣ 📓 23ds3000081_DLP_26T1_NPPE2.ipynb   # Main training notebook
 ┗ 📄 README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Install dependencies

```bash
pip install torch librosa datasets transformers scikit-learn jiwer
```

> **Note:** If running on Kaggle, most dependencies are pre-installed. Only `jiwer` needs to be added manually.

### 3. Download the dataset

Download the competition data from Kaggle and place it as follows:

```
/kaggle/input/competitions/multilingual-speech-recognition/
├── train.csv
├── test.csv
├── sample_submission.csv
└── competition_data/
    ├── train/    ← training audio clips
    └── test/     ← test audio clips
```

---

## 🚀 How It Works

| Step | Description |
|------|-------------|
| **1. Load & Normalize** | Reads CSVs, builds audio paths, lowercases/strips transcripts |
| **2. Train/Val Split** | 90/10 split with random seed 42 |
| **3. Feature Extraction** | Loads audio at 16 kHz, extracts log-Mel features via `WhisperProcessor` |
| **4. Fine-Tuning** | Trains with `Seq2SeqTrainer` using `fp16` for 500 steps |
| **5. Evaluation** | Measures WER on the validation split after training |
| **6. Inference** | Generates transcripts for all test clips |
| **7. Submission** | Saves predictions to `submission.csv` |

---

## 🏋️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `openai/whisper-small` |
| Max steps | 500 |
| Batch size (train/eval) | 4 |
| Gradient accumulation steps | 2 |
| Learning rate | 1e-5 |
| Warmup steps | 100 |
| Evaluation frequency | Every 200 steps |
| Mixed precision | fp16 ✅ |
| Metric | WER ↓ (lower is better) |

---

## 📊 Results

| Split | WER |
|-------|-----|
| Validation | *(fill after training)* |

---

## 📤 Output Files

- `./whisper-small-ft/` — saved model checkpoints
- `submission.csv` — final predictions in competition format

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-green)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-blue?logo=kaggle)

---
