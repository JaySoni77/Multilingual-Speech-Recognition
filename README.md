# Multilingual Speech Recognition — Whisper Fine-Tuning

A Kaggle competition notebook that fine-tunes OpenAI's **Whisper Small** model on a multilingual speech recognition dataset and generates submission predictions.

## Overview

The notebook loads audio clips and their reference transcripts, fine-tunes Whisper using Hugging Face's `Seq2SeqTrainer`, evaluates performance with **Word Error Rate (WER)**, and produces a `submission.csv` for the competition.

## Requirements

```bash
pip install jiwer
```

Core dependencies (pre-installed on Kaggle):

- `torch`, `librosa`, `datasets`, `transformers`
- `scikit-learn`, `pandas`, `numpy`, `jiwer`

## Dataset

Expected under `/kaggle/input/competitions/multilingual-speech-recognition/`:

```
train.csv                     # audio filename + transcript
test.csv                      # audio filename
sample_submission.csv
competition_data/train/       # training audio clips
competition_data/test/        # test audio clips
```

## Pipeline

1. **Load & normalize** — reads CSVs, lowercases/strips transcripts, builds audio file paths.
2. **Train/val split** — 90/10 split (random seed 42).
3. **Feature extraction** — loads audio at 16 kHz with `librosa`, extracts log-Mel features via `WhisperProcessor`.
4. **Fine-tuning** — runs `Seq2SeqTrainer` for 500 steps with `fp16`, batch size 4, LR `1e-5`, evaluating every 200 steps.
5. **Inference** — generates transcripts for all test clips and saves `submission.csv`.

## Training Configuration

| Parameter | Value |
|---|---|
| Model | `openai/whisper-small` |
| Max steps | 500 |
| Batch size | 4 |
| Learning rate | 1e-5 |
| Warmup steps | 100 |
| Evaluation metric | WER (lower is better) |

## Output

- `./whisper-small-ft/` — fine-tuned model checkpoints
- `submission.csv` — predictions in competition format
