# Medical-domain Machine Translation (VLSP-2025)

## Best model download
```
# Github release
https://github.com/minhnvm2307/Medical_MT/releases/download/v1/envit5-vlsp.zip

# Hugging face
https://huggingface.co/minhnvm/envit5-finetuned-medical-vlsp2025
```

## Setup
- Python 3.12+
- Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
- Transformer variants (v1, v2, v3):

```bash
python src/transformer_v1.py
python src/transformer_v2.py
python src/transformer_v3.py
```

- Qwen fine-tuning:

```bash
python src/qwen-finetune.py
```

- VietAI envit5 fine-tuning:

```bash
python src/vietai-finetune.py
```

- Unified evaluation:

```bash
python src/evaluation.py
```

## Dataset

IWLST

VLSP-2025

## Evaluation Results

### 1. Qwen on VLSP-2025 dataset (BLEU)

| Model | EN-VI |
| --- | --- |
| Base | 26.13 |
| 150k data | 30.12 |

### 2. VietAI envit5 on VLSP dataset (BLEU)

| Model | EN-VI | VI-EN
| --- | --- | --- |
| Base | 40.75 | 24.09 |
| 150k data | 45.66 | 34.50 |
| 300k data | 47.02 | 35.02 |
| 500k data | 48.80 | 38.62 |
