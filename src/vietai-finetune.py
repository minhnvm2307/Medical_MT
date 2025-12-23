import numpy as np
import torch
from datasets import Dataset, Features, Value
from sacrebleu.metrics import BLEU, TER
from nltk.translate.meteor_score import meteor_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

def bidirectional_generator(src_path, tgt_path):
    with open(src_path, "r", encoding="utf-8") as fs, open(tgt_path, "r", encoding="utf-8") as ft:
        for s, t in zip(fs, ft):
            s = s.strip()
            t = t.strip()
            if not s or not t:
                continue
            yield {
                "src": "en: " + s,
                "tgt": t,
                "direction": "en-vi",
            }
            yield {
                "src": "vi: " + t,
                "tgt": s,
                "direction": "vi-en",
            }
def build_dataset(src_path, tgt_path):
    features = Features({
        "src": Value("string"),
        "tgt": Value("string"),
        "direction": Value("string"),
    })
    return Dataset.from_generator(
        bidirectional_generator,
        gen_kwargs={"src_path": src_path, "tgt_path": tgt_path},
        features=features,
    )

MODEL_ID = "VietAI/envit5-translation"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

train_src_file = "/kaggle/input/vlsp-2025-data/train.en.txt"
train_tgt_file = "/kaggle/input/vlsp-2025-data/train.vi.txt"
val_src_file = "/kaggle/input/vlsp-2025-data/public_test.en.txt"
val_tgt_file = "/kaggle/input/vlsp-2025-data/public_test.vi.txt"
train_ds = build_dataset(train_src_file, train_tgt_file)
val_ds = build_dataset(val_src_file, val_tgt_file)
MAX_TRAIN_SAMPLES = 150_000
SEED = 42
train_ds = train_ds.shuffle(seed=SEED).select(range(min(MAX_TRAIN_SAMPLES, len(train_ds))))

def tokenize_batch(batch, max_length=128):
    model_inputs = tokenizer(
        batch["src"],
        truncation=True,
        padding=False,
        max_length=max_length,
    )
    labels = tokenizer(
        batch["tgt"],
        truncation=True,
        padding=False,
        max_length=max_length,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

MAX_LEN = 128
train_tok = train_ds.map(lambda x: tokenize_batch(x, MAX_LEN), remove_columns=train_ds.column_names, num_proc=2)
val_tok = val_ds.map(lambda x: tokenize_batch(x, MAX_LEN), remove_columns=val_ds.column_names, num_proc=2)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)

bleu_metric = BLEU()
ter_metric = TER()
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]
    bleu = bleu_metric.corpus_score(decoded_preds, [decoded_labels]).score
    ter = ter_metric.corpus_score(decoded_preds, [decoded_labels]).score
    meteor = float(np.mean([meteor_score([ref], pred) for ref, pred in zip(decoded_labels, decoded_preds)]))
    return {
        "bleu": bleu,
        "ter": ter,
        "meteor": meteor,
    }

args = Seq2SeqTrainingArguments(
    output_dir="/kaggle/working/envit5_bidirectional",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=1,
    warmup_ratio=0.05,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=2,
    predict_with_generate=True,
    generation_max_length=128,
    report_to="none",
)
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("/kaggle/working/envit5_bidirectional/final")
tokenizer.save_pretrained("/kaggle/working/envit5_bidirectional/final")
