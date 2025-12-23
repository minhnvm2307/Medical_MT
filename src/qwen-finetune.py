import numpy as np
import torch
from datasets import Dataset, Features, Value
from sacrebleu.metrics import BLEU, TER
from nltk.translate.meteor_score import meteor_score
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model

def bidirectional_generator(src_path, tgt_path):
    with open(src_path, "r", encoding="utf-8") as fs, open(tgt_path, "r", encoding="utf-8") as ft:
        for s, t in zip(fs, ft):
            s = s.strip()
            t = t.strip()
            if not s or not t:
                continue
            yield {
                "src": s,
                "tgt": t,
                "src_lang": "English",
                "tgt_lang": "Vietnamese",
                "direction": "en-vi",
            }
            yield {
                "src": t,
                "tgt": s,
                "src_lang": "Vietnamese",
                "tgt_lang": "English",
                "direction": "vi-en",
            }
def build_dataset(src_path, tgt_path):
    features = Features({
        "src": Value("string"),
        "tgt": Value("string"),
        "src_lang": Value("string"),
        "tgt_lang": Value("string"),
        "direction": Value("string"),
    })
    return Dataset.from_generator(
        bidirectional_generator,
        gen_kwargs={"src_path": src_path, "tgt_path": tgt_path},
        features=features,
    )

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map=None,
    trust_remote_code=True,
)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
)
model = get_peft_model(model, lora_config)
model.config.use_cache = False

SYSTEM_PROMPT = (
    "You are a professional medical translator specialized in clinical and biomedical text. "
    "Translate faithfully without adding or omitting information. "
    "Preserve numbers, units, dosage, and formatting. "
    "Use standard medical terminology and keep abbreviations as in the source. "
    "Output only the translation."
)
def build_chat_prompt(src_text, src_lang, tgt_lang):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Translate from {src_lang} to {tgt_lang}.
"
                f"Source ({src_lang}): {src_text}
"
                f"Translation ({tgt_lang}):"
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
def tokenize_with_mask(example, max_length=256):
    prompt = build_chat_prompt(example["src"], example["src_lang"], example["tgt_lang"])
    answer = example["tgt"].strip()
    full_text = prompt + answer + tokenizer.eos_token
    enc = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)
    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length, padding=False)["input_ids"]
    labels = enc["input_ids"].copy()
    labels[: len(prompt_ids)] = [-100] * min(len(prompt_ids), len(labels))
    enc["labels"] = labels
    return enc

train_src_file = "/kaggle/input/vlsp-2025-data/train.en.txt"
train_tgt_file = "/kaggle/input/vlsp-2025-data/train.vi.txt"
val_src_file = "/kaggle/input/vlsp-2025-data/public_test.en.txt"
val_tgt_file = "/kaggle/input/vlsp-2025-data/public_test.vi.txt"
train_ds = build_dataset(train_src_file, train_tgt_file)
val_ds = build_dataset(val_src_file, val_tgt_file)
MAX_TRAIN_SAMPLES = 50_000
SEED = 42
train_ds = train_ds.shuffle(seed=SEED).select(range(min(MAX_TRAIN_SAMPLES, len(train_ds))))
MAX_LEN = 256
train_tok = train_ds.map(lambda x: tokenize_with_mask(x, MAX_LEN), remove_columns=train_ds.column_names, num_proc=2)
val_tok = val_ds.map(lambda x: tokenize_with_mask(x, MAX_LEN), remove_columns=val_ds.column_names, num_proc=2)

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
    gen_lens = [len(p.split()) for p in decoded_preds]
    return {
        "bleu": bleu,
        "ter": ter,
        "meteor": meteor,
        "gen_len": float(np.mean(gen_lens)),
    }

args = Seq2SeqTrainingArguments(
    output_dir="/kaggle/working/qwen_mt_lora",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    fp16=True,
    gradient_checkpointing=True,
    predict_with_generate=True,
    generation_max_length=256,
    generation_num_beams=1,
    report_to="none",
    dataloader_num_workers=2,
    ddp_find_unused_parameters=False,
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
trainer.save_model("/kaggle/working/qwen_mt_lora/final")
tokenizer.save_pretrained("/kaggle/working/qwen_mt_lora/final")
