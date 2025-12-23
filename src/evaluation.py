import numpy as np
import torch
import sacrebleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

def load_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def build_prompts(texts, prompt_builder=None):
    if prompt_builder is None:
        return texts
    return [prompt_builder(t) for t in texts]

def generate_translations(
    src_lines,
    model,
    tokenizer,
    device="cuda",
    batch_size=32,
    max_input_len=256,
    max_new_tokens=128,
    num_beams=4,
    prompt_builder=None,
):
    model.to(device)
    model.eval()
    preds = []
    prompts = build_prompts(src_lines, prompt_builder)
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_len,
        ).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
        if model.config.is_encoder_decoder:
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        else:
            decoded = []
            input_ids = inputs["input_ids"]
            for row, out in enumerate(outputs):
                prompt_len = int((input_ids[row] != tokenizer.pad_token_id).sum().item())
                gen_ids = out[prompt_len:]
                decoded.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
        preds.extend([d.strip() for d in decoded])
    return preds

def evaluate_translations(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    ter = sacrebleu.corpus_ter(preds, [refs]).score
    meteor = 100 * float(
        np.mean([meteor_score([word_tokenize(r)], word_tokenize(p)) for p, r in zip(preds, refs)])
    )
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = 100 * float(np.mean([rouge.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]))
    return {"bleu": bleu, "ter": ter, "meteor": meteor, "rouge_l": rouge_l}

def run_eval(
    src_path,
    ref_path,
    model,
    tokenizer,
    device="cuda",
    prompt_builder=None,
    batch_size=32,
    max_input_len=256,
    max_new_tokens=128,
    num_beams=4,
):
    src_lines = load_lines(src_path)
    ref_lines = load_lines(ref_path)
    n = min(len(src_lines), len(ref_lines))
    src_lines = src_lines[:n]
    ref_lines = ref_lines[:n]
    preds = generate_translations(
        src_lines,
        model,
        tokenizer,
        device=device,
        batch_size=batch_size,
        max_input_len=max_input_len,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        prompt_builder=prompt_builder,
    )
    metrics = evaluate_translations(preds, ref_lines)
    return metrics, preds

MODEL_TYPE = "seq2seq"  # "seq2seq" or "causal"
MODEL_PATH = "/content/model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if MODEL_TYPE == "seq2seq":
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

def default_prompt(text):
    return text

src_path = "/content/vlsp-2025-data/versions/1/public_test.en.txt"
ref_path = "/content/vlsp-2025-data/versions/1/public_test.vi.txt"
metrics, predictions = run_eval(
    src_path,
    ref_path,
    model,
    tokenizer,
    device="cuda",
    prompt_builder=default_prompt,
    batch_size=32,
    max_input_len=256,
    max_new_tokens=128,
    num_beams=4,
)
print(metrics)
