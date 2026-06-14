"""
mini_pope.py  —  your hand-built POPE hallucination benchmark

This is the scaled-up, *measured* version of the dragon/bicycle probing you did by
hand. It runs the real POPE benchmark (lmms-lab/POPE) — a battery of yes/no
"Is there a {object} in the image?" questions, half with the object present, half
absent — against BOTH LLaVA-1.5 and Qwen2.5-VL, and scores them the way the papers
do.

We use POPE's **adversarial** split: the absent objects are ones that commonly
co-occur with what's actually in the image (e.g. asking about a "surfboard" in a
beach photo that has none). That is precisely the trap that made LLaVA invent a
"black bicycle" — a strong language prior pulling toward "yes." A well-grounded
model resists it; a weak one shows a high "yes-ratio."

Metrics (POPE-standard, treating "yes" as the positive class):
    accuracy   — overall correctness
    precision  — of the "yes" answers, how many were right
    recall     — of the truly-present objects, how many it caught
    F1         — harmonic mean
    yes_ratio  — fraction of answers that were "yes".  Ground truth is 50%, so
                 yes_ratio >> 0.5  ==  object-hallucination / yes-bias.

Models are loaded ONE AT A TIME (sequential) so two 7B models never sit in VRAM
together — fits comfortably in 24 GB.

Run:
    python mini_pope.py
Outputs a comparison table + saves mini_pope_results.json for inspection.
"""

import gc
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

# ----------------------------- CONFIG ------------------------------------
MODELS = [
    ("LLaVA-1.5-7B", "llava-hf/llava-1.5-7b-hf"),      # Hub repo id -> downloads to HF cache
    ("Qwen2.5-VL-7B", "Qwen/Qwen2.5-VL-7B-Instruct"),  # Hub repo id -> downloads to HF cache
]
CATEGORY = "adversarial"   # "adversarial" | "popular" | "random"
N_PER_LABEL = 30           # how many "yes" and how many "no" examples (so 2x total)
MAX_NEW = 8                # yes/no needs almost nothing
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
# -------------------------------------------------------------------------


def load_balanced_slice():
    """Stream POPE and collect a balanced yes/no sample from one category."""
    ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)
    yes, no = [], []
    for ex in ds:
        if ex.get("category") != CATEGORY:
            continue
        label = str(ex["answer"]).strip().lower()
        bucket = yes if label == "yes" else no
        if len(bucket) < N_PER_LABEL:
            bucket.append({
                "question": ex["question"],
                "answer": label,
                "image": ex["image"].convert("RGB"),
                "source": ex.get("image_source", ""),
            })
        if len(yes) >= N_PER_LABEL and len(no) >= N_PER_LABEL:
            break
    sample = yes + no
    print(f"Loaded {len(sample)} POPE/{CATEGORY} items "
          f"({len(yes)} present / {len(no)} absent)")
    return sample


def parse_yes_no(text):
    t = text.strip().lower()
    if t.startswith("yes") or t.startswith("y "):
        return "yes"
    if t.startswith("no") or t.startswith("n "):
        return "no"
    if "yes" in t and "no" not in t:
        return "yes"
    if "no" in t and "yes" not in t:
        return "no"
    return "?"   # unparseable / hedged


def ask(model, processor, image, question):
    q = question.strip() + " Please answer with a single word: yes or no."
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    in_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    return processor.batch_decode(gen[:, in_len:], skip_special_tokens=True)[0]


def evaluate(name, model_id, items):
    print(f"\n=== {name} ({model_id}) ===")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=DTYPE, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    tp = fp = fn = tn = unparsed = 0
    rows = []
    for i, it in enumerate(items, 1):
        raw = ask(model, processor, it["image"], it["question"])
        pred = parse_yes_no(raw)
        gt = it["answer"]
        if pred == "?":
            unparsed += 1
        elif pred == "yes" and gt == "yes":
            tp += 1
        elif pred == "yes" and gt == "no":
            fp += 1
        elif pred == "no" and gt == "yes":
            fn += 1
        else:
            tn += 1
        rows.append({"q": it["question"], "gt": gt, "pred": pred, "raw": raw.strip()})
        if i % 10 == 0:
            print(f"  {i}/{len(items)} done")

    n = len(items)
    yes_count = tp + fp
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    metrics = {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "yes_ratio": yes_count / n if n else 0.0,
        "unparsed": unparsed, "n": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }

    # free VRAM before the next model
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return metrics, rows


def main():
    items = load_balanced_slice()
    results = {}
    for name, model_id in MODELS:
        try:
            metrics, rows = evaluate(name, model_id, items)
            results[name] = {"metrics": metrics, "rows": rows}
        except Exception as e:
            print(f"  [skipped {name}: {e}]")
            results[name] = {"error": str(e)}

    # ---- comparison table ----
    print("\n" + "=" * 64)
    print(f"POPE / {CATEGORY}   ({N_PER_LABEL} present + {N_PER_LABEL} absent)")
    print("=" * 64)
    header = f"{'model':16} {'acc':>6} {'prec':>6} {'recall':>7} {'F1':>6} {'yes%':>6} {'?':>3}"
    print(header)
    print("-" * len(header))
    for name, _ in MODELS:
        r = results.get(name, {})
        if "metrics" not in r:
            print(f"{name:16}  (error)")
            continue
        m = r["metrics"]
        print(f"{name:16} {m['accuracy']*100:6.1f} {m['precision']*100:6.1f} "
              f"{m['recall']*100:7.1f} {m['f1']*100:6.1f} {m['yes_ratio']*100:6.1f} {m['unparsed']:3d}")
    print("\nyes% is the tell: ground truth is 50%. A model far above 50% is")
    print("answering 'yes' on absent objects — object hallucination, measured.")

    with open("mini_pope_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved per-question answers to mini_pope_results.json")


if __name__ == "__main__":
    main()
