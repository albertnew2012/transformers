"""
vqa_any.py  —  model-agnostic interactive VQA REPL

Same keyboard-driven loop as vqa_llava.py, but it loads ANY image-text-to-text
model via AutoModelForImageTextToText + the unified chat template. Point it at a
higher-resolution model to watch LLaVA-1.5's hallucinations shrink.

Why this fixes the "invented truck / handbags" problem that no LLaVA-1.5 setting
could: models like Qwen2.5-VL process near-NATIVE resolution (no 336 center-crop)
and LLaVA-NeXT TILES the image into several 336 crops. More real pixels -> a
stronger vision signal -> the language prior can no longer overrule what's there.

This also demonstrates the AutoModelForImageTextToText abstraction and the unified
processor.apply_chat_template(...) path from your Stage 6 study — one code path,
many architectures.

Run (downloads the model on first use):
    python vqa_any.py                                  # default: Qwen2.5-VL-7B
    python vqa_any.py Qwen/Qwen2.5-VL-7B-Instruct
    python vqa_any.py llava-hf/llava-v1.6-vicuna-7b-hf
    python vqa_any.py ./llava-1.5-7b-hf                # the baseline, for comparison
    python vqa_any.py <model> <image-path-or-url>      # also set a starting image

Commands: :image <src>  :reset  :quit/:q   (Ctrl-D / Ctrl-C also exit)
"""

import os
import sys
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# ----------------------------- CONFIG ------------------------------------
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"   # strong, low-hallucination, native-res
DEFAULT_IMAGE = "https://www.ilankelman.org/stopsigns/australia.jpg"
MAX_NEW = 200
# -------------------------------------------------------------------------

USE_CUDA = torch.cuda.is_available()
DTYPE = torch.float16 if USE_CUDA else torch.float32


def resolve_image(src):
    """Return a path/url usable in a chat message; download URLs once to avoid
    re-fetching every turn."""
    if not src.startswith("http"):
        return src
    import tempfile, requests
    r = requests.get(src, stream=True)
    suffix = os.path.splitext(src)[1] or ".jpg"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    return path


def load(model_id):
    print(f"Loading {model_id}  (dtype={DTYPE}, cuda={USE_CUDA}) ...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=DTYPE, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def answer(model, processor, messages):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(DTYPE)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    return processor.batch_decode(gen[:, input_len:], skip_special_tokens=True)[0].strip()


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    img_src = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_IMAGE

    model, processor = load(model_id)
    image_ref = resolve_image(img_src)
    messages = []

    print(f"\nModel : {model_id}")
    print(f"Image : {img_src}")
    print("Ask a question. :image <src> to switch, :reset to clear, :quit to exit.\n")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye."); break
        if not line:
            continue
        if line in (":quit", ":q"):
            print("bye."); break
        if line == ":reset":
            messages = []; print("(conversation cleared)"); continue
        if line.startswith(":image"):
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                print("usage: :image <path-or-url>"); continue
            image_ref = resolve_image(parts[1].strip())
            messages = []
            print(f"(image switched to {parts[1].strip()}; conversation reset)")
            continue

        # first turn carries the image; later turns are text-only follow-ups
        if not messages:
            content = [{"type": "image", "path": image_ref}, {"type": "text", "text": line}]
        else:
            content = [{"type": "text", "text": line}]
        messages.append({"role": "user", "content": content})

        try:
            reply = answer(model, processor, messages)
        except Exception as e:
            print(f"[error: {e}]")
            messages.pop(); continue

        messages.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
        print(f"bot> {reply}\n")


if __name__ == "__main__":
    main()
