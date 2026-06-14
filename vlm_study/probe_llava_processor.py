"""
probe_llava_processor.py  —  Stage 1, Experiment 1

Look at the INPUT side of LLaVA before any model runs. The processor is where a
single `<image>` placeholder gets expanded into one token per patch, and where
the image becomes a fixed 336x336 `pixel_values` tensor.

The key contract you are verifying here:

    number of <image> tokens in input_ids  ==  number of patches the vision
    tower will produce  ==  576  (24x24 patches, CLS dropped)

Later, the model's forward pass asserts this exact equality before it can scatter
image features into the sequence. Break it here and you will understand the error
you would hit in Stage 2, Experiment 3.

This script is intentionally lightweight: it loads ONLY the processor (tokenizer +
image-processor config), no 7B weights, so it runs in seconds on CPU.

Run:
    python probe_llava_processor.py
"""

from transformers import AutoProcessor
from PIL import Image

# ----------------------------- CONFIG ------------------------------------
MODEL_PATH = "llava-hf/llava-1.5-7b-hf"   # Hub repo id -> HF cache (or a local folder path)
IMAGE      = "https://www.ilankelman.org/stopsigns/australia.jpg"  # URL or local path
PROMPT     = "What is shown in this image?"
# -------------------------------------------------------------------------


def load_image(src):
    if src.startswith("http"):
        import requests
        return Image.open(requests.get(src, stream=True).raw).convert("RGB")
    return Image.open(src).convert("RGB")


def report(proc, image, label):
    conversation = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
    }]
    prompt = proc.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = proc(images=image, text=prompt, return_tensors="pt")

    n_image_tokens = int((inputs["input_ids"][0] == proc.image_token_id).sum())
    print(f"\n--- {label} ---")
    print(f"prompt template (note where <image> lands, before expansion):")
    print(f"  {prompt!r}")
    print(f"shapes        : {{ {', '.join(f'{k}: {tuple(v.shape)}' for k, v in inputs.items())} }}")
    print(f"image_token_id: {proc.image_token_id}")
    print(f"# <image> toks: {n_image_tokens}   <-- should be 576 for base LLaVA-1.5")
    return n_image_tokens


def main():
    print(f"Loading processor from {MODEL_PATH} (no model weights) ...")
    proc = AutoProcessor.from_pretrained(MODEL_PATH)
    print(f"patch_size={getattr(proc, 'patch_size', '?')}  "
          f"strategy={getattr(proc, 'vision_feature_select_strategy', '?')}")

    image = load_image(IMAGE)
    print(f"\nOriginal image size (W, H): {image.size}")

    # 1) the canonical case -> expect 576
    report(proc, image, "as-is")

    # 2) feed a non-square / different-resolution image.
    #    LLaVA-1.5 center-crops to a fixed 336x336, so the token count does NOT
    #    change. That fixed crop is exactly what LLaVA-NeXT later removes by
    #    tiling the image -> variable token counts. Watch this stay at 576.
    wide = image.resize((672, 336))
    report(proc, wide, "resized to 672x336 (still 576 -> motivates LLaVA-NeXT)")

    print("\nTakeaway: the processor reserves one <image> slot per patch up front;")
    print("the model will overwrite exactly those slots with vision features.\n")


if __name__ == "__main__":
    main()
