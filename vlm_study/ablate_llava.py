"""
ablate_llava.py  —  Stage 4

Build intuition by PERTURBING the model and watching the answer change. Two
ablations, both on a single loaded model:

  1) vision_feature_layer = -2  vs  -1
     LLaVA reads the second-to-last CLIP layer, not the last. The last layer is
     tuned for CLIP's contrastive objective (a global image/text match); the
     penultimate layer keeps more spatial, descriptive detail. Compare answers.

  2) zero the multimodal projector
     The projector is the only bridge from vision space (1024) to LLM space
     (4096). Zero its weights and the <image> rows become all-zeros -> the model
     goes "blind" and answers from the text prompt alone. Proves the bridge is
     load-bearing. (We can't use nn.Identity here: 1024 != 4096.)

Run:
    python ablate_llava.py
"""

import copy
import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

# ----------------------------- CONFIG ------------------------------------
MODEL_PATH = "llava-hf/llava-1.5-7b-hf"   # Hub repo id -> HF cache (or a local folder path)
IMAGE      = "https://www.ilankelman.org/stopsigns/australia.jpg"
PROMPT     = "What is shown in this image? Answer in one sentence."
USE_CUDA   = torch.cuda.is_available()
DTYPE      = torch.float16 if USE_CUDA else torch.float32
MAX_NEW    = 40
# -------------------------------------------------------------------------


def load_image(src):
    if src.startswith("http"):
        import requests
        return Image.open(requests.get(src, stream=True).raw).convert("RGB")
    return Image.open(src).convert("RGB")


def answer(model, processor, inputs):
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    text = processor.batch_decode(gen, skip_special_tokens=True)[0]
    # return just the assistant turn (after the last prompt token)
    return text.strip()


def main():
    print(f"Loading {MODEL_PATH} (dtype={DTYPE}) ...")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    image = load_image(IMAGE)
    conversation = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
    }]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    inputs["pixel_values"] = inputs["pixel_values"].to(DTYPE)

    # ---------- Ablation 1: feature layer -2 vs -1 ----------
    print("\n================  ABLATION 1: vision_feature_layer  ================")
    orig_layer = model.config.vision_feature_layer
    for layer in (-2, -1):
        model.config.vision_feature_layer = layer
        print(f"\nlayer={layer:>3}: {answer(model, processor, inputs)}")
    model.config.vision_feature_layer = orig_layer
    print("\n=> -2 (default) usually keeps richer spatial detail than the")
    print("   contrastive-tuned final layer. Note any quality difference.")

    # ---------- Ablation 2: zero the projector (go blind) ----------
    print("\n================  ABLATION 2: zero the projector  ================")
    projector = None
    for name, module in model.named_modules():
        if name.endswith("multi_modal_projector"):
            projector = module
            break

    saved = copy.deepcopy(projector.state_dict())  # so we can restore
    print(f"\nbaseline (bridge intact): {answer(model, processor, inputs)}")

    with torch.no_grad():
        for p in projector.parameters():
            p.zero_()
    print(f"projector zeroed (blind): {answer(model, processor, inputs)}")

    projector.load_state_dict(saved)  # restore
    print(f"restored (sanity check) : {answer(model, processor, inputs)}")
    print("\n=> With the bridge zeroed, image rows are all-zeros and the answer")
    print("   no longer depends on the picture. The projector is load-bearing.\n")


if __name__ == "__main__":
    main()
