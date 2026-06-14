"""
trace_llava_merge.py  —  Stage 2 (Experiment 2) + Stage 3 (Experiment 4)

This is the "prove it" companion to trace_llava_shapes.py. The shape trace showed
you the projector output is [batch, 576, 4096]. This script proves that those exact
576 vectors END UP inside the LLM's input sequence, replacing the <image> rows,
while every text row is left as its plain token embedding.

The single line that does the fusion is, in modeling_llava.py:248 :

    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

We verify two claims:

  (A) hidden_states[0] at <image> positions  ==  projector output   (image got in)
  (B) hidden_states[0] at text  positions     ==  token embeddings   (text untouched)

We also count how many times the VISION TOWER runs during a 40-token generation.
Answer: once. After the first forward, the 576 image vectors live in the KV cache,
so decoding never re-encodes the image. That is the core efficiency property of
token-merge VLMs (LLaVA, Qwen-VL, Fuyu) vs cross-attention VLMs (IDEFICS, mllama).

Run:
    python trace_llava_merge.py
"""

import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

# ----------------------------- CONFIG ------------------------------------
MODEL_PATH = "llava-hf/llava-1.5-7b-hf"   # Hub repo id -> HF cache (or a local folder path)
IMAGE      = "https://www.ilankelman.org/stopsigns/australia.jpg"
PROMPT     = "What is shown in this image?"
# fp16 on GPU (you have 24 GB, no quantization needed); fp32 on CPU
USE_CUDA   = torch.cuda.is_available()
DTYPE      = torch.float16 if USE_CUDA else torch.float32
# -------------------------------------------------------------------------


def load_image(src):
    if src.startswith("http"):
        import requests
        return Image.open(requests.get(src, stream=True).raw).convert("RGB")
    return Image.open(src).convert("RGB")


def find_module(model, suffix):
    for name, module in model.named_modules():
        if name.endswith(suffix):
            return module
    raise AttributeError(f"no module ending in '{suffix}'")


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

    # --- capture the projector output (the 576 image vectors) ---
    captured = {}

    def cap_proj(_m, _inp, out):
        captured["proj_out"] = out.detach().reshape(-1, out.shape[-1])  # [576, 4096]

    h_proj = find_module(model, "multi_modal_projector").register_forward_hook(cap_proj)

    # ============================================================
    # Experiment 2 — prove the masked_scatter merge
    # ============================================================
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    h_proj.remove()

    embeds_in = out.hidden_states[0][0]                      # [seq, 4096], post-merge
    input_ids = inputs["input_ids"][0]
    img_id = model.config.image_token_id
    img_mask = input_ids == img_id                          # [seq] bool

    proj_out = captured["proj_out"].to(embeds_in.dtype)
    image_rows = embeds_in[img_mask]                        # [576, 4096]

    # plain token embeddings (what text rows should still equal)
    tok_embeds = model.get_input_embeddings()(input_ids)    # [seq, 4096]
    text_rows_seq = embeds_in[~img_mask]
    text_rows_tok = tok_embeds[~img_mask]

    print("\n================  MERGE PROOF  ================")
    print(f"# <image> rows in sequence            : {int(img_mask.sum())}")
    print(f"projector output rows                 : {proj_out.shape[0]}")
    claim_a = torch.allclose(image_rows, proj_out, atol=1e-3, rtol=1e-3)
    claim_b = torch.allclose(text_rows_seq, text_rows_tok, atol=1e-3, rtol=1e-3)
    print(f"(A) image positions  == projector out : {claim_a}")
    print(f"(B) text  positions  == token embeds  : {claim_b}")
    if claim_a and claim_b:
        print("=> CONFIRMED: image vectors overwrote exactly the <image> rows;")
        print("   text rows were left untouched. That is the whole fusion.")
    else:
        print("=> Mismatch — inspect dtype/tolerance or model version.")
    print("==============================================")

    # ============================================================
    # Experiment 4 — vision tower runs ONCE per generation
    # ============================================================
    calls = {"n": 0}

    def count_vision(_m, _inp, _out):
        calls["n"] += 1

    h_cnt = find_module(model, "vision_tower").register_forward_hook(count_vision)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=40)
    h_cnt.remove()

    answer = processor.batch_decode(gen, skip_special_tokens=True)[0]
    new_tokens = gen.shape[1] - inputs["input_ids"].shape[1]

    print("\n============  VISION-TOWER CALLS  ============")
    print(f"new tokens generated                  : {new_tokens}")
    print(f"vision_tower forward calls            : {calls['n']}   <-- expect 1")
    print("=> The image is encoded once; its 576 vectors then live in the KV")
    print("   cache, so every decode step is pure text. Token-merge VLMs pay")
    print("   the vision cost exactly once.")
    print("==============================================")
    print("\nModel answer:\n" + answer + "\n")


if __name__ == "__main__":
    main()
