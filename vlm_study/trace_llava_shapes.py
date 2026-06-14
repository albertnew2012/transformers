"""
trace_llava_shapes.py

Watch a single image flow through LLaVA-1.5 and print the tensor shape
at every stage of the pipeline:

    image pixels -> patch embeddings -> projector output -> tokens the LLM sees

The projector is the "bridge": it remaps the vision encoder's 1024-dim patch
features into the LLM's 4096-dim embedding space. That 1024 -> 4096 step on the
same 576 patch tokens is the moment the whole architecture clicks.

Run:
    python trace_llava_shapes.py

Edit the CONFIG block below to point at your own image.
"""

import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

# ----------------------------- CONFIG ------------------------------------
MODEL_PATH = "llava-hf/llava-1.5-7b-hf"   # Hub repo id -> HF cache (or a local folder path)
IMAGE      = "https://www.ilankelman.org/stopsigns/australia.jpg"  # URL or local path; swap in your own
PROMPT     = "What is shown in this image?"
LOAD_IN_4BIT = torch.cuda.is_available()   # 4-bit on GPU (~5-6 GB VRAM); full precision on CPU
# -------------------------------------------------------------------------

compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


def load_image(src):
    if src.startswith("http"):
        import requests
        return Image.open(requests.get(src, stream=True).raw).convert("RGB")
    return Image.open(src).convert("RGB")


def find_module(model, suffix):
    """Locate a submodule by the tail of its name, robust to version-specific nesting."""
    for name, module in model.named_modules():
        if name.endswith(suffix):
            return module
    raise AttributeError(f"no module ending in '{suffix}' found")


def main():
    print(f"Loading {MODEL_PATH}  (4-bit={LOAD_IN_4BIT}) ...")
    load_kwargs = {"device_map": "auto"}
    if LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = LlavaForConditionalGeneration.from_pretrained(MODEL_PATH, **load_kwargs).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    image = load_image(IMAGE)

    # --- capture intermediate shapes with forward hooks ---
    captured = {}

    def cap_vision(_mod, _inp, out):
        hs = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        captured["vision_out"] = tuple(hs.shape)          # raw patch embeddings (incl. CLS token)

    def cap_proj(_mod, inp, out):
        captured["proj_in"]  = tuple(inp[0].shape)        # selected patches fed to the bridge
        captured["proj_out"] = tuple(out.shape)           # patches remapped into LLM dim

    h1 = find_module(model, "vision_tower").register_forward_hook(cap_vision)
    h2 = find_module(model, "multi_modal_projector").register_forward_hook(cap_proj)

    # --- build the multimodal input ---
    conversation = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
    }]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(compute_dtype)

    # --- one forward pass; hidden_states[0] = the embeddings the LLM actually sees ---
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    merged_seq = tuple(out.hidden_states[0].shape)

    h1.remove(); h2.remove()

    # --- count how many image tokens ended up in the sequence ---
    img_tok = getattr(model.config, "image_token_index",
                      getattr(model.config, "image_token_id", None))
    n_image_tokens = int((inputs["input_ids"][0] == img_tok).sum()) if img_tok is not None else "?"

    # --- report ---
    print("\n================  SHAPE TRACE  ================")
    print(f"1. pixel_values  (image into vision tower) : {tuple(inputs['pixel_values'].shape)}")
    print(f"     -> [batch, channels, 336, 336]")
    print(f"2. vision tower output (incl. CLS token)   : {captured['vision_out']}")
    print(f"     -> 577 = 576 patches (24x24) + 1 CLS")
    print(f"3. patch embeddings used by LLaVA          : {captured['proj_in']}")
    print(f"     -> CLS dropped, layer -2 selected: [batch, 576, 1024]")
    print(f"4. projector output  <-- THE BRIDGE        : {captured['proj_out']}")
    print(f"     -> same 576 tokens, remapped 1024 -> 4096 (the LLM's dim)")
    print(f"5. image tokens inserted into the sequence : {n_image_tokens}")
    print(f"6. merged sequence fed to the LLM          : {merged_seq}")
    print(f"     -> [batch, text_tokens + 576, 4096]")
    print("==============================================\n")

    # --- and the actual answer, for satisfaction ---
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=40)
    answer = processor.batch_decode(gen, skip_special_tokens=True)[0]
    print("Model answer:\n" + answer)


if __name__ == "__main__":
    main()