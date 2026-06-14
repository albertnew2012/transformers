"""
vqa_llava.py  —  interactive VQA REPL (a copy of trace_llava_shapes.py, kept separate)

Load LLaVA-1.5 ONCE, then ask question after question from the keyboard about an
image. The expensive part (7B weights + vision tower) is paid a single time at
startup; each question is just a forward + generate, so the loop feels snappy.

This is multi-turn by default: follow-up questions keep the conversation, so you
can ask "what color is it?" after "what is in the image?" and it remembers. The
image lives in the first turn; the model re-reads it each turn (no cross-call KV
cache), which is fine for interactive use.

Commands (type at the prompt):
    :image <path-or-url>   switch to a new image (resets the conversation)
    :reset                 clear the conversation, keep the same image
    :trace                 toggle a one-line shape readout after each answer
    :help                  show commands
    :quit / :q             exit   (Ctrl-D or Ctrl-C also exit)

Anything else is treated as a question about the current image.

Run:
    python vqa_llava.py
    python vqa_llava.py path/or/url/to/image.jpg     # optional starting image
"""

import sys
import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

# ----------------------------- CONFIG ------------------------------------
MODEL_PATH = "llava-hf/llava-1.5-7b-hf"   # Hub repo id -> HF cache (or a local folder path)
IMAGE      = "https://www.ilankelman.org/stopsigns/australia.jpg"  # starting image
MAX_NEW    = 200                   # max tokens per answer
# You have a 24 GB GPU -> fp16 full precision (better quality than 4-bit).
# Flip to True only if you ever need to fit on a smaller card.
LOAD_IN_4BIT = False
# -------------------------------------------------------------------------

USE_CUDA = torch.cuda.is_available()
COMPUTE_DTYPE = torch.float16 if USE_CUDA else torch.float32


def load_image(src):
    if src.startswith("http"):
        import requests
        return Image.open(requests.get(src, stream=True).raw).convert("RGB")
    return Image.open(src).convert("RGB")


def pad_to_square(img, fill):
    """Letterbox to a square so LLaVA's center-crop becomes lossless (keeps edges).

    LLaVA-1.5 resizes shortest-edge->336 then center-crops 336x336, discarding the
    long sides of a wide image. Pre-padding to a square means the resize/crop keeps
    the WHOLE scene (with neutral-gray bars) instead of throwing edges away.
    """
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def load_model():
    print(f"Loading {MODEL_PATH}  (4-bit={LOAD_IN_4BIT}, cuda={USE_CUDA}) ...")
    load_kwargs = {"device_map": "auto"}
    if LOAD_IN_4BIT and USE_CUDA:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        load_kwargs["torch_dtype"] = COMPUTE_DTYPE

    model = LlavaForConditionalGeneration.from_pretrained(MODEL_PATH, **load_kwargs).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor


def answer_question(model, processor, image, messages, show_trace, pad_square):
    """Run one VQA turn. `messages` already includes the new user question."""
    if pad_square:
        fill = tuple(round(m * 255) for m in processor.image_processor.image_mean)
        image = pad_to_square(image, fill)
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(COMPUTE_DTYPE)

    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    reply = processor.batch_decode(gen[:, input_len:], skip_special_tokens=True)[0].strip()

    if show_trace:
        img_id = getattr(model.config, "image_token_id",
                         getattr(model.config, "image_token_index", None))
        n_img = int((inputs["input_ids"][0] == img_id).sum()) if img_id is not None else "?"
        new_toks = gen.shape[1] - input_len
        print(f"   [trace] seq_in={input_len}  image_tokens={n_img}  "
              f"text_tokens={input_len - (n_img if isinstance(n_img, int) else 0)}  "
              f"generated={new_toks}")
    return reply


HELP = """\
commands:
  :image <path-or-url>   switch image (resets the conversation)
  :reset                 clear conversation, keep image
  :pad                   toggle pad-to-square (stop the center-crop dropping edges)
  :trace                 toggle shape readout after each answer
  :help                  this help
  :quit / :q             exit"""


def main():
    model, processor = load_model()

    img_src = sys.argv[1] if len(sys.argv) > 1 else IMAGE
    image = load_image(img_src)
    messages = []          # running conversation; image rides in the first user turn
    show_trace = False
    pad_square = False     # pad-to-square instead of center-crop (keeps wide edges)

    print(f"\nImage loaded: {img_src}  (size {image.size})")
    print("Ask a question about the image. Type :help for commands, :quit to exit.\n")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break

        if not line:
            continue

        # ----- commands -----
        if line in (":quit", ":q"):
            print("bye.")
            break
        if line == ":help":
            print(HELP)
            continue
        if line == ":reset":
            messages = []
            print("(conversation cleared)")
            continue
        if line == ":trace":
            show_trace = not show_trace
            print(f"(shape trace {'ON' if show_trace else 'OFF'})")
            continue
        if line == ":pad":
            pad_square = not pad_square
            print(f"(pad-to-square {'ON — full image, no crop' if pad_square else 'OFF — default center-crop'})")
            continue
        if line.startswith(":image"):
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                print("usage: :image <path-or-url>")
                continue
            try:
                image = load_image(parts[1].strip())
                messages = []
                print(f"(image switched to {parts[1].strip()}, size {image.size}; conversation reset)")
            except Exception as e:
                print(f"could not load image: {e}")
            continue

        # ----- a question -----
        # first turn carries the image placeholder; later turns are text-only
        if not messages:
            content = [{"type": "image"}, {"type": "text", "text": line}]
        else:
            content = [{"type": "text", "text": line}]
        messages.append({"role": "user", "content": content})

        try:
            reply = answer_question(model, processor, image, messages, show_trace, pad_square)
        except Exception as e:
            print(f"[error during generation: {e}]")
            messages.pop()   # drop the failed turn so the conversation stays valid
            continue

        messages.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
        print(f"llava> {reply}\n")


if __name__ == "__main__":
    main()
