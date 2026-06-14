"""
see_what_llava_sees.py  —  diagnose the "missing lion" by visualizing the crop

LLaVA-1.5 resizes an image's shortest edge to 336, then CENTER-CROPS to 336x336.
For a wide image that throws away the left/right edges entirely. This script
reconstructs the exact 336x336 tensor the model receives and saves it as a PNG,
so you can SEE what got cut. It also builds a pad-to-square version that keeps the
whole scene, for comparison.

No 7B weights needed — processor only, runs in seconds.

Run:
    python see_what_llava_sees.py [path-or-url]
Outputs:
    llava_sees.png          <- the cropped 336x336 the model actually sees (lion may be gone)
    llava_sees_padded.png   <- whole image letterboxed into 336x336 (nothing lost)
"""

import sys
import torch
from PIL import Image
from transformers import AutoProcessor

MODEL_PATH = "llava-hf/llava-1.5-7b-hf"   # Hub repo id -> HF cache (or a local folder path)
DEFAULT_IMG = "https://www.ilankelman.org/stopsigns/australia.jpg"


def load_image(src):
    if src.startswith("http"):
        import requests
        return Image.open(requests.get(src, stream=True).raw).convert("RGB")
    return Image.open(src).convert("RGB")


def pad_to_square(img, fill):
    """Letterbox an image to a square by padding the short side. Loses nothing."""
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def tensor_to_png(pixel_values, mean, std, path):
    """Undo CLIP normalization on pixel_values -> save a viewable PNG."""
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    denorm = (pixel_values.float() * std + mean).clamp(0, 1)[0]   # [3,336,336]
    arr = (denorm.permute(1, 2, 0) * 255).round().to(torch.uint8).numpy()
    Image.fromarray(arr).save(path)
    return path


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    proc = AutoProcessor.from_pretrained(MODEL_PATH)
    ip = proc.image_processor
    mean, std = ip.image_mean, ip.image_std
    fill = tuple(round(m * 255) for m in mean)   # neutral CLIP-gray padding

    img = load_image(src)
    print(f"original image: {img.size}  (W x H)")

    # 1) what the model REALLY sees (resize-shortest-edge + center-crop)
    pv = proc.image_processor(img, return_tensors="pt")["pixel_values"]
    p1 = tensor_to_png(pv, mean, std, "llava_sees.png")
    print(f"saved {p1}        <- the cropped 336x336 the model receives "
          f"(compare: is the 2nd lion still there?)")

    # 2) pad-to-square first -> resize/crop becomes lossless -> whole scene kept
    padded = pad_to_square(img, fill)
    pv2 = proc.image_processor(padded, return_tensors="pt")["pixel_values"]
    p2 = tensor_to_png(pv2, mean, std, "llava_sees_padded.png")
    print(f"saved {p2} <- whole image letterboxed, nothing cropped")

    print("\nOpen both PNGs. If the second lion is in llava_sees_padded.png but")
    print("missing from llava_sees.png, the center-crop is the culprit — and")
    print("pre-padding the image (option A below) lets the model see both.")


if __name__ == "__main__":
    main()
