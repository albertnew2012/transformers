"""
visualize_attention_map_onnx.py  —  ViT attention rollout + ONNX export, powered by 🤗 Transformers

This is a port of the classic ViT-pytorch `visualize_attention_map_onnx.py` demo, but instead of
the standalone `VisionTransformer(vis=True)` it drives this repository's
`ViTForImageClassification` (see `src/transformers/models/vit/modeling_vit.py`).

What it does:
  1. Loads `google/vit-base-patch16-224` (classifier head + ImageNet labels) with **eager**
     attention so that `outputs.attentions` is actually populated (SDPA/FA return `None`).
  2. Runs the image through the model and prints the top-5 ImageNet predictions.
  3. Computes "attention rollout" (Abnar & Zuidema, 2020): average heads, add the identity to
     account for residual connections, renormalize, then recursively multiply across layers.
  4. Overlays the CLS->patch attention on the original image (final map + every layer).
  5. Renders the autograd graph with torchviz  -> `model_visualization.png`.
  6. Exports the model to ONNX                 -> `vit_model.onnx`
     (and runs an optional onnxruntime parity check if onnxruntime is installed).

Run:
    python vit_study/visualize_attention_map_onnx.py [image-path-or-url]

    # options
    python vit_study/visualize_attention_map_onnx.py cat.jpg --model google/vit-base-patch16-224
    python vit_study/visualize_attention_map_onnx.py --no-show          # don't pop up windows

Outputs (written next to this script by default, override with --output-dir):
    vit_attention_rollout.png   <- original image vs. final rolled-out attention
    vit_attention_layers.png    <- attention overlay for each of the 12 layers
    model_visualization.png     <- torchviz autograd graph
    vit_model.onnx              <- exported ONNX model

Deps:
    pip install torch transformers matplotlib opencv-python numpy pillow torchviz
    # torchviz also needs the system Graphviz binaries: e.g. `apt install graphviz`
    # optional: pip install onnx onnxruntime   (for the parity check)
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from transformers import AutoImageProcessor, ViTForImageClassification


MODEL_PATH = "google/vit-base-patch16-224"  # has a classifier head + 1000 ImageNet id2label
DEFAULT_IMG = "http://images.cocodataset.org/val2017/000000039769.jpg"  # the HF docs "two cats" image


def load_image(src: str) -> Image.Image:
    """Load an RGB image from a local path or an http(s) URL."""
    if src.startswith("http"):
        import requests  # always available (transitive dep of huggingface_hub)

        data = requests.get(src, stream=True, timeout=30).content
        return Image.open(io.BytesIO(data)).convert("RGB")
    return Image.open(src).convert("RGB")


def attention_rollout(attentions: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """
    Recursively fuse the per-layer attention maps into a single CLS->token map.

    `attentions` is the tuple returned by 🤗 models when `output_attentions=True`:
    one tensor per layer, each shaped [batch, num_heads, seq, seq] (batch is assumed to be 1).

    Returns `joint_attentions` shaped [num_layers, seq, seq], where row `n` is the cumulative
    attention after fusing layers `0..n`.
    """
    # [num_layers, num_heads, seq, seq] -> drop the batch dim
    att_mat = torch.stack(attentions).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = att_mat.mean(dim=1)  # [num_layers, seq, seq]

    # To account for residual connections, add an identity matrix to the attention matrix
    # and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(-1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1, keepdim=True)

    # Recursively multiply the weight matrices.
    joint_attentions = torch.zeros_like(aug_att_mat)
    joint_attentions[0] = aug_att_mat[0]
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    return joint_attentions


def attention_overlay(joint_row: torch.Tensor, im_np: np.ndarray) -> np.ndarray:
    """Turn one CLS->patch attention row into an image-sized heatmap overlaid on `im_np`."""
    num_tokens = joint_row.shape[-1]
    grid_size = int(math.sqrt(num_tokens - 1))  # drop the CLS token, assume a square patch grid

    # Attention from the CLS/output token to every patch token.
    mask = joint_row[0, 1:].reshape(grid_size, grid_size).detach().cpu().numpy()
    # cv2.resize dsize is (width, height); PIL/np give us (H, W, C) so flip to (W, H).
    mask = cv2.resize(mask / mask.max(), (im_np.shape[1], im_np.shape[0]))[..., np.newaxis]
    return (mask * im_np).astype("uint8")


class ViTLogitsWrapper(nn.Module):
    """Thin wrapper so torch.onnx.export sees a plain `Tensor` output instead of a dataclass."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits


def export_graph(model: nn.Module, dummy_input: torch.Tensor, out_path: Path) -> None:
    """Render the autograd graph with torchviz (needs the system Graphviz binaries)."""
    try:
        from torchviz import make_dot
    except ImportError:
        print("[skip] torchviz not installed -> skipping autograd graph. `pip install torchviz`")
        return

    logits = model(pixel_values=dummy_input).logits
    dot = make_dot(logits, params=dict(model.named_parameters()))
    dot.format = "png"
    try:
        # render() appends ".png"; pass the stem so we land on `<out_path>`.
        dot.render(out_path.with_suffix(""), cleanup=True)
        print(f"[ok] autograd graph saved to '{out_path}'")
    except Exception as exc:  # graphviz executables missing, etc.
        print(f"[skip] could not render torchviz graph ({exc}). Is the Graphviz binary installed?")


def export_onnx(model: nn.Module, dummy_input: torch.Tensor, out_path: Path) -> None:
    """Export to ONNX through the logits wrapper, with a dynamic batch axis."""
    wrapper = ViTLogitsWrapper(model).eval()
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    print(f"[ok] ONNX model saved to '{out_path}'")

    # Optional parity check against the PyTorch model.
    try:
        import onnxruntime as ort
    except ImportError:
        print("[info] install onnxruntime to verify the export: `pip install onnxruntime`")
        return

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(["logits"], {"pixel_values": dummy_input.numpy()})[0]
    with torch.no_grad():
        torch_logits = wrapper(dummy_input).numpy()
    max_abs_diff = float(np.abs(onnx_logits - torch_logits).max())
    print(f"[ok] onnxruntime parity check: max|Δ logits| = {max_abs_diff:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", default=DEFAULT_IMG, help="image path or URL")
    parser.add_argument("--model", default=MODEL_PATH, help="HF model id or local path")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
        help="where to write the PNG/ONNX artifacts (defaults to this script's folder)",
    )
    parser.add_argument("--no-show", action="store_true", help="save figures without opening windows")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- model + preprocessing -------------------------------------------------------------
    # `attn_implementation="eager"` is REQUIRED: only the eager path returns attention weights.
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = ViTForImageClassification.from_pretrained(args.model, attn_implementation="eager")
    model.eval()

    im = load_image(args.image)
    im_np = np.array(im)
    inputs = processor(images=im, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    if not outputs.attentions:
        raise RuntimeError(
            "Model returned no attentions. Load the model with attn_implementation='eager' "
            "and pass output_attentions=True."
        )

    # --- top-5 predictions -----------------------------------------------------------------
    probs = torch.softmax(outputs.logits, dim=-1)
    top5 = torch.topk(probs, k=5, dim=-1)
    print("\nPrediction labels and attention map!\n")
    for score, idx in zip(top5.values[0], top5.indices[0]):
        print(f"{score.item():.5f} : {model.config.id2label[idx.item()]}")
    print()

    # --- attention rollout -----------------------------------------------------------------
    joint_attentions = attention_rollout(outputs.attentions)

    # Final fused attention (last layer) overlaid on the original image.
    result = attention_overlay(joint_attentions[-1], im_np)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))
    ax1.set_title("Original")
    ax2.set_title("Attention Map")
    ax1.imshow(im_np)
    ax2.imshow(result)
    for ax in (ax1, ax2):
        ax.axis("off")
    rollout_path = out_dir / "vit_attention_rollout.png"
    fig.savefig(rollout_path, bbox_inches="tight")
    print(f"[ok] rollout overlay saved to '{rollout_path}'")

    # Per-layer attention maps in a single grid figure.
    num_layers = joint_attentions.size(0)
    cols = 4
    rows = math.ceil(num_layers / cols)
    fig2, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    for i, ax in enumerate(axes.flat):
        if i < num_layers:
            ax.imshow(attention_overlay(joint_attentions[i], im_np))
            ax.set_title(f"Attention Map Layer {i + 1}")
        ax.axis("off")
    fig2.tight_layout()
    layers_path = out_dir / "vit_attention_layers.png"
    fig2.savefig(layers_path, bbox_inches="tight")
    print(f"[ok] per-layer overlays saved to '{layers_path}'")

    # --- graph + ONNX ----------------------------------------------------------------------
    image_size = model.config.image_size
    if isinstance(image_size, (list, tuple)):
        height, width = image_size
    else:
        height = width = image_size
    dummy_input = torch.randn(1, model.config.num_channels, height, width)

    export_graph(model, dummy_input, out_dir / "model_visualization.png")
    export_onnx(model, dummy_input, out_dir / "vit_model.onnx")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
