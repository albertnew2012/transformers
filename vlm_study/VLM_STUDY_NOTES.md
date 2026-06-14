# VLM Study Notes — what we learned by getting hands dirty

A consolidated reference for the LLaVA-vs-Qwen investigation. Pairs with the
curriculum in `~/.claude/plans/i-git-cloned-and-precious-zebra.md` and the scripts
listed at the bottom.

---

## 1. The one mental model to keep

> **A VLM is a language model gently steered by a low-bandwidth image signal.**
> Accuracy ≈ **how much the vision channel can override the language prior.**
> Vision bandwidth = resolution × encoder quality × training.

When the steering is weak (low-res, cropped, frozen encoder, small/old LLM), the
language prior takes the wheel → fluent, confident **fiction**. Strengthen the
signal and the pixels start winning. Almost everything below is a corollary.

(Corollary that bit *me*: I confidently called Qwen's "pedestrians" a hallucination
from my own downsampled view + a strong prior — the exact failure mode. Weak signal
+ strong prior = plausible fiction, in humans and models alike. Go get more signal —
zoom in / raise resolution — *before* committing.)

---

## 2. How LLaVA actually works (the pipeline you traced)

```
image ─► preprocess ─► vision tower ─► select+drop CLS ─► projector ─► MERGE ─► LLM ─► text
        336x336 crop   CLIP ViT-L/14    layer -2, 576      1024→4096   masked_   Vicuna
                       577 patch+CLS     patches            "the bridge" scatter   7B
```

Key numbers and where they come from:
- **336×336**: fixed crop (`image_size`). `336 ÷ 14 (patch) = 24` → **24×24 = 576 patches**.
- **577 → 576**: CLIP adds a CLS summary token; LLaVA drops it (`vision_feature_select_strategy="default"`, the `[:, 1:]` slice) and reads the **penultimate** layer (`-2`), because the last layer is tuned for CLIP's *contrastive* objective, not spatial detail.
- **The bridge**: a 2-layer MLP `1024 → 4096` (`LlavaMultiModalProjector`). Maps vision-space vectors into the LLM's embedding space.
- **The merge (the whole fusion)**: one line —
  `inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)`
  at `modeling_llava.py:248`. The 576 image vectors **overwrite** the 576 `<image>`
  placeholder rows; text rows are untouched.
- **The token-count contract**: the processor reserves exactly 576 `<image>` slots
  (`replace_image_token`, `processing_llava.py:70-76`); the model asserts
  `#tokens == #features` in `get_placeholder_mask` (`:206-212`). Violate it → error.
  Example sequence: `595 = 19 text tokens + 576 image tokens`.
- **Vision runs once per generation**: after the first forward the 576 vectors live
  in the KV cache, so decoding is pure text. (Token-merge VLMs pay vision cost once.)

---

## 3. The failure modes we discovered (on the stop-sign / Chinatown image)

| # | Symptom | Real cause | Fix |
|---|---|---|---|
| 1 | "one lion" (there are two) | **center-crop deleted the right lion** (~211px/side of a 1300px image) | pad-to-square (`:pad`) keeps edges; or a model that doesn't crop |
| 2 | identical answer every run | **greedy decoding is deterministic** — same input → same output, always | not a bug; use `do_sample=True` for variety |
| 3 | invented "people with handbags / a truck" | **object hallucination** — weak 336px signal, language prior fills the cliché | stronger/higher-res model; grounded prompt; shorter output |
| 4 | "the bicycle is black" (no bicycle) | **presupposition trap** — a WH-question ("what color is the X") presupposes X; weak models can't reject the premise | yes/no framing lets it say "no"; better grounding rejects it |
| 5 | archway text read as "17" (it's 中華門) | **OCR needs resolution** LLaVA doesn't have at 336px (worse when letterboxed) | native high-res model (Qwen reads it: "中华门 → China Gate") |

Sharp sub-lesson (#4): **failure is framing-dependent, not object-dependent.** Same
absent object, opposite outcomes — "*Do you see* a dragon?" → correctly "no";
"*What color is* the bicycle?" → confabulated. The presupposition in a WH-question
is the trap. (This is exactly what POPE's *adversarial* split tests.)

Important nuance: padding fixed the *missing lion* (a **pixel** problem) but LLaVA
**still hallucinated** the truck (a **prior** problem). Cropping was a real bug, but
it was never the main reason LLaVA hallucinated. The main reason was bandwidth — and
only a better model fixed that.

---

## 4. Why Qwen2.5-VL succeeded where LLaVA-1.5 failed

Same skeleton (image → encoder → connector → LLM), every component upgraded; two
were decisive for this image.

| Stage | LLaVA-1.5 | Qwen2.5-VL | Why it mattered |
|---|---|---|---|
| Preprocess | center-crop to 336² (drops edges) | **native dynamic resolution**, whole image kept | LLaVA deleted the 2nd lion |
| Encoder | **frozen** CLIP-336, contrastive | ViT **co-trained**, **2D-RoPE**, window attn | spatial + fine detail survive |
| Tokens | fixed **576** | **variable**, scales w/ image; 2×2 **PatchMerger** | more real pixels described |
| Connector | thin 2-layer MLP | MLP patch-merger (2×2 neighborhood) | preserves local structure |
| LLM | Vicuna-1.5-7B | Qwen2.5-7B | reports pixels vs completes a sentence |
| Training | ~1.2M, verbose GPT captions | huge, **grounding/OCR/counting**, says "no" | tuned to be accurate, not agreeable |

Two decisive factors:
1. **Resolution / no destructive crop** (perception gap) — Qwen *had* the evidence.
2. **Signal bandwidth vs language prior** (grounding gap) — Qwen's wide channel
   beats the prior *and* leading questions; it even gives **calibrated** answers
   ("at least three pedestrians, possibly more due to angle").

The field's 2-year arc LLaVA-1.5 → Qwen2.5-VL = "make the vision channel wider than
the language prior": stop cropping → native detail → preserve space (2D RoPE) →
train the encoder on grounding data → on a stronger LLM.

---

## 5. How VLMs are measured (and the tooling)

**Two layers — don't conflate them:**
- **Eval harnesses / toolkits** (the code): **lmms-eval**, **VLMEvalKit**.
- **Benchmarks / datasets** (data+metric the harness runs): POPE, MMMU, OCRBench…

Our manual probes were a hand-rolled micro-benchmark. Mapping:

| What we probed | Standard benchmark |
|---|---|
| dragon/bicycle (false object) | **POPE** (adversarial), HallusionBench, MMHal |
| archway OCR | **OCRBench**, TextVQA, DocVQA, ChartQA |
| counting lions | TallyQA, CountBench |
| "where is each lion" | RefCOCO / + / g |
| open description | VQAv2, GQA, OK-VQA |
| overall capability | **MMMU**, MMBench, MME, MM-Vet, MMStar |

**Recommendation for a learner:** hand-roll a tiny eval first (`mini_pope.py`) to
learn what a benchmark *is* → then **VLMEvalKit** for quick standard numbers
(simple one-command UX, leaderboard-aligned) → **lmms-eval** for paper-grade rigor.

Links: lmms-eval `github.com/EvolvingLMMs-Lab/lmms-eval` ·
VLMEvalKit `github.com/open-compass/VLMEvalKit` ·
POPE `huggingface.co/datasets/lmms-lab/POPE` ·
data hub `huggingface.co/lmms-lab`. (Leaderboard `opencompass/open_vlm_leaderboard`
is maintained but its *published snapshot* can lag the frontier by months — for
"what's best now" use LMArena Vision + model tech reports.)

---

## 6. The toolkit we built (repo root, personal study scripts — keep out of PRs)

| Script | Teaches |
|---|---|
| `trace_llava_shapes.py` (yours) | shape trace: pixels → patches → projector → merged sequence |
| `probe_llava_processor.py` | the 576-token contract; fixed crop stays 576 |
| `see_what_llava_sees.py` | reconstructs the 336² crop — *see* what's discarded |
| `trace_llava_merge.py` | proves `masked_scatter`; vision tower runs once/gen |
| `ablate_llava.py` | feature-layer −2 vs −1; zero-projector → blind |
| `vqa_llava.py` | LLaVA REPL with `:pad` / `:trace` toggles |
| `vqa_any.py` | model-agnostic REPL (AutoModelForImageTextToText) — the fix |
| `mini_pope.py` | measured hallucination benchmark, LLaVA vs Qwen (POPE) |

---

## 7. Stage 6 — VERIFIED: Qwen's merge is the SAME line; the difference is upstream

Read the code and a sharp correction emerged. Qwen2.5-VL uses the **identical fusion
primitive** as LLaVA — `inputs_embeds.masked_scatter(image_mask, image_embeds)` at
`modeling_qwen2_5_vl.py:1245`. The *merge* did not change at all. What changed is
**how the image features are produced before that line**:

- `Qwen2_5_VLPatchMerger` (`:138`) — RMSNorm → MLP over a **2×2 patch neighborhood**
  (`hidden_size = context_dim * spatial_merge_size²`), merging 4 patches into 1
  token. (LLaVA's connector is a plain per-token 2-layer MLP — no spatial merge.)
- `rot_pos_emb` / `apply_rotary_pos_emb_vision` (`:161, :384`) — **2D rotary**
  positions over the patch grid, so spatial layout is baked into attention.
- dynamic resolution via `grid_thw` — variable token count, whole image kept.

**So the precise lesson:** "LLaVA vs Qwen" is NOT "different merge" — it's the **same
`masked_scatter`, fed by far richer features** (high-res, 2D-positioned, spatially
merged). Bandwidth differs, not the plumbing.

"Different merge" *is* the right frame for the rest of the tour, though:
- **BLIP-2** — Q-Former: 32 learned query tokens replace the image (different connector)
- **IDEFICS / mllama** — gated **cross-attention** in each decoder layer; **no merge**
- **Fuyu** — patches projected straight into the text stream; no separate tower
- **Pixtral** — 2D RoPE in the vision encoder (like Qwen)

## 8. MEASURED: mini_pope.py results (POPE adversarial, 30 present + 30 absent)

```
model               acc   prec  recall     F1   yes%
LLaVA-1.5-7B       76.7   73.5    83.3   78.1   56.7
Qwen2.5-VL-7B      85.0   92.0    76.7   83.6   41.7
```

The anecdote, now quantified:
- **yes% = the hallucination tell.** Ground truth is 50%. LLaVA answers "yes" **56.7%**
  → leans toward *seeing things that aren't there*. Qwen **41.7%** → leans cautious.
- **Precision** is the sharpest gap: LLaVA **73.5** vs Qwen **92.0**. When LLaVA says
  "yes, it's there," it's **wrong ~1 in 4** — measured object hallucination. Qwen's
  "yes" is trustworthy 92% of the time.
- **The objects LLaVA false-positived: truck, handbag, bus, motorcycle, book** — the
  *same* hallucinations it volunteered in the open "what do you see?" (truck, handbags).
  The systematic bias and the anecdote are the same phenomenon.
- **The tradeoff is real, not free.** LLaVA's recall is *higher* (83.3 vs 76.7) — saying
  "yes" a lot catches more present objects. Qwen's caution costs a few misses (it said
  "no" to a present *skis* and *spoon* — small/ambiguous objects). So: **LLaVA errs
  toward confident false-positives (hallucination); Qwen errs toward cautious
  false-negatives (misses).** For hallucination-sensitive use, Qwen's error mode is the
  safer one — and its overall accuracy/F1 are clearly higher.

Even on this hardest (adversarial) split, neither is perfect — but the *direction* of
their errors is the whole story.
