#!/usr/bin/env python3
"""
Zero-shot photo screening using CLIP text descriptions - no training photos needed.

Unlike vision.py (a ResNet18 trained on your own labeled lumpy-skin photos), this
module scores a photo against plain-language descriptions of what a disease looks
like, using OpenCLIP. It has no training step: add a disease's visual description
to VISUAL_PROMPTS below and it can be screened for immediately.

This is meaningfully LESS accurate than a model trained on real labeled photos of
YOUR cattle, especially between similar-looking lesions. Always report it as a
"possible visual match - confirm with symptoms", never as a diagnosis, and feed it
into triage the same cautious way vision.py's low-confidence photos are handled.

Only diseases with an actual visual description belong in VISUAL_PROMPTS - most of
the 18 diseases (anthrax, brucellosis, TB, tick fevers, etc.) have no reliable photo
signature and must stay symptom-only; adding them here would just produce confident
nonsense.

    pip install open_clip_torch --break-system-packages

Usage:
    from zero_shot_vision import screen
    result = screen(pil_image)   # -> ranked list of {id, prompt_matched, score, caution}
"""
from pathlib import Path

import torch
from PIL import Image, ImageOps

# "-quickgelu" matters: OpenAI's original CLIP weights use the QuickGELU activation.
# Loading them under plain "ViT-B-32" silently applies the WRONG activation function -
# open_clip only warns about it ("quick_gelu mismatch"), it does not refuse to run - so this
# is not optional. Confirmed by testing: without "-quickgelu" all three verified test photos
# (known lumpy skin, known healthy, known FMD) scored the WRONG disease, one of them at 0.815.
MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"          # downloaded once from the OpenCLIP release, then cached locally
MAX_SIDE = 1024

# Baseline anchors so a photo has something to rank AGAINST, not just diseases to pick between.
# Without these, CLIP is forced to prefer one disease prompt over another even on a healthy photo.
BASELINE_PROMPTS = {
    "healthy": [
        "a photo of healthy cattle skin with no lesions",
        "a photo of a healthy cow's mouth with no blisters",
        "a photo of a normal, non-swollen cattle udder",
    ],
    "not_cattle": [
        "a photo with no cattle in it",
        "a photo of a person, building, or object, not an animal",
    ],
}

# One entry per disease that actually has a visual signature. Several phrasings each -
# zero-shot accuracy improves noticeably with prompt variety rather than one description.
VISUAL_PROMPTS = {
    "lumpy_skin_disease": [
        "a cow with many firm, raised, round skin nodules 2 to 5 centimetres wide",
        "cattle skin covered in hard lumps and swellings from lumpy skin disease",
    ],
    "foot_and_mouth_disease": [
        "a cow's mouth with blisters and raw ulcers on the tongue and gums",
        "a cow's hoof with blisters or raw sores between the claws",
        "a cow drooling heavily with stringy saliva from mouth sores",
    ],
    "mastitis": [
        "a cow's udder that is swollen, red and hot",
        "a cow's udder that is clearly asymmetric and inflamed in one quarter",
    ],
    # Add here once you have a description you trust, even with no training photos yet:
    # "ringworm": ["a cow with circular, crusty, hairless patches on the skin"],
    # "mange": ["a cow with rough, crusty, thickened skin and hair loss from mange mites"],
    # "warts": ["a cow with small, rough, cauliflower-like growths on the skin"],
}

CAUTION = ("Zero-shot visual screen only - no training photos were used for this disease, so this "
           "is less reliable than a trained classifier. Treat as a possible match to check, not a result.")

_cache = {}


def _model():
    if "m" not in _cache:
        import open_clip
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        model.to(device).eval()
        _cache["m"] = (model, preprocess, tokenizer, device)
    return _cache["m"]


def _all_prompts():
    """[(group_id, prompt), ...] - group_id is a disease id or a baseline id like 'healthy'."""
    out = []
    for gid, prompts in {**VISUAL_PROMPTS, **BASELINE_PROMPTS}.items():
        out.extend((gid, p) for p in prompts)
    return out


def _to_rgb(img):
    img = ImageOps.exif_transpose(img)
    try:
        return img.convert("RGB")
    except Exception:
        return img.convert("L").convert("RGB")


def screen(pil_img, top_k=3):
    """Returns ranked [{"id", "prompt_matched", "score", "caution"}], best match first.
    "id" is a disease id from VISUAL_PROMPTS, or "healthy" / "not_cattle" from the baseline.
    score is 0-1 (softmax over ALL prompts), so scores are comparable across diseases."""
    model, preprocess, tokenizer, device = _model()
    img = _to_rgb(pil_img)
    img.thumbnail((MAX_SIDE, MAX_SIDE))

    pairs = _all_prompts()
    text = tokenizer([p for _, p in pairs]).to(device)
    image = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        img_feat = model.encode_image(image)
        txt_feat = model.encode_text(text)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        sims = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1).squeeze(0).cpu()

    # collapse multiple prompts per disease down to its single best-matching phrasing
    best = {}
    for (gid, prompt), score in zip(pairs, sims.tolist()):
        if gid not in best or score > best[gid][1]:
            best[gid] = (prompt, score)

    ranked = sorted(best.items(), key=lambda kv: kv[1][1], reverse=True)
    return [
        {"id": gid, "prompt_matched": prompt, "score": round(score, 3),
         "caution": CAUTION if gid in VISUAL_PROMPTS else None}
        for gid, (prompt, score) in ranked[:top_k]
    ]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    args = ap.parse_args()
    for r in screen(Image.open(args.image)):
        tag = " [disease]" if r["caution"] else " [baseline]"
        print(f"{r['score']:.3f}  {r['id']:<24}{tag}  matched: {r['prompt_matched']}")
