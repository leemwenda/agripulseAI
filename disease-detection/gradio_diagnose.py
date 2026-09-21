#!/usr/bin/env python3
"""
Gradio diagnostic tool: upload a photo, see exactly what the cow detector
found and what crop the classifier actually judged, side by side with the
full probability breakdown.

Run from the disease-detection/ directory:
    python3 gradio_diagnose.py
"""
import sys
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent / "triage"))
import vision  # noqa: E402

DEFAULT_CHECKPOINT = str(Path(__file__).resolve().parent / "models" / "cattle_health_v5.pt")


def diagnose(img, checkpoint_path):
    if img is None:
        return None, None, "Upload a photo first."

    prepared = vision.prepare(img)
    box = vision.cow_box(prepared, fast=False)

    if box is None:
        return None, None, "No cow detected at all - this is a genuine detection miss, not a crop/background issue."

    boxed = prepared.copy()
    ImageDraw.Draw(boxed).rectangle(box, outline="red", width=4)

    crop = vision.pad_crop(prepared, box)
    if crop is None:
        crop = prepared

    ckpt = checkpoint_path.strip() or None
    result = vision.analyze(img, checkpoint=ckpt)

    lines = [f"Detected box: {tuple(round(v) for v in box)}"]
    if result["status"] == "ok":
        lines.append(f"Model version: {result['model_version']}")
        lines.append(f"Trained on cow crops: {result['cropped']}")
        lines.append("")
        lines.append("Probabilities:")
        for cls, p in sorted(result["probs"].items(), key=lambda kv: -kv[1]):
            lines.append(f"  {cls}: {p:.1%}")
        lines.append("")
        lines.append(f"Raw top class: {result['top']} ({result['top_prob']:.1%})")
        if result["thresholds"]:
            lines.append(f"Per-disease thresholds: {result['thresholds']}")
        # The REAL app verdict - this is what actually gets shown to a farmer.
        # analyze()'s own 'uncertain' field only checks the margin between the
        # top two classes; it does NOT apply the disease floor or per-disease
        # threshold. decide() is the only function that applies both, so it's
        # the only accurate way to know what the app would actually say.
        kind, label = vision.decide(result["probs"], result["margin"], result["thresholds"])
        lines.append("")
        lines.append(f">>> APP VERDICT: {kind.upper()}" + (f" ({label})" if kind != "uncertain" else ""))
    else:
        lines.append(f"Status: {result['status']}")

    return boxed, crop, "\n".join(lines)


with gr.Blocks(title="AgriPulse - Diagnose a Prediction") as demo:
    gr.Markdown("# Diagnose a Prediction")
    gr.Markdown(
        "Upload a photo to see exactly what the cow detector found and what crop "
        "the classifier actually judged - useful for figuring out WHY a prediction "
        "came out the way it did, not just what it was."
    )

    checkpoint_input = gr.Textbox(
        label="Checkpoint path", value=DEFAULT_CHECKPOINT,
        info="Change this to compare a different checkpoint on the same photo."
    )
    image_input = gr.Image(label="Photo", type="pil", sources=["upload", "webcam"])
    diagnose_btn = gr.Button("Diagnose", variant="primary")

    with gr.Row():
        boxed_output = gr.Image(label="Detected cow (red box)")
        crop_output = gr.Image(label="Exact crop the classifier judged")

    result_output = gr.Textbox(label="Full analysis", lines=12, interactive=False)

    diagnose_btn.click(
        diagnose,
        inputs=[image_input, checkpoint_input],
        outputs=[boxed_output, crop_output, result_output],
    )

if __name__ == "__main__":
    print("\nStarting on your LOCAL NETWORK only.")
    print("On your phone (same WiFi), open: http://<this-machine-IP>:7863")
    print("Find your IP with: hostname -I\n")
    demo.launch(server_name="0.0.0.0", server_port=7863)
