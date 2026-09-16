# Run this as a NEW CELL in your Colab notebook, AFTER your training loop (Section 7)
# has already run and `model`, `transform`, `device` exist in memory.
#
# This launches a Gradio UI directly from Colab and gives you a public link
# you can open in your Pop!_OS browser - no need to save/download a checkpoint.

!pip install gradio -q

import json
import gradio as gr
import torch
import torch.nn.functional as F

DATABASE = {}  # {cow_id: [embedding_tensor, ...]}  -- lives only in this Colab session
MATCH_THRESHOLD = 0.5

model.eval()

def get_embedding(pil_image):
    img = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img)
    return emb.squeeze(0).cpu()

def registered_cows_list():
    if not DATABASE:
        return "No cows registered yet."
    return "\n".join(f"- {cow_id}: {len(embs)} reference photo(s)"
                      for cow_id, embs in sorted(DATABASE.items()))

def register_cow_ui(cow_id, images):
    if not cow_id or not cow_id.strip():
        return "Enter a cow ID first.", registered_cows_list()
    if not images:
        return "Upload at least one photo.", registered_cows_list()
    cow_id = cow_id.strip()
    from PIL import Image
    new_embeddings = [get_embedding(Image.open(img.name)) for img in images]
    DATABASE.setdefault(cow_id, []).extend(new_embeddings)
    return (f"Registered {len(images)} photo(s) for '{cow_id}'. "
            f"Total reference photos now: {len(DATABASE[cow_id])}"), registered_cows_list()

def identify_cow_ui(image):
    if image is None:
        return "Upload a photo first.", ""
    if not DATABASE:
        return "No cows registered yet - register some first.", ""
    query_emb = get_embedding(image)
    scores = {}
    for cow_id, ref_embeddings in DATABASE.items():
        sims = [F.cosine_similarity(query_emb.unsqueeze(0), ref.unsqueeze(0)).item()
                for ref in ref_embeddings]
        scores[cow_id] = sum(sims) / len(sims)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    best_cow, best_sim = ranked[0]
    breakdown = "\n".join(f"  {cow_id}: {sim:.1%}" for cow_id, sim in ranked)
    if best_sim >= MATCH_THRESHOLD:
        result = f"Best match: {best_cow}  (confidence: {best_sim:.1%})"
    else:
        result = (f"No confident match (best guess {best_cow} at {best_sim:.1%}, "
                  f"below {MATCH_THRESHOLD:.0%} threshold). Flag for farmer confirmation.")
    return result, breakdown

with gr.Blocks(title="AgriPulse Cow Re-ID") as demo:
    gr.Markdown("# AgriPulse — Cow Re-Identification Test UI (running in Colab)")
    gr.Markdown(f"Model: ResNet50 + triplet loss embeddings | Device: {device}")

    with gr.Tab("Register Cow"):
        cow_id_input = gr.Textbox(label="Cow ID", placeholder="e.g. cow_001")
        register_images = gr.File(label="Photo(s)", file_count="multiple", file_types=["image"])
        register_btn = gr.Button("Register", variant="primary")
        register_output = gr.Textbox(label="Result", interactive=False)
        registered_list = gr.Textbox(label="Currently registered cows", interactive=False,
                                      value=registered_cows_list())
        register_btn.click(register_cow_ui, inputs=[cow_id_input, register_images],
                           outputs=[register_output, registered_list])

    with gr.Tab("Identify Cow"):
        identify_image = gr.Image(label="Photo to identify", type="pil")
        identify_btn = gr.Button("Identify", variant="primary")
        identify_result = gr.Textbox(label="Result", interactive=False)
        identify_breakdown = gr.Textbox(label="All similarity scores", interactive=False, lines=6)
        identify_btn.click(identify_cow_ui, inputs=[identify_image],
                           outputs=[identify_result, identify_breakdown])

demo.launch(share=True)  # share=True gives a public link that works outside Colab's network
