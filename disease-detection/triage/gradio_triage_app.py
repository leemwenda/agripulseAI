#!/usr/bin/env python3
"""
AgriPulse cattle detection - AUTOMATIC photo detection + symptom triage.

Same layout as the disease app (Snapshot and Live Monitoring tabs), but the result
is automatic and covers more than one disease:

  Snapshot tab   - add or take a photo and the result appears by itself. No button.
  Live tab       - point the camera at a cow and tick "Enable Live Monitoring"; the result
                   refreshes every ~2 s (smoothed over the last few frames).

For each photo the app (1) checks a cow is in the frame, (2) runs the cattle health
classifier, and (3) feeds anything it sees (e.g. lumpy skin -> "skin nodules") into the
symptom triage in triage/diseases/*.yaml, so the farmer gets a ranked list with urgency,
reportable-disease alerts and what to do.

A photo can only reveal what is visible on the skin. Every other disease in the knowledge
base needs signs from the "Add symptoms you can see" panel (fever, swollen lymph nodes,
breathing, etc.) - the ranked result updates as soon as you press "Update result".

Like the disease app, no raw probabilities are shown to the user (they go to the console),
and a disease is only named when the model is clearly ahead AND above a confidence floor.

Run:
    python3 triage/gradio_triage_app.py
Open http://127.0.0.1:7862  (phone on the same WiFi: http://<this-machine-IP>:7862).
Port 7862, so it never clashes with the other apps.
"""
import importlib.util
import time
from collections import deque
from pathlib import Path

import gradio as gr

HERE = Path(__file__).parent
CAPTURES_DIR = HERE.parent / "captures"        # shared with the disease app: real photos to sort later

DISEASE_CONFIDENCE_FLOOR = 0.80   # same rule as the disease app
LIVE_REFRESH_SECONDS = 2.0        # never run live inference more often than this
LIVE_SMOOTHING_WINDOW = 3         # average this many recent frames before showing a live verdict

UNCERTAIN_MSG = "Uncertain - move closer / improve lighting and try again."
NO_COW_MSG = "No cow detected - point the camera at cattle."
NO_SIGNS_HINT = ("Nothing to rank from the photo alone. Open **Add symptoms you can see** and tick what you "
                 "observe, then press **Update result**.")


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


engine = _load_module("triage_engine", "triage.py")

_vision = None
_vision_error = None


def get_vision():
    """Load vision.py on first use so the checklist still works if torch or the models are missing."""
    global _vision, _vision_error
    if _vision is None and _vision_error is None:
        try:
            _vision = _load_module("triage_vision", "vision.py")
        except Exception as exc:
            _vision_error = f"{type(exc).__name__}: {exc}"
    if _vision is None:
        raise RuntimeError(_vision_error)
    return _vision


BAND_ICON = {"likely": "\U0001F534", "possible": "\U0001F7E1"}


# ---------------------------------------------------------------- knowledge base helpers
def sign_choices(vocab):
    return [(v["ask"], s) for s, v in vocab.items()]


def photo_sign_map(vocab):
    """{photo model class -> sign id}, read from `from_photo_class` in symptoms.yaml."""
    return {v["from_photo_class"]: s for s, v in vocab.items() if v.get("from_photo_class")}


def coverage_text(vocab, diseases):
    photo = ", ".join(c.replace("_", " ") for c in photo_sign_map(vocab)) or "none yet"
    return (f"**Automatic photo detection covers:** {photo}.  \n"
            f"**Symptom triage covers ({len(diseases)}):** {', '.join(d['name'] for d in diseases)}.")


# ---------------------------------------------------------------- verdict logic (mirrors the disease app)
def verdict(probs, margin, classes):
    """-> (kind, top_label): kind is 'uncertain', 'healthy' or 'disease'."""
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top, top_p = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if (top_p - second) < margin:
        return "uncertain", top
    if top == "healthy":
        return "healthy", top
    if top_p >= DISEASE_CONFIDENCE_FLOOR:
        return "disease", top
    return "uncertain", top


def headline_for(kind, top, classes):
    if kind == "uncertain":
        return UNCERTAIN_MSG
    if kind == "healthy":
        looked_for = ", ".join(c.replace("_", " ") for c in classes if c != "healthy") or "disease"
        return f"No signs of {looked_for} seen in this photo. This does not rule out other diseases."
    return f"Possible {top.replace('_', ' ').title()} - have a vet confirm."


def photo_signs_for(kind, top, sign_map):
    return [sign_map[top]] if kind == "disease" and top in sign_map else []


# ---------------------------------------------------------------- triage rendering
def render(results):
    if not results:
        return ("**No disease in the knowledge base matches these signs strongly enough.**\n\n"
                "That does *not* mean the animal is healthy. If it is unwell, call a vet.\n\n"
                f"_{engine.DISCLAIMER}_")
    parts = []
    for i, r in enumerate(results, 1):
        icon = BAND_ICON.get(r["band"], "")
        block = [f"### {i}. {r['name']} - {icon} {r['band'].upper()} (score {r['score']}) - urgency: **{r['urgency']}**"]
        if r["notifiable"]:
            block.append("**Suspected reportable disease - notify the county veterinary officer / veterinary "
                         "authority before moving any animals or products.**")
        if r["zoonotic"]:
            block.append("**Warning: this disease can infect people.** Use protective gear and wash thoroughly.")
        if r["summary"]:
            block.append(f"> {r['summary']}")
        block.append("**Matched signs:** " + ", ".join(s.replace("_", " ") for s in r["matched_signs"]))
        if r["check_next"]:
            block.append("**Check next:**\n\n" + "\n".join(f"- {c['ask']}" for c in r["check_next"]))
        if r["differentials"]:
            block.append("**Also consider:** " + "; ".join(r["differentials"]))
        if r["confirm_with"]:
            block.append(f"**Confirmed by:** {r['confirm_with']}")
        block.append("**What to do:**\n\n" + "\n".join(f"- {a}" for a in r["advice"]))
        parts.append("\n\n".join(block))
    parts.append(f"---\n_{engine.DISCLAIMER}_")
    return "\n\n".join(parts)


def triage_markdown(photo_signs, manual_signs):
    signs = list(dict.fromkeys(list(photo_signs or []) + list(manual_signs or [])))
    if not signs:
        return NO_SIGNS_HINT
    try:
        results = engine.rank(engine.load_kb(HERE), signs)   # re-read every time: edited files apply at once
    except engine.KBError as exc:
        return f"**Knowledge base problem:** {exc}"
    prefix = ""
    if photo_signs:
        prefix = "Sign taken from the photo: **" + ", ".join(s.replace("_", " ") for s in photo_signs) + "**\n\n"
    return prefix + render(results)


# ---------------------------------------------------------------- captures
def save_capture(img, label):
    """Keep analysed photos (label in the filename) so real-world photos can be sorted into a test set."""
    if img is None:
        return
    try:
        CAPTURES_DIR.mkdir(exist_ok=True)
        safe = "".join(c if c.isalnum() else "_" for c in label)[:40]
        img.convert("RGB").save(CAPTURES_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_triage_{safe}.jpg", quality=90)
    except Exception as exc:
        print(f"Warning: could not save capture: {exc}")


# ---------------------------------------------------------------- Snapshot tab (automatic on photo)
def on_photo(img, manual):
    """Runs by itself whenever a photo is added, replaced or cleared."""
    if img is None:
        return "", [], ""
    try:
        res = get_vision().analyze(img, fast=False)
    except Exception as exc:
        return (f"Photo analysis is not available right now ({exc}). You can still use the symptom panel below.",
                [], triage_markdown([], manual))
    if res["status"] == "no_cow":
        save_capture(img, "no_cow")
        return NO_COW_MSG, [], triage_markdown([], manual)
    print(f"[snapshot] raw probabilities: {res['probs']}")            # developer-only, console
    kind, top = verdict(res["probs"], res["margin"], res["classes"])
    save_capture(img, f"{kind}_{top}")
    try:
        sign_map = photo_sign_map(engine.load_kb(HERE)[0])
    except engine.KBError as exc:
        return f"Knowledge base problem: {exc}", [], ""
    signs = photo_signs_for(kind, top, sign_map)
    return headline_for(kind, top, res["classes"]), signs, triage_markdown(signs, manual)


def on_refine(photo_signs, manual):
    return triage_markdown(photo_signs, manual)


# ---------------------------------------------------------------- Live tab (automatic, throttled + smoothed)
_last_live_run = 0.0
_live_buffer = deque(maxlen=LIVE_SMOOTHING_WINDOW)


def diagnose_live(img, live_enabled):
    global _last_live_run
    if not live_enabled or img is None:
        _live_buffer.clear()
        return gr.update(), gr.update()
    now = time.time()
    if now - _last_live_run < LIVE_REFRESH_SECONDS:
        return gr.update(), gr.update()          # not time yet
    _last_live_run = now
    try:
        res = get_vision().analyze(img, fast=True)
    except Exception as exc:
        return f"Photo analysis is not available right now ({exc}).", ""
    if res["status"] == "no_cow":
        _live_buffer.clear()
        return NO_COW_MSG, ""
    _live_buffer.append(res["probs"])
    if len(_live_buffer) < 2:
        return "Analyzing - hold steady...", gr.update()
    averaged = {c: sum(f[c] for f in _live_buffer) / len(_live_buffer) for c in res["classes"]}
    print(f"[live] averaged probabilities: {averaged}")                 # developer-only, console
    kind, top = verdict(averaged, res["margin"], res["classes"])
    if kind == "disease":
        save_capture(img, f"live_{top}")
    try:
        sign_map = photo_sign_map(engine.load_kb(HERE)[0])
    except engine.KBError as exc:
        return f"Knowledge base problem: {exc}", ""
    signs = photo_signs_for(kind, top, sign_map)
    return headline_for(kind, top, res["classes"]), (triage_markdown(signs, []) if signs else "")


# ---------------------------------------------------------------- knowledge-base reload
def reload_kb():
    try:
        vocab, diseases = engine.load_kb(HERE)
    except engine.KBError as exc:
        return gr.update(), f"**Knowledge base problem:** {exc}"
    return gr.update(choices=sign_choices(vocab), value=[]), coverage_text(vocab, diseases)


_vocab, _diseases = engine.load_kb(HERE)

with gr.Blocks(title="AgriPulse - Cattle Detection") as demo:
    gr.Markdown("# Cattle disease detection")
    gr.Markdown("Automatic photo detection plus symptom triage. This is a screening aid, not a diagnosis - "
                "always confirm with a vet.")
    coverage = gr.Markdown(coverage_text(_vocab, _diseases))
    photo_signs = gr.State([])

    with gr.Tab("Snapshot"):
        gr.Markdown("Add or take a photo - the result appears automatically.")
        snap_image = gr.Image(label="Photo", type="pil", sources=["upload", "webcam"])
        snap_headline = gr.Textbox(label="Result", interactive=False)
        with gr.Accordion("Add symptoms you can see (optional)", open=False):
            manual_signs = gr.CheckboxGroup(choices=sign_choices(_vocab), label="Signs observed")
            with gr.Row():
                refine_btn = gr.Button("Update result", variant="primary")
                reload_btn = gr.Button("Reload knowledge base")
        snap_triage = gr.Markdown()

        snap_image.change(on_photo, inputs=[snap_image, manual_signs],
                          outputs=[snap_headline, photo_signs, snap_triage], api_name="snapshot")
        refine_btn.click(on_refine, inputs=[photo_signs, manual_signs], outputs=[snap_triage], api_name="refine")
        reload_btn.click(reload_kb, outputs=[manual_signs, coverage])

    with gr.Tab("Live Monitoring"):
        gr.Markdown(f"Point the camera at a cow and enable live monitoring. A new result appears roughly every "
                    f"{LIVE_REFRESH_SECONDS:.0f} seconds (throttled to keep up with CPU speed).")
        live_enabled = gr.Checkbox(label="Enable Live Monitoring", value=False)
        live_image = gr.Image(label="Live Camera", type="pil", sources=["webcam"], streaming=True)
        live_headline = gr.Textbox(label="Live Result", interactive=False)
        live_triage = gr.Markdown()
        live_image.stream(fn=diagnose_live, inputs=[live_image, live_enabled],
                          outputs=[live_headline, live_triage], stream_every=0.5)

if __name__ == "__main__":
    print("\nCattle detection running.")
    print("This machine:  http://127.0.0.1:7862")
    print("Phone (same WiFi):  http://<this-machine-IP>:7862   (find IP with: hostname -I)\n")
    demo.launch(server_name="0.0.0.0", server_port=7862)
