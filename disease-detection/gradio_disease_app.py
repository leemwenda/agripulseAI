#!/usr/bin/env python3
"""
AgriPulse cattle disease detection - Gradio UI with webcam support.

Two ways to use the camera, in one tab:
  1. Snapshot mode  - take/upload one photo, click "Diagnose Photo".
  2. Live mode       - toggle "Enable Live Monitoring" and point the camera
                        at a cow; a prediction refreshes automatically every
                        ~2 seconds (throttled so CPU inference can keep up).

Loads the checkpoint saved by src/training/train_baseline.py. Works for
however many classes that checkpoint has, since the class list and
uncertain_margin are read from the checkpoint itself.

The farmer-facing result deliberately shows NO raw probabilities and will
not name a disease on shaky evidence - see headline_from_probs() below.
Raw probabilities are still printed to the console for developer review.

Run from the disease-detection/ directory:
    python3 gradio_disease_app.py
"""
import time
from collections import deque
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

CHECKPOINT_PATH = Path(__file__).parent / "models" / "cattle_health_classifier.pt"
LIVE_REFRESH_SECONDS = 2.0  # throttle: don't run inference more often than this
CAPTURES_DIR = Path(__file__).parent / "captures"
CAPTURES_DIR.mkdir(exist_ok=True)

# A disease is only named when the model is BOTH clearly ahead of the
# runner-up class (uncertain_margin, from the checkpoint) AND above this
# absolute confidence floor. A coinflip-confidence disease label is more
# frightening than useful to a farmer - anything short of this floor
# defaults to "healthy" or "uncertain" instead of naming a disease.
DISEASE_CONFIDENCE_FLOOR = 0.80


def save_capture(img, headline):
    """Save every diagnosed image alongside its prediction, so real-world
    test photos can be reviewed later (e.g. pasted into a chat for
    debugging) instead of vanishing after the UI moves on."""
    if img is None:
        return
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # keep the headline short and filesystem-safe
    safe_label = "".join(c if c.isalnum() else "_" for c in headline)[:40]
    path = CAPTURES_DIR / f"{timestamp}_{safe_label}.jpg"
    try:
        img.convert("RGB").save(path, quality=90)
    except Exception as exc:
        print(f"Warning: failed to save capture: {exc}")

# The disease classifier only knows the classes it was trained on - it has
# NO concept of "not a cow", so it will confidently misclassify a hand, a
# wall, a dog, etc. as one of its known classes. This general-purpose COCO
# detector runs FIRST to check "is there actually a cow-like animal here"
# before we trust the disease prediction at all. COCO class index 21 = "cow"
# in the 91-class list torchvision's pretrained detection models use.
COCO_COW_CLASS_INDEX = 21
COW_DETECTION_SCORE_THRESHOLD = 0.4
_cow_detector = None


def get_cow_detector(device):
    """Faster R-CNN, not SSDLite - real farm photos are often partially
    occluded by fence rails, cropped tight on just the head, or shot from
    unusual angles (a lying calf from above). The lighter SSDLite detector
    missed real cows in exactly these conditions; Faster R-CNN is slower
    but noticeably more reliable at detecting a cow despite occlusion or
    an atypical pose - same fix already validated in the re-id app."""
    global _cow_detector
    if _cow_detector is None:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        _cow_detector = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device)
        _cow_detector.eval()
    return _cow_detector


def looks_like_cow(pil_img, device):
    """Returns True if a cow-shaped object is detected anywhere in the frame
    with reasonable confidence."""
    detector = get_cow_detector(device)
    tensor = transforms.functional.to_tensor(pil_img.convert("RGB")).to(device)
    with torch.no_grad():
        prediction = detector([tensor])[0]
    for label, score in zip(prediction["labels"], prediction["scores"]):
        if label.item() == COCO_COW_CLASS_INDEX and score.item() >= COW_DETECTION_SCORE_THRESHOLD:
            return True
    return False


def build_model(num_classes):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {path}. Train one first with "
            f"src/training/train_baseline.py."
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((ckpt["image_size"], ckpt["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, ckpt["classes"], ckpt.get("uncertain_margin", 0.15), tf, device


print(f"Loading checkpoint from {CHECKPOINT_PATH} ...")
_model, _classes, _margin, _transform, _device = load_checkpoint(CHECKPOINT_PATH)
print(f"Loaded. Classes: {_classes}")

_last_live_run = 0.0  # module-level timestamp for throttling live mode

_LIVE_SMOOTHING_WINDOW = 3  # average this many recent frames before showing a live result
_live_prob_buffer = deque(maxlen=_LIVE_SMOOTHING_WINDOW)


def headline_from_probs(prob_dict):
    """Shared logic: turn a {label: prob} dict into a plain, non-alarming
    result for the farmer.

    Deliberately shows NO raw percentages - a borderline confidence number
    reads as alarming and isn't actionable for someone without an ML
    background. A disease is only named when the model is both clearly
    ahead of the runner-up class (uncertain_margin) AND above an absolute
    confidence floor (DISEASE_CONFIDENCE_FLOOR); anything short of that
    defaults to "healthy" or "uncertain" rather than a scary disease label
    on shaky evidence. Raw probabilities are still printed to the console
    for developer review - see diagnose_snapshot/diagnose_live.
    """
    sorted_probs = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_prob = sorted_probs[0]
    second_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
    margin_ok = (top_prob - second_prob) >= _margin

    if not margin_ok:
        return "Uncertain - move closer / improve lighting and try again."

    if top_label == "healthy":
        return "Healthy"

    if top_prob >= DISEASE_CONFIDENCE_FLOOR:
        pretty = top_label.replace("_", " ").title()
        return f"Possible {pretty} - have a vet confirm."

    return "Uncertain - move closer / improve lighting and try again."


def raw_predict_probs(img):
    """One forward pass through the classifier only (no cow-check, no
    headline formatting). Returns {label: prob, ...} or None if no cow
    was detected in the frame."""
    if img is None or not looks_like_cow(img, _device):
        return None
    tensor = _transform(img.convert("RGB")).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    return {cls: float(p) for cls, p in zip(_classes, probs)}


def predict_pil(img):
    """Single-shot prediction (snapshot mode) - one frame, no smoothing,
    since a deliberately-taken photo is already the best-quality input
    we're going to get. Returns just the headline text - no probabilities
    are surfaced to the farmer-facing UI."""
    if img is None:
        return "No image."
    prob_dict = raw_predict_probs(img)
    if prob_dict is None:
        return "No cow detected - point the camera at cattle."
    print(f"[snapshot] raw probabilities: {prob_dict}")  # developer-only, console
    return headline_from_probs(prob_dict)


def diagnose_snapshot(img):
    headline = predict_pil(img)
    save_capture(img, headline)
    return headline


def diagnose_live(img, live_enabled):
    """Bound to the streaming webcam's .stream() event - fires on every
    incoming frame, but we throttle actual inference to LIVE_REFRESH_SECONDS
    so a slow CPU doesn't fall permanently behind the video feed.

    Live webcam frames are noisier than a deliberate snapshot - motion
    blur, autofocus hunting, lower per-frame resolution - so a single
    frame's prediction can flicker between classes. To compensate, we
    average probabilities over the last few detected-cow frames
    (_LIVE_SMOOTHING_WINDOW) before deciding on a headline, rather than
    trusting any single noisy frame in isolation."""
    global _last_live_run
    if not live_enabled or img is None:
        _live_prob_buffer.clear()
        return gr.update()

    now = time.time()
    if now - _last_live_run < LIVE_REFRESH_SECONDS:
        return gr.update()  # skip this frame, not time yet
    _last_live_run = now

    prob_dict = raw_predict_probs(img)
    if prob_dict is None:
        _live_prob_buffer.clear()
        return "No cow detected - point the camera at cattle."

    headline = headline_from_probs(prob_dict)  # single-frame label, for the capture filename only
    save_capture(img, headline)
    _live_prob_buffer.append(prob_dict)

    # Average across whatever frames we've accumulated so far (up to
    # _LIVE_SMOOTHING_WINDOW). Needs >=2 frames before showing a real
    # verdict, so a single lucky/unlucky frame can't swing the display.
    if len(_live_prob_buffer) < 2:
        return "Analyzing - hold steady..."

    averaged = {
        cls: sum(frame[cls] for frame in _live_prob_buffer) / len(_live_prob_buffer)
        for cls in _classes
    }
    print(f"[live] averaged probabilities: {averaged}")  # developer-only, console
    return headline_from_probs(averaged)


with gr.Blocks(title="AgriPulse - Cattle Disease Detection") as demo:
    gr.Markdown("# Cattle Disease Detection")
    gr.Markdown(
        f"Classes this model recognizes: {', '.join(_classes)}. "
        "This is a screening aid, not a diagnosis - always confirm with a vet."
    )

    with gr.Tab("Snapshot"):
        gr.Markdown("Take or upload one photo, then diagnose it.")
        snap_image = gr.Image(label="Photo", type="pil", sources=["upload", "webcam"])
        snap_btn = gr.Button("Diagnose Photo", variant="primary")
        snap_headline = gr.Textbox(label="Result", interactive=False)
        snap_btn.click(diagnose_snapshot, inputs=[snap_image], outputs=[snap_headline])

    with gr.Tab("Live Monitoring"):
        gr.Markdown(
            f"Point the camera at a cow and enable live monitoring. "
            f"A new prediction runs roughly every {LIVE_REFRESH_SECONDS:.0f} seconds "
            f"(throttled to keep up with CPU speed)."
        )
        live_enabled_checkbox = gr.Checkbox(label="Enable Live Monitoring", value=False)
        live_image = gr.Image(label="Live Camera", type="pil", sources=["webcam"], streaming=True)
        live_headline = gr.Textbox(label="Live Result", interactive=False)

        live_image.stream(
            fn=diagnose_live,
            inputs=[live_image, live_enabled_checkbox],
            outputs=[live_headline],
            stream_every=0.5,  # gradio calls this often; our own throttle inside
                                 # diagnose_live() decides when to actually run inference
        )

if __name__ == "__main__":
    print("\nStarting on your LOCAL NETWORK only.")
    print("On your phone (same WiFi), open: http://<this-machine-IP>:7861")
    print("Find your IP with: hostname -I\n")
    demo.launch(server_name="0.0.0.0", server_port=7861)
