#!/usr/bin/env python3
"""
AgriPulse cow re-identification - Gradio test UI (with data collection + mobile access).

Every photo you register or confirm during identification gets automatically
saved into a growing dataset folder, organized by cow ID - so using this UI
builds real training data over time, not just tests the model.

The "Group Photos" tab lets you upload a batch of unsorted photos and have
the model cluster them by visual similarity (using the same embeddings as
identification), instead of eyeballing them yourself.

MOBILE ACCESS:
  Default (LAN):  python3 gradio_app.py
                  -> open http://<this-machine's-IP>:7860 on your phone,
                     same WiFi network. Find your IP with: hostname -I

  Public link:    SHARE=1 python3 gradio_app.py
                  -> prints a https://xxxxx.gradio.live link, works on any
                     network (WiFi or mobile data). Temporary, expires when
                     you stop the script. Don't leave it running unattended.

Run: python3 gradio_app.py
"""
import json
import os
from datetime import datetime

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

CHECKPOINT_PATH = "cow_embedding_resnet50_v1.pt"
DATABASE_PATH = "cow_database.json"
MATCH_THRESHOLD = 0.5
COLLECTED_DATA_DIR = os.path.expanduser("~/agripulse-ai/collected_data")


class CowEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.embedding = nn.Linear(2048, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)


transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

model = CowEmbeddingNet().to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def get_embedding(pil_image):
    img = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img)
    return emb.squeeze(0).cpu()


def _file_path(img_file):
    """gr.File's return type has changed across Gradio versions - older
    versions return an object with a `.name` attribute pointing to a temp
    file; Gradio 5/6+ returns the temp file path directly as a plain
    string. Support both so this keeps working regardless of installed
    Gradio version."""
    return getattr(img_file, "name", img_file)


def save_to_collected_data(cow_id, pil_image):
    """Saves a photo into the growing dataset, organized by cow ID."""
    cow_dir = os.path.join(COLLECTED_DATA_DIR, cow_id)
    os.makedirs(cow_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dst = os.path.join(cow_dir, f"{timestamp}.jpg")
    pil_image.convert("RGB").save(dst, "JPEG")
    return dst


def load_database():
    if not os.path.exists(DATABASE_PATH):
        return {}
    with open(DATABASE_PATH) as f:
        raw = json.load(f)
    return {cow_id: [torch.tensor(e) for e in embs] for cow_id, embs in raw.items()}


def save_database(database):
    raw = {cow_id: [e.tolist() for e in embs] for cow_id, embs in database.items()}
    with open(DATABASE_PATH, "w") as f:
        json.dump(raw, f)


def registered_cows_list():
    db = load_database()
    if not db:
        return "No cows registered yet."
    lines = [f"- {cow_id}: {len(embs)} reference photo(s)" for cow_id, embs in sorted(db.items())]
    return "\n".join(lines)


def register_cow_ui(cow_id, images, reject_non_cow=True):
    if not cow_id or not cow_id.strip():
        return "Enter a cow ID first.", registered_cows_list()
    if not images:
        return "Upload at least one photo.", registered_cows_list()

    cow_id = cow_id.strip()
    database = load_database()
    new_embeddings = []
    rejected_names = []
    error_names = []

    for img_file in images:
        name = os.path.basename(_file_path(img_file))
        try:
            pil_img = Image.open(_file_path(img_file))
            pil_img.load()  # force full decode now, so a truncated file fails here
            pil_img = pil_img.convert("RGB")
        except Exception as exc:
            error_names.append(f"{name}: {exc}")
            continue

        if reject_non_cow:
            try:
                is_cow, top_label = looks_like_cow(pil_img)
            except Exception:
                # If the detector itself misbehaves, don't block on it.
                is_cow, top_label = True, None
            if not is_cow:
                guess = top_label or "something else"
                rejected_names.append(f"{name}: looks like '{guess}', not cattle")
                continue

        new_embeddings.append(get_embedding(pil_img))
        save_to_collected_data(cow_id, pil_img)  # <-- collects data automatically

    if new_embeddings:
        database.setdefault(cow_id, []).extend(new_embeddings)
        save_database(database)

    result_lines = []
    if new_embeddings:
        result_lines.append(f"Registered {len(new_embeddings)} photo(s) for '{cow_id}' "
                            f"(also saved to collected_data/{cow_id}/). "
                            f"Total reference photos now: {len(database[cow_id])}")
    else:
        result_lines.append(f"Nothing registered for '{cow_id}' - all photos were filtered out.")

    if rejected_names:
        result_lines.append("")
        result_lines.append(f"Rejected {len(rejected_names)} non-cattle photo(s):")
        result_lines.extend(f"  {r}" for r in rejected_names)
    if error_names:
        result_lines.append("")
        result_lines.append(f"Skipped {len(error_names)} unreadable file(s):")
        result_lines.extend(f"  {e}" for e in error_names)

    return "\n".join(result_lines), registered_cows_list()


def identify_cow_ui(image, reject_non_cow=True):
    if image is None:
        return "Upload a photo first.", "", gr.update(visible=False), gr.update(choices=[])

    if reject_non_cow:
        try:
            is_cow, top_label = looks_like_cow(image)
        except Exception:
            is_cow, top_label = True, None
        if not is_cow:
            guess = top_label or "something else"
            return (f"This doesn't look like cattle (looks like '{guess}') - not identifying.",
                    "", gr.update(visible=False), gr.update(choices=[]))

    database = load_database()
    if not database:
        return "No cows registered yet - register some first.", "", gr.update(visible=False), gr.update(choices=[])

    query_emb = get_embedding(image)

    scores = {}
    for cow_id, ref_embeddings in database.items():
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
                  f"below {MATCH_THRESHOLD:.0%} threshold). Please confirm below.")

    cow_choices = sorted(database.keys())
    return result, breakdown, gr.update(visible=True), gr.update(choices=cow_choices, value=best_cow)


def confirm_identification(image, confirmed_cow_id):
    """When the farmer confirms which cow it actually was, save the photo
    as new training data AND add it as a new reference photo for that cow."""
    if image is None or not confirmed_cow_id:
        return "Nothing to confirm."

    save_to_collected_data(confirmed_cow_id, image)

    database = load_database()
    new_emb = get_embedding(image)
    database.setdefault(confirmed_cow_id, []).append(new_emb)
    save_database(database)

    return (f"Confirmed as '{confirmed_cow_id}'. Photo saved to collected_data/ "
            f"and added as a new reference photo (model gets a bit stronger each time).")


# ---------------------------------------------------------------------------
# Group Photos: upload an unsorted batch, cluster by embedding similarity,
# then let the farmer assign a cow ID to each resulting group and save.
# ---------------------------------------------------------------------------

_detector = None
_detector_transform = None
_detector_categories = None
NON_COW_SCORE_THRESHOLD = 0.4
PENDING_GROUPS_DIR = os.path.join(COLLECTED_DATA_DIR, "_pending_groups")


def _load_detector():
    """Lazily loads a general-purpose COCO-pretrained object detector the
    first time it's needed (downloads weights on first run if not cached).
    This is separate from the re-id model - it only tells cattle apart from
    humans/other objects, it never compares one cow to another."""
    global _detector, _detector_transform, _detector_categories
    if _detector is not None:
        return
    import torchvision.models.detection as detection_models
    weights = detection_models.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    _detector = detection_models.ssdlite320_mobilenet_v3_large(weights=weights).to(device)
    _detector.eval()
    _detector_transform = weights.transforms()
    _detector_categories = weights.meta["categories"]


def classify_photo_content(pil_image, score_threshold=NON_COW_SCORE_THRESHOLD):
    """Runs the detector and reports whether a 'cow' was found, plus the
    single highest-confidence label overall (for a helpful rejection
    message if it wasn't a cow). A COCO detector directly distinguishes
    cow/person/other-object, rather than forcing one label onto the whole
    photo the way a plain image classifier does - much more robust to
    close-up or unusual-angle cattle photos."""
    _load_detector()
    img = _detector_transform(pil_image).to(device)
    with torch.no_grad():
        output = _detector([img])[0]

    has_cow = False
    top_label = None
    top_score = 0.0
    for label_idx, score in zip(output["labels"].tolist(), output["scores"].tolist()):
        if score < score_threshold:
            continue
        label = _detector_categories[label_idx]
        if label == "cow":
            has_cow = True
        if score > top_score:
            top_score = score
            top_label = label

    return has_cow, top_label


def looks_like_cow(pil_image):
    """Accepts the photo unless the detector is confident about something
    specific that ISN'T a cow (a person, another animal, an object).
    Ambiguous close-ups with no confident detection at all are given the
    benefit of the doubt rather than rejected outright."""
    has_cow, top_label = classify_photo_content(pil_image)
    if has_cow:
        return True, top_label
    if top_label is None:
        return True, None  # nothing confidently detected - don't block on it
    return False, top_label


def save_pending_group_photos(session_id, groups, pil_images):
    """Auto-saves every grouped photo to a review folder right after
    grouping - organized by group NUMBER, not by cow ID, since no cow ID
    exists yet at this point. This never touches cow_database.json; that
    only happens once a real cow ID is assigned to a group and saved."""
    session_dir = os.path.join(PENDING_GROUPS_DIR, session_id)
    for gi, group in enumerate(groups, start=1):
        group_dir = os.path.join(session_dir, f"group_{gi}")
        os.makedirs(group_dir, exist_ok=True)
        for n, idx in enumerate(group):
            dst = os.path.join(group_dir, f"{n:03d}.jpg")
            pil_images[idx].convert("RGB").save(dst, "JPEG")
    return session_dir


def compute_embeddings_batch(image_files, reject_non_cow=True):
    """Loads each uploaded file once and returns (pil_images, embeddings,
    errors, rejected), all in matching order for the kept images.
    `rejected` holds (pil_image, reason) pairs so the rejected photos can
    still be shown, not just named. A single unreadable/corrupt photo - or,
    if reject_non_cow is set, a photo that doesn't look like cattle - is
    skipped instead of breaking the batch."""
    pil_images = []
    embeddings = []
    errors = []
    rejected = []
    for img_file in image_files:
        name = os.path.basename(_file_path(img_file))
        try:
            pil_img = Image.open(_file_path(img_file))
            pil_img.load()  # force full decode now, so a truncated file fails here
            pil_img = pil_img.convert("RGB")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue

        if reject_non_cow:
            try:
                is_cow, top_label = looks_like_cow(pil_img)
            except Exception:
                # If the detector itself misbehaves, don't block on it.
                is_cow, top_label = True, None
            if not is_cow:
                guess = top_label or "something else"
                rejected.append((pil_img, f"{name}: looks like '{guess}', not cattle"))
                continue

        try:
            emb = get_embedding(pil_img)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue

        pil_images.append(pil_img)
        embeddings.append(emb)
    return pil_images, embeddings, errors, rejected


def cluster_by_similarity(embeddings, threshold):
    """Complete-linkage agglomerative clustering: two clusters only merge if
    EVERY pair of photos across them meets the similarity threshold.

    This deliberately avoids single-linkage "chaining": if photo A is similar
    to B, and B is similar to C, single-linkage lumps A and C together even
    though A and C might not look alike at all - with a large batch this
    quickly collapses everything into one or two giant, wrong groups.
    Complete-linkage requires the whole group to mutually agree, which is
    much closer to "these are actually the same cow."
    """
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    sim = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
            sim[i][j] = sim[j][i] = s

    clusters = [[i] for i in range(n)]

    def cluster_min_sim(a, b):
        return min(sim[i][j] for i in a for j in b)

    while len(clusters) > 1:
        best_pair = None
        best_score = threshold  # only merge if the worst pair still clears the bar
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                score = cluster_min_sim(clusters[i], clusters[j])
                if score >= best_score:
                    best_score = score
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]

    return sorted(clusters, key=len, reverse=True)


def group_photos_ui(images, threshold, reject_non_cow):
    if not images:
        return ([], "Upload photos first.", gr.update(visible=False), None,
                gr.update(value=[], visible=False), "")

    try:
        pil_images, embeddings, errors, rejected = compute_embeddings_batch(
            images, reject_non_cow=reject_non_cow)

        rejected_gallery = [(img, reason) for img, reason in rejected]

        if not embeddings:
            msg_lines = ["Nothing left to group."]
            if rejected:
                msg_lines.append(f"\nRejected {len(rejected)} non-cattle photo(s) - see gallery below.")
            if errors:
                msg_lines.append(f"\nSkipped {len(errors)} unreadable file(s):")
                msg_lines.extend(f"  {e}" for e in errors)
            return ([], "\n".join(msg_lines), gr.update(visible=False), None,
                    gr.update(value=rejected_gallery, visible=bool(rejected)), "")

        groups = cluster_by_similarity(embeddings, threshold)

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = save_pending_group_photos(session_id, groups, pil_images)

        gallery_items = []
        summary_lines = [f"Auto-saved raw groups to: {session_dir}", ""]
        for gi, group in enumerate(groups, start=1):
            summary_lines.append(f"Group {gi}: {len(group)} photo(s)")
            for idx in group:
                gallery_items.append((pil_images[idx], f"Group {gi}"))

        if rejected:
            summary_lines.append("")
            summary_lines.append(f"Rejected {len(rejected)} non-cattle photo(s) - see gallery below.")
        if errors:
            summary_lines.append("")
            summary_lines.append(f"Skipped {len(errors)} unreadable file(s):")
            summary_lines.extend(f"  {e}" for e in errors)

        summary = "\n".join(summary_lines)
        state = {"pil_images": pil_images, "embeddings": embeddings, "groups": groups}

        # Pre-fill one suggested, session-traceable ID per group - edit these
        # to real cow IDs before saving. Nothing is registered until you save.
        suggested_assignments = "\n".join(
            f"{gi}: new_{session_id}_g{gi}" for gi in range(1, len(groups) + 1))

        return (gallery_items, summary, gr.update(visible=True), state,
                gr.update(value=rejected_gallery, visible=bool(rejected)), suggested_assignments)
    except Exception as exc:
        # Never let a bad batch take down the whole app - surface the error instead.
        return ([], f"Grouping failed: {exc}", gr.update(visible=False), None,
                gr.update(value=[], visible=False), "")


def save_grouped_photos_ui(assignment_text, state):
    if not state:
        return "Run grouping first."

    try:
        pil_images = state["pil_images"]
        embeddings = state["embeddings"]
        groups = state["groups"]

        # Parse lines like "1: AGP-00127"
        assignments = {}
        for line in assignment_text.strip().splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            num_part, cow_id = line.split(":", 1)
            num_part = num_part.strip()
            cow_id = cow_id.strip()
            if num_part.isdigit() and cow_id:
                assignments[int(num_part)] = cow_id

        if not assignments:
            return "No valid assignments found. Use lines like: 1: AGP-00127"

        database = load_database()
        results = []
        for group_num, cow_id in sorted(assignments.items()):
            if group_num < 1 or group_num > len(groups):
                results.append(f"Group {group_num}: doesn't exist, skipped.")
                continue
            indices = groups[group_num - 1]
            for idx in indices:
                try:
                    save_to_collected_data(cow_id, pil_images[idx])
                except Exception as exc:
                    results.append(f"Group {group_num}: failed to save one photo ({exc}).")
                    continue
            database.setdefault(cow_id, []).extend(embeddings[idx] for idx in indices)
            results.append(f"Group {group_num}: saved {len(indices)} photo(s) as '{cow_id}'.")

        save_database(database)
        return "\n".join(results)
    except Exception as exc:
        return f"Saving failed: {exc}"


with gr.Blocks(title="AgriPulse Cow Re-ID") as demo:
    gr.Markdown("# AgriPulse — Cow Re-Identification Test UI")
    gr.Markdown(f"Model: ResNet50 + triplet loss embeddings | Device: {device}")
    gr.Markdown(f"All photos are automatically saved to `{COLLECTED_DATA_DIR}` for future training.")

    with gr.Tab("Register Cow"):
        gr.Markdown("Add reference photos for a cow. Upload multiple angles/conditions for better matching.")
        cow_id_input = gr.Textbox(label="Cow ID", placeholder="e.g. AGP-00127 or cow_001")
        register_images = gr.File(label="Photo(s)", file_count="multiple", file_types=["image"])
        register_reject_checkbox = gr.Checkbox(label="Reject photos that don't look like cattle", value=True)
        register_btn = gr.Button("Register", variant="primary")
        register_output = gr.Textbox(label="Result", interactive=False)
        registered_list = gr.Textbox(label="Currently registered cows", interactive=False,
                                      value=registered_cows_list())
        register_btn.click(register_cow_ui, inputs=[cow_id_input, register_images, register_reject_checkbox],
                           outputs=[register_output, registered_list])

    with gr.Tab("Identify Cow"):
        gr.Markdown("Upload a new photo to find the best-matching registered cow.")
        identify_image = gr.Image(label="Photo to identify", type="pil", sources=["upload", "webcam"])
        identify_reject_checkbox = gr.Checkbox(label="Reject photos that don't look like cattle", value=True)
        identify_btn = gr.Button("Identify", variant="primary")
        identify_result = gr.Textbox(label="Result", interactive=False)
        identify_breakdown = gr.Textbox(label="All similarity scores", interactive=False, lines=6)

        with gr.Group(visible=False) as confirm_group:
            gr.Markdown("**Confirm the correct cow** (this improves the model and saves the photo as training data):")
            confirm_dropdown = gr.Dropdown(label="Actual cow ID", choices=[])
            confirm_btn = gr.Button("Confirm")
            confirm_output = gr.Textbox(label="Confirmation result", interactive=False)

        identify_btn.click(identify_cow_ui, inputs=[identify_image, identify_reject_checkbox],
                           outputs=[identify_result, identify_breakdown, confirm_group, confirm_dropdown])
        confirm_btn.click(confirm_identification, inputs=[identify_image, confirm_dropdown],
                          outputs=[confirm_output])

    with gr.Tab("Group Photos"):
        gr.Markdown("Upload a batch of unsorted photos and automatically cluster them by visual "
                    "similarity, using the model's embeddings - no manual eyeballing needed. "
                    "Non-cattle photos are filtered out first, and every grouped photo is "
                    "auto-saved to a review folder right away (no cow ID needed yet).")
        group_images = gr.File(label="Photos", file_count="multiple", file_types=["image"])
        reject_checkbox = gr.Checkbox(label="Reject photos that don't look like cattle", value=True)
        threshold_slider = gr.Slider(minimum=0.1, maximum=0.9, value=MATCH_THRESHOLD, step=0.05,
                                      label="Similarity threshold (higher = stricter grouping)")
        group_btn = gr.Button("Group Photos", variant="primary")
        group_gallery = gr.Gallery(label="Grouped photos", columns=4, height=500)
        group_summary = gr.Textbox(label="Group summary", interactive=False, lines=6)
        rejected_gallery = gr.Gallery(label="Rejected (non-cattle) photos", columns=4, height=300, visible=False)
        group_state = gr.State(None)

        with gr.Group(visible=False) as assign_group:
            gr.Markdown("**Assign a cow ID to each group** (one per line, e.g. `1: AGP-00127`). "
                        "Groups are pre-filled with a placeholder ID - edit any you want to name "
                        "for real before saving; leave a line out to skip that group entirely.")
            assign_textbox = gr.Textbox(label="Group -> Cow ID assignments", lines=6,
                                        placeholder="1: AGP-00127\n2: cow_002")
            save_groups_btn = gr.Button("Save Assigned Groups", variant="primary")
            save_groups_output = gr.Textbox(label="Save result", interactive=False)

        group_btn.click(group_photos_ui, inputs=[group_images, threshold_slider, reject_checkbox],
                        outputs=[group_gallery, group_summary, assign_group, group_state,
                                 rejected_gallery, assign_textbox])
        save_groups_btn.click(save_grouped_photos_ui, inputs=[assign_textbox, group_state],
                              outputs=[save_groups_output])

if __name__ == "__main__":
    use_share = os.environ.get("SHARE", "0") == "1"
    if use_share:
        print("\nStarting with a PUBLIC link (works on any network, any device).")
        print("This link is temporary and stops working when you close this script.\n")
        demo.launch(share=True)
    else:
        print("\nStarting on your LOCAL NETWORK only.")
        print("On your phone (same WiFi), open: http://<this-machine-IP>:7860")
        print("Find your IP with: hostname -I\n")
        demo.launch(server_name="0.0.0.0")
