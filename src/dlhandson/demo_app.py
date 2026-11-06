"""Ship it: a two-tab Gradio web app for both models.

Why a demo app in a deep-learning lecture? Because a model nobody can use has
zero business value. Gradio turns a Python function into a web UI in a few
lines, and `share=True` creates a temporary public HTTPS link (via Gradio's
relay servers) that opens on any phone — with camera and photo-gallery upload.

SECURITY / PRIVACY NOTES
* The share link is PUBLIC while the notebook runs (anyone with the URL can use
  it) and expires automatically (about a week). Close it with demo.close().
* Uploaded images go to YOUR Colab machine only; nothing is stored by this code.
* Don't upload photos of people, documents, or anything sensitive.
"""
from __future__ import annotations

import torch

LEAF_INTRO = ("Upload a photo **from your phone's gallery**, take one with the camera, or paste one. "
              "The model knows only 3 classes, so watch what it says about things that are *not* bean leaves.")
TEXT_INTRO = "Type a customer message. The model routes it to one of 77 banking intents."


def _decision(conf: float, label: str, threshold: float, human: str) -> str:
    if conf >= threshold:
        return f"### ✅ Auto-decision: **{label}**\nConfidence {conf:.0%} ≥ threshold {threshold:.0%}"
    return (f"### 🧑 Routed to a {human}\nTop guess *{label}* is only {conf:.0%} confident "
            f"(< {threshold:.0%}). In production this goes to a person, not an automatic action.")


def make_vision_predictor(model, transform, class_names, device, threshold: float = 0.8):
    @torch.no_grad()
    def predict(image):
        if image is None:
            return {}, "Upload or take a photo first."
        model.eval()
        x = transform(image.convert("RGB")).unsqueeze(0).to(device)
        probs = torch.softmax(model(x).float(), dim=1)[0].cpu()
        k = int(probs.argmax())
        return ({c: float(p) for c, p in zip(class_names, probs)},
                _decision(float(probs[k]), class_names[k], threshold, "human agronomist"))
    return predict


def make_text_predictor(model, tokenizer, id2label, device, threshold: float = 0.7, max_length: int = 64):
    @torch.no_grad()
    def predict(text: str):
        if not text or not text.strip():
            return {}, "Type a message first."
        model.eval()
        enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        probs = torch.softmax(model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.float(),
                              dim=1)[0].cpu()
        top = torch.topk(probs, k=min(5, len(probs)))
        labels = {id2label[int(i)]: float(p) for p, i in zip(top.values, top.indices)}
        k = int(top.indices[0])
        return labels, _decision(float(top.values[0]), id2label[k], threshold, "human support agent")
    return predict


def build_demo(vision_predict=None, text_predict=None, title: str = "Deep Learning Hands-On · Live Demo"):
    import gradio as gr

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        if vision_predict is not None:
            with gr.Tab("🌱 Leaf Doctor"):
                gr.Markdown(LEAF_INTRO)
                with gr.Row():
                    img = gr.Image(type="pil", sources=["upload", "webcam", "clipboard"], label="Photo")
                    with gr.Column():
                        lab = gr.Label(num_top_classes=3, label="Class probabilities")
                        dec = gr.Markdown()
                btn = gr.Button("Diagnose", variant="primary")
                btn.click(vision_predict, inputs=img, outputs=[lab, dec])
                img.change(vision_predict, inputs=img, outputs=[lab, dec])
        if text_predict is not None:
            with gr.Tab("💬 Support Router"):
                gr.Markdown(TEXT_INTRO)
                txt = gr.Textbox(lines=2, label="Customer message",
                                 placeholder="e.g. I was charged twice for the same coffee")
                lab2 = gr.Label(num_top_classes=5, label="Top intents")
                dec2 = gr.Markdown()
                gr.Examples(["I still haven't received my new card", "Why was I charged twice?",
                             "Can I top up with Apple Pay?", "what's the weather like in Portland?"], inputs=txt)
                gr.Button("Route", variant="primary").click(text_predict, inputs=txt, outputs=[lab2, dec2])
                txt.submit(text_predict, inputs=txt, outputs=[lab2, dec2])
    return demo


def launch(demo, share: bool = True) -> str | None:
    """Launch without blocking the notebook; return the public URL (or None)."""
    try:
        result = demo.launch(share=share, prevent_thread_lock=True, quiet=True)
    except Exception as exc:
        print(f"⚠️  Could not start the public link ({exc}).\n"
              "   Fallback: use the in-notebook upload cell instead.")
        return None
    url = getattr(demo, "share_url", None)
    if not url and isinstance(result, tuple) and len(result) >= 3:
        url = result[2] or result[1]
    return url


def qr_code(url: str, box_size: int = 8):
    """A QR code (PIL image) so phones can open the demo link instantly."""
    import qrcode

    qr = qrcode.QRCode(box_size=box_size, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").get_image()
