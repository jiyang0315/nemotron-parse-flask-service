import io
import os
from typing import Any, Dict, List

import torch
from flask import Flask, jsonify, request
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer, GenerationConfig

import fitz  # PyMuPDF

# =========================
# Basic configuration (env overridable)
# =========================
MODEL_PATH = os.getenv("MODEL_PATH", "nvidia/NVIDIA-Nemotron-Parse-v1.1")
DEVICE = os.getenv("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32

TASK_PROMPT = os.getenv(
    "TASK_PROMPT", "</s><s><predict_bbox><predict_classes><output_markdown>"
)

# PDF rendering controls (no poppler needed)
PDF_DPI = int(os.getenv("PDF_DPI", "100"))          # 100~200 usually ok; too high may break some pages
PDF_MAX_PAGES = int(os.getenv("PDF_MAX_PAGES", "20"))  # safety limit
DEBUG_SAVE_PAGES = os.getenv("DEBUG_SAVE_PAGES", "1") == "0"  # save debug_page_*.png

PORT = int(os.getenv("PORT", "8000"))

# =========================
# Flask app
# =========================
app = Flask(__name__)
app.json.ensure_ascii = False  # IMPORTANT: allow Chinese in JSON response


# =========================
# Load model/processor once at startup
# =========================
model = (
    AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=DTYPE,
    )
    .to(DEVICE)
    .eval()
)

# Not strictly required for text-only output, but kept (some models need tokenizer/processor)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
generation_config = GenerationConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)


# =========================
# Helpers
# =========================
def _parse_image_bytes(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise ValueError("Empty image payload")
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _parse_pdf_bytes(pdf_bytes: bytes) -> List[Image.Image]:
    """
    Render PDF bytes to a list of PIL Images (one per page) using PyMuPDF.
    No poppler dependency.
    """
    if not pdf_bytes:
        raise ValueError("Empty PDF payload")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Failed to open PDF: {exc}") from exc

    if doc.page_count == 0:
        raise ValueError("PDF has no pages")

    # PDF default is 72 DPI. Convert desired DPI -> zoom factor.
    zoom = PDF_DPI / 72.0
    mat = fitz.Matrix(zoom, zoom)

    images: List[Image.Image] = []
    page_count = min(doc.page_count, PDF_MAX_PAGES)

    for i in range(page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

        if DEBUG_SAVE_PAGES:
            img.save(f"./debug_page_{i+1}.png")

        images.append(img)

    return images


def _run_inference_text_only(image: Image.Image, task_prompt: str) -> Dict[str, Any]:
    """
    Run Nemotron-Parse on one image and return ONLY raw_text.
    """
    with torch.inference_mode():
        inputs = processor(
            images=[image],
            text=task_prompt,
            return_tensors="pt",
        ).to(DEVICE)

        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
        )

    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    # Debug prints
    print("generated_text_len:", len(generated_text))
    print("generated_text_head:", generated_text[:200])

    return {
        "prompt": task_prompt,
        "raw_text": generated_text,
    }


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify(
        {
            "status": "ok",
            "device": DEVICE,
            "dtype": str(DTYPE),
            "model": MODEL_PATH,
            "pdf_dpi": PDF_DPI,
            "pdf_max_pages": PDF_MAX_PAGES,
        }
    )

@app.route("/parse", methods=["POST"])
def parse() -> Any:
    """
    POST /parse
    - Form-data: file=<pdf|png|jpg>
    Response:
    - If image: {type:"image", prompt, raw_text}
    - If pdf:   {type:"pdf", prompt, page_count, pages:[{page,width,height,raw_text}]}
    """
    if "file" not in request.files:
        return jsonify({"error": "Missing upload field 'file'"}), 400

    f = request.files["file"]
    data = f.read()

    # Robust PDF detection: extension/mimetype OR magic header
    filename = (f.filename or "").lower()
    mimetype = (f.mimetype or "").lower()
    is_pdf = (
        filename.endswith(".pdf")
        or mimetype == "application/pdf"
        or data[:4] == b"%PDF"
    )

    try:
        if is_pdf:
            page_images = _parse_pdf_bytes(data)
            pages_out: List[Dict[str, Any]] = []

            for idx, page_img in enumerate(page_images, start=1):
                print("=== infer page", idx, "size", page_img.size)
                r = _run_inference_text_only(page_img, task_prompt=TASK_PROMPT)
                pages_out.append(
                    {
                        "page": idx,
                        "raw_text": r["raw_text"],
                    }
                )

            return jsonify(
                {
                    "type": "pdf",
                    "prompt": TASK_PROMPT,
                    "page_count": len(pages_out),
                    "pages": pages_out,
                }
            )

        # Image
        image = _parse_image_bytes(data)
        r = _run_inference_text_only(image, task_prompt=TASK_PROMPT)
        return jsonify({"type": "image", **r})

    except Exception as exc:
        # Ensure Chinese errors can return too
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    # Tip: if you want to suppress Flask's default ascii escaping further,
    # you already set app.json.ensure_ascii = False above.
    app.run(host="0.0.0.0", port=PORT)
