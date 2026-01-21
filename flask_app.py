import io
import os
from typing import Any, Dict, List, Tuple

import torch
from flask import Flask, jsonify, request
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
)

import fitz  # PyMuPDF

from postprocessing import (
    extract_classes_bboxes,
    postprocess_text,
    transform_bbox_to_original,
)

# Basic configuration - override with environment variables if needed.
MODEL_PATH = os.getenv("MODEL_PATH", "nvidia/NVIDIA-Nemotron-Parse-v1.1")
DEVICE = os.getenv("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32
TASK_PROMPT = os.getenv(
    "TASK_PROMPT", "</s><s><predict_bbox><predict_classes><output_markdown>"
)

# PDF rendering controls (no poppler needed)
PDF_DPI = int(os.getenv("PDF_DPI", "100"))        # 200-300 is typical
PDF_MAX_PAGES = int(os.getenv("PDF_MAX_PAGES", "20"))  # safety limit

app = Flask(__name__)
app.json.ensure_ascii = False
# Load heavy assets once at startup so requests stay fast.
model = (
    AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=DTYPE,
    )
    .to(DEVICE)
    .eval()
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
generation_config = GenerationConfig.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)


def _parse_image_bytes(image_bytes: bytes) -> Image.Image:
    """Parse raw bytes into a PIL Image."""
    if not image_bytes:
        raise ValueError("Empty image payload")
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _parse_pdf_bytes(pdf_bytes: bytes) -> List[Image.Image]:
    """
    Render a PDF (bytes) into a list of PIL Images (one per page) using PyMuPDF.
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
        # alpha=False to avoid RGBA unless you need it
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        img.save(f"./debug_page_{i+1}.png")
        images.append(img)

    return images


def _run_inference(
    image: Image.Image,
    task_prompt: str,
    table_format: str = "latex",
    text_format: str = "markdown",
    blank_text_in_figures: bool = False,
) -> Dict[str, Any]:
    """Run Nemotron-Parse on a single image and return structured results."""
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
    print("generated_text_len:", len(generated_text))
    print("generated_text_head:", generated_text[:200])
    classes, bboxes, texts = extract_classes_bboxes(generated_text)

    bboxes = [
        transform_bbox_to_original(bbox, image.width, image.height)
        for bbox in bboxes
    ]

    texts = [
        postprocess_text(
            text,
            cls=cls,
            table_format=table_format,
            text_format=text_format,
            blank_text_in_figures=blank_text_in_figures,
        )
        for text, cls in zip(texts, classes)
    ]

    items: List[Dict[str, Any]] = []
    for cls, bbox, text in zip(classes, bboxes, texts):
        items.append(
            {
                "class": cls,
                "bbox": {
                    "left": bbox[0],
                    "top": bbox[1],
                    "right": bbox[2],
                    "bottom": bbox[3],
                },
                "text": text,
            }
        )

    return {
        "prompt": task_prompt,
        "items": items,
        "raw_text": generated_text,
    }


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({"status": "ok", "device": DEVICE, "model": MODEL_PATH})


@app.route("/parse", methods=["POST"])
def parse() -> Any:
    """
    POST /parse
    - Form-data: file=<pdf|png|jpg>
    - Optional query params:
        table_format: latex | HTML | markdown
        text_format: markdown | plain
        blank_text_in_figures: bool (default false)

    Response:
    - If image: {type:"image", prompt, items, raw_text}
    - If pdf:   {type:"pdf", prompt, page_count, pages:[{page,width,height,items,raw_text}]}
    """
    if "file" not in request.files:
        return jsonify({"error": "Missing upload field 'file'"}), 400

    table_format = request.args.get("table_format", "latex")
    text_format = request.args.get("text_format", "markdown")
    blank_text_in_figures = (
        request.args.get("blank_text_in_figures", "false").lower() == "true"
    )

    f = request.files["file"]
    data = f.read()

    # Robust PDF detection: extension/mimetype OR magic header
    filename = (f.filename or "").lower()
    mimetype = (f.mimetype or "").lower()
    is_pdf = filename.endswith(".pdf") or mimetype == "application/pdf" or data[:4] == b"%PDF"

    try:
        if is_pdf:
            page_images = _parse_pdf_bytes(data)
            pages_out: List[Dict[str, Any]] = []

            for idx, page_img in enumerate(page_images, start=1):
                print("=== infer page", idx, "size", page_img.size)
                r = _run_inference(
                    page_img,
                    task_prompt=TASK_PROMPT,
                    table_format=table_format,
                    text_format=text_format,
                    blank_text_in_figures=blank_text_in_figures,
                )
                pages_out.append(
                    {
                        "page": idx,
                        "width": page_img.width,
                        "height": page_img.height,
                        "items": r["items"],
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

        # image
        image = _parse_image_bytes(data)
        r = _run_inference(
            image,
            task_prompt=TASK_PROMPT,
            table_format=table_format,
            text_format=text_format,
            blank_text_in_figures=blank_text_in_figures,
        )
        return jsonify({"type": "image", **r})

    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
