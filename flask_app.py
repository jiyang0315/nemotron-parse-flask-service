#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import re
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List

import torch
from flask import Flask, jsonify, request
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer, GenerationConfig

import fitz  # PyMuPDF

from postprocessing import (
    extract_classes_bboxes,
    postprocess_text,
    transform_bbox_to_original,
)

# =========================================================
# Config (env overridable)
# =========================================================
MODEL_PATH = os.getenv("MODEL_PATH", "nvidia/NVIDIA-Nemotron-Parse-v1.1")
DEVICE = os.getenv("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32

TASK_PROMPT = os.getenv("TASK_PROMPT", "</s><s><predict_bbox><predict_classes><output_markdown>")

# PDF render
PDF_DPI = int(os.getenv("PDF_DPI", "100"))                 # 100~200 usually ok
PDF_MAX_PAGES = int(os.getenv("PDF_MAX_PAGES", "50"))      # safety limit

# Debug
DEBUG_SAVE_PAGES = os.getenv("DEBUG_SAVE_PAGES", "0") == "1"  # set 1 to save debug_page_*.png
PORT = int(os.getenv("PORT", "8000"))

# Save parsed json to file (optional)
SAVE_JSON_DIR = os.getenv("SAVE_JSON_DIR", "")  # e.g. ./outputs ; empty means do not save

# Output format
TABLE_FORMAT = os.getenv("TABLE_FORMAT", "latex")          # latex | markdown | HTML
TEXT_FORMAT = os.getenv("TEXT_FORMAT", "markdown")         # markdown | plain
BLANK_TEXT_IN_FIGURES = os.getenv("BLANK_TEXT_IN_FIGURES", "0") == "1"  # remove text in Picture class

# If you only want raw_text without items/bboxes, set this to 1
TEXT_ONLY = os.getenv("TEXT_ONLY", "0") == "1"


# =========================================================
# Flask
# =========================================================
app = Flask(__name__)
app.json.ensure_ascii = False  # IMPORTANT: allow Chinese output in JSON response


# =========================================================
# Load model once
# =========================================================
model = (
    AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=DTYPE,
    )
    .to(DEVICE)
    .eval()
)

# keep tokenizer/processor for decoding and preprocess
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
generation_config = GenerationConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)


# =========================================================
# Helpers
# =========================================================
def _parse_image_bytes(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise ValueError("Empty image payload")
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _parse_pdf_bytes(pdf_bytes: bytes) -> List[Image.Image]:
    if not pdf_bytes:
        raise ValueError("Empty PDF payload")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Failed to open PDF: {exc}") from exc

    if doc.page_count == 0:
        raise ValueError("PDF has no pages")

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


def strip_latex_table(text: str) -> str:
    """
    Remove LaTeX tabular wrapper and normalize to readable plain text.

    Input example:
      \\begin{tabular}{ccc}
      a & b & c\\\\
      ...
      \\end{tabular}

    Output:
      a | b | c
      ...
    """
    if not text:
        return text

    # Remove begin/end
    text = re.sub(r"\\begin\{tabular\}\{.*?\}", "", text)
    text = re.sub(r"\\end\{tabular\}", "", text)

    # Normalize row/col delimiters
    text = text.replace("\\\\", "\n")
    text = text.replace("&", " | ")

    # Remove excessive blank lines
    text = re.sub(r"\n[ \t]*\n", "\n", text)

    return text.strip()


def clean_item_text_for_extra_field(cls: str, text: str) -> str:
    """
    Keep original 'text' untouched.
    Create an extra cleaned field ONLY for Table class; otherwise return text as-is.
    """
    if cls == "Table":
        return strip_latex_table(text)
    return text


def _run_inference(image: Image.Image, task_prompt: str) -> Dict[str, Any]:
    with torch.inference_mode():
        inputs = processor(images=[image], text=task_prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(**inputs, generation_config=generation_config)

    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return {"prompt": task_prompt, "raw_text": generated_text}


def _build_items(image: Image.Image, raw_text: str) -> List[Dict[str, Any]]:
    classes, bboxes, texts = extract_classes_bboxes(raw_text)

    # Convert model's normalized bbox to original image pixel coordinates
    bboxes = [transform_bbox_to_original(b, image.width, image.height) for b in bboxes]

    # Postprocess model text (table/text format, etc.)
    texts = [
        postprocess_text(
            t,
            cls=c,
            table_format=TABLE_FORMAT,
            text_format=TEXT_FORMAT,
            blank_text_in_figures=BLANK_TEXT_IN_FIGURES,
        )
        for t, c in zip(texts, classes)
    ]

    items: List[Dict[str, Any]] = []
    for c, b, t in zip(classes, bboxes, texts):
        # IMPORTANT: keep original `text` as-is, add `text_clean` as extra
        items.append(
            {
                "class": c,
                "bbox": {"left": b[0], "top": b[1], "right": b[2], "bottom": b[3]},
                "text": t,  # 原始不动
                # "text_clean": clean_item_text_for_extra_field(c, t),  # 新增清洗字段
            }
        )
    return items


def _maybe_save_json(payload: Dict[str, Any], base_name: str = "parse") -> str:
    """
    Save response JSON to SAVE_JSON_DIR if configured.
    Return saved path or "".
    """
    if not SAVE_JSON_DIR:
        return ""

    os.makedirs(SAVE_JSON_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rid = uuid.uuid4().hex[:8]
    out_path = os.path.join(SAVE_JSON_DIR, f"{base_name}_{ts}_{rid}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_path


# =========================================================
# Routes
# =========================================================
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
            "table_format": TABLE_FORMAT,
            "text_format": TEXT_FORMAT,
            "blank_text_in_figures": BLANK_TEXT_IN_FIGURES,
            "text_only": TEXT_ONLY,
            "save_json_dir": SAVE_JSON_DIR,
        }
    )


@app.route("/parse", methods=["POST"])
def parse() -> Any:
    """
    POST /parse
      - form-data: file=@xxx.pdf / xxx.png / xxx.jpg
    Response:
      - pdf:  {type:"pdf", prompt, page_count, pages:[{page,width,height, raw_text, items?}]}
      - image:{type:"image", prompt, width,height, raw_text, items?}

    If SAVE_JSON_DIR is set, response will include "saved_json_path".
    """
    if "file" not in request.files:
        return jsonify({"error": "Missing upload field 'file'"}), 400

    f = request.files["file"]
    data = f.read()

    filename = (f.filename or "").lower()
    mimetype = (f.mimetype or "").lower()
    is_pdf = filename.endswith(".pdf") or mimetype == "application/pdf" or data[:4] == b"%PDF"

    try:
        if is_pdf:
            page_images = _parse_pdf_bytes(data)
            pages_out: List[Dict[str, Any]] = []

            for idx, page_img in enumerate(page_images, start=1):
                r = _run_inference(page_img, TASK_PROMPT)
                page_obj: Dict[str, Any] = {
                    "page": idx,
                    "width": page_img.width,
                    "height": page_img.height,
                    "raw_text": r["raw_text"],
                }
                if not TEXT_ONLY:
                    page_obj["items"] = _build_items(page_img, r["raw_text"])
                pages_out.append(page_obj)

            payload: Dict[str, Any] = {
                "type": "pdf",
                "prompt": TASK_PROMPT,
                "page_count": len(pages_out),
                "pages": pages_out,
            }

            saved_path = _maybe_save_json(payload, base_name="pdf_parse")
            if saved_path:
                payload["saved_json_path"] = saved_path

            return jsonify(payload)

        # Image
        image = _parse_image_bytes(data)
        r = _run_inference(image, TASK_PROMPT)

        payload = {
            "type": "image",
            "prompt": TASK_PROMPT,
            "width": image.width,
            "height": image.height,
            "raw_text": r["raw_text"],
        }
        if not TEXT_ONLY:
            payload["items"] = _build_items(image, r["raw_text"])

        saved_path = _maybe_save_json(payload, base_name="img_parse")
        if saved_path:
            payload["saved_json_path"] = saved_path

        return jsonify(payload)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
