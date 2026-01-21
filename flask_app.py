import io
import os
from typing import Any, Dict, List

import torch
from flask import Flask, jsonify, request
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
)

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

app = Flask(__name__)

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


def _parse_image(file_storage) -> Image.Image:
    """Parse an uploaded file into a PIL Image."""
    image_bytes = file_storage.read()
    if not image_bytes:
        raise ValueError("Empty image payload")
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image


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

    generated_text = processor.batch_decode(
        outputs, skip_special_tokens=True
    )[0]
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
    - Form-data: image=<file>
    - Optional query params:
        table_format: latex | HTML | markdown
        text_format: markdown | plain
        blank_text_in_figures: bool (default false)
    """
    if "image" not in request.files:
        return jsonify({"error": "Missing image file field 'image'"}), 400

    table_format = request.args.get("table_format", "latex")
    text_format = request.args.get("text_format", "markdown")
    blank_text_in_figures = (
        request.args.get("blank_text_in_figures", "false").lower() == "true"
    )

    try:
        image = _parse_image(request.files["image"])
        result = _run_inference(
            image,
            task_prompt=TASK_PROMPT,
            table_format=table_format,
            text_format=text_format,
            blank_text_in_figures=blank_text_in_figures,
        )
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

