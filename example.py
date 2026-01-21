import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoConfig, AutoImageProcessor, GenerationConfig
from postprocessing import extract_classes_bboxes, transform_bbox_to_original, postprocess_text

# Load model and processor
model_path = "nvidia/NVIDIA-Nemotron-Parse-v1.1"  # Or use a local path
device = "cuda:0"

model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# Load image
image = Image.open("./debug_page_1.png")
task_prompt = "</s><s><predict_bbox><predict_classes><output_markdown>"

# Process image
inputs = processor(images=[image], text=task_prompt, return_tensors="pt").to(device)
prompt_ids = processor.tokenizer.encode(task_prompt, return_tensors="pt", add_special_tokens=False).cuda()


generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
# Generate text
outputs = model.generate(**inputs,  generation_config=generation_config)

# Decode the generated text
generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
classes, bboxes, texts = extract_classes_bboxes(generated_text) 
bboxes = [transform_bbox_to_original(bbox, image.width, image.height) for bbox in bboxes]

# Specify output formats for postprocessing
table_format = 'latex' # latex | HTML | markdown
text_format = 'markdown' # markdown | plain
blank_text_in_figures = False # remove text inside 'Picture' class
texts = [postprocess_text(text, cls = cls, table_format=table_format, text_format=text_format, blank_text_in_figures=blank_text_in_figures) for text, cls in zip(texts, classes)]

for cl, bb, txt in zip(classes, bboxes, texts):
    print(cl, ': ', txt)

# OPTIONAL - Draw bounding boxes
draw = ImageDraw.Draw(image)
for bbox in bboxes:
  draw.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]), outline="red")

# 保存结果图
out_path = "./debug_page_1_out.png"
image.save(out_path)
print(f"Saved: {out_path}")