from vllm import LLM, SamplingParams
from PIL import Image

def main():
    sampling_params = SamplingParams(
        temperature=0,
        top_k=1,
        repetition_penalty=1.1,
        max_tokens=9000,
        skip_special_tokens=False,
    )
    
    llm = LLM(
        model="nvidia/NVIDIA-Nemotron-Parse-v1.1",
        max_num_seqs=64,
        limit_mm_per_prompt={"image": 1},
        dtype="bfloat16",
        trust_remote_code=True,
    )
    
    image = Image.open("./test.png")
    
    prompts = [
        {  # Implicit prompt
            "prompt": "</s><s><predict_bbox><predict_classes><output_markdown>",
            "multi_modal_data": {
                "image": image
            },
        },
        {  # Explicit encoder/decoder prompt
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "image": image
                },
            },
            "decoder_prompt": "</s><s><predict_bbox><predict_classes><output_markdown>",
        },
    ]
    
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Decoder prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    main()
