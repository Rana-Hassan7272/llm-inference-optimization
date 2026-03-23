import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_with_timing(
    prompt: str,
    tokenizer,
    model,
    device: str,
    max_new_tokens: int = 50,
):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    elapsed_seconds = time.perf_counter() - start

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text, elapsed_seconds


def main():
    parser = argparse.ArgumentParser(description="Basic TinyLlama inference with timing.")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt text.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HF model name.")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Max new tokens.")
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)
    text, elapsed = generate_with_timing(
        prompt=args.prompt,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"Device: {device}")
    print(f"Inference time (s): {elapsed:.4f}")
    print("Generated text:")
    print(text)


if __name__ == "__main__":
    main()
