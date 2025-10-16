#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# conda run --live-stream --name pytorch_2.5.0 python c:/Users/15222/Desktop/NNABC/model/Transformer/test.py ./tmp/finetuned_distilgpt2

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ./.../test.py <model_path>")
        exit(0)

    model_path = sys.argv[1]  # "./tmp/finetuned_distilgpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧪 Testing model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)

    model.eval()
    test_prompts = ["The future of", "In the year", "Technology will"]
    for prompt in test_prompts:
        print(f"\n🎯 Prompt: '{prompt}'")
        print("-" * 50)

        encoded = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        for i in range(3):
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.size(1) + 30,
                    num_return_sequences=1,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                )

            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"💬 {i+1}: '{generated_text}'")

        # RLHF to fine-tune the model for alignment
        # ...
