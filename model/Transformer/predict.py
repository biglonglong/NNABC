import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ./.../test.py <model_path>")
        exit(0)

    model_path = sys.argv[1]  # "./tmp/distilgpt2_finetuned"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧪 Testing model on {device} from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()

    test_prompts = ["The future of AI is"]
    n = 3
    more_length = 50
    for prompt in test_prompts:
        print(f"\n🎯 Prompt: '{prompt}'")
        print("-" * 50)

        encoded = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        for i in range(n):
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.size(1) + more_length,
                    num_return_sequences=1,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                )

            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"💬 {i+1}: '{generated_text}'")
