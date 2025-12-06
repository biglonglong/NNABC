import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# Example: reward if answer contains '42'
def compute_reward(output_text):
    return 1.0 if "42" in output_text else 0.0


model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

dataset = load_dataset("gsm8k", "main")


# Dummy GRPO training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
for step, sample in enumerate(dataset["train"]):
    inputs = tokenizer(sample["question"], return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=64)
    text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)

    reward = compute_reward(text_out)
    # Normally: compute GRPO loss here
    loss = -torch.tensor(reward, dtype=torch.float32, requires_grad=True)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step} | Reward: {reward} | Output: {text_out}")

print("Training complete.")
