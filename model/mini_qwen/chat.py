import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from vllm import LLM, SamplingParams
from utils import SYSTEM_PROMPT


def chat(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir, dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)

    while True:
        prompt = input("User: what's your question?\n")
        if prompt.lower() in ("exit", "bye", "quit"):
            print("Assistant: Bye👋")
            break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_completion_length,
            temperature=args.temperature,
        )
        output_ids = [
            oids[len(iids) :]
            for iids, oids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        generated_text = response[0]
        print(f"Assistant:\n{generated_text}")


def chat_vllm(args):
    llm = LLM(model=args.checkpoint_dir, gpu_memory_utilization=0.6)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.0,
        max_tokens=args.max_completion_length,
    )

    while True:
        prompt = input("User: what's your question?\n")
        if prompt.lower() in ("exit", "bye", "quit"):
            print("Assistant: Bye👋")
            break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = llm.generate([text], sampling_params)
        generated_text = outputs[0].outputs[0].text
        print(f"Assistant:\n{generated_text}")


def chat_lora(args):
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        args.checkpoint_dir,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_dir, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    while True:
        prompt = input("User: what's your question?\n")
        if prompt.lower() in ("exit", "bye", "quit"):
            print("Assistant: Bye👋")
            break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_completion_length,
            temperature=args.temperature,
        )
        output_ids = [
            oids[len(iids) :]
            for iids, oids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        generated_text = response[0]
        print(f"Assistant:\n{generated_text}")
