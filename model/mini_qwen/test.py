import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from vllm import LLM, SamplingParams
from utils import SYSTEM_PROMPT
from utils import get_dataset
from reward import extract_number_from_boxed_string


def test(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = get_dataset(split="test")
    correct_num = 0
    for i in tqdm(range(len(data))):
        q = data[i]["prompt"][1]["content"]
        a = data[i]["answer"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
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

        extracted_number = extract_number_from_boxed_string(generated_text)
        if extracted_number == a.replace(" ", "").replace(",", ""):
            correct_num += 1
        # else:
        #     print(f"Error! Extracted: {extracted_number} Ans: {a}\n")

    print("Accuracy:", correct_num / len(data) if data else "N/A")


def test_vllm(args):
    llm = LLM(model=args.checkpoint_dir, gpu_memory_utilization=0.6)
    tokenizer = llm.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.0,
        max_tokens=args.max_completion_length,
    )

    data = get_dataset(split="test")
    correct_num = 0
    for i in tqdm(range(len(data))):
        q = data[i]["prompt"][1]["content"]
        a = data[i]["answer"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = llm.generate([text], sampling_params)
        generated_text = outputs[0].outputs[0].text

        extracted_number = extract_number_from_boxed_string(generated_text)
        if extracted_number == a.replace(" ", "").replace(",", ""):
            correct_num += 1
        # else:
        #     print(f"Error! Extracted: {extracted_number} Ans: {a}\n")

    print("Accuracy:", correct_num / len(data) if data else "N/A")


def test_lora(args):
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
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_dir, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = get_dataset(split="test")
    correct_num = 0
    for i in tqdm(range(len(data))):
        q = data[i]["prompt"][1]["content"]
        a = data[i]["answer"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
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

        extracted_number = extract_number_from_boxed_string(generated_text)
        if extracted_number == a.replace(" ", "").replace(",", ""):
            correct_num += 1
        # else:
        #     print(f"Error! Extracted: {extracted_number} Ans: {a}\n")

    print("Accuracy:", correct_num / len(data) if data else "N/A")
