import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from swanlab.integration.transformers import SwanLabCallback
from utils import get_dataset
from reward import REWARD_MAP


def train(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        dtype=torch.bfloat16 if args.bf16 else None,
        use_cache=False,
        device_map=None,
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward_funcs = [REWARD_MAP[func.strip()] for func in args.reward_funcs.split(",")]

    training_args = GRPOConfig(
        output_dir=args.checkpoint_dir,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        log_on_each_node=False,
        report_to=args.report_to,
        num_generations=args.num_generations,
        use_vllm=args.use_vllm,
        vllm_gpu_memory_utilization=args.vllm_gpu_ratio,
    )

    swanlab_callback = SwanLabCallback(
        project="huggingface", experiment_name="mini-qwen-grpo-001"
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=get_dataset(split_half=args.split_half),
        callbacks=[swanlab_callback],
    )
    trainer.train()
