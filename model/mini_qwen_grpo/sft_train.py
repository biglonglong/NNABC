import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from swanlab.integration.transformers import SwanLabCallback
from utils import get_dataset


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

    training_args = SFTConfig(
        output_dir=args.checkpoint_dir,
        max_length=args.max_seq_length,
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
        bf16=args.bf16,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        log_on_each_node=False,
        report_to=args.report_to,
    )

    swanlab_callback = SwanLabCallback(
        project="huggingface", experiment_name="mini-qwen-sft-001"
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=get_dataset(sft=True, split_half=args.split_half),
        callbacks=[swanlab_callback],
    )
    trainer.train()
