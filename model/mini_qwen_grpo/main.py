import argparse
from grpo_train import train as grpo_train
from sft_train import train as sft_train
from chat import chat, chat_vllm
from test import test, test_vllm


def main():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        default="sft_train",
        choices=["grpo_train", "sft_train", "chat", "chat_vllm", "test", "test_vllm"],
        help="The task to be performed.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        default="./checkpoint",
        help="Path to save the model or load checkpoint to infer.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=False,
        default=None,
        help="The model name or path of the pre-trained model from huggingface.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=False,
        default="./huggingface",
        help="Path to a directory in which a downloaded pre-trained model configuration and weights should be cached.",
    )
    parser.add_argument(
        "--split_half",
        type=str,
        required=False,
        default=None,
        choices=["first_half", "second_half"],
        help="Whether to use only the first half or the second half or the whole of the dataset for training.",
    )
    parser.add_argument(
        "--reward_funcs",
        type=str,
        required=False,
        default="strict_format_reward_func,correctness_reward_func",
        help="A comma-separated list of reward functions to be used during RL training.",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        required=False,
        default=2560,
        help="Maximum length of the tokenized sequence during sft training. Sequences longer are truncated from the right.",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        required=False,
        default=256,
        help="Maximum length of the tokenized prompt during RL training. Prompt longer are truncated from the left.",
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        required=False,
        default=2048,
        help="Maximum length of the generated completion.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        required=False,
        default=1,
        help="The batch size per device for training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=False,
        default=5e-6,
        help="The initial learning rate for [`AdamW`] optimizer.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        required=False,
        default="cosine",
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        required=False,
        default=0.1,
        help="Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        required=False,
        default=0.9,
        help="The beta1 hyperparameter for the [`AdamW`] optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        required=False,
        default=0.99,
        help="The beta2 hyperparameter for the [`AdamW`] optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        required=False,
        default=0.1,
        help="The weight decay to apply to all layers except all bias and LayerNorm weights in the optimizer.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        required=False,
        default=4,
        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        required=False,
        default=1.0,
        help="Maximum gradient norm for gradient clipping.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to use bf16 (mixed) precision training instead of 32-bit training. Requires Ampere or higher NVIDIA architecture or using CPU or Ascend NPU.",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        required=False,
        default="steps",
        choices=["no", "steps", "epoch", "best"],
        help="The checkpoint save strategy to adopt during training.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        required=False,
        default=100,
        help='Number of updates steps before two checkpoint saves if `save_strategy="steps"`.',
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        required=False,
        default=10,
        help="Number of update steps between two logs.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        required=False,
        default="none",
        help="training report",
    )

    parser.add_argument(
        "--num_generations",
        type=int,
        required=False,
        default=4,
        help="Number of generations per prompt to sample. The global batch size (num_processes * per_device_batch_size) must be divisible by this value.",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Whether to use vLLM for generating completions.",
    )
    parser.add_argument(
        "--vllm_gpu_ratio",
        type=float,
        required=False,
        default=0.9,
        help="Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the"
        "device dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus"
        "improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors"
        "during initialization.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=0.3,
        help="Temperature for sampling during chat or test. The higher the temperature, the more random the completions.",
    )

    args = parser.parse_args()
    if args.task == "grpo_train":
        grpo_train(args)
    elif args.task == "sft_train":
        sft_train(args)
    elif args.task == "chat":
        chat(args)
    elif args.task == "chat_vllm":
        chat_vllm(args)
    elif args.task == "test":
        test(args)
    elif args.task == "test_vllm":
        test_vllm(args)


if __name__ == "__main__":
    # https://hugging-face.cn/docs/trl/index
    # https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
    # python3 ./model/mini_qwen_grpo/main.py --task=sft_train --model_name_or_path=Qwen/Qwen2.5-0.5B-Instruct --checkpoint_dir=./checkpoint/sft --bf16 --save_strategy=epoch
    # python3 ./model/mini_qwen_grpo/main.py --task=grpo_train --model_name_or_path=Qwen/Qwen2.5-0.5B-Instruct --checkpoint_dir=./checkpoint/grpo --bf16 --save_strategy=epoch
    # python3 ./model/mini_qwen_grpo/main.py --task=chat --checkpoint_dir=./checkpoint/???
    # python3 ./model/mini_qwen_grpo/main.py --task=test --checkpoint_dir=./checkpoint/???
    main()

    # norm  43.9%
    # sft   34.5%
    # grpo
