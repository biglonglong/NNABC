import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from data import TextDataset


def data_filter(prefix, data, num_limit, min_length=50):
    filtered_data = []
    for example in data:
        text = example["text"].strip()
        if len(text) > min_length:
            filtered_data.append(example)
            if len(filtered_data) >= num_limit:
                break

    for i, example in enumerate(filtered_data[:2]):
        print(f"{prefix} {i}: {example['text'][:100]}...")
    return filtered_data


if __name__ == "__main__":

    # configs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_LENGTH = 128
    MODEL_PATH = "distilgpt2"
    NUM_EPOCHS = 1
    BATCH_SIZE = 4
    LEARNING_RATE = 5e-5
    OUTPUT_DIR = "./tmp"
    print(
        f'🚀 微调 "{MODEL_PATH}" on {DEVICE} \n'
        f"📏 长度:{MAX_LENGTH} 🔄 轮数:{NUM_EPOCHS} 💼 批量:{BATCH_SIZE} ⚡ 学习率:{LEARNING_RATE} \n"
        f"📂 输出:{OUTPUT_DIR} \n"
        f"-----------------------------------"
    )

    # load data & filter
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir="./data")
    print(f"Dataset shape: {dataset.shape} \n" f"Dataset cache: {dataset.cache_files}")
    train_data, valid_data = dataset["train"], dataset["validation"]
    filtered_train_data = data_filter(
        "Train", train_data, num_limit=1000, min_length=50
    )
    filtered_valid_data = data_filter(
        "Validation", valid_data, num_limit=10, min_length=50
    )

    # load model, tokenizer（setting token: padding bos eos unk sep cls mask）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir="./huggingface")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, cache_dir="./huggingface")
    print(model)

    # preprocess
    train_dataset = TextDataset(
        texts=[example["text"] for example in filtered_train_data],
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    valid_dataset = TextDataset(
        texts=[example["text"] for example in filtered_valid_data],
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    # arguments
    training_args = TrainingArguments(
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=50,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        remove_unused_columns=False,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
    )  # , compute_metrics=compute_metrics

    # finetune
    trainer.train()
    print(trainer.evaluate())

    # save
    tokenizer.save_pretrained(OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR + "/finetuned_distilgpt2")
