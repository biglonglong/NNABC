import os
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

    return filtered_data


if __name__ == "__main__":

    # configs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_EPOCHS = 1
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-5
    MODEL_NAME = "distilgpt2"
    OUTPUT_DIR = "./tmp"
    BEST_MODEL_DIR = OUTPUT_DIR + "/" + MODEL_NAME + "_finetuned"
    print(
        f'🚀 微调 "{MODEL_NAME}" on {DEVICE} \n'
        f"🔄 轮数:{NUM_EPOCHS} 💼 批量:{BATCH_SIZE} ⚡ 学习率:{LEARNING_RATE} \n"
        f"📂 输出:{OUTPUT_DIR} \n"
        f"📂 最佳模型输出:{BEST_MODEL_DIR} \n"
        f"-----------------------------------"
    )

    # load data
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir="./data")
    # print(f"Dataset shape: {dataset.shape} \n" f"Dataset cache: {dataset.cache_files}")

    # filter data
    train_data, valid_data = dataset["train"], dataset["validation"]
    filtered_train_data = data_filter(
        "Train", train_data, num_limit=1000, min_length=50
    )
    filtered_valid_data = data_filter(
        "Validation", valid_data, num_limit=10, min_length=50
    )
    # for i, example in enumerate(filtered_train_data[:2]):
    #     print(f"Train {i}: {example['text'][:100]}...")

    # load tokenizer & model
    if os.path.exists(BEST_MODEL_DIR):
        # latter train: load best model from previous train
        # first train: manual download from ModelScope or Huggingface and put files under BEST_MODEL_DIR
        tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(BEST_MODEL_DIR)
        print(f'✅ Load best model from "{BEST_MODEL_DIR}"')
    else:
        # first train: auto download from huggingface model hub to ./huggingface
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="./huggingface")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, cache_dir="./huggingface"
        )
        print(f'✅ Load model "{MODEL_NAME}" from Huggingface')

    # set tokenizer & model
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # print(model)

    # preprocess
    train_dataset = TextDataset(
        texts=[example["text"] for example in filtered_train_data],
        tokenizer=tokenizer,
        max_length=128,
    )
    valid_dataset = TextDataset(
        texts=[example["text"] for example in filtered_valid_data],
        tokenizer=tokenizer,
        max_length=128,
    )

    # arguments
    training_args = TrainingArguments(
        remove_unused_columns=False,  # Preserve all columns from the dataset (don't remove unused ones)
        num_train_epochs=NUM_EPOCHS,  # Total number of training epochs
        per_device_train_batch_size=BATCH_SIZE,  # Training batch size per device (GPU/TPU)
        learning_rate=LEARNING_RATE,  # Initial learning rate for optimizer
        warmup_steps=50,  # Warmup steps: gradually increase learning rate to target over first 50 steps
        logging_steps=50,  # Log training metrics every 50 steps
        output_dir=OUTPUT_DIR,  # Directory to save model checkpoints and outputs
        overwrite_output_dir=True,  # Overwrite the output directory if it already exists
        save_strategy="epoch",  # Save model checkpoint at the end of each epoch
        save_total_limit=2,  # Keep only the most recent checkpoint (delete older ones)
        load_best_model_at_end=True,  # Load the best model checkpoint at the end of training
        per_device_eval_batch_size=BATCH_SIZE,  # Evaluation batch size per device
        eval_strategy="epoch",  # Evaluate model at the end of each epoch
        metric_for_best_model="eval_loss",  # Metric used to determine the best model
        greater_is_better=False,  # Lower evaluation loss indicates better performance
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
    )  # , compute_metrics=compute_metrics

    # train & eval
    trainer.train()
    # trainer.evaluate()

    # save
    trainer.save_model(BEST_MODEL_DIR)
    # tokenizer.save_pretrained(BEST_MODEL_DIR)
