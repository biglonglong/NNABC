import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd

# tensor
from torch.utils.data import TensorDataset

# image
from torchvision.datasets import ImageFolder

# legacy_text !!Attention: may be under torchtext.legacy.data
import torchtext
from torchtext.data import Iterator, BucketIterator
from torchtext.data import Field
from torchtext.data import TabularDataset

# self_text
from data import TextDataset

# transformers
from datasets import load_dataset, load_from_disk


def csv_preprocess(df):
    # Create final result DataFrame to store ALL processed data
    result_df = pd.DataFrame()

    # Filter rows with labels A, B, C, D, E only
    df = df[df["label"].isin(["A", "B", "C", "D", "E"])]

    # Convert column A to one-hot encoding
    if "A" in df.columns:
        a_processed = pd.get_dummies(df["A"], prefix="A")
        result_df = pd.concat([result_df, a_processed], axis=1)

    # Fill missing values in column B with 0 and merge to result
    if "B" in df.columns:
        b_processed = df[["B"]].fillna(0)
        result_df = pd.concat([result_df, b_processed], axis=1)

    # Modify column C with prefix and merge to result
    if "C" in df.columns:
        c_processed = df[["C"]].copy().add_prefix("Modified_")
        result_df = pd.concat([result_df, c_processed], axis=1)

    # Add D column (target) to result_df
    if "D" in df.columns:
        d_processed = df[["D"]].copy()
        result_df = pd.concat([result_df, d_processed], axis=1)

    # Merge other columns (except A, B, C, D) to result
    other_cols = [col for col in df.columns if col not in ["A", "B", "C", "D"]]
    if other_cols:
        result_df = pd.concat([result_df, df[other_cols]], axis=1)

    # Remove rows with missing values
    result_df = result_df.dropna()

    # Split result_df into features (x_df) and target (y_df)
    if "D" in result_df.columns:
        y_df = result_df[["D"]].copy()  # Target column
        x_df = result_df.drop("D", axis=1)  # All other columns as features
    else:
        raise ValueError("Column 'D' not found in processed data")

    # Save processed data
    result_df.to_csv("data_processed.csv", index=False)

    return x_df, y_df


def image_preprocess():
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((224, 224))
    normalize = transforms.Normalize(
        mean=[0.162, 0.151, 0.138], std=[0.058, 0.052, 0.048]
    )
    horizon_flip = transforms.RandomHorizontalFlip()
    vertical_flip = transforms.RandomVerticalFlip()
    rotate = transforms.RandomRotation()

    transform = transforms.Compose(
        [resize, normalize, horizon_flip, vertical_flip, rotate]
    )
    return transform


if __name__ == "__main__":
    # csv preprocessing
    raw = pd.read_csv(".\data\data.csv")
    x_df, y_df = csv_preprocess(raw)

    dataset = TensorDataset(
        torch.tensor(x_df.values, dtype=torch.float32),
        torch.tensor(y_df.values.flatten(), dtype=torch.float32),
    )
    print(
        f"Dataset: {len(dataset)} samples, Features: {x_df.shape}, Target: {y_df.shape}"
    )

    train_dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)

    # image preprocessing
    raw = r".\dataset_split\train"
    transform_train = image_preprocess()

    dataset = ImageFolder(
        root=raw,
        transform=transform_train,
        target_transform=lambda t: torch.tensor(t),
    )
    print(
        f"Image Dataset: {len(dataset)} samples, Classes: {dataset.classes}, Class to Index: {dataset.class_to_idx}"
    )

    train_dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)

    # legacy_text preprocessing
    raw = r".\data\raw.csv"
    tokenizer = lambda x: x.split()
    TEXT = Field(sequential=True, tokenize=tokenizer, lower=True, batch_first=True)
    LABEL = Field(sequential=False, use_vocab=False, batch_first=True)
    train_fields = [
        ("id", None),
        ("comment_text", TEXT),
        ("toxic", LABEL),
        ("severe_toxic", None),
        ("threat", None),
        ("obscene", None),
        ("insult", None),
        ("identity_hate", None),
    ]

    dataset = TabularDataset(
        path=raw, format="csv", skip_header=True, fields=train_fields
    )
    print(
        f"{dataset[0].__dict__.keys()} \n"
        f"First Train Data: \n"
        f"{dataset[0].__dict__.values()} \n"
        f"the front five of First Train Data's Comment Text: \n"
        f"{dataset[0].comment_text[:5]} "
    )
    TEXT.build_vocab(dataset, max_size=10000, min_freq=10)
    print(
        f"Size:\t {len(TEXT.vocab)} \n"
        f"Index:\t {TEXT.vocab.stoi} \n"
        f"Vocab:\t {TEXT.vocab.itos} \n"
        f"MostCom:\t {TEXT.vocab.freqs.most_common(10)} \n"
        f"LeaCom:\t {TEXT.vocab.freqs.most_common()[:-11:-1]}"
    )

    train_bucketiterator = BucketIterator(dataset=dataset, batch_size=128, shuffle=True)
    batch = next(iter(train_bucketiterator))
    b_x = batch.comment_text
    b_y = batch.toxic

    # self_text preprocessing
    raw = r".\data\raw.csv"
    text, label = pd.read_csv(raw, usecols=["comment_text", "toxic"]).values.T
    tokenizer = lambda x: x.split()

    dataset = TextDataset(
        text,
        label,
        tokenizer=tokenizer,
        build_vocab=True,
        vocab=None,
        max_length=64,
        min_freq=2,
    )
    print(
        f"Vocabulary size: {len(dataset.vocab)} \n"
        f"First 15 tokens: {list(dataset.vocab.get_itos())[:15]} \n"
        f"Most frequent tokens: {dict(list(dataset.vocab.get_stoi().items())[:10])} \n"
        f"Dataset length: {len(dataset)} \n"
        f"First sample: {dataset[0]}"
    )

    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),  # stack text
            torch.stack([item[1] for item in batch]),  # stack label
        ),
    )

    # transformers preprocessing
    dataset = load_dataset(
        "madao33/new-title-chinese", cache_dir="./data"
    )  # , split="train"
    print(f"Dataset shape: {dataset.shape} \n" f"Dataset cache: {dataset.cache_files}")
    train_data, valid_data = dataset["train"], dataset["validation"]

    filtered_train_data = train_data.filter(lambda example: "中国" in example["text"])
    for i, example in enumerate(filtered_train_data[:2]):
        print(f"Example {i}: {example}")
