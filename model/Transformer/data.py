import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        super().__init__()
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,  # 截断超长文本
            max_length=self.max_length,  # 统一序列长度
            padding="max_length",  # <Pad>填充到最大长度
            return_tensors="pt",  # 返回pt精度张量
        )

        input_ids = encoding["input_ids"].squeeze()
        return {"input_ids": input_ids, "labels": input_ids.clone()}
