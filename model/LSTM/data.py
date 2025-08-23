import torch
from sklearn.preprocessing import OneHotEncoder
import torch.utils.data as Data
import numpy as np

TIME_STEP = 50


class TextDataset(Data.Dataset):
    def __init__(self, text, vocab):
        super().__init__()
        self.text = text
        self.vocab = vocab
        self.time_step = TIME_STEP

        self.int2word = {i: word for i, word in enumerate(vocab)}
        self.word2int = {word: i for i, word in self.int2word.items()}

        self.text_indices = [self.word2int[word] for word in self.text]

        self.encoder = OneHotEncoder(sparse_output=False).fit(self.vocab.reshape(-1, 1))

    def __len__(self):
        return len(self.text_indices) - self.time_step

    def __getitem__(self, idx):
        input_indices = self.text_indices[idx : idx + self.time_step]
        target_indices = self.text_indices[idx + self.time_step]

        input_onehot = self.encoder.transform(
            np.array(
                [self.int2word[word_indice] for word_indice in input_indices]
            ).reshape(-1, 1)
        )

        return torch.tensor(input_onehot, dtype=torch.float32), torch.tensor(
            target_indices, dtype=torch.long
        )
