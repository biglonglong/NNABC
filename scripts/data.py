import torch
from torch.utils.data import Dataset
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class TextDataset(Dataset):
    def __init__(
        self,
        texts,
        labels,
        tokenizer=None,
        build_vocab=True,
        vocab=None,
        max_length=512,
        min_freq=10,
    ):
        """
        build_vocab & vocab ->
        - True & None:    build vocab from texts by tokenizer
        - False & None:   return the tokens only after tokenizer
        - Any & not None: use provided vocab
        """

        super().__init__()
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer or get_tokenizer("basic_english")
        self.max_length = max_length
        self.min_freq = min_freq

        if vocab is None and build_vocab:
            self.vocab = self._build_vocab()
        else:
            self.vocab = vocab

    def _build_vocab(self):
        # use when tokenize2tokens without 
        
        # Method 1: torchtext.vocab.build_vocab_from_iterator
        def yield_tokens():
            for text in self.texts:
                tokens = self.tokenizer(text.lower())
                yield tokens
        vocab = build_vocab_from_iterator(
            yield_tokens(), min_freq=self.min_freq, specials=["<unk>", "<pad>"]
        )
        vocab.set_default_index(vocab["<unk>"])
        return vocab

        # # Method 2: collections.Counter
        # # Uncomment and comment above to use this method
        # from collections import Counter
        # from torchtext.vocab import Vocab
        # counter = Counter()
        # counter.update(self.tokenizer(text.lower()))
        # filtered_counter = Counter({token: count for token, count in counter.items() if count >= self.min_freq})
        # vocab = Vocab(filtered_counter, specials=['<unk>', '<pad>'])
        # vocab.set_default_index(vocab['<unk>'])
        # return vocab

    def __len__(self):
        # data length
        return len(self.texts)

    def __getitem__(self, idx):
        # transfrom texts[idx], labels[idx] -> Tensor(token_ids, label)
        text = self.texts[idx]
        label = self.labels[idx]

        # tokenize2ids
        if hasattr(self.tokenizer, "encode"):
            token_ids = self.tokenizer.encode(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).squeeze()
            return token_ids, torch.tensor(label, dtype=torch.long)

        # tokenize2tokens
        else:
            tokens = self.tokenizer(text.lower())

            if self.vocab:
                token_ids = [self.vocab[token] for token in tokens]

                # Padding
                if len(token_ids) < self.max_length:
                    token_ids.extend(
                        [self.vocab["<pad>"]] * (self.max_length - len(token_ids))
                    )
                else:
                    token_ids = token_ids[: self.max_length]

                return torch.tensor(token_ids, dtype=torch.long), torch.tensor(
                    label, dtype=torch.long
                )

            return tokens, torch.tensor(label, dtype=torch.long)
