import torch
from torch.utils.data import Dataset
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class TextDataset(Dataset):
    def __init__(self, text, label, 
                tokenizer=None, build_vocab=True, vocab=None,
                max_length=512, min_freq=10):
        '''
        build_vocab & vocab ->
        - True & None:    build vocab from text by tokenizer
        - False & None:   return the tokens only after tokenizer
        - Any & not None: use provided vocab
        '''

        super().__init__()
        self.text = text
        self.label = label
        self.tokenizer = tokenizer or get_tokenizer('basic_english')
        self.max_length = max_length
        self.min_freq = min_freq
        
        if vocab is None and build_vocab:
            self.vocab = self._build_vocab()
        else:
            self.vocab = vocab

    def _build_vocab(self):
        # use when tokenizer without vocab
        def yield_tokens():
            for text in self.text:
                try:
                    if isinstance(text, str):
                        tokens = self.tokenizer(text.lower())
                        yield tokens
                    else:
                        tokens = str(text).lower().split()
                        yield tokens
                except (TypeError, AttributeError, ValueError):
                    tokens = str(text).lower().split()
                    yield tokens

        vocab = build_vocab_from_iterator(
            yield_tokens(),
            min_freq=self.min_freq,
            specials=['<unk>', '<pad>']
        )
        vocab.set_default_index(vocab['<unk>'])
        return vocab
    
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
        return len(self.text)

    def __getitem__(self, idx):
        # transfrom text[idx], label[idx] -> Tensor(token_ids, label)
        text = self.text[idx]
        label = self.label[idx]
        tokens = self.tokenizer(text.lower())

        if self.vocab:
            token_ids = [self.vocab[token] for token in tokens]

            # Padding
            if len(token_ids) < self.max_length:
                token_ids.extend([self.vocab['<pad>']] * (self.max_length - len(token_ids)))
            else:
                token_ids = token_ids[:self.max_length]

            return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)
        
        return tokens, label