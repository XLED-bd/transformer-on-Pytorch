from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from typing import Iterable, List
from torchtext.vocab import build_vocab_from_iterator
import os
import torch

import pandas as pd
import string

class IMDBDataset(Dataset):
    def __init__(self, root_dir, vocab=None, max_length=600):
        self.max_length = max_length
        self.tokenizer = get_tokenizer('basic_english')
        
        self.texts = []
        self.labels = []
        
        pos_path = os.path.join(root_dir, 'pos')
        for file in os.listdir(pos_path):
            with open(os.path.join(pos_path, file), 'r', encoding='utf-8') as f:
                self.texts.append(f.read())
                self.labels.append(1)
        
        neg_path = os.path.join(root_dir, 'neg')
        for file in os.listdir(neg_path):
            with open(os.path.join(neg_path, file), 'r', encoding='utf-8') as f:
                self.texts.append(f.read())
                self.labels.append(0)
        
        if vocab is None:
            self.vocab = self._create_vocab()
        else:
            self.vocab = vocab

    def _create_vocab(self, max_tokens=20000):
        def yield_tokens() -> Iterable[List[str]]:
            for text in self.texts:
                yield self.tokenizer(text)
        
        vocab = build_vocab_from_iterator(
            yield_tokens(),
            specials=['<pad>', '<unk>'],
            max_tokens=max_tokens - 2 
        )
        vocab.set_default_index(vocab['<unk>'])
        return vocab

    def _process_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text)
        ids = [self.vocab[token] for token in tokens]
        if len(ids) < self.max_length:
            ids = ids + [self.vocab['<pad>']] * (self.max_length - len(ids))
        else:
            ids = ids[:self.max_length]
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return self._process_text(text), torch.tensor(label, dtype=torch.long)
    

class AGDataset(Dataset):
    def __init__(self, root_dir, vocab=None, max_length=600):
        self.max_length = max_length
        self.tokenizer = get_tokenizer('basic_english')
        
        self.texts = []
        self.labels = []

        df_label = pd.read_csv(root_dir, usecols=["Class Index"])["Class Index"].tolist()
        df_text = pd.read_csv(root_dir, usecols=["Title", "Description"])

        for i in range(0, len(df_text)):
            self.texts.append(df_text["Title"][i].replace('"', " ").replace("\\", " ") + " " + df_text["Description"][i].replace('"', " ").replace("\\", " "))

            self.labels.append([1, 0, 0, 0] if df_label[i] == 1 else
                                [0, 1, 0, 0] if df_label[i] == 2 else
                                  [0, 0, 1, 0] if df_label[i] == 3 else
                                  [0, 0, 0, 1])


        if vocab is None:
            self.vocab = self._create_vocab()
        else:
            self.vocab = vocab


    def _create_vocab(self, max_tokens=30000):
        def yield_tokens() -> Iterable[List[str]]:
            for text in self.texts:
                yield self.tokenizer(text)
        
        vocab = build_vocab_from_iterator(
            yield_tokens(),
            specials=['<pad>', '<unk>'],
            max_tokens=max_tokens - 2 
        )
        vocab.set_default_index(vocab['<unk>'])
        return vocab

        
    def _process_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text)
        ids = [self.vocab[token] for token in tokens]
        if len(ids) < self.max_length:
            ids = ids + [self.vocab['<pad>']] * (self.max_length - len(ids))
        else:
            ids = ids[:self.max_length]
        return torch.tensor(ids, dtype=torch.long)
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return self._process_text(text), torch.tensor(label, dtype=torch.long)
    


class TranslateDataset(Dataset):
    def __init__(self, root_dir, tokenizer_1, tokenizer_2,vocab_1=None, vocab_2=None, max_length=40, max_tokens=15000):
        self.max_length = max_length

        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        

        
        self.texts_1 = []
        self.texts_2 = []

        with open(root_dir, 'r') as f:
            texts = f.readlines()

        for i in range(len(texts)):
            self.texts_1.append(''.join(char for char in texts[i].replace('\n', '').lower().split('\t')[0] if char not in string.punctuation + "¿"))
            self.texts_2.append(''.join(char for char in texts[i].replace('\n', '').lower().split('\t')[1] if char not in string.punctuation + "¿"))
            

        if vocab_1 is None:
            self.vocab_1 = self._create_vocab_1(max_tokens=max_tokens)
            self.vocab_2 = self._create_vocab_2(max_tokens=max_tokens)
        else:
            self.vocab_1 = vocab_1
            self.vocab_2 = vocab_2


    def _create_vocab_1(self, max_tokens):
        def yield_tokens() -> Iterable[List[str]]:
            for text in self.texts_1:
                yield self.tokenizer_1(text)
        
        vocab_1 = build_vocab_from_iterator(
            yield_tokens(),
            specials=['<pad>', '<unk>', '<start>', '<end>'],
            max_tokens=max_tokens - 4 
        )
        vocab_1.set_default_index(vocab_1['<unk>'])
        return vocab_1

    def _create_vocab_2(self, max_tokens=15000):
        def yield_tokens() -> Iterable[List[str]]:
            for text in self.texts_2:
                yield self.tokenizer_2(text)
        
        vocab_2 = build_vocab_from_iterator(
            yield_tokens(),
            specials=['<pad>', '<unk>', '<start>', '<end>'],
            max_tokens=max_tokens - 4 
        )
        vocab_2.set_default_index(vocab_2['<unk>'])
        return vocab_2
        
    def _process_text_1(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer_1(text)
        ids = [self.vocab_1[token] for token in tokens]
        if len(ids) < self.max_length:
            ids = ids + [self.vocab_1['<pad>']] * (self.max_length - len(ids))
        else:
            ids = ids[:self.max_length]
        return torch.tensor(ids, dtype=torch.long)
    
    def _process_text_2(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer_2(text)
        ids = [self.vocab_2[token] for token in tokens]
        if len(ids) < self.max_length:
            ids = [self.vocab_2['<start>']] + ids + [self.vocab_2['<end>']] + [self.vocab_2['<pad>']] * (self.max_length - len(ids) - 2)
        else:
            ids = [self.vocab_2['<start>']] + ids[:self.max_length - 2] + [self.vocab_2['<end>']]
        return torch.tensor(ids, dtype=torch.long)
    
    def _process_text_2_target(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer_2(text)
        ids = [self.vocab_2[token] for token in tokens]
        if len(ids) < self.max_length:
            ids = ids + [self.vocab_2['<end>']] + [self.vocab_2['<pad>']] * (self.max_length - len(ids) - 1)
        else:
            ids = ids[:self.max_length - 1] + [self.vocab_2['<end>']]
        return torch.tensor(ids, dtype=torch.long)
    
    def __len__(self):
        return len(self.texts_1)

    def __getitem__(self, idx):
        text_1 = self.texts_1[idx]
        text_2 = self.texts_2[idx]

        return self._process_text_1(text_1), self._process_text_2(text_2), self._process_text_2_target(text_2)
