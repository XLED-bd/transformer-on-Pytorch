from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from typing import Iterable, List
from torchtext.vocab import build_vocab_from_iterator
import os
import torch

import pandas as pd

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
    


class EngSpaDataset(Dataset):
    def __init__(self, root_dir, vocab_eng=None, vocab_spa=None, max_length=200):
        self.max_length = max_length
        self.tokenizer_eng = get_tokenizer('spacy', language='en')
        self.tokenizer_spa = get_tokenizer('spacy', language='es')

        
        self.texts = []
        self.labels = []

        with open(root_dir, 'r') as f:
            texts = f.readlines()

        for i in range(0, len(texts)):
            self.texts.append(texts[i].split('\t')[0])
            self.labels.append(texts[i].split('\t')[1].replace('\n', ''))
            

        if vocab_eng is None:
            self.vocab_eng = self._create_vocab_eng()
            self.vocab_spa = self._create_vocab_spa()
        else:
            self.vocab_eng = vocab_eng
            self.vocab_spa = vocab_spa


    def _create_vocab_eng(self, max_tokens=30000):
        def yield_tokens() -> Iterable[List[str]]:
            for text in self.texts:
                yield self.tokenizer_eng(text)
        
        vocab_eng = build_vocab_from_iterator(
            yield_tokens(),
            specials=['<pad>', '<unk>', '<start>', '<end>'],
            max_tokens=max_tokens - 4 
        )
        vocab_eng.set_default_index(vocab_eng['<unk>'])
        return vocab_eng

    def _create_vocab_spa(self, max_tokens=30000):
        def yield_tokens() -> Iterable[List[str]]:
            for text in self.texts:
                yield self.tokenizer_spa(text)
        
        vocab_spa = build_vocab_from_iterator(
            yield_tokens(),
            specials=['<pad>', '<unk>', '<start>', '<end>'],
            max_tokens=max_tokens - 4 
        )
        vocab_spa.set_default_index(vocab_spa['<unk>'])
        return vocab_spa
        
    def _process_text_eng(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer_eng(text)
        ids = [self.vocab_eng[token] for token in tokens]
        if len(ids) < self.max_length - 2:
            ids = [self.vocab_eng['<start>']] + ids + [self.vocab_eng['<pad>']] * (self.max_length - len(ids) - 2) + [self.vocab_eng['<end>']]
        else:
            ids = [self.vocab_eng['<start>']] + ids[:self.max_length - 2] + [self.vocab_eng['<end>']]
        return torch.tensor(ids, dtype=torch.long)
    
    def _process_text_spa(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer_spa(text)
        ids = [self.vocab_spa[token] for token in tokens]
        if len(ids) < self.max_length:
            ids = [self.vocab_eng['<start>']] + ids + [self.vocab_spa['<pad>']] * (self.max_length - len(ids)) + [self.vocab_eng['<end>']]
        else:
            ids = [self.vocab_eng['<start>']] + ids[:self.max_length] + [self.vocab_eng['<end>']]
        return torch.tensor(ids, dtype=torch.long)
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return self._process_text_eng(text), self._process_text_spa(label)
    

dataset = EngSpaDataset("spa-eng/spa.txt")

print(dataset.texts[1])
print(dataset.labels[1])

print(dataset[1])
