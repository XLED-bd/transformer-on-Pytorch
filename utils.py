import os
from torchtext.vocab import build_vocab_from_iterator
from dataset import IMDBDataset, AGDataset, TranslateDataset
from torch.utils.data import DataLoader


def create_imdb_dataloader(train_dir, batch_size=32):
    train_dataset = IMDBDataset(train_dir)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return train_dataloader, train_dataset.vocab


def create_ag_dataloader(train_dir, batch_size=32):
    train_dataset = AGDataset(train_dir)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return train_dataloader, train_dataset.vocab


def create_translate_dataloader(train_dir, tokenizer_1=None, tokenizer_2=None, batch_size=32):
    train_dataset = TranslateDataset(train_dir, tokenizer_1=tokenizer_1, tokenizer_2=tokenizer_2)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return train_dataloader, train_dataset.vocab_1, train_dataset.vocab_2