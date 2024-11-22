import os
from torchtext.vocab import build_vocab_from_iterator
from dataset import IMDBDataset
from torch.utils.data import DataLoader


def create_imdb_dataloader(train_dir, batch_size=32):
    train_dataset = IMDBDataset(train_dir)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return train_dataloader, train_dataset.vocab