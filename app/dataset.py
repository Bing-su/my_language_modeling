from typing import Optional

import pytorch_lightning as pl
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast


class TextDataset(Dataset):
    def __init__(self, dataset: HFDataset, tokenizer: PreTrainedTokenizerFast):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["text"]
        inputs = self.tokenizer(text, truncation=True, return_special_tokens_mask=True)
        return inputs


class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        batch_size: int = 32,
        num_workers: int = 12,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.4)

    def prepare_data(self) -> None:
        load_dataset(
            "Bingsu/my-korean-training-corpus",
            split="train",
            use_auth_token=True,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = load_dataset(
            "Bingsu/my-korean-training-corpus", split="train", use_auth_token=True
        )
        self.train_dataset = TextDataset(dataset, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            shuffle=True,
            num_workers=self.num_workers,
        )
