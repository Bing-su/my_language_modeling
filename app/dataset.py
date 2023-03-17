from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pytorch_lightning as pl
import torch
from datasets import Dataset, load_dataset, load_from_disk
from loguru import logger
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class TokenizeFunction:
    def __init__(self, tokenizer: "PreTrainedTokenizerBase", column: str = "text"):
        self.tokenizer = tokenizer
        self.column = column

    def __call__(self, examples):
        return self.tokenizer(examples[self.column], return_special_tokens_mask=True)


class GroupText:
    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        column: str = "text",
        seq_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.column = column
        self.seq_length = seq_length

    def __call__(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.seq_length:
            total_length = (total_length // self.seq_length) * self.seq_length
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + self.seq_length]
                for i in range(0, total_length, self.seq_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result


class TextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        model_type: str = "mlm",
        data_path: str = "data",
        batch_size: int = 32,
        num_workers: int = 12,
        seq_length: int = 512,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_type = model_type
        self.data_path = data_path
        self.seq_length = seq_length
        self.collator = DataCollatorForLanguageModeling(
            tokenizer, mlm=model_type == "mlm", mlm_probability=0.4
        )

    def prepare_data(self) -> None:
        if not Path(self.data_path).exists():
            logger.info("Downloading dataset...")
            raw_dataset: Dataset = load_dataset(
                "Bingsu/my-korean-training-corpus",
                split="train",
                use_auth_token=True,
            )
            tokenized_dataset = raw_dataset.map(
                TokenizeFunction(self.tokenizer),
                batched=True,
                num_proc=self.num_workers,
                remove_columns=list(raw_dataset.features),
                desc="Running tokenizer on dataset",
            )
            dataset = tokenized_dataset.map(
                GroupText(self.tokenizer, seq_length=self.seq_length),
                batched=True,
                num_proc=self.num_workers,
                desc="Grouping texts",
            )
            logger.info("Saving dataset...")
            dataset.save_to_disk(self.data_path)

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = load_from_disk(self.data_path)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )
