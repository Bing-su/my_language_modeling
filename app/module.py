import pytorch_lightning as pl
import torch
from loguru import logger
from transformers import PreTrainedTokenizerFast
from torchmetrics import Accuracy

from .util import create_optimizer


class TextMLMModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerFast,
        optimizer: str = "adamw",
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.tokenizer = tokenizer

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.accuracy = Accuracy(num_classes=tokenizer.vocab_size, ignore_index=-100)

    @property
    def forward(self):
        return self.model.forward

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output.loss

        preds = torch.argmax(output.logits, dim=-1)
        labels = batch["labels"]
        self.accuracy(preds, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log("train/accuracy", self.accuracy, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        params = self.model.named_parameters()

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        opt_class = create_optimizer(self.optimizer)
        optimizer = opt_class(
            optimizer_grouped_parameters,
            lr=5e-5,
        )

        if "bnb" in self.optimizer:
            from bitsandbytes.optim import GlobalOptimManager

            manager = GlobalOptimManager.get_instance()
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Embedding) or name.endswith("decoder"):
                    manager.register_module_override(
                        module, "weight", {"optim_bits": 32}
                    )
                    logger.debug(f"bitsandbytes: will optimize {module} in fp32")

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
        )

        scheduler_config = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_config]

    def save(self, save_path: str):
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
