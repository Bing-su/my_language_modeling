import inspect
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
from loguru import logger
from torchmetrics import Accuracy

from .util import create_optimizer

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class TextMLMModule(pl.LightningModule):
    def __init__(
        self,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizerBase",
        model_type: str = "mlm",
        optimizer: str = "adamw",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.acc = Accuracy(
            task="multiclass", num_classes=model.config.vocab_size, ignore_index=-100
        )

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output.loss

        self.calc_accuracy(output.logits, batch["labels"])

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", self.acc, on_step=True, on_epoch=True)
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
        signiture = inspect.signature(opt_class)
        opt_kwargs = {}
        true_kwargs = [
            "capturable",
            "weight_decouple",
            "decouple_decay",
            "decoupled_weight_decay",
        ]
        for key in true_kwargs:
            if key in signiture.parameters:
                opt_kwargs[key] = True

        optimizer = opt_class(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            **opt_kwargs,
        )

        if "bnb" in self.optimizer:
            from bitsandbytes.optim import GlobalOptimManager

            manager = GlobalOptimManager.get_instance()
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Embedding) or name.endswith(
                    ("decoder", "lm_head")
                ):
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
        self.tokenizer.save_pretrained(save_path)

    def calc_accuracy(self, logits, labels):
        preds = torch.argmax(logits, dim=-1)
        # preds.shape == labels.shape == (batch_size, seq_len)
        if self.model_type == "mlm":
            self.acc(preds, labels)
        else:  # "clm"
            preds = preds[:, :-1]
            labels = labels[:, 1:]
            self.acc(preds, labels)
