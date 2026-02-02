import pytorch_lightning as pl
import torch
import torch.optim as optim

class Module(pl.LightningModule):
    def __init__(self, model_cls, cfg: dict, train_cfg: dict, model_cls_args: dict):
        super().__init__()
        self.__dict__.update(cfg)
        self.__dict__.update(train_cfg)

        self.model = model_cls(input_shape=self.input_shape, **model_cls_args)

        if hasattr(self.model, '_visualise_step'):
            self._visualise_step = lambda batch: self.model._visualise_step(batch[0])
            self._visualisation_labels = self.model._visualisation_labels

    def forward(self, batch, **kwargs):
        return self.model(batch, **kwargs)

    def training_step(self, batch, batch_idx):
        batch = batch[0]
        results = self.forward(batch)
        loss = self.model.loss_function(
            batch, results, M_N=batch.size(0) / self.len_train_ds
        )
        self.log_dict({f"train/{k}": v for k, v in loss.items()}, on_epoch=True)
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        batch = batch[0]
        results = self.forward(batch)
        loss = self.model.loss_function(
            batch, results, M_N=batch.size(0) / self.len_val_ds
        )
        self.log_dict({f"valid/{k}": v for k, v in loss.items()}, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer