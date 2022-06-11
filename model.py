import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import monai
import torch
from torchmetrics import AUROC


class MRSClassfication(pl.LightningModule):

    def __init__(self,model_name,model_hparams,learning_rate=0.001,multi_class=False):
        super().__init__()

        self.model = self.create_model(model_name,model_hparams)
        self.multi_class = multi_class
        self.loss = nn.CrossEntropyLoss() if self.multi_class else nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate

    def create_model(self, model_name, model_hparams):

        if model_name == 'DenseNet121':
            return monai.networks.nets.DenseNet121(**model_hparams)
        elif model_name == 'ViT':
            return monai.networks.nets.ViT(**model_hparams)
        elif model_name == 'SEResNet101':
            return monai.networks.nets.SEResNet101(**model_hparams)
        else :
            assert False, f'Unknown model name "{model_name}".'

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_pred = self(x)
        loss = self.loss (y_pred, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        return y_pred, y

    def validation_epoch_end(self, validation_step_outputs):
        y_preds = []
        ys = []

        for y_pred, y in validation_step_outputs:
            y_preds.extend(y_pred)
            ys.extend(y)

        y_preds = torch.stack(y_preds)
        ys = torch.stack(ys).type(torch.int)

        auc  = AUROC()
        auc_score = auc(y_preds.squeeze(),ys.squeeze())

        print(f"auc_score : {auc_score:.4f}")
        self.log("val_auc", auc_score,prog_bar=True, logger=True) 

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=2e-5)

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_pred = self.model(x)

        return y_pred, y
    
    def test_epoch_end(self, test_step_outputs):
        y_preds = []
        ys = []

        for y_pred, y in test_step_outputs:
            y_preds.extend(y_pred)
            ys.extend(y)

        y_preds = torch.stack(y_preds)
        ys = torch.stack(ys).type(torch.int)

        auc  = AUROC()
        auc_score = auc(y_preds.squeeze(),ys.squeeze())

        print(f"auc_score : {auc_score:.4f}")