import torch
import torch.nn as nn

import pytorch_lightning as pl 

import os
import confuse


class LMFramework(pl.LightningModule):

    def __init__(self, lm_model, learning_rate, loaders):
        super(LMFramework, self).__init__()
        self.model = lm_model

        self.learning_rate = learning_rate
        self.loaders = loaders

        self.criterion = nn.NLLLoss(reduction='none')

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_nb):
        y = batch['lm_label']
        loss_mask = batch['loss_mask']

        y_hat = self.forward(batch)
        batch_size, seq_len, vocab = y_hat.shape[0], y_hat.shape[1], y_hat.shape[2]
        y_hat = y_hat.view(-1, vocab)
        y = y.view(batch_size * seq_len)
        loss_mask = loss_mask.view(batch_size * seq_len)
        loss_val = self.criterion(y_hat, y)
        loss_val = (loss_val * loss_mask)
        loss_val = loss_val[loss_val > 0].mean()
        return {
            'loss': loss_val, 
            'log': {'train_loss': loss_val},
            }
    
    def validation_step(self, batch, batch_nb):
        y = batch['lm_label']
        loss_mask = batch['loss_mask']
        
        y_hat = self.forward(batch)
        
        batch_size, seq_len, vocab = y_hat.shape[0], y_hat.shape[1], y_hat.shape[2]
        y_hat = y_hat.view(-1, vocab)
        y = y.view(batch_size * seq_len)
        loss_mask = loss_mask.view(batch_size * seq_len)
        loss_val = self.criterion(y_hat, y)
        loss_val = loss_val * loss_mask
        loss_val = loss_val[loss_val > 0]
        return {
            'loss_val': loss_val
            }

    def validation_epoch_end(self, outputs):
        loss_val = torch.cat([x['loss_val'] for x in outputs]).mean()
        perp = torch.exp(loss_val)
        return {
            'val_loss': loss_val,
            'perplexity': perp,
            'log': {'val_loss': loss_val, 'perplexity': perp}
            }

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return opt

    def train_dataloader(self):
        return self.loaders['train']

    def val_dataloader(self):
        return self.loaders['valid']
    
    def on_save_checkpoint(self, checkpoint):
        if isinstance(self.logger, pl.logging.CometLogger):
            self.logger.experiment.log_model(checkpoint)

