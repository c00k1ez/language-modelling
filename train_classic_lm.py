import torch
import torch.nn as nn

import pytorch_lightning as pl 

from utils import load_dataloaders, seed_all
from framework.models import ClassicLanguageModel

from pytorch_lightning.callbacks import ModelCheckpoint

import os
import confuse

import nltk

class ClassicLMFramework(pl.LightningModule):

    def __init__(self, lm_model, learning_rate, loaders):
        super(ClassicLMFramework, self).__init__()
        self.model = lm_model

        self.learning_rate = learning_rate
        self.loaders = loaders

        self.criterion = nn.NLLLoss(reduction='none')

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_nb):
        x, y = batch['text'], batch['lm_label']
        loss_mask = batch['loss_mask']
        y_hat = self.model(x)
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
        x, y = batch['text'], batch['lm_label']
        loss_mask = batch['loss_mask']
        y_hat = self.model(x)
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

    def validation_end(self, outputs):
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

    @pl.data_loader
    def train_dataloader(self):
        return self.loaders['train']

    @pl.data_loader
    def val_dataloader(self):
        return self.loaders['valid']


if __name__ == "__main__":

    nltk.download('punkt')

    config = confuse.Configuration('research')
    config.set_file('./configs/lm_base_config.yaml')

    seed_all(config['general']['seed'].get())

    model = ClassicLanguageModel(**config['model'].get())
    loaders = load_dataloaders(**config['dataloaders'].get())
    framework = ClassicLMFramework(model, **config['optimizer'].get(), loaders=loaders)

    checkpoint_callback = ModelCheckpoint(
        filepath=config['general']['checkpoint_path'].get(),
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix='classic_lm_'
    )

    trainer = pl.Trainer(**config['trainer_params'].get(),
                        checkpoint_callback=checkpoint_callback,
                        print_nan_grads=True
                        )
    trainer.fit(framework)