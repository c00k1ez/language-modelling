import torch
import torch.nn as nn

import pytorch_lightning as pl 

from utils import load_dataloaders, seed_all
from framework.models import AttentionLanguageModel
from train_classic_lm import ClassicLMFramework

from pytorch_lightning.callbacks import ModelCheckpoint

import os
import confuse

import nltk


if __name__ == '__main__':

    nltk.download('punkt')

    config = confuse.Configuration('research')
    config.set_file('./configs/lm_base_config.yaml')

    seed_all(config['general']['seed'].get())

    model = AttentionLanguageModel(**config['model'].get())
    loaders = load_dataloaders(**config['dataloaders'].get())
    framework = ClassicLMFramework(model, **config['optimizer'].get(), loaders=loaders)

    checkpoint_callback = ModelCheckpoint(
        filepath=config['general']['checkpoint_path'].get(),
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix='attention_lm_'
    )

    trainer = pl.Trainer(**config['trainer_params'].get(),
                        checkpoint_callback=checkpoint_callback,
                        print_nan_grads=True
                        )
    trainer.fit(framework)