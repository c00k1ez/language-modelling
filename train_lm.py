import torch
import torch.nn as nn

import pytorch_lightning as pl 

from utils import load_dataloaders, seed_all
from framework.models import ClassicLanguageModel, AttentionLanguageModel
from framework.lm_framework import LMFramework

from pytorch_lightning.callbacks import ModelCheckpoint

import os
import confuse

import nltk
import argparse


if __name__ == "__main__":

    nltk.download('punkt')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./configs/lm_base_config.yaml')
    parser.add_argument('--model', type=str, default='classic_lm')
    args = parser.parse_args()

    config = confuse.Configuration('research')
    config.set_file(args.config_file)

    model = None
    if args.model == 'classic_lm':
        model = ClassicLanguageModel(**config['model'].get())
    elif args.model == 'attention_lm':
        model = AttentionLanguageModel(**config['model'].get())
    else:
        raise ValueError("You have wrong --model parameter")

    seed_all(config['general']['seed'].get())

    loaders = load_dataloaders(**config['dataloaders'].get())
    framework = LMFramework(model, **config['optimizer'].get(), loaders=loaders)

    if not os.path.isdir(config['general']['checkpoint_path'].get()):
        os.makedirs(config['general']['checkpoint_path'].get())
    
    if not os.path.isdir(config['trainer_params']['default_save_path'].get()):
        os.makedirs(config['trainer_params']['default_save_path'].get())

    checkpoint_callback = ModelCheckpoint(
        filepath=config['general']['checkpoint_path'].get(),
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=args.model
    )

    trainer = pl.Trainer(**config['trainer_params'].get(),
                        checkpoint_callback=checkpoint_callback,
                        print_nan_grads=True,
                        profiler=True
                        )
    trainer.fit(framework)