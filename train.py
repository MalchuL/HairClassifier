import os.path

import hydra
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, QuantizationAwareTraining
from pytorch_lightning.loggers import TensorBoardLogger

from experiment import HairClassifier


@hydra.main(config_path="configs/main_config.yml")
def main(cfg):
    print(cfg.pretty())

    logger = TensorBoardLogger("logs")
    checkpoint_callback = ModelCheckpoint(
        filename='model_{epoch}_{f1_score}_{roc_auc}',
        verbose=True,
        monitor='f1_score',
        mode='max',
        save_last=True,
        save_top_k=cfg.train.save_top_k,
        save_weights_only=cfg.train.save_weight_only
    )

    callbacks = [LearningRateMonitor('epoch'), checkpoint_callback, QuantizationAwareTraining()]

    model = HairClassifier(cfg)

    trainer = Trainer(gpus=cfg.gpu_ids,
                      max_epochs=cfg.train.epoches,
                      logger=logger,
                      limit_val_batches=cfg.train.val_steps_limit,
                      # limit_train_batches=cfg.steps_limit,
                      flush_logs_every_n_steps=cfg.train.log_freq,
                      resume_from_checkpoint=cfg.checkpoint_path,
                      check_val_every_n_epoch=cfg.train.val_freq,
                      precision=cfg.train.precision,
                      gradient_clip_val=cfg.train.gradient_clip_val,
                      callbacks=callbacks)

    trainer.fit(model)

    # Save jit model
    torch.jit.save(torch.jit.script(model), "quant_model.pth")


if __name__ == '__main__':
    main()
