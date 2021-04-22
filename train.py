import os.path

import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from experiment import HairClassifier


@hydra.main(config_path="configs/main_config.yml")
def main(cfg):
    print(cfg.pretty())

    logger = TensorBoardLogger("logs", default_hp_metric='f1_score')
    checkpoint_callback = ModelCheckpoint(
        filename='model_{epoch}_{f1_score}',
        verbose=True,
        monitor='f1_score',
        mode='max'
    )


    model = HairClassifier(cfg)

    trainer = Trainer(gpus=cfg.gpu_ids,
                      max_epochs=cfg.train.epoches,
                      logger=logger,
                      limit_val_batches=cfg.train.val_steps_limit,
                      # limit_train_batches=cfg.steps_limit,
                      log_every_n_steps=cfg.train.log_freq,
                      flush_logs_every_n_steps=cfg.train.log_freq,
                      resume_from_checkpoint=cfg.checkpoint_path,
                      check_val_every_n_epoch=cfg.train.val_freq,
                      precision=cfg.train.precision,
                      gradient_clip_val=cfg.train.gradient_clip_val,
                      callbacks=[LearningRateMonitor('epoch'), checkpoint_callback])

    trainer.fit(model)


if __name__ == '__main__':
    main()
