from argparse import Namespace


import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn

from data import MyImageDataset

from registry import registries
from transforms.transform import get_transform
import torch
import models  # Adds effnets to registry
from utils.confusion_matrix import plot_confusion_matrix, render_figure_to_tensor

from utils.util import make_weights_for_balanced_classes
import sklearn.metrics as metrics

class HairClassifier(pl.LightningModule):

    def __init__(self,
                 hparams: Namespace) -> None:
        super(HairClassifier, self).__init__()

        self.hparams = hparams

        self.model = self.create_model()
        self.loss = self.create_loss()

    def create_loss(self):
        return registries.CRITERION.get_from_params(**self.hparams.train.loss)

    def get_scheduler(self, optimizer, scheduler_params):
        return registries.SCHEDULERS.get_from_params(**{'optimizer': optimizer, **scheduler_params})

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer_params = self.hparams.train.optimizer_params
        scheduler_params = self.hparams.train.scheduler_params
        optimizer = registries.OPTIMIZERS.get_from_params(**{'params': params, **optimizer_params})

        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': self.get_scheduler(optimizer, scheduler_params), 'interval': 'step'}}

    def create_model(self):
        return registries.MODELS.get_from_params(**self.hparams.model)

    def forward(self, input):
        return self.model(input)

    # Fix bug in lr scheduling
    def on_epoch_end(self):
        if (self.current_epoch + 1) % self.hparams.train.val_freq != 0:
            with torch.no_grad():
                self.trainer.optimizer_connector.update_learning_rates(interval='epoch')

    def training_step(self, batch, batch_idx):
        image, labels = batch
        result = self(image)

        labels = labels.unsqueeze(1).type_as(image)
        loss = self.loss(result, labels)

        log = {'loss': loss, 'mean_class': labels.mean()}
        out = {}

        self.log_dict(log, prog_bar=True)

        return {'loss': loss, 'out': out}

    def validation_step(self, batch, batch_nb):
        image, labels = batch
        result = torch.sigmoid(self(image))

        pred = (result.squeeze(1) > self.hparams.threshold).int()

        return {'pred': pred.detach().cpu().numpy(), 'target': labels.cpu().numpy()}

    def merge_dict(self, outputs):
        if not outputs:
            return {}
        keys = outputs[0].keys()
        result = {}
        for key in keys:
            merged_values = []
            for value in outputs:
                merged_values.append(value[key])
            result[key] = np.concatenate(merged_values, axis=0)

        return result

    def validation_epoch_end(self, outputs):
        outputs = self.merge_dict(outputs)

        pred, target = outputs['pred'], outputs['target']
        f1_score = metrics.f1_score(target, pred)

        self.log('f1_score', f1_score)
        self.log('hp_metric', f1_score)

        confusion_matrix = metrics.confusion_matrix(target, pred)
        self.logger.experiment.add_image('CM', render_figure_to_tensor(plot_confusion_matrix(confusion_matrix,
                                                                     self.train_dataset.classes,
                                                                     normalize=True,
                                                                     show=False, )), self.global_step)

        return {'f1_score': f1_score}

    def get_transforms(self):
        return get_transform(self.hparams.datasets.train, True), get_transform(self.hparams.datasets.val, False)

    def prepare_data(self):
        train_transforms, val_transforms = self.get_transforms()

        train_params = self.hparams.datasets.train
        self.train_dataset = MyImageDataset(train_params.dataroot, train_transforms, self.hparams.class_to_id)

        val_params = self.hparams.datasets.val
        self.val_dataset = MyImageDataset(val_params.dataroot, val_transforms, self.hparams.class_to_id)

    def val_dataloader(self):
        val_params = self.hparams.datasets.val
        return DataLoader(self.val_dataset,
                          batch_size=val_params.batch_size,
                          shuffle=False,
                          drop_last=False,
                          num_workers=val_params.n_workers)

    def train_dataloader(self):
        train_params = self.hparams.datasets.train
        sampler = WeightedRandomSampler(
            make_weights_for_balanced_classes(self.train_dataset.targets, len(self.train_dataset.classes)),
            len(self.train_dataset))
        return DataLoader(self.train_dataset,
                          batch_size=train_params.batch_size,
                          shuffle=sampler is None,
                          drop_last=True,
                          num_workers=train_params.n_workers,
                          sampler=sampler)
