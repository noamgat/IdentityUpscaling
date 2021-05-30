import argparse
import functools
import os
import sys
from collections import OrderedDict

import PIL.Image
import pytorch_lightning as pl
import torchvision
import yaml
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.utils import save_image

from bicubic import BicubicDownsampleTargetSize
from celeba_aligned import build_aligned_celeba, CelebAPairsDataset, CelebAAdverserialDataset
from face_comparer import FaceComparer
import torch
import torch.nn.functional as F
import numpy as np
import pl_transfer_learning_helpers

NUM_WORKERS = 0

class FaceComparerModule(LightningModule):
    def __init__(self, *args, face_comparer_params=None, **kwargs):
        args = args[1:]
        face_comparer_params = face_comparer_params or {}
        #face_comparer_params = kwargs.pop('face_comparer_params', {})
        face_comparer_params.setdefault('feature_extractor_model', 'facenet')
        face_comparer_params.setdefault('load_pretrained', True)
        self.include_adverserial_faces = kwargs.pop('include_adverserial_faces', 0)
        self.milestones = kwargs.pop('milestones', [500000, 1000000])
        self.train_bn = kwargs.pop('train_bn', 0)
        super().__init__(*args, **kwargs)
        self.face_comparer = FaceComparer(**face_comparer_params)
        # TODO ENABLE FOR LEARNING
        # pl_transfer_learning_helpers.freeze(self.feature_extractor, train_bn=self.train_bn)
        #self.face_comparer.cuda()
        #self.device = self.face_comparer.tail[0].weight.device # TODO : Easiest way?
        self.lr_scheduler_gamma = 1e-1
        self.lr = 1e-2

    @property
    def feature_extractor(self):
        return self.face_comparer.face_features_extractor[-1]

    def forward(self, x1, x2):
        return self.face_comparer.forward(x1, x2)

    def get_dataloader(self, split='train', same_ratio=0.5, batch_size=16):
        shuffle = split == 'train'
        large = build_aligned_celeba('CelebA_Raw', 'CelebA_large', split=split)
        pairs_dataset = CelebAPairsDataset(large, same_ratio=same_ratio, num_samples=10000)
        if not self.include_adverserial_faces:
            return DataLoader(pairs_dataset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=shuffle)

        transform_func = functools.partial(BicubicDownsampleTargetSize.downsample_single, size=256, mode='area')
        transform = torchvision.transforms.Lambda(transform_func)

        withidentity = build_aligned_celeba('CelebA_Raw', 'CelebA_withidentity', new_image_suffix='_0', split=split, extra_transforms=[transform])
        large_matching_withidentity = build_aligned_celeba('CelebA_Raw', 'CelebA_large',
                                                           custom_indices=withidentity.filtered_indices, split=split)
        adverserial_dataset = CelebAAdverserialDataset(withidentity, large_matching_withidentity)
        pairs_dataset = CelebAPairsDataset(large, same_ratio=same_ratio, num_samples=len(adverserial_dataset))
        concat_dataset = ConcatDataset([pairs_dataset, adverserial_dataset])
        return DataLoader(concat_dataset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=shuffle)

    @pl.data_loader
    def train_dataloader(self):
        return self.get_dataloader()

    @pl.data_loader
    def val_dataloader(self):
        return self.get_dataloader(split='valid')

    # @pl.data_loader
    # def test_dataloader(self):
    #     return self.get_dataloader(split='test')

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_nb):
        x1, x2, y = batch
        y = y.unsqueeze(1)
        prediction = self(x1, x2)
        num_correct = int(((prediction.sign() / 2) + 0.5 == y).to(float).sum().item())
        loss = F.binary_cross_entropy_with_logits(prediction.to(torch.double), y.to(torch.double))
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs, 'num_correct': num_correct}

    def test_or_validation_step(self, batch, batch_idx, prefix='val'):
        x1, x2, y = batch
        y = y.unsqueeze(1)

        save_image(torch.cat((x1, x2)), 'resources/validationinputs.jpg')

        # implement your own
        prediction = self(x1, x2)
        loss = F.binary_cross_entropy_with_logits(prediction.to(torch.double), y.to(torch.double))
        num_correct = int(((prediction.sign() / 2) + 0.5 == y).to(float).sum().item()) / (len(y) * 1.0)

        # all optional...
        # return whatever you need for the collation function test_end
        output = OrderedDict({
            f'{prefix}_loss': loss,
            f'{prefix}_acc': torch.tensor(num_correct),  # everything must be a tensor
        })

        # return an optional dict
        return output

    def validation_step(self, batch, batch_idx):
        return self.test_or_validation_step(batch, batch_idx, prefix='val')

    #def test_step(self, batch, batch_idx):
    #    return self.test_or_validation_step(batch, batch_idx, prefix='test')

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc_mean = torch.stack([x['val_acc'] for x in outputs]).mean()
        results = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        print(f"Epoch {self.current_epoch} Validation: Acc = {val_acc_mean.item()}, Loss = {val_loss_mean.item()}, B={self.face_comparer.tail[-1].bias.data.item()}")
        return {'progress_bar': results, 'log': results, 'val_loss': results['val_loss']}

    def train(self, mode=True):
        super().train(mode=mode)

        if mode:
            epoch = self.current_epoch
            if epoch < self.milestones[0]:
                # feature extractor is frozen (except for BatchNorm layers)
                pl_transfer_learning_helpers.freeze(module=self.feature_extractor, train_bn=self.train_bn)

            elif self.milestones[0] <= epoch < self.milestones[1]:
                # Unfreeze last two layers of the feature extractor
                pl_transfer_learning_helpers.freeze(module=self.feature_extractor, n=-2, train_bn=self.train_bn)

    def on_epoch_start(self):
        """Use `on_epoch_start` to unfreeze layers progressively."""
        optimizer = self.trainer.optimizers[0]
        if self.current_epoch == self.milestones[0]:
            print("Hit first milestone, unfreezing last two layers")
            pl_transfer_learning_helpers.unfreeze(module=self.feature_extractor,
                                                  optimizer=optimizer,
                                                  train_bn=self.train_bn,
                                                  start_n=-2)

        elif self.current_epoch == self.milestones[1]:
            print("Hit first milestone, unfreezing all layers")
            pl_transfer_learning_helpers.unfreeze(
                                          module=self.feature_extractor,
                                          optimizer=optimizer,
                                          train_bn=self.train_bn,
                                          end_n=-2)

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      self.parameters()),
                               lr=self.lr)

        scheduler = MultiStepLR(optimizer,
                                milestones=self.milestones,
                                gamma=self.lr_scheduler_gamma)

        return [optimizer], [scheduler]


def load_face_comparer_module(config_file_path, opts=None, for_eval=False):
    with open(config_file_path) as fd:
        config = yaml.safe_load(fd)
    trainer_params = config['trainer_params']
    model_params = config['model_params']
    checkpoint_params = config['checkpoint_params']
    if 'filepath' in checkpoint_params:
        os.makedirs(checkpoint_params['filepath'], exist_ok=True)
    checkpoint_callback = ModelCheckpoint(**checkpoint_params)
    if checkpoint_callback.save_last:
        last_ckpt = os.path.join(checkpoint_callback.dirpath, checkpoint_callback.prefix + 'last.ckpt')
    force_restart = opts and opts.force_restart
    last_ckpt = last_ckpt if os.path.exists(last_ckpt) and not force_restart else None
    if for_eval:
        if last_ckpt:
            net = FaceComparerModule.load_from_checkpoint(last_ckpt, **model_params)
        else:
            print("Warning: for_eval=True with no checkpoint.")
            net = FaceComparerModule(**model_params)

        net.cuda()
        net.eval()
        net.freeze()
        trainer = None
    else:
        net = FaceComparerModule(**model_params)
        trainer = Trainer(gpus=[torch.cuda.current_device()],
                          logger=False,
                          #fast_dev_run=False,
                          checkpoint_callback=checkpoint_callback,
                          resume_from_checkpoint=last_ckpt,
                          **trainer_params)
    #print(F'Trainer running at {trainer.logger.log_dir}')

    return net, trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FaceComparerTrainer')

    # I/O arguments
    parser.add_argument('-config_file', type=str, default='configs/linear_basic.yml', help='Config file')
    parser.add_argument('-force_restart', default=False, help='Start training from scratch even if exists', action='store_true')
    parser.add_argument('-gpu_id', default=2, type=int, help='Which gpu to use')
    parser.add_argument('-seed', default=7652252, type=int, help='Random seed')
    opts = parser.parse_args()

    torch.random.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    torch.cuda.set_device(opts.gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu_id)

    net, trainer = load_face_comparer_module(opts.config_file, opts=opts, for_eval=False)
    trainer.fit(net)
