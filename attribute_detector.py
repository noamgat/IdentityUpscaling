import argparse
import json
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CelebA


#FEATURE_INDEX = 0
from fairface_dataset import FairfaceDataset


def build_mlp(input_dim, hidden_dims, output_dim):
    dims = [input_dim] + hidden_dims + [output_dim]
    layers = []
    for i in range(1, len(dims)):
        layers.append(torch.nn.Linear(dims[i-1], dims[i]))
        if i < len(dims)-1:
            layers.append(torch.nn.BatchNorm1d(num_features=dims[i]))
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


class CelebAFeaturesDataset(Dataset):
    def __init__(self, feature_dict, split):
        celeb_a_train = CelebA(root='CelebA_Raw', split=split)
        for attrib_idx, attrib_name in enumerate(celeb_a_train.attr_names):
            print(f"Attribute {attrib_idx} : {attrib_name}")
        self.num_features = len(celeb_a_train.attr_names)
        self.feature_attrib_pairs = []
        attrib_vector_sum = torch.zeros(40, dtype=torch.float)
        num_entries = 0
        for i, fn in enumerate(celeb_a_train.filename):
            fn = os.path.splitext(fn)[0]
            if fn in feature_dict:
                feature_vector = feature_dict[fn]
                attrib_vector = celeb_a_train.attr[i]
                attrib_vector_sum += attrib_vector
                #attrib_vector = attrib_vector[FEATURE_INDEX:FEATURE_INDEX+1]
                self.feature_attrib_pairs.append((feature_vector, attrib_vector))
                num_entries += 1
        self.attrib_vector_mean = attrib_vector_sum / num_entries

    def __len__(self):
        return len(self.feature_attrib_pairs)

    def __getitem__(self, item):
        feature, attrib = self.feature_attrib_pairs[item]
        feature_vector = torch.FloatTensor(feature)
        attrib_vector = attrib.type(torch.float)
        return feature_vector, attrib_vector


class FairfaceFeaturesDataset(Dataset):
    def __init__(self, feature_dict, split):
        if split == 'valid':
            split = 'val'
        fairface_dataset = FairfaceDataset(split=split)
        for attrib_idx, attrib_name in enumerate(fairface_dataset.races):
            print(f"Attribute {attrib_idx} : {attrib_name}")
        self.num_features = len(fairface_dataset.races)
        # self.num_features = 2
        self.feature_attrib_pairs = []
        for i in range(len(fairface_dataset)):
            fn, attrib_vector = fairface_dataset[i]
            try:
                fn = os.path.basename(os.path.splitext(fn)[0])
                if fn in feature_dict:
                    feature_vector = feature_dict[fn]
                    # attrib_vector = attrib_vector[FEATURE_INDEX:FEATURE_INDEX+1]
                    self.feature_attrib_pairs.append((feature_vector, attrib_vector))
            except:
                print("AH")
        print("Done")
        self.dataset_balance = None

    def __len__(self):
        return len(self.feature_attrib_pairs)

    def __getitem__(self, item):
        feature, attrib = self.feature_attrib_pairs[item]
        feature_vector = torch.FloatTensor(feature)
        attrib_vector = torch.from_numpy(attrib).type(torch.float)
        attrib_vector = torch.FloatTensor([attrib_vector[0], 1 - attrib_vector[0]])
        return feature_vector, attrib_vector


class AttributeDetectorModule(pl.LightningModule):
    def __init__(self, num_outputs=None, is_one_hot=False, dataset_balance=None):
        super().__init__()
        #self.l1 = torch.nn.Linear(512, 40)
        num_outputs = num_outputs or 40
        self.l1 = build_mlp(512, [1024], num_outputs)
        self.is_one_hot = is_one_hot
        self.dataset_balance = dataset_balance

    def forward(self, x):
        logits = self.l1(x)
        if self.is_one_hot:
            return logits
        else:
            return torch.sigmoid(logits)

    def num_attributes(self):
        return self.l1[-1].out_features

    def shared_training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        if self.is_one_hot:
            loss = F.binary_cross_entropy_with_logits(y_hat, y)
            num_correct = (y_hat.argmax(dim=1) == y.argmax(dim=1)).to(float).mean().item() * 100
        else:
            diff = F.mse_loss(y_hat, y, reduce=False)
            y_weights = torch.zeros(y.shape)
            for batch_idx, labels in enumerate(y.to(int)):
                for attr_idx, attr_val in enumerate(labels):
                    balance_factor = self.dataset_balance[attr_idx] if attr_val else 1 - self.dataset_balance[attr_idx]
                    y_weights[batch_idx][attr_idx] = 1 / balance_factor
            y_weights = y_weights.to(diff.device)
            loss = (diff * y_weights).mean()
            num_correct = int((y_hat.round() == y).to(float).mean().item() * 100)
        return y_hat, loss, num_correct

    def training_step(self, batch, batch_idx):
        y_hat, loss, num_correct = self.shared_training_step(batch)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs, 'num_correct': num_correct}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, loss, num_correct = self.shared_training_step(batch)
        confusion_matrix = torch.zeros(self.num_attributes(), 2, 2, dtype=torch.float)
        y_predict = y_hat.round().to(int)
        for predict, answer in zip(y_predict, y.to(int)):
            for attr_idx, (single_predict, single_label) in enumerate(zip(predict, answer)):
                confusion_matrix[attr_idx, single_label.item(), single_predict.item()] += 1
        accuracy_per_attr = (y_hat.round() == y).float().mean(dim=0).detach().cpu()  # TODO ONE HOT
        metrics = {'val_acc': num_correct, 'val_loss': loss, 'val_acc_per_attr': accuracy_per_attr, 'val_confusion': confusion_matrix}
        return metrics

    def validation_epoch_end(self, outputs):
        accs = [float(output['val_acc']) for output in outputs]
        val_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        acc = sum(accs) / len(accs)
        print(f"Epoch Accuracy: {acc:.2f}")
        attr_accs = [output['val_acc_per_attr'] for output in outputs]
        attr_acc = (torch.stack(attr_accs).mean(dim=0) * 100).round()
        print(f"Per attribute accuracy:")
        print(attr_acc.tolist())
        confusion_matrix = torch.stack([output['val_confusion'] for output in outputs]).sum(dim=0) + 1e-4
        for attr_idx in range(confusion_matrix.shape[0]):
            for label_val in [0, 1]:
                confusion_matrix[attr_idx][label_val] /= confusion_matrix[attr_idx][label_val].sum()
        conf_acc = [attr_mat.diag().mean().item() * 100 for attr_mat in confusion_matrix]
        print(f"Per attribute confusion accuracy:")
        print(['%.1f' % acc for acc in conf_acc])
        return {'Training/_accuracy': acc, 'val_loss': val_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def load_attribute_detector_from_checkpoint(ckpt_file):
    loaded_model = AttributeDetectorModule.load_from_checkpoint(ckpt_file, map_location='cuda:0')
    return loaded_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attribute Detector')
    parser.add_argument('-gpu_id', default='2', type=str, help='Which gpu to use. Can also use multigpu format')
    parser.add_argument('-face_comparer_config', default='configs/arcface_adv.yml', type=str,
                        help='YML file of face comparer')
    parser.add_argument('-batch_size', type=int, default=16, help='Batch size to use during optimization')
    parser.add_argument('-ckpt', type=str, default=None, help='Checkpoint to start training from')
    parser.add_argument('-dataset', type=str, default='celeba', help='Which data set to use? celeba / fairface')
    kwargs = vars(parser.parse_args())

    gpu_id = kwargs['gpu_id']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    #torch.cuda.set_device(gpu_id)

    dataset_name = kwargs['dataset']
    is_fairface = dataset_name == 'fairface'
    is_one_hot = is_fairface
    dataset_suffix = ".fairface_train_features.json" if is_fairface else ".celeba_features.json"
    features_file = kwargs['face_comparer_config'] + dataset_suffix
    if not os.path.exists(features_file):
        raise Exception(f"Features json f{features_file} does not exist")
    feature_dict = json.load(open(features_file, "r"))

    dataset_class = FairfaceFeaturesDataset if is_fairface else CelebAFeaturesDataset
    dataset_train = dataset_class(feature_dict, 'train')
    print(f"Loaded dataset with {len(dataset_train)} feature vectors")
    dataset_valid = dataset_class(feature_dict, 'valid')
    train_loader = DataLoader(dataset_train, batch_size=kwargs['batch_size'], shuffle=True)#, num_workers=2)
    val_loader = DataLoader(dataset_valid, batch_size=kwargs['batch_size'])#, num_workers=2)
    config_dir, config_file = os.path.split(kwargs['face_comparer_config'])
    attribute_model_file = os.path.splitext(config_file)[0] + "_" + dataset_name + "_"
    print("Saving trained model to " + attribute_model_file)
    checkpoint_callback = ModelCheckpoint(config_dir, save_weights_only=True, prefix=attribute_model_file)

    trainer = pl.Trainer(gpus=[torch.cuda.current_device()],
                         checkpoint_callback=checkpoint_callback)
    if kwargs['ckpt']:
        model = load_attribute_detector_from_checkpoint(kwargs['ckpt'])
    else:
        model = AttributeDetectorModule(num_outputs=dataset_train.num_features, is_one_hot=is_one_hot, dataset_balance=dataset_train.attrib_vector_mean)

    trainer.fit(model, train_loader, val_dataloaders=val_loader)
