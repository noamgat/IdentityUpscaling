import copy
import json
import math

import pandas
from tqdm import tqdm

from PULSE import PULSE
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
import torch
from pathlib import Path
from PIL import Image
import torchvision
from math import log10, ceil
import argparse
import os
from torchvision.datasets import CelebA
import numpy as np

class Images(Dataset):
    def __init__(self, root_dir, duplicates, targets_dir=None, filename_prefix='', celeba_db: CelebA = None, extension='png', return_target_attributes=False):
        self.root_path = Path(root_dir)
        self.image_list = list(self.root_path.glob(f"{filename_prefix}*.{extension}"))
        self.duplicates = duplicates # Number of times to duplicate the image in the dataset to produce multiple HR images
        self.targets_path = Path(targets_dir) if targets_dir else ''
        self.celeba_db = celeba_db
        self.return_target_attributes = return_target_attributes

    def __len__(self):
        return self.duplicates*len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx//self.duplicates]
        target_image = ''
        if self.targets_path:
            img_filename = os.path.split(img_path)[-1]
            if self.targets_path.is_file():
                target_img_path = self.targets_path
            elif self.targets_path.is_dir():
                target_img_path = self.targets_path.joinpath(img_filename)
                if self.celeba_db:
                    celeba_prefixes = [os.path.splitext(f)[0] for f in self.celeba_db.filename]
                    image_idx, image_prefix = \
                    [(i, prefix) for i, prefix in enumerate(celeba_prefixes) if img_filename.startswith(prefix)][0]
                    identity_idx = self.celeba_db.identity[image_idx].item()
                    identity_image_indices = [idx for idx, ident in enumerate(self.celeba_db.identity) if
                                              ident.item() == identity_idx]
                    if len(identity_image_indices) > 1:
                        identity_image_indices.remove(image_idx)
                    target_idx = identity_image_indices[(idx % self.duplicates) % len(identity_image_indices)]
                    target_filename_prefix = celeba_prefixes[target_idx]
                    target_filename = target_filename_prefix + img_filename[len(target_filename_prefix):]
                    target_img_path = self.targets_path.joinpath(target_filename)
            else:
                raise Exception(f"Invalid target image location {self.targets_path}")
            if not target_img_path.exists():
                raise Exception(f"Target image not found at {target_img_path}")
            target_image = torchvision.transforms.ToTensor()(Image.open(target_img_path))
            if self.return_target_attributes:
                target_image = self.celeba_db.attr[image_idx].to(torch.float)
        image = torchvision.transforms.ToTensor()(Image.open(img_path))
        if(self.duplicates == 1):
            return image,img_path.stem,target_image
        else:
            return image,img_path.stem+f"_{(idx % self.duplicates)+1}",target_image


class FairfaceImages(Dataset):
    def __init__(self, root_dir, fairface_csv, filename_prefix='', extension='png', **kwargs):
        self.root_path = Path(root_dir)
        self.image_list = list(self.root_path.glob(f"{filename_prefix}*.{extension}"))
        fairface_dataset = pandas.read_csv(fairface_csv)
        image_list_stems = [p.stem[:-2] for p in self.image_list]
        dataset_stems = [Path(entry).stem for entry in fairface_dataset['file']]
        dataset_lookup = {stem: idx for idx, stem in enumerate(dataset_stems)}
        #This is not the order of the model
        #race_names = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
        race_names = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White']
        image_list_race_names = [fairface_dataset['race'][dataset_lookup[stem]] for stem in image_list_stems]
        self.image_list_races = [race_names.index(race_name) for race_name in image_list_race_names]
        print("Done")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        target_image_path = self.image_list[item]
        target_image = torchvision.transforms.ToTensor()(Image.open(target_image_path))
        target_race = self.image_list_races[item]
        return target_image, target_image_path.stem, target_race


parser = argparse.ArgumentParser(description='PULSE')

#I/O arguments
parser.add_argument('-input_dir', type=str, default='input', help='input data directory')
#parser.add_argument('-targets_dir', type=str, default='targets', help='targets data directory')
parser.add_argument('-targets_dir', type=str, default=None, help='targets data directory')
parser.add_argument('-output_dir', type=str, default='runs', help='output data directory')
parser.add_argument('-output_suffix', type=str, default='0', help='output data directory')
parser.add_argument('-cache_dir', type=str, default='cache', help='cache directory for model weights')
parser.add_argument('-duplicates', type=int, default=1, help='How many HR images to produce for every image in the input directory')
parser.add_argument('-batch_size', type=int, default=1, help='Batch size to use during optimization')
parser.add_argument('-overwrite', action='store_true', help='Recreate files even if the output file exists')
parser.add_argument('-input_prefix', type=str, default='', help='Only operate on filenames begnning with X')
parser.add_argument('-output_image_type', type=str, default='jpg', help='What image type to create? png/jpg')
parser.add_argument('-copy_target', action='store_true', help='Copy the target image besides the output')
parser.add_argument('-celeba_pairs', action='store_true', help='Return target images of different images of same person')
parser.add_argument('-celeba_attributes', action='store_true', help='Aim for target attribute vectors instead of face feature vector')
parser.add_argument('-generate_celeba_feature_vectors', action='store_true', help='Should we generate the celeba feature vectors?')
parser.add_argument('-test_fairface', action='store_true', help='Should we test fairface accuracy?')
parser.add_argument('-fairface_csv_path', type=str, default='', help='Use fairface dataset instead of target dataset')


#PULSE arguments
parser.add_argument('-seed', type=int, help='manual seed to use')
parser.add_argument('-loss_str', type=str, default="100*L2+0.05*GEOCROSS+0.1*L2_IDENTITY", help='Loss function to use')
parser.add_argument('-loss_str_2', type=str, default=None, help='Loss function to use for 2nd wave of optimization')
parser.add_argument('-eps', type=float, default=2e-3, help='Target for downscaling loss (L2)')
parser.add_argument('-noise_type', type=str, default='trainable', help='zero, fixed, or trainable')
parser.add_argument('-num_trainable_noise_layers', type=int, default=5, help='Number of noise layers to optimize')
parser.add_argument('-tile_latent', action='store_true', help='Whether to forcibly tile the same latent 18 times')
parser.add_argument('-bad_noise_layers', type=str, default="17", help='List of noise layers to zero out to improve image quality')
parser.add_argument('-opt_name', type=str, default='adam', help='Optimizer to use in projected gradient descent')
parser.add_argument('-learning_rate', type=float, default=0.4, help='Learning rate to use during optimization')
parser.add_argument('-steps', type=int, default=100, help='Number of optimization steps')
parser.add_argument('-lr_schedule', type=str, default='linear1cycledrop', help='fixed, linear1cycledrop, linear1cycle')
parser.add_argument('-save_intermediate', action='store_true', help='Whether to store and save intermediate HR and LR images during optimization')
parser.add_argument('-gpu_id', default='0', type=str, help='Which gpu to use. Can also use multigpu format')
parser.add_argument('-face_comparer_config', default='configs/linear_basic.yml', type=str, help='YML file of face comparer')
parser.add_argument('-use_stylegan2', action='store_true', help='Whether to use stylegan2 (default=stylegan1)')

kwargs = vars(parser.parse_args())

#torch.cuda.set_device('cuda:' + kwargs['gpu_id'])
os.environ['CUDA_VISIBLE_DEVICES'] = str(kwargs['gpu_id'])

celeb_a = CelebA(root='CelebA_Raw', split='all') if (kwargs["celeba_pairs"] or kwargs["celeba_attributes"]) else None
dataset = Images(kwargs["input_dir"],
                 duplicates=kwargs["duplicates"],
                 targets_dir=kwargs["targets_dir"],
                 filename_prefix=kwargs["input_prefix"],
                 celeba_db=celeb_a,
                 return_target_attributes=kwargs["celeba_attributes"])
if kwargs['fairface_csv_path']:
    dataset = FairfaceImages(kwargs["input_dir"], kwargs['fairface_csv_path'], kwargs["input_prefix"])
    print("Using fairface dataset, will replace ATTR_SOURCE_IS_1 in loss string with high res source's race")
print(f"Running on {len(dataset)} files")
#targets_dataset = Images(kwargs["targets_dir"], duplicates=1)
out_path = Path(kwargs["output_dir"])
output_suffix = kwargs["output_suffix"]
out_path.mkdir(parents=True, exist_ok=True)
ouptut_image_type = kwargs["output_image_type"]
copy_target = kwargs["copy_target"]
dataloader = DataLoader(dataset, batch_size=kwargs["batch_size"])

model = PULSE(cache_dir=kwargs["cache_dir"],
              face_comparer_config=kwargs['face_comparer_config'],
              use_stylegan2=kwargs['use_stylegan2']
              )
print("Loaded model")
#model = model.cuda()
# model = DataParallel(model)

toPIL = torchvision.transforms.ToPILImage()

#from bicubic import BicubicDownsampleTargetSize
#test = BicubicDownsampleTargetSize.downsampling(target_identity_im.unsqueeze(0), (50, 50), mode='area').squeeze(0)
#toPIL(test.cpu().detach().clamp(0, 1)).save('runs/downsample.png')
#toPIL(target_identity_im.cpu().detach().clamp(0, 1)).save('runs/input.png')
#exit(0)

def vector_angle(f1, f2):
    dot_product = (f1 * f2).sum(-1)
    normalized = (f1.norm(dim=-1) * f2.norm(dim=-1) + 1e-5)
    cosdistance = dot_product / normalized
    angledistance = torch.acos(cosdistance) * (180 / math.pi)
    return angledistance

if kwargs['generate_celeba_feature_vectors']:
    features_files = [
        kwargs['face_comparer_config'] + '.celeba_features.json',
        kwargs['face_comparer_config'] + '.fairface_train_features.json',
        kwargs['face_comparer_config'] + '. fairface_val_features.json'
    ]
    image_dirs = [
        'CelebA_Raw/celeba/img_align_celeba',
        'fairface/train',
        'fairface/val'
    ]
    for features_file, image_dir in zip(features_files, image_dirs):
        if not os.path.exists(features_file) or kwargs['overwrite']:
            print(f"Started generating feature vectors {features_file}")
            feature_dict = {}
            celeba_images = Images(image_dir, duplicates=1, extension='jpg')
            bs = kwargs['batch_size']
            dataloader = DataLoader(celeba_images, batch_size=bs)
            for im, stem, target_image in tqdm(dataloader):
                feature_vector = model.face_features_extractor.face_features_extractor.forward(im.cuda())
                for i in range(bs):
                    try:
                        feature_list = feature_vector[i].detach().cpu().numpy().tolist()
                        feature_dict[stem[i]] = feature_list
                    except Exception as e:
                        print(f"Error: {e}")
            open(features_file, "w").write(json.dumps(feature_dict, indent=2, sort_keys=True))
            print(f"Finished generating feature vectors {features_file}")
    # averages_file = kwargs['face_comparer_config'] + '.attribs.npy'
    # celeb_a_train = CelebA(root='CelebA_Raw', split='train')
    # if not os.path.exists(averages_file) or kwargs['overwrite']:
    #     feature_dict = json.load(open(features_file, "r"))
    #     num_attrs = len(celeb_a_train.attr_names)
    #     num_features = len(next(iter(feature_dict.values())))
    #     attr_match_matrix = torch.zeros((num_attrs, num_features), dtype=torch.float)
    #     attr_mismatch_matrix = torch.zeros((num_attrs, num_features), dtype=torch.float)
    #     for i, fn in enumerate(celeb_a_train.filename):
    #         fn = os.path.splitext(fn)[0]
    #         if fn in feature_dict:
    #             feature_vector = torch.FloatTensor(feature_dict[fn]).unsqueeze(0)
    #             attrib_vector = celeb_a_train.attr[i].unsqueeze(1).type(torch.float)
    #             average_contribution = (attrib_vector * feature_vector)
    #             attr_match_matrix += average_contribution
    #             attrib_vector = 1 - attrib_vector
    #             average_contribution = (attrib_vector * feature_vector)
    #             attr_mismatch_matrix += average_contribution
    #     num_attrib_matches = celeb_a_train.attr.sum(dim=0)
    #     num_attrib_mismatches = len(celeb_a_train.filename) - num_attrib_matches
    #     for attr_idx in range(num_attrs):
    #         attr_match_matrix[attr_idx] /= num_attrib_matches[attr_idx]
    #         attr_mismatch_matrix[attr_idx] /= num_attrib_mismatches[attr_idx]
    #
    #     torch.save({'match': attr_match_matrix, 'mismatch': attr_mismatch_matrix}, averages_file)
    #     print(f"Finished generating attribute vectors for config {kwargs['face_comparer_config']}")
    # attr_match_obj = torch.load(averages_file)
    # attr_match_matrix = attr_match_obj['match']
    # attr_mismatch_matrix = attr_match_obj['mismatch']
    #
    # feature_dict = json.load(open(features_file, "r"))
    # num_attrs = attr_match_matrix.shape[0]
    # attrib_accuracy = torch.zeros((num_attrs, ), dtype=torch.float)
    # num_tests = 0
    # for i, fn in enumerate(celeb_a_train.filename):
    #     fn = os.path.splitext(fn)[0]
    #     if fn in feature_dict:
    #         feature_vector = torch.FloatTensor(feature_dict[fn]).unsqueeze(0)
    #         attrib_vector = celeb_a_train.attr[i].type(torch.float)
    #         match_scores = torch.matmul(attr_match_matrix, feature_vector.T).squeeze(1)
    #         mismatch_scores = torch.matmul(attr_mismatch_matrix, feature_vector.T).squeeze(1)
    #         match_decision = match_scores > mismatch_scores
    #         success_vector = match_decision == attrib_vector
    #         attrib_accuracy += success_vector
    #         num_tests += 1
    # attrib_accuracy /= num_tests
    #
    # for attr_idx in range(num_attrs):
    #     attr_vec = attr_match_matrix[attr_idx]
    #     attr_vec_abs = attr_vec.abs()
    #     attr_norm = attr_vec_abs.sum(dim=-1)
    #     num_large_attributes = (attr_vec_abs > (attr_norm * 0.01)).sum()
    #     attr_angle = vector_angle(attr_match_matrix[attr_idx], attr_mismatch_matrix[attr_idx])
    #     #print(f"Match<->Mismatch Angle for Attribute {celeb_a_train.attr_names[attr_idx]} : {attr_angle}")
    #     #print(f"Meaningful dims: {num_large_attributes}")
    #     print(f"Accuracy for attribute {celeb_a_train.attr_names[attr_idx]} : {100*attrib_accuracy[attr_idx]:.1f}")

    exit(0)

if kwargs['test_fairface']:
    from fairface_dataset import FairfaceDataset
    test_dataset = FairfaceDataset(split='val')
    num_correct = 0
    num_images = 0
    try:
        for img_path, attr_vector in tqdm(test_dataset):
            im = torchvision.transforms.ToTensor()(Image.open(img_path))
            im = im.unsqueeze(0)
            feature_vector = model.face_features_extractor.face_features_extractor.forward(im.cuda())
            race_vector = model.face_features_extractor.face_features_extractor.race_detector(feature_vector)
            selected_race = race_vector[0].argmax()
            correct_race = attr_vector.argmax()
            is_correct = selected_race == correct_race
            if is_correct:
                num_correct += 1
            num_images += 1
    except:
        pass

    accuracy = num_correct / num_images
    print(f"TEST FAIRFACE accuracy : {num_correct} / {num_images} ({accuracy*100:.2f}%)")
    exit(0)

for ref_im, ref_im_name, target_identity_im in dataloader:
    if not kwargs['overwrite']:
        skip_batch = True
        for i in range(kwargs["batch_size"]):
            output_filename = out_path / f"{ref_im_name[i]}_{output_suffix}.png"
            if not os.path.exists(output_filename):
                skip_batch = False
                break
        if skip_batch:
            print(f"Skipping batch of files {ref_im_name} as the outputs exist")
            continue
    ref_im = ref_im.cuda()
    kwargs_copy = copy.deepcopy(kwargs)
    if kwargs['celeba_attributes']:
        target_identity_im = target_identity_im.type(torch.long)
    if isinstance(target_identity_im, torch.LongTensor):
        target_race_or_attr_vector = target_identity_im
        if kwargs['celeba_attributes']:
            for i in range(len(target_race_or_attr_vector[0])):
                kwargs_copy['loss_str'] = kwargs_copy['loss_str'].replace(f'ATTR_{i}_IS_SOURCE',
                                                                          f'ATTR_{i}_IS_{target_race_or_attr_vector[0][i].item()}')
        else:
            kwargs_copy['loss_str'] = kwargs_copy['loss_str'].replace('ATTR_SOURCE_IS_1',
                                                                      f'ATTR_{target_race_or_attr_vector.item()}_IS_1')
    else:
        target_race_or_attr_vector = None
    if isinstance(target_identity_im, torch.FloatTensor):
        target_identity_im = target_identity_im.cuda()
    else:
        target_identity_im = None
    if(kwargs["save_intermediate"]):
        padding = ceil(log10(100))
        for i in range(kwargs["batch_size"]):
            int_path_HR = Path(out_path / ref_im_name[i] / "HR")
            int_path_LR = Path(out_path / ref_im_name[i] / "LR")
            int_path_HR.mkdir(parents=True, exist_ok=True)
            int_path_LR.mkdir(parents=True, exist_ok=True)
        for j,(HR,LR) in enumerate(model(ref_im,target_identity_im,**kwargs_copy)):
            for i in range(kwargs["batch_size"]):
                toPIL(HR[i].cpu().detach().clamp(0, 1)).save(
                    int_path_HR / f"{ref_im_name[i]}_{j:0{padding}}_{output_suffix}.{ouptut_image_type}")
                toPIL(LR[i].cpu().detach().clamp(0, 1)).save(
                    int_path_LR / f"{ref_im_name[i]}_{j:0{padding}}_{output_suffix}.{ouptut_image_type}")
    else:
        #out_im = model(ref_im,**kwargs)
        for j,(HR,LR) in enumerate(model(ref_im, target_identity_im, **kwargs_copy)):
            for i in range(kwargs["batch_size"]):
                output_filename = out_path / f"{ref_im_name[i]}_{output_suffix}_{j}.{ouptut_image_type}"
                toPIL(HR[i].cpu().detach().clamp(0, 1)).save(output_filename)
                print(f"Created {output_filename}")
                if copy_target:
                    output_filename = out_path / f"{ref_im_name[0]}_{output_suffix}_target.{ouptut_image_type}"
                    toPIL(target_identity_im[i].cpu().detach().clamp(0, 1)).save(output_filename)
                    print(f"Copied target {output_filename}")

