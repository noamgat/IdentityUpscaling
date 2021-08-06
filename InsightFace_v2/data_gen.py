import os
import pickle

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from celeba_aligned_copy import build_aligned_celeba, CelebAAdverserialDataset, CelebAPairsDataset
from config import IMG_DIR
from config import pickle_file

from fairface_dataset import FairfaceDataset
import numpy as np

# Data augmentation and normalization for training
# Just normalization for validation
from utils import get_central_face_attributes

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class ArcFaceDataset(Dataset):
    def __init__(self, split):
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        self.split = split
        self.samples = data
        self.transformer = data_transforms['train']

    def __getitem__(self, i):
        sample = self.samples[i]
        filename = sample['img']
        label = sample['label']

        filename = os.path.join(IMG_DIR, filename)
        img = Image.open(filename).convert('RGB')
        img = self.transformer(img)

        return img, label

    def __len__(self):
        return len(self.samples)


class AdverserialFaceDataset(Dataset):
    def __init__(self, split):
        celeba_raw = build_aligned_celeba('../CelebA_Raw', '../CelebA_large', split=split)
        #celeba_orig = CelebA(root='../CelebA_Raw', split='all', download=False, target_type='identity')
        #celeba = celeba_orig

        generated_suffix = 'withidentity_256'  # can also be 'generated', 'metaepoch2', 'withidentity'
        generated = build_aligned_celeba('../CelebA_Raw', f'../CelebA_{generated_suffix}', new_image_suffix='_0', split=split)
        large_matching_generated = build_aligned_celeba('../CelebA_Raw', '../CelebA_large',
                                                        custom_indices=generated.filtered_indices, split=split)
        adverserial_dataset_1 = CelebAAdverserialDataset(generated, large_matching_generated, return_indices=True)
        pairs_dataset = CelebAPairsDataset(celeba_raw, same_ratio=1, num_samples=len(adverserial_dataset_1), return_indices=True)
        self.generated_dataset = adverserial_dataset_1
        self.pairs_dataset = pairs_dataset

        import celeba_eval

        generated_samples_file = f'data/adv_generated_{generated_suffix}.pkl'
        if not os.path.exists(generated_samples_file):
            celeba_eval.process(None, self.generated_dataset, f'data/adv_generated_{generated_suffix}_pairs.txt', generated_samples_file)
        with open(generated_samples_file, 'rb') as file:
            data = pickle.load(file)
            self.generated_samples = data['samples']

        pairs_sample_file = 'data/adv_pairs.pkl'
        if not os.path.exists(pairs_sample_file):
            celeba_eval.process(None, self.pairs_dataset, 'data/adv_pairs_pairs.txt', pairs_sample_file)
        with open(pairs_sample_file, 'rb') as file:
            data = pickle.load(file)
            self.pairs_samples = data['samples']
        self.transformer = data_transforms[split]


    def __len__(self):
        return len(self.generated_dataset) + len(self.pairs_dataset)

    def __getitem__(self, item):
        import celeba_eval
        data_source = self.generated_dataset if (item % 2) == 0 else self.pairs_dataset
        samples = self.generated_samples if (item % 2) == 0 else self.pairs_samples
        data_index = item // 2
        (data_source_1, idx1), (data_source_2, idx2), is_different = data_source[data_index]
        fn1 = data_source_1.filename[idx1]
        fn2 = data_source_2.filename[idx2]
        img1 = celeba_eval.get_image(samples, self.transformer, fn1)
        img2 = celeba_eval.get_image(samples, self.transformer, fn2)
        if img1 is None or img2 is None:
            # Corrupt image or faulty landmark detection
            item = (item + 1) % len(self)
            return self[item]
        return img1, img2, is_different


class FairfaceImageDataset(Dataset):

    def __init__(self, split):
        self.inner_dataset = FairfaceDataset('../fairface', split=split)
        self.transformer = data_transforms[split]

    def __getitem__(self, i):
        item, file, label = self.inner_dataset[i]
        from config import device

        #Copy-paste from get_image
        bbox, landmarks = self.inner_dataset.get_face_points(item)
        if landmarks is None:
            #print(f"landmarks {file} has no landmarks - None")
            return self[(i+1) % len(self)]
        num_landmarks = np.prod(np.array(landmarks).shape)
        if num_landmarks != 10:
            #print(f"Image {file} has no landmarks - number of elements = {num_landmarks}")
            return self[(i+1) % len(self)]

        from utils import align_face
        img = align_face(file, landmarks)  # BGR
        # img = blur_and_grayscale(img)
        img = img[..., ::-1]  # RGB
        img = Image.fromarray(img, 'RGB')  # RGB
        img = self.transformer(img)
        img = img.to(device)

        return img, label

    def __len__(self):
        return len(self.inner_dataset)


class CelebAAttributesDataset(Dataset):
    def __init__(self, split):
        from torchvision.datasets import CelebA
        celeb_a = CelebA(root='../CelebA_Raw', split='valid' if split == 'val' else split)
        for attrib_idx, attrib_name in enumerate(celeb_a.attr_names):
            print(f"Attribute {attrib_idx} : {attrib_name}")
        self.num_features = len(celeb_a.attr_names)
        attrib_vector_sum = torch.zeros(40, dtype=torch.float)
        num_entries = 0
        for i, attrib_vector in enumerate(celeb_a.attr):
            attrib_vector_sum += attrib_vector
            num_entries += 1
        self.attrib_vector_mean = attrib_vector_sum / num_entries
        self.weight_multiplier_label_1 = (1 / self.attrib_vector_mean)
        self.weight_multiplier_label_0 = (1 / (1 - self.attrib_vector_mean))
        self.celeb_a = celeb_a
        self.transformer = data_transforms[split]

    def __len__(self):
        return len(self.celeb_a)

    def __getitem__(self, item):
        file = self.celeb_a.filename[item]
        img_path = os.path.join(self.celeb_a.root, self.celeb_a.base_folder, "img_align_celeba", file)
        from config import device

        # Copy-paste from get_image
        bbox, landmarks = self.get_face_points(img_path)
        if landmarks is None:
            # print(f"landmarks {file} has no landmarks - None")
            return self[(item + 1) % len(self)]
        num_landmarks = np.prod(np.array(landmarks).shape)
        if num_landmarks != 10:
            # print(f"Image {file} has no landmarks - number of elements = {num_landmarks}")
            return self[(item + 1) % len(self)]

        from utils import align_face
        img = align_face(img_path, landmarks)  # BGR
        # img = blur_and_grayscale(img)
        img = img[..., ::-1]  # RGB
        img = Image.fromarray(img, 'RGB')  # RGB
        img = self.transformer(img)
        img = img.to(device)

        label = self.celeb_a.attr[item]
        label_weights = label * self.weight_multiplier_label_1 + (1 - label) * self.weight_multiplier_label_0
        return img, label, label_weights


    def get_face_points(self, img_path):
        bbox_path = img_path + ".bbox.pt"
        if os.path.exists(bbox_path):
            bbox, landmark = torch.load(bbox_path)
            return bbox, landmark
        bboxes, landmarks = get_central_face_attributes(img_path)
        if bboxes:
            torch.save([bboxes[0], landmarks[0]], bbox_path)
        else:
            torch.save([None, None], bbox_path)
        return self.get_face_points(img_path)

