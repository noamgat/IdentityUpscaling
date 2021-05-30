import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
import os
import numpy as np
import torch
import random

from tqdm import tqdm


def build_aligned_celeba(orig_celeba_folder, new_celeba_folder, split='all', new_image_suffix='', custom_indices=None, extra_transforms=[]):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()] + extra_transforms)
    celeb_a = CelebA(root=orig_celeba_folder,
                     split=split,
                     download=False,
                     target_type='identity',
                     transform=transform)
    celeb_a.root = new_celeba_folder
    celeb_a.filename = [f"{os.path.splitext(fn)[0]}_0{new_image_suffix}.png" for fn in celeb_a.filename]
    img_folder = os.path.join(new_celeba_folder, celeb_a.base_folder, "img_align_celeba")
    existing_indices = [os.path.exists(os.path.join(img_folder, fn)) for fn in celeb_a.filename]
    if custom_indices:
        assert len(existing_indices) == len(custom_indices)
        for existing_bool, custom_bool in zip(existing_indices, custom_indices):
            if custom_bool and not existing_bool:
                raise Exception("custom_indices array refers to image that does not exist in dataset")
        existing_indices = custom_indices
    print(f"{sum(existing_indices)} / {len(celeb_a.filename)} images exist in {new_celeba_folder} split {split}")

    for list_attr in ['filename', 'identity', 'bbox', 'landmarks_align', 'attr']:
        attr_val = getattr(celeb_a, list_attr)
        filtered_list = np.array(attr_val)[existing_indices]
        if isinstance(attr_val, torch.Tensor):
            filtered_list = torch.Tensor(filtered_list).to(dtype=attr_val.dtype)
        else:
            filtered_list = list(filtered_list)
        setattr(celeb_a, list_attr, filtered_list)
    celeb_a.filtered_indices = existing_indices
    return celeb_a


class CelebAPairsDataset(Dataset):
    def __init__(self, celeb_a: CelebA, same_ratio=0.5, num_samples=10000):
        super(CelebAPairsDataset, self).__init__()
        self.celeb_a = celeb_a
        from collections import defaultdict
        identity_dicts = defaultdict(list)
        for idx, identity_idx in enumerate(celeb_a.identity):
            identity_dicts[identity_idx.item()].append(idx)
        self.identity_dicts = identity_dicts
        self.identity_indices = list(self.identity_dicts.keys())
        self.same_ratio = same_ratio
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        is_same = int(item * self.same_ratio) < int((item + 1) * self.same_ratio)
        if is_same:
            identity = 0
            while len(self.identity_dicts[identity]) < 2:
                identity, = np.random.choice(self.identity_indices, 1)
            idx1, idx2 = np.random.choice(self.identity_dicts[identity], 2, replace=False)
        else:
            identities = np.random.choice(self.identity_indices, 2, replace=False)
            idx1, = np.random.choice(self.identity_dicts[identities[0]], 1)
            idx2, = np.random.choice(self.identity_dicts[identities[1]], 1)
        return self.celeb_a[idx1][0], self.celeb_a[idx2][0], 0 if is_same else 1


class CelebAAdverserialDataset(Dataset):
    def __init__(self, celeb_a_1: CelebA, celeb_a_2: CelebA,):
        super(CelebAAdverserialDataset, self).__init__()
        self.celeb_a_1 = celeb_a_1
        self.celeb_a_2 = celeb_a_2
        assert len(self.celeb_a_1) == len(self.celeb_a_2)

    def __len__(self):
        return len(self.celeb_a_1)

    def __getitem__(self, item):
        is_same = False
        return self.celeb_a_1[item][0], self.celeb_a_2[item][0], 0 if is_same else 1

if __name__ == '__main__':
    large = build_aligned_celeba('CelebA_Raw', 'CelebA_large')
    small = build_aligned_celeba('CelebA_Raw', 'CelebA_small')
    pairs_dataset = CelebAPairsDataset(large, same_ratio=0.5)

    # Test deltas between feature vector of different photos of same person
    #from bicubic import BicubicDownsampleTargetSize
    #from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
    #downsample_to_160 = BicubicDownsampleTargetSize(160, True)
    #inception_resnet = InceptionResnetV1(pretrained='vggface2', classify=False).eval()
    #face_features_extractor = torch.nn.Sequential(downsample_to_160, inception_resnet)

    from train_face_comparer import load_face_comparer_module
    #face_comparer_module, _ = load_face_comparer_module('configs/linear_basic.yml', for_eval=True)
    face_comparer_module, _ = load_face_comparer_module('configs/arcface_basic.yml', for_eval=True)
    #face_comparer_module, _ = load_face_comparer_module('configs/sphereface_basic.yml', for_eval=True)
    face_features_extractor = face_comparer_module.face_comparer.face_features_extractor

    def run_experiment(target_dataset, num_trials, compare_feature_vectors_directly=True):
        same_person_deltas = []
        different_person_deltas = []
        for i in tqdm(range(num_trials)):
            p1, p2, is_different = target_dataset[i]
            p1 = p1.cuda()
            p2 = p2.cuda()
            if compare_feature_vectors_directly:
                images = [p1, p2]
                feature_vectors = [face_features_extractor(img.unsqueeze(0)) for img in images]
                delta_feature = (feature_vectors[1] - feature_vectors[0]).abs().sum().item()
            else:
                delta_feature = face_comparer_module.face_comparer.forward(p1.unsqueeze(0), p2.unsqueeze(0))
                delta_feature = torch.sigmoid(delta_feature).mean(1).sum().detach().cpu().item()
            if is_different:
                different_person_deltas.append(delta_feature)
            else:
                same_person_deltas.append(delta_feature)

        print(f"Number of experiments: {num_trials}")
        if len(same_person_deltas) > 0:
            m1 = np.mean(same_person_deltas)
            std1 = np.std(same_person_deltas)
            print(f"Average Same Person Delta: {m1}. STD: {std1}")
        if len(different_person_deltas) > 0:
            m2 = np.mean(different_person_deltas)
            std2 = np.std(different_person_deltas)
            print(f"Average Different Person Delta: {m2}. STD: {std2}")
        if len(same_person_deltas) > 0 and len(different_person_deltas) > 0:
            cutoff_point = m1 + (m2 - m1) * (std1 / (std1 + std2))
            cutoff_accuracy = ((np.array(same_person_deltas) < cutoff_point).sum() + (np.array(different_person_deltas) > cutoff_point).sum()) / num_trials
            print(f"Cutoff training accuracy: {100 * cutoff_accuracy}")


    #run_experiment(pairs_dataset, 1000, compare_feature_vectors_directly=True)
    # print("Aligned Test complete")
    run_experiment(pairs_dataset, 1000, compare_feature_vectors_directly=False)
    print("Aligned Test complete (Full scorer)")


    # Test prediction accuracy on adverserial dataset
    generated = build_aligned_celeba('CelebA_Raw', 'CelebA_generated', new_image_suffix='_0')
    withidentity = build_aligned_celeba('CelebA_Raw', 'CelebA_withidentity', new_image_suffix='_0')
    large_matching_generated = build_aligned_celeba('CelebA_Raw', 'CelebA_large', custom_indices=generated.filtered_indices)
    large_matching_withidentity = build_aligned_celeba('CelebA_Raw', 'CelebA_large', custom_indices=withidentity.filtered_indices)

    adverserial_dataset_1 = CelebAAdverserialDataset(generated, large_matching_generated)
    adverserial_dataset_2 = CelebAAdverserialDataset(withidentity, large_matching_withidentity)

    # Check that the datasets output matching images
    # toPIL = torchvision.transforms.ToPILImage()
    # im1, im2, is_same = adverserial_dataset_1[99]
    # toPIL(im1).save('im1.png')
    # toPIL(im2).save('im2.png')

    #run_experiment(adverserial_dataset_1, 1000)
    #print("Adverserial Test Complete")
    #run_experiment(adverserial_dataset_2, 3000)
    #print("Adverserial With Identity Test Complete")
    run_experiment(adverserial_dataset_2, 3000, compare_feature_vectors_directly=False)
    print("Adverserial With Identity Test Complete (Full scorer test)")



