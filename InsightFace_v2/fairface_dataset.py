import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

from utils import get_central_face_attributes


class FairfaceDataset(Dataset):

    def __init__(self, path='fairface', split='train'):
        self.path = path
        self.labels = pd.read_csv(os.path.join(path, f'fairface_label_{split}.csv'))
        self.race_one_hot = pd.get_dummies(self.labels.race)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image_path = os.path.join(self.path, self.labels.file[item])
        attr_vector = self.race_one_hot.values[item]
        return item, image_path, attr_vector

    def set_face_points(self, item, bbox, landmark):
        bbox_path = os.path.join(self.path, self.labels.file[item]) + ".bbox.pt"
        torch.save([bbox, landmark], bbox_path)

    def get_face_points(self, item):
        bbox_path = os.path.join(self.path, self.labels.file[item]) + ".bbox.pt"
        if os.path.exists(bbox_path):
            bbox, landmark = torch.load(bbox_path)
            return bbox, landmark
        return None, None

    @property
    def races(self):
        return list(self.race_one_hot.columns)


if __name__ == '__main__':
    train_dataset = FairfaceDataset(path='../fairface', split='train')
    #print(train_dataset.races)
    for idx, img_path, attr_vector in tqdm(train_dataset):
        bboxes, landmarks = get_central_face_attributes(img_path)
        if bboxes:
            train_dataset.set_face_points(idx, bboxes[0], landmarks[0])
            # load_test = train_dataset.get_face_points(idx)

