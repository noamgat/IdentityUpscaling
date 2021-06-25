import math

from bicubic import BicubicDownsampleTargetSize
# from sphereface_pytorch.net_sphere import sphere20a
from stylegan import G_synthesis,G_mapping
from dataclasses import dataclass
from SphericalOptimizer import SphericalOptimizer
from pathlib import Path
import numpy as np
import time
import torch
from loss import LossBuilder
from functools import partial
from drive import open_url
from facenet_pytorch import InceptionResnetV1


def build_mlp(input_dim, hidden_dims, output_dim):
    dims = [input_dim] + hidden_dims + [output_dim]
    layers = []
    for i in range(1, len(dims)):
        layers.append(torch.nn.Linear(dims[i-1], dims[i]))
        if i < len(dims)-1:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)

class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class FaceComparer(torch.nn.Module):
    FEATURE_NORMALIZATION_ABS = 0
    FEATURE_NORMALIZATION_SIGN = 1
    FEATURE_NORMALIZATION_SQUARE = 2
    FEATURE_NORMALIZATION_ANGLE = 3
    FEATURE_NORMALIZATION_DOT = 4

    def __init__(self,
                 feature_extractor_model='facenet',
                 load_pretrained=True,
                 hidden_dims=[],
                 initial_bias=None,
                 feature_normalization_mode=FEATURE_NORMALIZATION_ABS,
                 feature_extractor_params={},
                 feature_normalization_scale=1.0):
        super(FaceComparer, self).__init__()
        if feature_extractor_model == 'facenet':
            self.face_features_extractor = self.create_facenet_features_extractor(load_pretrained, **feature_extractor_params)
        elif feature_extractor_model == 'arcface':
            self.face_features_extractor = self.create_arcface_features_extractor(load_pretrained, **feature_extractor_params)
        else:
            raise Exception(f"Invalid feature extractor model '{feature_extractor_model}'")
        first_tail_dimension = 1 if feature_normalization_mode in \
                                    [self.FEATURE_NORMALIZATION_ANGLE, self.FEATURE_NORMALIZATION_DOT] else 512
        self.tail = build_mlp(first_tail_dimension, hidden_dims, 1)
        self.feature_normalization_mode = feature_normalization_mode
        last_fc = self.tail[0]
        if initial_bias is not None:
            last_fc.weight.data = torch.ones_like(last_fc.weight.data)
            last_fc.bias.data = torch.ones_like(last_fc.bias.data) * initial_bias
        self.feature_normalization_scale = feature_normalization_scale
        print("Done")

    def create_facenet_features_extractor(self, load_pretrained):
        downsample_to_160 = BicubicDownsampleTargetSize(160, True)
        pretrained_name = 'vggface2' if load_pretrained else None
        feature_extractor = InceptionResnetV1(pretrained=pretrained_name, classify=False)
        facenet_features_extractor = torch.nn.Sequential(downsample_to_160, feature_extractor)
        return facenet_features_extractor

    def create_arcface_features_extractor(self, load_pretrained, model_path=None):
        from arcface_features_extractor import ArcfaceFeaturesExtractor
        return ArcfaceFeaturesExtractor(load_pretrained, model_path)

    def extract_features(self, image_or_features):
        if len(image_or_features.shape) == 2:
            # Channels: Batch Size, Features
            return image_or_features
        else:
            # Channels: Batch Size, CHW
            #images = self.image_extractor(image_or_features)
            return self.face_features_extractor(image_or_features)

    def extract_race_vector(self, image_or_features):
        features = self.extract_features(image_or_features)
        race_vector = self.face_features_extractor.race_detector(features)
        return race_vector

    def forward(self, x_1, x_2):
        # Allow the forward pass to accept both images and pre-calculated feature vectors
        features_1 = self.extract_features(x_1)
        features_2 = self.extract_features(x_2)
        features_diff = features_1 - features_2

        if self.feature_normalization_mode == self.FEATURE_NORMALIZATION_ABS:
            features_diff = abs(features_diff)
        elif self.feature_normalization_mode == self.FEATURE_NORMALIZATION_SQUARE:
            features_diff = torch.pow(features_diff, 2)
        elif self.feature_normalization_mode == self.FEATURE_NORMALIZATION_SIGN:
            # Make sure the first element of every vector is non-negative - flip if negative
            is_nonnegative = features_diff[:, 0] >= 0
            sign_vector = ((is_nonnegative * 1) - 0.5) * 2
            features_diff *= sign_vector.unsqueeze(0).T
        elif self.feature_normalization_mode == self.FEATURE_NORMALIZATION_ANGLE:
            f1 = features_1
            f2 = features_2
            # https://github.com/pytorch/pytorch/issues/18027 No batch dot product
            dot_product = (f1 * f2).sum(-1)
            if f1.norm(dim=1).mean().item() > 1000000:
                print("WARNING: Angle loss: Norms: %.2f, %.2f" % (f1.norm(dim=1).mean(), f2.norm(dim=1).mean()))
                features_1 = self.extract_features(x_1)
            normalized = (f1.norm(dim=1) * f2.norm(dim=1) + 1e-5)
            cosdistance = dot_product / normalized
            angledistance = torch.acos(cosdistance) * (180 / math.pi)
            ## Change from -1 (opposite) -> 1 (same) range to 0 (same) - 1 (different)
            #features_diff = (torch.ones_like(cosdistance) - cosdistance) / 2
            features_diff = angledistance.unsqueeze(1)
        else:
            raise Exception("unknown normalization mode " + self.feature_normalization_mode)

        mlp_output = self.tail(features_diff)
        #mlp_output = mlp_output.squeeze(1)
        # decision = torch.sigmoid(mlp_output) #Using BCE loss, that will sigmoid
        decision = mlp_output
        decision = decision * self.feature_normalization_scale
        #threshold_decision = features_diff.sum(dim=1)
        #threshold_decision = threshold_decision - 21
        #threshold_decision = threshold_decision.unsqueeze(1)
        #return threshold_decision
        return decision
