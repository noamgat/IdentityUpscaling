import functools

import torch
from bicubic import BicubicDownSample

class LossBuilder(torch.nn.Module):
    def __init__(self, ref_im, target_identity_im, face_features_extractor, attribute_detector, loss_str, eps):
        super(LossBuilder, self).__init__()
        print(f"Loss_str = {loss_str}")
        assert ref_im.shape[2]==ref_im.shape[3]
        im_size = ref_im.shape[2]
        factor=1024//im_size
        assert im_size*factor==1024
        self.D = BicubicDownSample(factor=factor)
        self.ref_im = ref_im
        if target_identity_im is not None:
            if len(target_identity_im.shape) > 2:
                # This is an image we aim to be similar to
                self.target_identity_vector = face_features_extractor.extract_features(target_identity_im)
                self.target_identity_is_attributes = False
            else:
                # This is an attribute vector we aim to be like
                self.target_identity_vector = target_identity_im
                self.target_identity_is_attributes = True
        else:
            self.target_identity_vector = None
            self.target_identity_is_attributes = False
        self.face_features_extractor = face_features_extractor
        self.attribute_detector = attribute_detector
        self.parsed_loss = [loss_term.split('*') for loss_term in loss_str.split('+')]
        self.eps = eps
        self.iter_index = 0

    # Takes a list of tensors, flattens them, and concatenates them into a vector
    # Used to calculate euclidian distance between lists of tensors
    def flatcat(self, l):
        l = l if(isinstance(l, list)) else [l]
        return torch.cat([x.flatten() for x in l], dim=0)

    def _loss_l2(self, gen_im_lr, ref_im, **kwargs):
        return ((gen_im_lr - ref_im).pow(2).mean((1, 2, 3)).clamp(min=self.eps).sum())

    def _loss_l1(self, gen_im_lr, ref_im, **kwargs):
        return 10*((gen_im_lr - ref_im).abs().mean((1, 2, 3)).clamp(min=self.eps).sum())

    def _loss_l2_identity(self, gen_im, target_identity_vector, **kwargs):
        gen_identity_vector = self.face_features_extractor.extract_features(gen_im)
        if self.target_identity_is_attributes:
            gen_identity_vector = self.attribute_detector.forward(gen_identity_vector)
        return ((gen_identity_vector - target_identity_vector).pow(2).mean(1).sum())

    def _loss_l1_identity(self, gen_im, target_identity_vector, **kwargs):
        gen_identity_vector = self.face_features_extractor.extract_features(gen_im)
        if self.target_identity_is_attributes:
            gen_identity_vector = self.attribute_detector.forward(gen_identity_vector)
        return 10*((gen_identity_vector - target_identity_vector).abs().mean(1).sum())

    def _loss_identity_score_sigmoid(self, gen_im, target_identity_vector, **kwargs):
        logit = self.face_features_extractor.forward(gen_im, target_identity_vector)
        return torch.sigmoid(logit).mean(1).sum()

    def _loss_identity_score_l1(self, gen_im, target_identity_vector, **kwargs):
        logit = self.face_features_extractor.forward(gen_im, target_identity_vector)
        logit[logit < 0] = 0
        return logit.mean(1).sum()

    def _loss_face_attribute(self, gen_im, attr_index, target_attr_value, **kwargs):
        gen_identity_vector = self.face_features_extractor.extract_features(gen_im)
        temperature = max(100 * (0.95 ** self.iter_index), 1)
        attr_vector = self.attribute_detector.forward(gen_identity_vector) #, temperature=temperature)
        # HARD CODED HACK FOR FAIRFACE
        attr_vector = torch.sigmoid(attr_vector)
        predicted_attr_value = attr_vector[0, attr_index]
        return (predicted_attr_value - target_attr_value).pow(2).sum()

    # Uses geodesic distance on sphere to sum pairwise distances of the 18 vectors
    def _loss_geocross(self, latent, **kwargs):
        if(latent.shape[1] == 1):
            return 0
        else:
            X = latent.view(-1, 1, 18, 512)
            Y = latent.view(-1, 18, 1, 512)
            A = ((X-Y).pow(2).sum(-1)+1e-9).sqrt()
            B = ((X+Y).pow(2).sum(-1)+1e-9).sqrt()
            D = 2*torch.atan2(A, B)
            D = ((D.pow(2)*512).mean((1, 2))/8.).sum()
            return D

    def forward(self, latent, gen_im):
        self.iter_index += 1
        gen_im_lr = gen_im if gen_im.shape == self.ref_im.shape else self.D(gen_im)
        var_dict = {'latent': latent,
                    'gen_im_lr': gen_im_lr,
                    'ref_im': self.ref_im,
                    #'gen_identity_vector': gen_identity_vector,
                    'target_identity_vector': self.target_identity_vector,
                    'gen_im': gen_im
                    }
        loss = 0
        loss_fun_dict = {
            'L2': self._loss_l2,
            'L1': self._loss_l1,
            'GEOCROSS': self._loss_geocross,
            'L2_IDENTITY': self._loss_l2_identity,
            'L1_IDENTITY': self._loss_l1_identity,
            'IDENTITY_SCORE': self._loss_identity_score_sigmoid,
            'IDENTITY_SCORE_L1': self._loss_identity_score_l1,
        }
        for attr_idx in range(40):
            for attr_value in [0, 1]:
                attr_value_copy = attr_value
                loss_name = f'ATTR_{attr_idx}_IS_{attr_value}'
                loss_func = functools.partial(self._loss_face_attribute,
                                              attr_index=attr_idx, target_attr_value=attr_value_copy)
                loss_fun_dict[loss_name] = loss_func

        losses = {}
        for weight, loss_type in self.parsed_loss:
            tmp_loss = loss_fun_dict[loss_type](**var_dict)
            losses[loss_type] = tmp_loss
            loss += float(weight)*tmp_loss
        return loss, losses
