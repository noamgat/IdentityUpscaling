from collections import defaultdict

import torchvision
from PIL import Image
from tqdm import tqdm

from data_gen import CelebAAttributesDataset
import torch


def celeba_attr_test(feature_extractor, tail_fc):
    feature_extractor.eval()
    tail_fc.eval()
    test_dataset = CelebAAttributesDataset(split='val')
    confusion_matrix = torch.zeros(40, 2, 2, dtype=torch.float)

    test_i = 0
    try:
        for im, y, attr_weight in tqdm(test_dataset):
            im = im.unsqueeze(0)
            feature_vector = feature_extractor(im.cuda())
            y_hat = tail_fc(feature_vector)
            y_hat = torch.nn.functional.sigmoid(y_hat)

            y_predict = y_hat[0].round().to(int)
            for attr_idx, (single_predict, single_label) in enumerate(zip(y_predict, y)):
                confusion_matrix[attr_idx, single_label.item(), single_predict.item()] += 1
            test_i += 1
            if test_i > 100:
                break
    except Exception as e:
        pass

    for attr_idx in range(confusion_matrix.shape[0]):
        for label_val in [0, 1]:
            confusion_matrix[attr_idx][label_val] /= confusion_matrix[attr_idx][label_val].sum()
    conf_acc = [attr_mat.diag().mean().item() * 100 for attr_mat in confusion_matrix]
    print(f"Per attribute confusion accuracy:")
    print(['%.1f' % acc for acc in conf_acc])

    accuracy = sum(conf_acc) / len(conf_acc)
    print(f"TEST CELEBA ATTR accuracy : ({accuracy*100:.2f}%)")
    return accuracy, 70


if __name__ == "__main__":
    import torch
    import sys
    print(f"len(sys.argv)={len(sys.argv)}")
    if len(sys.argv) == 1:
        checkpoint = 'pretrained/BEST_checkpoint_r101.tar'
    else:
        checkpoint = sys.argv[1]

    print(f"Loading checkpoint {checkpoint}")
    checkpoint_name = checkpoint[:-4].replace('/', '_')
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model'].module
    fairface_mlp = checkpoint['attr_fc'].module
    from config import device
    model = model.to(device)
    model.eval()

    acc, threshold = celeba_attr_test(model, fairface_mlp)
    print("Done")


