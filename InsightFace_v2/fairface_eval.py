from collections import defaultdict

import torchvision
from PIL import Image
from tqdm import tqdm

from data_gen import FairfaceImageDataset
from fairface_dataset import FairfaceDataset


def fairface_test(feature_extractor, tail_fc):
    feature_extractor.eval()
    tail_fc.eval()
    test_dataset = FairfaceImageDataset(split='val')
    num_correct = 0
    num_images = 0
    per_class_num_correct = defaultdict(int)
    per_class_num_images = defaultdict(int)
    try:
        for im, attr_vector in tqdm(test_dataset):
            im = im.unsqueeze(0)
            feature_vector = feature_extractor(im.cuda())
            race_vector = tail_fc(feature_vector)
            selected_race = race_vector[0].argmax()
            correct_race = attr_vector.argmax()
            is_correct = selected_race == correct_race
            if is_correct:
                num_correct += 1
                per_class_num_correct[correct_race] += 1
            num_images += 1
            per_class_num_images[correct_race] += 1
    except Exception as e:
        pass

    accuracy = num_correct / num_images
    print(f"TEST FAIRFACE accuracy : {num_correct} / {num_images} ({accuracy*100:.2f}%)")
    for class_idx, num_images in per_class_num_images.items():
        num_correct = per_class_num_correct[class_idx]
        accuracy = num_correct / num_images
        print(f"TEST FAIRFACE accuracy class {class_idx}: {num_correct} / {num_images} ({accuracy * 100:.2f}%)")
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
    fairface_mlp = checkpoint['race_fc'].module
    from config import device
    model = model.to(device)
    model.eval()

    acc, threshold = fairface_test(model, fairface_mlp)
    print("Done")


