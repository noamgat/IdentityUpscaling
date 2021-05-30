import math
import os
import pickle
import tarfile
import time

import cv2 as cv
import numpy as np
import scipy.stats
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.datasets import CelebA
from torchvision.utils import save_image
from tqdm import tqdm

from celeba_aligned_copy import build_aligned_celeba, CelebAPairsDataset, CelebAAdverserialDataset
from config import device
from data_gen import data_transforms, AdverserialFaceDataset
from utils import align_face, get_central_face_attributes, get_all_face_attributes, draw_bboxes
import sys

celeba_split = 'all'  # train / valid / test / all

normal_angles_file = f'data/celeba_angles_{celeba_split}.txt'
normal_lfw_pickle = f'data/celeba_funneled_{celeba_split}.pkl'
normal_test_pair_file = f'data/celeba_test_pair_{celeba_split}.txt'
checkpoint_name = 'checkpoint'

N = 6000
celeba_raw = build_aligned_celeba('../CelebA_Raw', '../CelebA_large')
celeba_orig = CelebA(root='../CelebA_Raw', split=celeba_split, download=False, target_type='identity')
celeba = celeba_orig

generated_suffix = 'metaepoch2'  # can also be 'generated', 'withidentity', 'metaepoch2'
if __name__ == '__main__':
    if len(sys.argv) > 2:
        generated_suffix = sys.argv[2]
        print(f"Using generated_suffix={generated_suffix}")

generated = build_aligned_celeba('../CelebA_Raw', f'../CelebA_{generated_suffix}', new_image_suffix='_0', split=celeba_split)
large_matching_generated = build_aligned_celeba('../CelebA_Raw', '../CelebA_large', custom_indices=generated.filtered_indices, split=celeba_split)
adverserial_dataset_1 = CelebAAdverserialDataset(generated, large_matching_generated, return_indices=True)

adverserial_pickle = f'data/celeba_adverserial_funneled_{celeba_split}_{generated_suffix}.pkl'
adverserial_test_pair_file = f'data/celeba_adverserial_test_pair_{celeba_split}_{generated_suffix}.txt'
adverserial_angles_file = f'data/celeba_adverserial_angles_{celeba_split}_{generated_suffix}.txt'

angles_file = normal_angles_file
lfw_pickle = normal_lfw_pickle
test_pair_file = normal_test_pair_file
is_adverserial = False

def set_is_adverserial(is_adverserial_flag):
    global angles_file
    global lfw_pickle
    global test_pair_file
    global is_adverserial
    global celeba
    is_adverserial = is_adverserial_flag
    angles_file = adverserial_angles_file if is_adverserial else normal_angles_file
    lfw_pickle = adverserial_pickle if is_adverserial else normal_lfw_pickle
    test_pair_file = adverserial_test_pair_file if is_adverserial else normal_test_pair_file
    celeba = generated if is_adverserial else celeba_orig


def extract(filename):
    with tarfile.open(filename, 'r') as tar:
        tar.extractall('data')


celeba_dataset = CelebAPairsDataset(celeba, num_samples=N, return_indices=True)

def process(data_source, data_set, output_test_pair_file, output_pickle_file):
    lines = []
    data_sources = set()
    for i in tqdm(range(N)):
        (data_source_1, idx1), (data_source_2, idx2), is_different = data_set[i]
        data_sources.add(data_source_1)
        data_sources.add(data_source_2)
        file1 = data_source_1.filename[idx1]
        file2 = data_source_2.filename[idx2]
        is_same = 1 - is_different
        is_same_check = int(data_source_1.identity[idx1] == data_source_2.identity[idx2] and data_source_1 == data_source_2)
        assert is_same == is_same_check, "Bad is_same flag"
        line = f"{file1} {file2} {is_same}"
        lines.append(line)
    with open(output_test_pair_file, 'w') as file:
        file.write("\r\n".join(lines))

    file_names = []

    for data_source in data_sources:
        subjects = list(range(1, data_source.identity.max().item() + 1))
        # assert (len(subjects) == 10177), "Number of subjects is: {}!".format(len(subjects))
        images = np.array(range(len(data_source.filename)))
        for i in tqdm(range(len(subjects))):
            sub = subjects[i]
            indices = images[(data_source.identity == sub).squeeze()]
            folder = os.path.join(data_source.root, data_source.base_folder, 'img_align_celeba')
            files = np.array(data_source.filename)[indices]
            for file in files:
                filename = os.path.join(folder, file)
                file_names.append({'filename': filename, 'class_id': i, 'subject': sub})

    #197016 / 202599
    #assert (len(file_names) == 202599), "Number of files is: {}!".format(len(file_names))



    samples = []
    for item in tqdm(file_names):
        filename = item['filename']
        class_id = item['class_id']
        sub = item['subject']

        try:
            bboxes, landmarks = get_central_face_attributes(filename)

            samples.append(
                {'class_id': class_id, 'subject': sub, 'full_path': filename, 'bounding_boxes': bboxes,
                 'landmarks': landmarks})
        except KeyboardInterrupt:
            raise
        except Exception as err:
            print(err)

    with open(output_pickle_file, 'wb') as file:
        save = {
            'samples': samples
        }
        pickle.dump(save, file, pickle.HIGHEST_PROTOCOL)




def get_image(samples, transformer, file):
    filtered = [sample for sample in samples if file in sample['full_path'].replace('\\', '/')]
    if len(filtered) != 1:
        print(f"Image {file} can't be found (filtered = {len(filtered)})")
        return None
    assert (len(filtered) == 1), 'len(filtered): {} file:{}'.format(len(filtered), file)
    sample = filtered[0]
    full_path = sample['full_path']
    landmarks = sample['landmarks']
    num_landmarks = np.prod(np.array(landmarks).shape)
    if num_landmarks != 10:
        print(f"Image {file} has no landmarks - number of elements = {num_landmarks}")
        return None
    img = align_face(full_path, landmarks)  # BGR
    # img = blur_and_grayscale(img)
    img = img[..., ::-1]  # RGB
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    img = img.to(device)
    return img


def evaluate(model):
    model.eval()

    with open(lfw_pickle, 'rb') as file:
        data = pickle.load(file)

    samples = data['samples']

    filename = test_pair_file
    with open(filename, 'r') as file:
        lines = file.readlines()

    transformer = data_transforms['val']

    angles = []

    start = time.time()
    with torch.no_grad():
        for line in tqdm(lines):
            tokens = line.split()
            file0 = tokens[0]
            img0 = get_image(samples, transformer, file0)

            file1 = tokens[1]
            img1 = get_image(samples, transformer, file1)
            if img0 is None or img1 is None:
                continue
            #os.makedirs('images/celebanetinput', exist_ok=True)
            #save_image(img0, 'images/celebanetinput/img0.jpg')
            #save_image(img1, 'images/celebanetinput/img1.jpg')
            #raise Exception("MANUAL STOP")
            imgs = torch.zeros([2, 3, 112, 112], dtype=torch.float, device=device)
            imgs[0] = img0
            imgs[1] = img1

            output = model(imgs)

            feature0 = output[0].cpu().numpy()
            feature1 = output[1].cpu().numpy()
            x0 = feature0 / np.linalg.norm(feature0)
            x1 = feature1 / np.linalg.norm(feature1)
            cosine = np.dot(x0, x1)
            cosine = np.clip(cosine, -1.0, 1.0)
            theta = math.acos(cosine)
            theta = theta * 180 / math.pi
            is_same = tokens[2]
            angles.append('{} {}\n'.format(theta, is_same))

    elapsed_time = time.time() - start
    print('elapsed time(sec) per image: {}'.format(elapsed_time / (6000 * 2)))

    with open(angles_file, 'w') as file:
        file.writelines(angles)


def visualize(threshold):
    with open(angles_file) as file:
        lines = file.readlines()

    ones = []
    zeros = []

    for line in lines:
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        if type == 1:
            ones.append(angle)
        else:
            zeros.append(angle)

    bins = np.linspace(0, 180, 181)

    if is_adverserial:
        plt.hist(zeros, bins, density=True, alpha=0.5, label='2', facecolor='yellow')
    else:
        plt.hist(zeros, bins, density=True, alpha=0.5, label='0', facecolor='red')
        plt.hist(ones, bins, density=True, alpha=0.5, label='1', facecolor='blue')

    mu_0 = np.mean(zeros)
    sigma_0 = np.std(zeros)
    y_0 = scipy.stats.norm.pdf(bins, mu_0, sigma_0)
    plt.plot(bins, y_0, 'r--')
    mu_1 = np.mean(ones)
    sigma_1 = np.std(ones)
    y_1 = scipy.stats.norm.pdf(bins, mu_1, sigma_1)
    plt.plot(bins, y_1, 'b--')
    plt.xlabel('theta')
    plt.ylabel('theta j Distribution')
    plt.title(
        '{}\nHistogram : mu_0={:.4f},sigma_0={:.4f}, mu_1={:.4f},sigma_1={:.4f}'.
            format(checkpoint_name, mu_0, sigma_0, mu_1, sigma_1))

    print('threshold: ' + str(threshold))
    print('mu_0: ' + str(mu_0))
    print('sigma_0: ' + str(sigma_0))
    print('mu_1: ' + str(mu_1))
    print('sigma_1: ' + str(sigma_1))

    plt.legend(loc='upper right')
    plt.plot([threshold, threshold], [0, 0.05], 'k-', lw=2)
    plt.savefig(f'images/theta_dist/celeba_theta_dist_{checkpoint_name}_adv{is_adverserial}.png')
    if is_adverserial:
        plt.close()
    # plt.show()


def accuracy(threshold):
    with open(angles_file) as file:
        lines = file.readlines()

    wrong = 0
    for line in lines:
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        if type == 1:
            if angle > threshold:
                wrong += 1
        else:
            if angle <= threshold:
                wrong += 1

    accuracy = 1 - wrong / 6000
    return accuracy


def adverserial_accuracy(threshold):
    ds1 = generated
    ds2 = large_matching_generated
    for i in range(N):
        filename1 = ds1[i].filename
        filename2 = ds2[i].filename


def show_bboxes(folder):
    with open(lfw_pickle, 'rb') as file:
        data = pickle.load(file)

    samples = data['samples']
    for sample in tqdm(samples):
        full_path = sample['full_path']
        bounding_boxes = sample['bounding_boxes']
        landmarks = sample['landmarks']
        img = cv.imread(full_path)
        img = draw_bboxes(img, bounding_boxes, landmarks)
        filename = os.path.basename(full_path)
        filename = os.path.join(folder, filename)
        cv.imwrite(filename, img)


def error_analysis(threshold):
    with open(angles_file) as file:
        angle_lines = file.readlines()

    fp = []
    fn = []
    for i, line in enumerate(angle_lines):
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        if angle <= threshold and type == 0:
            fp.append(i)
        if angle > threshold and type == 1:
            fn.append(i)

    print('len(fp): ' + str(len(fp)))
    print('len(fn): ' + str(len(fn)))

    num_fp = len(fp)
    num_fn = len(fn)

    with open(lfw_pickle, 'rb') as file:
        data = pickle.load(file)
    samples = data['samples']

    filename = test_pair_file
    with open(filename, 'r') as file:
        pair_lines = file.readlines()

    for i in tqdm(range(num_fp)):
        fp_id = fp[i]
        fp_line = pair_lines[fp_id]
        tokens = fp_line.split()
        file0 = tokens[0]
        copy_file(samples, file0, '{}_celeba_fp_0.jpg'.format(i))
        save_aligned(samples, file0, '{}_celeba_fp_0_aligned.jpg'.format(i))
        file1 = tokens[1]
        copy_file(samples, file1, '{}_celeba_fp_1.jpg'.format(i))
        save_aligned(samples, file1, '{}_celeba_fp_1_aligned.jpg'.format(i))

    for i in tqdm(range(num_fn)):
        fn_id = fn[i]
        fn_line = pair_lines[fn_id]
        tokens = fn_line.split()
        file0 = tokens[0]
        copy_file(samples, file0, '{}_celeba_fn_0.jpg'.format(i))
        save_aligned(samples, file0, '{}_celeba_fn_0_aligned.jpg'.format(i))
        file1 = tokens[1]
        copy_file(samples, file1, '{}_celeba_fn_1.jpg'.format(i))
        save_aligned(samples, file1, '{}_celeba_fn_1_aligned.jpg'.format(i))


def save_aligned(samples, old_fn, new_fn):
    # folder = os.path.join(celeba.root, celeba.base_folder, 'img_align_celeba')
    # old_fn = os.path.join(folder, old)

    filtered = [sample for sample in samples if old_fn in sample['full_path'].replace('\\', '/')]
    assert (len(filtered) == 1), 'len(filtered): {} file:{}'.format(len(filtered), old_fn)
    sample = filtered[0]
    old_fn = sample['full_path']

    _, landmarks = get_central_face_attributes(old_fn)
    img = align_face(old_fn, landmarks)
    new_fn = os.path.join('images', new_fn)
    cv.imwrite(new_fn, img)


def copy_file(samples, old, new):
    #folder = os.path.join(celeba.root, celeba.base_folder, 'img_align_celeba')
    #old_fn = os.path.join(folder, old)

    filtered = [sample for sample in samples if old in sample['full_path'].replace('\\', '/')]
    assert (len(filtered) == 1), 'len(filtered): {} file:{}'.format(len(filtered), old)
    sample = filtered[0]
    old_fn = sample['full_path']

    img = cv.imread(old_fn)
    bboxes, landmarks = get_all_face_attributes(old_fn)
    draw_bboxes(img, bboxes, landmarks)
    cv.resize(img, (224, 224))
    new_fn = os.path.join('images', new)
    cv.imwrite(new_fn, img)


def get_threshold(normal_mistake_weight=1.0, adverserial_mistake_weight=1.0):

    data = []
    current_adverserial_flag = is_adverserial
    for adverserial_flag in [False, True]:
        mistake_weight = adverserial_mistake_weight if adverserial_flag else normal_mistake_weight
        set_is_adverserial(adverserial_flag)

        print(f"Reading {angles_file} for angles")
        with open(angles_file, 'r') as file:
            lines = file.readlines()

        for line in lines:
            tokens = line.split()
            angle = float(tokens[0])
            type = int(tokens[1])
            data.append({'angle': angle, 'type': type, 'weight': mistake_weight})

    set_is_adverserial(current_adverserial_flag)
    min_error = 12000
    min_threshold = 0

    for d in data:
        threshold = d['angle']
        type1 = sum([s['weight'] for s in data if s['angle'] <= threshold and s['type'] == 0])
        type2 = sum([s['weight'] for s in data if s['angle'] > threshold and s['type'] == 1])
        num_errors = type1 + type2
        if num_errors < min_error:
            min_error = num_errors
            min_threshold = threshold

    # print(min_error, min_threshold)
    return min_threshold


def celeba_test(model, normal_weight=1.0, adverserial_weight=1.0):
    #filename = 'data/lfw-funneled.tgz'
    #if not os.path.isdir('data/lfw_funneled'):
    #    print('Extracting {}...'.format(filename))
    #    extract(filename)

    if not os.path.isfile(lfw_pickle):
        print('Processing {}...'.format(lfw_pickle))
        process(celeba, celeba_dataset, test_pair_file, lfw_pickle)

    if not os.path.isfile(adverserial_pickle):
        print('Processing {}...'.format(adverserial_pickle))
        process(large_matching_generated, adverserial_dataset_1, adverserial_test_pair_file, adverserial_pickle)

    #raise Exception("Early exit")

    recalc_every_time = True
    if recalc_every_time or not os.path.isfile(angles_file):
        print('Evaluating {}...'.format(angles_file))
        evaluate(model)

    set_is_adverserial(True)
    if recalc_every_time or not os.path.isfile(angles_file):
        print('Evaluating {}...'.format(angles_file))
        evaluate(model)

    set_is_adverserial(False)

    # raise Exception("Early exit")

    print('Calculating threshold...')
    # threshold = 70.36
    thres = get_threshold(normal_weight, adverserial_weight)
    print('Calculating accuracy...')
    acc1 = accuracy(thres)
    print('Accuracy: {}%, threshold: {}'.format(acc1 * 100, thres))

    set_is_adverserial(True)
    print('Calculating Adverserial accuracy...')
    acc2 = accuracy(thres)
    print('Adverserial Accuracy: {}%, threshold: {}'.format(acc2 * 100, thres))

    acc = (acc1 * normal_weight + acc2 * adverserial_weight) / (normal_weight + adverserial_weight)
    return acc, thres


def paper_acc_test():
    global normal_angles_file
    global adverserial_angles_file

    models = ['pretrained_BEST_checkpoint_101_20200831_', 'pretrained_BEST_checkpoint_101_adv_20200908_']
    datasets = ['generated', 'metaepoch2', 'withidentity']
    for model in models:
        for dataset in datasets:
            print("----------------------------------")
            print(f"MODEL: {model} DATASET: {dataset}")
            normal_angles_file = f'eval_results/{model}{dataset}/celeba_angles_all.txt'
            adverserial_angles_file = f'eval_results/{model}{dataset}/celeba_adverserial_angles_all_{dataset}.txt'
            #print(normal_angles_file)
            thres = get_threshold(1.0, 0.1)
            print('Calculating accuracy...')
            for adverserial in [False, True]:
                set_is_adverserial(adverserial)
                acc1 = accuracy(thres)
                print('Adv:{}, Accuracy: {}%, threshold: {}'.format(adverserial, acc1 * 100, thres))

if __name__ == "__main__":
    paper_acc_test()
    raise Exception("Done")
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
    model = model.to(device)
    model.eval()

    acc, threshold = celeba_test(model)

    print('Visualizing {}...'.format(angles_file))
    set_is_adverserial(False)
    #visualize(threshold)
    set_is_adverserial(True)
    #visualize(threshold)

    print('error analysis...')
    # error_analysis(threshold)

    # Copy the generated files to another place for further analysis
    output_dir = f'eval_results/{checkpoint_name}_{generated_suffix}'
    os.makedirs(output_dir, exist_ok=True)
    import shutil
    for fn in [normal_angles_file, normal_lfw_pickle, normal_test_pair_file,
               adverserial_pickle, adverserial_test_pair_file, adverserial_angles_file]:
        shutil.copy(fn, output_dir)




