import math
import os
from functools import partial
from shutil import copyfile

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from celeba_attr_eval import celeba_attr_test
from celeba_eval import celeba_test
from config import device, grad_clip, print_freq
from data_gen import ArcFaceDataset, AdverserialFaceDataset, FairfaceImageDataset, CelebAAttributesDataset
from focal_loss import FocalLoss
from lfw_eval import lfw_test
from models import resnet18, resnet34, resnet50, resnet101, resnet152, ArcMarginModel
from optimizer import InsightFaceOptimizer
from utils import parse_args, save_checkpoint, AverageMeter, accuracy, get_logger
from fairface_eval import fairface_test

def full_log(epoch):
    full_log_dir = 'data/full_log'
    if not os.path.isdir(full_log_dir):
        os.mkdir(full_log_dir)
    filename = 'angles_{}.txt'.format(epoch)
    dst_file = os.path.join(full_log_dir, filename)
    src_file = 'data/angles.txt'
    copyfile(src_file, dst_file)


def build_simple_mlp(input_dim, hidden_dims, output_dim):
    dims = [input_dim] + hidden_dims + [output_dim]
    layers = []
    for i in range(1, len(dims)):
        layers.append(torch.nn.Linear(dims[i-1], dims[i]))
        if i < len(dims)-1:
            layers.append(torch.nn.BatchNorm1d(num_features=dims[i]))
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_acc = float('-inf')
    writer = SummaryWriter()
    epochs_since_improvement = 0
    is_adverserial = args.adverserial
    adverserial_weight = args.adverserial_weight
    adverserial_test_weight = args.adverserial_test_weight
    is_fairface = args.fairface
    fairface_weight = args.fairface_weight
    fairface_mlp = None
    is_attr = args.attr
    attr_weight = args.attr_weight
    attr_mlp = None
    #if is_adverserial:
        # https://github.com/pytorch/pytorch/issues/40403
    #    torch.multiprocessing.set_start_method('spawn')  # good solution !!!!

    # Initialize / load checkpoint
    if checkpoint is None:
        if args.network == 'r18':
            model = resnet18(args)
        elif args.network == 'r34':
            model = resnet34(args)
        elif args.network == 'r50':
            model = resnet50(args)
        elif args.network == 'r101':
            model = resnet101(args)
        elif args.network == 'r152':
            model = resnet152(args)
        else:
            raise TypeError('network {} is not supported.'.format(args.network))

        # print(model)
        model = nn.DataParallel(model)
        metric_fc = ArcMarginModel(args)
        metric_fc = nn.DataParallel(metric_fc)
        fairface_mlp = None

        if args.optimizer == 'sgd':
            optimizer = InsightFaceOptimizer(
                torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                lr=args.lr, momentum=args.mom, weight_decay=args.weight_decay))
        else:
            optimizer = InsightFaceOptimizer(
                torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                 lr=args.lr, weight_decay=args.weight_decay))

        if is_fairface:
            fairface_mlp = build_simple_mlp(512, [256], 7)
            fairface_mlp = fairface_mlp.to(device)
            fairface_mlp = nn.DataParallel(fairface_mlp)

        if is_attr:
            attr_mlp = build_simple_mlp(512, [1024], 40)
            attr_mlp = attr_mlp.to(device)
            attr_mlp = nn.DataParallel(attr_mlp)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        try:
            model = model.module
        except:
            pass
        model = nn.DataParallel(model)
        metric_fc = checkpoint['metric_fc']
        try:
            metric_fc = metric_fc.module
        except:
            pass
        metric_fc = nn.DataParallel(metric_fc)
        optimizer = checkpoint['optimizer']

        if is_fairface:
            try:
                fairface_mlp = checkpoint['race_fc']
                fairface_mlp = fairface_mlp.module
            except:
                pass
            if not fairface_mlp:
                fairface_mlp = build_simple_mlp(512, [256], 7)
            fairface_mlp = fairface_mlp.to(device)
            fairface_mlp = nn.DataParallel(fairface_mlp)

        if is_attr:
            try:
                attr_mlp = checkpoint['race_fc']
                attr_mlp = attr_mlp.module
            except:
                pass
            if not attr_mlp:
                attr_mlp = build_simple_mlp(512, [1024], 40)
            attr_mlp = attr_mlp.to(device)
            attr_mlp = nn.DataParallel(attr_mlp)

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)
    metric_fc = metric_fc.to(device)

    # Loss function
    if args.focal_loss:
        criterion = FocalLoss(gamma=args.gamma).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    num_workers = 0 if args.debug else 4
    train_dataset = ArcFaceDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)

    test_func = partial(celeba_test, normal_weight=1.0, adverserial_weight=adverserial_test_weight) if is_adverserial else lfw_test
    if is_adverserial:
        adv_dataset = AdverserialFaceDataset('train')
        # Adverserial loader requires CUDA on its own, therefore need 'spawn' multiprocess mode
        adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=args.batch_size//2, shuffle=True,
                                                 num_workers=num_workers,
                                                 multiprocessing_context=None if num_workers == 0 else 'spawn')
        adv_criterion = nn.BCEWithLogitsLoss().to(device)
        if start_epoch > 0:
            logger.info('Model already exists, calculating first threshold for adverserial training\n')
            start_acc, threshold = (0.5, 75) if args.debug else test_func(model)
            logger.info(f'Starting accuracy={start_acc}, threshold={threshold}\n')
    max_train_rounds = 10000 if is_adverserial else -1

    if is_fairface:
        fairface_dataset = FairfaceImageDataset('train')
        fairface_loader = torch.utils.data.DataLoader(fairface_dataset, batch_size=args.batch_size,
                                                      shuffle=True,
                                                      num_workers=num_workers,
                                                      multiprocessing_context=None if num_workers == 0 else 'spawn')
        test_func = partial(fairface_test, tail_fc=fairface_mlp)

    if is_attr:
        attr_dataset = CelebAAttributesDataset('train')
        attr_loader = torch.utils.data.DataLoader(attr_dataset, batch_size=args.batch_size,
                                                      shuffle=True,
                                                      num_workers=num_workers,
                                                      multiprocessing_context=None if num_workers == 0 else 'spawn')
        test_func = partial(celeba_attr_test, tail_fc=attr_mlp)

    if args.debug:
        max_train_rounds = 10
        print_freq = 2

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # One epoch's training
        post_batch_generators = []
        if is_adverserial and epoch > 0:
            post_batch_generator = create_train_adv_generator(train_loader=adv_loader,
                                                              model=model,
                                                              threshold=threshold,
                                                              criterion=adv_criterion,
                                                              optimizer=optimizer,
                                                              epoch=epoch,
                                                              logger=logger,
                                                              loss_multiplier=adverserial_weight)
            post_batch_generators.append(post_batch_generator)
        if is_fairface:
            post_batch_generator = create_train_fairface_generator(train_loader=fairface_loader,
                                                                   model=model,
                                                                   optimizer=optimizer,
                                                                   epoch=epoch,
                                                                   logger=logger,
                                                                   loss_multiplier=fairface_weight,
                                                                   decision_mlp=fairface_mlp)
            post_batch_generators.append(post_batch_generator)
        if is_attr:
            post_batch_generator = create_train_attributes_generator(train_loader=attr_loader,
                                                                     model=model,
                                                                     optimizer=optimizer,
                                                                     epoch=epoch,
                                                                     logger=logger,
                                                                     loss_multiplier=attr_weight,
                                                                     decision_mlp=attr_mlp)
            post_batch_generators.append(post_batch_generator)


        # Standard epoch
        train_loss, train_acc = train(train_loader=train_loader,
                                      model=model,
                                      metric_fc=metric_fc,
                                      criterion=criterion,
                                      optimizer=optimizer,
                                      epoch=epoch,
                                      logger=logger,
                                      max_rounds=max_train_rounds,
                                      post_batch_generators=post_batch_generators)

        writer.add_scalar('model/train_loss', train_loss, epoch)
        writer.add_scalar('model/train_acc', train_acc, epoch)

        logger.info('Learning rate={}, step number={}\n'.format(optimizer.lr, optimizer.step_num))

        # One epoch's validation

        lfw_acc, threshold = test_func(model)

        # lfw_acc, threshold = 0, 75

        writer.add_scalar('model/valid_acc', lfw_acc, epoch)
        writer.add_scalar('model/valid_thres', threshold, epoch)

        # Check if there was an improvement
        is_best = lfw_acc > best_acc
        best_acc = max(lfw_acc, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # if is_adverserial:
        #     # Adverserial epoch
        #     train_loss, train_acc =
        #
        #     writer.add_scalar('model/adv_train_loss', train_loss, epoch)
        #     writer.add_scalar('model/adv_train_acc', train_acc, epoch)
        #
        #     logger.info('Learning rate={}, step number={}\n'.format(optimizer.lr, optimizer.step_num))
        #
        #     # One epoch's validation
        #
        #     writer.add_scalar('model/adv_valid_acc', lfw_acc, epoch)
        #     writer.add_scalar('model/adv_valid_thres', threshold, epoch)
        #
        #     # Check if there was an improvement
        #     is_best = lfw_acc > best_acc
        #     best_acc = max(lfw_acc, best_acc)
        #     if not is_best:
        #         epochs_since_improvement += 1
        #         print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        #     else:
        #         epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, metric_fc, fairface_mlp, attr_mlp, optimizer, best_acc, is_best, args.checkpoint_suffix)


def train(train_loader, model, metric_fc, criterion, optimizer, epoch, logger, max_rounds=-1, post_batch_generators=[]):
    model.train()  # train mode (dropout and batchnorm is used)
    metric_fc.train()

    losses = AverageMeter()
    top5_accs = AverageMeter()

    # Batches
    for i, (img, label) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.to(device)

        #import torchvision
        #torchvision.utils.save_image(img.cpu(), 'debug_insightface.png')
        label = label.to(device)  # [N, 1]

        # Forward prop.
        feature = model(img)  # embedding => [N, 512]
        output = metric_fc(feature, label)  # class_id_out => [N, 93431]

        # Calculate loss
        loss = criterion(output, label)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        optimizer.clip_gradient(grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        top5_accuracy = accuracy(output, label, 5)
        top5_accs.update(top5_accuracy)

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Top5 Accuracy {top5_accs.val:.3f} ({top5_accs.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                         loss=losses,
                                                                                         top5_accs=top5_accs))
        # NOAM: Trying to overfit the adverserial training, make this part shorter
        if 0 < max_rounds <= i:
            break

        if post_batch_generators is not None:
            should_stop = False
            for post_batch_generator in post_batch_generators:
                try:
                    next(post_batch_generator)
                except StopIteration:
                    # Match the number of iterations between the two trainer types
                    should_stop = True
                    break
            if should_stop:
                break

    return losses.avg, top5_accs.avg


def create_train_adv_generator(train_loader, model, threshold, criterion, optimizer, epoch, logger, loss_multiplier):
    #model.train()  # train mode (dropout and batchnorm is used)
    #metric_fc.train()
    yield
    losses = AverageMeter()
    top5_accs = AverageMeter()

    # Batches
    for i, (img1, img2, is_different_label) in enumerate(train_loader):
        # Move to GPU, if available
        bs = img1.shape[0]
        img1 = img1.to(device)
        img2 = img2.to(device)

        #os.makedirs('images/training_batches', exist_ok=True)
        #save_image(torch.cat([torch.stack((i1, i2), dim=0) for i1, i2 in zip(img1, img2)]),
        #           'images/training_batches/adv_batch.jpg')
        #open('images/training_batches/label.txt', 'w').write(str(is_different_label.numpy().reshape(-1, 4)))

        is_different_label = is_different_label.to(device).type_as(img1)  # [N, 1]

        #img1 = img1[:1,:,:,:]
        imgs_concatted = torch.cat((img1, img2), 0)
        features_concat = model(imgs_concatted)

        f1 = features_concat[:bs]
        f2 = features_concat[bs:]
        assert f1.shape == f2.shape
        # Forward prop.
        #f1 = model(img1)  # embedding => [N, 512]
        #f2 = model(img1)  # embedding => [N, 512]

        # https://github.com/pytorch/pytorch/issues/18027 No batch dot product
        dot_product = (f1 * f2).sum(-1)
        normalized = (f1.norm(dim=1) * f2.norm(dim=1) + 1e-5)
        cosdistance = dot_product / normalized
        angle = torch.acos(cosdistance) * 180 / math.pi
        # Change from -1 (opposite) -> 1 (same) range to 0 (same) - 1 (different)
        #features_diff = (torch.ones_like(cosdistance) - cosdistance) / 2
        #features_diff = features_diff.unsqueeze(1)


        #output = metric_fc(feature, label)  # class_id_out => [N, 93431]
        #threshold_decision = features_diff.sum(dim=1) - threshold
        threshold_decision = angle - threshold

        # Calculate loss
        loss = criterion(threshold_decision, is_different_label) * loss_multiplier

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        optimizer.clip_gradient(grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        binary_accuracy = ((threshold_decision > 0).type(torch.int) == is_different_label).sum().item() / bs
        #top5_accuracy = accuracy(threshold_decision, label, 5)
        top5_accs.update(binary_accuracy)

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Adv Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Adv Accuracy {top5_accs.val:.3f} ({top5_accs.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                         loss=losses,
                                                                                         top5_accs=top5_accs))
        yield

    return losses.avg, top5_accs.avg


def create_train_fairface_generator(train_loader, model, optimizer, epoch, logger, loss_multiplier, decision_mlp):
    #model.train()  # train mode (dropout and batchnorm is used)
    decision_mlp.train()
    yield
    losses = AverageMeter()
    top5_accs = AverageMeter()

    # Batches
    for i, (img, race_vector) in enumerate(train_loader):
        bs = img.shape[0]
        # Move to GPU, if available
        #import torchvision
        #torchvision.utils.save_image(img.cpu(), 'debug_fairface.png')
        #raise Exception("DONE DEBUG")
        img = img.to(device)

        #os.makedirs('images/training_batches', exist_ok=True)
        #save_image(torch.cat([torch.stack((i1, i2), dim=0) for i1, i2 in zip(img1, img2)]),
        #           'images/training_batches/adv_batch.jpg')
        #open('images/training_batches/label.txt', 'w').write(str(is_different_label.numpy().reshape(-1, 4)))

        race_vector = race_vector.to(device).type_as(img)  # [N, NUM_RACES]

        #img1 = img1[:1,:,:,:]

        features = model(img)
        decision = decision_mlp(features)

        y_hat = decision
        y = race_vector
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y) * loss_multiplier
        num_correct = (y_hat.argmax(dim=1) == y.argmax(dim=1)).to(float).sum().item()

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        optimizer.clip_gradient(grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        binary_accuracy = num_correct / bs
        #top5_accuracy = accuracy(threshold_decision, label, 5)
        top5_accs.update(binary_accuracy)

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Fairface Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Fairface Accuracy {top5_accs.val:.3f} ({top5_accs.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                         loss=losses,
                                                                                         top5_accs=top5_accs))
        yield

    return losses.avg, top5_accs.avg


def create_train_attributes_generator(train_loader, model, optimizer, epoch, logger, loss_multiplier, decision_mlp):
    #model.train()  # train mode (dropout and batchnorm is used)
    decision_mlp.train()
    yield
    losses = AverageMeter()
    top5_accs = AverageMeter()

    # Batches
    for i, (img, attr_vector, attr_weights) in enumerate(train_loader):
        bs = img.shape[0]
        # Move to GPU, if available
        #import torchvision
        #torchvision.utils.save_image(img.cpu(), 'debug_fairface.png')
        #raise Exception("DONE DEBUG")
        img = img.to(device)

        #os.makedirs('images/training_batches', exist_ok=True)
        #save_image(torch.cat([torch.stack((i1, i2), dim=0) for i1, i2 in zip(img1, img2)]),
        #           'images/training_batches/adv_batch.jpg')
        #open('images/training_batches/label.txt', 'w').write(str(is_different_label.numpy().reshape(-1, 4)))

        attr_vector = attr_vector.to(device).type_as(img)  # [N, NUM_RACES]
        attr_weights = attr_weights.to(device).type_as(img)  # [N, NUM_RACES]

        #img1 = img1[:1,:,:,:]

        features = model(img)
        decision = decision_mlp(features)
        decision = torch.nn.functional.sigmoid(decision)

        y_hat = decision
        y = attr_vector
        loss = torch.nn.functional.mse_loss(y_hat, y, reduce=False) * attr_weights
        loss = loss.mean() * loss_multiplier
        num_correct = (y_hat.round() == y.round()).to(float).sum().item()

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        optimizer.clip_gradient(grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        binary_accuracy = num_correct / (bs * 40)
        #top5_accuracy = accuracy(threshold_decision, label, 5)
        top5_accs.update(binary_accuracy)

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'CelebA Attr Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'CelebA Attr Accuracy {top5_accs.val:.3f} ({top5_accs.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                         loss=losses,
                                                                                         top5_accs=top5_accs))
        yield

    return losses.avg, top5_accs.avg

def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
