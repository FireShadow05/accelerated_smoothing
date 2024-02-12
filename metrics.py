import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import get_dataset, DATASETS
from core import Smooth
from utils import get_test_dataloader, CustomDataset, JS_divergence, Testing, confidence_bound
from distribution import StandardGaussian
import numpy as np
from architectures import get_architecture
import json
import sys

softmax = nn.Softmax()
original_stdout = sys.stdout
parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("surrogate", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("data_path", type=str, help="path to the folder containing the sampled data")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
args = parser.parse_args()

if __name__ == '__main__':
    sigma = args.sigma
    label_file = os.path.join(args.data_path, f'label_{args.split}.pth')
    smooth_outs_file = os.path.join(args.data_path, f'smooth_out_{args.split}_{args.N}.pth')
    x_file = os.path.join(args.data_path, f'x_{args.split}.pth')
    custom_dataset = CustomDataset(label_file, smooth_outs_file, x_file)
    batch_size = args.batch
    dataloader = DataLoader(custom_dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    softmax = nn.Softmax()

    base_classifier = args.base_classifier
    checkpoint = torch.load(base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], 'cifar10')
    base_classifier.load_state_dict(checkpoint['state_dict'])

    smooth_classifier = Smooth(base_classifier, 10, sigma)

    surrogate = args.surrogate
    model_dir = surrogate.split('checkpoint.pth.tar')[0]
    checkpoint = torch.load(surrogate)
    surrogate = get_architecture(checkpoint["arch"], 'cifar10')
    surrogate.load_state_dict(checkpoint['state_dict'])

    N = 100000
    surrogate.eval()
    dist = StandardGaussian(d=3072, scale=sigma)

    acc_100 = 0
    acc_100000 = 0
    MSE = 0
    y_true = []
    y_predicted = []
    radius_act = []
    radius_pred = []
    gt = []
    pd = []
    baseline = []
    for i, (labels, smooth_outs, images) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        smooth_outs = smooth_outs.to(device)
        base_predicted = smooth_classifier.predict(images, 100, 0.001, 100)
        cahat, base_radius = smooth_classifier.certify(images, 100, 100, 0.001, 100)
        if cahat == labels:
            baseline.append(base_radius)

        output = N * softmax(surrogate(images))
        nA = output.detach().cpu().floor().squeeze()
        nA_1 = smooth_outs.detach().cpu().floor().squeeze()
        actual_predicted = torch.argmax(smooth_outs)
        _, predicted = torch.max(output.data, 1)
        realN = nA.sum()
        realN_1 = nA_1.sum()

        p1low_1, p1high_1 = confidence_bound(nA[nA.argmax().item()].item(), realN, 0.001)
        p_actual, _ = confidence_bound(nA_1[nA_1.argmax().item()].item(), realN_1, 0.001)
        if actual_predicted == labels:
            r_gt = dist.certify_radius(p_actual)
            gt.append(r_gt)
        
        if predicted == labels and base_predicted == predicted:
            r_p = dist.certify_radius(p1low_1)
            pd.append(r_p)

        if predicted == labels and predicted == actual_predicted:
            acc_100 += 1
            r_p = dist.certify_radius(p1low_1)
            r_gt = dist.certify_radius(p_actual)
            radius_act.append(r_gt)
            radius_pred.append(r_p)

    with open(model_dir+'baseline_test.json', 'w') as json_file:
        json.dump(baseline, json_file)

    with open(model_dir+'gt_test.json', 'w') as json_file:
        json.dump(gt, json_file)

    with open(model_dir+'acr_ours.json', 'w') as json_file:
        json.dump(pd, json_file)

    less = []
    more = []
    avg_radius_more = []
    avg_radius_less = []
    error_less = []
    error_more = []

    for i in range(len(radius_pred)):
        if radius_pred[i] < 0.25 or radius_act[i] < 0.25:
            continue
        if radius_pred[i] <= radius_act[i]:
            less.append((radius_act[i] - radius_pred[i])/radius_act[i])
            avg_radius_less.append(radius_act[i])
            error_less.append(radius_act[i] - radius_pred[i])
        else:
            more.append((radius_pred[i] - radius_act[i])/radius_act[i])
            avg_radius_more.append(radius_act[i])
            error_more.append(radius_pred[i] - radius_act[i])
    more = np.array(more)
    error_less = np.array(error_less)
    less = np.array(less)
    error_more = np.array(error_more)
    with open(model_dir+'variance.txt', 'w') as f:
        sys.stdout = f
        print('Total Examples for which predicted radius is less than robust radius :', len(less))
        print(f'Average percentage error when radius predicted is less than the robust radius is {100*less.mean():.2f}')
        print(f'Average error when radius predicted is less than the robust radius is {error_less.mean():.4f} with variance {error_less.var():.4f}')
        print(f'Average ground truth radius when radius predicted is less than the robust radius is {np.array(avg_radius_less).mean():.4f}')
        print(f'Total Examples for which predicted radius is more than robust radius :', len(more))
        print(f'Average percentage error when radius predicted is more than the robust radius is {100*more.mean():.2f}')
        print(f'Average error when radius predicted is less than the robust radius is {error_more.mean():.4f} with variance {error_more.var():.4f}')
        print(f'Average ground truth radius when radius predicted is more than the robust radius is {np.array(avg_radius_more).mean():.4f}')

    sys.stdout = original_stdout
