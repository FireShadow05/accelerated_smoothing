import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

def read_acr(acr_list, name):
    thresholds = np.arange(0, 1.6, 0.25)

    # Initialize counts dictionary for each threshold
    counts = {threshold: 0 for threshold in thresholds}

    # Iterate through values and count occurrences for each threshold
    for r in acr_list:
        for threshold in thresholds:
            if r > threshold:
                counts[threshold] += 1

    for threshold in thresholds:
        certified_accuracy = counts[threshold] / 10000
        print(f"For {name}, Certified accuracy for r > {threshold:.2f} is {certified_accuracy:.2%}")
    print('ACR :', np.array(acr_list).sum() / 10000)

def graph_acr(acr_list, name, thresholds):
    counts = {threshold: 0 for threshold in thresholds}
    for r in acr_list:
        for threshold in thresholds:
            if r > threshold:
                counts[threshold] += 1
    certified_accuracies = [counts[threshold] / 10000 for threshold in thresholds]
    plt.plot(thresholds, certified_accuracies, linestyle='-', label=name)
    plt.xlabel('Threshold')
    plt.ylabel('Certified Accuracy')
    plt.title('Certified Accuracy vs. Threshold')
    plt.legend()
    plt.grid(True)

def main(file_path):
    with open(file_path + '/cifar_resnet110-100000/acr_ours.json', 'r') as json_file:
        ours_110 = json.load(json_file)

    with open(file_path + '/cifar_resnet56-100000/acr_ours.json', 'r') as json_file:
        ours_56 = json.load(json_file)

    with open(file_path + '/cifar_resnet20-100000/acr_ours.json', 'r') as json_file:
        ours_20 = json.load(json_file)

    with open(file_path + '/cifar_resnet20-100000/gt_test.json', 'r') as json_file:
        gt = json.load(json_file)

    with open(file_path + '/cifar_resnet20-100000/baseline_test.json', 'r') as json_file:
        base = json.load(json_file)

    read_acr(base, 'baseline')
    read_acr(gt, 'ground truth')
    read_acr(ours_20, 'ours resnet20')
    read_acr(ours_56, 'ours resnet56')
    read_acr(ours_110, 'ours resnet110')

    thresholds = np.arange(0, 2.0, 0.01)

    plt.figure(figsize=(10, 6))
    graph_acr(base, 'Cohen (N = 100)', thresholds)
    graph_acr(gt, 'Cohen (N = 100000)', thresholds)
    graph_acr(ours_20, 'ours resnet20', thresholds)
    graph_acr(ours_56, 'ours resnet56', thresholds)
    graph_acr(ours_110, 'ours resnet110', thresholds)
    plt.savefig(file_path + '/certified_accuracy_plot.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot certified accuracy vs. threshold graphs')
    parser.add_argument('file_path', type=str, help='Path to the JSON files directory')
    args = parser.parse_args()
    main(args.file_path)
