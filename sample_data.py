# evaluate a smoothed classifier on a dataset
import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import numpy as np
import datetime
from architectures import get_architecture
from utils import sample_count_list, count_arr

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("--batch", type=int, default=1024, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--start", type=int, default=0, help="start index of the dataset")
parser.add_argument("--end", type = int, default = 50000, help="end index of the dataset")
parser.add_argument("--itr", type = int, default = 0, help="end index of the dataset")
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    smooth_classifier = Smooth(base_classifier, 10, args.sigma)
    model_name = args.base_classifier.split('/')[-1].split('.pth')[0]
    out_dir = f'./sampled_dataset/{args.dataset}/{model_name}-{args.N}-{args.sigma}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    num_classes = get_num_classes(args.dataset)
    x_list = []
    label_list = []
    counts_list = []
    total_time = 0
    start = args.start
    end = args.end
    outfile = out_dir+f'/logs.txt'
    f = open(outfile, 'w')
    acc = 0
    base_classifier.eval()
    print(args.sigma)
    for i in range(start,end):
        print(i)

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        count = sample_count_list(base_classifier, x, args.N, args.batch, num_classes,args.sigma)
        after_time = time()
        x_list.append(x.cpu())
        label_list.append(torch.Tensor([label]).cpu())
        counts_list.append(torch.tensor(count).cpu())
        total_time += after_time - before_time
        
    smooth_outputs = torch.stack(counts_list, dim = 0)
    x_stacked = torch.stack(x_list, dim = 0)
    label_stacked = torch.stack(label_list, dim = 0)

    torch.save(smooth_outputs, os.path.join(out_dir, f'smooth_out_{args.split}_{args.N}.pth'))
    torch.save(x_stacked, os.path.join(out_dir, f'x_{args.split}.pth'))
    torch.save(label_stacked, os.path.join(out_dir, f'label_{args.split}.pth'))

    print(f"total time elapsed {total_time}, time per example {total_time/(end-start)}", file=f, flush=True)
    print(acc)
    f.close()

