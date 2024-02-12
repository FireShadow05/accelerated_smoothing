import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from statsmodels.stats.proportion import proportion_confint
import scipy
import numpy as np
from math import ceil
import random

def lambertWlog(logx, mode='lowerbound'):
    # compute lambertW(x) with log x as the input
    # stolen from Greg Yang et al's rs4a @ GitHub
    z = logx
    '''Computes LambertW(e^z) numerically safely.
    For small value of z, we use `scipy.special.lambertw`.
    For large value of z, we apply the approximation

        z - log(z) < W(e^z) < z - log(z) - log(1 - log(z)/z).
    '''
    if z > 500:
        if mode == 'lowerbound':
            return z - np.log(z)
        elif mode == 'upperbound':
            return z - np.log(z) - np.log(1 - np.log(z) / z)
        else:
            return np.NaN
            # raise ValueError('Unknown mode: ' + str(mode))
    else:
        return scipy.special.lambertw(np.exp(z)).real


# def calc_adaptive_sigma(sigma:float, d:int, k:int):
#     return sigma * np.sqrt((d + 2) / (d + 2 - 2 * k))


def read_pAs(file_path):
    """
        The format of sampling file:
        line could be empty, start with x, or start with o
        If start with x, metadata
        If start with o, then it will follow three numbers: #no, pA lower bound, pA upper bound
        Here, the pA is the probability of the true class.
    :param file_path:
    :return:
    """
    arr = list()
    no_set = set()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            fields = line.split(' ')
            if len(fields) > 0 and fields[0] == 'o':
                no, pAL, pAU = fields[1:]
                no, pAL, pAU = int(no), float(pAL), float(pAU)
                if no not in no_set:
                    arr.append((no, pAL, pAU))
                    no_set.add(no)
    return arr

def read_orig_Rs(file_path, num_stds):
    """
        The format of original R file:
        Each line corresponds to a sample.
    :param file_path:
    :param aux_stds:
    :return: [instance_no, radius, p1low, p1high, [[other-p1low1, other-p1high1], ..., [other-p1lowN, other-p1highN]]]
    """
    res = list()
    res_in_dict = dict()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) > 0:
                fields = line.strip().split(' ')
                no, r = int(fields[0]), float(fields[1])
                cur_line = [no, r, None, None, [[None, None] for _ in range(len(num_stds))]]
                res_in_dict[no] = cur_line
    for i in sorted(res_in_dict.keys()):
        res.append(res_in_dict[i])
    return res

def KL_divergence(input, target):
    eps = 1e-10
    input = torch.log(input + eps)
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    loss = kl_loss(input + eps, target + eps)
    return loss

def JS_divergence(input, target):
    m = 0.5 * (input + target)
    return 0.5 * KL_divergence(input, m) + 0.5 * KL_divergence(target, m)

def confidence_bound(NA: int, N: int, alpha: float) -> (float, float):
    """ Returns a (1 - alpha) confidence *interval* on a bernoulli proportion.

    This function uses the Clopper-Pearson method.

    :param NA: the number of "successes"
    :param N: the number of total draws
    :param alpha: the confidence level
    :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
    """
    ret = proportion_confint(NA, N, alpha=alpha, method="beta")
    return ret[0], ret[1]

class CustomDataset(Dataset):
    def __init__(self, label_file, smooth_outs_file, x_file, transform=None):
        self.label_data = torch.load(label_file)
        self.smooth_outs_data = torch.load(smooth_outs_file)
        self.x_data = torch.load(x_file)
        self.transform = transform

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        label = self.label_data[idx]
        smooth_outs = self.smooth_outs_data[idx]
        x = self.x_data[idx]
        if self.transform:
            x = self.transform(x)
        return label, smooth_outs, x

# Define a custom DataLoader for the test set
def get_test_dataloader(batch_size=32, num_samples = None):
    transform = transforms.Compose([transforms.ToTensor()])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    if num_samples == None:
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    else:
        indices = random.sample(range(len(testset)), num_samples)
    # Create a Subset of the testset using the sampled indices
        subset = torch.utils.data.Subset(testset, indices)

        # Create a DataLoader for the subset
        testloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=1)
    return testloader

def Testing(model,test_loader,device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = nn.Softmax()(model(images))

            """ Cross-Entropy Calculation"""
            target = torch.zeros_like(output,device=device)
            target[labels] = 1
            """ Accuracy Calculation"""
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            """ Calculating the avg radii"""
            # avg_radii += get_avg_radii(images,model)


    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    return correct/total

def count_arr(arr: np.ndarray, length: int) -> np.ndarray:
    counts = np.zeros(length, dtype=int)
    for idx in arr:
        counts[idx] += 1
    return counts

def sample_count_list(base_classifier, x: torch.tensor, num: int, batch_size, num_classes = 10, sigma = 0.5) -> np.ndarray:
    with torch.no_grad():
        counts = np.zeros(num_classes, dtype=int)
        for _ in range(ceil(num / batch_size)):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size

            batch = x.repeat((this_batch_size, 1, 1, 1))
            noise = torch.randn_like(batch, device='cuda') * sigma
            predictions = base_classifier(batch + noise).argmax(1)
            counts += count_arr(predictions.cpu().numpy(), num_classes)
        return counts

