import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture
# from utils import CustomDataset, get_test_dataloader, Testing, JS_divergence, KL_divergence
from utils import get_test_dataloader, CustomDataset, JS_divergence, Testing, confidence_bound
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
import time
from distribution import StandardGaussian
import json
import sys 

# Save the current standard output
original_stdout = sys.stdout

softmax = nn.Softmax()

def calculate_acr(model, N, sigma,test_loader):
    dist = StandardGaussian(d = 3072, scale = sigma)
    acr = 0
    acr_list = []
    count = 0
    i = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            i+=1
            # print(i)
            images = images.to(device)
            labels = labels.to(device)
            output = N*softmax(model(images))
            nA = output.detach().cpu().floor()
            nA = nA.squeeze()
            _, predicted = torch.max(output.data, 1)
            realN = nA.sum()
            p1low_1, p1high_1 = confidence_bound(nA[nA.argmax().item()].item(), realN, 0.001)
            if predicted == labels:
                r = dist.certify_radius(p1low_1)
                acr_list.append(r)
                acr += r
                # print(acr)
    print(f'Average Certified Radius for all examples is: {acr/len(test_loader)}')
    return acr_list, acr/len(test_loader)

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("model_type", type=str, help="type of the surrogate model")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("data_path", type=str, help="path to the folder containing the sampled data")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
args = parser.parse_args()

if __name__ == '__main__':
    model = get_architecture(args.model_type, args.dataset)
    sigma = args.sigma
    label_file = args.data_path + f'/label_{args.split}.pth'
    smooth_outs = args.data_path + f'/smooth_out_{args.split}_{args.N}.pth'
    x_file = args.data_path + f'/x_{args.split}.pth'
    custom_dataset = CustomDataset(label_file, smooth_outs, x_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define a custom dataset (as shown in the previous response)

    # Create a DataLoader
    batch_size = args.batch
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001, 
            betas=(0.5,0.999)
        )

    # Learning rate scheduler (Reduce learning rate on a schedule)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training loop
    num_epochs = 200  # You can adjust this
    model.to(device)
    test_loader = get_test_dataloader(batch_size=1024)
    eval_set = get_test_dataloader(batch_size=1,num_samples = 1000)
    acr_set = get_test_dataloader(batch_size=1)
    n = args.N
    total_training_time = 0
    train_time_acr = []
    train_time_acc = []
    train_time_loss = []
    output_dir = f'./models/{args.dataset}/{sigma}/{args.model_type}-{args.N}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir+'/log.txt', 'w') as f:
    # Redirect standard output to the file
        sys.stdout = f
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            start_time = time.time()
            for batch_idx, (labels, smooth_outs, images) in enumerate(dataloader):
                # print('Training...')
                smooth_outs = smooth_outs.to(device).float()/n
                # print(smooth_outs)
                labels = labels.to(device).int()
                images = images.to(device)

                optimizer.zero_grad()

                outputs = model(images)  
                targets = softmax(outputs)
                loss = JS_divergence(targets, smooth_outs)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            scheduler.step()
            end_time = time.time()
            epoch_training_time = end_time - start_time
            total_training_time += epoch_training_time
            epoch_loss = running_loss / len(dataloader)
            print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}")
            acc = Testing(model,test_loader,device)
            _, acr = calculate_acr(model,n,sigma,eval_set)
            train_time_acc.append(acc)
            train_time_acr.append(acr)
            train_time_loss.append(epoch_loss)

            # Adjust learning rate
        start_time = time.time()
        acr_list, acr = calculate_acr(model,n,sigma,acr_set)
        end_time = time.time()
        epoch_evaluation_time = end_time - start_time
        with open(output_dir+f'/acr_test.json', 'w') as f:
            json.dump(acr_list, f)
        with open(output_dir+f'/train_time_acc.json', 'w') as f:
            json.dump(train_time_acc, f)
        with open(output_dir+f'/train_time_acr.json', 'w') as f:
            json.dump(train_time_acr, f)
        with open(output_dir+f'/train_time_loss.json', 'w') as f:
            json.dump(train_time_loss, f)
        plt.plot(train_time_acc, label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()
        plt.savefig(output_dir+f'/training_accuracy.png')
        plt.close()

        plt.plot(train_time_acr, label='ACR')
        plt.xlabel('Epoch')
        plt.ylabel('ACR')
        plt.title('Average Certified Radius (ACR)')
        plt.legend()
        plt.savefig(output_dir+f'/acr.png')
        plt.close()

        plt.plot(train_time_loss, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig(output_dir+f'/training_loss.png')
        plt.close()
        torch.save({
                'arch': args.model_type,
                'state_dict': model.state_dict(),
            }, os.path.join(output_dir, 'checkpoint.pth.tar'))

        # Calculate and print total training and evaluation times
        print(f"Total training time: {total_training_time} seconds")
        print(f"Total evaluation time: {epoch_evaluation_time} seconds")

        print("Training finished.")
    sys.stdout = original_stdout
