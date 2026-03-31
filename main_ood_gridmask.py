import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models, transforms
import os
import random
from model.model import *
from data.oracle_mnist import get_datasets
import os.path as osp
from config import *
from model.resnet import *
import albumentations as albu
import cv2
import albumentations.pytorch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help="name of the experiment")
    parser.add_argument('--data_root', type=str,
                        default="./data/oracle-mnist-GrideMask224/oracle-mnist-size224",
                        help="the path to dataset with trian and val")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--input_size', type=int, default=28, help="size of input")
    parser.add_argument('--if_val', type=int, default=1, help="if val when training")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument("--seed", type=int, default=15, help='random seed to reproduce')
    parser.add_argument("--ckpt", type=str, default='ckpt', help='path to save result')
    args = parser.parse_args()
    return args

def main(args):
    # to save result
    ckpt_root = osp.join(args.ckpt, args.name)
    model_name = osp.join(ckpt_root, f"split_{args.split}_latest.pth")
    file_name = osp.join(ckpt_root, f'split_{args.split}_running.txt')
    save_content = []
    save_content.append(f"Experiment split {args.split}\n")
    save_content.append(f"known classes: {args.known_classes}\n")
    save_content.append(f"unknown classes: {args.unknown_classes}\n")
    AUROCs = {}

    # to reproduce random process
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    # to reproduce DataLoader
    data_generator = torch.Generator()
    data_generator.manual_seed(seed)

    # Define transforms for the training and validation sets
    #train_transform = transforms.Compose([
    #    transforms.RandomResizedCrop((args.input_size, args.input_size)),
    #    # transforms.RandomRotation(15),
    #    # transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
    #    transforms.Grayscale(),
    #    transforms.ToTensor(),
    #    transforms.Normalize([0.5], [0.5])
    #    ])
    train_transform = albu.Compose([
        albu.GridDropout(num_grid=(3, 3), fill_value=0, p=0.5), # 应用GridMask 3x3
        #albu.RandomResizedCrop(args.input_size),
        albu.Resize(args.input_size, args.input_size),

        #albu.Grayscale(1.0),
        albu.ToGray(),
        albu.pytorch.transforms.ToTensorV2(),  # 转换为Tensor格式
        albu.Normalize([0.5], [0.5])
        ])

    val_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
        ])
    data_dir = args.data_root
    train_root = os.path.join(data_dir, 'train')
    val_root = os.path.join(data_dir, 'test')
    datasets = get_datasets(
        train_transform, val_transform, train_root, val_root,
        train_classes=args.known_classes, open_set_classes=args.unknown_classes
    )
    data_loader = {}
    for k, v, in datasets.items():
        shuffle = True if k == 'train' else False
        data_loader[k] = torch.utils.data.DataLoader(
            v, batch_size=args.batch_size, shuffle=shuffle,
            sampler=None, num_workers=args.num_workers, pin_memory=True,
            prefetch_factor=4,
            generator=data_generator
        )
    train_loader = data_loader['train']
    val_loader = data_loader['val']
    out_loader = data_loader['test_unknown']
    info = f"num image train: {len(datasets['train'])} val: {len(datasets['val'])}, ood: {len(datasets['test_unknown'])}"
    print(info)
    save_content.append(info)
    save_content.append('\n')

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    #model = Net1(num_classes=len(args.known_classes)).to(device)
    model = ResNet18(num_c=len(args.known_classes)).to(device)


    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[80])

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_net(model, train_loader, criterion, optimizer, device)
        # val
        if args.if_val:
            val_loss, val_acc = val_net(model, val_loader, criterion, device)
            info = f'Epoch {epoch} train Loss: {train_loss:.4f} Acc: {train_acc:.4f}, test Loss: {val_loss:.4f} Acc: {val_acc:.4f}'
            print(info)
            save_content.append(info)
            save_content.append('\n')
        else:
            info = f'Epoch {epoch} train Loss: {train_loss:.4f} Acc: {train_acc:.4f}'
            print(info)
            save_content.append(info)
            save_content.append('\n')
        scheduler.step()
        # compute AUROC
        for ood_method in ["max_prob", "max_logit", "energy", "GEN", "shannon_entropy"]:
            acc, auroc = eval_ood(model, val_loader, out_loader, device, method=ood_method)
            info = f"AUROC {ood_method}: {auroc*100:.4f}"
            print(info)
            AUROCs[ood_method] = auroc
            save_content.append(info)
            save_content.append('\n')
    torch.save(model, model_name)
    with open(file_name, 'w+') as file:
        file.writelines(save_content)

    return acc, AUROCs


if __name__ == '__main__':
    args = get_args()
    expr_root = osp.join(args.ckpt, args.name)
    os.makedirs(expr_root, exist_ok=True)
    file_name = osp.join(expr_root, 'summary.txt')
    # random train 5 times with different known and unknown class splits
    for item in range(5):
        args.split = item
        args.known_classes = OrcaleMNIST_split[item][0]
        args.unknown_classes = OrcaleMNIST_split[item][1]
        acc, AUROCs = main(args)

        # save every split result
        summary = []
        summary.append(f"Split: {item}\n")
        summary.append(f"Latest acc: {acc:.3f}\n")
        for key in AUROCs.keys():
            summary.append(f"AUROC {key}: {AUROCs[key]:.3f}")
            summary.append('\n')
        summary.append('\n')
        with open(file_name, 'a+') as file:
            file.writelines(summary)

