import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models, transforms
import os
import random
from model.util_model_data_aug import *  #把原来的 model 代码换成 util_model_data_aug
from timm_oracle_mnist import get_datasets  #把原来的 代码换成 timm的代码
#from data.oracle_mnist import get_datasets
import os.path as osp
from config import *
from model.resnet import *
import torchsummary

#from torchsummary import summary

#from timm.data.mixup import Mixup

'''
2024年8月10日
自己写的不行  在许老师指导下重新实现 写到 da_main_ood_cutout.py 里面去了
'''


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="train_norm224", help="name of the experiment")
    parser.add_argument('--data_root', type=str,
                        default=r"D:\PycharmProjects\OracleMNIST\data\oralce-mnist-size224",
                        #default="./data/oracle-mnist-size224",
                        help="the path to dataset with trian and val")  #替换成绝对路径
    parser.add_argument('--train_path', type=str,
                        default="./data/oracle-mnist-size224/train",
                        help="Training set path, for easy adjustment")

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    #parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--input_size', type=int, default=224, help="size of input for data augment")
    parser.add_argument('--real_input_size', type=int, default=28, help="size of real input")
    parser.add_argument('--if_val', type=int, default=1, help="if val when training")
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument("--seed", type=int, default=15, help='random seed to reproduce')
    parser.add_argument("--ckpt", type=str, default='batch_224', help='path to save result')

    parser.add_argument('--da_method', type=str, default='cutmix_intra',
                        choices=['base',
                                 'mixup_intra', 'mixup_inter',
                                 'mosaic_intra', 'mosaic_inter',
                                 'cutmix_intra', 'cutmix_inter'],
                        help='data augmentation method')

    parser.add_argument("--alpha", type=float, default=0.5, help='hyperparameter alpha for beta distribution beta(alpha,alpha)')
    parser.add_argument("--check_augmented_img", type=int, default=1, help='if save the check_augmented image')
    parser.add_argument("--print_model", type=int, default=0, help='if print model information')

    args = parser.parse_args()
    return args


def main(args):
    # to save result
    ckpt_root = osp.join(args.ckpt, args.name).replace('\\', '/')  #解决windows 下 join 反斜杠问题
    model_name = osp.join(ckpt_root, f"split_{args.split}_latest.pth").replace('\\', '/')  #解决windows 下 join 反斜杠问题
    file_name = osp.join(ckpt_root, f'split_{args.split}_running.txt').replace('\\', '/')  #解决windows 下 join 反斜杠问题
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
    train_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        #transforms.RandomResizedCrop(args.input_size),

        transforms.RandomRotation(15),
        transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    data_dir = args.data_root
    train_root = os.path.join(data_dir, 'train').replace('\\', '/')  #解决windows 下 join 反斜杠问题
    val_root = os.path.join(data_dir, 'test').replace('\\', '/')
    print("train_root: ", train_root)  # 自增 看变量
    print("val_root: ", val_root)  # 自增 看变量 目前没问题了

    datasets = get_datasets(
        train_transform, val_transform, train_root, val_root,
        train_classes=args.known_classes, open_set_classes=args.unknown_classes
    )
    #print("args:",args)  # 自增 看变量
    #print("datasets:",datasets)  # 自增 看变量
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
    # torchsummary.summary(model, (1, 28, 28))
    # 使用 summary 函数打印模型详细信息
    model = ResNet18(num_c=len(args.known_classes)).to(device)
    #torchsummary.summary(model, (1, 28, 28))

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # Adam 学习率优化
    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[80])  #学习率调度器
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr)  #模拟退火

    for epoch in range(args.epochs):
        # train
        #train_loss, train_acc = train_net(model, train_loader, criterion, optimizer, device, args)
        train_loss = train_net(model, train_loader, criterion, optimizer, device, args)
        # val
        if args.if_val:
            #val_loss, val_acc = val_net(model, val_loader, criterion, device)
            val_loss, val_acc = val_net(model, val_loader, criterion, device, args)
            #info = f'Epoch {epoch} train Loss: {train_loss:.4f} Acc: {train_acc:.4f}, test Loss: {val_loss:.4f} Acc: {val_acc:.4f}'
            info = f'Epoch {epoch} train Loss: {train_loss:.4f} , test Loss: {val_loss:.4f} Acc: {val_acc:.4f}'
            print(info)
            save_content.append(info)
            save_content.append('\n')
        else:
            info = f'Epoch {epoch} train Loss: {train_loss:.4f} '
            print(info)
            save_content.append(info)
            save_content.append('\n')
        scheduler.step()
        # compute AUROC
        for ood_method in ["max_prob", "max_logit", "energy", "GEN", "shannon_entropy"]:
            #acc, auroc = eval_ood(model, val_loader, out_loader, device, method=ood_method)
            acc, auroc = eval_ood(model, val_loader, out_loader, device, args, method=ood_method)
            #acc, auroc, auroc_resutl = eval_ood(model, val_loader, out_loader, device, method=ood_method)
            info = f"AUROC {ood_method}: {auroc * 100:.4f}"
            print(info)
            #print(auroc_resutl)
            AUROCs[ood_method] = auroc
            save_content.append(info)
            save_content.append('\n')

    if args.print_model:
        torch.save(model, model_name)

    with open(file_name, 'w+') as file:
        file.writelines(save_content)

    return acc, AUROCs


if __name__ == '__main__':
    args = get_args()

    print(args)  #自增看变量
    expr_root = osp.join(args.ckpt, args.name).replace('\\', '/')  # name："train_layer_norm"
    os.makedirs(expr_root, exist_ok=True)
    file_name = osp.join(expr_root, 'summary.txt').replace('\\', '/')  #解决windows 下 join 反斜杠问题
    # random train 5 times with different known and unknown class splits
    for item in range(5):
        args.split = item
        args.known_classes = OrcaleMNIST_split[item][0]
        args.unknown_classes = OrcaleMNIST_split[item][1]
        acc, AUROCs = main(args)

        # save every split result
        summary = []
        summary.append(f"Split: {item}\n")
        summary.append(f"Latest acc: {acc:.4f}\n")
        for key in AUROCs.keys():
            summary.append(f"AUROC {key}: {AUROCs[key]:.4f}")
            summary.append('\n')
        summary.append('\n')
        with open(file_name, 'a+') as file:
            file.writelines(summary)
