import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models, transforms
import os
import random
from model.util_data_aug import *   #把原来的 model 代码换成 timm_mixup_model
from timm_oracle_mnist import get_datasets  #把原来的 代码换成 timm的代码
#from data.oracle_mnist import get_datasets
import os.path as osp
from config import *
from model.resnet import *
import torchsummary
import timm  #(for Vision Transformer tiny models)

'''
2024年7月19日 创建文件 目的是使用 timm库
首先实现 mixup的实用
# ViT-Tiny 
class_names_ViT_tiny = [
    'big', 'sun', 'moon', 'cattle', 'next', 'field', 'not', 'arrow', 'time', 'wood']
'''

def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--name', type=str, required=True, help="name of the experiment")
    parser.add_argument('--name', type=str, default="train_norm224", help="name of the experiment")
    parser.add_argument('--data_root', type=str,
                        #default="./data/oracle-mnist-size224",
                        default=r"D:\deeplearningsoft\PycharmProjects\OracleMNIST\data\oralce-mnist-size224",
                        help="the path to dataset with trian and val")
    parser.add_argument('--train_path', type=str,
                        default="./data/oracle-mnist-size224/train",
                        help="Training set path, for easy adjustment")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default="0")  # 默认使用第一个GPU（如果有的话）
    parser.add_argument('--input_size', type=int, default=224, help="size of input for data augmentation,To speed up, you can adjust it to 112.")
    parser.add_argument('--real_input_size', type=int, default=28, help="size of input for model")
    parser.add_argument('--if_val', type=int, default=1, help="if val when training")
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=200, help='default 200')
    parser.add_argument("--seed", type=int, default=15, help='random seed to reproduce')
    parser.add_argument("--ckpt", type=str, default='ckpt_pth_224', help='path to save result')
    #parser.add_argument("--ckpt", type=str, default='ckpt_timm_224', help='path to save result')

    parser.add_argument('--da_method', type=str, default='dynamic',
                        choices=['base',
                                 'mixup_intra', 'mixup_inter',
                                 'mosaic_intra', 'mosaic_inter',
                                 'cutmix_intra', 'cutmix_inter',
                                 'gridmask', 'dynamic',
                                 'combination111_intra',  #  Intra-mixup  Intra-CutMix  再 Dynamic GridMask
                                 'combination224_intra', # 1/2 Intra-mixup 1/2 Intra-CutMix  再 Dynamic GridMask
                                 'combination333_intra',  # 1/3 Intra-mixup 1/3 Intra-CutMix  1/3 Dynamic GridMask
                                 'combination12_intra',  #  Intra-mixup  Intra-CutMix
                                 'combination13_intra',  # Intra-mixup  Dynamic GridMask
                                 'combination23_intra',  # Intra-CutMix  再 Dynamic GridMask
                                 ],
                        help='data augmentation method')
    parser.add_argument("--mixup_alpha", type=float, default=0.005, help='hyperparameter alpha alpha for lambda distribution')
    parser.add_argument('--check_augmented_img', type=int, default=1, help='if save the augmented image to check')
    parser.add_argument('--check_combination_augmented_img', type=int, default=0, help='if save the combination augmented image to check')
    parser.add_argument('--print_model', type=int, default=0, help='if print model information')
    parser.add_argument("--DynamicGridMask_DelProb", type=float, default=1.0, help="0.9 best;hyperparameter probility of every unit's deletion ")
    parser.add_argument("--aspect_ratio", type=str, default='4:3', help="the aspect ratio of the delete block")
    parser.add_argument("--Cutmix_Prob", type=float, default=1.0, help="0.9 best;hyperparameter probility of Cutmix ")
    parser.add_argument("--Cutmix_alpha", type=float, default=0.1, help='hyperparameter alpha for lambda distribution')
    parser.add_argument("--Mosaic_Prob", type=float, default=0.1, help='hyperparameter alpha for lambda distribution')

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
        transforms.RandomRotation(15),
        transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
        ])
    val_transform = transforms.Compose([
        transforms.Resize((args.real_input_size, args.real_input_size)),
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

    data_loader = {}

    for k, v, in datasets.items():
        shuffle = True if k == 'train' else False
        drop_last = True if k == 'train' else False
        data_loader[k] = torch.utils.data.DataLoader(
            v, batch_size=args.batch_size, shuffle=shuffle,
            sampler=None, num_workers=args.num_workers, pin_memory=True,
            prefetch_factor=4,
            drop_last=drop_last,
            generator=data_generator
        )

    train_loader = data_loader['train']
    val_loader = data_loader['val']
    out_loader = data_loader['test_unknown']
    info = f"num image train: {len(datasets['train'])} val: {len(datasets['val'])}, ood: {len(datasets['test_unknown'])}"
    print(info)
    save_content.append(info)
    save_content.append('\n')

    #device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    device = args.device
    print("Gpu server's device:{0}".format(device))
    # model = Net1(num_classes=len(args.known_classes)).to(device)
    #model = model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10).to(device)
    model = ResNet18(num_c=len(args.known_classes)).to(device)
    if args.print_model:
        torchsummary.summary(model, (1, 28, 28))

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[80])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)
    #generate gridmask 10000
    if 'gridmask' in args.da_method:
        gridmasks_tensor = Generate_GridMasks(args).to(device) #希望运行更快
    elif 'dynamic' in args.da_method:
        gridmasks_tensor = Generate_DynamicGridMask(args).to(device)  # 希望运行更快
    elif 'cutmix' in args.da_method:
        cutmix_masks_tensor, cutmix_lam_tensor = Generate_CutMixMasksandLam(args)
        cutmix_masks_tensor = cutmix_masks_tensor.to(device)
        cutmix_lam_tensor = cutmix_lam_tensor.to(device)
    elif 'combination' in args.da_method:
        gridmasks_tensor = Generate_DynamicGridMask(args).to(device)  # 希望运行更快
        cutmix_masks_tensor, cutmix_lam_tensor = Generate_CutMixMasksandLam(args)
        cutmix_masks_tensor = cutmix_masks_tensor.to(device)
        cutmix_lam_tensor = cutmix_lam_tensor.to(device)

    # else:
    #     masks_tensor = Generate_DynamicGridMask(args).to(device)

    for epoch in range(args.epochs):
        save_content = []
        if 'gridmask' in args.da_method or 'dynamic' in args.da_method:
            train_loss = train_net(model, train_loader, criterion, optimizer, device, args, gridmasks_tensor)
        elif 'cutmix' in args.da_method:
            train_loss = train_net(model, train_loader, criterion, optimizer, device, args, cutmix_masks_tensor, cutmix_lam_tensor)
        elif 'combination' in args.da_method:
            train_loss = combination_train_net(model, train_loader, criterion, optimizer, device, args, gridmasks_tensor, cutmix_masks_tensor, cutmix_lam_tensor)
        else:
            train_loss = train_net(model, train_loader, criterion, optimizer, device, args)
        if args.if_val:
            val_loss, val_acc = val_net(model, val_loader, criterion, device, args)
            info = f'Epoch {epoch} train loss: {train_loss:.4f}, test loss: {val_loss:.4f} test acc: {val_acc:.4f}'
            print(info)
            save_content.append(info)
            save_content.append('\n')
        else:
            info = f'Epoch {epoch} train loss: {train_loss:.4f}'
            print(info)
            save_content.append(info)
            save_content.append('\n')
        scheduler.step()
        # compute AUROC
        for ood_method in ["max_prob", "max_logit", "energy", "GEN", "shannon_entropy"]:
            acc, auroc = eval_ood(model, val_loader, out_loader, device, args, method=ood_method)
            info = f"AUROC {ood_method}: {auroc*100:.4f}"
            print(info)
            AUROCs[ood_method] = auroc
            save_content.append(info)
            save_content.append('\n')
        with open(file_name, 'a+') as file:
            file.writelines(save_content)
    torch.save(model, model_name)
    return acc, AUROCs

'''
#生成 10000个 gridmask 
'''
def Generate_GridMasks(args):
    mask_numbers = 10000
    h = w = args.input_size
    ds_min = int(h/224 * 96)   # [224 min 96 max 224]
    # 生成 count_of_ones 个  超参
    ds = torch.from_numpy(np.random.randint(ds_min, h, mask_numbers))  # 改成和原始论文一模一样 不再提前指定 d is the length of one unit.
    deta_ys = torch.from_numpy(np.random.randint(0, ds - 1, mask_numbers))
    deta_xs = torch.from_numpy(np.random.randint(0, ds - 1, mask_numbers))
    rs = torch.full((mask_numbers,), 0.4)  # r: is the ratio of the shorter gray edge in a unit. 原文设置为 0.4

    width_units = deta_xs + rs * ds  # rs * ds 逐个元素相乘
    masks = torch.from_numpy(np.ones((mask_numbers, h, w), dtype=np.float32))  #生成对应的 mask个数

    # Y-axis direction
    for index,mask in enumerate(masks):
        for i in range(h // ds[index]):
            s_h = ds[index] * i
            t_h = s_h + deta_ys[index]
            s_h = max(min(s_h, h), 0)  # 防止超出边界
            t_h = max(min(t_h, h), 0)
            # X-axis direction
            for j in range(w // int(width_units[index])):
                s_x = ds[index] * j
                t_x = s_x + deta_xs[index]
                s_x = max(min(s_x, w), 0)  # 防止超出边界
                t_x = max(min(t_x, w), 0)
                # 最后一个 要不要自己决定 默认不要
                if (t_h - s_h) < deta_ys[index] or (t_x - s_x) < deta_xs[index]:
                    continue

                mask[s_h:t_h, s_x:t_x] *= 0  # rows from s to t =0

    return masks

'''
#生成 10000个 dynamic gridmask 
'''
def Generate_DynamicGridMask(args):
    mask_numbers = 10000
    h = w = args.input_size
    ds_min = int(h / 224 * 96 )  # [224 min 96 max 224] 先和gridmask 一样 后面需要调整 0.5:[48,112],[96,224],1.25:[120,224]
    # 生成 count_of_ones 个  超参
    ds = torch.from_numpy(np.random.randint(ds_min, h, mask_numbers))  # d is the length of one unit.
    deta_ys = torch.from_numpy(np.random.randint(3, ds - 1, mask_numbers))
    # 依据字体特征 设置 宽高比在一定范围内即可 默认[1/3,3x] 实际不能超过 d 和下面的二选1

    if args.aspect_ratio == '4:3': #删除块宽高比为4:3
        ####deta_x=[1/3,1]d,r = d-deta_x 换算出留存率k 约为0.51< 最佳0.64 #####################
        deta_x_max = deta_ys
        deta_x_min = deta_ys // 3
        deta_xs = [0] * mask_numbers
        for index, _ in enumerate(deta_xs):
            deta_xs[index] = np.random.randint(deta_x_min[index], int(deta_x_max[index]))
        deta_xs_tensor = torch.tensor(deta_xs, dtype=torch.int)
    elif args.aspect_ratio == '1:3':
        deta_xs_tensor = torch.from_numpy(np.random.randint(1, ds // 3, mask_numbers))
    elif args.aspect_ratio == '1:2':
        deta_xs_tensor = torch.from_numpy(np.random.randint(1, ds // 2, mask_numbers))
    else:
        #args.aspect_ratio == '1:1':  # 删除块宽高比为1:1
        deta_xs_tensor = torch.from_numpy(np.random.randint(1, ds - 1, mask_numbers))


    #rs = torch.full((mask_numbers,), 0.4)  # r: is the ratio of the shorter gray edge in a unit. 原文设置为 0.4

    #width_units = deta_xs_tensor + rs * ds  # rs * ds 逐个元素相乘  the same to GridMask
    width_units = ds  # the same to ds, remove the r Hyperparameter
    masks = torch.from_numpy(np.ones((mask_numbers, h, w), dtype=np.float32))  # 生成对应的 mask个数

    # Y-axis direction
    for index, mask in enumerate(masks):
        for i in range(h // ds[index]):
            s_h = ds[index] * i
            t_h = s_h + deta_ys[index]
            s_h = max(min(s_h, h), 0)  # 防止超出边界
            t_h = max(min(t_h, h), 0)
            # X-axis direction
            for j in range(w // int(width_units[index])):
                # 随机数  0.5 就是当前删除块以 0.5的概率进行删除
                # 依据论文 Grid Mask 中最佳保持图片比例在 67% 因此在此设置概率为 0.7  目前设置为 0.9 和 GridMask对比用
                if np.random.rand() > args.DynamicGridMask_DelProb:
                    continue

                s_x = ds[index] * j
                t_x = s_x + deta_xs_tensor[index]
                s_x = max(min(s_x, w), 0)  # 防止超出边界
                t_x = max(min(t_x, w), 0)
                # 最后一个 要不要自己决定 默认不要
                if (t_h - s_h) < deta_ys[index] or (t_x - s_x) < deta_xs_tensor[index]:
                    continue

                # 随机因子  防止每次删除块都从左上角开始  操作块大小 d
                max_random_h = ds[index] - (t_h - s_h)
                max_random_x = ds[index] - (t_x - s_x)
                if max_random_h > 0:
                    random_h = np.random.randint(0, max_random_h)
                else:
                    random_h = 0
                if max_random_x > 0:
                    random_x = np.random.randint(0, max_random_x)
                else:
                    random_x = 0

                s_h_j = s_h  # 防止随着J循环 random_h累加
                t_h_j = t_h
                s_h_j = s_h_j + random_h
                t_h_j = t_h_j + random_h
                t_h_j = max(min(t_h_j, h), 0)

                s_x = s_x + random_x
                t_x = t_x + random_x
                t_x = max(min(t_x, w), 0)
                mask[s_h_j:t_h_j, s_x:t_x] *= 0  # rows from s to t =0

    return masks

def Choice_GPU(args):
    device = torch.device("cpu")
    # 检查GPU是否可用
    if torch.cuda.is_available():
        # 获取GPU数量
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        # 检查指定的设备编号是否有效
        try:
            device_id = int(args.device)
            if device_id < 0 or device_id >= num_gpus:
                raise ValueError(f"Invalid GPU device ID: {args.device}. Should be in the range [0, {num_gpus - 1}].")

                # 设置设备为指定的GPU
            device = torch.device(f"cuda:{device_id}")
            print(f"Using GPU: {device}")
        except ValueError as e:
            print(e)
            # 如果指定的设备编号无效，则使用默认的第一个GPU（如果有的话）
            device = torch.device(f"cuda:0") if num_gpus > 0 else torch.device("cpu")
            print(f"Falling back to default device: {device}")
    else:
        # 如果没有GPU可用，则使用CPU
        device = torch.device("cpu")
        print("No GPU available. Using CPU.")
    return device

'''
#生成 10000个 CutMix mask and label 
'''
def Generate_CutMixMasksandLam(args):
    mask_numbers = 10000
    alpha = args.Cutmix_alpha
    W = H = args.input_size

    # Generate binary mask
    rand = np.random.beta(alpha, alpha, mask_numbers)
    cut_rat = np.sqrt(1. - rand)
    cut_w = (W * cut_rat).astype(int)
    rand = np.random.beta(alpha, alpha, mask_numbers)
    cut_rat = np.sqrt(1. - rand)
    cut_h = (H * cut_rat).astype(int)

    # Uniformly choose where to cut
    cx = np.random.randint(0, W, mask_numbers)
    cy = np.random.randint(0, H, mask_numbers)

    # Save the bounding box of the image to be cutmix'ed
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Create mask for the region to be cut and pasted
    masks = torch.from_numpy(np.zeros((mask_numbers, H, W), dtype=np.float32))  # 生成对应的 mask个数

    for index, mask in enumerate(masks):
        mask[bby1[index]:bby2[index], bbx1[index]:bbx2[index]] = 1.

    # 依据面积 计算
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))  # 10000 个
    lam_tensor = torch.from_numpy(lam)

    # for index,lam in enumerate(lam_tensor):
    #     if bbx2[index] < bbx1[index]:
    #         print('bbx2[index]:{0},bbx1[index]:{1}'.format(bbx2[index],bbx1[index]))
    #     if bby2[index] < bby1[index]:
    #         print('bby2[index]:{0},bby1[index]:{1}'.format(bby2[index],bby1[index]))

    return masks, lam_tensor



if __name__ == '__main__':
    args = get_args()
    args.device = Choice_GPU(args)  # judge GPU

    expr_root = osp.join(args.ckpt, args.name).replace('\\', '/')    # name："train_layer_norm"
    os.makedirs(expr_root, exist_ok=True)
    file_name = osp.join(expr_root, 'summary.txt').replace('\\', '/')  #解决windows 下 join 反斜杠问题
    # random train 5 times with different known and unknown class splits
    for item in range(1):
    #for item in range(1):
        args.split = item
        args.known_classes = OrcaleMNIST_split[item][0]
        args.unknown_classes = OrcaleMNIST_split[item][1]
        acc, AUROCs = main(args)

        #save every split result
        summary = []
        summary.append(f"Split: {item}\n")
        summary.append(f"Latest acc: {acc:.4f}\n")
        for key in AUROCs.keys():
            summary.append(f"AUROC {key}: {AUROCs[key]:.4f}")
            summary.append('\n')
        summary.append('\n')
        with open(file_name, 'a+') as file:
            file.writelines(summary)

