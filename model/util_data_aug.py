import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
#from sklearn.metrics import roc_auc_score
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score, roc_curve, auc
import random
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
import os
import uuid
#from Data_Augmentation.MyGridMask import GridMask_NN  # 替换为你的 GridMask 类所在的模块


def train_net(model, loader, criterion, optimizer, device, args, masks_tensor: torch.Tensor = None, lam_tensor: torch.Tensor = None):
    running_loss = 0.0
    num_sample = 0
    for inputs, labels, _ in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # do data augmentation
        combine_method = 0 # 0:no method ;1 mixup; 2:cutmix; 4:dynamic;8:mosaic;
        if 'mixup' in args.da_method:
            inputs_new, labels_new = mixup(inputs, labels, args)
            if args.check_augmented_img:
                intra = True if "intra" in args.da_method else False
                prefix = "mixup_intra_class" if intra else "mixup_inter_class"
                Save_ImageTensor_ToImage(inputs_new[0], labels_new[0], prefix)
            combine_method = 1
        elif 'cutmix' in args.da_method:
            if( combine_method > 0 ):
                inputs_new, labels_new = batch_cutmix(inputs_new, labels_new, args, masks_tensor, lam_tensor)
                combine_method += 2
            else:
                inputs_new, labels_new = batch_cutmix(inputs, labels, args, masks_tensor, lam_tensor)
                if args.check_augmented_img:
                    intra = True if "intra" in args.da_method else False
                    prefix = "Cutmix_intra_class" if intra else "Cutmix_inter_class"
                    Save_ImageTensor_ToImage(inputs_new[0], labels_new[0], prefix)
                combine_method = 2
        elif 'gridmask' in args.da_method:
             inputs_new, labels_new = Batch_GridMask(inputs, labels, args, masks_tensor )
        elif 'dynamic' in args.da_method:
            if (combine_method > 0):
                inputs_new, labels_new = Batch_DynamicGridMask(inputs_new, labels_new, args, masks_tensor)
                combine_method += 4
            else:
                inputs_new, labels_new = Batch_DynamicGridMask(inputs, labels, args, masks_tensor)
                if args.check_augmented_img:
                    Save_ImageTensor_ToImage(inputs_new[0], labels_new[0], "DynamicGridmask")
                combine_method = 4
        elif 'mosaic' in args.da_method:
            if (combine_method > 0):
                inputs_new, labels_new = Batch_Mosaic4(inputs_new, labels_new, args)
                combine_method += 8
            else:
                inputs_new, labels_new = Batch_Mosaic4(inputs, labels, args)
                combine_method = 8
        else:
            inputs_new, labels_new = inputs, F.one_hot(labels, num_classes=len(args.known_classes))
            if args.check_augmented_img:
                prefix = 'base'
                Save_ImageTensor_ToImage(inputs_new[0], labels_new[0], prefix)

        if inputs_new.size()[-1] != args.real_input_size:
            size = [args.real_input_size, args.real_input_size]
            inputs_new = F.interpolate(inputs_new, size=size, mode='bilinear', align_corners=False)

        optimizer.zero_grad()
        num_sample += inputs_new.size(0)
        #print("inputs_new.size(0):{0}".format(inputs_new.size(0)))
        if(inputs_new.size(0) < 1): #还真有最后一个分配为0的情况 5555
            continue

        logits = model(inputs_new)
        # loss = criterion(logits, labels_new)
        loss = get_loss(logits, labels_new)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    loss = running_loss / num_sample
    return loss

'''Combination train'''
def combination_train_net(model, loader, criterion, optimizer, device, args, gridmasks_tensor: torch.Tensor = None, cutmix_masks_tensor: torch.Tensor = None, lam_tensor: torch.Tensor = None):
    running_loss = 0.0
    num_sample = 0
    for inputs, labels, _ in loader:
        if (labels.size(0) < 2):  # 还真有最后一个分配为0的情况 5555
            continue

        inputs = inputs.to(device)
        labels = labels.to(device)
        if '224' in args.da_method: # 1/2 Intra-mixup 1/2CutMix  再 Dynamic GridMask

            '''          
            split_combination = int(labels.size(0) * 0.5 ) #组合比例

            inputs_dynamic, labels_dynamic = inputs.clone() , F.one_hot(labels, num_classes=len(args.known_classes))

            inputs_dynamic[:split_combination], labels_dynamic[:split_combination] = inputs_mixup[:split_combination], labels_mixup[:split_combination]

            cutmix_split = labels_dynamic.size(0) - split_combination
            if labels_cutmix.size(0) > cutmix_split:
                inputs_dynamic[split_combination:], labels_dynamic[split_combination:] = inputs_cutmix[:cutmix_split], labels_cutmix[:cutmix_split]
            else:
                inputs_dynamic[split_combination: split_combination + labels_cutmix.size(0)], labels_dynamic[split_combination:split_combination + labels_cutmix.size(0)] = inputs_cutmix[:cutmix_split], labels_cutmix[:cutmix_split]

            inputs_new, labels_new = Batch_DynamicGridMask(inputs_dynamic, labels_dynamic, args, gridmasks_tensor)
            '''
            inputs_mixup, labels_mixup = mixup(inputs.clone(), labels.clone(), args)
            inputs_cutmix, labels_cutmix = batch_cutmix(inputs.clone(), labels.clone(), args, cutmix_masks_tensor, lam_tensor)
            inputs_dynamic = torch.cat((inputs_mixup, inputs_cutmix), dim=0)
            labels_dynamic = torch.cat((labels_mixup, labels_cutmix), dim=0)
            inputs_new, labels_new = Batch_DynamicGridMask(inputs_dynamic, labels_dynamic, args, gridmasks_tensor)
            if args.check_combination_augmented_img:
                prefix = 'combination224_intra'
                Save_ImageTensor_ToImage(inputs_new[0], labels_new[0], prefix)

        elif '333' in args.da_method:
            inputs_mixup, labels_mixup = mixup(inputs.clone(), labels.clone(), args)
            inputs_cutmix, labels_cutmix = batch_cutmix(inputs.clone(), labels.clone(), args, cutmix_masks_tensor, lam_tensor)
            inputs_dynamic, labels_dynamic = Batch_DynamicGridMask(inputs.clone(), labels.clone(), args, gridmasks_tensor)

            inputs_new = torch.cat((inputs_mixup, inputs_cutmix, inputs_dynamic), dim=0)
            labels_new = torch.cat((labels_mixup, labels_cutmix, labels_dynamic), dim=0)

            if args.check_combination_augmented_img:
                prefix = 'combination333_intra'
                Save_ImageTensor_ToImage(inputs_new[0], labels_new[0], prefix)

            '''
            split_combination = int(labels.size(0) / 3)  # 组合比例
            inputs_new, labels_new = inputs.clone(), F.one_hot(labels, num_classes=len(args.known_classes))

            inputs_new[:split_combination], labels_new[:split_combination] = inputs_mixup[:split_combination], labels_mixup[:split_combination]

            cutmix_split = 2*split_combination - split_combination
            inputs_new[split_combination:2*split_combination], labels_new[split_combination:2*split_combination] = inputs_cutmix[:cutmix_split], labels_cutmix[:cutmix_split]

            dynamic_split = labels_new.size(0) - 2 * split_combination
            inputs_new[2 * split_combination:], labels_new[2 * split_combination:] = inputs_dynamic[:dynamic_split], labels_dynamic[:dynamic_split]
            '''
        elif '111' in args.da_method:
            inputs_mixup, labels_mixup = mixup(inputs, labels, args)

            if labels_mixup.size(0) < labels.size(0):
                if labels_mixup.dim() != labels.dim():
                    labels = F.one_hot(labels, num_classes=len(args.known_classes))
                inputs[:labels_mixup.size(0)], labels[:labels_mixup.size(0)] = inputs_mixup[:labels_mixup.size(0)], labels_mixup[:labels_mixup.size(0)]
                inputs_cutmix, labels_cutmix = batch_cutmix(inputs, labels, args, cutmix_masks_tensor,lam_tensor)
            else:
                inputs_cutmix, labels_cutmix = batch_cutmix(inputs_mixup, labels_mixup, args, cutmix_masks_tensor, lam_tensor)

            if labels_cutmix.size(0) < labels.size(0):
                if labels_cutmix.dim() != labels.dim():
                    labels = F.one_hot(labels, num_classes=len(args.known_classes))
                inputs[:labels_cutmix.size(0)], labels[:labels_cutmix.size(0)] = inputs_cutmix[:labels_cutmix.size(0)], labels_cutmix[:labels_cutmix.size(0)]
                inputs_new, labels_new = Batch_DynamicGridMask(inputs, labels, args, gridmasks_tensor)
            else:
                inputs_new, labels_new = Batch_DynamicGridMask(inputs_cutmix, labels_cutmix, args, gridmasks_tensor)
        elif '12' in args.da_method:
            inputs_mixup, labels_mixup = mixup(inputs.clone(), labels.clone(), args)
            inputs_cutmix, labels_cutmix = batch_cutmix(inputs.clone(), labels.clone(), args, cutmix_masks_tensor, lam_tensor)

            inputs_new = torch.cat((inputs_mixup, inputs_cutmix ), dim=0)
            labels_new = torch.cat((labels_mixup, labels_cutmix), dim=0)
            if args.check_combination_augmented_img:
                prefix = 'combination12_intra'
                Save_ImageTensor_ToImage(inputs_new[0], labels_new[0], prefix)
        elif '13' in args.da_method:
            inputs_mixup, labels_mixup = mixup(inputs.clone(), labels.clone(), args)
            inputs_dynamic, labels_dynamic = Batch_DynamicGridMask(inputs.clone(), labels.clone(), args, gridmasks_tensor)

            inputs_new = torch.cat((inputs_mixup,  inputs_dynamic), dim=0)
            labels_new = torch.cat((labels_mixup,  labels_dynamic), dim=0)
            if args.check_combination_augmented_img:
                prefix = 'combination13_intra'
                Save_ImageTensor_ToImage(inputs_new[0], labels_new[0], prefix)
        elif '23' in args.da_method:
            inputs_cutmix, labels_cutmix = batch_cutmix(inputs.clone(), labels.clone(), args, cutmix_masks_tensor, lam_tensor)
            inputs_dynamic, labels_dynamic = Batch_DynamicGridMask(inputs.clone(), labels.clone(), args, gridmasks_tensor)

            inputs_new = torch.cat(( inputs_cutmix, inputs_dynamic), dim=0)
            labels_new = torch.cat(( labels_cutmix, labels_dynamic), dim=0)
            if args.check_combination_augmented_img:
                prefix = 'combination23_intra'
                Save_ImageTensor_ToImage(inputs_new[0], labels_new[0], prefix)
        else:
            inputs_new, labels_new = inputs, F.one_hot(labels, num_classes=len(args.known_classes))

        if inputs_new.size()[-1] != args.real_input_size:
            size = [args.real_input_size, args.real_input_size]
            inputs_new = F.interpolate(inputs_new, size=size, mode='bilinear', align_corners=False)

        optimizer.zero_grad()
        num_sample += inputs_new.size(0)

        logits = model(inputs_new)
        # loss = criterion(logits, labels_new)
        loss = get_loss(logits, labels_new)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs_new.size(0)
    loss = running_loss / num_sample
    return loss


def get_loss(logit, target):
    preds = logit.log_softmax(dim=-1)
    loss = torch.sum(-target * preds, dim=-1)
    return loss.mean()


def val_net(model, loader, criterion, device, args):
    with torch.no_grad():
        running_loss = 0.0
        running_corrects = 0
        num_sample = 0
        for inputs, labels, _ in loader:
            inputs = inputs.to(device)
            num_sample += inputs.size(0)
            labels = labels.to(device)
            if inputs.size()[-1] != args.real_input_size:
                size = [args.real_input_size, args.real_input_size]
                inputs = F.interpolate(inputs, size=size, mode='bilinear', align_corners=False)
            logits = model(inputs)
            _, preds = torch.max(logits, 1)
            loss = criterion(logits, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        loss = running_loss / num_sample
        acc = running_corrects.double() / num_sample
        return loss, acc


class Net1(nn.Module):
    def __init__(self, num_classes=6):
        super(Net1, self).__init__()
        ndf = 32
        self.model = nn.Sequential(
            nn.Conv2d(1, ndf, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([ndf, 14, 14]),
            nn.Conv2d(ndf, ndf * 2, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([ndf * 2, 7, 7]),
            nn.Conv2d(ndf * 2, ndf * 4, 7, 1, padding=0),
        )
        self.fc = nn.Sequential(
            nn.Linear(ndf * 4, ndf * 2),
            nn.LeakyReLU(),
            nn.Linear(ndf * 2, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Net0(nn.Module):
    def __init__(self):
        super(Net0, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_ood_score(logit, prob,  method='max_logit'):
    logit = logit.detach().cpu().numpy()
    #print(logit.device)  # 这应该输出 'cpu'
    #logit_cpu = logit.detach().cpu()
    #print(logit_cpu.device)  # 这应该还是输出 'cpu'
    #logit = logit_cpu.numpy()  # 现在应该不会出现错误了
    prob = prob.detach().cpu().numpy()
    if method == 'max_logit':
        return -np.max(logit, axis=1).flatten()
    elif method == 'max_prob':
        return -np.max(prob, axis=1).flatten()
    elif method == 'shannon_entropy':
        known_entropy = entropy(prob, axis=1).flatten()
        return known_entropy
    elif method == 'energy':
        #return -np.log(np.exp(logit).sum(axis=1)).flatten() overflow
        return -logsumexp(logit)
    elif method == 'GEN':
        M = 6
        gamma = 0.1
        sorted_prob = np.sort(prob, axis=-1)[:, ::-1]
        gen_entropy = np.sum(sorted_prob[:, :M] ** gamma * (1 - sorted_prob[:, :M]) ** gamma, axis=-1).flatten()
        return gen_entropy
    #elif method == 'ours':
    #    print(labels)
    #    print(len(labels))
    #    print(logit)
    #    print(len(logit))
    #    print(prob)
    #    print(len(prob))
    #    fpr, tpr, thresholds = roc_curve(labels, logit)
    #    roc_auc_manual = auc(fpr, tpr)
    #    return roc_auc_manual
    else:
        #raise NotImplemented('Please check the method to compute ood score')
        raise NotImplementedError('Please check the method to compute ood score')


def eval_ood(model, known_loader, unknown_loader, device, args, method='max_logit'):
    model.eval()
    ##################################################################
    # for known
    ##################################################################
    correct, num_known, = 0, 0
    known_ood_scores = []
    with torch.no_grad():
        for inputs, labels, _ in known_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if inputs.size()[-1] != args.real_input_size:
                size = [args.real_input_size, args.real_input_size]
                inputs = F.interpolate(inputs, size=size, mode='bilinear', align_corners=False)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            pred = logits.max(1)[1]
            num_known += inputs.size(0)
            correct += (pred == labels).sum()
            #known_ood_scores.append(get_ood_score(logits, probs, method))
            known_ood_scores.append(get_ood_score(logits, probs, method))
    known_ood_scores = np.hstack(known_ood_scores).flatten()
    assert known_ood_scores.shape[0] == num_known
    ##################################################################
    # for unknown
    ##################################################################
    num_unknown = 0
    unknown_ood_scores = []
    with torch.no_grad():
        for inputs, labels, _ in unknown_loader:
            inputs = inputs.to(device)

            # 使用双线性插值将图片尺寸从224x224缩放到28x28  从112x112缩放到28x28
            if inputs.size()[-1] != args.real_input_size:
                size = [args.real_input_size, args.real_input_size]
                inputs = F.interpolate(inputs, size=size, mode='bilinear', align_corners=False)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            num_unknown += inputs.size(0)
            unknown_ood_scores.append(get_ood_score(logits, probs, method))
            #unknown_ood_scores.append(get_ood_score(logits, probs, labels, method))
    unknown_ood_scores = np.hstack(unknown_ood_scores)
    assert unknown_ood_scores.shape[0] == num_unknown
    ##################################################################
    # compute output: acc and auroc
    ##################################################################
    acc = float(correct) / float(num_known)
    labels = np.hstack((np.zeros_like(known_ood_scores), np.ones_like(unknown_ood_scores)))
    ood_scores = np.hstack((known_ood_scores, unknown_ood_scores))
    auroc = roc_auc_score(labels, ood_scores)
    #if method=='ours':
	#auroc = ood_scores
    #    auroc = sum(ood_scores)/len(ood_scores)
    #else:
    #    auroc = roc_auc_score(labels, ood_scores)

    return acc, auroc


'''Prevent overflow'''
def logsumexp(logit):
    max_value = np.max(logit, axis=1, keepdims=True)
    shifted_logit = logit - max_value
    sum_exp = np.exp(shifted_logit).sum(axis=1)
    return np.log(sum_exp) + max_value.flatten()


def get_fuse_index(labels, intra=True):
    perm = np.random.permutation(labels.size(0))
    if intra:
        new_index = []
        for id in range(labels.size(0)):
            flag = False
            for new_id in perm:
                if labels[id] == labels[new_id] and id != new_id:
                    new_index.append(new_id)
                    flag = True
                    break
                else:
                    continue
            if not flag:
                new_index.append(id)
        return new_index
    else:
        return perm

'''label [256,6]'''
def get_fuse_index_six(labels, intra=True):
    if labels.dim() != 2:
        raise ValueError("labels must be a 2D tensor")

    perm = np.random.permutation(labels.size(0))
    if intra:
        new_index = []
        for id in range(labels.size(0)):
            flag = False
            for new_id in perm:
                # 注意：这里使用了 all() 方法来检查所有元素是否都满足条件
                if torch.all(labels[id] == labels[new_id]) and id != new_id:
                    new_index.append(new_id)
                    flag = True
                    break
                else:
                    continue
            if not flag:
                new_index.append(id)
        return new_index
    else:
        return perm


def mixup(inputs, labels, args):
    alpha = args.mixup_alpha
    intra = True if "intra" in args.da_method else False
    if labels.dim() == 2:
        new_index = get_fuse_index_six(labels, intra=intra)
    else:
        new_index = get_fuse_index(labels, intra=intra)

    #new_index = get_fuse_index(labels, intra=intra)
    rand = np.random.beta(alpha, alpha, size=len(new_index))
    rand = torch.from_numpy(rand).to(inputs.device).unsqueeze(-1)
    if labels.dim() == 1:
        labels_mixup = F.one_hot(labels, num_classes=len(args.known_classes)).float()
    else:
        labels_mixup = labels

    labels_new = rand * labels_mixup + (1 - rand) * labels_mixup[new_index]
    rand = rand.unsqueeze(-1).unsqueeze(-1).type_as(inputs)
    inputs_new = rand * inputs + (1 - rand) * inputs[new_index]

    # if args.check_augmented_img:
    #     prefix = "mixup_intra_class" if intra else "mixup_inter_class"
    #     Save_ImageTensor_ToImage(inputs_new[0], labels_new[0], prefix)
    return inputs_new, labels_new


'''
Use CutMix with a 100% probability alpha = beta = 2
'''
def CutMix(inputs, labels, inputs_new, labels_new, intraClass = True):
    for index, value in enumerate(inputs):
        # 使用随机函数选择一个键
        xj_index = random.randint(0, len(inputs) - 1)  # [0,256)
        count = 0  # 初始化计数器
        # 类内融合
        if intraClass:
            # 初始化一个候选索引列表，排除当前索引
            candidates = [i for i in range(len(inputs)) if i != index and labels[i] == labels[index]]
            # 从候选索引列表中随机选择一个索引
            if len(candidates) > 0:
                xj_index = random.choice(candidates)
            else:
                xj_index = index
        else:
            # 初始化一个候选索引列表，排除当前索引
            candidates = [i for i in range(len(inputs)) if i != index and labels[i] != labels[index]]
            # 从候选索引列表中随机选择一个索引
            if len(candidates) > 0:
                xj_index = random.choice(candidates)
            else:
                xj_index = index

        # 生成一个Beta分布的随机数
        alpha = 2.0
        beta = 2.0
        Lambda = np.random.beta(alpha, beta)

        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs[index].shape, Lambda)

        # 裁剪图像
        cropped_xi = inputs[index][:, bby1:bby2, bbx1:bbx2]
        # 确保 image_xj 上的区域与 cropped_xi 大小相同
        cropped_xj = inputs[xj_index][:, bby1:bby2, bbx1:bbx2]
        # 一种方法是将 cropped_xj 粘贴（paste）回 image_xi
        inputs_new[index][:, bby1:bby2, bbx1:bbx2] = cropped_xj
        # 现在 image_xi 的 (bbx1, bby1, bbx2, bby2) 区域已经被替换为 image_xj 的相应区域
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs[index].shape[-1] * inputs[index].shape[-2]))

        labels_new[index] = int(labels[index]) * lam + int(labels[xj_index]) * (1. - lam)

        if intraClass :
            prefix = "intra-calssCutMix"
        else:
            prefix = "inter-calssCutMix"
        #Save_ImageTensor_ToImage(inputs_new[index], labels_new[index], prefix)
    return inputs_new, labels_new


'''
Use CutMix 配套函数
'''
def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    elif len(size) == 2:
        W = size[0]
        H = size[1]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

'''
Use Mosaic4 
'''
def Mosaic4(inputs, labels, inputs_new, labels_new, intraClass = True):
    for index, value in enumerate(inputs):
        # 使用随机函数选择一个键
        xj_index = random.randint(0, len(inputs) - 1)  # [0,256)
        count = 0  # 初始化计数器
        # 类内融合
        if intraClass:
            # 初始化一个候选索引列表，排除当前索引
            candidates = [i for i in range(len(inputs)) if i != index and labels[i] == labels[index]]
        else:
            # 初始化一个候选索引列表，排除当前索引
            candidates = [i for i in range(len(inputs)) if i != index and labels[i] != labels[index]]
        # 从候选索引列表中随机选择3 个索引 组合成 图片集合
        if len(candidates) < 3:   #少于3个不弄了
            continue
        image_tensor_list = []
        for i in random.sample(candidates,3):
            image_tensor_list.append(inputs[i])
        image_tensor_list.append(inputs[index])

        inputs_new[index], labels_new[index] = Mosaic4_Pocess(image_tensor_list, labels, inputs[index].shape[-1], inputs[index].shape[-2], 0.25, 0.25)

    return inputs_new, labels_new

'''
Use Mosaic4  process details   image_file_list:Tensor
'''
def Mosaic4_Pocess(image_tensor_list, labels, mo_w, mo_h, scale_x, scale_y):
    """
        Implement mosaic augmentation
        image_file_list: list of 4 images
        mo_h: height of mosaic-augmented image
        mo_w: width of mosaic-augmented image
    """
    if len(image_tensor_list) != 4:
        assert("please input 4 images")
        return

    # 如果你需要 tensor 的深拷贝（即数据本身的副本），你需要对每个 tensor 调用 .clone()  #新的拷贝，所有操作在这上面，不影响原来的
    image_tensor_list_new = [tensor.clone() for tensor in image_tensor_list]
    new_img = image_tensor_list[0].clone()* 0


    # split points
    div_point_x = int(mo_w * scale_x) # 28*0.25 =7
    div_point_y = int(mo_h * scale_y) # 28*0.25 =7

    # loop through images
    for i in range(len(image_tensor_list_new)):
        # top left image, img_0
        if i == 0:
            # width and height of the top left image
            w0 = div_point_x
            h0 = div_point_y
            img_0  = random_rotate(image_tensor_list_new[0], h0, w0 )  # 7*7
            # top left
            new_img[:, :div_point_y, :div_point_x] = img_0

        # top right image
        elif i == 1:
            w1 = mo_w - div_point_x  # trừ sẽ khớp
            h1 = div_point_y  # giữ nguyên như cái i=0
            img_1 = random_rotate(image_tensor_list_new[1], h1, w1)  # 7*21  h*w
            new_img[:,:div_point_y, div_point_x:] = img_1

        # bottom left image
        elif i == 2:
            w2 = div_point_x
            h2 = mo_h - div_point_y
            img_2 = random_rotate(image_tensor_list_new[2], h2, w2)   # 21*7 h*w
            #new_img[div_point_y:, :div_point_x, :] = img_2.reshape(h2, w2, 1)
            new_img[:, div_point_y:, :div_point_x] = img_2

        # bottom right image
        else:
            w3 = mo_w - div_point_x
            h3 = mo_h - div_point_y
            img_3 = random_rotate(image_tensor_list_new[3], h3, w3)  # 21*21
            #new_img[div_point_y:, div_point_x:, :] = img_3.reshape(h3, w3, 1)
            new_img[:, div_point_y:, div_point_x:] = img_3

    new_label = labels[0] * 0.25 * 0.25  + labels[1] * 0.25 * 0.75 + labels[2] * 0.75 * 0.25 + labels[3] * 0.75 * 0.75

    #Save_ImageTensor_ToImage(new_img, new_label,"Mosaic4_")

    return  new_img, new_label


'''
# Random rotate image with predefined size 
'''
def random_rotate(image_tensor, expected_h, expected_w, p = 0.25):
    """
        Implement random crop image
        image_name: for example img_0.jpg
        image_dir: original image directory
        expected_h: expected height of the image, it depends on the scale_y
        expected_w: expected width of the image, it depends on the scale_x
    """
    # 将 Tensor 转换为 PIL 图像
    to_pil = ToPILImage()
    pil_image = to_pil(image_tensor.squeeze(0))  # 去除批次维度，因为 PIL 图像不需要批次维度
    if random.random() < p:  # 如果随机数小于p，则应用旋转变换
        # 旋转图像，这里以旋转 30 度为例
        angle = random.randint(-40,40)  # 可以是 -40 到 40 之间的任何值
        rotated_pil_image = pil_image.rotate(angle, expand=True)  # expand=True 用于保持图像内容的完整性
        #rotated_pil_image = pil_image.rotate(angle)  # expand=True 用于保持图像内容的完整性
    else:
        rotated_pil_image = pil_image

    target_size = (expected_w, expected_h)  #width x height
    resized_pil_image = rotated_pil_image.resize(target_size, Image.ANTIALIAS) # (width, height)
    # 转换 PIL 图像回 Tensor，并添加回批次维度
    to_tensor = ToTensor()
    resized_tensor = to_tensor(resized_pil_image)  #(1,7,7)

    return resized_tensor

def get_Mosaic_fuse_index(labels, intra=True):
    perm = np.random.permutation(labels.size(0))
    if intra:
        #new_index = np.zeros((256, 3))  # 生成一个256x3的二维数组，所有元素为零
        new_index = np.zeros((labels.shape[0], 3))  # 生依据batch大小生成
        for id in range(labels.size(0)):
            count = 0
            for new_id in perm:
                if labels[id] == labels[new_id] and id != new_id:
                    new_index[id][count] = new_id
                    count = count + 1
                    if count > 2:
                        break
                else:
                    continue
            if count < 3:
                for my_i in range(3 - count ):
                    new_index[id][count - my_i -1] = id
        return new_index
    else:
        perm1 = np.random.permutation(labels.size(0))
        perm2 = np.random.permutation(labels.size(0))
        # 使用numpy.stack沿着新的轴（列）堆叠它们
        perm_mosaic = np.stack((perm, perm1, perm2), axis=1)  # 256x3
        return perm_mosaic

'''
Batch_Mosaic 4 images, Batch  256
'''
def Batch_Mosaic4(inputs, labels, args):
    intra = True if "intra" in args.da_method else False
    new_index = get_Mosaic_fuse_index(labels, intra=intra)  # 256x3
    mosaic_start = int(args.Mosaic_Prob * args.batch_size) #有多少个图片要进行 mosaic

    #images_leftop = inputs[:mosaic_start].clone()  # 显式地克隆前20个元素 到一个新的张量

    # 图片拼合
    h = w = args.input_size
    images_leftop = inputs.clone()
    images_leftop = F.interpolate(images_leftop, size=[int(h*0.25),int(w*0.25)], mode='bilinear', align_corners=False)
    #print("images_leftop.shap:{0}".format(images_leftop.shape))
    images_righttop = inputs[new_index[:, 0]].clone()
    images_righttop = F.interpolate(images_righttop, size=[int(h * 0.25), int(w * 0.75)], mode='bilinear', align_corners=False)
    images_leftbottom = inputs[new_index[:, 1]].clone()
    images_leftbottom = F.interpolate(images_leftbottom, size=[int(h * 0.75), int(w * 0.25)], mode='bilinear', align_corners=False)
    images_rightbottom = inputs[new_index[:, 2]].clone()
    images_rightbottom = F.interpolate(images_rightbottom, size=[int(h * 0.75), int(w * 0.75)], mode='bilinear', align_corners=False)

    # 计算缩放后的大小
    size_left_right =size_top_bottom = int(w * 0.25)  #
    # 处理 最后一个batch < 256的情况
    # if inputs.shape[0] <  args.batch_size:
    #     # 创建一个形状为 (2, 112, 112) 的全零张量 例如最后一个batch 的 input为  [2, 1, 112, 112]
    #     padded_images = torch.zeros(inputs.shape[0], h, w, device=images_leftop.device,
    #                                 dtype=images_leftop.dtype)  # [2,112,112]
    # else:
    #     # 创建一个形状为 (256, 112, 112) 的全零张量
    #     padded_images = torch.zeros(args.batch_size, h, w, device=images_leftop.device, dtype=images_leftop.dtype) # 256,112,112
    padded_images = torch.zeros(inputs.shape[0], h, w, device=images_leftop.device, dtype=images_leftop.dtype)  # [2,112,112]
    padded_images = padded_images.unsqueeze(1)  # [2,1,112,112]

    padded_images[:,:, :size_left_right, :size_left_right] = images_leftop # 将 images_leftop 复制到 padded_images 的左上角

    padded_images[:,:, :size_left_right, size_left_right:] = images_righttop  # 将 images_righttop 复制到 padded_images 的右上角

    padded_images[:,:, size_left_right:, :size_left_right] = images_leftbottom  # 将 images_righttop 复制到 padded_images 的右上角

    padded_images[:,:, size_left_right:, size_left_right:] = images_rightbottom  # 将 images_rightbottom 复制到 padded_images 的右下角

    # label拼合
    labels = F.one_hot(labels, num_classes=len(args.known_classes)).float()
    label_lefttop = labels.clone()
    label_righttop = labels[new_index[:, 0]].clone()     # 0列
    label_leftbottom = labels[new_index[:, 1]].clone()   # 1列
    label_rightbottom = labels[new_index[:, 2]].clone()  # 2列
    labels_new = label_lefttop * 0.25 * 0.25 + label_righttop * 0.25 * 0.75 + label_leftbottom * 0.75 * 0.25 + label_rightbottom * 0.75 * 0.75

    #计算概率问题
    mosaic_start = int(args.Mosaic_Prob * args.batch_size)
    padded_images[mosaic_start:] = inputs[mosaic_start:]
    labels_new[mosaic_start:] = labels[mosaic_start:]

    #for index, image in enumerate(padded_images):
    #    Save_ImageTensor_ToImage(padded_images[index], labels[0], "Mosaic")

    return padded_images, labels_new


'''
Use GridMask 
原始GridMask  Batch  image gridmask 256
'''
def Batch_GridMask(inputs, labels, args, masks_tensor):
    prob = 0.9 #gridmask 的起作用概率
    # 生成一个长度为256的随机tensor
    random_tensor = torch.rand(256)
    # 根据条件生成新的tensor，满足条件的设为1（True），不满足的设为0（False）
    probability = (random_tensor < prob)  # 或者使用.int()来得到整数类型的tensor
    count_of_ones = torch.sum(probability).item()  # 统计probability中1（即True）的元素个数,即有多少个需要 gridmask

    # 生成 0 到 9999 之间的 count_of_ones 个随机索引（不重复）
    indices = torch.randperm(masks_tensor.size(0))[:count_of_ones]
    # 使用这些索引来选择子集
    selected_masks = masks_tensor[indices] # [230, 112, 112]
    masks_expanded = selected_masks.unsqueeze(1) # [230,1,112,112]
    # 进行切片和乘法
    if inputs.shape[0] < count_of_ones:
        inputs = inputs * masks_expanded[:inputs.shape[0]]
    else:
        inputs[:count_of_ones] = inputs[:count_of_ones] * masks_expanded
    if labels.dim() == 1:
        labels_new = F.one_hot(labels, num_classes=len(args.known_classes)).float()
    else:
        labels_new = labels

    #for index, image in enumerate(inputs) :
    #    Save_ImageTensor_ToImage(inputs[index], labels[0], "GridMask")

    return inputs, labels_new


    '''
    #封装成一个类  创建一个 GridMask 实例
    grid_mask = GridMask_NN(use_h=True, use_w=True, rotate=1, offset=False, ratio=0.4, mode=1, prob=0.9)
    # 应用 GridMask
    inputs_new = grid_mask(inputs)
    labels_new = F.one_hot(labels, num_classes=len(args.known_classes)).float()

    #for index, image in enumerate(inputs_new) :
    #   Save_ImageTensor_ToImage(inputs_new[index], labels[0], "GridMask")

    return  inputs_new, labels_new
    '''


'''
Use GridMask 
原始GridMask  one by one image gridmask
'''
def OnebyOne_GridMask(inputs, labels, args):
    inputs_new = inputs.clone()
    for index, value in enumerate(inputs):
        # 每张图的变量都随机生成
        d = random.randint(96, 224)  # 改成和原始论文一模一样 不再提前指定 d is the length of one unit.
        deta_y = random.randint(1, d - 1)
        deta_x = random.randint(1, d - 1)
        r = 0.4  # r: is the ratio of the shorter gray edge in a unit. 原文设置为 0.4
        probability = 0.9

        inputs_new[index] = OneImage_OriginalGridmask(inputs[index], deta_x, deta_y, d, r, probability)

        #Save_ImageTensor_ToImage( inputs_new[index], labels_new[index], "GridMask")
    return inputs_new, labels


"""
#gridmask  和论文中的基本一致
deta_x, deta_y,r,d refence to the paper 
"""
def OneImage_OriginalGridmask(image_tensor,deta_x, deta_y, d, r = 0.5,  prob=1.0):
    """
     original gridmask image
    :param image_tensor: Gray image    data shape is [channel, height, width]
    :param deta_x:
    :param deta_y:
    :param r: 默认为 0.6  CIFAR10 中 0.4 效果最好   ImageNet 0.6
    :param d:
    :param prob: 概率
    :return: processed_image  type pil image
    """
    # Set whether to apply gridmask. The default probability is to use all,
    # such as user prob=0.3, then the 30% images will use gridmask
    if np.random.rand() > prob:
        return image_tensor

    h, w = image_tensor.shape[-1], image_tensor.shape[-2]

    mask = np.ones((h, w), np.float32)

    width_unit = deta_x + int(r*d)
    #Y-axis direction
    for i in range(h // d ):
        s_h = d * i
        t_h = s_h + deta_y
        s_h = max(min(s_h, h), 0) #防止超出边界
        t_h = max(min(t_h, h), 0)
        # X-axis direction
        for j in range(w // width_unit):
            s_x = d * j
            t_x = s_x + deta_x
            s_x = max(min(s_x, w), 0)  # 防止超出边界
            t_x = max(min(t_x, w), 0)
            #最后一个 要不要自己决定 默认不要
            if (t_h - s_h) < deta_y or (t_x - s_x) < deta_x:
                continue

            mask[s_h:t_h, s_x:t_x] *= 0  #rows from s to t =0

    # # 保存生成的 mask 为图片
    # mask_array = np.copy(mask)  # shape[224,224] min:0.0 max:1.0
    # scaled_array = (mask_array * 255).astype(np.uint8)  # shape[224,224] min:0.0 max:255.0
    # for x in range(0, image.size[0], d):
    #     for y in range(0, image.size[1], d):
    #         x_end = min(image.size[0], x + d)
    #         y_end = min(image.size[1], y + d)
    #         scaled_array[x:x_end, y_end -1:y_end] = 255
    #         scaled_array[x_end -1:x_end, y:y_end] = 255
    #
    # mask_image = Image.fromarray(scaled_array)
    # #image.show()
    # mask_save_path = "D:/PycharmProjects/OracleMNIST/data/oralce-mnist-size224/grimask/test_grimask_mask_E.bmp"
    # mask_image.save(mask_save_path)

    # 检查CUDA是否可用
    if torch.cuda.is_available():
        mask = torch.from_numpy(mask).float().cuda()
        mask_cpu = mask.cpu()  # 将PyTorch Tensor从GPU拷贝到CPU
        mask_numpy = mask_cpu.numpy()  # 将CPU上的PyTorch Tensor转换为numpy数组
    else:
        mask = torch.from_numpy(mask).float()  # 不使用.cuda()，张量将留在CPU上
        mask_numpy = mask.numpy()  # 将 PyTorch Tensor 转换为 numpy 数组


    # mask 是一个 numpy数组 [28,28]
    # 将NumPy数组转换为PyTorch Tensor
    mask_tensor = torch.from_numpy(mask_numpy)
    # 例如，如果image_tensor是[C, H, W]，而mask_numpy是[H, W]，你可能需要添加一个维度
    if mask_tensor.dim() < image_tensor.dim():
        mask_tensor = mask_tensor.unsqueeze(0)  # 在第一个维度上增加一个维度
    # 现在使用expand_as来扩展mask_tensor
    expanded_mask = mask_tensor.expand_as(image_tensor)

    # linux 检测到不在同一个设备上 让二者到一个设备上
    expanded_mask = expanded_mask.to(image_tensor.device)
    gridmask_image_tensor = image_tensor * expanded_mask

    return gridmask_image_tensor


'''
Use DynamicGridMask 
'''
def DynamicGridMask(inputs, labels, args):
    for index, value in enumerate(inputs):
        d = random.randint(12, 28)
        deta_y = random.randint(3, d - 1)  # 保证删除块 水平方向 > 7 y方向不做限制
        # 依据字体特征 设置 宽高比在一定范围内即可 默认[1/3,3x]
        deta_x_max = 3 * deta_y
        deta_x_min = deta_y // 3
        deta_x = random.randint(deta_x_min, deta_x_max)
        if deta_x > d:
            deta_x = min(d - 1, deta_x)  # 保证 r至少等于1

        mid_r = random.randint(int(d * 0.35), int(d * 0.6))  # 对应于原始论文中的 r =0.4  是短边和长边的比例 控制在0.35-0.6之间
        r = mid_r - deta_x
        probability = 0.5

        inputs[index] = DynamicGridmask_oneImage(inputs[index], deta_x, deta_y, d, r, probability)

        #Save_ImageTensor_ToImage(inputs_new[index], labels_new[index], "DynamicGridMask")
    return inputs, labels


'''
Use DynamicGridMask 
'''
def Batch_DynamicGridMask(inputs, labels, args, masks_tensor):
    prob = 1.0  # gridmask 的起作用概率  已经在生成 masks使用过概率了
    # 生成一个长度为256的随机tensor
    random_tensor = torch.rand(256)
    # 根据条件生成新的tensor，满足条件的设为1（True），不满足的设为0（False）
    probability = (random_tensor < prob)  # 或者使用.int()来得到整数类型的tensor
    count_of_ones = torch.sum(probability).item()  # 统计probability中1（即True）的元素个数,即有多少个需要 gridmask

    # 生成 0 到 9999 之间的 count_of_ones 个随机索引（不重复）
    indices = torch.randperm(masks_tensor.size(0))[:count_of_ones]
    # 使用这些索引来选择子集
    selected_masks = masks_tensor[indices]  # [230, 112, 112]
    masks_expanded = selected_masks.unsqueeze(1)  # [230,1,112,112]
    # 进行切片和乘法
    if inputs.shape[0] < count_of_ones:
        inputs = inputs * masks_expanded[:inputs.shape[0]]
    else:
        inputs[:count_of_ones] = inputs[:count_of_ones] * masks_expanded

    if labels.dim() == 1:
        labels_new = F.one_hot(labels, num_classes=len(args.known_classes)).float()
    else:
        labels_new = labels

    #for index, image in enumerate(inputs) :
    #   Save_ImageTensor_ToImage(inputs[index], labels[0], "DynamicGridmask")
    # if args.check_augmented_img:
    #     Save_ImageTensor_ToImage(inputs[0], labels[0], "DynamicGridmask")


    return inputs, labels_new



"""
#DynamicGridmask_oneImage   处理单张图片
deta_x, deta_y,r,d refence to the paper 
"""
def DynamicGridmask_oneImage(image_tensor,deta_x, deta_y, d, r = 0.5,  prob=0.7):
    """
     dynamic gridmask image
    :param image_tensor: Gray image    data shape is [channel, height, width]
    :param deta_x:
    :param deta_y:
    :param r: 默认为 0.6  CIFAR10 中 0.4 效果最好   ImageNet 0.6
    :param d:
    :param prob: 概率
    :return: processed_image  type pil image
    """
    h, w = image_tensor.shape[-1], image_tensor.shape[-2]
    mask = np.ones((h, w), np.float32)
    # Y-axis direction
    if d == 0:
        raise ValueError("d cannot be 0")  # 实际上这行代码永远不会执行，因为randint不会生成0

    for i in range(h // d):
        s_h = d * i
        t_h = s_h + deta_y
        s_h = max(min(s_h, h), 0)  # 防止超出边界
        t_h = max(min(t_h, h), 0)
        # X-axis direction
        for j in range(w // (deta_x + r)):
            # 随机数  0.5 就是当前删除块以 0.5的概率进行删除  #依据论文 Grid Mask 中最佳保持图片比例在 67% 因此在此设置概率为 0.7
            if np.random.rand() > prob:
                continue
            s_x = d * j
            t_x = s_x + deta_x
            s_x = max(min(s_x, w), 0)  # 防止超出边界
            t_x = max(min(t_x, w), 0)
            # 最后一个 要不要自己决定 默认不要
            if (t_h - s_h) < deta_y or (t_x - s_x) < deta_x:
                continue

            # 随机因子  防止每次删除块都从左上角开始  操作块大小 d
            max_random_h = d - (t_h - s_h)
            max_random_x = d - (t_x - s_x)
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


    # 检查CUDA是否可用
    if torch.cuda.is_available():
        mask = torch.from_numpy(mask).float().cuda()
        mask_cpu = mask.cpu()  # 将PyTorch Tensor从GPU拷贝到CPU
        mask_numpy = mask_cpu.numpy()  # 将CPU上的PyTorch Tensor转换为numpy数组
    else:
        mask = torch.from_numpy(mask).float()  # 不使用.cuda()，张量将留在CPU上
        mask_numpy = mask.numpy()  # 将 PyTorch Tensor 转换为 numpy 数组

    # mask 是一个 numpy数组 [28,28]
    # 将NumPy数组转换为PyTorch Tensor
    mask_tensor = torch.from_numpy(mask_numpy)
    # 例如，如果image_tensor是[C, H, W]，而mask_numpy是[H, W]，你可能需要添加一个维度
    if mask_tensor.dim() < image_tensor.dim():
        mask_tensor = mask_tensor.unsqueeze(0)  # 在第一个维度上增加一个维度
    # 现在使用expand_as来扩展mask_tensor
    expanded_mask = mask_tensor.expand_as(image_tensor)

    #linux 检测到不在同一个设备上 让二者到一个设备上
    expanded_mask = expanded_mask.to(image_tensor.device)
    gridmask_image_tensor = image_tensor * expanded_mask

    return gridmask_image_tensor


'''
save processed image to bmp 
'''
def Save_ImageTensor_ToImage(image_tensor, label, str_prefix):
    if image_tensor.size()[-1] != 224:
        size = [224, 224]
        image_tensor = image_tensor.unsqueeze(1)   #[1,112,112] -》[1,1,112,112]
        image_tensor = F.interpolate(image_tensor, size=size, mode='bilinear', align_corners=False)
        image_tensor = image_tensor.squeeze(1) #[1,1,112,112] ->[1,112,112]
    # 去除单通道维度（这一步是可选的，但有助于后续处理）
    gray_tensor = image_tensor.squeeze(0)  # 现在形状为[224, 224] min:0 max:1.0

    image_array = np.copy(
        gray_tensor.cpu())  # shape[224,224] min:0.0 max:1.0  can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first
    image_array = image_array * 0.5 + 0.5
    scaled_array = (image_array * 255).astype(np.uint8)  # shape[224,224] min:0.0 max:255.0
    image = Image.fromarray(scaled_array)

    # 指定路径和文件名保存图片
    # 生成UUID
    unique_id = uuid.uuid4()
    # 转换为字符串并去除连字符
    image_name = str_prefix + "_" + str(label) + "_" + str(unique_id).replace('-', '') + '.bmp'
    image_full_path = os.path.join('D:/PycharmProjects/CAM-master/obi_pth/local_images/dynamic/', image_name).replace('\\', '/')
    #image_full_path = os.path.join('./ckpt_pth_224/images/', image_name).replace('\\', '/')
    # image_full_path = os.path.join('/home/OracleMNIST/ImageTensor_Results/', image_name).replace('\\', '/')
    image.save(image_full_path)


'''2024年10月14日按照批量处理方法改写的cutmix代码'''
def batch_cutmix(inputs, labels, args,masks_tensor, lam_tensor):
    prob = args.Cutmix_Prob  # Cutmix 的起作用概率 0.9 230
    random_tensor = torch.rand(labels.size(0)) # 生成一个长度为256的随机tensor
    # 根据条件生成新的tensor，满足条件的设为1（True），不满足的设为0（False）
    probability = (random_tensor < prob)  # 或者使用.int()来得到整数类型的tensor
    count_of_ones = torch.sum(probability).item()  # 统计probability中1（即True）的元素个数,即有多少个需要 gridmask

    # 生成 0 到 9999 之间的 count_of_ones 个随机索引（不重复）
    indices = torch.randperm(masks_tensor.size(0))[:count_of_ones]
    # 使用这些索引来选择子集
    selected_masks = masks_tensor[indices]  # [230, 112, 112]
    masks_expanded = selected_masks.unsqueeze(1)  # [230,1,112,112]
    selected_lams = lam_tensor[indices] # [230]
    selected_lams = selected_lams.unsqueeze(1) #[230,1]

    alpha = args.Cutmix_alpha
    intra = True if "intra" in args.da_method else False
    if labels.dim() == 2:
        new_index = get_fuse_index_six(labels, intra=intra)
    else:
        new_index = get_fuse_index(labels, intra=intra)

    shuffled_data = inputs[new_index]

    if labels.dim() == 1:
        labels = F.one_hot(labels, num_classes=len(args.known_classes)).float()

    #labels = F.one_hot(labels, num_classes=len(args.known_classes)).float()
    target_a = labels
    target_b = labels[new_index]
    #labels = F.one_hot(labels, num_classes=len(args.known_classes)).float()
    #labels_new = rand * labels + (1 - rand) * labels[new_index]
    # 进行切片和乘法
    if inputs.shape[0] < count_of_ones:
        mixed_data = inputs * (1 - masks_expanded[:inputs.shape[0]]) + shuffled_data * masks_expanded[:inputs.shape[0]]
        mixed_targets = target_a[:inputs.shape[0]] * (1 - selected_lams[:inputs.shape[0]]) + target_b[:inputs.shape[0]] * selected_lams[:inputs.shape[0]]
    else:
        mixed_data = inputs[:count_of_ones] * (1 - masks_expanded) + shuffled_data[:count_of_ones] * masks_expanded
        mixed_targets = target_a[:count_of_ones] * (1 - selected_lams[:count_of_ones]) + target_b[:count_of_ones] * selected_lams[:count_of_ones]
        # 沿着特征维度（第1个维度）拼接
        mixed_data = torch.cat((mixed_data, inputs[count_of_ones:]), dim=0)
        mixed_targets = torch.cat((mixed_targets, labels[count_of_ones:]), dim=0)

    # for index, image in enumerate(mixed_data) :
    #    Save_ImageTensor_ToImage(mixed_data[index], mixed_targets[0], "Cutmix")
    # if args.check_augmented_img:
    #     prefix = "Cutmix_intra_class" if intra else "Cutmixup_inter_class"
    #     Save_ImageTensor_ToImage(mixed_data[0], mixed_targets[0], prefix)

    return mixed_data, mixed_targets