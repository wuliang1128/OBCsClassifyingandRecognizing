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


def train_net(model, loader, criterion, optimizer, device, args):
    running_loss = 0.0
    running_corrects = 0
    num_sample = 0
    for inputs, labels, _ in loader:
        inputs = inputs.to(device)
        num_sample += inputs.size(0)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        _, preds = torch.max(logits, 1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        Count_batch = 1
        for AugmentionType, Is_Active in args.DataAugmentionTypes.items():
            if Is_Active and AugmentionType == 'intra-classMosaic':
                inputs_new = inputs.detach().clone()  # 拷贝出来新的张量副本 不影响原来的张量 挑选在原来的 inpus挑选，结果保存在 新的 inputs_new 避免相互影响
                labels_new = labels.detach().clone()
                num_sample += inputs_new.size(0)
                Mosaic4(inputs, labels, inputs_new, labels_new, intraClass=True)  # Mosaic4

                optimizer.zero_grad()
                logits = model(inputs_new)
                _, preds = torch.max(logits, 1)
                loss = criterion(logits, labels_new)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                Count_batch += 1

            if Is_Active and AugmentionType == 'inter-classMosaic':
                inputs_new = inputs.detach().clone()  # 拷贝出来新的张量副本 不影响原来的张量 挑选在原来的 inpus挑选，结果保存在 新的 inputs_new 避免相互影响
                labels_new = labels.detach().clone()
                num_sample += inputs_new.size(0)
                Mosaic4(inputs, labels, inputs_new, labels_new, intraClass=False)

                optimizer.zero_grad()
                logits = model(inputs_new)
                _, preds = torch.max(logits, 1)
                loss = criterion(logits, labels_new)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                Count_batch += 1

            if Is_Active and AugmentionType == 'intra-classmixup':
                inputs_new = inputs.detach().clone()  # 拷贝出来新的张量副本 不影响原来的张量 挑选在原来的 inpus挑选，结果保存在 新的 inputs_new 避免相互影响
                labels_new = labels.detach().clone()
                num_sample += inputs_new.size(0)
                mixup(inputs, labels, inputs_new, labels_new, intraClass=True)

                optimizer.zero_grad()
                logits = model(inputs_new)
                _, preds = torch.max(logits, 1)
                loss = criterion(logits, labels_new)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                Count_batch += 1

            if Is_Active and AugmentionType == 'inter-classmixup':
                inputs_new = inputs.detach().clone()  # 拷贝出来新的张量副本 不影响原来的张量 挑选在原来的 inpus挑选，结果保存在 新的 inputs_new 避免相互影响
                labels_new = labels.detach().clone()
                num_sample += inputs_new.size(0)
                mixup(inputs, labels, inputs_new, labels_new, intraClass=False)

                optimizer.zero_grad()
                logits = model(inputs_new)
                _, preds = torch.max(logits, 1)
                loss = criterion(logits, labels_new)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                Count_batch += 1

            if Is_Active and AugmentionType == 'intra-classCutMix':
                inputs_new = inputs.detach().clone()  # 拷贝出来新的张量副本 不影响原来的张量 挑选在原来的 inpus挑选，结果保存在 新的 inputs_new 避免相互影响
                labels_new = labels.detach().clone()
                num_sample += inputs_new.size(0)
                CutMix(inputs, labels, inputs_new, labels_new, intraClass=True)

                optimizer.zero_grad()
                logits = model(inputs_new)
                _, preds = torch.max(logits, 1)
                loss = criterion(logits, labels_new)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                Count_batch += 1

            if Is_Active and AugmentionType == 'inter-classCutMix':
                inputs_new = inputs.detach().clone()  # 拷贝出来新的张量副本 不影响原来的张量 挑选在原来的 inpus挑选，结果保存在 新的 inputs_new 避免相互影响
                labels_new = labels.detach().clone()
                num_sample += inputs_new.size(0)
                CutMix(inputs, labels, inputs_new, labels_new, intraClass=False)

                optimizer.zero_grad()
                logits = model(inputs_new)
                _, preds = torch.max(logits, 1)
                loss = criterion(logits, labels_new)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                Count_batch += 1

            if Is_Active and AugmentionType == 'GridMask':
                inputs_new = inputs.detach().clone()  # 拷贝出来新的张量副本 不影响原来的张量 挑选在原来的 inpus挑选，结果保存在 新的 inputs_new 避免相互影响
                labels_new = labels.detach().clone()
                num_sample += inputs_new.size(0)
                GridMask(inputs, inputs_new, labels_new)

                optimizer.zero_grad()
                logits = model(inputs_new)
                _, preds = torch.max(logits, 1)
                loss = criterion(logits, labels_new)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                Count_batch += 1

            if Is_Active and AugmentionType == 'DynamicGridMask':
                inputs_new = inputs.detach().clone()  # 拷贝出来新的张量副本 不影响原来的张量 挑选在原来的 inpus挑选，结果保存在 新的 inputs_new 避免相互影响
                labels_new = labels.detach().clone()
                num_sample += inputs_new.size(0)
                DynamicGridMask(inputs, inputs_new, labels_new)

                optimizer.zero_grad()
                logits = model(inputs_new)
                _, preds = torch.max(logits, 1)
                loss = criterion(logits, labels_new)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                Count_batch += 1

        #mixup(inputs, labels, inputs_new, labels_new,intraClass = True) # Mixup
        #CutMix(inputs, labels, inputs_new, labels_new, intraClass=True)  # CutMix
        #Mosaic4(inputs, labels, inputs_new, labels_new, intraClass=True)  # Mosaic4
        #GridMask(inputs, inputs_new)  # GridMask
        #DynamicGridMask(inputs, inputs_new)  # DynamicGridMask

    loss = running_loss / num_sample *  (1.0 / Count_batch ) #
    acc = running_corrects.double() / num_sample *  (1.0 / Count_batch )
    return loss, acc


def val_net(model, loader, criterion, device):
    with torch.no_grad():
        running_loss = 0.0
        running_corrects = 0
        num_sample = 0
        for inputs, labels, _ in loader:
            inputs = inputs.to(device)
            num_sample += inputs.size(0)
            labels = labels.to(device)
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


def eval_ood(model, known_loader, unknown_loader, device, method='max_logit'):
    model.eval()
    ##################################################################
    # for known
    ##################################################################
    correct, num_known, = 0, 0
    known_ood_scores = []
    with torch.no_grad():
        for data, labels, _ in known_loader:
            data = data.to(device)
            labels = labels.to(device)
            logits = model(data)
            probs = F.softmax(logits, dim=1)
            pred = logits.max(1)[1]
            num_known += data.size(0)
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
        for data, labels, _ in unknown_loader:
            data = data.to(device)
            logits = model(data)
            probs = F.softmax(logits, dim=1)
            num_unknown += data.size(0)
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

'''
Use mixup with a 100% probability  alpha = beta = 1
'''
def mixup(inputs, labels, inputs_new, labels_new, intraClass = True):
    for index, value in enumerate(inputs):
        # 使用随机函数选择一个键
        xj_index = random.randint(0,len(inputs) - 1 ) # [0,256)
        count = 0  # 初始化计数器
        #类内融合
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
        alpha = 1.0
        beta = 1.0
        Lambda = np.random.beta(alpha, beta)

        inputs_new[index] = Lambda * inputs[index] + (1 - Lambda) * inputs[xj_index]
        labels_new[index] = Lambda * labels[index] + (1 - Lambda) * labels[xj_index]

        # if intraClass :
        #     prefix = "intra-calssmixup"
        # else:
        #     prefix = "inter-calssmixup"
        #Save_ImageTensor_ToImage(inputs_new[index], labels_new[index], prefix )

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

        inputs_new[index], labels_new[index] =  Mosaic4_Pocess(image_tensor_list, labels, inputs[index].shape[-1], inputs[index].shape[-2], 0.25, 0.25)


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
        #rotated_pil_image = pil_image.rotate(angle, expand=True)  # expand=True 用于保持图像内容的完整性
        rotated_pil_image = pil_image.rotate(angle)  # expand=True 用于保持图像内容的完整性
    else:
        rotated_pil_image = pil_image

    target_size = (expected_w, expected_h)  #width x height
    resized_pil_image = rotated_pil_image.resize(target_size, Image.ANTIALIAS) # (width, height)
    # 转换 PIL 图像回 Tensor，并添加回批次维度
    to_tensor = ToTensor()
    resized_tensor = to_tensor(resized_pil_image)  #(1,7,7)

    return resized_tensor


'''
Use GridMask 
'''
def GridMask(inputs, inputs_new, labels_new):
    for index, value in enumerate(inputs):
        # 每张图的变量都随机生成
        d = random.randint(12, 28)  # 改成和原始论文一模一样 不再提前指定 d is the length of one unit.
        deta_y = random.randint(1, d - 1)
        deta_x = random.randint(1, d - 1)
        r = 0.4  # r: is the ratio of the shorter gray edge in a unit. 原文设置为 0.4
        probability = 1.0

        inputs_new[index] = OriginalGridmask(inputs_new[index], deta_x, deta_y, d, r, probability)

        Save_ImageTensor_ToImage( inputs_new[index], labels_new[index], "GridMask")


"""
#gridmask  和论文中的基本一致
deta_x, deta_y,r,d refence to the paper 
"""
def OriginalGridmask(image_tensor,deta_x, deta_y, d, r = 0.5,  prob=1.0):
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


    gridmask_image_tensor = image_tensor * expanded_mask

    return gridmask_image_tensor


'''
Use DynamicGridMask 
'''
def DynamicGridMask(inputs, inputs_new, labels_new):
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
        probability = 1.0

        inputs_new[index] = DynamicGridmask_oneImage(inputs_new[index], deta_x, deta_y, d, r, probability)

        #Save_ImageTensor_ToImage(inputs_new[index], labels_new[index], "DynamicGridMask")


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

    gridmask_image_tensor = image_tensor * expanded_mask

    return gridmask_image_tensor

'''
save processed image to bmp 
'''
def Save_ImageTensor_ToImage(image_tensor,label,str_prefix):
    # 去除单通道维度（这一步是可选的，但有助于后续处理）
    gray_tensor = image_tensor.squeeze(0)  # 现在形状为[28, 28]
    # 将Tensor转换为PIL Image
    to_pil = transforms.ToPILImage(mode='L')  # 'L'模式表示灰度图
    image = to_pil(gray_tensor)
    # 指定路径和文件名保存图片
    # 生成UUID
    unique_id = uuid.uuid4()
    # 转换为字符串并去除连字符
    image_name = str_prefix + str(label) + str(unique_id).replace('-', '') + '.bmp'
    image_full_path = os.path.join('D:/ImageTensor_Results/', image_name).replace('\\', '/')
    image.save(image_full_path)

