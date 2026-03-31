import numpy as np
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import torch

# code sorce  https://github.com/ZhenglinZhou/Data_Augmentation_Zoo_for_Object_Detection/blob/master/augmentation_zoo/MyGridMask.py
#2024年9月25日
# 首先完成原始的 gridmask

class GridMask_Object(object):
    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def __call__(self, sample):
        img, annots = sample['img'], sample['annot']
        if np.random.rand() > self.prob:
            return sample
        h = img.shape[0]
        w = img.shape[1]
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        # d = self.d
        #        self.l = int(d*self.ratio+0.5)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        #        mask = 1*(np.random.randint(0,3,[hh,ww])>0)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        if self.mode == 1:
            mask = 1 - mask
        mask = np.expand_dims(mask.astype(np.float), axis=2)
        mask = np.tile(mask, [1, 1, 3])
        if self.offset:
            offset = np.float(2 * (np.random.rand(h, w) - 0.5))
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask
        return {'img': img, 'annot': annots}

#from paper author
#https://github.com/dvlab-research/GridMask/blob/master/detection_grid/maskrcnn_benchmark/data/transforms/grid.py


class GridMask_NN(nn.Module):
    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.4, mode=1, prob=0.9):
        super(GridMask_NN, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch  # + 1.#0.5

    def forward(self, x):
        n, c, h, w = x.size()
        # 生成一个长度为256的随机tensor
        random_tensor = torch.rand(n)
        # 根据条件生成新的tensor，满足条件的设为1（True），不满足的设为0（False）
        probability = (random_tensor < self.prob)  # 或者使用.int()来得到整数类型的tensor
        count_of_ones = torch.sum(probability).item()  # 统计probability中1（即True）的元素个数,即有多少个需要 gridmask

        ds_min = int(h / 224 * 96)  # [224 min 96 max 224]
        # 生成 count_of_ones 个  超参
        ds = torch.from_numpy(np.random.randint(ds_min, h, size=count_of_ones))  # 改成和原始论文一模一样 不再提前指定 d is the length of one unit.
        deta_ys = torch.from_numpy(np.random.randint(0, ds - 1, size=count_of_ones))
        deta_xs = torch.from_numpy(np.random.randint(0, ds - 1, size=count_of_ones))
        rs = torch.full((count_of_ones,),  self.ratio)  # r: is the ratio of the shorter gray edge in a unit. 原文设置为 0.4

        #width_units = deta_xs + rs * ds  # rs * ds 逐个元素相乘

        x = x.view(-1, h, w)  #[256,112,112]
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        ls = []
        for d in ds:
            ls.append( min(max(int(d * self.ratio + 0.5), 1), d - 1))
        masks = torch.from_numpy(np.ones((count_of_ones, hh, ww), dtype=np.float32))  # 生成对应的 mask个数

        st_hs = np.random.randint(ds)
        st_ws = np.random.randint(ds)
        if self.use_h:
            for index,mask in enumerate(masks):
                for i in range(hh // ds[index]):
                    s = ds[index] * i + st_hs[index]
                    t = min(s + ls[index], hh)
                    mask[s:t, :] *= 0
        if self.use_w:
            for index, mask in enumerate(masks):
                for i in range(ww // ds[index]):
                    s = ds[index] * i + st_ws[index]
                    t = min(s + ls[index], ww)
                    mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask_new = []
        for index,mask in enumerate(masks):
            mask = Image.fromarray(np.uint8(mask))
            mask = mask.rotate(r)
            mask = np.asarray(mask)
            #        mask = 1*(np.random.randint(0,3,[hh,ww])>0)
            mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]
            if not mask.flags.writeable:
                mask = mask.copy()  # 复制数组以使其可写
            if torch.cuda.is_available():
                mask = torch.from_numpy(mask).float().cuda()
            else:
                mask = torch.from_numpy(mask).float()
            if self.mode == 1:
                mask = 1 - mask
            mask_new.append(mask)

        #将每个列表转换为 torch.Tensor，并添加一个新的维度以匹配[1, 112, 112]
        mask_new_tensors = [m.clone().detach().to(torch.float32) for m in mask_new]
        # 使用 torch.stack 将它们堆叠成一个新的张量  现在 mask_new_tensor 的形状是 [230, 1, 112, 112]
        mask_new_tensors = torch.stack(mask_new_tensors, dim=0)


        #mask_new = mask_new.expand_as(x)
        um_samples = mask_new_tensors.size(0)  # 或者 masks_expanded.shape[0]  第一维维度
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float().cuda()
            #x = x * masks + offset * (1 - masks)
            x[:um_samples] = x[:um_samples] * mask_new_tensors + offset * (1 - mask_new_tensors)
        else:
            x[:um_samples]  = x[:um_samples]  * mask_new_tensors
        x = x.unsqueeze(1)  # [256,1,112,112]

        # 进行切片和乘法
        #inputs[:um_samples] = inputs[:um_samples] * masks_expanded

        return x


'''
Use GridMask 依据batch  现场生成 mask 然后 参与运算
原始GridMask  Batch  image gridmask 256
inputs 256,1,112,112
labels 256个label
'''
def Batch_GridMask(inputs, labels, args):
    # 函数内执行
    h, w = inputs[0].shape[-1], inputs[0].shape[-2]
    # 256个概率
    prob = 0.9
    # 生成一个长度为256的随机tensor
    random_tensor = torch.rand(256)
    # 根据条件生成新的tensor，满足条件的设为1（True），不满足的设为0（False）
    probability = (random_tensor < prob) # 或者使用.int()来得到整数类型的tensor
    count_of_ones = torch.sum(probability).item()  # 统计probability中1（即True）的元素个数,即有多少个需要 gridmask

    ds_min = int(h/224 * 96)   # [224 min 96 max 224]
    # 生成 count_of_ones 个  超参
    ds = torch.from_numpy(np.random.randint(ds_min, h, size=count_of_ones))  # 改成和原始论文一模一样 不再提前指定 d is the length of one unit.
    deta_ys = torch.from_numpy(np.random.randint(0, ds - 1, size=count_of_ones))
    deta_xs = torch.from_numpy(np.random.randint(0, ds - 1, size=count_of_ones))
    rs = torch.full((count_of_ones,), 0.4)  # r: is the ratio of the shorter gray edge in a unit. 原文设置为 0.4

    width_units = deta_xs + rs * ds  # rs * ds 逐个元素相乘
    masks = torch.from_numpy(np.ones((count_of_ones, h, w), dtype=np.float32))  #生成对应的 mask个数

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
    masks_expanded = masks.unsqueeze(1) # [230,1,112,112]
    um_samples = masks_expanded.size(0)  # 或者 masks_expanded.shape[0]  第一维维度
    # 进行切片和乘法
    inputs[:um_samples] = inputs[:um_samples] * masks_expanded
    labels_new = F.one_hot(labels, num_classes=len(args.known_classes)).float()

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

