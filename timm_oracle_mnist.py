from pathlib import Path
import sys

import matplotlib.pyplot as plt

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import torchvision
import numpy as np
import torch
import os
import random
import shutil
from copy import deepcopy
# from config import oracle_train_root, oracle_val_root


def split_train_val(root):
    dataset_dir = os.path.join(root).replace('\\', '/')  #解决windows 下 join 反斜杠问题
    val_dir = os.path.join(dataset_dir, 'val').replace('\\', '/')  #解决windows 下 join 反斜杠问题
    train_dir = os.path.join(dataset_dir, 'train').replace('\\', '/')  #解决windows 下 join 反斜杠问题
    class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for c_name in class_name:
        c_dir = os.path.join(dataset_dir, c_name).replace('\\', '/')  #解决windows 下 join 反斜杠问题
        save_dir = os.path.join(val_dir, c_name).replace('\\', '/')  #解决windows 下 join 反斜杠问题
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"make new folder{save_dir}")
        else:
            print("the path is existing")
        pathdir = os.listdir(c_dir)
        random.seed(10)
        file_number = len(pathdir)
        rate = 0.3
        picknumber = int(file_number * rate)
        samples = random.sample(pathdir, picknumber)
        list_len = len(samples)
        print(f"the number of val sample is {list_len}")
        for sample in samples:
            path_img = os.path.join(c_dir, sample).replace('\\', '/')  #解决windows 下 join 反斜杠问题
            shutil.move(path_img, save_dir)


def move_file(root):
    dataset_dir = os.path.join(root).replace('\\', '/')  #解决windows 下 join 反斜杠问题
    train_dir = os.path.join(dataset_dir, 'train').replace('\\', '/')  #解决windows 下 join 反斜杠问题
    class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for c_name in class_name:
        c_dir = os.path.join(dataset_dir, c_name).replace('\\', '/')  #解决windows 下 join 反斜杠问题
        save_dir = os.path.join(train_dir, c_name).replace('\\', '/')  #解决windows 下 join 反斜杠问题
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"make new folder{save_dir}")
        else:
            print("the path is existing")
        pathdir = os.listdir(c_dir)
        list_len = len(pathdir)
        print(f"the number of val sample is {list_len}")
        for sample in pathdir:
            path_img = os.path.join(c_dir, sample).replace('\\', '/')  #解决windows 下 join 反斜杠问题
            shutil.move(path_img, save_dir)


class OracleMNIST(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root, transform)
        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]
        return img, label, uq_idx

'''
增加代码  在windows 上碰到错误提示：
AttributeError: Can't pickle local object 'subsample_classes.<locals>.<lambda>'
替换掉 lambda 函数

class SubsampledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, cls_idxs, target_xform_dict):
        super(SubsampledDataset, self).__init__()
        self.dataset = dataset
        self.indices = cls_idxs
        self.target_xform_dict = target_xform_dict

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        data, target = self.dataset[sample_idx]
        transformed_target = self.target_xform_dict[target]
        return data, transformed_target
'''

def subsample_dataset(dataset, idxs):
    dataset.imgs = [x for i, x in enumerate(dataset.imgs) if i in idxs]
    dataset.samples = [x for i, x in enumerate(dataset.samples) if i in idxs]
    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


'''
#原来的代码
def subsample_classes(dataset, include_classes=range(10)):
    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)
    #dataset.target_transform = lambda x: target_xform_dict[x]
    # return dataset

    #dataset = SubsampledDataset(dataset, cls_idxs, target_xform_dict)
    set_target_transform(dataset, target_xform_dict)

    return dataset
'''
##################################################################
''' 
错误提示：AttributeError: Can't pickle local object 'subsample_classes.<locals>.<lambda>'
这个错误是由于在 Windows 环境下使用 Python 的 multiprocessing 模块时，无法序列化（pickle）局部定义的 lambda 函数。
在 Windows 系统中，multiprocessing 模块通过创建新的 Python 解释器进程来并行执行代码，而这些新进程需要能够访问并重新创建所有必要的对象和数据。
然而，lambda 函数作为局部作用域内的匿名函数，它们不能被序列化并传递给新的进程。

在你的代码中，你设置了 dataset.target_transform 为一个 lambda 函数，这个 lambda 函数引用了外部的 target_xform_dict 字典。
当 PyTorch 的 DataLoader 试图在多个进程中并行加载数据时，它尝试序列化这个 lambda 函数，导致失败。
'''
#使用类
class TargetTransformer:
    def __init__(self, target_xform_dict):
        self.target_xform_dict = target_xform_dict

    def __call__(self, x):
        return self.target_xform_dict[x]


def subsample_classes(dataset, include_classes=range(10)):
    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]


    # 创建一个字典，将原始类别标签映射到新的索引
    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)


    #dataset.target_transform = target_xform

    try:
        #dataset.target_transform = lambda x: target_xform_dict[x]
        # 创建一个 TargetTransformer 实例
        target_transformer = TargetTransformer(target_xform_dict)
        dataset.target_transform = target_transformer

    except KeyError as e:
        print(f"KeyError: {e} - 确保target_xform_dict包含所有必要的键")
    except Exception as e:
        print(f"An error occurred: {e}")

    #dataset.target_transform = lambda x: target_xform_dict[x] #windows 下查看数据流 注释掉  linux下运行 必须得有
    # lambda x: target_xform_dict[x] 这个lambda函数接受一个参数 x（在这里，x 代表数据集中的一个目标或标签），
    # 并返回 target_xform_dict 字典中键为 x 的值。这意味着，对于数据集中的每一个目标（或标签），这个转换都会查找 target_xform_dict 字典，
    # 找到与当前目标相对应的值，并将这个值作为转换后的目标返回。

    return dataset

#-----------修改半天没用上-----------
# 直接将定义的函数作为 target_transform   以下两个都有问题 没有真正赋值给 subsampled_dataset.target_transform
    #subsampled_dataset.target_transform = lambda x: transform_target(target_xform_dict, x)
def transform_target(target_xform_dict, target):
    return target_xform_dict[target]

    # 或者，避免使用 lambda，直接传递函数和字典
def set_target_transform(dataset, target_dict):
    transform = transform_target(target_dict,target)

    dataset.target_transform = transform

#*********以下两个函数没用到********
 # 假设您有一个函数可以基于索引来子采样数据集
# 这里只是一个示例，您需要根据实际情况来实现这个函数
def subsample_dataset_func(dataset, indices):
    # 示例实现，仅返回子采样的数据
    return SubsampledData(dataset, indices)
        # 使用假设的函数来子采样数据集
# 假设的 SubsampledData 类，用于表示子采样后的数据集
class SubsampledData(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        dataset.imgs = [x for i, x in enumerate(dataset.imgs) if i in indices]
        dataset.samples = [x for i, x in enumerate(dataset.samples) if i in indices]
        dataset.targets = np.array(dataset.targets)[indices].tolist()
        dataset.uq_idxs = dataset.uq_idxs[indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        data, target = self.dataset[sample_idx]
        if hasattr(self.dataset, 'target_transform') and self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)
        return data, target

###########################################################################

def get_train_val_split(train_dataset, val_split=0.2):
    val_dataset = deepcopy(train_dataset)
    train_dataset = deepcopy(train_dataset)

    train_classes = np.unique(train_dataset.targets)  #  train_dataset.targets 目前为空 所以报错[1 2 3 4 5 8]

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(train_dataset.targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    # Get training/validation datasets based on selected idxs
    train_dataset = subsample_dataset(train_dataset, train_idxs)
    val_dataset = subsample_dataset(val_dataset, val_idxs)

    return train_dataset, val_dataset


def get_equal_len_datasets(dataset1, dataset2):
    """
    Make two datasets the same length
    """

    if len(dataset1) > len(dataset2):

        rand_idxs = np.random.choice(range(len(dataset1)), size=(len(dataset2, )))
        subsample_dataset(dataset1, rand_idxs)

    elif len(dataset2) > len(dataset1):

        rand_idxs = np.random.choice(range(len(dataset2)), size=(len(dataset1, )))
        subsample_dataset(dataset2, rand_idxs)

    return dataset1, dataset2


def get_datasets(train_transform, test_transform,
                 train_root, val_root,
                 train_classes=range(10), open_set_classes=range(10), balance_open_set_eval=False, split_train_val=True,  #报错 target 8 i sour of  bounds 的处理
                 seed=0):
    np.random.seed(seed)

    # Init train dataset and subsample training classes
    train_dataset_whole = OracleMNIST(root=train_root, transform=train_transform)
    train_dataset_whole = subsample_classes(train_dataset_whole, include_classes=train_classes)

    # Split into training and validation sets
    train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
    val_dataset_split.transform = test_transform

    #增加代码，在此对 train_dataset_split 的图片进行 操作

    # Get test set for known classes
    test_dataset_known = OracleMNIST(root=val_root, transform=test_transform)
    test_dataset_known = subsample_classes(test_dataset_known, include_classes=train_classes)

    # Get test set for unknown classes
    test_dataset_unknown = OracleMNIST(root=val_root, transform=test_transform)
    test_dataset_unknown = subsample_classes(test_dataset_unknown, include_classes=open_set_classes)

    if balance_open_set_eval:
        test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)

    # Either split train into train and val or use test set as val
    train_dataset = train_dataset_split if split_train_val else train_dataset_whole
    val_dataset = val_dataset_split if split_train_val else test_dataset_known

    all_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test_known': test_dataset_known,
        'test_unknown': test_dataset_unknown,
    }

    return all_datasets


if __name__ == '__main__':
    # split_train_val('./datasets/PaddyDoctor10407/raw')
    # move_file('./datasets/PaddyDoctor10407/raw')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    x = get_datasets(transform, transform, balance_open_set_eval=False, split_train_val=False)
    print([len(v) for k, v in x.items()])

    #train_loader = torch.utils.data.DataLoader(x['train'], batch_size=2, num_workers=1)
    train_loader = torch.utils.data.DataLoader(x['train'], batch_size=2, num_workers=1)
    for data, target, idx in train_loader:
        print(data.size())
        print(data.min())
        print(data.max())
        plt.imshow(data[0].permute(1, 2, 0))
        plt.show()
        break
    debug = 0
