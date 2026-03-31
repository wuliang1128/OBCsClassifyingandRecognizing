#加载  数据
import time
import numpy as np
import pandas as pd
from torchvision import models, transforms
import os
from timm_oracle_mnist import get_datasets  #把原来的 代码换成 timm的代码
from config import *
import torch
#导入 PCA 和可视化所需的库
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


#有点问题  还没修正 应该是不正确的


def load_OracleMNIST():
    input_size =28
    data_dir = "D:/PycharmProjects/OracleMNIST/data/oralce-mnist-size224"
    known_classes = OrcaleMNIST_split[0][0]
    unknown_classes = OrcaleMNIST_split[0][1]
    batch_size = 256
    num_workers = 4
    data_generator = torch.Generator()

    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(15),
        transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_root = os.path.join(data_dir, 'train').replace('\\', '/')  # 解决windows 下 join 反斜杠问题
    val_root = os.path.join(data_dir, 'test').replace('\\', '/')
    print("train_root: ", train_root)  # 自增 看变量
    print("val_root: ", val_root)  # 自增 看变量 目前没问题了

    datasets = get_datasets(
        train_transform, val_transform, train_root, val_root,
        train_classes=known_classes, open_set_classes=unknown_classes,
        split_train_val = False
    )

    data_loader = {}
    for k, v, in datasets.items():
        shuffle = True if k == 'train' else False
        drop_last = True if k == 'train' else False
        data_loader[k] = torch.utils.data.DataLoader(
            v, batch_size=batch_size, shuffle=shuffle,
            sampler=None, num_workers=num_workers, pin_memory=True,
            prefetch_factor=4,
            drop_last=drop_last,
            generator=data_generator
        )

    train_loader = data_loader['train']
    val_loader = data_loader['val']

    # 假设 train_loader 是一个已经定义好的 DataLoader 实例
    all_data_and_labels = list(train_loader)  # 这将返回一个包含多个 (data, label) 元组的列表
    # 如果你想分别获取所有的数据和标签，可以这样做：
    train_data = [item[0] for item in all_data_and_labels]  # 获取所有的数据
    train_labels = [item[1] for item in all_data_and_labels]  # 获取所有的标签
    #注意：这里假设所有张量的形状除了第一个维度外都是相同的
    train_concatenated_data = torch.cat(train_data, dim=0)
    train_concatenated_data = train_concatenated_data.squeeze(1)
    train_concatenated_labels = torch.cat(train_labels, dim=0)  #6类
    # 使用 torch.unique 函数来获取唯一值，并返回唯一值的数量
    # unique_labels = torch.unique(train_concatenated_labels)
    # num_classes = len(unique_labels)
    # print(f"Number of unique classes: {num_classes}")

    all_data_and_labels = list(val_loader)  # 这将返回一个包含多个 (data, label) 元组的列表
    # 如果你想分别获取所有的数据和标签，可以这样做：
    test_data = [item[0] for item in all_data_and_labels]  # 获取所有的数据
    test_labels = [item[1] for item in all_data_and_labels]  # 获取所有的标签
    test_concatenated_data = torch.cat(test_data, dim=0) #[3065,1,28,28]
    test_concatenated_data = test_concatenated_data.squeeze(1)
    test_concatenated_labels = torch.cat(test_labels, dim=0) #6类

    train_concatenated_data = torch.cat((train_concatenated_data, test_concatenated_data), dim=0)
    train_concatenated_labels = torch.cat((train_concatenated_labels, test_concatenated_labels), dim=0)  #6 类



    return (train_concatenated_data, train_concatenated_labels), (test_concatenated_data, test_concatenated_labels)


if __name__ == '__main__':
    #获取 MNIST 训练和测试数据并检查训练数据的形状
    # 加载 MNIST 数据集

    (X_train, y_train) , (X_test, y_test) = load_OracleMNIST()
    X_train.shape

    #创建一个包含多个图像和图像中的像素数的数组，并将 X_train 数据复制到 X
    X = np.zeros((X_train.shape[0], 784))
    for i in range(X_train.shape[0]):
        X[i] = X_train[i].flatten()

    #打乱数据集，取出 10% 的 MNIST 训练数据并将其存储在数据框中。
    X = pd.DataFrame(X)
    Y = pd.DataFrame(y_train)
    X = X.sample(frac=0.1, random_state=10).reset_index(drop=True)
    Y = Y.sample(frac=0.1, random_state=10).reset_index(drop=True)
    df = X

    #使用 sklearn.decomposition 中的 PCA 库应用 PCA。
    time_start = time.time()
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(df.values)
    print('PCA done! Time elapsed: {} seconds'.format(time.time() - time_start))
    #PCA 生成两个维度，主成分 1 和主成分 2。将两个 PCA 成分与标签一起添加到数据框中。
    pca_df = pd.DataFrame(data=pca_results
                          , columns=['pca_1', 'pca_2'])
    pca_df['label'] = Y
    #绘制 PCA 结果
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    # Y 有几类就是几个颜色和标签
    all_colors = ['yellow', 'black', 'cyan', 'green', 'blue', 'red', 'brown', 'crimson', 'gold', 'indigo']
    all_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y_labels = np.unique(Y)
    # 使用这些唯一标签作为索引来从 all_colors 中选择相应的颜色
    colors = [all_colors[label] for label in y_labels]
    targets = y_labels

    for target, color in zip(targets, colors):
        indicesToKeep = pca_df['label'] == target
        ax.scatter(pca_df.loc[indicesToKeep, 'pca_1']
                   , pca_df.loc[indicesToKeep, 'pca_2']
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    # 显示图形
    plt.show()

    print('result:')







