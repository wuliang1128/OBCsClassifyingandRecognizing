import sys
import os
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#
#from config import *
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上级目录
parent_dir = os.path.dirname(current_dir)
# 将上级目录添加到 sys.path
sys.path.append(parent_dir)
# 现在你可以导入 config 模块了
from config import *

from torchvision import transforms
from timm_oracle_mnist import get_datasets  #把原来的 代码换成 timm的代码
from model.resnet import *
import itertools
#来自许老师

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def visualize_tsne_points(tx, ty, labels, save_path, epoch, flag, classes, dataset, unknown_in_one, split_idx):
    colors_per_class_old = get_colors_per_class(dataset)
    file_name = os.path.join(save_path, f"split{split_idx}_epoch{epoch}_" + flag + "_tsne_figure.jpg")
    colors_per_class = {}
    class_name = get_class_name(dataset)
    new_class_name = [class_name[x] for x in classes]
    for name in new_class_name:
        colors_per_class.update(
            {
                name: colors_per_class_old[name]
            }
        )
    if unknown_in_one:
        for name in new_class_name[-21:]:
            colors_per_class.update(
                {
                    name: [192, 192, 192]
                }
            )
    # initialize matplotlib plot
    fig = plt.figure(figsize=(10.4, 8.8))
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label, key in enumerate(colors_per_class):
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[key][::-1]], dtype=np.float64) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=key)

    # build a legend using the labels we set previously
    ax.legend(loc='upper right')

    # finally, show the plot
    # plt.show()
    plt.savefig(file_name, bbox_inches="tight", dpi=400)
    plt.clf()


def visualize_tsne(features, labels, save_dir, epoch, flag, classes, dataset, unknown_in_one=False, split_idx=0):
    tsne = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(features)
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    # visualize the plot: samples as colored points
    visualize_tsne_points(tx, ty, labels, save_dir, epoch, flag, classes, dataset, unknown_in_one, split_idx=split_idx)

def get_colors_per_class(dataset):
    colors_per_class_old = {}
    if dataset == 'paddy_rice':
        colors_per_class_old = {
            'BLB': [254, 202, 87],
            'BLS': [255, 107, 107],
            'BPB': [10, 189, 227],
            'BS': [255, 159, 243],
            'Blast': [16, 172, 132],
            'DH': [128, 80, 128],
            'DM': [87, 101, 116],
            'Healthy': [52, 31, 151],
            'Hispa': [100, 100, 255],
            'Tungro': [255, 0, 0],
        }
    elif dataset == "cifar-10-10":
        colors_per_class_old = {
            'airplane': [254, 202, 87],
            'automobile': [255, 107, 107],
            'bird': [10, 189, 227],
            'cat': [255, 159, 243],
            'deer': [16, 172, 132],
            'dog': [128, 80, 128],
            'frog': [87, 101, 116],
            'horse': [52, 31, 151],
            'ship': [100, 100, 255],
            'truck': [255, 0, 0],
        }
    elif dataset == "mnist" or dataset == "svhn":
        colors_per_class_old = {
            '0': [254, 202, 87],
            '1': [255, 107, 107],
            '2': [10, 189, 227],
            '3': [255, 159, 243],
            '4': [16, 172, 132],
            '5': [128, 80, 128],
            '6': [87, 101, 116],
            '7': [52, 31, 151],
            '8': [100, 100, 255],
            '9': [255, 0, 0],
        }
    elif dataset == "Oracle-MNIST":
        colors_per_class_old = {
            'big': [254, 202, 87],
            'sun': [255, 107, 107],
            'moon': [10, 189, 227],
            'cattle': [255, 159, 243],
            'next': [16, 172, 132],
            'field': [128, 80, 128],
            'not': [87, 101, 116],
            'arrow': [52, 31, 151],
            'time': [100, 100, 255],
            'wood': [255, 0, 0],
        }
    # elif dataset == "":


    return colors_per_class_old


def get_class_name(dataset):
    if dataset == 'cifar-10-10':
        class_name = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    elif dataset == 'mnist':
        class_name = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    elif dataset == 'paddy_rice':
        class_name = ['BLB', 'BLS', 'BPB', 'BS', 'Blast', 'DH', 'DM', 'Healthy', 'Hispa', 'Tungro']
    elif dataset == 'Oracle-MNIST':
        class_name = ["big", "sun", "moon", "cattle", "next", "field", "not", "arrow", "time", "wood"]
    else:
        raise NotImplemented("please check the dataset name")

    return class_name

def load_OracleMNIST():
    input_size =28
    data_dir = "D:/deeplearningsoft/PycharmProjects/OracleMNIST/data/oralce-mnist-size224"
    #data_dir = "../data/oracle-mnist-size224"
    known_classes = OrcaleMNIST_split[0][0]
    unknown_classes = OrcaleMNIST_split[0][1]
    all_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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
        #train_classes=known_classes, open_set_classes=unknown_classes,
        train_classes=all_classes, open_set_classes=all_classes, #10分类用
        split_train_val=False
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
    test_known_loader = data_loader['test_known']
    test_unknown_loader = data_loader['test_unknown']

    return train_loader, val_loader, test_known_loader, test_unknown_loader
    # # 假设 train_loader 是一个已经定义好的 DataLoader 实例
    # all_train_data_and_labels = list(train_loader)  # 这将返回一个包含多个 (data, label) 元组的列表
    # # 如果你想分别获取所有的数据和标签，可以这样做：
    # train_data = [item[0] for item in all_train_data_and_labels]  # 获取所有的数据
    # train_labels = [item[1] for item in all_train_data_and_labels]  # 获取所有的标签
    # #注意：这里假设所有张量的形状除了第一个维度外都是相同的
    # train_concatenated_data = torch.cat(train_data, dim=0)  #[16128,1,28,28]
    # #train_concatenated_data = train_concatenated_data.squeeze(1)
    # train_concatenated_labels = torch.cat(train_labels, dim=0)  #6类

    # # 假设 val_loader 是一个已经定义好的 DataLoader 实例
    # all_val_data_and_labels = list(val_loader)  # 这将返回一个包含多个 (data, label) 元组的列表
    # # 如果你想分别获取所有的数据和标签，可以这样做：
    # val_data = [item[0] for item in all_val_data_and_labels]  # 获取所有的数据
    # val_labels = [item[1] for item in all_val_data_and_labels]  # 获取所有的标签
    # # 注意：这里假设所有张量的形状除了第一个维度外都是相同的
    # val_concatenated_data = torch.cat(val_data, dim=0)  # [16128,1,28,28]
    # # train_concatenated_data = train_concatenated_data.squeeze(1)
    # val_concatenated_labels = torch.cat(val_labels, dim=0)  # 6类
    #
    # # 假设 test_loader 是一个已经定义好的 DataLoader 实例
    # all_test_data_and_labels = list(test_loader)  # 这将返回一个包含多个 (data, label) 元组的列表
    # # 如果你想分别获取所有的数据和标签，可以这样做：
    # test_data = [item[0] for item in all_test_data_and_labels]  # 获取所有的数据
    # test_labels = [item[1] for item in all_test_data_and_labels]  # 获取所有的标签
    # test_concatenated_data = torch.cat(test_data, dim=0) #[1200,1,28,28]
    # #test_concatenated_data = test_concatenated_data.squeeze(1) #[3065,28,28]
    # test_concatenated_labels = torch.cat(test_labels, dim=0) #6类
    #
    # # 使用 torch.unique 函数来获取唯一值，并返回唯一值的数量
    # unique_labels = torch.unique(val_concatenated_labels)
    # num_classes = len(unique_labels)
    # print(f"Number of unique train_concatenated_data classes: {num_classes}")
    #
    # #return (train_concatenated_data, train_concatenated_labels), (val_concatenated_data, val_concatenated_labels ), (test_concatenated_data, test_concatenated_labels)
    # return  (val_concatenated_data, val_concatenated_labels), (test_concatenated_data, test_concatenated_labels)


'''返回模型 # 加载训练好的权重'''
def load_rest18feature_model(rest18_model_type):
    # base = 'D:/PycharmProjects/CAM-master/obi_pth/Batch_base_new/split_0_latest.pth'
    #
    # Batch_mixup_intra_new = 'D:/PycharmProjects/CAM-master/obi_pth/Batch_mixup_intra_new/split_0_latest.pth'
    # Batch_cutmix_intra_new = 'D:/PycharmProjects/CAM-master/obi_pth/Batch_cutmix_intra_new/split_0_latest.pth'
    # Batch_dynamic_new = 'D:/PycharmProjects/CAM-master/obi_pth/Batch_dynamic_new/split_0_latest.pth'
    #
    # Batch_combination12_intra_new = 'D:/PycharmProjects/CAM-master/obi_pth/Batch_combination12_intra_new/split_0_latest.pth'
    # Batch_combination23_intra_new = 'D:/PycharmProjects/CAM-master/obi_pth/Batch_combination23_intra_new/split_0_latest.pth'
    # Batch_combination13_intra_new = 'D:/PycharmProjects/CAM-master/obi_pth/Batch_combination13_intra_new/split_0_latest.pth'
    #
    # Batch_combination333_intra_new = 'D:/PycharmProjects/CAM-master/obi_pth/Batch_combination333_intra_new/split_0_latest.pth'
    # Batch_combination224_intra_new = 'D:/PycharmProjects/CAM-master/obi_pth/Batch_combination224_intra_new/split_0_latest.pth'

    base = "../ckpt_pth_224/Batch_base_new/split_0_latest.pth"

    Batch_mixup_intra_new = '../ckpt_pth_224/Batch_mixup_intra_new/split_0_latest.pth'
    Batch_cutmix_intra_new = '../ckpt_pth_224/Batch_cutmix_intra_new/split_0_latest.pth'
    Batch_dynamic_new = '../ckpt_pth_224/Batch_dynamic_new/split_0_latest.pth'

    Batch_combination12_intra_new = '../ckpt_pth_224/Batch_combination12_intra_new/split_0_latest.pth'
    Batch_combination23_intra_new = '../ckpt_pth_224/Batch_combination23_intra_new/split_0_latest.pth'
    Batch_combination13_intra_new = '../ckpt_pth_224/Batch_combination13_intra_new/split_0_latest.pth'

    Batch_combination333_intra_new = '../ckpt_pth_224/Batch_combination333_intra_new/split_0_latest.pth'
    Batch_combination224_intra_new = '../ckpt_pth_224/Batch_combination224_intra_new/split_0_latest.pth'

    pth_sets_dice = {
        '01base': base,

        '02mixup_intra_new': Batch_mixup_intra_new,
        '03cutmix_intra_new': Batch_cutmix_intra_new,
        '04dynamic_new': Batch_dynamic_new,

        '05combination12_intra_new': Batch_combination12_intra_new,
        '06combination23_intra_new': Batch_combination23_intra_new,
        '07combination13_intra_new': Batch_combination13_intra_new,

        '08combination333_intra_new': Batch_combination333_intra_new,
        '09combination224_intra_new': Batch_combination224_intra_new
    }

    # 如果你有训练好的模型权重文件（例如 'resnet18_trained.pth'），你可以加载它
    # 遍历字典，找到匹配的键
    for key in pth_sets_dice:
        if key.startswith(rest18_model_type):
            model_path = pth_sets_dice[key]
            break  # 找到第一个匹配的键后退出循环
    else:
        # 如果没有找到匹配的键，value 将保持未定义或设置一个默认值
        model_path = pth_sets_dice['01base']

    # 检查GPU是否可用
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{0}")
    # 实例化模型
    model = ResNet18(num_c=10).to(device)
    # 加载状态字典
    model_instance = torch.load(model_path, map_location=device)

    return model_instance


if __name__ == '__main__':
    #准备输入数据 形状为 (concat_batch_size,channels, height, width )
    #(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_OracleMNIST()
    # (X_val, y_val), (X_test, y_test) = load_OracleMNIST()
    # print(f"X_train.shape:{X_val.shape}")
    train_loader, val_loader, test_known_loader, test_unknown_loader = load_OracleMNIST()

    tsne_models_choice = {
        '01base',

        '02mixup_intra_new',
        '03cutmix_intra_new',
        '04dynamic_new',

        '05combination12_intra_new',
        '06combination23_intra_new',
        '07combination13_intra_new',

        '08combination333_intra_new',
        '09combination224_intra_new'
    }

    for item in tsne_models_choice:
        # 加载模型  预加载训练好的模型
        model = load_rest18feature_model(item)

        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{0}")

        features_layer4_lists = []
        labels_layer4_lists = []

        # for inputs, labels, _ in test_known_loader:
        #     inputs = inputs.to(device)
        #     labels = labels.to(device)
        #     if inputs.size()[-1] != 28:
        #         size = [28, 28]
        #         inputs = F.interpolate(inputs, size=size, mode='bilinear', align_corners=False)
        #
        #     features_layer4, y = model(inputs, return_feature=True)  # 现在，当你进行前向传播时，你可以获取第4层的特征
        #
        #     features_layer4_lists.append(features_layer4)
        #     labels_layer4_lists.append(labels)

        for inputs, labels, _ in test_unknown_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if inputs.size()[-1] != 28:
                size = [28, 28]
                inputs = F.interpolate(inputs, size=size, mode='bilinear', align_corners=False)
            features_layer4, y = model(inputs, return_feature=True)  # 现在，当你进行前向传播时，你可以获取第4层的特征
            features_layer4_lists.append(features_layer4)
            labels_layer4_lists.append(labels)


        features_layer4_all = torch.cat(features_layer4_lists, dim=0)  # [16128,1,28,28]
        labels_layer4_all = torch.cat(labels_layer4_lists, dim=0)  # [16128,1,28,28]
        # 使用 torch.unique 去除重复项
        unique_labels = torch.unique(labels_layer4_all)
        # visualize_tsne(features_known, labels_known, args.save_root, epoch, "known",
        #                args.train_classes, args.dataset, split_idx=args.split_idx)
        #save_path = 'D:/PycharmProjects/CAM-master/obi_pth/tsne/tsne—test_unknown_loader/'
        save_path = '../ckpt_pth_224/tsne/tsne—test_unknown_loader/'
        train_classes = OrcaleMNIST_split[0][0] + OrcaleMNIST_split[0][1]
        dataset_class = "Oracle-MNIST"
        split_idx = 0
        flag = "test_" + item
        print("flag:{0}".format(flag))
        # 将特征列表转换为NumPy数组（这里只有一个特征向量，但为了匹配TSNE的输入格式，我们仍然这样做）
        #features_np = features_layer4_all.detach().numpy()  LINUX报错
        features_layer4_all_cpu = features_layer4_all.detach().cpu()  # 先detach()以避免计算图跟踪，然后调用cpu()移动到CPU
        features_np = features_layer4_all_cpu.numpy()  # 现在可以安全地转换为NumPy数组了

        visualize_tsne(features_np, labels_layer4_all, save_path, 200, flag,
                       train_classes, dataset_class, split_idx=split_idx)


    # 将模型设置为评估模式
    #model.eval()





