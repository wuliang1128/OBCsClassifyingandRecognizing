from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import torchvision
import torch
from matplotlib import pyplot as plt
import numpy as np
#失败

def myimshows(imgs, titles=False, fname="test.jpg", size=6):
    lens = len(imgs)
    fig = plt.figure(figsize=(size * lens, size))
    if titles == False:
        titles = "0123456789"
    for i in range(1, lens + 1):
        cols = 100 + lens * 10 + i
        plt.xticks(())
        plt.yticks(())
        plt.subplot(cols)
        if len(imgs[i - 1].shape) == 2:
            plt.imshow(imgs[i - 1], cmap='Reds')
        else:
            plt.imshow(imgs[i - 1])
        plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


def tensor2img(tensor, heatmap=False, shape=(224, 224)):
    np_arr = tensor.detach().numpy()  # [0]
    # 对数据进行归一化
    if np_arr.max() > 1 or np_arr.min() < 0:
        np_arr = np_arr - np_arr.min()
        np_arr = np_arr / np_arr.max()
    # np_arr=(np_arr*255).astype(np.uint8)
    if np_arr.shape[0] == 1:
        np_arr = np.concatenate([np_arr, np_arr, np_arr], axis=0)
    np_arr = np_arr.transpose((1, 2, 0))
    return np_arr


path = r"D:\\daxiang.jpg"
bin_data = torchvision.io.read_file(path)  # 加载二进制数据
img = torchvision.io.decode_image(bin_data) / 255  # 解码成CHW的图片
img = img.unsqueeze(0)  # 变成BCHW的数据，B==1; squeeze
input_tensor = torchvision.transforms.functional.resize(img, [224, 224])

# 对图像进行水平翻转，得到两个数据
input_tensors = torch.cat([input_tensor, input_tensor.flip(dims=(3,))], axis=0)

model = resnet50(pretrained=True)
target_layers = [model.layer4[-1]]  # 如果传入多个layer，cam输出结果将会取均值

# cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
with GradCAM(model=model, target_layers=target_layers, use_cuda=False) as cam:
    targets = [ClassifierOutputTarget(386), ClassifierOutputTarget(386)]  # 指定查看class_num为386的热力图
    # aug_smooth=True, eigen_smooth=True 使用图像增强是热力图变得更加平滑
    grayscale_cams = cam(input_tensor=input_tensors, targets=targets)  # targets=None 自动调用概率最大的类别显示
    for grayscale_cam, tensor in zip(grayscale_cams, input_tensors):
        # 将热力图结果与原图进行融合
        rgb_img = tensor2img(tensor)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        myimshows([rgb_img, grayscale_cam, visualization], ["image", "cam", "image + cam"])




