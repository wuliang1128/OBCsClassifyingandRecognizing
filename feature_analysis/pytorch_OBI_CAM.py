# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
# last update by BZ, June 30, 2021

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import io
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json
from model.resnet import *

import sys
print(sys.path)

def load_all_calss():
    LABELS_file = 'imagenet-obi-labels.json'
    # load the imagenet category list
    with open(LABELS_file) as f:
        classes = json.load(f)
    return classes


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


# normalize = transforms.Normalize(
#    mean=[0.485, 0.456, 0.406],
#    std=[0.229, 0.224, 0.225]
# )
# preprocess = transforms.Compose([
#    transforms.Resize((224,224)),
#    transforms.ToTensor(),
#    normalize
# ])


def generate_CAM(model_path_param = None, image_full_path = None, true_label_str = None ):
    # input image
    # LABELS_file = 'imagenet-simple-labels.json'
    # image_file = 'test.jpg'

    if image_full_path is None:
        image_file = "D:/PycharmProjects/CAM-master/obi_pth/jpg/cattle/mixup_intra_class_tensor([0., 0., 0., 0., 1., 0.].jpg"
    else:
        image_file = image_full_path

    # networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
    model_id = 2
    if model_id == 1:
        net = models.squeezenet1_1(pretrained=True)
        finalconv_name = 'features'  # this is the last conv layer of the network
    elif model_id == 2:
        # net = models.resnet18(pretrained=True)
        net = ResNet18(6)
        finalconv_name = 'layer4'
    elif model_id == 3:
        net = models.densenet161(pretrained=True)
        finalconv_name = 'features'

    # # 修改最后一层以适应10个输出类别
    # num_ftrs = net.fc.in_features
    # net.fc = torch.nn.Linear(num_ftrs, 10)

    # 如果你有训练好的模型权重文件（例如 'resnet18_trained.pth'），你可以加载它
    if model_path_param is None:
        model_path = 'D:/PycharmProjects/CAM-master/obi_pth/Batch_mixup_intra_new/split_0_latest.pth'
    else:
        model_path = model_path_param

    # 加载状态字典
    model_instance = torch.load(model_path, map_location=torch.device('cpu'))  # 如果你有 GPU，可以用 'cuda' 替换 'cpu'
    state_dict = model_instance.state_dict()
    # 检查 state_dict 的类型（调试用）
    print(type(state_dict))  # 应该输出 <class 'dict'>
    # 将状态字典加载到模型中
    net.load_state_dict(state_dict)

    net.eval()

    # hook the feature extractor
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    net._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    preprocess = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomRotation(15),
        transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # load test image
    img_pil = Image.open(image_file)
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = net(img_variable)

    classes = load_all_calss()

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # output the prediction
    for i in range(0, 6):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
        if( classes[idx[i]] == true_label_str):
            CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[i]])

    return  CAMs


def save_CAM_image(CAMs,output_image_path=None, output_image_CAM_path=None ):
    # render the CAM and output
    # img = cv2.imread("combination224_intra_tensor([0., 0., 0., 0., 0., 1.], device='cuda_0', dtype=torch.float64)_0895e930045e4062bbc5d5a8659ba84b.jpg")
    if output_image_path is None:
        img = cv2.imread(
            "D:/PycharmProjects/CAM-master/obi_pth/jpg/cattle/mixup_intra_class_tensor([0., 0., 0., 0., 1., 0.].jpg")
    else:
        img = cv2.imread(output_image_path)

    if output_image_CAM_path is None:
        img_savename = "D:/PycharmProjects/CAM-master/obi_pth/CAM/" + "OBI_test.jpg"
    else:
        img_savename = output_image_CAM_path

    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite(img_savename, result)


if __name__ == '__main__':
    base ='D:/PycharmProjects/CAM-master/obi_pth/Batch_base_new/split_0_latest.pth'

    Batch_mixup_intra_new = 'D:/PycharmProjects/CAM-master/obi_pth/Batch_mixup_intra_new/split_0_latest.pth'
    Batch_cutmix_intra_new = 'D:/PycharmProjects/CAM-master/obi_pth/Batch_cutmix_intra_new/split_0_latest.pth'
    Batch_dynamic_new = 'D:/PycharmProjects/CAM-master/obi_pth/Batch_dynamic_new/split_0_latest.pth'


    Batch_combination12_intra_new = 'D:/PycharmProjects/CAM-master/obi_pth/Batch_combination12_intra_new/split_0_latest.pth'
    Batch_combination23_intra_new = 'D:/PycharmProjects/CAM-master/obi_pth/Batch_combination23_intra_new/split_0_latest.pth'
    Batch_combination13_intra_new = 'D:/PycharmProjects/CAM-master/obi_pth/Batch_combination13_intra_new/split_0_latest.pth'

    Batch_combination333_intra_new = 'D:/PycharmProjects/CAM-master/obi_pth/Batch_combination333_intra_new/split_0_latest.pth'
    Batch_combination224_intra_new = 'D:/PycharmProjects/CAM-master/obi_pth/Batch_combination224_intra_new/split_0_latest.pth'



    pth_sets_dice = {
        '01base': base,

        '02mixup_intra_new': Batch_mixup_intra_new,
        '03cutmix_intra_new':Batch_cutmix_intra_new,
        '04dynamic_new':Batch_dynamic_new,

        '05combination12_intra_new':Batch_combination12_intra_new,
        '06combination23_intra_new': Batch_combination23_intra_new,
        '07combination13_intra_new':Batch_combination13_intra_new,

        '08combination333_intra_new':Batch_combination333_intra_new,
        '09combination224_intra_new': Batch_combination224_intra_new
    }
    #image_full_path = "D:/PycharmProjects/OracleMNIST/data/oralce-mnist-size224/test/6/big_same_label_CutMixup1.bmp"
    #image_full_path = "D:/PycharmProjects/OracleMNIST/data/oralce-mnist-size224/train_IntraMixup/0/002000_d00110 a_1_IntraMixup.bmp"
    image_full_path = r'D:\PycharmProjects\OracleMNIST\data\oralce-mnist-size224\mixup_2\002000_h01473_1_IntraCutMix.bmp'

    true_label_str = "big"
    # 检查文件是否存在
    if os.path.exists(image_full_path):
        img = cv2.imread(image_full_path)
        if img is not None:
            print("图像加载成功")
            # 显示图像（可选）
            cv2.imshow('Loaded Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("图像文件无法被OpenCV读取，可能是文件损坏或格式不支持（尽管这里是BMP格式，但还是要检查）")
    else:
        print(f"文件路径不存在：{image_full_path}。请检查路径是否正确。")


    for k,model_path_param in pth_sets_dice.items():

        CAMs = generate_CAM(model_path_param, image_full_path, true_label_str) #  label_str 图片的真实标签

        #output_image_CAM_path = "D:/PycharmProjects/CAM-master/obi_pth/CAM/TestImage_CAM/IntraMixup_" + k + "_CAM.jpg"
        output_image_CAM_path = "D:/PycharmProjects/OracleMNIST/data/oralce-mnist-size224/mixup_2/big_IntraCutMix_" + k + "_CAM.jpg"

        save_CAM_image(CAMs, image_full_path, output_image_CAM_path) #保存cam图


'''
OrcaleMNIST_split = [
        #[known clsses], [unknown classes]
        [[2, 4, 8, 1, 3, 5], [7, 9, 6, 0]],
        [[8, 0, 7, 9, 1, 3], [6, 2, 4, 5]],
        [[0, 7, 1, 8, 5, 9], [6, 4, 3, 2]],
        [[1, 3, 5, 7, 8, 4], [2, 0, 9, 6]],
        [[5, 0, 6, 2, 8, 4], [3, 9, 1, 7]],
]
[
"moon",
"next",
"time",
"sun",
"cattle",
"field"
]

[
"big",
"sun",
"moon",
"cattle",
"next",
"field",
"not",
"arrow",
"time",
"wood"
]

'''