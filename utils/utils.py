import os
import zipfile
import random
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
from .colorize_mask import cityscapes_colorize_mask, camvid_colorize_mask, seed_colorize_mask
import shutil


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_predict(output, gt, img_name, dataset, save_path, output_grey=False, output_color=True, gt_color=False):
    if output_grey:
        if dataset == 'remote':
            # _, img_name = img_name.split('/')
            # f_path = os.path.join(save_path, img_name + '.png')
            # f_path = f_path.replace('*', '$')
            # cv2.imwrite(f_path, output, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            output_grey = Image.fromarray(output).convert("L")
            output_grey.save(os.path.join(save_path, img_name + '.png'))

            # f_path = os.path.join(save_path, img_name + '.png')
            # f_path = f_path.replace('*', '$')
            # output_grey.save(f_path)
        else:
            output_grey = Image.fromarray(output).convert("L")
            # output_grey.save(os.path.join(save_path, img_name + '.png'))

            f_path = os.path.join(save_path, img_name + '.png')
            f_path = f_path.replace('*', '$')
            output_grey.save(f_path)

    if output_color:
        if dataset == 'cityscapes':
            output_color = cityscapes_colorize_mask(output)
        elif dataset == 'camvid':
            output_color = camvid_colorize_mask(output)
        elif dataset == 'seed':
            output_color = seed_colorize_mask(output)

        # output_color.save(os.path.join(save_path, img_name + '_color.png'))

        f_path = os.path.join(save_path, img_name + '.png')
        f_path = f_path.replace('*', '$')
        output_color.save(f_path)

    if gt_color:
        if dataset == 'cityscapes':
            gt_color = cityscapes_colorize_mask(gt)
        elif dataset == 'camvid':
            gt_color = camvid_colorize_mask(gt)

        gt_color.save(os.path.join(save_path, img_name + '_gt.png'))


def zipDir(dirpath, outFullName):
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        print(dirpath)

        fpath = "results"  # 保留目录结构，可指定字符串
        # fpath = path.replace(dirpath, '') # 不保留

        for filename in filenames:
            zip.write(os.path.join(path, filename),
                      os.path.join(fpath, filename))
    zip.close()


def netParams(model):
    """
    computing total network parameters
    args:
       model: model
    return: the number of parameters
    """
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters


def binMask():
    f_path = 'F:\\ez\\Efficient-Segmentation-Networks-master\\dataset\\seed'
    dir_list = ['train', 'val', 'test']

    def BGR2GRAY(img):
        b = img[:, :, 0].copy()
        g = img[:, :, 1].copy()
        r = img[:, :, 2].copy()

        # Gray scale
        out = 0.2126 * r + 0.7152 * g + 0.0722 * b
        out = out.astype(np.uint8)

        return out

    # binalization

    def binarization(img, th=128):
        img[img < th] = 0
        img[img >= th] = 255
        return img

    for dir_name in dir_list:
        dir_name = f_path + dir_name
        for img in os.listdir(dir_name):
            if 'mask' in img:
                img_path = dir_name + '/' + img
                # img = cv2.imread(img_path)
                # print(img.shape)
                # if img.shape[0] > H_MAX:
                #     H_MAX = img.shape[0]
                # if img.shape[1] > W_MAX:
                #     W_MAX = img.shape[1]

                # Read image
                img = cv2.imread(img_path).astype(np.float32)

                # Grayscale
                out = BGR2GRAY(img)

                # Binarization
                out = binarization(out)

                # Save result
                cv2.imwrite(img_path, out)

def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


def split_dataset():
    '''
    split pos232 into 3 part and generate corresponding txt file
    :return:
    '''
    dataset_dir = os.path.join("..", "dataset", "seed", "pos732")
    train_dir = os.path.join("..", "dataset", "seed", "train")
    val_dir = os.path.join("..", "dataset", "seed", "val")
    test_dir = os.path.join("..", "dataset", "seed", "test")

    makedir(train_dir)
    makedir(val_dir)
    makedir(test_dir)

    train_per = 0.8
    valid_per = 0.1
    test_per = 0.1

    imgs_list = []
    for img in os.listdir(dataset_dir):
        if 'mask' not in img:
            img_path = dataset_dir + '/' + img
            imgs_list.append(img_path)

    random.seed(666)
    random.shuffle(imgs_list)

    imgs_num = len(imgs_list)
    train_point = int(imgs_num * train_per)
    valid_point = int(imgs_num * (train_per + valid_per))

    for i in range(imgs_num):
        if i < train_point:
            shutil.copy(imgs_list[i], train_dir)
            shutil.copy(imgs_list[i].replace('.jpg', '_mask.jpg'), train_dir)
        elif i < valid_point:
            shutil.copy(imgs_list[i], val_dir)
            shutil.copy(imgs_list[i].replace('.jpg', '_mask.jpg'), val_dir)
        else:
            shutil.copy(imgs_list[i], test_dir)
            shutil.copy(imgs_list[i].replace('.jpg', '_mask.jpg'), test_dir)


def generate_txt():
    '''
    generate txt file for dataset
    '''

    train_txt_path = os.path.join("..", "dataset", "seed", "train.txt")
    val_txt_path = os.path.join("..", "dataset", "seed", "val.txt")
    test_txt_path = os.path.join("..", "dataset", "seed", "test.txt")

    def gen_txt(txt_path, dataset_type):
        f = open(txt_path, 'w')
        # file_name = dataset_type + '.txt'
        dataset_dir = os.path.join("..", "dataset", "seed", dataset_type)

        for img in os.listdir(dataset_dir):
            if 'mask' not in img:
                img_path = dataset_type + '/' + img
                img_mask_path = dataset_type + '/' + img.replace('.jpg', '_mask.jpg')
                f.write(img_path + ' ' + img_mask_path + '\n')

        f.close()

    gen_txt(train_txt_path, 'train')
    gen_txt(val_txt_path, 'val')
    gen_txt(test_txt_path, 'test')


if __name__ == "__main__":
    split_dataset()
    generate_txt()
