import os
import time

import os.path as osp
import numpy as np
from PIL import Image
import cv2
from torch.utils import data
import pickle
import torchvision.transforms.functional as TF
import random

Image.MAX_IMAGE_PIXELS = 2300000000


class MyTrainDataSet(data.Dataset):

    def __init__(self, root='', list_path='', max_iters=None, transform=None):
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.transform = transform

        for name in self.img_ids:
            img_name, label_name = name.split()
            img_file = osp.join(self.root, img_name)
            label_file = osp.join(self.root, label_name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name,
            })

        print("length of train set: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        # print(datafiles)
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        # label = cv2.imread(datafiles["label"], -1)
        # label = convert_to_msk(label)
        size = image.size
        name = datafiles["name"]

        # image, label = co_transforms(image, label)

        if self.transform is not None:
            image = self.transform(image)

        label = np.array(label)
        print(image.shape)
        return image, label.copy(), np.array(size), name


class MyValDataSet(data.Dataset):

    def __init__(self, root='', list_path='', transform=None):
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        self.transform = transform

        for name in self.img_ids:
            img_name, label_name = name.split()
            img_file = osp.join(self.root, img_name)
            label_file = osp.join(self.root, label_name)
            # print(label_file)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name,
            })

        print("length of Validation set: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        size = image.size
        name = datafiles["name"]

        # image, label = co_transforms(image, label)

        if self.transform is not None:
            image = self.transform(image)

        label = np.array(label)

        return image, label.copy(), np.array(size), name


class MyTestDataSet(data.Dataset):

    def __init__(self, root='', list_path='', transform=None):
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        self.transform = transform

        for name in self.img_ids:
            img_name = name.strip()
            # img_name, label_name = name.split()
            img_file = osp.join(self.root, img_name)
            # label_file = osp.join(self.root, label_name)
            self.files.append({
                "img": img_file,
                # "label": label_file,
                "name": name.split('/')[1].replace('.jpg', ''),
            })
        print("lenth of test set ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')

        size = image.size
        name = datafiles["name"]

        if self.transform is not None:
            image = self.transform(image)

        # return image.copy(), np.array(size), name
        return image, np.array(size), name


class MyTrainInform:
    """ To get statistical information about the train set, such as mean, std, class distribution.
        The class is employed for tackle class imbalance.
    """

    def __init__(self, data_dir='', classes=7, train_set_file="",
                 inform_data_file="", normVal=1.10):
        """
        Args:
           data_dir: directory where the dataset is kept
           classes: number of classes in the dataset
           inform_data_file: location where cached file has to be stored
           normVal: normalization value, as defined in ERFNet paper
        """
        self.data_dir = data_dir
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        self.train_set_file = train_set_file
        self.inform_data_file = inform_data_file

    def compute_class_weights(self, histogram):
        """to compute the class weights
        Args:
            histogram: distribution of class samples
        """
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))

    def readWholeTrainSet(self, fileName, train_flag=True):
        """to read the whole train set of current dataset.
        Args:
        fileName: train set file that stores the image locations
        trainStg: if processing training or validation data

        return: 0 if successful
        """
        global_hist = np.zeros(self.classes, dtype=np.float32)

        no_files = 0
        min_val_al = 0
        max_val_al = 0
        with open(self.data_dir + '/' + fileName, 'r') as textFile:
            # with open(fileName, 'r') as textFile:
            for line in textFile:
                # we expect the text file to contain the data in following format
                # <RGB Image> <Label Image>
                img_name, label_name = line.split()

                img_file = ((self.data_dir).strip() + '/' + img_name).strip()
                label_file = ((self.data_dir).strip() + '/' + label_name).strip()
                label_img = Image.open(label_file)
                label_img = np.array(label_img)
                # label_img = cv2.imread(label_file, -1)
                # label_img = convert_to_msk(label_img)
                unique_values = np.unique(label_img)
                max_val = max(unique_values)
                min_val = min(unique_values)

                max_val_al = max(max_val, max_val_al)
                min_val_al = min(min_val, min_val_al)

                if train_flag == True:
                    hist = np.histogram(label_img, self.classes, [0, self.classes - 1])
                    global_hist += hist[0]

                    rgb_img = cv2.imread(img_file)
                    # bgr -> rgb
                    self.mean[0] += np.mean(rgb_img[:, :, 2])
                    self.mean[1] += np.mean(rgb_img[:, :, 1])
                    self.mean[2] += np.mean(rgb_img[:, :, 0])

                    self.std[0] += np.std(rgb_img[:, :, 2])
                    self.std[1] += np.std(rgb_img[:, :, 1])
                    self.std[2] += np.std(rgb_img[:, :, 0])

                else:
                    print("we can only collect statistical information of train set, please check")

                if max_val > (self.classes - 1) or min_val < 0:
                    print('Labels can take value between 0 and number of classes.')
                    print('Some problem with labels. Please check. label_set:', unique_values)
                    print('Label Image ID: ' + label_file)
                no_files += 1

        # divide the mean and std values by the sample space size
        self.mean /= no_files * 255
        self.std /= no_files * 255

        # compute the class imbalance information
        self.compute_class_weights(global_hist)
        return 0

    def collectDataAndSave(self):
        """ To collect statistical information of train set and then save it.
        The file train.txt should be inside the data directory.
        """
        print('Processing training data')
        return_val = self.readWholeTrainSet(fileName=self.train_set_file)

        print('Pickling data')
        if return_val == 0:
            data_dict = dict()
            data_dict['mean'] = self.mean
            data_dict['std'] = self.std
            data_dict['classWeights'] = self.classWeights
            pickle.dump(data_dict, open(self.inform_data_file, "wb"))
            return data_dict
        return None


def gen_part_list_txt(t_path="img_train", m_path="lab_train", z_path="img_testA"):
    lst = []
    print("./remote/"+t_path)
    for name in os.listdir("./remote/"+t_path):
        img_path = t_path + "/" + name
        lab_path = m_path + "/" + name.replace('jpg', 'png')
        # print(img_path+"\n")
        # print(img_path + " " + lab_path+"\n")
        lst.append(img_path + " " + lab_path)
    test_lst = []
    for name in os.listdir("./remote/"+z_path):
        img_path = z_path + "/" + name
        # print(img_path+"\n")
        # print(img_path + " " + lab_path+"\n")
        test_lst.append(img_path)

    random.shuffle(lst)
    lst = lst[: 48000]
    len_lst = len(lst)
    print(len_lst)
    print(len(test_lst))
    m = 0.95
    with open('./remote/remote_train_list.txt', mode='w') as f:
        for x in lst[: int(m*len_lst)]:
            f.write(x + "\n")
    with open('./remote/remote_val_list.txt', mode='w') as f:
        for x in lst[int(m * len_lst):]:
            f.write(x + "\n")
    with open('./remote/remote_test_list.txt', mode='w') as f:
        for x in test_lst:
            f.write(x + "\n")
    with open('./remote/remote_trainval_list.txt', mode='w') as f:
        for x in lst:
            f.write(x + "\n")

def gen_whole_list_txt(t_path="img_train", m_path="lab_train", z_path="img_testA"):
    lst = []
    print("./remote/"+t_path)
    for name in os.listdir("./remote/"+t_path):
        img_path = t_path + "/" + name
        lab_path = m_path + "/" + name.replace('jpg', 'png')
        # print(img_path+"\n")
        # print(img_path + " " + lab_path+"\n")
        lst.append(img_path + " " + lab_path)
    test_lst = []
    for name in os.listdir("./remote/"+z_path):
        img_path = z_path + "/" + name
        # print(img_path+"\n")
        # print(img_path + " " + lab_path+"\n")
        test_lst.append(img_path)
    len_lst = len(lst)
    print(len_lst)
    print(len(test_lst))
    m = 0.99
    with open('./remote/remote_train_list.txt', mode='w') as f:
        for x in lst[: int(m*len_lst)]:
            f.write(x + "\n")
    with open('./remote/remote_val_list.txt', mode='w') as f:
        for x in lst[int(m * len_lst):]:
            f.write(x + "\n")
    with open('./remote/remote_test_list.txt', mode='w') as f:
        for x in test_lst:
            f.write(x + "\n")
    with open('./remote/remote_trainval_list.txt', mode='w') as f:
        for x in lst:
            f.write(x + "\n")


def gen_test_list_txt(z_path="img_train"):
    lst = []
    print("./remote/"+z_path)
    test_lst = []
    for name in os.listdir("./remote/"+z_path):
        img_path = z_path + "/" + name
        # print(img_path+"\n")
        # print(img_path + " " + lab_path+"\n")
        test_lst.append(img_path)
    len_lst = len(lst)
    print(len_lst)
    print(len(test_lst))
    m = 0.99
    with open('./remote/remote_testB_list.txt', mode='w') as f:
        for x in test_lst:
            f.write(x + "\n")


def convert_to_msk(img):
    palette = {
        # bgr
        "[64, 128, 0]": 2, # forest
        "[192, 128, 96]": 3, # water
        "[0, 128, 64]": 1, # filed
        "[64, 64, 96]": 6, # else
        "[192, 128, 32]": 4, # road
        "[0, 128, 96]": 0, # building
        "[64, 128, 64]": 5, # grass
        "[255, 255, 255]": 255,
    }
    rt = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            a, b, c = img[i, j]
            idx = f"[{a}, {b}, {c}]"
            rt[i, j] = palette[idx]
    return rt


def co_transforms(img, msk):
    if random.random() > 0.5:
        angle = random.randint(-30, 30)
        img = TF.rotate(img, angle)
        msk = TF.rotate(msk, angle)
    if random.random() > 0.5:
        if random.random() > 0.5:
            img = TF.hflip(img)
            msk = TF.hflip(msk)
    if random.random() > 0.5:
        if random.random() > 0.5:
            img = TF.vflip(img)
            msk = TF.vflip(msk)
    # more transforms ...
    return img, msk


if __name__ == "__main__":
    gen_test_list_txt()
    # gen_part_list_txt()
