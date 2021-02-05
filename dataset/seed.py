import cv2
import numpy as np
import os
import time

import os.path as osp
import numpy as np
from PIL import Image
import random
import cv2
from torch.utils import data
import pickle
import random

Image.MAX_IMAGE_PIXELS = 2300000000


class SeedDataSet(data.Dataset):

    def __init__(self, root='', list_path='', max_iters=None, transform=None):
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.transform = transform

        # img_name, str(y0), str(y1), str(x0), str(x1)))
        for name in self.img_ids:
            img_name, msk_name, y0, y1, x0, x1 = name.split()
            img_file = osp.join(self.root, img_name)
            msk_file = osp.join(self.root, msk_name)
            self.files.append({
                "img": img_file,
                "mask": msk_file,
                "name": name,
                "y0": int(y0),
                "y1": int(y1),
                "x0": int(x0),
                "x1": int(x1)
            })

        print("length of train set: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        image = image.crop((datafiles["x0"], datafiles["y0"], datafiles["x1"], datafiles["y1"]))
        mask = cv2.imread(datafiles["mask"],
                          cv2.IMREAD_GRAYSCALE)[datafiles["y0"]: datafiles["y1"], datafiles["x0"]: datafiles["x1"]]
        size = image.size
        name = datafiles["name"]

        if self.transform is not None:
            image = self.transform(image)

        mask[mask < 128] = 0
        mask[mask > 128] = 1

        return image, mask.copy(), np.array(size), name


class SeedValDataSet(data.Dataset):

    def __init__(self, root='', list_path='', transform=None):
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        self.transform = transform

        for name in self.img_ids:
            img_name, msk_name, y0, y1, x0, x1 = name.split()
            img_file = osp.join(self.root, img_name)
            msk_file = osp.join(self.root, msk_name)
            # print(label_file)
            self.files.append({
                "img": img_file,
                "mask": msk_file,
                "name": name,
                "y0": int(y0),
                "y1": int(y1),
                "x0": int(x0),
                "x1": int(x1)
            })

        print("length of Validation set: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        image = image.crop((datafiles["x0"], datafiles["y0"], datafiles["x1"], datafiles["y1"]))
        mask = cv2.imread(datafiles["mask"],
                          cv2.IMREAD_GRAYSCALE)[datafiles["y0"]: datafiles["y1"], datafiles["x0"]: datafiles["x1"]]
        size = image.size
        name = datafiles["name"]

        if self.transform is not None:
            image = self.transform(image)

        mask[mask < 128] = 0
        mask[mask > 128] = 1

        return image, mask.copy(), np.array(size), name


class SeedTestDataSet(data.Dataset):

    def __init__(self, root='', list_path='', transform=None):
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        self.transform = transform

        for name in self.img_ids:
            img_name, msk_name, y0, y1, x0, x1 = name.split()
            img_file = osp.join(self.root, img_name)
            msk_file = osp.join(self.root, msk_name)
            self.files.append({
                "img": img_file,
                "mask": msk_file,
                "name": f"{img_file}_{y0}_{y1}_{x0}_{x1}",
                "y0": int(y0),
                "y1": int(y1),
                "x0": int(x0),
                "x1": int(x1)
            })
        print("lenth of test set ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        image = image.crop((datafiles["x0"], datafiles["y0"], datafiles["x1"], datafiles["y1"]))

        size = image.size
        name = datafiles["name"]

        if self.transform is not None:
            image = self.transform(image)

        # return image.copy(), np.array(size), name
        return image, np.array(size), name


class SeedTrainInform:
    """ To get statistical information about the train set, such as mean, std, class distribution.
        The class is employed for tackle class imbalance.
    """

    def __init__(self, data_dir='', classes=2, train_set_file="",
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
                img_name, msk_name, y0, y1, x0, x1 = line.split()
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

                img_file = ((self.data_dir).strip() + '/' + img_name).strip()
                label_file = ((self.data_dir).strip() + '/' + msk_name).strip()

                label_img = cv2.imread(label_file, 0)[y0: y1, x0: x1]
                label_img[label_img > 128] = 1
                label_img[label_img < 128] = 0
                unique_values = np.unique(label_img)
                max_val = max(unique_values)
                min_val = min(unique_values)

                max_val_al = max(max_val, max_val_al)
                min_val_al = min(min_val, min_val_al)

                if train_flag == True:
                    hist = np.histogram(label_img, self.classes, [0, self.classes - 1])
                    global_hist += hist[0]

                    rgb_img = cv2.imread(img_file)[y0: y1, x0: x1]
                    self.mean[0] += np.mean(rgb_img[:, :, 0])
                    self.mean[1] += np.mean(rgb_img[:, :, 1])
                    self.mean[2] += np.mean(rgb_img[:, :, 2])

                    self.std[0] += np.std(rgb_img[:, :, 0])
                    self.std[1] += np.std(rgb_img[:, :, 1])
                    self.std[2] += np.std(rgb_img[:, :, 2])

                else:
                    print("we can only collect statistical information of train set, please check")

                if max_val > (self.classes - 1) or min_val < 0:
                    print('Labels can take value between 0 and number of classes.')
                    print('Some problem with labels. Please check. label_set:', unique_values)
                    print('Label Image ID: ' + label_file)
                no_files += 1

        # divide the mean and std values by the sample space size
        self.mean /= no_files
        self.std /= no_files
        self.mean /= 255
        self.std /= 255

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


def gen_tuples(stride=512, threshold=0.8, wsi_path="pos232/"):
    t0 = time.time()
    full_path = "./seed/" + wsi_path
    pos_list = []
    neg_list = []

    def gen(offset=0, mask=True):
        for img_name in os.listdir(full_path):
            if '_mask' not in img_name:
                # print(img)
                img = cv2.imread(full_path+img_name, 1)
                if mask:
                    msk_name = img_name.replace('.', '_mask.')
                    msk = cv2.imread(full_path+msk_name, 0)

                    # print(msk.shape)
                    h_iter = (msk.shape[0] - offset) // stride
                    w_iter = (msk.shape[1] - offset) // stride
                else:
                    h_iter = (img.shape[0] - offset) // stride
                    w_iter = (img.shape[1] - offset) // stride

                for j in range(h_iter):
                    for i in range(w_iter):

                        if mask:
                            x0, y0 = i * stride + offset, j * stride + offset
                            x1, y1 = (i + 1) * stride + offset, (j + 1) * stride + offset
                            msk_patch = msk[y0: y1, x0: x1]

                            # normal region
                            if 255 * stride ** 2 * (1 - threshold) < np.sum(msk_patch) < 255 * stride ** 2 * 0.5:
                                neg_list.append((wsi_path+img_name, str(y0), str(y1), str(x0), str(x1)))

                            # lesion region
                            elif 255 * stride ** 2 * 0.5 < np.sum(msk_patch) < 255 * stride ** 2 * threshold:
                                pos_list.append((wsi_path+img_name, str(y0), str(y1), str(x0), str(x1)))
                        else:
                            x0, y0 = i * stride + offset, j * stride
                            x1, y1 = (i + 1) * stride + offset, (j + 1) * stride
                            img_patch = img[y0: y1, x0: x1]
                            # if np.mean(img_patch) < 190:
                            neg_list.append((wsi_path+img_name, '0', str(y0), str(y1), str(x0), str(x1)))

    gen(offset=0)
    gen(offset=stride//2)
    print(len(pos_list), len(neg_list))
    random.shuffle(pos_list)
    random.shuffle(neg_list)
    min_len = min(len(pos_list), len(neg_list))
    pos_list = pos_list[: min_len]
    neg_list = neg_list[: min_len]

    p_nums = len(pos_list)
    n_nums = len(neg_list)
    train_list = pos_list[: int(p_nums*0.8)] + neg_list[: int(n_nums*0.8)]
    val_list = pos_list[int(p_nums*0.8): int(p_nums*0.9)] + neg_list[int(n_nums*0.8): int(n_nums*0.9)]
    test_list = pos_list[int(p_nums*0.9):] + neg_list[int(n_nums*0.9):]

    print(len(pos_list), len(neg_list))

    with open('./seed/seed_train_list.txt', mode='w') as f:
        write_list_txt(f, train_list)
    with open('./seed/seed_val_list.txt', mode='w') as f:
        write_list_txt(f, val_list)
    with open('./seed/seed_test_list.txt', mode='w') as f:
            write_list_txt(f, test_list)
    with open('./seed/seed_trainval_list.txt', mode='w') as f:
        write_list_txt(f, train_list)
        write_list_txt(f, val_list)

    t1 = time.time()
    t = t1 - t0
    print(len(train_list), len(val_list), len(test_list))
    print("patches sampling consumes time:", t/60)


def write_list_txt(f, mode_list):
    for x in mode_list:
        f.write(x[0] + ' ' + x[0].replace('.jpg', '_mask.jpg') + ' ' + x[1] + ' ' + x[2] + ' ' + x[3] + ' ' + x[4] + '\n')


if __name__ == "__main__":
    gen_tuples()
