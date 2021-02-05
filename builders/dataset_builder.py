import os
import pickle
from torchvision import transforms
from torch.utils import data
from dataset.cityscapes import *
from dataset.camvid import *
from dataset.seed import *
from dataset.remote import *


# register needed for new dataset, format -> dataset: fn_prefix
loader_name_space = {
    'cityscapes': 'Cityscapes',
    'camvid': 'CamVid',
    'seed': 'Seed',
    'remote': 'Remote',
}


def build_dataset_train(dataset, num_classes, input_size, batch_size, train_type, random_scale, random_mirror, num_workers):
    # import relevant functon
    cap_name = loader_name_space[dataset]
    informFN = eval(cap_name + "TrainInform")
    trainDataSet = eval(cap_name + "DataSet")
    valDataSet = eval(cap_name + "ValDataSet")

    data_dir = os.path.join('./dataset/', dataset)
    dataset_list = dataset + '_trainval.txt'
    train_data_list = os.path.join(data_dir, dataset + '_' + train_type + '.txt')
    val_data_list = os.path.join(data_dir, dataset + '_val' + '.txt')
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')

    # inform_data_file collect the information of mean, std and weigth_class
    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))
        dataCollect = informFN(data_dir=data_dir,
                                 classes=num_classes,
                                 train_set_file=dataset_list,
                                 inform_data_file=inform_data_file,
                                 normVal=1.10)

        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))

    trainLoader = data.DataLoader(
        trainDataSet(data_dir, train_data_list, crop_size=input_size, scale=random_scale,
                          mirror=random_mirror, mean=datas['mean']),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True)

    valLoader = data.DataLoader(
        valDataSet(data_dir, val_data_list, f_scale=1, mean=datas['mean']),
        batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True,
        drop_last=True)

    return datas, trainLoader, valLoader


def build_dataset_test(dataset, num_classes, num_workers=4, none_gt=False):
    # import relevant functon
    cap_name = loader_name_space[dataset]
    valDataSet = eval(cap_name + "DataSet")
    testDataSet = eval(cap_name + "DataSet")

    data_dir = os.path.join('./dataset/', dataset)
    dataset_list = dataset + '_trainval.txt'
    test_data_list = os.path.join(data_dir, dataset + '_test' + '.txt')
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')

    # inform_data_file collect the information of mean, std and weigth_class
    informFN = eval(cap_name + "TrainInform")
    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))
        dataCollect = informFN(data_dir=data_dir,
                               classes=num_classes,
                               train_set_file=dataset_list,
                               inform_data_file=inform_data_file,
                               normVal=1.10)

        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))

    if dataset == "cityscapes":
        # for cityscapes, if test on validation set, set none_gt to False
        # if test on the test set, set none_gt to True

        if none_gt:
            testLoader = data.DataLoader(
                testDataSet(data_dir, test_data_list, mean=datas['mean']),
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        else:
            test_data_list = os.path.join(data_dir, dataset + '_val' + '.txt')
            testLoader = data.DataLoader(
                valDataSet(data_dir, test_data_list, mean=datas['mean']),
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

        return datas, testLoader

    else:
        testLoader = data.DataLoader(
            valDataSet(data_dir, test_data_list, mean=datas['mean']),
            batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

        return datas, testLoader
