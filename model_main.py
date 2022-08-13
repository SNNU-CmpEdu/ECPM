from Initial_data import DataSet

from ECPM import ECPM
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import torch

if __name__ == '__main__':
    basedir = '.'
    batch_size = 8

    lr = 1e-3

    epochs = 25

    device = 'cuda:0'

    dataSet = DataSet(basedir)

    D = dataSet.get_martix_course()
    P = dataSet.get_martix()


    train_data = dataSet.train_data
    test_data = dataSet.test_data

    stu_idx_loader = DataLoader(TensorDataset(torch.tensor(dataSet.train_total_stu_list).float()),
                                batch_size=batch_size, shuffle=True)

    model = ECPM(
        csa=dataSet.csa,
        # ca=dataSet.ca,
        P=dataSet.get_martix(),
        D=dataSet.get_martix_course(),
        train_stu_num = dataSet.train_stu_num,
        train_cour_num=dataSet.train_cour_num,  # 课程数量
        behavior_num=dataSet.behavior_num,  # 隐特征维度，对应的是行为特征个数
        lr=lr,
        device=device)


    model.fit(
        index_loader=stu_idx_loader,
        train_data=train_data,
        test_data=test_data,
        epochs=epochs,
        save_dir=dataSet.save_dir, save_size=(dataSet.train_stu_num, dataSet.train_cour_num))

    # model.test(
    #     index_loader=stu_idx_loader,
    #     train_data=train_data, test_data=test_data,
    #     save_dir=dataSet.save_dir, save_size=(dataSet.train_stu_num, dataSet.train_cour_num))

    save_dir = dataSet.save_dir
    model.save_parameter(dataSet.save_dir)
    #
    # path = 'model_save/NewNet_weight_MSE.pth'
    # model.load_net(path)
    #
    #
    # model.net.get_Q()
    #
    # model.net.get_A()
