from Initial_data import DataSet

from NewModel import NewModel
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import torch

if __name__ == '__main__':
    # ----------基本参数--------------
    basedir = '.'
    batch_size = 8
    # #隐特征维度
    # hidden_feat = 16
    # 划分比例
    # test_ratio = 0.2
    lr = 1e-3
    # lr = 0.01
    epochs = 25
    # epochs = 2
    # epochs = 300
    device = 'cuda:0'
    # device = 'cpu'
    # ----------基本参数--------------

    dataSet = DataSet(basedir)
    # W = dataSet.get_martix_user()
    # D = dataSet.get_martix_user()
    D = dataSet.get_martix_course()
    P = dataSet.get_martix()

    # 读取训练集 测试集
    train_data = dataSet.train_data
    test_data = dataSet.test_data
    # valid_data = dataSet.valid_data

    # 这里代表需要用来做训练的学习者，是学生索引加载器  train test 里面的学生个数是一样的，所以在哪里读取都一样
    stu_idx_loader = DataLoader(TensorDataset(torch.tensor(dataSet.train_total_stu_list).float()),
                                batch_size=batch_size, shuffle=True)
    # # 加载模型
    # model = Model(
    #     W=W,
    #     D=D,
    #     P=P,
    #     train_cour_num=dataSet.train_cour_num,  # 课程数量
    #     behavior_num=dataSet.behavior_num,  # 隐特征维度，对应的是行为特征个数
    #     lr=lr,
    #     device=device)
    # 加载模型
    model = NewModel(
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

    path = 'model_save/NewNet_weight_MSE.pth'
    model.load_net(path)


    model.net.get_Q()

    model.net.get_A()
