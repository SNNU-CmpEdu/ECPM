import pandas as pd
import numpy as np
import torch


# 定义DataSet类
class DataSet():
    # 初始化函数，self指向的是实例化对象
    def __init__(self, basedir):
        self.basedir = basedir
        # read_dir = basedir + '/model_input_data/'
        # save_dir = basedir + '/model_output_data/'
        #
        read_dir = basedir + '/model_input_data/'
        save_dir = basedir + '/model_output_data/'
        self.train_data = pd.read_csv(read_dir + "model_train_data.csv", index_col='username') #第三列做归一化
        self.test_data = pd.read_csv(read_dir + "model_test_data.csv", index_col='username')

        # print(self.train_data['complete_rate'])
        max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

        self.train_data['complete'].apply(max_min_scaler)
        self.test_data['complete'].apply(max_min_scaler)
        # self.valid_data = pd.read_csv(read_dir + "model_valid_data.csv", index_col='username')

        # self.ca = pd.read_csv(read_dir + 'ca.csv', index_col=0)
        # self.ca = torch.Tensor(self.ca.values)

        self.csa = torch.load(read_dir + 'data_csa.pt') #第二维度做归一化softmax
        self.csa = torch.Tensor(self.csa)
        self.csa = torch.softmax(self.csa, dim=2)

        # 读取课程-行为-平均值矩阵
        # martix_user = pd.read_csv(read_dir + "avg_user.csv", index_col='course_id')
        martix_course = pd.read_csv(read_dir + "avg_course.csv", index_col='username')
        martix = pd.read_csv(read_dir + "adjacency_weight.csv", index_col=0)

        # self.martix_user = martix_user
        self.martix_course = martix_course
        self.martix = martix
        # 下方划分数据集的时候需要调用self.record，将uid设置为索引号，
        # 返回的record是一个以uid为索引的dataframe
        # self.train_record = self.train_data.set_index('username')
        # self.test_record = self.test_data.set_index('username')
        # self.valid_record = self.valid_data.set_index('username')
        self.train_record = self.train_data
        self.test_record = self.test_data
        # self.valid_record = self.valid_data
        # print(self.train_record.index)

        # 将record重点索引去重后，以列表形式保存
        self.train_total_stu_list = list(set(self.train_record.index))
        self.test_total_stu_list = list(set(self.test_record.index))
        # self.valid_total_stu_list = list(set(self.valid_record.index))

        # 分别获取学生数和课程数，set具有去重的功能
        self.train_stu_num = len(set(self.train_data.index))
        self.train_cour_num = len(set(self.train_data['course_id']))

        self.test_stu_num = len(set(self.test_data.index))
        self.test_cour_num = len(set(self.test_data['course_id']))

        # self.valid_stu_num = len(set(self.valid_data.index))
        # self.valid_cour_num = len(set(self.valid_data['course_id']))

        # self.behavior_num = martix_user.shape[1]
        self.behavior_num = 19
        self.save_dir = save_dir

    # def get_martix_user(self):
    #     martix_user = self.martix_user
    #     W = np.array(martix_user.values.tolist())
    #     return torch.tensor(W, dtype=torch.double)

    def get_martix_course(self):
        martix_course = self.martix_course
        D = np.array(martix_course.values.tolist())
        return torch.tensor(D, dtype=torch.double)

    def get_martix(self, ):
        martix = self.martix
        P = np.array(martix.values.tolist())
        return torch.tensor(P, dtype=torch.double)
