import pandas as pd
data = pd.read_csv('./model_output_data/pred_score.csv',header=None)
# data = pd.read_csv('pred_score.csv', header=None)
# data = pd.read_csv('pred_score_no_g_s.csv', header=None)
print("read over!")

#
data = pd.DataFrame(data)

row_index = data.index.values
col_index = data.columns.values

result = []
#遍历每行每列，将模型最后输出的结果由矩阵形式转化为三列
for row in row_index:
    for col in col_index:
        activity = data.loc[row,col]
        #将遍历到的每行每列的数值保存在result这个全局变量里面，方便后面的保存
        result.append([row,col,activity])
result = pd.DataFrame(result)
result.columns=['username', 'course_id', 'complete']

print("result!")

pred_data = result
# #读取这个数据是为了取的truth
# get_truth = pd.read_csv('./model_input_data/new_activity.csv')
get_truth = pd.read_csv('./data_process/model_input_willingness.csv')
del get_truth['complete']

dropout_test_data = pd.merge(pred_data,get_truth,on=['username','course_id'],how='outer')





recommend_data =  dropout_test_data.fillna(-1)
# train_data = pd.read_csv('./model_input_data/model_train_data.csv',usecols=[0,1])
train_data = pd.read_csv('./model_input_data/model_train_data.csv',usecols=[0,1])
recommend_data = recommend_data.append(train_data)
recommend_data = recommend_data.append(train_data)
df1 = recommend_data.drop_duplicates(subset=['username', 'course_id'],keep=False)


test_data = pd.read_csv('./model_input_data/model_test_data.csv')

del test_data['complete']
dropout = pd.merge(dropout_test_data,test_data,on=['username','course_id'],how='inner')



from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
y_test = dropout.iloc[:,3].values

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as AUC
import matplotlib.pyplot as plt
y_score = 1-dropout.iloc[:,2].values

# 利用roc_curve函数获得假阳性率FPR，和真阳性率recall，都是一系列值
FPR, recall, thresholds = roc_curve(y_test,y_score)
# 计算AUC面积
area = AUC(y_test, y_score)

# 画图
plt.figure()
plt.plot(FPR,recall,label='ROC curve (area = %0.6f)' % area)
plt.legend(loc="lower right")
plt.show()