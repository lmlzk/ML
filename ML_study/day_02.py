from sklearn.datasets import load_iris, fetch_20newsgroups, load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np


# li = load_iris()
# #
# # print(li.data)
# # print(li.target)
# # print(li.feature_names)
# # print(li.target_names)
#
# # 数据集的划分, 训练集的特征值，测试集的特征值，训练集的目标值，测试集的目标值
# # x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, test_size=0.25, random_state=24)
# #
# #
# # x_train1, x_test1, y_train1, y_test1= train_test_split(li.data, li.target, test_size=0.25, random_state=24)
# #
# # print(x_train == x_train1)
#
#
# # news = fetch_20newsgroups(subset='all')
# #
# # print(news.data)
#
#
# lb = load_boston()
#
# print(lb.data)
# print(lb.target)


def knncls():
    """
    K-近邻算法实现入住
    :return: None
    """
    # 获取数据，分析数据
    data = pd.read_csv("./data/FBlocation/train.csv")

    # print(data)

    # 缩小数据的范围，防止运算时间过长
    data = data.query("x > 1.0 & x < 1.25 & y > 2.5 & y < 2.75")

    # 处理时间日期，分割时间，增加一些日期的详细特征
    time_value = pd.to_datetime(data['time'], unit='s')

    # 把时间格式转换成字典格式，获取年，月，日
    time_value = pd.DatetimeIndex(time_value)

    # 构造新的特征，weekday, day ,hour
    data['weekday'] = time_value.weekday
    data['day'] = time_value.day
    data['hour'] = time_value.hour

    data = data.drop(['time'], axis=1)

    # 删除一些签到位置少的签到点
    place_count = data.groupby('place_id').aggregate(np.count_nonzero)
    tf = place_count[place_count.row_id > 3].reset_index()

    data = data[data['place_id'].isin(tf.place_id)]

    # 取出特征值和目标值
    y = data['place_id']

    x = data.drop(['place_id'], axis=1)


    # 进行数据的分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 进行标准化
    std = StandardScaler()

    x_train = std.fit_transform(x_train)

    # x_test = std.fit_transform(x_test)
    x_test = std.transform(x_test)

    # estimaotr估计器流程
    knn = KNeighborsClassifier()

    # # fit数据
    # knn.fit(x_train, y_train)
    #
    # # 预测结果
    #
    # # 得出准确率
    # score = knn.score(x_test, y_test)

    param = {"n_neighbors": [1, 3, 5]}

    # 使用网格搜索
    gs = GridSearchCV(knn, param_grid=param, cv=2)

    # 输入数据
    gs.fit(x_train, y_train)

    # 得出测试集的准确率
    print("测试集的准确率：", gs.score(x_test, y_test))

    print("在交叉验证当中的最好验证结果：", gs.best_score_)

    print("选择了模型：", gs.best_estimator_)

    print("每个超参数每一个交叉验证：", gs.cv_results_)

    return None


def navie_bayes():
    """
    朴素贝叶斯对新闻分类
    :return: None
    """
    # 获取新闻的数据集
    news = fetch_20newsgroups(subset='all')

    # 进行数据集的分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)

    # 进行特征抽取
    tf = TfidfVectorizer()

    x_train = tf.fit_transform(x_train)

    x_test = tf.transform(x_test)

    # 进行朴素贝叶斯分类
    mlb = MultinomialNB(alpha=1.0)

    mlb.fit(x_train, y_train)

    y_predict = mlb.predict(x_test)

    print("预测的文章类型结果：", y_predict)

    score = mlb.score(x_test, y_test)

    print("准确率：", score)

    print(classification_report(y_test, y_predict, target_names=news.target_names))

    return None


if __name__ == "__main__":
    knncls()















































