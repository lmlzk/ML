from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, LogisticRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report
import pandas as pd
import numpy as np

def linear():
    """
    线性回归和梯度下降对波士顿数据集进行处理
    :return: None
    """
    # 1、获取数据，进行分割
    lb = load_boston()

    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)

    # 2、进行标准化处理，特征值和目标值都需要
    std_x = StandardScaler()

    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 目标值进行标准化
    std_y = StandardScaler()

    y_train = std_y.fit_transform(y_train)

    y_std_test = std_y.transform(y_test)

    # 把经过标准化之后的数据预测值真实值，转换成能理解的原来的数据格式
    y_test = std_y.inverse_transform(y_std_test)

    # 线性回归的实现
    lr = LinearRegression()

    lr.fit(x_train, y_train)

    print("正规方程得出的参数：", lr.coef_)

    # 得出正规方程的预测值
    y_lr_predict = lr.predict(x_test)

    y_lr_predict = std_y.inverse_transform(y_lr_predict)

    print("正规方程的均方误差：", mean_squared_error(y_test, y_lr_predict))

    # 梯度下降得出结果
    sgd = SGDRegressor()

    sgd.fit(x_train, y_train)

    print("梯度下降得出的参数：", sgd.coef_)

    y_sgd_predict = sgd.predict(x_test)

    y_sgd_predict = std_y.inverse_transform(y_sgd_predict)

    print("梯度下降的均方误差：", mean_squared_error(y_test, y_sgd_predict))

    # Ridge回归
    rd = Ridge(alpha=0.01)

    rd.fit(x_train, y_train)

    print("岭回归得出的参数：", rd.coef_)

    y_rd_predict = rd.predict(x_test)

    y_rd_predict = std_y.inverse_transform(y_rd_predict)

    print("岭回归的均方误差：", mean_squared_error(y_test, y_rd_predict))

    return None


def logistic():
    """
    逻辑回归对肿瘤数据进行分析
    :return: None
    """
    # 获取数据，去除一些空值
    column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", names=column_names)

    data = data.replace(to_replace='?', value=np.nan)

    # dropna    float
    data = data.dropna()

    # 对特征值和目标值进行分割，训练集和测试集

    x = data[column_names[1:10]]

    y = data[column_names[10]]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 逻辑回归流程
    lr = LogisticRegression()

    lr.fit(x_train, y_train)

    print(lr.coef_)

    print("准确率：", lr.score(x_test, y_test))

    # 得出预测结果
    y_predict = lr.predict(x_test)

    print("精确率和召回率：", classification_report(y_test, y_predict, labels=[2, 4], target_names=['良性', '恶性']))

    return None


if __name__ == "__main__":
    logistic()































